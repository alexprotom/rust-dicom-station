#!/usr/bin/env bash
# Start the simulator build on a simulated iPad or iPhone and see that it
# stays up.
#
#   ./simulator-smoke.sh [ipad|iphone] [OUT_DIR]   # default: ipad, out/smoke/<device>
#
# Needs the app from `./build-ipa.sh --simulator` (out/simulator/
# RustDICOMStation.app) and Xcode with at least one iOS simulator runtime.
# Picks the newest runtime and an iPad (an iPad Pro when there is one) or an
# iPhone (a Pro Max when there is one) of the newest generation it offers,
# creates a fresh simulator (so nothing from an earlier run is on it), boots
# it, installs and launches the app, waits, and then:
#
#   * saves a screenshot of the simulator's screen and the program's
#     standard output and error into OUT_DIR;
#   * fails if the program wrote last_panic.txt into its Documents folder
#     (src/main.rs does that on any panic), and prints it;
#   * fails if the program is no longer running (launchd's list of the
#     simulator's jobs, saved as launchctl.txt, has no PID for it).
#
# The simulator is deleted again at the end, pass or fail.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
family="${1:-ipad}"
case "$family" in
    ipad) family_name=iPad; prefer="Pro" ;;
    iphone) family_name=iPhone; prefer="Pro Max" ;;
    *) echo "usage: $0 [ipad|iphone] [OUT_DIR]" >&2; exit 2 ;;
esac
out="${2:-$here/out/smoke/$family}"
app="$here/out/simulator/RustDICOMStation.app"
wait_s="${RDS_SMOKE_SECONDS:-45}"

[ -d "$app" ] || { echo "$app is missing - run ./build-ipa.sh --simulator first" >&2; exit 1; }
mkdir -p "$out"
bundle_id="$(/usr/libexec/PlistBuddy -c 'Print :CFBundleIdentifier' "$app/Info.plist")"

# ---- the simulated device ------------------------------------------------------
# The newest available iOS runtime, and among the device types it supports
# for the family asked for the preferred model (else any) listed last,
# which is the newest.
choice="$(xcrun simctl list -j | FAMILY="$family_name" PREFER="$prefer" python3 -c '
import json, os, sys
d = json.load(sys.stdin)
family, prefer = os.environ["FAMILY"], os.environ["PREFER"]
rts = [r for r in d["runtimes"] if r.get("platform") == "iOS" and r.get("isAvailable")]
if not rts:
    sys.exit("no available iOS simulator runtime")
rt = max(rts, key=lambda r: [int(x) for x in r["version"].split(".")])
types = rt.get("supportedDeviceTypes") or d["devicetypes"]
fits = [t for t in types if t.get("productFamily") == family]
if not fits:
    sys.exit("no " + family + " device type for " + rt["name"])
best = [t for t in fits if t["name"].endswith(prefer) or (prefer + " ") in t["name"]] or fits
t = best[-1]
print(rt["identifier"], t["identifier"], t["name"].replace(" ", "_"))
')"
read -r runtime devtype devname <<EOF
$choice
EOF
[ -n "$devtype" ] || { echo "could not choose a simulated $family_name" >&2; exit 1; }
echo "runtime $runtime, device ${devname//_/ }"

udid="$(xcrun simctl create rds-smoke "$devtype" "$runtime")"
cleanup() {
    xcrun simctl shutdown "$udid" >/dev/null 2>&1 || true
    xcrun simctl delete "$udid" >/dev/null 2>&1 || true
}
trap cleanup EXIT

xcrun simctl boot "$udid"
xcrun simctl bootstatus "$udid" -b >/dev/null
xcrun simctl install "$udid" "$app"

# ---- the program ---------------------------------------------------------------
xcrun simctl launch \
    --stdout="$out/stdout.log" \
    --stderr="$out/stderr.log" \
    "$udid" "$bundle_id"
echo "launched $bundle_id; waiting ${wait_s}s"
sleep "$wait_s"

xcrun simctl io "$udid" screenshot "$out/screenshot.png" >/dev/null
echo "screenshot: $out/screenshot.png"

data="$(xcrun simctl get_app_container "$udid" "$bundle_id" data)"
ls -la "$data/Documents" > "$out/documents.txt" 2>&1 || true
if [ -f "$data/Documents/last_panic.txt" ]; then
    cp "$data/Documents/last_panic.txt" "$out/"
    echo "::error::the program panicked on the simulator:"
    cat "$data/Documents/last_panic.txt"
    exit 1
fi

# Into a file first, not piped into `grep -q`: grep stops reading at the
# match, `simctl spawn` then dies of SIGPIPE writing the rest ("Child process
# terminated with signal 13: Broken pipe"), and under pipefail that turned a
# running program into a "not running" one. A line counts only with a PID in
# its first column; an app that has exited can keep its line, with "-" there.
xcrun simctl spawn "$udid" launchctl list > "$out/launchctl.txt" 2>&1 || true
if awk -v job="UIKitApplication:$bundle_id" \
        'index($3, job) == 1 && $1 ~ /^[0-9]+$/ { up = 1; print } END { exit !up }' \
        "$out/launchctl.txt"; then
    echo "the program is running ${wait_s}s after launch"
else
    grep -F "UIKitApplication:$bundle_id" "$out/launchctl.txt" || true
    echo "::error::the program is not running ${wait_s}s after launch; its standard error:"
    tail -n 60 "$out/stderr.log" || true
    exit 1
fi

echo "--- standard error (last lines) ---"
tail -n 20 "$out/stderr.log" || true

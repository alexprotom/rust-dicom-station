# The PACS server: `rds-pacs`

Every station keeps its own archive (*Tools ▶ 🏥 PACS*, [pacs.md](pacs.md)).
The PACS server puts a door on one station's archive so that other stations
can reach it: list its patients, copy studies down, work on them with every
tool the viewer has, send back what they drew, and hand the server
workflows to run on its own machine. The other stations need nothing
installed beyond the viewer itself; every build of Rust DICOM Station is a
client, the Android and iOS ones included.

The server is an optional component. A default installation is a client
only, and nothing listens on the network until someone starts the server.

```text
  station A (any OS)              station B (PC or Mac)                  station C (iPad)
  +----------------+   HTTPS      +-------------------------------+      +----------------+
  | viewer         | -----------> | rds-pacs   (the server)       | <--- | viewer         |
  |  mirror of B's |  pinned      |   serves   <data>/archive     |      |  mirror of B's |
  |  studies       |  certificate |   runs     tasks (workflows)  |      |  studies       |
  +----------------+  + token     | viewer     Tools > PACS works |      +----------------+
                                  |            on the same folder |
                                  +-------------------------------+
```

## Contents

1. [What it does](#what-it-does)
2. [Setting up the server](#setting-up-the-server)
3. [Reaching it on the local network](#reaching-it-on-the-local-network)
4. [Reaching it over the internet](#reaching-it-over-the-internet)
5. [Connecting a station](#connecting-a-station)
6. [Working on a server's studies](#working-on-a-servers-studies)
7. [Tasks: letting the server do the work](#tasks-letting-the-server-do-the-work)
8. [Phones and tablets](#phones-and-tablets)
9. [Security](#security)
10. [The configuration file](#the-configuration-file)
11. [The command line](#the-command-line)
12. [Where the files are](#where-the-files-are)
13. [Troubleshooting](#troubleshooting)
14. [For developers: the protocol](#for-developers-the-protocol)

## What it does

| | |
|---|---|
| **Serve the archive** | The station's own archive folder, the same one *Tools ▶ PACS* shows on that machine, listed and downloadable by paired stations |
| **Mirror mode** | A station pulls studies into a local copy (its *mirror* of that server), works on them with or without a connection, and syncs back the structure sets and segmentations it made |
| **Task mode** | A station hands the server a workflow bound to studies of its archive; the server runs it on its own hardware and files the results, which the station then pulls |
| **Pairing** | A station is let in by a one-time pairing code from the server's operator, then by its own key; the operator sees every paired station and can revoke any of them |
| **TLS always** | Every connection is encrypted, on the local network as much as over the internet; the server's certificate is pinned when a station pairs |

What it is not: a DICOM network node in the classical sense (no DIMSE,
no C-STORE from a scanner), a cloud service, or a medical device. Like
the rest of the station it is for research and QA use.

## Setting up the server

### 1. Install it

The server is the program `rds-pacs`, installed beside the viewer:

| Installation | How |
|---|---|
| **Windows installer** | Tick *Install the PACS server* on the options page (unticked by default), or run the setup with `--pacs`. An update keeps the choice. |
| **macOS disk image** | Always in the bundle: `Rust DICOM Station.app/Contents/MacOS/rds-pacs`. Installed through Homebrew, `rds-pacs` is on the PATH. |
| **Linux AppImage** | Always inside: `rust-dicom-station.AppImage pacs serve` runs it. A link named `rds-pacs` to the AppImage works too. |
| **Snap** | The snap's third command, `rust-dicom-station.rds-pacs`. |
| **Flatpak** | `flatpak run --command=rds-pacs io.github.alexprotom.rust-dicom-station`. |
| **From source** | `cargo build --release --features pacs-server` writes `target/release/rds-pacs` beside the viewer. |

The viewer's *Settings ▶ PACS server* says whether it is installed.

### 2. Start it

Open *Settings ▶ PACS server ▶ Open the server window* and press
**▶ Start**. The viewer starts `rds-pacs` as a program of its own: it
keeps serving when the window, and the viewer, are closed. **⏹ Stop** in
the same window stops it.

The first start makes the server's certificate and shows its fingerprint,
sixteen groups of four characters:

```text
Certificate   3F2A 9C41 0B7E ... 77D0
```

**The firewall.** Windows may ask once, in a *Windows Security Alert*,
whether `rds-pacs.exe` may accept connections: *Allow* it for *private*
networks (and *public* ones only if the computer really is reached through
one). The server window says what the Windows firewall currently does with
`rds-pacs`: *allowed* on which kinds of network, *no rule* (nothing asked,
or the alert was closed) or *blocked* (the alert was answered with
*Cancel*, which makes a rule that blocks). **🔓 Allow through the Windows
firewall** puts it right from the window: Windows asks for administrator
permission, and a rule is made that lets `rds-pacs.exe` accept TCP
connections on private and domain networks (tick *also on networks
Windows calls public* when the computer's network is classed as public:
Windows does that with an unknown network, and often with a home network
until it is marked private in the network settings). Without such a rule
other stations see *connection timed out*, however right the address is.
macOS asks the same question when the server first starts; *Allow*. On
Linux a firewall is the operator's own (`ufw allow 11443/tcp`).

The same from a terminal: `rds-pacs serve` (or the command of the table
above with `serve`) runs the server in the foreground and prints where it
listens, the fingerprint and a connection line; Ctrl+C stops it. Starting
it at log-on is up to the operating system's own tools (Task Scheduler on
Windows, a `launchd` agent on macOS, a `systemd --user` unit on Linux)
running `rds-pacs serve`.

### 3. Choose what it serves (optional)

The window's *Settings* section writes `pacs.toml`
([the configuration file](#the-configuration-file)); a running server
takes changes at its next start. By default it serves the station's own
archive on port 11443 on every network, runs tasks, and downloads no
model weights.

### 4. Pair the stations

For each station: choose its **role**, press **New pairing code**, and
give the person at that station two things - the **connection details**
(*📋 Copy connection details*: one line with the address and the
certificate fingerprint) and the **code**. *📋 Copy invitation* puts both
in one line. A code is good for one station and for ten minutes (or as
long as chosen), and it is never shown again.

| Role | May |
|---|---|
| `view` | list the archive, download studies |
| `edit` | also send structure sets, segmentations and whole studies into it |
| `run` | also hand the server workflows to run |
| `admin` | also remove studies and patients, make pairing codes, see and revoke stations, read the activity log |

The *Paired stations* list shows each station with its role, when it was
last seen and from where; **Revoke** stops its key at once.

## Reaching it on the local network

Nothing more is needed. The server window lists, under *How other
stations reach it*, every address of the computer with the port
(`192.168.1.20:11443`, the interface it belongs to beside it: *Ethernet*,
*Wi-Fi*, *Tailscale*; the one marked *the usual one* is the address the
computer sends from by default) and the machine's name with the port. The
📋 beside an address copies a connection line with that address; *Copy
connection details* takes the first. A computer with a wired and a
wireless link, or a VPN, has several addresses, and the other station must
be on the same network as the one it is given. The list is live: an
address the router hands out can change (after a reboot, a new lease, a
switch from cable to Wi-Fi), and a connection line copied before that
points nowhere. Give the server's computer a fixed address in the router
(a *DHCP reservation*) so that the stations' saved address stays right.

If a station cannot connect (*connection timed out*): check that the
server runs, that the address is one the window lists *now*, that the
window says the firewall allows `rds-pacs` (above), and that both machines
are on the same network (a guest Wi-Fi often keeps devices apart; so does
*AP isolation* on some routers).

## Reaching it over the internet

The server listens on the computer it runs on; how the internet reaches
that computer is the network's business. Three ways, from the simplest:

### A. A mesh VPN (recommended)

Install [Tailscale](https://tailscale.com), [ZeroTier](https://www.zerotier.com)
or a [WireGuard](https://www.wireguard.com) tunnel on the server and on
each station. Every device gets an address of the VPN (`100.x.y.z` for
Tailscale) that works from anywhere, and nothing is opened on any router.
Use the server's VPN address (shown in the VPN's app) in the connection
line: replace the address in `rds-pacs://192.168.1.20:11443/#sha256=...`
with it and keep the rest. Nothing else changes: the certificate pin and
the key work exactly as on the local network.

This is the recommended way because the server is then reachable only by
devices of the VPN, before any of the server's own checks.

### B. A forwarded port and a name

1. In the router, forward TCP port 11443 (or the port chosen) to the
   server's computer, whose local address should be fixed (see above).
2. If the internet address of the connection changes, give it a name
   through a dynamic DNS service (the router usually has one built in):
   `ward-pacs.example.net`.
3. Add that name to *Extra names* in the window's settings and press
   *New certificate* (only while the server is stopped; every station
   paired before must pair again), so the certificate carries it. The
   pinned stations do not need this - they trust the certificate's
   fingerprint, not its names - but other tools do.
4. Use the name in the connection line:
   `rds-pacs://ward-pacs.example.net:11443/#sha256=...`.

What is exposed: one TLS listener that answers nothing but its own name
and version without a key, lets a station in only with a pairing code the
operator made in the last minutes, and slows guessing down to one try per
second per address. That is a small surface, but it is a surface on the
internet; route A avoids it.

### C. A reverse proxy with a public certificate

For an operator who already runs a web server with a real certificate
(Caddy, nginx, Traefik with Let's Encrypt):

1. In the window's settings choose *Listen on: this computer only*
   (`bind = "127.0.0.1"`, `behind_proxy = true`), so that only the proxy
   reaches the server.
2. Point the proxy at `https://127.0.0.1:11443` and let it accept the
   server's self-signed certificate on that hop (Caddy:
   `reverse_proxy https://127.0.0.1:11443 { transport http { tls_insecure_skip_verify } }`;
   nginx: `proxy_pass https://127.0.0.1:11443; proxy_ssl_verify off;`).
   Pass `X-Forwarded-For`, so the activity log shows the stations'
   addresses rather than the proxy's. Raise the proxy's request size limit
   to `max_upload_mb` (nginx: `client_max_body_size 2g;`).
3. Stations pair with the proxy's public name and tick *The server has a
   certificate the system trusts* instead of checking a fingerprint.

Alternatively give `rds-pacs` the certificate files themselves
(`tls_cert`, `tls_key`) and forward the port as in B; stations then pair
the same way.

## Connecting a station

On the station: *Tools ▶ 🏥 PACS ▶ ➕ Add server*.

1. Paste the **connection line** (or type the server's address, `host` or
   `host:port`) and press **🔍 Look at the server**. Nothing is sent but a
   request for the server's name; the window shows who answered and the
   certificate it presented.
2. **Check the certificate.** With a connection line it is compared for
   you (*✔ the same as in the connection line*). With a bare address,
   compare the fingerprint shown with the one in the server's window (or
   the operator's message) and tick that they are the same. A fingerprint
   that does not match means a different server, or one whose certificate
   was renewed: do not pair, ask the operator.
3. Type the **pairing code** (case and the dash do not matter) and, if you
   like, a different name for this station, and press **🔒 Pair**.

The server now appears in the row of sources at the top of the PACS
window, beside *This station*. What the station keeps about it is in
`pacs-servers.json` in its configuration folder
([where the files are](#where-the-files-are)): the address, the pinned
fingerprint, the role and the key. That file is a secret - whoever has it
can act as this station - and it is written readable by its owner only.

*🔒 Pair again* (with a new code) mends a revoked key or a renewed server
certificate; *✖ Forget server* removes the server from the station (what
was copied down stays in the mirror folder until removed by hand).

## Working on a server's studies

Select the server at the top of the PACS window. The **Studies** tab lists
its patients and studies like the local archive, with a ✔ on every study
this station holds a copy of.

| Button | What happens |
|---|---|
| **📥 Pull** | Copies the selected study (or the whole patient) into this station's mirror of the server. Only what the mirror does not hold yet is fetched, in bundles; a pull that breaks off keeps what arrived. |
| **📩 Load into workspace A / B** | Pulls what is missing, then loads the mirrored folder into the workspace, exactly like *File ▶ Add DICOM folder*. Every tool of the viewer works on it from here on, connected or not. |
| **📤 Send workspace A / B** | Gives the server the workspace's structure sets and segmentation series (new SOP Instance UIDs, the original Study and Frame of Reference UIDs, a reference to the image series), filed under the study they were drawn on, and files them into the mirror as well. Images are never sent again. If the server cannot be reached, the objects wait in the **outbox**. |
| **⟲ Sync** | Both directions for every study held here: delivers the outbox, fetches what others filed into those studies meanwhile, and sends what the server lacks. Studies never pulled are not fetched. |
| **📤 Upload folder** | Sends every DICOM file of a folder on this computer into the server's archive (a new study for the server, say). |
| right-click | *Remove the local copy* (the server keeps the study); with the admin role also *Remove from the server*. |

Nothing is ever overwritten, on either side: both archives only grow,
files are named by their SOP Instance UIDs, and every object the station
makes gets UIDs of its own. Two people who send structures for the same
study both find theirs there afterwards, as two structure sets. Sending
the same objects twice files them once.

When the server cannot be reached, the window shows its archive as last
seen, everything held here loads and works as usual, and sends go to the
outbox. The next **Sync** delivers them.

## Tasks: letting the server do the work

A station with the `run` role has a **Tasks** tab: the server runs a
workflow ([workflows.md](workflows.md)) on studies of its archive, on its
own CPU and GPU, while the station does something else - or is switched
off. One task runs at a time, the rest wait in order.

1. Choose a workflow. The server offers its operator's saved workflows,
   the examples the program ships, and a few one-step templates: *Body
   contour*, *Organs (TotalSegmentator, fast)*, *Heart on every phase of
   a 4DCT*.
2. For each input of the workflow choose a study of the server's archive
   (a *DICOM folder* input can also take a whole patient; a *DICOM
   folders* batch takes a patient and runs once per study).
3. Leave *File the results into the server's archive* ticked, so the
   results can be pulled, and press **▶ Run on the server**.

The queue shows each task's state and progress; *Details* shows its log
and the files it wrote besides DICOM (reports and tables, each with a
*💾* button to save it here); *✖ Cancel* stops it. When a task is done,
**📥 Pull results** pulls the studies it filed into; load them as usual.

Engines that need model weights use the server's model folder. With
`allow_model_download = false` (the default) a missing model is an error
the task reports; the operator downloads it once in the server station's
*Tools ▶ Downloaded models*, or allows downloads.

What a task may not do: write anywhere but its own run folder (a workflow
that names an absolute output folder is refused), read a folder of the
server's other than the studies it was bound to, or read a file of the
server's when the workflow was sent along with the request.

## Phones and tablets

The Android and iOS builds are clients: *Tools ▶ PACS ▶ Add server*
works as on a desktop, and a mirror on a tablet is the usual way to take
studies along and work on them offline. The mirror lives in the app's
data folder (on iOS the Files app shows it under *Rust DICOM Station*).
Downloads run while the app is in the foreground. The server itself
does not run on a phone or a tablet.

On Android, the connection line and the pairing code fields have a
**📋 Paste** button: the keyboard's own paste does not reach the app
(the window library has no way to the system clipboard there), the
button does. Copy the line from the mail or chat it came in, open *Add
server*, press *Paste*. An invitation line pasted into the code field
gives up its code. What the app's copy buttons copy reaches the system
clipboard the same way.

## Security

* **Encryption.** TLS (rustls) on every connection, loopback included.
  There is no plain-HTTP mode.
* **The server's identity.** A self-signed certificate, made once and
  kept; a station pins its SHA-256 when it pairs and from then on accepts
  that certificate and no other. A certificate that changed fails the TLS
  handshake, before the station's key is sent. The handshake's signatures
  are checked against the pinned certificate's key, so showing the right
  certificate is not enough without holding its key. With an operator's
  own certificate, stations verify through the system's roots instead.
* **Who may connect.** A pairing code: eight characters, one station,
  minutes long, with a role. Each station then has its own key (32 random
  bytes); the server stores only its SHA-256 and compares in constant
  time. Revoking a station stops its key at once.
* **Guessing.** One pairing attempt per second per address; a code that
  has seen five wrong attempts is void.
* **What a request can reach.** Nothing is built from the URL: UIDs are
  checked to be UIDs and then compared with the archive's folder names;
  an upload is streamed to disk, cut off past `max_upload_mb`, and
  unpacked under numbered names, never under its entries' own.
* **The activity log** (`audit-YYYY-MM-DD.log` in the server's data
  folder) records who did what to which study, never a patient's name.
* **Patient identity.** A PACS carries it by nature: what leaves the
  server is DICOM data, over TLS, to stations its operator paired. Run
  *Tools ▶ Anonymize* before filing anything into a server that people
  outside the department reach.
* **The local operator.** The viewer on the server's own computer talks
  to the server with a key the server writes into its state folder on
  every start, readable by its owner only. Only that key can stop the
  server.

## The configuration file

`pacs.toml` in the station's configuration folder (the server window
shows the path). Every key may be left out:

```toml
name = ""                  # what stations see; empty: this computer's name
bind = "0.0.0.0"           # 127.0.0.1 behind a reverse proxy
port = 11443
archive_dir = ""           # empty: the viewer's archive (Tools > PACS)
advertise = []             # extra names the certificate carries
tls_cert = ""              # PEM files of a certificate of your own;
tls_key = ""               #   empty: the self-signed one
behind_proxy = false       # take the station's address from X-Forwarded-For
tasks = true               # run the workflows stations hand in
max_upload_mb = 2048       # the largest single upload
max_queued_tasks = 16
task_timeout_minutes = 240
pairing_minutes = 10       # how long a code is good for by default
models_dir = ""            # empty: the viewer's model folder
allow_model_download = false
volume_cache_mb = 4096     # image volumes a task keeps between its steps
workflows_dir = ""         # empty: the viewer's workflow folder
audit_log = true
```

`rds-pacs --check` prints what the file says and exits.

## The command line

```text
rds-pacs [--config PATH] [serve]        run the server (the default)
rds-pacs --check                        print the configuration and exit
rds-pacs pair [--role R] [--minutes N]  a pairing code (the server must run)
rds-pacs clients [--revoke NAME]        the paired stations, or revoke one
rds-pacs cert [--regenerate]            the certificate's fingerprint, or a new one
rds-pacs stop                           stop the running server
```

`pair` prints the code on standard output and the invitation line on
standard error.

## Where the files are

| | Windows | Linux | macOS |
|---|---|---|---|
| configuration folder | `%LOCALAPPDATA%\RustDICOMStation` | `~/.config/RustDICOMStation` | `~/Library/Application Support/RustDICOMStation` |
| data folder | `%LOCALAPPDATA%\RustDICOMStation` | `~/.local/share/RustDICOMStation` | `~/Library/Application Support/RustDICOMStation` |

(In a snap both are under `~/snap/rust-dicom-station/common/`, in the
Flatpak under `~/.var/app/io.github.alexprotom.rust-dicom-station/`.)

```text
<configuration folder>/pacs.toml          the server's configuration
<configuration folder>/pacs/              the server's state
    server.crt, server.key                its certificate (the key: owner only)
    identity.json                         its id (names the stations' mirrors)
    clients.json                          paired stations, keys as hashes only
    local-admin.token                     the local operator's key (owner only)
    running.json                          while it listens: port, fingerprint
<data folder>/archive/                    the archive it serves (by default)
<data folder>/pacs/                       its work
    audit-YYYY-MM-DD.log                  the activity log
    server.log                            what it printed when the viewer started it
    tasks/<id>/TASK.json, run/            one folder per task
<configuration folder>/pacs-servers.json  a station's paired servers (a secret)
<data folder>/pacs-mirror/<server id>/    a station's copy of one server
    archive/                              the copy, in the archive's layout
    outbox/                               what waits to be sent
```

Backing up a server means the archive folder and `<configuration
folder>/pacs/`. Moving the server to another computer with both keeps
every station's pairing.

## Troubleshooting

| Symptom | Cause, and what to do |
|---|---|
| *the server cannot be reached: ... connection timed out* | Nothing answers at that address: the server's computer has another address now (the window's *How other stations reach it* lists the current ones), its firewall drops the connection (the window says whether `rds-pacs` is allowed; *Allow through the Windows firewall*), or the two machines are not on the same network. Try the address from the station's browser as `https://address:11443/rds/v1/server` (a certificate warning there is expected and says the server answers). |
| *the server cannot be reached: ... connection refused* | The computer is there but nothing listens on that port: the server does not run, or runs on another port. |
| *the server presented a different certificate* | The server's certificate was renewed (*New certificate*, `rds-pacs cert --regenerate`, or its state folder was lost), or this is not the server. Ask the operator for the new connection line and *Pair again*. |
| *not let in: this station's token is not accepted* | The station was revoked, or the server's `clients.json` was reset. *Pair again* with a new code. |
| *this pairing code is not valid* | Mistyped, already used, expired, or spoiled by wrong attempts. Make a new one. |
| *not allowed: ... paired with the view role* | Ask the operator to pair the station again with a role that allows it. |
| A task fails with a missing model | The server has no weights for that engine and downloads are off; see [Tasks](#tasks-letting-the-server-do-the-work). |
| The server does not start from the window | `server.log` in the server's data folder says why; a port taken by another program is the usual reason (change `port`). |

## For developers: the protocol

The code is in `src/pacs/` (the client side, always compiled) with the
server half behind the cargo feature `pacs-server`:

```text
src/pacs/protocol.rs   the JSON both sides share (serde, #[serde(default)])
src/pacs/client.rs     Remote: ureq + the pinning verifier; bundles, uploads
src/pacs/servers.rs    pacs-servers.json
src/pacs/mirror.rs     the mirror, the outbox, pull / send / sync
src/pacs/config.rs     pacs.toml
src/pacs/local.rs      the server on this machine: folders, running.json, start
src/pacs/tls.rs        the certificate (rcgen), the rustls server configuration
src/pacs/auth.rs       pairing codes, tokens, roles
src/pacs/server.rs     the routes (axum over tokio)
src/pacs/tasks.rs      the queue and the runner over workflow::graph::exec
src/bin/rds-pacs.rs    the executable
src/app/pacs_remote.rs, pacs_server_win.rs   the windows
```

Every route is under `/rds/v1/` and, except the first two, needs
`Authorization: Bearer <key>`:

| Method and path | Role | |
|---|---|---|
| `GET /server` | none | name, version, protocol version, server id, fingerprint, whether it runs tasks |
| `POST /pair` | a code | `{code, client_name}` → `{token, role, server_id, client_name}` |
| `GET /whoami` | view | the caller's name and role |
| `GET /patients` | view | the listing |
| `GET /studies/{uid}` | view | the manifest: every instance with its SOP Instance UID and size |
| `GET /studies/{uid}/instances/{sop}` | view | one file |
| `POST /studies/{uid}/bundle` | view | `{sops}` → a zip of as many as fit 64 MB |
| `POST /studies` | edit | a DICOM file or a zip of them → what the archive did |
| `DELETE /studies/{uid}`, `/patients/{key}` | admin | remove |
| `GET /workflows` | run | what the server offers, with each workflow's inputs |
| `POST /tasks`, `GET /tasks`, `GET /tasks/{id}?log_from=N` | run | hand in, list, follow |
| `POST /tasks/{id}/cancel` | run | the station that handed it in, or an admin |
| `GET /tasks/{id}/files/{path}` | run | a report or table the task wrote |
| `POST /admin/pairing-codes`, `GET /admin/clients`, `POST /admin/clients/{name}/revoke`, `GET /admin/audit` | admin | the operator's window |
| `POST /admin/shutdown` | the local operator | stop |

Errors are `{error, code}` with the HTTP status. `ServerInfo` carries the
protocol's major version; a station refuses a server whose major it does
not know, and every answer reads across minor changes because unknown
fields are ignored and missing ones default.

```bash
cargo build --release --features pacs-server
cargo clippy --features pacs-server --all-targets -- -D warnings
cargo test --features pacs-server --test pacs_server --test pacs_mirror
```

`tests/pacs_server.rs` starts real servers on loopback ports and holds
them to the rules above: nothing without a key, a role is a ceiling, a code
works once and guessing is throttled, revoking is immediate, a changed
certificate is refused before the key leaves, no path is reachable through
a URL, an oversized upload leaves the archive alone, and the certificate,
the identity and the keys survive a restart. `tests/pacs_mirror.rs` is
`tests/archive.rs` over the wire (pull, load, draw, send back, the outbox
while the server is down, sync in both directions, a study the server
dropped) and a task run end to end (the body-contour template, bound to a
study, filed back and pulled).

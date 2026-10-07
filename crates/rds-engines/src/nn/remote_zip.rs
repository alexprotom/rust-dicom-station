//! A few members out of a remote ZIP archive without downloading the rest.
//!
//! The nnU-Net v1 pretrained models are published as one archive per task
//! (about 5 GB: every configuration, five folds each, with their optimizer
//! states), of which one network is a few hundred megabytes. When the server
//! answers HTTP range requests, [`HttpRange`] presents the remote file as
//! `Read + Seek`: the `zip` crate reads the central directory from the end
//! of the file and then just the members asked for, each through one
//! request that streams from the member's local header onwards. A server
//! that does not do ranges gets the whole archive downloaded once, to a
//! temporary file next to the destination, and the members are taken out
//! of that ([`fetch_members`]).

use anyhow::{bail, Context, Result};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::progress::{ProgressSink, CANCELLED};

/// How much of the end of the file is fetched up front: the central
/// directory of an archive of a few hundred files, and the end records.
const TAIL: u64 = 256 * 1024;
/// The first request at a new place asks for this much; every request that
/// continues where the last one ended asks for twice as much as it did, up
/// to [`MAX_CHUNK`]. A large member is a dozen requests, and a jump costs at
/// most one small read-ahead.
const FIRST_CHUNK: u64 = 256 * 1024;
const MAX_CHUNK: u64 = 64 << 20;

/// A remote file read through HTTP range requests.
pub struct HttpRange {
    agent: ureq::Agent,
    url: String,
    len: u64,
    pos: u64,
    /// The last `tail.len()` bytes of the file.
    tail: Vec<u8>,
    /// An open response body, the file offset of its next byte and the
    /// offset it ends at.
    stream: Option<(Box<dyn Read + Send + Sync>, u64, u64)>,
    /// What the next request continuing the current one asks for.
    chunk: u64,
    /// Requests made and bytes received, for progress and the tests.
    pub requests: u32,
    pub received: u64,
}

/// Parse `Content-Range: bytes a-b/total` into `(a, total)`.
fn content_range(v: &str) -> Option<(u64, Option<u64>)> {
    let rest = v.trim().strip_prefix("bytes")?.trim();
    let (range, total) = rest.split_once('/')?;
    let (a, _) = range.split_once('-')?;
    Some((a.trim().parse().ok()?, total.trim().parse().ok()))
}

impl HttpRange {
    /// Open `url`, or `Ok(None)` when its server sends whole files only.
    pub fn open(agent: &ureq::Agent, url: &str) -> Result<Option<HttpRange>> {
        let resp = agent
            .get(url)
            .set("Range", &format!("bytes=-{TAIL}"))
            .call()
            .with_context(|| format!("open {url}"))?;
        if resp.status() != 206 {
            return Ok(None);
        }
        let Some((start, Some(len))) = resp.header("Content-Range").and_then(content_range) else {
            return Ok(None);
        };
        let mut tail = Vec::with_capacity((len - start) as usize);
        resp.into_reader()
            .take(len - start)
            .read_to_end(&mut tail)
            .context("read the end of the archive")?;
        if tail.len() as u64 != len - start {
            bail!("{url}: short read at the end of the archive");
        }
        Ok(Some(HttpRange {
            agent: agent.clone(),
            url: url.to_string(),
            len,
            pos: 0,
            received: tail.len() as u64,
            tail,
            stream: None,
            chunk: FIRST_CHUNK,
            requests: 1,
        }))
    }

    /// The file's size in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// A body for the bytes `[at, end)`.
    fn request(&mut self, at: u64, end: u64) -> std::io::Result<Box<dyn Read + Send + Sync>> {
        let resp = self
            .agent
            .get(&self.url)
            .set("Range", &format!("bytes={at}-{}", end - 1))
            .call()
            .map_err(|e| std::io::Error::other(format!("{}: {e}", self.url)))?;
        self.requests += 1;
        let start = resp.header("Content-Range").and_then(content_range);
        if resp.status() != 206 || start.map(|s| s.0) != Some(at) {
            return Err(std::io::Error::other(format!(
                "{}: the server did not honour a range request",
                self.url
            )));
        }
        Ok(Box::new(resp.into_reader()))
    }
}

impl Read for HttpRange {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.pos >= self.len || buf.is_empty() {
            return Ok(0);
        }
        let tail_at = self.len - self.tail.len() as u64;
        if self.pos >= tail_at {
            let from = (self.pos - tail_at) as usize;
            let n = buf.len().min(self.tail.len() - from);
            buf[..n].copy_from_slice(&self.tail[from..from + n]);
            self.pos += n as u64;
            return Ok(n);
        }
        // A little way ahead within the open response (a member's data right
        // after its local header): read through to it rather than ask again.
        if let Some((s, at, end)) = self.stream.as_mut() {
            if *at < self.pos && self.pos < *end && self.pos - *at <= 1 << 20 {
                let skip = self.pos - *at;
                let n = std::io::copy(&mut (&mut *s).take(skip), &mut std::io::sink())?;
                self.received += n;
                *at += n;
            }
        }
        let live = matches!(&self.stream, Some((_, at, end)) if *at == self.pos && self.pos < *end);
        if !live {
            // Continuing straight on from the last request: ask for more.
            let sequential = matches!(&self.stream, Some((_, _, end)) if *end == self.pos);
            self.chunk = if sequential {
                (self.chunk * 2).min(MAX_CHUNK)
            } else {
                FIRST_CHUNK
            };
            // Never through the stream what the cached tail already holds.
            let end = (self.pos + self.chunk).min(tail_at);
            let s = self.request(self.pos, end)?;
            self.stream = Some((s, self.pos, end));
        }
        let (s, at, end) = self.stream.as_mut().expect("opened above");
        let want = buf.len().min((*end - *at) as usize);
        let n = s.read(&mut buf[..want])?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "the range response ended early",
            ));
        }
        *at += n as u64;
        self.pos += n as u64;
        self.received += n as u64;
        Ok(n)
    }
}

impl Seek for HttpRange {
    fn seek(&mut self, to: SeekFrom) -> std::io::Result<u64> {
        let p = match to {
            SeekFrom::Start(p) => p as i128,
            SeekFrom::End(d) => self.len as i128 + d as i128,
            SeekFrom::Current(d) => self.pos as i128 + d as i128,
        };
        if p < 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "seek before the start",
            ));
        }
        self.pos = p as u64;
        Ok(self.pos)
    }
}

/// One member to take out of an archive: chosen by [`fetch_members`]'s
/// `pick` from the archive's entry names, written to `dest`.
pub struct Member {
    pub name: String,
    pub dest: PathBuf,
}

/// Copy the archive members `pick` chooses (given every entry name) to
/// their destinations. Through range requests when the server allows them;
/// otherwise the whole archive is downloaded to `scratch` first and removed
/// afterwards. Returns whether ranges were used.
pub fn fetch_members(
    url: &str,
    full_bytes: u64,
    scratch: &Path,
    label: &str,
    pick: &dyn Fn(&[String]) -> Result<Vec<Member>>,
    sink: &dyn ProgressSink,
) -> Result<bool> {
    let agent = super::cache::download_agent();
    sink.report(0.0, &format!("Reading the archive index ({label})"));
    if let Some(mut remote) = HttpRange::open(&agent, url)? {
        let len = remote.len();
        extract(&mut remote, len, label, pick, sink)?;
        return Ok(true);
    }
    super::cache::download_with(&agent, url, scratch, full_bytes, label, sink)?;
    let res = (|| {
        let mut f = std::fs::File::open(scratch)?;
        let len = f.metadata()?.len();
        extract(&mut f, len, label, pick, sink)
    })();
    let _ = std::fs::remove_file(scratch);
    res.map(|()| false)
}

/// One entry of a zip's central directory.
#[derive(Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    /// 0 stored, 8 deflated.
    pub method: u16,
    pub compressed: u64,
    pub size: u64,
    pub crc: u32,
    /// Offset of its local header.
    pub header: u64,
}

fn u16_at(b: &[u8], i: usize) -> u16 {
    u16::from_le_bytes([b[i], b[i + 1]])
}
fn u32_at(b: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(b[i..i + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

fn read_at<R: Read + Seek>(r: &mut R, at: u64, n: usize) -> Result<Vec<u8>> {
    r.seek(SeekFrom::Start(at))?;
    let mut b = vec![0u8; n];
    r.read_exact(&mut b).context("read the archive")?;
    Ok(b)
}

/// The central directory of the zip `r` (of `len` bytes), read from its end
/// (ZIP64 included) without touching the entries themselves. The `zip`
/// crate validates every entry's local header when it opens an archive,
/// which on a remote archive would be one request per entry; this reads
/// the end records and the directory, and nothing else.
pub fn central_directory<R: Read + Seek>(r: &mut R, len: u64) -> Result<Vec<Entry>> {
    const EOCD: u32 = 0x0605_4b50;
    let span = len.min(22 + 65_535);
    let end = read_at(r, len - span, span as usize)?;
    let at = (0..end.len().saturating_sub(21))
        .rev()
        .find(|&i| u32_at(&end, i) == EOCD)
        .context("not a zip archive (no end of central directory)")?;
    let e = &end[at..];
    let mut count = u16_at(e, 10) as u64;
    let mut cd_size = u32_at(e, 12) as u64;
    let mut cd_off = u32_at(e, 16) as u64;
    if count == 0xFFFF || cd_size == 0xFFFF_FFFF || cd_off == 0xFFFF_FFFF {
        let eocd_at = len - span + at as u64;
        let loc = read_at(r, eocd_at.checked_sub(20).context("zip64 locator")?, 20)?;
        if u32_at(&loc, 0) != 0x0706_4b50 {
            bail!("zip64 archive without its locator");
        }
        let rec = read_at(r, u64_at(&loc, 8), 56)?;
        if u32_at(&rec, 0) != 0x0606_4b50 {
            bail!("zip64 end record missing");
        }
        count = u64_at(&rec, 32);
        cd_size = u64_at(&rec, 40);
        cd_off = u64_at(&rec, 48);
    }
    if cd_off + cd_size > len {
        bail!("the central directory lies past the end of the archive");
    }
    let cd = read_at(r, cd_off, cd_size as usize)?;
    let mut out = Vec::with_capacity(count as usize);
    let mut i = 0usize;
    while i + 46 <= cd.len() && u32_at(&cd, i) == 0x0201_4b50 {
        let method = u16_at(&cd, i + 10);
        let crc = u32_at(&cd, i + 16);
        let mut compressed = u32_at(&cd, i + 20) as u64;
        let mut size = u32_at(&cd, i + 24) as u64;
        let n = u16_at(&cd, i + 28) as usize;
        let x = u16_at(&cd, i + 30) as usize;
        let k = u16_at(&cd, i + 32) as usize;
        let mut header = u32_at(&cd, i + 42) as u64;
        let name = String::from_utf8_lossy(&cd[i + 46..i + 46 + n]).into_owned();
        // ZIP64 extended information: the 64-bit values of the fields set
        // to all ones, in this order.
        let mut j = i + 46 + n;
        let xe = j + x;
        while j + 4 <= xe {
            let id = u16_at(&cd, j);
            let l = u16_at(&cd, j + 2) as usize;
            if id == 0x0001 {
                let mut p = j + 4;
                for f in [&mut size, &mut compressed, &mut header] {
                    if *f == 0xFFFF_FFFF && p + 8 <= j + 4 + l {
                        *f = u64_at(&cd, p);
                        p += 8;
                    }
                }
            }
            j += 4 + l;
        }
        out.push(Entry {
            name,
            method,
            compressed,
            size,
            crc,
            header,
        });
        i += 46 + n + x + k;
    }
    if out.len() as u64 != count {
        bail!(
            "the central directory lists {} of {count} entries",
            out.len()
        );
    }
    Ok(out)
}

fn extract<R: Read + Seek>(
    r: &mut R,
    len: u64,
    label: &str,
    pick: &dyn Fn(&[String]) -> Result<Vec<Member>>,
    sink: &dyn ProgressSink,
) -> Result<()> {
    let entries = central_directory(r, len)?;
    let names: Vec<String> = entries.iter().map(|e| e.name.clone()).collect();
    let members = pick(&names)?;
    let chosen: Vec<(&Member, &Entry)> = members
        .iter()
        .map(|m| {
            entries
                .iter()
                .find(|e| e.name == m.name)
                .map(|e| (m, e))
                .with_context(|| format!("archive member {}", m.name))
        })
        .collect::<Result<_>>()?;
    let total: u64 = chosen.iter().map(|(_, e)| e.compressed).sum::<u64>().max(1);
    let mut done = 0u64;
    for (m, e) in chosen {
        if e.method != 0 && e.method != 8 {
            bail!(
                "{}: compression method {} is not supported",
                m.name,
                e.method
            );
        }
        let local = read_at(r, e.header, 30)?;
        if u32_at(&local, 0) != 0x0403_4b50 {
            bail!("{}: no local header where the directory says", m.name);
        }
        let start = e.header + 30 + u16_at(&local, 26) as u64 + u16_at(&local, 28) as u64;
        r.seek(SeekFrom::Start(start))?;
        let raw = (&mut *r).take(e.compressed);
        let mut data: Box<dyn Read + '_> = if e.method == 8 {
            Box::new(flate2::read::DeflateDecoder::new(raw))
        } else {
            Box::new(raw)
        };
        if let Some(dir) = m.dest.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
        }
        let tmp = m.dest.with_extension("part");
        let mut out = std::io::BufWriter::new(
            std::fs::File::create(&tmp).with_context(|| format!("create {}", tmp.display()))?,
        );
        let mut crc = crc32fast::Hasher::new();
        let mut buf = vec![0u8; 1 << 20];
        let mut got = 0u64;
        let file = m.name.rsplit('/').next().unwrap_or(&m.name).to_string();
        let copied = (|| -> Result<()> {
            loop {
                if sink.cancelled() {
                    bail!(CANCELLED);
                }
                let n = data
                    .read(&mut buf)
                    .with_context(|| format!("read {}", m.name))?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                crc.update(&buf[..n]);
                got += n as u64;
                // Uncompressed bytes against the compressed total: close
                // enough for the weights, which deflate hardly shrinks.
                let frac = (done + got.min(e.compressed)) as f32 / total as f32;
                sink.report(
                    frac.min(1.0),
                    &format!(
                        "Downloading {label}: {file} {} / {} MB",
                        got / 1_000_000,
                        e.size / 1_000_000
                    ),
                );
            }
            out.flush()?;
            if got != e.size || crc.clone().finalize() != e.crc {
                bail!(
                    "{}: the unpacked member does not match its checksum",
                    m.name
                );
            }
            Ok(())
        })();
        drop(out);
        if let Err(err) = copied {
            let _ = std::fs::remove_file(&tmp);
            return Err(err);
        }
        std::fs::rename(&tmp, &m.dest)?;
        done += e.compressed;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::{BufRead, BufReader};
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// A one-file HTTP/1.1 server on the loopback interface: `GET /f`
    /// answers with the bytes, honouring `Range` when `ranges` is set.
    /// Returns the URL and the count of body bytes sent.
    pub(crate) fn serve(bytes: Vec<u8>, ranges: bool) -> (String, Arc<AtomicU64>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let sent = Arc::new(AtomicU64::new(0));
        let counter = sent.clone();
        let data = Arc::new(bytes);
        std::thread::spawn(move || {
            for conn in listener.incoming() {
                let Ok(mut conn) = conn else { continue };
                let data = data.clone();
                let counter = counter.clone();
                std::thread::spawn(move || {
                    let mut reader = BufReader::new(conn.try_clone().unwrap());
                    loop {
                        let mut range: Option<String> = None;
                        let mut first = String::new();
                        if reader.read_line(&mut first).unwrap_or(0) == 0 {
                            return;
                        }
                        loop {
                            let mut h = String::new();
                            if reader.read_line(&mut h).unwrap_or(0) == 0 {
                                return;
                            }
                            let h = h.trim_end();
                            if h.is_empty() {
                                break;
                            }
                            if let Some((k, v)) = h.split_once(':') {
                                if k.eq_ignore_ascii_case("range") {
                                    range = Some(v.trim().to_string());
                                }
                            }
                        }
                        let n = data.len() as u64;
                        let span = range.filter(|_| ranges).and_then(|r| {
                            let r = r.strip_prefix("bytes=")?;
                            let (a, b) = r.split_once('-')?;
                            Some(if a.is_empty() {
                                let k: u64 = b.parse().ok()?;
                                (n.saturating_sub(k), n - 1)
                            } else {
                                let a: u64 = a.parse().ok()?;
                                let b = if b.is_empty() { n - 1 } else { b.parse().ok()? };
                                (a, b.min(n - 1))
                            })
                        });
                        let (head, body) = match span {
                            Some((a, b)) => (
                                format!(
                                    "HTTP/1.1 206 Partial Content\r\nContent-Length: {}\r\n\
                                     Content-Range: bytes {a}-{b}/{n}\r\n\r\n",
                                    b - a + 1
                                ),
                                &data[a as usize..=b as usize],
                            ),
                            None => (
                                format!("HTTP/1.1 200 OK\r\nContent-Length: {n}\r\n\r\n"),
                                &data[..],
                            ),
                        };
                        if conn.write_all(head.as_bytes()).is_err() {
                            return;
                        }
                        // Write in pieces so a client that hangs up early is
                        // noticed and the count stays close to what was read.
                        for piece in body.chunks(64 * 1024) {
                            if conn.write_all(piece).is_err() {
                                return;
                            }
                            counter.fetch_add(piece.len() as u64, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        (format!("http://{addr}/f"), sent)
    }

    struct Quiet;
    impl ProgressSink for Quiet {
        fn report(&self, _: f32, _: &str) {}
        fn cancelled(&self) -> bool {
            false
        }
    }

    /// An archive with a small stored member, a small deflated one, and a
    /// large one of noise (incompressible, like weights) in between.
    pub(crate) fn sample_zip() -> (Vec<u8>, Vec<u8>) {
        let mut big = vec![0u8; 3_000_000];
        let mut s: u64 = 0x1234_5678;
        for b in big.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *b = (s >> 56) as u8;
        }
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip::ZipWriter::new(&mut buf);
            let stored: zip::write::FileOptions<()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            let deflated: zip::write::FileOptions<()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            z.start_file("Task/plans.txt", stored).unwrap();
            z.write_all(b"the plans").unwrap();
            z.start_file("Task/fold_0/big.model", stored).unwrap();
            z.write_all(&big).unwrap();
            z.start_file("Task/fold_1/big.model", stored).unwrap();
            z.write_all(&big).unwrap();
            z.start_file("Task/post.json", deflated).unwrap();
            z.write_all(&b"{\"x\": 1}".repeat(100)).unwrap();
            z.finish().unwrap();
        }
        (buf.into_inner(), big)
    }

    fn pick_two(dir: &Path) -> impl Fn(&[String]) -> Result<Vec<Member>> + '_ {
        move |names: &[String]| {
            assert!(names.iter().any(|n| n == "Task/fold_1/big.model"));
            Ok(vec![
                Member {
                    name: "Task/plans.txt".into(),
                    dest: dir.join("plans.txt"),
                },
                Member {
                    name: "Task/post.json".into(),
                    dest: dir.join("post.json"),
                },
            ])
        }
    }

    #[test]
    fn members_come_through_range_requests_without_the_rest() {
        let (zip_bytes, big) = sample_zip();
        let (url, sent) = serve(zip_bytes.clone(), true);
        let dir = std::env::temp_dir().join("rds_remote_zip_ranges");
        let _ = std::fs::remove_dir_all(&dir);
        let pick = pick_two(&dir);
        let ranged = fetch_members(
            &url,
            zip_bytes.len() as u64,
            &dir.join("all.zip"),
            "test",
            &pick,
            &Quiet,
        )
        .unwrap();
        assert!(ranged);
        assert_eq!(std::fs::read(dir.join("plans.txt")).unwrap(), b"the plans");
        assert_eq!(
            std::fs::read(dir.join("post.json")).unwrap(),
            b"{\"x\": 1}".repeat(100)
        );
        // Neither 3 MB member came over the wire: the end of the file and
        // one first-size request at its start, no more.
        let n = sent.load(Ordering::Relaxed);
        assert!(
            n <= TAIL + FIRST_CHUNK,
            "{n} bytes sent for a few hundred wanted"
        );

        // A large member, read through the stream.
        let pick_big = |_: &[String]| -> Result<Vec<Member>> {
            Ok(vec![Member {
                name: "Task/fold_0/big.model".into(),
                dest: dir.join("big.model"),
            }])
        };
        assert!(fetch_members(&url, 0, &dir.join("all.zip"), "test", &pick_big, &Quiet).unwrap());
        assert_eq!(std::fs::read(dir.join("big.model")).unwrap(), big);
        assert!(!dir.join("all.zip").exists());
    }

    #[test]
    fn a_server_without_ranges_gets_one_whole_download() {
        let (zip_bytes, _) = sample_zip();
        let (url, sent) = serve(zip_bytes.clone(), false);
        let dir = std::env::temp_dir().join("rds_remote_zip_whole");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let pick = pick_two(&dir);
        let ranged = fetch_members(
            &url,
            zip_bytes.len() as u64,
            &dir.join("all.zip"),
            "test",
            &pick,
            &Quiet,
        )
        .unwrap();
        assert!(!ranged);
        assert_eq!(std::fs::read(dir.join("plans.txt")).unwrap(), b"the plans");
        assert!(sent.load(Ordering::Relaxed) >= zip_bytes.len() as u64);
        // The scratch copy is gone.
        assert!(!dir.join("all.zip").exists());
    }

    #[test]
    fn the_directory_is_read_from_the_end_records_zip64_too() {
        let (zip_bytes, _) = sample_zip();
        let mut c = std::io::Cursor::new(zip_bytes.clone());
        let e = central_directory(&mut c, zip_bytes.len() as u64).unwrap();
        let names: Vec<&str> = e.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "Task/plans.txt",
                "Task/fold_0/big.model",
                "Task/fold_1/big.model",
                "Task/post.json"
            ]
        );
        assert_eq!(e[1].size, 3_000_000);
        assert_eq!((e[0].method, e[3].method), (0, 8));
        // The same members in a ZIP64 archive (every entry forced large).
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut z = zip::ZipWriter::new(&mut buf);
            let o: zip::write::FileOptions<()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored)
                .large_file(true);
            z.start_file("a.txt", o).unwrap();
            z.write_all(b"first").unwrap();
            z.start_file("b/c.bin", o).unwrap();
            z.write_all(&[7u8; 1000]).unwrap();
            z.finish().unwrap();
        }
        let bytes = buf.into_inner();
        let mut c = std::io::Cursor::new(bytes.clone());
        let e = central_directory(&mut c, bytes.len() as u64).unwrap();
        assert_eq!(e.len(), 2);
        assert_eq!((e[1].name.as_str(), e[1].size), ("b/c.bin", 1000));
        // And they unpack, checksums checked.
        let dir = std::env::temp_dir().join("rds_remote_zip64");
        let _ = std::fs::remove_dir_all(&dir);
        let pick = |_: &[String]| -> Result<Vec<Member>> {
            Ok(vec![Member {
                name: "b/c.bin".into(),
                dest: dir.join("c.bin"),
            }])
        };
        let mut c = std::io::Cursor::new(bytes.clone());
        extract(&mut c, bytes.len() as u64, "t", &pick, &Quiet).unwrap();
        assert_eq!(std::fs::read(dir.join("c.bin")).unwrap(), vec![7u8; 1000]);
        // A damaged member is refused, and nothing is left behind.
        let mut bad = bytes.clone();
        let at = bad.windows(5).position(|w| w == b"first").unwrap();
        bad[at] = b'F';
        let pick_a = |_: &[String]| -> Result<Vec<Member>> {
            Ok(vec![Member {
                name: "a.txt".into(),
                dest: dir.join("a.txt"),
            }])
        };
        let mut c = std::io::Cursor::new(bad.clone());
        assert!(extract(&mut c, bad.len() as u64, "t", &pick_a, &Quiet).is_err());
        assert!(!dir.join("a.txt").exists() && !dir.join("a.part").exists());
    }

    #[test]
    fn content_range_headers_parse() {
        assert_eq!(content_range("bytes 10-19/100"), Some((10, Some(100))));
        assert_eq!(content_range("bytes 0-0/*"), Some((0, None)));
        assert_eq!(content_range("items 1-2/3"), None);
    }
}

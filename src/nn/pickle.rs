//! Minimal pure-Rust reader for PyTorch checkpoint files (`.pth`).
//!
//! A torch checkpoint is a ZIP archive containing `<prefix>/data.pkl` (a
//! Python pickle, protocol 2, describing the object tree) plus one raw
//! little-endian storage blob per tensor under `<prefix>/data/<key>`.
//!
//! This module implements just enough of the pickle virtual machine to walk
//! such a checkpoint and extract a state dict — every tensor's storage key,
//! dtype, shape, stride and storage offset — either from the archive root
//! (a bare `state_dict()`, as Hugging Face's `pytorch_model.bin` files are
//! saved) or from a named entry of the root dict (`network_weights` in an
//! nnU-Net training checkpoint). No Python, no libtorch: the opcodes below
//! cover everything `torch.save` (protocol 2) emits for these files, plus a
//! few protocol-4 opcodes for robustness.
//!
//! Reference: CPython `pickletools` / `pickle.py`, and
//! `torch/serialization.py` (`_rebuild_tensor_v2`, persistent IDs of the form
//! `('storage', <StorageClass>, key, device, numel)`).

use anyhow::{bail, Context, Result};
use std::cell::RefCell;
use std::collections::HashMap;
use std::io::Read;
use std::path::Path;
use std::rc::Rc;

use super::half::f16_to_f32;

/// Element type of a torch storage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    F32,
    F64,
    F16,
    I64,
    I32,
    I16,
    U8,
    Bool,
}

impl Dtype {
    fn from_storage_class(name: &str) -> Option<Dtype> {
        Some(match name {
            "FloatStorage" => Dtype::F32,
            "DoubleStorage" => Dtype::F64,
            "HalfStorage" => Dtype::F16,
            "LongStorage" => Dtype::I64,
            "IntStorage" => Dtype::I32,
            "ShortStorage" => Dtype::I16,
            "ByteStorage" | "CharStorage" => Dtype::U8,
            "BoolStorage" => Dtype::Bool,
            _ => return None,
        })
    }
    pub fn size(self) -> usize {
        match self {
            Dtype::F64 | Dtype::I64 => 8,
            Dtype::F32 | Dtype::I32 => 4,
            Dtype::F16 | Dtype::I16 => 2,
            Dtype::U8 | Dtype::Bool => 1,
        }
    }
}

/// Description of one tensor inside the checkpoint (data still on disk).
#[derive(Clone, Debug)]
pub struct TensorMeta {
    pub storage_key: String,
    pub dtype: Dtype,
    pub storage_numel: usize,
    pub storage_offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
}

impl TensorMeta {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
    /// Row-major (C-contiguous) check — true for all nnU-Net weights.
    pub fn is_contiguous(&self) -> bool {
        let mut expect = 1usize;
        for (dim, st) in self.shape.iter().zip(self.stride.iter()).rev() {
            if *dim > 1 && *st != expect {
                return false;
            }
            expect *= *dim;
        }
        true
    }
}

// ---- pickle value model -------------------------------------------------

type List = Rc<RefCell<Vec<Value>>>;
type Dict = Rc<RefCell<Vec<(Value, Value)>>>;

#[derive(Clone, Debug)]
enum Value {
    None,
    /// Payload kept for pickle-VM fidelity even where never inspected.
    #[allow(dead_code)]
    Bool(bool),
    Int(i64),
    #[allow(dead_code)]
    Float(f64),
    Str(Rc<str>),
    #[allow(dead_code)]
    Bytes(Rc<[u8]>),
    Tuple(Rc<Vec<Value>>),
    List(List),
    Dict(Dict),
    Global(Rc<str>, Rc<str>),
    /// Persistent-ID storage reference: (key, dtype, numel).
    Storage(Rc<str>, Dtype, usize),
    Tensor(Rc<TensorMeta>),
    /// Anything we do not need (numpy scalars in the training log, …).
    Opaque,
    Mark,
}

impl Value {
    fn as_usize(&self) -> Result<usize> {
        match self {
            Value::Int(v) if *v >= 0 => Ok(*v as usize),
            _ => bail!("expected non-negative int, got {:?}", self),
        }
    }
}

fn tuple_usizes(v: &Value) -> Result<Vec<usize>> {
    match v {
        Value::Tuple(items) => items.iter().map(|i| i.as_usize()).collect(),
        _ => bail!("expected tuple of ints, got {:?}", v),
    }
}

// ---- the pickle machine -------------------------------------------------

struct Machine<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Value>,
    memo: HashMap<u32, Value>,
}

impl<'a> Machine<'a> {
    fn u8(&mut self) -> Result<u8> {
        let b = *self.data.get(self.pos).context("pickle: eof")?;
        self.pos += 1;
        Ok(b)
    }
    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self
            .data
            .get(self.pos..self.pos + n)
            .context("pickle: eof")?;
        self.pos += n;
        Ok(s)
    }
    fn u16le(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()))
    }
    fn u32le(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn i32le(&mut self) -> Result<i32> {
        Ok(i32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64le(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn line(&mut self) -> Result<&'a str> {
        let start = self.pos;
        while *self.data.get(self.pos).context("pickle: eof")? != b'\n' {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.data[start..self.pos]).context("pickle: bad utf8")?;
        self.pos += 1;
        Ok(s)
    }
    fn pop(&mut self) -> Result<Value> {
        self.stack.pop().context("pickle: stack underflow")
    }
    /// Pop values above the topmost MARK; remove the MARK too.
    fn pop_mark(&mut self) -> Result<Vec<Value>> {
        let mark = self
            .stack
            .iter()
            .rposition(|v| matches!(v, Value::Mark))
            .context("pickle: no mark")?;
        let items = self.stack.split_off(mark + 1);
        self.stack.pop(); // the mark itself
        Ok(items)
    }

    fn reduce(&mut self, callable: Value, args: Value) -> Result<Value> {
        let (module, name) = match &callable {
            Value::Global(m, n) => (m.as_ref(), n.as_ref()),
            // e.g. the result of a previous REDUCE used as a callable — the
            // training-log section does this with numpy dtypes; irrelevant.
            _ => return Ok(Value::Opaque),
        };
        match (module, name) {
            ("collections", "OrderedDict") => Ok(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
            ("torch._utils", "_rebuild_tensor_v2") => {
                let args = match args {
                    Value::Tuple(t) => t,
                    _ => bail!("_rebuild_tensor_v2: args not a tuple"),
                };
                if args.len() < 4 {
                    bail!("_rebuild_tensor_v2: expected >=4 args, got {}", args.len());
                }
                let (key, dtype, numel) = match &args[0] {
                    Value::Storage(k, d, n) => (k.clone(), *d, *n),
                    other => bail!("_rebuild_tensor_v2: arg0 not a storage ({:?})", other),
                };
                Ok(Value::Tensor(Rc::new(TensorMeta {
                    storage_key: key.to_string(),
                    dtype,
                    storage_numel: numel,
                    storage_offset: args[1].as_usize()?,
                    shape: tuple_usizes(&args[2])?,
                    stride: tuple_usizes(&args[3])?,
                })))
            }
            ("torch", "device") => Ok(Value::Opaque),
            // numpy machinery in the training log — value never inspected.
            _ => Ok(Value::Opaque),
        }
    }

    fn persistent_load(&mut self, pid: Value) -> Result<Value> {
        // torch: ('storage', StorageClass, key: str, device: str, numel: int)
        if let Value::Tuple(items) = &pid {
            if items.len() >= 5 {
                if let (Value::Str(tag), Value::Global(m, cls), Value::Str(key)) =
                    (&items[0], &items[1], &items[2])
                {
                    if tag.as_ref() == "storage" && m.as_ref() == "torch" {
                        let dtype = Dtype::from_storage_class(cls)
                            .with_context(|| format!("unsupported torch storage class {cls}"))?;
                        let numel = items[4].as_usize()?;
                        return Ok(Value::Storage(key.clone(), dtype, numel));
                    }
                }
            }
        }
        bail!("unsupported persistent id: {:?}", pid)
    }

    fn run(&mut self) -> Result<Value> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    let _proto = self.u8()?;
                }
                b'.' => return self.pop(), // STOP
                b'(' => self.stack.push(Value::Mark),
                b'}' => self
                    .stack
                    .push(Value::Dict(Rc::new(RefCell::new(Vec::new())))),
                b']' => self
                    .stack
                    .push(Value::List(Rc::new(RefCell::new(Vec::new())))),
                b')' => self.stack.push(Value::Tuple(Rc::new(Vec::new()))),
                b'N' => self.stack.push(Value::None),
                0x88 => self.stack.push(Value::Bool(true)), // NEWTRUE
                0x89 => self.stack.push(Value::Bool(false)), // NEWFALSE
                b'J' => {
                    let v = self.i32le()?;
                    self.stack.push(Value::Int(v as i64));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Value::Int(v as i64));
                }
                b'M' => {
                    let v = self.u16le()?;
                    self.stack.push(Value::Int(v as i64));
                }
                0x8a => {
                    // LONG1: n bytes little-endian signed
                    let n = self.u8()? as usize;
                    let raw = self.bytes(n)?;
                    let mut v: i64 = 0;
                    for (i, b) in raw.iter().enumerate().take(8) {
                        v |= (*b as i64) << (8 * i);
                    }
                    if n > 0 && n <= 8 && raw[n - 1] & 0x80 != 0 && n < 8 {
                        v |= -1i64 << (8 * n); // sign-extend
                    }
                    self.stack.push(Value::Int(v));
                }
                b'G' => {
                    // BINFLOAT: big-endian f64
                    let raw = self.bytes(8)?;
                    self.stack
                        .push(Value::Float(f64::from_be_bytes(raw.try_into().unwrap())));
                }
                b'X' => {
                    let n = self.u32le()? as usize;
                    let s = std::str::from_utf8(self.bytes(n)?).context("bad utf8")?;
                    self.stack.push(Value::Str(Rc::from(s)));
                }
                0x8c => {
                    // SHORT_BINUNICODE
                    let n = self.u8()? as usize;
                    let s = std::str::from_utf8(self.bytes(n)?).context("bad utf8")?;
                    self.stack.push(Value::Str(Rc::from(s)));
                }
                b'U' => {
                    // SHORT_BINSTRING
                    let n = self.u8()? as usize;
                    let raw = self.bytes(n)?;
                    self.stack.push(Value::Bytes(Rc::from(raw)));
                }
                b'T' => {
                    // BINSTRING
                    let n = self.u32le()? as usize;
                    let raw = self.bytes(n)?;
                    self.stack.push(Value::Bytes(Rc::from(raw)));
                }
                0x8d => {
                    // BINUNICODE8
                    let n = self.u64le()? as usize;
                    let s = std::str::from_utf8(self.bytes(n)?).context("bad utf8")?;
                    self.stack.push(Value::Str(Rc::from(s)));
                }
                b'c' => {
                    // GLOBAL: two newline-terminated strings
                    let module = self.line()?.to_owned();
                    let name = self.line()?.to_owned();
                    self.stack.push(Value::Global(
                        Rc::from(module.as_str()),
                        Rc::from(name.as_str()),
                    ));
                }
                0x93 => {
                    // STACK_GLOBAL
                    let name = self.pop()?;
                    let module = self.pop()?;
                    match (module, name) {
                        (Value::Str(m), Value::Str(n)) => self.stack.push(Value::Global(m, n)),
                        _ => bail!("STACK_GLOBAL: non-string operands"),
                    }
                }
                b'q' => {
                    let id = self.u8()? as u32;
                    let top = self.stack.last().context("BINPUT: empty stack")?.clone();
                    self.memo.insert(id, top);
                }
                b'r' => {
                    let id = self.u32le()?;
                    let top = self
                        .stack
                        .last()
                        .context("LONG_BINPUT: empty stack")?
                        .clone();
                    self.memo.insert(id, top);
                }
                0x94 => {
                    // MEMOIZE (protocol 4): key = current memo size
                    let id = self.memo.len() as u32;
                    let top = self.stack.last().context("MEMOIZE: empty stack")?.clone();
                    self.memo.insert(id, top);
                }
                b'h' => {
                    let id = self.u8()? as u32;
                    let v = self.memo.get(&id).context("BINGET: missing memo")?.clone();
                    self.stack.push(v);
                }
                b'j' => {
                    let id = self.u32le()?;
                    let v = self
                        .memo
                        .get(&id)
                        .context("LONG_BINGET: missing memo")?
                        .clone();
                    self.stack.push(v);
                }
                0x85 => {
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a])));
                }
                0x86 => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a, b])));
                }
                0x87 => {
                    let c = self.pop()?;
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(Value::Tuple(Rc::new(vec![a, b, c])));
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::Tuple(Rc::new(items)));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Value::List(Rc::new(RefCell::new(items))));
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let mut pairs = Vec::with_capacity(items.len() / 2);
                    let mut it = items.into_iter();
                    while let (Some(k), Some(v)) = (it.next(), it.next()) {
                        pairs.push((k, v));
                    }
                    self.stack.push(Value::Dict(Rc::new(RefCell::new(pairs))));
                }
                b'a' => {
                    // APPEND
                    let v = self.pop()?;
                    match self.stack.last() {
                        Some(Value::List(l)) => l.borrow_mut().push(v),
                        _ => bail!("APPEND: no list on stack"),
                    }
                }
                b'e' => {
                    // APPENDS
                    let items = self.pop_mark()?;
                    match self.stack.last() {
                        Some(Value::List(l)) => l.borrow_mut().extend(items),
                        _ => bail!("APPENDS: no list on stack"),
                    }
                }
                b's' => {
                    // SETITEM
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.stack.last() {
                        Some(Value::Dict(d)) => d.borrow_mut().push((k, v)),
                        Some(Value::Opaque) => {}
                        _ => bail!("SETITEM: no dict on stack"),
                    }
                }
                b'u' => {
                    // SETITEMS
                    let items = self.pop_mark()?;
                    match self.stack.last() {
                        Some(Value::Dict(d)) => {
                            let mut d = d.borrow_mut();
                            let mut it = items.into_iter();
                            while let (Some(k), Some(v)) = (it.next(), it.next()) {
                                d.push((k, v));
                            }
                        }
                        Some(Value::Opaque) => {}
                        _ => bail!("SETITEMS: no dict on stack"),
                    }
                }
                b'R' => {
                    // REDUCE
                    let args = self.pop()?;
                    let callable = self.pop()?;
                    let v = self.reduce(callable, args)?;
                    self.stack.push(v);
                }
                0x81 => {
                    // NEWOBJ: cls.__new__(cls, *args) — treat like REDUCE
                    let args = self.pop()?;
                    let cls = self.pop()?;
                    let v = self.reduce(cls, args)?;
                    self.stack.push(v);
                }
                b'b' => {
                    // BUILD: obj.__setstate__(state) — state unused for our targets
                    let _state = self.pop()?;
                }
                b'Q' => {
                    // BINPERSID
                    let pid = self.pop()?;
                    let v = self.persistent_load(pid)?;
                    self.stack.push(v);
                }
                0x95 => {
                    // FRAME (protocol 4): 8-byte length, ignore
                    let _ = self.u64le()?;
                }
                b'2' => {
                    // DUP
                    let top = self.stack.last().context("DUP: empty stack")?.clone();
                    self.stack.push(top);
                }
                b'0' => {
                    // POP
                    let _ = self.pop()?;
                }
                b'1' => {
                    // POP_MARK
                    let _ = self.pop_mark()?;
                }
                other => bail!(
                    "unsupported pickle opcode 0x{other:02x} at offset {}",
                    self.pos - 1
                ),
            }
        }
    }
}

// ---- public API ---------------------------------------------------------

/// All tensors of one state dict inside a checkpoint, plus access to the
/// raw storage bytes still inside the ZIP archive.
pub struct PthReader {
    archive: zip::ZipArchive<std::fs::File>,
    /// e.g. "checkpoint_final" — first path component inside the zip.
    prefix: String,
    /// state-dict entries in file order: (parameter name, tensor meta).
    pub tensors: Vec<(String, TensorMeta)>,
}

impl PthReader {
    /// Open a `.pth` checkpoint and extract the tensor table of the state
    /// dict found under `top_key` (e.g. `"network_weights"`). If `top_key` is
    /// empty the checkpoint root itself must be the state dict.
    pub fn open(path: &Path, top_key: &str) -> Result<PthReader> {
        let file = std::fs::File::open(path)
            .with_context(|| format!("open checkpoint {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file).context("checkpoint is not a zip archive")?;
        // find "<prefix>/data.pkl"
        let mut pkl_name = None;
        for i in 0..archive.len() {
            let name = archive.by_index_raw(i)?.name().to_owned();
            if name.ends_with("/data.pkl") || name == "data.pkl" {
                pkl_name = Some(name);
                break;
            }
        }
        let pkl_name = pkl_name.context("no data.pkl inside checkpoint")?;
        let prefix = pkl_name
            .strip_suffix("data.pkl")
            .unwrap()
            .trim_end_matches('/')
            .to_owned();
        let mut pkl = Vec::new();
        archive
            .by_name(&pkl_name)?
            .read_to_end(&mut pkl)
            .context("read data.pkl")?;
        let mut m = Machine {
            data: &pkl,
            pos: 0,
            stack: Vec::new(),
            memo: HashMap::new(),
        };
        let root = m.run().context("unpickle checkpoint")?;
        let state = if top_key.is_empty() {
            root
        } else {
            let Value::Dict(d) = &root else {
                bail!("checkpoint root is not a dict");
            };
            let d = d.borrow();
            d.iter()
                .find(|(k, _)| matches!(k, Value::Str(s) if s.as_ref() == top_key))
                .map(|(_, v)| v.clone())
                .with_context(|| format!("checkpoint has no '{top_key}' entry"))?
        };
        let Value::Dict(state) = &state else {
            bail!("state dict is not a dict");
        };
        let mut tensors = Vec::new();
        for (k, v) in state.borrow().iter() {
            if let (Value::Str(name), Value::Tensor(meta)) = (k, v) {
                tensors.push((name.to_string(), (**meta).clone()));
            }
        }
        if tensors.is_empty() {
            bail!("state dict contains no tensors");
        }
        Ok(PthReader {
            archive,
            prefix,
            tensors,
        })
    }

    /// Read one tensor as f32 (converting from its storage dtype), honoring
    /// storage offset and (contiguous) strides.
    pub fn read_f32(&mut self, meta: &TensorMeta) -> Result<Vec<f32>> {
        if !meta.is_contiguous() {
            bail!("non-contiguous tensor (storage {})", meta.storage_key);
        }
        let entry = if self.prefix.is_empty() {
            format!("data/{}", meta.storage_key)
        } else {
            format!("{}/data/{}", self.prefix, meta.storage_key)
        };
        let mut raw = Vec::new();
        self.archive
            .by_name(&entry)
            .with_context(|| format!("storage entry {entry}"))?
            .read_to_end(&mut raw)
            .context("read storage")?;
        let esize = meta.dtype.size();
        let start = meta.storage_offset * esize;
        let need = meta.numel() * esize;
        if raw.len() < start + need {
            bail!(
                "storage {} too small: {} < {}",
                meta.storage_key,
                raw.len(),
                start + need
            );
        }
        let raw = &raw[start..start + need];
        let mut out = Vec::with_capacity(meta.numel());
        match meta.dtype {
            Dtype::F32 => {
                for c in raw.chunks_exact(4) {
                    out.push(f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            Dtype::F64 => {
                for c in raw.chunks_exact(8) {
                    out.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
                }
            }
            Dtype::F16 => {
                for c in raw.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes(c.try_into().unwrap())));
                }
            }
            other => bail!("unsupported tensor dtype {:?}", other),
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- a synthetic torch checkpoint ----------------------------------
    //
    // `torch.save` writes a ZIP holding `archive/data.pkl` (a protocol-2
    // pickle) plus one raw little-endian blob per storage under
    // `archive/data/<key>`. These helpers emit exactly that, so the reader is
    // exercised end to end — including the root-level `state_dict()` layout
    // that Hugging Face checkpoints use — without PyTorch, and without the
    // 700 MB download.

    fn unicode(out: &mut Vec<u8>, s: &str) {
        out.push(b'X');
        out.extend_from_slice(&(s.len() as u32).to_le_bytes());
        out.extend_from_slice(s.as_bytes());
    }
    fn global_(out: &mut Vec<u8>, module: &str, name: &str) {
        out.push(b'c');
        out.extend_from_slice(module.as_bytes());
        out.push(b'\n');
        out.extend_from_slice(name.as_bytes());
        out.push(b'\n');
    }
    fn int_(out: &mut Vec<u8>, v: u64) {
        if v < 256 {
            out.push(b'K');
            out.push(v as u8);
        } else {
            out.push(b'J');
            out.extend_from_slice(&(v as u32).to_le_bytes());
        }
    }
    fn tuple_of(out: &mut Vec<u8>, vals: &[usize]) {
        out.push(b'(');
        for v in vals {
            int_(out, *v as u64);
        }
        out.push(b't');
    }

    struct Entry {
        name: &'static str,
        storage_key: &'static str,
        storage_class: &'static str,
        storage_numel: usize,
        offset: usize,
        shape: Vec<usize>,
        stride: Vec<usize>,
    }

    fn contiguous(name: &'static str, key: &'static str, shape: &[usize]) -> Entry {
        let mut stride = vec![1usize; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
        Entry {
            name,
            storage_key: key,
            storage_class: "FloatStorage",
            storage_numel: shape.iter().product(),
            offset: 0,
            shape: shape.to_vec(),
            stride,
        }
    }

    /// Emit the `_rebuild_tensor_v2(storage, offset, size, stride, ...)` call
    /// for one entry.
    fn push_tensor(out: &mut Vec<u8>, e: &Entry) {
        global_(out, "torch._utils", "_rebuild_tensor_v2");
        out.push(b'(');
        // persistent id: ('storage', torch.<Class>, key, 'cpu', numel)
        out.push(b'(');
        unicode(out, "storage");
        global_(out, "torch", e.storage_class);
        unicode(out, e.storage_key);
        unicode(out, "cpu");
        int_(out, e.storage_numel as u64);
        out.push(b't');
        out.push(b'Q'); // BINPERSID
        int_(out, e.offset as u64);
        tuple_of(out, &e.shape);
        tuple_of(out, &e.stride);
        out.push(0x89); // requires_grad = False
        out.push(b'}'); // backward_hooks = {}
        out.push(b't');
        out.push(b'R'); // REDUCE
    }

    /// Build a `.pth`. `top_key` empty means the archive root *is* the state
    /// dict; otherwise the root is `{top_key: <state dict>, "epoch": 7}`.
    fn build_pth(entries: &[Entry], storages: &[(&str, Vec<u8>)], top_key: &str) -> Vec<u8> {
        let mut pkl = vec![0x80, 0x02];
        if !top_key.is_empty() {
            pkl.push(b'}');
            pkl.push(b'(');
            unicode(&mut pkl, top_key);
        }
        pkl.push(b'}'); // the state dict itself
        pkl.push(b'(');
        for e in entries {
            unicode(&mut pkl, e.name);
            push_tensor(&mut pkl, e);
        }
        pkl.push(b'u'); // SETITEMS
        if !top_key.is_empty() {
            unicode(&mut pkl, "epoch");
            int_(&mut pkl, 7);
            pkl.push(b'u');
        }
        pkl.push(b'.'); // STOP

        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let opts: zip::write::FileOptions<()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Stored);
            zip.start_file("archive/data.pkl", opts).unwrap();
            std::io::Write::write_all(&mut zip, &pkl).unwrap();
            for (key, bytes) in storages {
                zip.start_file(format!("archive/data/{key}"), opts).unwrap();
                std::io::Write::write_all(&mut zip, bytes).unwrap();
            }
            zip.finish().unwrap();
        }
        buf.into_inner()
    }

    fn f32_bytes(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join("rds_pickle_tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn reads_a_root_level_state_dict() {
        // The Hugging Face layout: torch.save(model.state_dict(), ...) with no
        // wrapper dict. This is how the SegVol checkpoint is published.
        let vals: Vec<f32> = (0..6).map(|i| i as f32 * 0.5 - 1.0).collect();
        let entries = vec![
            contiguous("image_encoder.patch_embedding.weight", "0", &[2, 3]),
            contiguous("mask_decoder.iou_token.weight", "1", &[2]),
        ];
        let pth = build_pth(
            &entries,
            &[("0", f32_bytes(&vals)), ("1", f32_bytes(&[9.0, -9.0]))],
            "",
        );
        let path = write_temp("root.pth", &pth);
        let mut r = PthReader::open(&path, "").unwrap();
        assert_eq!(r.tensors.len(), 2);
        let names: Vec<&str> = r.tensors.iter().map(|(n, _)| n.as_str()).collect();
        assert_eq!(
            names,
            [
                "image_encoder.patch_embedding.weight",
                "mask_decoder.iou_token.weight"
            ]
        );
        let (_, meta) = r.tensors[0].clone();
        assert_eq!(meta.shape, [2, 3]);
        assert_eq!(meta.numel(), 6);
        assert!(meta.is_contiguous());
        assert_eq!(r.read_f32(&meta).unwrap(), vals);
        let (_, meta1) = r.tensors[1].clone();
        assert_eq!(r.read_f32(&meta1).unwrap(), [9.0, -9.0]);
    }

    #[test]
    fn reads_a_nested_state_dict_and_ignores_siblings() {
        // The nnU-Net training-checkpoint layout the auto-segmentation module
        // already relies on: {"network_weights": {...}, "epoch": 7}.
        let vals = vec![1.5f32, 2.5, 3.5, 4.5];
        let entries = vec![contiguous(
            "encoder.stages.0.0.convs.0.conv.weight",
            "0",
            &[4],
        )];
        let pth = build_pth(&entries, &[("0", f32_bytes(&vals))], "network_weights");
        let path = write_temp("nested.pth", &pth);
        let mut r = PthReader::open(&path, "network_weights").unwrap();
        assert_eq!(r.tensors.len(), 1);
        let (name, meta) = r.tensors[0].clone();
        assert_eq!(name, "encoder.stages.0.0.convs.0.conv.weight");
        assert_eq!(r.read_f32(&meta).unwrap(), vals);
        // asking for the root instead finds no tensors, since the root holds
        // only the nested dict and an int
        assert!(PthReader::open(&path, "").is_err());
        // and a key that is not there is an error, not a silent empty result
        assert!(PthReader::open(&path, "optimizer_state").is_err());
    }

    #[test]
    fn honors_storage_offset_and_shared_storages() {
        // Two tensors viewing one storage at different offsets — torch emits
        // this whenever parameters were sliced out of a single buffer.
        let all: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut a = contiguous("a", "0", &[4]);
        a.storage_numel = 10;
        let mut b = contiguous("b", "0", &[3]);
        b.storage_numel = 10;
        b.offset = 6;
        let pth = build_pth(&[a, b], &[("0", f32_bytes(&all))], "");
        let path = write_temp("offset.pth", &pth);
        let mut r = PthReader::open(&path, "").unwrap();
        let (_, ma) = r.tensors[0].clone();
        let (_, mb) = r.tensors[1].clone();
        assert_eq!(r.read_f32(&ma).unwrap(), [0.0, 1.0, 2.0, 3.0]);
        assert_eq!(r.read_f32(&mb).unwrap(), [6.0, 7.0, 8.0]);
    }

    #[test]
    fn widens_half_precision_storages() {
        // f16 checkpoints are read at full precision.
        let bits: Vec<u16> = vec![0x3c00, 0xc000, 0x0000]; // 1, -2, 0
        let raw: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let mut e = contiguous("half", "0", &[3]);
        e.storage_class = "HalfStorage";
        let pth = build_pth(&[e], &[("0", raw)], "");
        let path = write_temp("half.pth", &pth);
        let mut r = PthReader::open(&path, "").unwrap();
        let (_, meta) = r.tensors[0].clone();
        assert_eq!(meta.dtype, Dtype::F16);
        assert_eq!(r.read_f32(&meta).unwrap(), [1.0, -2.0, 0.0]);
    }

    #[test]
    fn rejects_non_contiguous_and_truncated_storages() {
        // A transposed view: strides do not match the shape's row-major order.
        let mut t = contiguous("t", "0", &[2, 3]);
        t.stride = vec![1, 2];
        let pth = build_pth(&[t], &[("0", f32_bytes(&[0.0; 6]))], "");
        let path = write_temp("noncontig.pth", &pth);
        let mut r = PthReader::open(&path, "").unwrap();
        let (_, meta) = r.tensors[0].clone();
        assert!(!meta.is_contiguous());
        let err = r.read_f32(&meta).unwrap_err().to_string();
        assert!(err.contains("non-contiguous"), "{err}");

        // A storage blob shorter than the shape demands must be an error, not
        // a panic or a silently short vector.
        let short = contiguous("s", "0", &[8]);
        let pth = build_pth(&[short], &[("0", f32_bytes(&[1.0, 2.0]))], "");
        let path = write_temp("short.pth", &pth);
        let mut r = PthReader::open(&path, "").unwrap();
        let (_, meta) = r.tensors[0].clone();
        assert!(r
            .read_f32(&meta)
            .unwrap_err()
            .to_string()
            .contains("too small"));
    }

    #[test]
    fn rejects_a_file_that_is_not_a_checkpoint() {
        let path = write_temp("garbage.pth", b"not a zip at all");
        assert!(PthReader::open(&path, "").is_err());
    }

    #[test]
    fn contiguity() {
        let m = TensorMeta {
            storage_key: "0".into(),
            dtype: Dtype::F32,
            storage_numel: 864,
            storage_offset: 0,
            shape: vec![32, 1, 3, 3, 3],
            stride: vec![27, 27, 9, 3, 1],
        };
        assert!(m.is_contiguous());
        assert_eq!(m.numel(), 864);
    }
}

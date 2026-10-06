//! Plain Python pickles (`.pkl`) as JSON values: dicts, lists, tuples,
//! numbers, strings, and the numpy arrays, scalars and dtypes that nnU-Net
//! v1 keeps its plans in (`plans.pkl`).
//!
//! This is a second, separate pickle machine. The one in [`super::pickle`]
//! walks torch checkpoints and only has to find tensors; this one has to
//! keep every value, numpy's included, which means honouring `BUILD` (a
//! numpy array is created empty by `_reconstruct` and filled by its state)
//! and memo references to mutable objects. Nothing is ever executed: the
//! callables a pickle names are matched against the handful understood
//! here, and anything else becomes `null`.
//!
//! The JSON is what a reader of the plans needs: dict keys become strings
//! (`0`, `(1, 2)` the way Python prints a tuple), tuples and sets become
//! arrays, a numpy array becomes nested arrays in C order, a numpy scalar a
//! number, a dtype its descriptor (`<f8`). Non-finite floats become `null`.

use anyhow::{bail, Context, Result};
use serde_json::{Map, Number, Value};
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Debug)]
enum Obj {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(Rc<str>),
    Bytes(Rc<Vec<u8>>),
    Tuple(Rc<Vec<Obj>>),
    List(Rc<RefCell<Vec<Obj>>>),
    Dict(Rc<RefCell<Vec<(Obj, Obj)>>>),
    Global(Rc<str>, Rc<str>),
    Dtype(Rc<RefCell<Dtype>>),
    Array(Rc<RefCell<Array>>),
    /// An object of a class not modelled here.
    Other,
    Mark,
}

#[derive(Clone, Debug)]
struct Dtype {
    /// `f8`, `i4`, `u1`, `b1`, `U3`, `O8`...
    code: String,
    /// `<`, `>`, `|` or `=`.
    order: char,
}

#[derive(Clone, Debug, Default)]
struct Array {
    shape: Vec<usize>,
    fortran: bool,
    dtype: Option<Dtype>,
    /// Raw bytes, or for object arrays the elements.
    raw: Option<Vec<u8>>,
    items: Option<Vec<Obj>>,
}

fn dtype_of(code: &str) -> Dtype {
    let (order, code) = match code.chars().next() {
        Some(c @ ('<' | '>' | '|' | '=')) => (c, &code[1..]),
        _ => ('=', code),
    };
    Dtype {
        code: code.to_string(),
        order,
    }
}

impl Dtype {
    fn big_endian(&self) -> bool {
        self.order == '>'
    }

    /// One element at `b` as a JSON number (or bool), when the type is
    /// numeric.
    fn decode(&self, b: &[u8]) -> Option<Value> {
        let be = self.big_endian();
        macro_rules! num {
            ($t:ty, $n:expr) => {{
                let a: [u8; $n] = b.get(..$n)?.try_into().ok()?;
                if be {
                    <$t>::from_be_bytes(a)
                } else {
                    <$t>::from_le_bytes(a)
                }
            }};
        }
        let v = match self.code.as_str() {
            "f8" => float(num!(f64, 8)),
            "f4" => float(num!(f32, 4) as f64),
            "i8" => Value::from(num!(i64, 8)),
            "i4" => Value::from(num!(i32, 4)),
            "i2" => Value::from(num!(i16, 2)),
            "i1" => Value::from(*b.first()? as i8),
            "u8" => Value::from(num!(u64, 8)),
            "u4" => Value::from(num!(u32, 4)),
            "u2" => Value::from(num!(u16, 2)),
            "u1" => Value::from(*b.first()?),
            "b1" => Value::Bool(*b.first()? != 0),
            _ => return None,
        };
        Some(v)
    }

    fn size(&self) -> Option<usize> {
        let digits: String = self.code.chars().filter(|c| c.is_ascii_digit()).collect();
        digits.parse().ok()
    }
}

fn float(f: f64) -> Value {
    Number::from_f64(f)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

struct Machine<'a> {
    data: &'a [u8],
    pos: usize,
    stack: Vec<Obj>,
    memo: HashMap<u32, Obj>,
}

impl<'a> Machine<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self
            .data
            .get(self.pos..self.pos + n)
            .context("pickle: unexpected end")?;
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    fn le<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().unwrap())
    }
    fn line(&mut self) -> Result<&'a str> {
        let rest = &self.data[self.pos..];
        let end = rest
            .iter()
            .position(|&b| b == b'\n')
            .context("pickle: unterminated line")?;
        self.pos += end + 1;
        std::str::from_utf8(&rest[..end]).context("pickle: not utf-8")
    }
    fn pop(&mut self) -> Result<Obj> {
        self.stack.pop().context("pickle: stack underflow")
    }
    fn top(&mut self) -> Result<&mut Obj> {
        self.stack.last_mut().context("pickle: empty stack")
    }
    fn pop_mark(&mut self) -> Result<Vec<Obj>> {
        let m = self
            .stack
            .iter()
            .rposition(|o| matches!(o, Obj::Mark))
            .context("pickle: no mark")?;
        let items = self.stack.split_off(m + 1);
        self.stack.pop();
        Ok(items)
    }
    fn str_obj(&mut self, n: usize) -> Result<Obj> {
        let b = self.take(n)?;
        Ok(Obj::Str(Rc::from(
            std::str::from_utf8(b).context("pickle: not utf-8")?,
        )))
    }
    fn bytes_obj(&mut self, n: usize) -> Result<Obj> {
        Ok(Obj::Bytes(Rc::new(self.take(n)?.to_vec())))
    }
    fn memo_put(&mut self, k: u32) -> Result<()> {
        let v = self
            .stack
            .last()
            .context("pickle: PUT on empty stack")?
            .clone();
        self.memo.insert(k, v);
        Ok(())
    }
    fn memo_get(&mut self, k: u32) -> Result<()> {
        let v = self
            .memo
            .get(&k)
            .with_context(|| format!("pickle: memo {k} missing"))?
            .clone();
        self.stack.push(v);
        Ok(())
    }

    fn call(&mut self, f: Obj, args: Obj) -> Result<Obj> {
        let Obj::Global(m, n) = f else {
            return Ok(Obj::Other);
        };
        let args: Vec<Obj> = match args {
            Obj::Tuple(t) => t.as_ref().clone(),
            _ => Vec::new(),
        };
        let module = m.as_ref().replace("numpy._core", "numpy.core");
        Ok(match (module.as_str(), n.as_ref()) {
            ("collections", "OrderedDict") | ("builtins", "dict") | ("__builtin__", "dict") => {
                Obj::Dict(Rc::new(RefCell::new(Vec::new())))
            }
            ("builtins", "set") | ("builtins", "frozenset") | ("builtins", "list") => {
                let items = match args.first() {
                    Some(Obj::List(l)) => l.borrow().clone(),
                    Some(Obj::Tuple(t)) => t.as_ref().clone(),
                    _ => Vec::new(),
                };
                Obj::List(Rc::new(RefCell::new(items)))
            }
            ("numpy", "dtype") => match args.first() {
                Some(Obj::Str(s)) => Obj::Dtype(Rc::new(RefCell::new(dtype_of(s)))),
                _ => Obj::Other,
            },
            ("numpy.core.multiarray", "_reconstruct") => {
                Obj::Array(Rc::new(RefCell::new(Array::default())))
            }
            ("numpy.core.multiarray", "scalar") => {
                let (Some(Obj::Dtype(dt)), Some(raw)) = (args.first(), args.get(1)) else {
                    return Ok(Obj::Other);
                };
                let raw = match raw {
                    Obj::Bytes(b) => b.as_ref().clone(),
                    Obj::Str(s) => latin1(s),
                    _ => return Ok(Obj::Other),
                };
                match dt.borrow().decode(&raw) {
                    Some(Value::Bool(b)) => Obj::Bool(b),
                    Some(Value::Number(x)) => match x.as_i64() {
                        Some(i) => Obj::Int(i),
                        None => Obj::Float(x.as_f64().unwrap_or(f64::NAN)),
                    },
                    _ => Obj::Other,
                }
            }
            ("_codecs", "encode") => match args.first() {
                Some(Obj::Str(s)) => Obj::Bytes(Rc::new(latin1(s))),
                _ => Obj::Other,
            },
            _ => Obj::Other,
        })
    }

    fn build(&mut self, state: Obj) -> Result<()> {
        match self.top()?.clone() {
            Obj::Array(a) => {
                // (version, shape, dtype, is_fortran, raw bytes or items)
                let Obj::Tuple(t) = state else {
                    bail!("numpy array state is not a tuple");
                };
                let base = if t.len() == 5 { 1 } else { 0 };
                let shape = match t.get(base) {
                    Some(Obj::Tuple(s)) => s
                        .iter()
                        .map(|v| match v {
                            Obj::Int(i) if *i >= 0 => Ok(*i as usize),
                            _ => bail!("numpy shape entry"),
                        })
                        .collect::<Result<Vec<_>>>()?,
                    _ => bail!("numpy array shape"),
                };
                let dtype = match t.get(base + 1) {
                    Some(Obj::Dtype(d)) => d.borrow().clone(),
                    _ => bail!("numpy array dtype"),
                };
                let fortran = matches!(t.get(base + 2), Some(Obj::Bool(true)));
                let mut arr = a.borrow_mut();
                arr.shape = shape;
                arr.fortran = fortran;
                arr.dtype = Some(dtype);
                match t.get(base + 3) {
                    Some(Obj::Bytes(b)) => arr.raw = Some(b.as_ref().clone()),
                    Some(Obj::Str(s)) => arr.raw = Some(latin1(s)),
                    Some(Obj::List(l)) => arr.items = Some(l.borrow().clone()),
                    _ => bail!("numpy array data"),
                }
            }
            Obj::Dtype(d) => {
                // (version, byte order, ...)
                if let Obj::Tuple(t) = state {
                    if let Some(Obj::Str(o)) = t.get(1) {
                        if let Some(c) = o.chars().next() {
                            d.borrow_mut().order = c;
                        }
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn run(&mut self) -> Result<Obj> {
        loop {
            let op = self.u8()?;
            match op {
                0x80 => {
                    self.u8()?;
                }
                0x95 => {
                    self.le::<8>()?;
                }
                b'.' => return self.pop(),
                b'(' => self.stack.push(Obj::Mark),
                b'}' => self.stack.push(Obj::Dict(Rc::default())),
                b']' => self.stack.push(Obj::List(Rc::default())),
                b')' => self.stack.push(Obj::Tuple(Rc::default())),
                0x8f => self.stack.push(Obj::List(Rc::default())),
                b'N' => self.stack.push(Obj::None),
                0x88 => self.stack.push(Obj::Bool(true)),
                0x89 => self.stack.push(Obj::Bool(false)),
                b'J' => {
                    let v = i32::from_le_bytes(self.le()?);
                    self.stack.push(Obj::Int(v as i64));
                }
                b'K' => {
                    let v = self.u8()?;
                    self.stack.push(Obj::Int(v as i64));
                }
                b'M' => {
                    let v = u16::from_le_bytes(self.le()?);
                    self.stack.push(Obj::Int(v as i64));
                }
                0x8a | 0x8b => {
                    let n = if op == 0x8a {
                        self.u8()? as usize
                    } else {
                        u32::from_le_bytes(self.le()?) as usize
                    };
                    let b = self.take(n)?;
                    if n > 8 {
                        bail!("pickle: integer wider than 64 bits");
                    }
                    let mut v: i64 = 0;
                    for (i, &x) in b.iter().enumerate() {
                        v |= (x as i64) << (8 * i);
                    }
                    if n > 0 && n < 8 && b[n - 1] & 0x80 != 0 {
                        v -= 1i64 << (8 * n);
                    }
                    self.stack.push(Obj::Int(v));
                }
                b'I' => {
                    let l = self.line()?.trim();
                    self.stack.push(match l {
                        "00" => Obj::Bool(false),
                        "01" => Obj::Bool(true),
                        _ => Obj::Int(l.parse().context("pickle: INT")?),
                    });
                }
                b'L' => {
                    let l = self.line()?.trim().trim_end_matches('L');
                    self.stack
                        .push(Obj::Int(l.parse().context("pickle: LONG")?));
                }
                b'G' => {
                    let v = f64::from_be_bytes(self.le()?);
                    self.stack.push(Obj::Float(v));
                }
                b'F' => {
                    let l = self.line()?.trim();
                    self.stack
                        .push(Obj::Float(l.parse().context("pickle: FLOAT")?));
                }
                b'X' => {
                    let n = u32::from_le_bytes(self.le()?) as usize;
                    let o = self.str_obj(n)?;
                    self.stack.push(o);
                }
                0x8c => {
                    let n = self.u8()? as usize;
                    let o = self.str_obj(n)?;
                    self.stack.push(o);
                }
                0x8d => {
                    let n = u64::from_le_bytes(self.le()?) as usize;
                    let o = self.str_obj(n)?;
                    self.stack.push(o);
                }
                b'T' => {
                    let n = u32::from_le_bytes(self.le()?) as usize;
                    let o = self.bytes_obj(n)?;
                    self.stack.push(o);
                }
                b'U' => {
                    let n = self.u8()? as usize;
                    let o = self.bytes_obj(n)?;
                    self.stack.push(o);
                }
                b'B' => {
                    let n = u32::from_le_bytes(self.le()?) as usize;
                    let o = self.bytes_obj(n)?;
                    self.stack.push(o);
                }
                b'C' => {
                    let n = self.u8()? as usize;
                    let o = self.bytes_obj(n)?;
                    self.stack.push(o);
                }
                0x8e | 0x96 => {
                    let n = u64::from_le_bytes(self.le()?) as usize;
                    let o = self.bytes_obj(n)?;
                    self.stack.push(o);
                }
                b'V' => {
                    let l = self.line()?.to_string();
                    self.stack.push(Obj::Str(Rc::from(l.as_str())));
                }
                b'S' => {
                    let l = self.line()?.trim();
                    let inner = l
                        .strip_prefix('\'')
                        .and_then(|s| s.strip_suffix('\''))
                        .or_else(|| l.strip_prefix('"').and_then(|s| s.strip_suffix('"')))
                        .unwrap_or(l);
                    self.stack.push(Obj::Str(Rc::from(inner)));
                }
                b'c' => {
                    let m = self.line()?;
                    let n = self.line()?;
                    self.stack.push(Obj::Global(Rc::from(m), Rc::from(n)));
                }
                0x93 => {
                    let n = self.pop()?;
                    let m = self.pop()?;
                    match (m, n) {
                        (Obj::Str(m), Obj::Str(n)) => self.stack.push(Obj::Global(m, n)),
                        _ => bail!("pickle: STACK_GLOBAL needs two strings"),
                    }
                }
                b't' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Obj::Tuple(Rc::new(items)));
                }
                0x85..=0x87 => {
                    let k = (op - 0x84) as usize;
                    let at = self
                        .stack
                        .len()
                        .checked_sub(k)
                        .context("pickle: stack underflow")?;
                    let items = self.stack.split_off(at);
                    self.stack.push(Obj::Tuple(Rc::new(items)));
                }
                b'l' => {
                    let items = self.pop_mark()?;
                    self.stack.push(Obj::List(Rc::new(RefCell::new(items))));
                }
                b'd' => {
                    let items = self.pop_mark()?;
                    let pairs = items
                        .chunks(2)
                        .filter(|c| c.len() == 2)
                        .map(|c| (c[0].clone(), c[1].clone()))
                        .collect();
                    self.stack.push(Obj::Dict(Rc::new(RefCell::new(pairs))));
                }
                b'a' => {
                    let v = self.pop()?;
                    match self.top()? {
                        Obj::List(l) => l.borrow_mut().push(v),
                        _ => bail!("pickle: APPEND to a non-list"),
                    }
                }
                b'e' | 0x90 => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Obj::List(l) => l.borrow_mut().extend(items),
                        _ => bail!("pickle: APPENDS to a non-list"),
                    }
                }
                b's' => {
                    let v = self.pop()?;
                    let k = self.pop()?;
                    match self.top()? {
                        Obj::Dict(d) => d.borrow_mut().push((k, v)),
                        Obj::Other => {}
                        _ => bail!("pickle: SETITEM on a non-dict"),
                    }
                }
                b'u' => {
                    let items = self.pop_mark()?;
                    match self.top()? {
                        Obj::Dict(d) => {
                            let mut d = d.borrow_mut();
                            for c in items.chunks(2) {
                                if c.len() == 2 {
                                    d.push((c[0].clone(), c[1].clone()));
                                }
                            }
                        }
                        Obj::Other => {}
                        _ => bail!("pickle: SETITEMS on a non-dict"),
                    }
                }
                0x91 => {
                    let items = self.pop_mark()?;
                    self.stack.push(Obj::List(Rc::new(RefCell::new(items))));
                }
                b'R' | 0x81 => {
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let v = self.call(f, args)?;
                    self.stack.push(v);
                }
                0x92 => {
                    let _kwargs = self.pop()?;
                    let args = self.pop()?;
                    let f = self.pop()?;
                    let v = self.call(f, args)?;
                    self.stack.push(v);
                }
                b'b' => {
                    let state = self.pop()?;
                    self.build(state)?;
                }
                b'q' => {
                    let k = self.u8()? as u32;
                    self.memo_put(k)?;
                }
                b'r' => {
                    let k = u32::from_le_bytes(self.le()?);
                    self.memo_put(k)?;
                }
                0x94 => {
                    let k = self.memo.len() as u32;
                    self.memo_put(k)?;
                }
                b'p' => {
                    let k: u32 = self.line()?.trim().parse().context("pickle: PUT")?;
                    self.memo_put(k)?;
                }
                b'h' => {
                    let k = self.u8()? as u32;
                    self.memo_get(k)?;
                }
                b'j' => {
                    let k = u32::from_le_bytes(self.le()?);
                    self.memo_get(k)?;
                }
                b'g' => {
                    let k: u32 = self.line()?.trim().parse().context("pickle: GET")?;
                    self.memo_get(k)?;
                }
                b'2' => {
                    let v = self.top()?.clone();
                    self.stack.push(v);
                }
                b'0' => {
                    self.pop()?;
                }
                b'1' => {
                    self.pop_mark()?;
                }
                other => bail!(
                    "pickle: unsupported opcode 0x{other:02x} at offset {}",
                    self.pos - 1
                ),
            }
        }
    }
}

/// What `str.encode('latin1')` gives: each character's code point as a byte.
fn latin1(s: &str) -> Vec<u8> {
    s.chars().map(|c| c as u32 as u8).collect()
}

/// How Python prints a dict key.
fn key_text(k: &Obj) -> String {
    match k {
        Obj::Str(s) => s.to_string(),
        Obj::Int(i) => i.to_string(),
        Obj::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Obj::Float(f) => format!("{f:?}"),
        Obj::None => "None".to_string(),
        Obj::Tuple(t) => {
            let parts: Vec<String> = t.iter().map(key_text).collect();
            if parts.len() == 1 {
                format!("({},)", parts[0])
            } else {
                format!("({})", parts.join(", "))
            }
        }
        _ => "?".to_string(),
    }
}

fn to_json(o: &Obj) -> Value {
    match o {
        Obj::None | Obj::Other | Obj::Mark | Obj::Global(..) | Obj::Bytes(_) => Value::Null,
        Obj::Bool(b) => Value::Bool(*b),
        Obj::Int(i) => Value::from(*i),
        Obj::Float(f) => float(*f),
        Obj::Str(s) => Value::String(s.to_string()),
        Obj::Tuple(t) => Value::Array(t.iter().map(to_json).collect()),
        Obj::List(l) => Value::Array(l.borrow().iter().map(to_json).collect()),
        Obj::Dict(d) => {
            let mut m = Map::new();
            for (k, v) in d.borrow().iter() {
                m.insert(key_text(k), to_json(v));
            }
            Value::Object(m)
        }
        Obj::Dtype(d) => {
            let d = d.borrow();
            Value::String(format!("{}{}", d.order, d.code))
        }
        Obj::Array(a) => array_json(&a.borrow()),
    }
}

/// A numpy array as nested JSON arrays, in C order.
fn array_json(a: &Array) -> Value {
    let n: usize = a.shape.iter().product();
    let flat: Vec<Value> = if let Some(items) = &a.items {
        items.iter().map(to_json).collect()
    } else {
        let (Some(dt), Some(raw)) = (&a.dtype, &a.raw) else {
            return Value::Null;
        };
        let Some(size) = dt.size().filter(|&s| s > 0) else {
            return Value::Null;
        };
        raw.chunks(size)
            .take(n)
            .map(|c| dt.decode(c).unwrap_or(Value::Null))
            .collect()
    };
    if flat.len() != n {
        return Value::Null;
    }
    // Index of C-order position `i` in the stored order.
    let stored = |i: usize| -> usize {
        if !a.fortran || a.shape.len() < 2 {
            return i;
        }
        let mut rem = i;
        let mut idx = vec![0usize; a.shape.len()];
        for d in (0..a.shape.len()).rev() {
            idx[d] = rem % a.shape[d];
            rem /= a.shape[d];
        }
        let mut f = 0;
        let mut stride = 1;
        for (k, n) in idx.iter().zip(&a.shape) {
            f += k * stride;
            stride *= n;
        }
        f
    };
    fn nest(shape: &[usize], at: &mut usize, get: &dyn Fn(usize) -> Value) -> Value {
        if shape.is_empty() {
            let v = get(*at);
            *at += 1;
            return v;
        }
        Value::Array((0..shape[0]).map(|_| nest(&shape[1..], at, get)).collect())
    }
    let mut at = 0;
    nest(&a.shape, &mut at, &|i| flat[stored(i)].clone())
}

/// A pickle's object as JSON.
pub fn load(bytes: &[u8]) -> Result<Value> {
    let mut m = Machine {
        data: bytes,
        pos: 0,
        stack: Vec::new(),
        memo: HashMap::new(),
    };
    let root = m.run()?;
    Ok(to_json(&root))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `pickle.dumps(obj, protocol=2)` of
    /// `{'plans_per_stage': {0: {'patch_size': np.array([96, 160, 160]),
    /// 'current_spacing': np.array([2.5, 0.8, 0.8])}}, 'num_classes': 2,
    /// 'mean': np.float64(-3.5), 'flag': np.bool_(True),
    /// 'pool': [[1, 2, 2], [2, 2, 2]], (1, 2): 'tuple key',
    /// 'F': np.asfortranarray(np.arange(6, dtype=np.int32).reshape(2, 3))}`
    /// at protocol 3 by CPython 3.13 with numpy 2.5, its `numpy._core`
    /// module path renamed to the `numpy.core` the plans of 2020 carry.
    #[test]
    fn numpy_values_come_out_as_numbers() {
        let pkl = include_bytes!("../../../../tests/data/nnunet-v1-plans.pkl");
        let v = load(pkl).unwrap();
        assert_eq!(v["num_classes"], 2);
        assert_eq!(v["mean"], -3.5);
        assert_eq!(v["flag"], true);
        assert_eq!(
            v["plans_per_stage"]["0"]["patch_size"],
            serde_json::json!([96, 160, 160])
        );
        assert_eq!(
            v["plans_per_stage"]["0"]["current_spacing"],
            serde_json::json!([2.5, 0.8, 0.8])
        );
        assert_eq!(v["pool"], serde_json::json!([[1, 2, 2], [2, 2, 2]]));
        assert_eq!(v["(1, 2)"], "tuple key");
        assert_eq!(v["F"], serde_json::json!([[0, 1, 2], [3, 4, 5]]));
    }

    #[test]
    fn plain_protocols_parse() {
        // pickle.dumps({'a': [1, 2.5, 'x', None, True], 'b': (3, -4)}, protocol=0)
        let p0 = b"(dp0\nVa\np1\n(lp2\nI1\naF2.5\naVx\np3\naNaI01\nasVb\np4\n(I3\nI-4\ntp5\ns.";
        let v = load(p0).unwrap();
        assert_eq!(
            v,
            serde_json::json!({"a": [1, 2.5, "x", null, true], "b": [3, -4]})
        );
        // The same at protocol 4 (FRAME, SHORT_BINUNICODE, MEMOIZE).
        let p4: &[u8] = &[
            0x80, 0x04, 0x95, 0x17, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x7d, 0x94, 0x8c,
            0x01, 0x61, 0x94, 0x5d, 0x94, 0x28, 0x4b, 0x01, 0x47, 0x40, 0x04, 0x00, 0x00, 0x00,
            0x00, 0x00, 0x00, 0x65, 0x73, 0x2e,
        ];
        assert_eq!(load(p4).unwrap(), serde_json::json!({"a": [1, 2.5]}));
    }
}

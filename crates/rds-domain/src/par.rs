//! Parallel sums that come out the same on every run.
//!
//! The codebase's rule is that a sum which decides something - a threshold,
//! a normalisation, the next step of an optimizer - must reproduce itself
//! exactly (CLAUDE.md, "Determinism"). rayon's own `sum` and `reduce` do not
//! promise that: they combine partial results in the order the work
//! happened to be split and stolen, which varies from run to run and with
//! the number of threads, and floating-point addition is not associative, so
//! the last bits of the result wander.
//!
//! [`ordered_fold`] keeps the parallelism and fixes the order. The input is
//! cut into pieces of a fixed length ([`CHUNK`], not one piece per thread,
//! so the machine does not matter either), each piece is folded on its own
//! in parallel, and the per-piece results are combined one after another in
//! input order. For a few hundred thousand items the sequential combine is a
//! hundred additions; the cost over an unordered reduce is nothing.

use rayon::prelude::*;

/// Items per piece. Small enough that a few hundred thousand samples still
/// spread over every core, large enough that the per-piece results are few.
pub const CHUNK: usize = 4096;

/// Fold `items` in parallel, in fixed pieces of [`CHUNK`], and combine the
/// per-piece results in input order.
///
/// `piece` gets a piece and the index of its first item in `items` (so a
/// caller can walk a second slice alongside); `combine` merges two results,
/// the earlier one first. The result depends only on `items`, never on the
/// thread count or on scheduling.
pub fn ordered_fold<T, A>(
    items: &[T],
    piece: impl Fn(&[T], usize) -> A + Sync + Send,
    combine: impl FnMut(A, A) -> A,
    identity: A,
) -> A
where
    T: Sync,
    A: Send,
{
    ordered_fold_by(items, CHUNK, piece, combine, identity)
}

/// [`ordered_fold`] with a piece length of the caller's choosing, for inputs
/// of a few thousand costly items (an optimizer's samples) that one
/// [`CHUNK`] would leave on a single core. The length must be a constant or
/// derived from the input alone, never from the thread count.
pub fn ordered_fold_by<T, A>(
    items: &[T],
    chunk: usize,
    piece: impl Fn(&[T], usize) -> A + Sync + Send,
    combine: impl FnMut(A, A) -> A,
    identity: A,
) -> A
where
    T: Sync,
    A: Send,
{
    let chunk = chunk.max(1);
    let parts: Vec<A> = items
        .par_chunks(chunk)
        .enumerate()
        .map(|(i, c)| piece(c, i * chunk))
        .collect();
    parts.into_iter().fold(identity, combine)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Values whose sum depends on the order they are added in.
    fn awkward(n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let x = (i as f64 * 0.618_033_988_75).fract();
                if i % 3 == 0 {
                    x * 1e12
                } else {
                    x * 1e-3
                }
            })
            .collect()
    }

    fn sum_of(v: &[f64]) -> f64 {
        ordered_fold(v, |c, _| c.iter().sum::<f64>(), |a, b| a + b, 0.0)
    }

    #[test]
    fn the_sum_does_not_depend_on_the_thread_count() {
        let v = awkward(100_003);
        let one = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap()
            .install(|| sum_of(&v));
        let three = rayon::ThreadPoolBuilder::new()
            .num_threads(3)
            .build()
            .unwrap()
            .install(|| sum_of(&v));
        assert_eq!(one.to_bits(), three.to_bits());
        // And it is the sequential sum of the per-piece sums.
        let by_hand = v
            .chunks(CHUNK)
            .map(|c| c.iter().sum::<f64>())
            .fold(0.0, |a, b| a + b);
        assert_eq!(one.to_bits(), by_hand.to_bits());
    }

    #[test]
    fn the_offset_lets_a_second_slice_be_walked_alongside() {
        let a: Vec<usize> = (0..10_000).collect();
        let b: Vec<usize> = (0..10_000).map(|i| 2 * i).collect();
        let bad = ordered_fold(
            &a,
            |c, off| {
                c.iter()
                    .enumerate()
                    .filter(|(i, &x)| b[off + i] != 2 * x)
                    .count()
            },
            |x, y| x + y,
            0,
        );
        assert_eq!(bad, 0);
    }

    #[test]
    fn an_empty_input_gives_the_identity() {
        let v: Vec<f64> = Vec::new();
        assert_eq!(sum_of(&v), 0.0);
    }
}

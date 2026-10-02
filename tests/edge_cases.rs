//! Edge-case regression tests: invalid input, extreme magnitudes, and the worst
//! case for the floating-point rounding of the sampling pointers.
//!
//! Most of these are DETERMINISTIC: they either use an RNG whose output is
//! fixed, or choose weights whose expected counts are whole numbers (in which
//! case SUS must return exactly that many copies no matter what the random
//! offset is). Deterministic tests can never flake, so any failure is a real bug.

use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng, TryRng};
use std::convert::Infallible;

use stochastic_universal_sampling::choose_multiple_weighted as sus;

/// A fake "random" number generator that always returns the largest possible
/// value. Two consequences for the code under test:
///
///  * `random::<f64>()` yields 1 - 2^-53, the largest number below 1. SUS uses
///    this as the pointer offset (as a fraction of the pointer spacing), so this
///    is the most extreme offset that can ever occur.
///  * `shuffle` always swaps an element with itself, i.e. it is the identity
///    permutation, which keeps the layout of the weights predictable.
///
/// In real use the offset gets this close to 1 with probability around 1e-13, so
/// no amount of ordinary random testing would ever find a bug that appears here.
struct MaxRng;

impl TryRng for MaxRng {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(u32::MAX)
    }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(u64::MAX)
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> {
        dst.fill(0xFF);
        Ok(())
    }
}

/// Check MaxRng works as advertised.
#[test]
fn test_fixture_max_rng() {
    for _ in 0..100 {
        assert_eq!(MaxRng.random::<f64>(), 1.0_f64.next_down());
    }
}

/// How many times each index appears in `picked`.
fn tally(picked: &[usize], n: usize) -> Vec<usize> {
    let mut counts = vec![0; n];
    for &i in picked {
        counts[i] += 1;
    }
    counts
}

/// Floating-point rounding can push the last sampling pointer onto, or a hair
/// past, the end of the cumulative weights. The code must then fall back on the
/// last item that has a non-zero weight, never on a trailing zero-weight item.
#[test]
fn worst_case_offset_never_selects_a_zero_weight_item() {
    for positives in 1..=6 {
        for k in 1..=64 {
            for scale in [1.0, 0.1, 0.3, 1.0 / 3.0, 7.0, 1e-3] {
                // Zero-weight item at the END
                let mut w = vec![scale; positives];
                w.push(0.0);
                let picked = sus(&mut MaxRng, k, &w);
                assert_eq!(tally(&picked, w.len())[positives], 0, "w = {w:?}, k = {k}");

                // Zero-weight item at the START
                let mut w = vec![0.0];
                w.extend(std::iter::repeat_n(scale, positives));
                let picked = sus(&mut MaxRng, k, &w);
                assert_eq!(tally(&picked, w.len())[0], 0, "w = {w:?}, k = {k}");
            }
        }
    }
}

/// Weights whose SUM exceeds f64::MAX must still work, because every individual
/// weight is a perfectly valid finite number. The expected counts are whole
/// numbers (4, 2, 1, 1), so the output multiset is fully determined.
#[test]
fn sum_of_weights_may_exceed_f64_max() {
    let m = f64::MAX;
    let w = [m, m / 2.0, m / 4.0, m / 4.0];
    let mut rng = StdRng::seed_from_u64(1);
    for _ in 0..200 {
        assert_eq!(tally(&sus(&mut rng, 8, &w), 4), [4, 2, 1, 1]);
    }
}

/// Subnormal numbers are the tiny values below f64::MIN_POSITIVE (about 2.2e-308)
/// that trade away precision to get closer to zero. The smallest one is 5e-324.
/// A naive `total / amount` rounds to exactly zero here, which would stack every
/// pointer on one item. These weights are exact multiples of 5e-324, so the
/// expected counts (4, 2, 1, 1 and 2, 2) are whole numbers.
#[test]
fn subnormal_weights_are_sampled_correctly() {
    let tiny = 5e-324;
    let mut rng = StdRng::seed_from_u64(2);
    for _ in 0..200 {
        let w = [4.0 * tiny, 2.0 * tiny, tiny, tiny];
        assert_eq!(tally(&sus(&mut rng, 8, &w), 4), [4, 2, 1, 1]);
        assert_eq!(tally(&sus(&mut rng, 4, &[tiny, tiny]), 2), [2, 2]);
    }
}

/// Bad weights are rejected even when no samples were requested, so a bug in the
/// caller is reported consistently rather than only when `amount > 0`.
#[test]
#[should_panic(expected = "invalid weight")]
fn negative_weight_panics_even_if_amount_is_zero() {
    sus(&mut StdRng::seed_from_u64(3), 0, &[1.0, -1.0]);
}

#[test]
#[should_panic(expected = "invalid weight")]
fn nan_weight_panics() {
    sus(&mut StdRng::seed_from_u64(4), 3, &[1.0, f64::NAN]);
}

#[test]
#[should_panic(expected = "invalid weight")]
fn infinite_weight_panics() {
    sus(&mut StdRng::seed_from_u64(5), 3, &[1.0, f64::INFINITY]);
}

//! Tests that the *input order* of the weights does not bias which combinations
//! of items can be selected.
//!
//! Background
//! ----------
//! SUS places `k` pointers a fixed distance `total / k` apart and slides them
//! across the cumulative weights with ONE random offset. The pointers are rigidly
//! coupled, so nearby items in the input array are selected in a correlated way.
//! Shuffling only the *output* hides the order of the results but not *which
//! sets* of items are reachable.
//!
//! Crafted input
//! -------------
//! Take `n` equal, non-zero weights and draw `k` items, with `k` dividing `n`.
//! The pointer spacing is then exactly `n / k` items, so the selected indices are
//! always `{i, i + n/k, i + 2n/k, ...}` for a single random `i`. Only `n/k`
//! distinct outcomes are reachable, out of C(n, k) possible k-subsets:
//!
//!   n = 4, k = 2:  only {0,2} and {1,3}            (of 6 pairs)
//!   n = 6, k = 3:  only {0,2,4} and {1,3,5}        (of 20 triples)
//!
//! If the input were randomly permuted before sampling, every k-subset would
//! become reachable with equal probability.
//!
//! All tests use a seeded RNG, so they are deterministic. The failure probability
//! for a correct (input-shuffling) implementation is (1/3)^500 and ~0 respectively.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::collections::BTreeSet;

use stochastic_universal_sampling::choose_multiple_weighted as sus;

type Sampler = fn(&mut StdRng, usize, &[f64]) -> Vec<usize>;

const SEED: u64 = 0x5EED_CAFE;

/// Run `sampler` many times and collect the distinct *sets* of selected indices
/// (each outcome sorted, so output order is irrelevant).
fn distinct_outcomes(
    sampler: Sampler,
    trials: usize,
    amount: usize,
    weights: &[f64],
) -> BTreeSet<Vec<usize>> {
    let mut rng = StdRng::seed_from_u64(SEED);
    (0..trials)
        .map(|_| {
            let mut picked = sampler(&mut rng, amount, weights);
            picked.sort_unstable();
            picked
        })
        .collect()
}

/// Reference implementation of the desired behaviour: permute the inputs, run
/// SUS, then map the chosen indices back to the caller's original indices.
fn sus_with_shuffled_input(rng: &mut StdRng, amount: usize, weights: &[f64]) -> Vec<usize> {
    let mut perm: Vec<usize> = (0..weights.len()).collect();
    perm.shuffle(rng);
    let permuted: Vec<f64> = perm.iter().map(|&i| weights[i]).collect();
    sus(rng, amount, &permuted)
        .into_iter()
        .map(|j| perm[j])
        .collect()
}

// ---------------------------------------------------------------------------
// Checks (each takes a sampler, so they can be run against any implementation)
// ---------------------------------------------------------------------------

/// 2 draws from 4 equal weights: adjacent / wrap-around pairs must be reachable.
/// Without input shuffling the index gap is always exactly 2.
fn check_pairs_not_locked_to_gap_two(sampler: Sampler) {
    let outcomes = distinct_outcomes(sampler, 500, 2, &[1.0; 4]);
    let gaps: BTreeSet<usize> = outcomes.iter().map(|o| o[1] - o[0]).collect();
    assert!(
        gaps.iter().any(|&g| g != 2),
        "every sampled pair had index gap 2 (outcomes seen: {outcomes:?}); \
         selection is locked to the input order"
    );
}

/// 3 draws from 6 equal weights: far more than 2 of the 20 triples must appear.
fn check_many_distinct_triples(sampler: Sampler) {
    let outcomes = distinct_outcomes(sampler, 2000, 3, &[1.0; 6]);
    assert!(
        outcomes.len() > 2,
        "only {} distinct triples reachable out of 20 (outcomes seen: {outcomes:?}); \
         selection is locked to the input order",
        outcomes.len()
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// EXPECTED TO FAIL on the current crate: demonstrates input order matters.
#[test]
fn input_order_does_not_constrain_pairs() {
    check_pairs_not_locked_to_gap_two(sus::<StdRng>);
}

/// EXPECTED TO FAIL on the current crate: demonstrates input order matters.
#[test]
fn input_order_does_not_constrain_triples() {
    check_many_distinct_triples(sus::<StdRng>);
}

/// Validates the checks themselves: a wrapper that shuffles its input must pass.
#[test]
fn checks_pass_for_shuffled_input_reference() {
    check_pairs_not_locked_to_gap_two(sus_with_shuffled_input);
    check_many_distinct_triples(sus_with_shuffled_input);
}

/// Documents the exact structure the current implementation exhibits, so the
/// root cause is visible in a failing diff if this ever changes. This one
/// PASSES on the current crate and should be deleted once input shuffling is added.
#[test]
fn current_behaviour_only_reaches_strided_subsets() {
    let pairs = distinct_outcomes(sus::<StdRng>, 500, 2, &[1.0; 4]);
    let expected_pairs: BTreeSet<Vec<usize>> = [vec![0, 2], vec![1, 3]].into_iter().collect();
    assert_eq!(pairs, expected_pairs);

    let triples = distinct_outcomes(sus::<StdRng>, 2000, 3, &[1.0; 6]);
    let expected_triples: BTreeSet<Vec<usize>> =
        [vec![0, 2, 4], vec![1, 3, 5]].into_iter().collect();
    assert_eq!(triples, expected_triples);
}

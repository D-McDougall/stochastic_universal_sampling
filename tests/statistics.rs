//! Statistical tests for stochastic universal sampling (SUS).
//!
//! THE DEFINING PROPERTY
//! ---------------------
//! Let `e_i = k * w_i / sum(w)` be item i's expected number of copies in `k`
//! draws. SUS lays `k` pointers `sum(w)/k` apart and slides them over the
//! cumulative weights with ONE uniform random phase. In units of the pointer
//! spacing, item i owns an interval of length `e_i`, so the number of pointers
//! landing in it is *exactly*
//!
//!     count_i = floor(e_i) + Bernoulli(frac(e_i))
//!
//! Four layers of checks, from strongest to weakest claim:
//!
//!   1. Bound         count_i is ALWAYS floor(e_i) or ceil(e_i), in every trial.
//!                    (This is what separates SUS from roulette-wheel sampling,
//!                    where count_i ~ Binomial(k, p_i) strays arbitrarily far.)
//!   2. Marginal law  P(count_i = ceil(e_i)) = frac(e_i). Exact binomial test per
//!                    item. Implies E[count_i] = e_i *and* Var = f(1-f).
//!   3. Unbiasedness  E[count_i] = e_i, one-sample t-test. Weaker than 2, kept
//!                    because it is the claim people care about, and because it
//!                    shows why 2 is needed: roulette passes it.
//!   4. Exchangeable  Output position j holds item i with probability
//!                    w_i / sum(w). Trials are i.i.d., so this is a textbook
//!                    Pearson chi-squared (n-1 d.o.f.) per position. It tests the
//!                    crate's final shuffle.
//!
//! MULTIPLE TESTING
//! ----------------
//! Each check runs many tests (one per item). Counts are negatively dependent
//! (they sum to k) and are all functions of one random phase, so we use
//! Bonferroni, which is valid under arbitrary dependence: reject iff
//! min p < ALPHA / m. With ALPHA = 1e-6 a correct implementation fails for about
//! one seed in a million; the tests are seeded anyway, hence deterministic.
//!
//! `checks_have_power_against_known_bad_samplers` proves the checks can actually
//! reject, by running them against deliberately broken samplers.

use rand::prelude::*;
use rand::rngs::StdRng;
use statrs::distribution::{Binomial, ChiSquared, ContinuousCDF, DiscreteCDF, StudentsT};
use std::sync::OnceLock;

use stochastic_universal_sampling::choose_multiple_weighted as sus;

type Sampler = fn(&mut StdRng, usize, &[f64]) -> Vec<usize>;

fn seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| rand::random())
}

/// Family-wise significance level, split across tests by Bonferroni.
const ALPHA: f64 = 1e-6;
/// Floating-point slack when deciding whether an expectation is an integer.
const TOL: f64 = 1e-9;
const TRIALS: usize = 50_000;
/// The t-test's normal approximation needs this many "events" of variance.
const MIN_EVENTS: f64 = 10.0;

// ===========================================================================
// Statistics helpers (thin wrappers over statrs)
// ===========================================================================

/// Exact two-sided binomial test: twice the smaller tail, capped at 1.
/// Valid for any n and p, including very rare events.
fn binomial_p(x: usize, n: usize, p: f64) -> f64 {
    let b = Binomial::new(p, n as u64).expect("probability in (0, 1)");
    let lower = b.cdf(x as u64); // P(X <= x)
    let upper = if x == 0 { 1.0 } else { b.sf(x as u64 - 1) }; // P(X >= x)
    (2.0 * lower.min(upper)).min(1.0)
}

fn t_p(t: f64, df: f64) -> f64 {
    2.0 * StudentsT::new(0.0, 1.0, df).expect("df > 0").sf(t.abs())
}

fn chi2_p(stat: f64, df: usize) -> f64 {
    ChiSquared::new(df as f64).expect("df > 0").sf(stat)
}

fn pearson(observed: &[usize], expected: &[f64]) -> f64 {
    observed
        .iter()
        .zip(expected)
        .map(|(&o, &e)| (o as f64 - e).powi(2) / e)
        .sum()
}

/// One p-value together with a description of what it tested.
struct Evidence {
    p: f64,
    label: String,
}

/// Bonferroni decision for a family of tests; prints the smallest p-value.
fn bonferroni(what: &str, evidence: Vec<Evidence>) -> Result<(), String> {
    let m = evidence.len();
    assert!(m > 0, "{what}: no testable items");
    let threshold = ALPHA / m as f64;
    let worst = evidence
        .iter()
        .min_by(|a, b| a.p.total_cmp(&b.p))
        .expect("non-empty");
    eprintln!(
        "{what}: {m} tests, min p = {:.3e} (threshold {threshold:.3e}) at {}",
        worst.p, worst.label
    );
    if worst.p < threshold {
        Err(format!(
            "{what}: REJECTED at {} with p = {:.3e} < {threshold:.3e} ({m} tests)",
            worst.label, worst.p
        ))
    } else {
        Ok(())
    }
}

// ===========================================================================
// Fixtures and tallying
// ===========================================================================

struct Fixture {
    name: String,
    weights: Vec<f64>,
    k: usize,
}

fn fx(name: &str, weights: Vec<f64>, k: usize) -> Fixture {
    Fixture {
        name: name.to_string(),
        weights,
        k,
    }
}

fn fixtures() -> Vec<Fixture> {
    vec![
        fx("ascending", vec![1.0, 2.0, 3.0, 4.0], 7),
        fx(
            "one dominant item",
            std::iter::once(100.0)
                .chain(std::iter::repeat_n(1.0, 10))
                .collect(),
            5,
        ),
        // Every expectation is an integer, so the output multiset is deterministic.
        fx(
            "integer expectations with zeros",
            vec![0.0, 2.0, 0.0, 4.0, 2.0],
            4,
        ),
        fx("k much larger than n", vec![0.1, 0.25, 0.65], 999),
        fx("single item", vec![3.5], 5),
        fx("k = 1 (plain proportional)", vec![1.0, 2.0, 3.0, 4.0], 1),
        fx(
            "three decades of weight",
            vec![1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
            20,
        ),
        fx(
            "very small items",
            (0..30).map(|i| 1e-9 * ((i * 37) % 101) as f64).collect(),
            7,
        ),
        // Rare events at both ends (expected extras ~5 in 50_000 trials): too
        // sparse for a chi-squared approximation, fine for the exact test.
        fx("rare events", vec![1000.0, 1000.0, 0.05], 4),
        fx(
            "many items",
            (0..200).map(|i| ((i * 37) % 101 + 1) as f64).collect(),
            57,
        ),
    ]
}

/// One item of a fixture: its expectation and the two counts SUS may produce.
struct Item {
    /// Index
    i: usize,
    /// Expected count
    e: f64,
    /// Lower bound of expected count
    lo: u32,
    /// Upper bound of expected count
    hi: u32,
}

impl Item {
    fn new(i: usize, e: f64) -> Self {
        Item {
            i,
            e,
            lo: (e + TOL).floor() as u32,
            hi: (e - TOL).ceil() as u32,
        }
    }
    /// False when `e` is an integer: the count is then deterministic.
    fn is_random(&self) -> bool {
        self.hi > self.lo
    }
    fn frac(&self) -> f64 {
        self.e - self.lo as f64
    }
}

/// Count histograms from many draws of one fixture: `hist[item][count]`.
struct Run {
    name: String,
    trials: usize,
    e: Vec<f64>,
    hist: Vec<Vec<usize>>,
}

impl Run {
    fn items(&self) -> impl Iterator<Item = Item> + '_ {
        self.e.iter().enumerate().map(|(i, &e)| Item::new(i, e))
    }

    fn n_at(&self, item: &Item, count: u32) -> usize {
        self.hist[item.i].get(count as usize).copied().unwrap_or(0)
    }

    /// Trials in which the item's count was neither floor(e) nor ceil(e).
    fn n_outside(&self, item: &Item) -> usize {
        let hi = if item.is_random() {
            self.n_at(item, item.hi)
        } else {
            0
        };
        self.trials - self.n_at(item, item.lo) - hi
    }

    /// Sample mean and unbiased sample variance of an item's count.
    fn mean_var(&self, item: &Item) -> (f64, f64) {
        let t = self.trials as f64;
        let (sum, sum_sq) = self.hist[item.i]
            .iter()
            .enumerate()
            .fold((0.0, 0.0), |(s, s2), (c, &n)| {
                (s + c as f64 * n as f64, s2 + (c * c) as f64 * n as f64)
            });
        let mean = sum / t;
        (mean, (sum_sq - t * mean * mean) / (t - 1.0))
    }
}

/// Calculates what the histgram should look like
fn expected_counts(weights: &[f64], k: usize) -> Vec<f64> {
    let total: f64 = weights.iter().sum();
    weights.iter().map(|&w| k as f64 * w / total).collect()
}

fn run_fixture(sampler: Sampler, fx: &Fixture, trials: usize, seed: u64) -> Run {
    let n = fx.weights.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut hist = vec![vec![0usize; fx.k + 1]; n];
    let mut counts = vec![0usize; n];
    for _ in 0..trials {
        let out = sampler(&mut rng, fx.k, &fx.weights);
        assert_eq!(out.len(), fx.k, "[{}] wrong number of samples", fx.name);
        counts.fill(0);
        for &i in &out {
            assert!(i < n, "[{}] index {i} out of range (n = {n})", fx.name);
            counts[i] += 1;
        }
        for (h, &c) in hist.iter_mut().zip(&counts) {
            h[c] += 1;
        }
    }
    Run {
        name: fx.name.clone(),
        trials,
        e: expected_counts(&fx.weights, fx.k),
        hist,
    }
}

fn run_all(sampler: Sampler, trials: usize) -> Vec<Run> {
    fixtures()
        .iter()
        .enumerate()
        .map(|(i, f)| run_fixture(sampler, f, trials, seed() + i as u64))
        .collect()
}

/// The real crate's runs, simulated once and shared by all tests that need them.
fn crate_runs() -> &'static [Run] {
    static RUNS: OnceLock<Vec<Run>> = OnceLock::new();
    RUNS.get_or_init(|| run_all(sus::<StdRng>, TRIALS))
}

// ===========================================================================
// Checks (each returns Err(description) so broken samplers can be asserted on)
// ===========================================================================

/// Layer 1: every count is floor(e) or ceil(e) in every single trial.
fn check_bounds(runs: &[Run]) -> Result<(), String> {
    for r in runs {
        for it in r.items() {
            let bad = r.n_outside(&it);
            if bad > 0 {
                return Err(format!(
                    "[{}] item {}: e = {:.6} so count must be in {{{}, {}}}, \
                     but {bad} of {} trials fell outside",
                    r.name, it.i, it.e, it.lo, it.hi, r.trials
                ));
            }
        }
    }
    Ok(())
}

/// Layer 2: P(count = ceil(e)) = frac(e), exact binomial test per item.
fn check_count_distribution(runs: &[Run]) -> Result<(), String> {
    check_bounds(runs)?;
    let mut evidence = Vec::new();
    for r in runs {
        for it in r.items().filter(Item::is_random) {
            let extras = r.n_at(&it, it.hi);
            evidence.push(Evidence {
                p: binomial_p(extras, r.trials, it.frac()),
                label: format!("[{}] item {} (f = {:.4})", r.name, it.i, it.frac()),
            });
        }
    }
    bonferroni("count distribution", evidence)
}

/// Layer 3: E[count] = e, one-sample t-test with the sample standard error.
fn check_unbiased(runs: &[Run]) -> Result<(), String> {
    let mut evidence = Vec::new();
    for r in runs {
        let t = r.trials as f64;
        for it in r.items() {
            let (mean, var) = r.mean_var(&it);
            let label = format!("[{}] item {} (e = {:.4})", r.name, it.i, it.e);
            if var <= 1e-12 {
                // No variation observed: a constant but wrong count is
                // infinitely significant.
                if (mean - it.e).abs() > 1e-6 {
                    evidence.push(Evidence { p: 0.0, label });
                }
            } else if var * t >= MIN_EVENTS {
                let stat = (mean - it.e) / (var / t).sqrt();
                evidence.push(Evidence {
                    p: t_p(stat, t - 1.0),
                    label,
                });
            }
        }
    }
    bonferroni("unbiasedness", evidence)
}

/// Layer 4: output position j holds item i with probability w_i / sum(w).
fn check_position_distribution(sampler: Sampler) -> Result<(), String> {
    const T: usize = 60_000;
    let weights = [1.0, 2.0, 3.0, 4.0, 10.0];
    let k = 8;
    let total: f64 = weights.iter().sum();
    let expected: Vec<f64> = weights.iter().map(|w| T as f64 * w / total).collect();

    let mut rng = StdRng::seed_from_u64(seed() ^ 0x1234);
    let mut observed = vec![vec![0usize; weights.len()]; k];
    for _ in 0..T {
        for (pos, &item) in sampler(&mut rng, k, &weights).iter().enumerate() {
            observed[pos][item] += 1;
        }
    }
    let evidence = observed
        .iter()
        .enumerate()
        .map(|(pos, o)| Evidence {
            p: chi2_p(pearson(o, &expected), weights.len() - 1),
            label: format!("output position {pos}"),
        })
        .collect();
    bonferroni("position distribution", evidence)
}

// ===========================================================================
// Tests on the real crate
// ===========================================================================

/// Layer 1 on hand-picked fixtures.
#[test]
fn counts_are_always_floor_or_ceil_of_expectation() {
    check_bounds(crate_runs()).unwrap();
}

/// Layer 1 on hundreds of random weight vectors (zeros, wide dynamic range).
#[test]
fn random_weight_vectors_obey_floor_ceil_bounds() {
    let mut rng = StdRng::seed_from_u64(seed() ^ 0xABCD);
    for case in 0..400 {
        let n = 1 + (rng.random::<f64>() * 40.0) as usize;
        let k = 1 + (rng.random::<f64>() * 100.0) as usize;
        let mut w: Vec<f64> = (0..n)
            .map(|_| {
                if rng.random::<f64>() < 0.2 {
                    0.0
                } else {
                    10f64.powf(-3.0 + 6.0 * rng.random::<f64>())
                }
            })
            .collect();
        if w.iter().all(|&x| x == 0.0) {
            w[0] = 1.0;
        }
        let f = fx(&format!("random case {case} (n={n}, k={k})"), w, k);
        check_bounds(&[run_fixture(sus::<StdRng>, &f, 200, seed() + case)]).unwrap();
    }
}

/// Layer 2: each item's count is floor(e) + Bernoulli(frac(e)).
#[test]
fn count_distribution_is_floor_plus_bernoulli_of_fraction() {
    check_count_distribution(crate_runs()).unwrap();
}

/// Layer 3: E[count_i] = k * p_i.
#[test]
fn expected_count_is_proportional_to_weight() {
    check_unbiased(crate_runs()).unwrap();
}

/// Layer 4: the final shuffle makes every output position equally likely to
/// hold any copy, so position j shows item i with probability p_i.
#[test]
fn output_positions_are_proportional_and_exchangeable() {
    check_position_distribution(sus::<StdRng>).unwrap();
}

/// Selection probabilities must not depend on the absolute scale of the weights.
#[test]
fn bounds_hold_at_every_weight_scale() {
    let bases = [
        fx("ascending", vec![1.0, 2.0, 3.0, 4.0], 7),
        fx("mixed", vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0], 10),
    ];
    let mut failures = Vec::new();
    for scale in [1e-30, 1e-18, 1e-12, 1e-6, 1.0, 1e6, 1e12, 1e100] {
        for b in &bases {
            let scaled = fx(
                &format!("{} x {scale:e}", b.name),
                b.weights.iter().map(|w| w * scale).collect(),
                b.k,
            );
            if let Err(e) = check_bounds(&[run_fixture(sus::<StdRng>, &scaled, 2_000, seed())]) {
                failures.push(e);
            }
        }
    }
    assert!(
        failures.is_empty(),
        "selection depends on weight scale ({} failures):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

// ===========================================================================
// Test-the-tests: samplers with known defects must be rejected
// ===========================================================================

/// SUS reimplemented here. `phase_scale < 1` restricts the random offset to the
/// first part of the interval (a subtle bias); `shuffle = false` omits the
/// final shuffle of the output.
fn sweep(rng: &mut StdRng, k: usize, w: &[f64], phase_scale: f64, shuffle: bool) -> Vec<usize> {
    let total: f64 = w.iter().sum();
    let spacing = total / k as f64;
    let offset = phase_scale * rng.random::<f64>() * spacing;
    let mut out = Vec::with_capacity(k);
    let (mut idx, mut cum) = (0usize, w[0]);
    for j in 0..k {
        let arm = j as f64 * spacing + offset;
        while cum < arm && idx + 1 < w.len() {
            idx += 1;
            cum += w[idx];
        }
        out.push(idx);
    }
    if shuffle {
        out.shuffle(rng);
    }
    out
}

fn reference_sus(rng: &mut StdRng, k: usize, w: &[f64]) -> Vec<usize> {
    sweep(rng, k, w, 1.0, true)
}
fn biased_phase_sus(rng: &mut StdRng, k: usize, w: &[f64]) -> Vec<usize> {
    sweep(rng, k, w, 0.8, true)
}
fn unshuffled_sus(rng: &mut StdRng, k: usize, w: &[f64]) -> Vec<usize> {
    sweep(rng, k, w, 1.0, false)
}
/// Classic roulette wheel: k independent draws with replacement. Unbiased, but
/// counts are Binomial(k, p_i), not floor/ceil.
fn roulette_wheel(rng: &mut StdRng, k: usize, w: &[f64]) -> Vec<usize> {
    let cum: Vec<f64> = w
        .iter()
        .scan(0.0, |s, &x| {
            *s += x;
            Some(*s)
        })
        .collect();
    let total = *cum.last().expect("non-empty");
    (0..k)
        .map(|_| {
            let u = rng.random::<f64>() * total;
            cum.partition_point(|&c| c <= u).min(w.len() - 1)
        })
        .collect()
}

#[test]
fn checks_have_power_against_known_bad_samplers() {
    // Negative control: a correct independent implementation passes everything.
    let runs = run_all(reference_sus, TRIALS);
    check_bounds(&runs).expect("reference SUS must satisfy bounds");
    check_count_distribution(&runs).expect("reference SUS must pass the binomial test");
    check_unbiased(&runs).expect("reference SUS must be unbiased");
    check_position_distribution(reference_sus).expect("reference SUS output must be exchangeable");

    // Roulette wheel: right mean, wrong variance. The bound and the exact test
    // must catch it; the mean-only test must NOT, which is why the stronger
    // checks exist.
    let runs = run_all(roulette_wheel, TRIALS);
    assert!(
        check_bounds(&runs).is_err(),
        "roulette wheel violates floor/ceil bound"
    );
    assert!(
        check_count_distribution(&runs).is_err(),
        "roulette wheel has wrong count law"
    );
    assert!(
        check_unbiased(&runs).is_ok(),
        "roulette wheel is unbiased; the mean test alone cannot tell it from SUS"
    );

    // Offset biased towards the start: obeys the bound but gets the Bernoulli
    // probabilities wrong, so only the statistical checks can see it.
    let runs = run_all(biased_phase_sus, TRIALS);
    assert!(
        check_bounds(&runs).is_ok(),
        "biased phase still obeys floor/ceil"
    );
    assert!(
        check_count_distribution(&runs).is_err(),
        "binomial test must detect biased phase"
    );
    assert!(
        check_unbiased(&runs).is_err(),
        "mean test must detect biased phase"
    );

    // Missing output shuffle: count statistics are perfect, positions are not.
    let runs = run_all(unshuffled_sus, TRIALS);
    check_count_distribution(&runs).expect("unshuffled SUS has the right counts");
    assert!(
        check_position_distribution(unshuffled_sus).is_err(),
        "position test must detect a missing output shuffle"
    );
}

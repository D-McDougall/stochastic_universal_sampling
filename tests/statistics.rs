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
//! one seed in a million.
//!
//! The master seed is random on every run (see `seed()`), so that flukes and real
//! bugs get flushed out over time. Failure messages print the seed; set
//! `SUS_TEST_SEED=<n>` to replay a run exactly.
//!

use rand::prelude::*;
use rand::rngs::StdRng;
use statrs::distribution::{Binomial, ChiSquared, ContinuousCDF, DiscreteCDF, StudentsT};
use std::sync::OnceLock;

use stochastic_universal_sampling::choose_multiple_weighted as sus;

type Sampler = fn(&mut StdRng, usize, &[f64]) -> Vec<usize>;

/// The master seed for this file.
///
/// Every random sample is derived from this seed number. By default it truly
/// random (OS sourced). To replay a failure, set the environment variable:
///
///     SUS_TEST_SEED=123456789 cargo test --test statistics
///
/// Failure messages from statistical checks always print the seed.
fn seed() -> u64 {
    static SEED: OnceLock<u64> = OnceLock::new();
    *SEED.get_or_init(|| match std::env::var("SUS_TEST_SEED") {
        Ok(s) => s
            .parse()
            .expect("SUS_TEST_SEED must be an unsigned 64-bit integer"),
        Err(_) => rand::random(),
    })
}

/// Suffix appended to failure messages so a failing run can be reproduced.
fn replay_hint() -> String {
    format!("[replay with SUS_TEST_SEED={}]", seed())
}

/// Family-wise significance level, split across tests by Bonferroni.
const ALPHA: f64 = 1e-6;
/// Floating-point slack when deciding whether an expectation is an integer.
const TOL: f64 = 1e-9;
const TRIALS: usize = 50_000;
/// Minimum number of *expected* "minority events" before the t-test of layer 3
/// is trusted on an item. See `check_unbiased` for what that means and for why
/// 1000 (rather than the textbook "10 or so") is the right size when the
/// significance threshold is as extreme as ours (about 4e-9).
const MIN_EVENTS: f64 = 1_000.0;

// ===========================================================================
// Statistics helpers (thin wrappers over statrs)
// ===========================================================================
//
// A 60-second primer on how these tests decide "pass" or "fail"
// -------------------------------------------------------------
// The sampler is random, so its output never exactly matches theory, just as
// 1000 fair coin flips rarely give exactly 500 heads. A statistical test asks:
//
//     "If the implementation were CORRECT, how likely is a result at least this
//      far from the theoretical prediction?"
//
// That probability is called a *p-value*. A big p-value (0.3, say) means "totally
// ordinary, nothing to see". A tiny one (1e-9) means "a correct implementation
// would practically never do this", so we conclude the implementation is wrong.
// A p-value is NOT the probability that the code is buggy; it is the probability
// of the evidence assuming the code is fine.
//
// Each helper below turns raw tallies into one p-value, using a different
// statistical model depending on what is being compared.

/// Exact two-sided binomial test.
///
/// QUESTION: a coin lands heads with probability `p`. We tossed it `n` times and
/// saw `x` heads. How surprising is `x`, if the coin really has probability `p`?
///
/// HOW: the binomial distribution gives the exact probability of every possible
/// head-count 0..=n. Two tail probabilities measure surprise:
///
///   lower = P(X <= x)   chance of seeing this few heads, or fewer
///   upper = P(X >= x)   chance of seeing this many heads, or more
///
/// Whichever tail is smaller is the one we are surprised by. We double it
/// because we would be equally suspicious of "too many" and "too few" (that is
/// what "two-sided" means), and cap at 1 because probabilities cannot exceed 1.
/// (Doubling the smaller tail is a standard convention; it can overstate the
/// p-value but never understate it, so it never rejects more often than
/// advertised.)
fn binomial_p(x: usize, n: usize, p: f64) -> f64 {
    let b = Binomial::new(p, n as u64).expect("probability in (0, 1)");
    // P(X <= x). ("cdf" = cumulative distribution function.)
    let lower = b.cdf(x as u64);
    // `sf` is the "survival function", 1 - cdf. sf(x - 1) = P(X > x - 1) = P(X >= x).
    // (When x == 0 there is no x - 1, and P(X >= 0) is trivially 1.)
    let upper = if x == 0 { 1.0 } else { b.sf(x as u64 - 1) };
    (2.0 * lower.min(upper)).min(1.0)
}

/// Two-sided p-value for a Student's t statistic with `df` degrees of freedom.
///
/// BACKGROUND: a t statistic says "how many standard errors is my sample average
/// away from the value I expected?" (see `check_unbiased`). If the data are
/// roughly bell-shaped, t follows Student's t-distribution: a bell curve with
/// slightly fatter tails than the standard normal, because the spread had to be
/// *estimated* from the same noisy data. `df` ("degrees of freedom", here the
/// number of samples minus 1) controls how fat: with ~50 000 samples the curve is
/// indistinguishable from the standard normal bell curve.
///
/// `sf(|t|)` is the area under the curve to the right of |t|, i.e. P(T > |t|).
/// The bell curve is symmetric, so doubling it also covers the left tail.
fn t_p(t: f64, df: f64) -> f64 {
    2.0 * StudentsT::new(0.0, 1.0, df).expect("df > 0").sf(t.abs())
}

/// p-value of a chi-squared statistic: P(a correct model gives a statistic >= `stat`).
///
/// `df` (degrees of freedom) is the number of buckets minus 1. Minus one because
/// the bucket tallies must add up to the known total number of trials, so once
/// you know all but one bucket, the last is forced: only `buckets - 1` of them
/// are free to wobble.
///
/// Only the upper tail is used: a chi-squared statistic is a sum of squared
/// misses, so it is small when the fit is good and large when it is bad.
/// Unusually *small* values (suspiciously perfect fits) are not what we hunt.
fn chi2_p(stat: f64, df: usize) -> f64 {
    ChiSquared::new(df as f64).expect("df > 0").sf(stat)
}

/// Pearson's chi-squared statistic, a "goodness of fit" score:
///
///     X^2 = sum over buckets of (observed - expected)^2 / expected
///
/// In words, for every bucket: measure how far the tally strayed from its
/// prediction; square that (so overshoot and undershoot both count, and big
/// misses count extra); divide by the prediction (missing by 10 matters when you
/// expected 20, but is noise when you expected 20 000); then add everything up.
/// A good fit scores around `buckets - 1`. A score far above that means the
/// observed tallies do not look like the prediction.
///
/// Caveat: the p-value computed from this score (`chi2_p`) is an approximation
/// that needs every bucket's expectation to be reasonably large (rule of thumb:
/// at least 5). That is why rare-event items use `binomial_p` instead.
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
///
/// THE PROBLEM: if you run `m` tests and each has, say, a 1-in-a-million chance
/// of a false alarm, then across all `m` the chance that at least one cries wolf
/// is about `m` in a million. It is like buying m lottery tickets: each is a long
/// shot, but together you win more often. Run 259 tests and your "one in a
/// million" test becomes "one in four thousand".
///
/// THE FIX (Bonferroni correction): demand that each individual p-value beat
/// `ALPHA / m`. By Boole's inequality (the probability that at least one of
/// several events happens is at most the SUM of their individual probabilities),
/// the chance that ANY test false-alarms is then at most m * (ALPHA / m) = ALPHA.
/// Crucially that inequality needs no independence assumption, which matters
/// here because the item counts are correlated (they must sum to k).
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
            "{what}: REJECTED at {} with p = {:.3e} < {threshold:.3e} ({m} tests) {}",
            worst.label,
            worst.p,
            replay_hint()
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
    ///
    /// We never stored the individual counts, only a histogram: `hist[c]` = how
    /// many trials produced count `c`. That is all we need, because both numbers
    /// only depend on two running totals:
    ///
    ///     sum    = Σ x         (add up every observed count)
    ///     sum_sq = Σ x²        (add up the square of every observed count)
    ///
    /// A bucket `c` that occurred `n` times contributes `c * n` and `c² * n`.
    ///
    ///     mean     = sum / T
    ///     variance = (sum_sq − T·mean²) / (T − 1)
    ///
    /// Variance measures how spread out the counts are: the average squared
    /// distance from the mean. Dividing by `T − 1` instead of `T` is "Bessel's
    /// correction": the data were already used once to compute the mean, which
    /// makes them look slightly tighter around it than the truth, and `T − 1`
    /// compensates exactly, so that on average this equals the true variance.
    ///
    /// Numerical note: "sum_sq − T·mean²" is a textbook trap when the mean is huge
    /// compared to the spread, because two nearly-equal big floats cancel and
    /// leave mostly rounding noise. It is harmless here: counts are small
    /// integers, so `sum` and `sum_sq` are exactly-representable integers
    /// (< 2^53), and the cancellation costs only about 1e-10 of relative
    /// accuracy, far below the statistical noise of the test.
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

/// Calculates what the histogram should look like
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
        .map(|(i, f)| run_fixture(sampler, f, trials, seed().wrapping_add(i as u64)))
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
                     but {bad} of {} trials fell outside {}",
                    r.name,
                    it.i,
                    it.e,
                    it.lo,
                    it.hi,
                    r.trials,
                    replay_hint()
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
///
/// THE TEST. For one item we observed `T` counts (one per trial). If the sampler
/// is unbiased their average should land near the expectation `e`. But how near
/// is "near"? That depends on how noisy the counts are, so we measure the gap in
/// units of the *standard error*, the typical wobble of an average:
///
///     t = (mean − e) / (s / √T)        where s² is the sample variance
///
/// Why `s / √T`? The Central Limit Theorem says an average of many independent
/// draws is approximately bell-curve (normal) distributed around the true mean,
/// with spread `σ / √T` (σ = spread of a single draw), whatever the shape of the
/// individual draws. Estimating σ by `s` makes `t` follow Student's t-distribution,
/// which `t_p` converts into a p-value. A gap of 6 standard errors or more would
/// occur by pure chance only about once in a billion times.
///
/// THE CATCH: "approximately a bell curve" is only true when the data contain
/// plenty of information about the spread. Consider the "rare events" fixture,
/// weights [1000, 1000, 0.05], k = 4. Item 0 expects e = 1.99995 copies. SUS gives
/// it 2 copies in all but one in 20 000 trials, where it gets 1. In T = 50 000
/// trials we therefore expect only 2.5 such deviations. Poisson statistics
/// (the law of rare events) say there is an e^−2.5 ≈ 8% chance of seeing NONE.
/// Then every count is 2, the sample variance is exactly 0, and the t statistic
/// is 0/0. The previous version of this check treated "no variation, and mean ≠ e"
/// as proof of bias ("a constant but wrong count is infinitely significant"). But
/// mean ≠ e (2 vs 1.99995) only because the rare deviations did not happen to
/// show up, which is the single most likely outcome. That made this test fail on
/// roughly 1 run in 6 with a correct implementation.
///
/// THE RULE. Decide whether an item is testable from the NULL MODEL alone,
/// before looking at any data (peeking at the data to choose tests voids the
/// p-value guarantee). The relevant quantity is the expected number
/// of "minority events": trials in which the item takes its less common
/// value, floor(e) or ceil(e). With f = frac(e), that is
///
///     T · min(f, 1 − f)
///
/// If this is below `MIN_EVENTS` we skip the item here; it is still covered by
/// the exact binomial test of layer 2, which is valid at any rarity.
///
/// WHY MIN_EVENTS = 1000 and not the textbook ~10. With only a few hundred
/// events the bell curve is a poor model for the *extreme* tail we use, because
/// the same sparse data estimates both the mean and the spread, so a chance
/// shortfall of events also shrinks the estimated spread and inflates `t`. We
/// quantified it by enumerating the exact binomial distribution at T = 50 000
/// and our Bonferroni threshold (~4e-9). The actual false-alarm probability per
/// item, as a multiple of the intended one:
///
///     expected events:   10     100    1 000   10 000
///     × nominal:         ≫1     ~160     ~4      ~1.1
///
/// 1000 keeps the per-item error within a small factor. Nearly all items sit far
/// above that (thousands of events), so the whole family stays within ~1.2× ALPHA.
///
/// ZERO OBSERVED VARIANCE:
///  * random item that passed the gate: the null model says variation is
///    near-certain (a variation-free run has probability below e^−1000), so a
///    constant, wrong mean really is overwhelming evidence of bias.
///  * deterministic item (e is an integer): SUS must return exactly e every time,
///    so a constant wrong count is a genuine defect. Any other sampler with
///    nonzero variance on such an item is simply t-tested (roulette does this).
fn check_unbiased(runs: &[Run]) -> Result<(), String> {
    let mut evidence = Vec::new();
    for r in runs {
        let t = r.trials as f64;
        for it in r.items() {
            // Null-model gate (never looks at the data). Integer expectations
            // are exempt: their count is deterministic, nothing to approximate.
            let expected_minority_events = t * it.frac().min(1.0 - it.frac());
            if it.is_random() && expected_minority_events < MIN_EVENTS {
                continue;
            }

            let (mean, var) = r.mean_var(&it);
            let label = format!("[{}] item {} (e = {:.4})", r.name, it.i, it.e);
            if var <= 1e-12 {
                // Every trial gave the identical count; the t statistic would
                // be 0/0, so judge the constant directly (see doc comment).
                if (mean - it.e).abs() > 1e-6 {
                    evidence.push(Evidence { p: 0.0, label });
                }
            } else {
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
fn stats_counts_are_always_floor_or_ceil_of_expectation() {
    check_bounds(crate_runs()).unwrap();
}

/// Layer 1 on hundreds of random weight vectors (zeros, wide dynamic range).
#[test]
fn stats_random_weight_vectors_obey_floor_ceil_bounds() {
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
        check_bounds(&[run_fixture(
            sus::<StdRng>,
            &f,
            200,
            seed().wrapping_add(case),
        )])
        .unwrap();
    }
}

/// Layer 2: each item's count is floor(e) + Bernoulli(frac(e)).
#[test]
fn stats_count_distribution_is_floor_plus_bernoulli_of_fraction() {
    check_count_distribution(crate_runs()).unwrap();
}

/// Layer 3: E[count_i] = k * p_i.
#[test]
fn stats_expected_count_is_proportional_to_weight() {
    check_unbiased(crate_runs()).unwrap();
}

/// Layer 4: the final shuffle makes every output position equally likely to
/// hold any copy, so position j shows item i with probability p_i.
#[test]
fn stats_output_positions_are_proportional_and_exchangeable() {
    check_position_distribution(sus::<StdRng>).unwrap();
}

/// Selection probabilities must not depend on the absolute scale of the weights.
#[test]
fn stats_bounds_hold_at_every_weight_scale() {
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

/// This test proves the checks can actually reject, by running them against
/// deliberately broken samplers.
#[test]
fn stats_checks_have_power_against_known_bad_samplers() {
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

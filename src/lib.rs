#![doc = include_str!("../README.md")]

use rand::prelude::*;

const NO_DATA: &str = "no data: can not choose from empty set";

/// The stochastic universal sampling algorithm
///
/// Chooses `amount` elements at random, with repetition, and in random order.
/// The likelihood of each element’s inclusion in the output is specified by
/// the `weights` array.  All weights must be finite and greater than or equal
/// to zero. If all of the weights are equal, even if they are all zero, then
/// each element has an equal likelihood of being selected.
///
/// Each element `i` is returned either `floor(e_i)` or `ceil(e_i)` times, where
/// `e_i = amount * weights[i] / sum(weights)` is its expected number of copies.
/// (This is what distinguishes SUS from independent "roulette wheel" draws,
/// where the number of copies can stray arbitrarily far from `e_i`.)
///
/// The input order is randomized internally, so the position of an element in
/// `weights` has no influence on which elements are chosen *together*.
///
/// Returns a vector of indices into the weights array.
///
/// # Panics
///
/// This function panics in the following cases:
///
/// - If `amount > 0` and `weights` is empty (no data to choose from).
/// - If any element in `weights` is negative (invalid probability).
/// - If any element in `weights` is not finite (e.g., NaN or infinity).
///
pub fn choose_multiple_weighted<R>(rng: &mut R, amount: usize, weights: &[f64]) -> Vec<usize>
where
    R: Rng + ?Sized,
{
    // Validate the input first so that a bad `weights` array is reported
    // consistently instead of only when it happens to be used.
    let mut max_weight: f64 = 0.0;
    for (i, &w) in weights.iter().enumerate() {
        assert!(
            w.is_finite() && w >= 0.0,
            "invalid weight: weights[{i}] = {w} (must be finite and >= 0)"
        );
        max_weight = max_weight.max(w);
    }
    if amount == 0 {
        return vec![];
    } else {
        assert!(!weights.is_empty(), "{NO_DATA}");
    }
    // If every weight is zero then fall back to uniform sampling.
    if max_weight == 0.0 {
        return choose_multiple(rng, amount, weights.len());
    }

    // Shuffle the input locations. The sampling arms are rigidly coupled: one
    // random number places all of them, so without this, which elements get
    // picked *together* would depend on the order of the inputs
    // (e.g. adjacent inputs would be less likely to be picked together).
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.shuffle(rng);

    // Apply a cumulative summation to the (rescaled) weights, accessed in
    // shuffled order. Element `order[j]` owns the half-open interval
    // `[cumulative[j-1], cumulative[j])` of the number line, whose length is
    // its weight. A pointer landing inside that interval selects the element.
    // Zero-weight elements own an empty interval and so can never be hit.
    let cumulative: Vec<f64> = order
        .iter()
        .scan(0.0, |running_total, &original_index| {
            // Rescale so the largest weight is 1.0.
            //
            // WHY: floating-point numbers have a limited range. Adding up huge weights
            // can *overflow* to infinity (f64::MAX + f64::MAX = inf) and dividing a
            // tiny total by `amount` can *underflow* to zero, in which case every
            // pointer lands on the same spot and the output is garbage. Dividing every
            // weight by the largest weight is harmless, because only the *ratios* between
            // weights matter, but it guarantees that `1.0 <= total <= weights.len()`:
            // safely away from both cliffs, for any finite input.
            *running_total += weights[original_index] / max_weight;
            Some(*running_total)
        })
        .collect();
    let total_weight = *cumulative.last().unwrap(); // Safe to unwrap: `weights` is not empty
    debug_assert!(total_weight.is_finite() && total_weight >= 1.0);

    // Find the last element that can legitimately be selected. This finds and
    // discards any zero-weight elements at the end of the shuffled weights array.
    let mut range_end = cumulative.len();
    while range_end > 0 && cumulative[range_end - 1] >= total_weight {
        range_end -= 1;
    }

    // Generate the random number to sample from the weights cumsum
    let arm_spacing = total_weight / (amount as f64);
    let arm_offset = rng.random::<f64>() * arm_spacing;

    // Find the indices of random numbers in the weights cumsum
    let mut samples = Vec::with_capacity(amount);
    let mut index = 0;
    for arm in 0..amount {
        let arm = (arm as f64) * arm_spacing + arm_offset;
        while index < range_end && cumulative[index] <= arm {
            index += 1;
        }
        let original_index = order[index]; // Undo the input order shuffle
        samples.push(original_index);
    }

    // Shuffle the random sample to break up any runs of repeated elements.
    // Needed BC: repeated elements will always be adjacent because elements
    // are chosen via a single scan through the cumulative weights array.
    samples.shuffle(rng);
    samples
}

/// The stochastic universal sampling algorithm, with uniform weights
///
/// Chooses `amount` elements from the range `0..items` at random, with
/// repetition, and in random order. All elements have an equal likelihood of
/// being selected. This is equivalent to calling [choose_multiple_weighted()]
/// with weights that are all equal.
///
/// Returns a vector of indices in the range `(0..items)`.
///
/// # Panics
///
/// This function panics if `amount > 0` and `items == 0` (no data to choose from).
///
pub fn choose_multiple<R>(rng: &mut R, amount: usize, items: usize) -> Vec<usize>
where
    R: Rng + ?Sized,
{
    assert!(amount == 0 || items > 0, "{NO_DATA}");
    let mut results = Vec::with_capacity(amount);
    while results.len() < amount {
        let num_samples = amount - results.len();
        // If over-sampling then get uniform coverage of the items using non-random numbers.
        if num_samples >= items {
            results.extend(0..items);
        } else {
            // Select the final elements using choose-without-replacement.
            results.extend(rand::seq::index::sample(rng, items, num_samples));
        }
    }
    results.shuffle(rng);
    results
}

#[cfg(test)]
mod tests {
    use super::choose_multiple_weighted as sus;

    fn assert_data_eq(a: &mut [usize], b: &mut [usize]) {
        a.sort();
        b.sort();
        assert_eq!(a, b);
    }

    #[test]
    fn no_data() {
        let mut rng = rand::rng();
        assert_data_eq(&mut sus(&mut rng, 0, &[]), &mut []);
        assert_data_eq(&mut sus(&mut rng, 0, &[1.0, 2.0, 3.0]), &mut []);
    }

    #[test]
    #[should_panic]
    fn no_data_panic() {
        let mut rng = rand::rng();
        sus(&mut rng, 100, &[]);
    }

    #[test]
    fn not_enough_data() {
        let mut rng = rand::rng();
        assert_data_eq(&mut sus(&mut rng, 2, &[1.0]), &mut [0, 0]);
    }

    #[test]
    fn zero_data() {
        let mut rng = rand::rng();
        assert_data_eq(&mut sus(&mut rng, 1, &[0.0]), &mut [0]);
        assert_data_eq(
            &mut sus(&mut rng, 10, &[0.0; 10]),
            &mut [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        );
        assert_data_eq(&mut sus(&mut rng, 6, &[0.0; 3]), &mut [0, 0, 1, 1, 2, 2]);
        sus(&mut rng, 7, &[0.0; 3]); // must not crash
    }

    #[test]
    fn round_robin() {
        let mut rng = rand::rng();
        assert_data_eq(&mut sus(&mut rng, 3, &[1.0; 3]), &mut [0, 1, 2]);
        assert_data_eq(&mut sus(&mut rng, 6, &[1.0; 3]), &mut [0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn it_works() {
        let mut rng = rand::rng();
        assert_data_eq(&mut sus(&mut rng, 2, &[1.0, 0.0, 1.0]), &mut [0, 2]);
        assert_data_eq(&mut sus(&mut rng, 3, &[2.0, 0.0, 1.0]), &mut [0, 0, 2]);
        assert_data_eq(&mut sus(&mut rng, 3, &[1.0, 0.0, 0.5]), &mut [0, 0, 2]);
        assert_data_eq(
            &mut sus(&mut rng, 6, &[1.0, 2.0, 3.0]),
            &mut [0, 1, 1, 2, 2, 2],
        );
    }

    #[test]
    fn sample_one() {
        let mut rng = rand::rng();
        let mut data = [0.0; 10000];
        data[1234] = 0.0000001;
        assert_data_eq(&mut sus(&mut rng, 1, &data), &mut [1234]);
    }

    #[test]
    fn random_data() {
        let mut rng = rand::rng();
        assert!(sus(&mut rng, 13, &[1.0; 10000]) != sus(&mut rng, 13, &[1.0; 10000]));
        assert!(sus(&mut rng, 40, &[1.0; 2000]) != sus(&mut rng, 40, &[1.0; 2000]));
    }

    #[test]
    fn random_order() {
        let mut rng = rand::rng();
        let mut a = sus(&mut rng, 2000, &[1.0; 2000]);
        let mut b = sus(&mut rng, 2000, &[1.0; 2000]);
        assert!(a != b);
        assert_data_eq(&mut a, &mut b);
    }

    #[test]
    fn random_order_repeats() {
        let mut rng = rand::rng();
        for _ in 0..100 {
            let mut a = sus(&mut rng, 100, &[1.0; 2]);
            let mut b = sus(&mut rng, 100, &[1.0; 2]);
            assert!(a != b);
            assert_data_eq(&mut a, &mut b);
        }
    }

    #[test]
    fn uniform_weights() {
        let mut rng = rand::rng();

        let mut a = sus(&mut rng, 13, &[0.3; 13]);
        let mut b = sus(&mut rng, 13, &[0.0; 13]);
        let mut c = super::choose_multiple(&mut rng, 13, 13);
        assert_data_eq(&mut a, &mut b);
        assert_data_eq(&mut a, &mut c);

        let mut x = super::choose_multiple(&mut rng, 1000, 1033);
        let mut y = super::choose_multiple(&mut rng, 1000, 1033);

        x.sort();
        y.sort();
        assert!(x != y);
    }

    #[test]
    #[ignore = "performance benchmark"]
    fn benchmark() {
        use rand::RngExt;
        let mut rng = rand::rng();
        let amount = 1000;
        let num_weights = 1_000_000;
        let weights: Vec<f64> = (0..num_weights).map(|_| rng.random()).collect();
        println!("Running SUS(amount: {amount}, num_weights: {num_weights}) ...",);
        std::thread::yield_now();
        let start_time = std::time::Instant::now();
        std::hint::black_box(sus(&mut rng, amount, &weights));
        let elapsed_time = start_time.elapsed();
        println!("Elapsed time: {elapsed_time:?}");
    }
}

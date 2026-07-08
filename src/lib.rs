use rand::prelude::*;

const NO_DATA: &'static str = "no data: can not choose from empty set";

/// The stochastic universal sampling algorithm
///
/// Chooses `amount` elements at random, with repetition, and in random order.
/// The likelihood of each element’s inclusion in the output is specified by
/// the `weights` array.  All weights must be greater than or equal to zero. If
/// all of the weights are equal, even if they are all zero, then each element
/// has an equal likelihood of being selected.
///
/// Returns a vector of indices into the weights array.
pub fn choose_multiple_weighted<R>(rng: &mut R, amount: usize, weights: &[f64]) -> Vec<usize>
where
    R: Rng + ?Sized,
{
    if amount == 0 {
        return vec![];
    } else {
        assert!(!weights.is_empty(), "{NO_DATA}");
    }
    // Apply a cumulative summation to the weights.
    let weights: Vec<_> = (0..weights.len())
        .scan(0.0, |sum, idx| {
            debug_assert!(weights[idx] >= 0.0);
            *sum += weights[idx];
            Some(*sum)
        })
        .collect();
    // Check for all zero weights.
    let total_weight = *weights.last().expect("Internal Error");
    if total_weight <= f64::EPSILON * weights.len() as f64 {
        return choose_multiple(rng, amount, weights.len());
    }
    assert!(total_weight.is_finite());
    // Generate the random numbers to sample from the weights cumsum.
    let arm_spacing = total_weight / (amount as f64);
    let arm_offset = rng.random::<f64>() * arm_spacing;
    // Find the indices of random numbers in the weights cumsum.
    let mut samples = Vec::with_capacity(amount);
    let mut idx = 0;
    for arm in 0..amount {
        let arm = (arm as f64) * arm_spacing + arm_offset;
        while idx < weights.len() && weights[idx] < arm {
            idx += 1;
        }
        samples.push(idx);
    }
    // Shuffle the random sample to break up any runs of repeated elements.
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
/// Returns a vector of indices into the items range.
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
    return results;
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
        sus(&mut rng, 7, &[0.0; 3]);
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
        assert!(sus(&mut rng, 1, &[1.0; 10000]) != sus(&mut rng, 1, &[1.0; 10000]));
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
    #[ignore]
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

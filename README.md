# Stochastic Universal Sampling

Weighted random selection for the [`rand`](https://crates.io/crates/rand)
crate, with a hard guarantee on how far the results can stray from their
expected values.

Given a list of weights, this crate picks items with repetition, where each item
is chosen in proportion to its weight. It returns indices into the weights
array, in random order. The stochastic universal sampling (SUS) algorithm has
low variance in the expected *counts*: item are picked exactly as often as
their weight entitles them to be picked.

* [**crates.io**](https://crates.io/crates/stochastic_universal_sampling)
* [**github.com**](https://github.com/D-McDougall/stochastic_universal_sampling)
* [**docs.rs**](https://docs.rs/stochastic_universal_sampling)

# Installation

To install this package into your current rust project, run the command:

```bash
$ cargo add stochastic_universal_sampling
```

# Examples

```rust
use stochastic_universal_sampling as sus;
let rng = &mut rand::rng();

println!("{:?}", sus::choose_multiple_weighted(rng, 8, &[1.0, 2.0, 3.0, 4.0]));
// Example Output: [3, 1, 2, 2, 3, 3, 1, 0]

println!("{:?}", sus::choose_multiple(rng, 4, 2));
// Example Output: [1, 0, 1, 0]
```

# Statistical Properties

Let `w_i` be the weight of item `i`, and let

```text
e_i = k · w_i / (w_1 + w_2 + … + w_n)
```

be the **expected number of copies** of item `i` in `k` picks. For example,
with weights `[1, 2, 3, 4]` and `k = 8`, the expected counts are `0.8, 1.6, 2.4,
3.2`. SUS has the following externally visible properties.

1. **Bounded counts.** In every single run, item `i` appears either
   `floor(e_i)` or `ceil(e_i)` times, never anything else. In the example
   above, item 0 appears 0 or 1 times, item 1 appears 1 or 2 times, item 2
   appears 2 or 3 times, and item 3 appears 3 or 4 times. The actual count
   is never more than one away from the expected count.
2. **Unbiased, with minimal variance.** `P(count_i = ceil(e_i)) = frac(e_i)`.
   So the average count over many runs is exactly `e_i`, and the variance is
   `f · (1 − f)` with `f = frac(e_i)`.
3. **No unlucky misses.** An item with `e_i ≥ 1` is *always* picked at least
   once. An item with weight zero is *never* picked.
4. **Each position is an ordinary weighted draw.** The output is shuffled, so
   looking at any one position in the result, the item there is distributed
   exactly like a single weighted draw: item `i` with probability
   `w_i / (w_1 + … + w_n)`.
5. **Picks within one call are not independent.** The counts are negatively
   correlated, since they must add up to `k`. Whenever one item gets its extra
   copy, another gets one fewer.

## Comparison with Roulette-Wheel Sampling

Roulette-wheel sampling makes `k` independent weighted draws. Each draw is
correct on its own, so it matches SUS on property 4 and is also unbiased: the
average count of item `i` is `e_i` for both methods. What differs is the
**spread** of the counts around that average. In roulette sampling, `count_i`
follows a Binomial(`k`, `p_i`) distribution with `p_i = w_i / Σw`. Its variance
is `k · p_i · (1 − p_i)`, which grows with `k`, and its range is the whole of
`0..=k`.

Using the same example (weights `[1, 2, 3, 4]`, `k = 8`):

| Item | Expected count | SUS: possible counts | SUS: variance | Roulette: possible counts | Roulette: variance |
|-----:|---------------:|:--------------------:|--------------:|:-------------------------:|-------------------:|
| 0    | 0.8            | 0 or 1               | 0.16          | 0 to 8                    | 0.72               |
| 1    | 1.6            | 1 or 2               | 0.24          | 0 to 8                    | 1.28               |
| 2    | 2.4            | 2 or 3               | 0.24          | 0 to 8                    | 1.68               |
| 3    | 3.2            | 3 or 4               | 0.16          | 0 to 8                    | 1.92               |

The difference is easy to feel in small cases:

* With weights `[1, 2, 3]` and `k = 6`, the expected counts are exactly
  `1, 2, 3`, so SUS returns indices `{0, 1, 1, 2, 2, 2}` (in some order) **every
  time**. A roulette wheel would leave item 0 out entirely in about 33% of runs
  (`(5/6)^6 ≈ 0.335`), even though its expected count is 1.
* With weights `[1, 1, 1]` and `k = 2`, SUS always returns two *distinct*
  items. A roulette wheel returns the same item twice one run in three.

This matters in evolutionary algorithms, where selection noise is a nuisance:
with roulette selection, a mediocre individual can be over-represented, or an
excellent one missed, purely by chance. SUS keeps selection pressure
consistently faithful to the weights.

## How it works

The classic SUS algorithm (Baker, 1987) is a roulette wheel with `k` equally
spaced pointers and a single spin. This crate follows it closely, plus a few
changes that make the output safe to use as a general-purpose sampler. The
function is `choose_multiple_weighted(rng, k, weights)`, and each step below
corresponds to one stage of it.

1) **Validate.** Every weight must be non-negative and finite. If `k == 0`,
   return an empty vector immediately. Otherwise, `weights` must be non-empty.
   Violations panic.

2) **Shuffle the input order** *(modification)*. Randomly permute the input
   `weights` array to decorrelates adjacent elements. The per-item count guarantee
   holds regardless, but this step makes every *combination* of items reachable.

3) **Lay the weights end to end.** A cumulative summation over the shuffled
   weights turns each item into an interval on a number line from `0` to `Σw`,
   whose length equals its weight.

4) **Place the pointers.** The pointer spacing is `Σw / k`. A single random
   offset in `[0, spacing)` fixes the position of the first pointer, and the
   others follow at exactly one spacing apart. This single random number is
   the *only* randomness in the selection itself, and it is the source of the
   count guarantee: an item whose interval is `e_i` spacings long can contain only
   `floor(e_i)` or `ceil(e_i)` pointers.

5) **Read off the items.** The pointers are visited in ascending order. For each
   pointer, we advance past every interval that ends at or before it, and the
   interval it lands in is the selected item. Zero-weight items have empty
   intervals and are skipped.

6) **Shuffle the output** *(modification)*. The scan emits repeated items in
   contiguous blocks. A final shuffle ensures that the output is in random order.

### All-Zero Weights

If every weight is zero the function falls back to `choose_multiple(rng, k, weights.len())`.
This is the "equal weights" case, and it is also exposed on its own. It deals
out the indices `0..n` in complete rounds while at least `n` picks remain, then
picks the remainder uniformly *without* replacement, and shuffles the result.
Every item therefore appears `floor(k / n)` or `ceil(k / n)` times, which meets
the same SUS guarantees but for uniform weights.

### Computational Cost

For `n` weights and `k` picks, time is **O(n + k)** and memory is **O(n + k)**.

# References

* [**Wikipedia**](https://en.wikipedia.org/wiki/Stochastic_universal_sampling)
* **Introduction to Evolutionary Computing**  
  A.E. Eiben and J.E. Smith, 2003, 2015  
  <https://doi.org/10.1007/978-3-662-44874-8>  
  _(See chapter 5)_
* **Reducing Bias and Inefficiency in the Selection Algorithm**  
  James E. Baker (1987)  
  Proceedings of the Second International Conference on Genetic Algorithms and Their Application.  
  Hillsdale, New Jersey: L. Erlbaum Associates  
  Pages 14–21.  

# Copyright & License

Copyright 2024 David McDougall.

Licensed under the MIT No Attribution (MIT-0) license.

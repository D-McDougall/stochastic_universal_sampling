# Stochastic Universal Sampling

This package implements the stochastic universal sampling (SUS) algorithm for
the rand crate. The SUS algorithm is essentially a random selection algorithm.
SUS guarantees that highly-weighted samples will not dominate the selection
beyond their proportional weight. This is useful for evolutionary algorithms.
For more information see:

* [**crates.io**](https://crates.io/crates/stochastic_universal_sampling)
* [**docs.rs**](https://docs.rs/stochastic_universal_sampling)
* [**Wikipedia**](https://en.wikipedia.org/wiki/Stochastic_universal_sampling)
* Introduction to Evolutionary Computing  
  A.E. Eiben and J.E. Smith, 2003, 2015  
  <https://doi.org/10.1007/978-3-662-44874-8>  
  _(See chapter 5)_

# Copyright & License

Copyright 2024 David McDougall.

Licensed under the MIT No Attribution (MIT-0) license.

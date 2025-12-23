//! # BayesianMultiOutputConfig - Trait Implementations
//!
//! This module contains trait implementations for `BayesianMultiOutputConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for BayesianMultiOutputConfig {
    fn default() -> Self {
        Self {
            weight_prior: PriorDistribution::Normal(0.0, 1.0),
            bias_prior: PriorDistribution::Normal(0.0, 1.0),
            noise_prior: PriorDistribution::Gamma(1.0, 1.0),
            inference_method: InferenceMethod::Variational,
            n_samples: 1000,
            burn_in: 100,
            thin: 1,
            max_iter: 1000,
            tol: 1e-6,
            random_state: None,
        }
    }
}

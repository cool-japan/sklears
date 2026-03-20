//! # ADVIGaussianMixture - Trait Implementations
//!
//! This module contains trait implementations for `ADVIGaussianMixture`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Fit`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::ArrayView2;
use scirs2_core::random::{thread_rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
};

use super::types::{ADVIGaussianMixture, ADVIGaussianMixtureTrained};

impl Default for ADVIGaussianMixture<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator<Untrained> for ADVIGaussianMixture<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for ADVIGaussianMixture<Untrained> {
    type Fitted = ADVIGaussianMixtureTrained;
    fn fit(self, X: &ArrayView2<f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _n_features) = X.dim();
        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than number of components".to_string(),
            ));
        }
        let mut rng = match self.random_state {
            Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
            None => scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng()),
        };
        let mut best_model = None;
        let mut best_lower_bound = f64::NEG_INFINITY;
        for _ in 0..self.n_init {
            let (
                weight_concentration,
                mean_precision,
                mean_values,
                precision_values,
                degrees_of_freedom,
                scale_matrices,
            ) = self.initialize_parameters(X, &mut rng)?;
            let result = self.run_advi(
                X,
                weight_concentration,
                mean_precision,
                mean_values,
                precision_values,
                degrees_of_freedom,
                scale_matrices,
                &mut rng,
            )?;
            if result.lower_bound > best_lower_bound {
                best_lower_bound = result.lower_bound;
                best_model = Some(result);
            }
        }
        match best_model {
            Some(model) => Ok(model),
            None => Err(SklearsError::ConvergenceError {
                iterations: self.max_iter,
            }),
        }
    }
}

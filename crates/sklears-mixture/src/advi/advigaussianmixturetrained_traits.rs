//! # ADVIGaussianMixtureTrained - Trait Implementations
//!
//! This module contains trait implementations for `ADVIGaussianMixtureTrained`.
//!
//! ## Implemented Traits
//!
//! - `Estimator`
//! - `Predict`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Predict, Trained},
};

use super::types::ADVIGaussianMixtureTrained;

impl Estimator<Trained> for ADVIGaussianMixtureTrained {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<usize>> for ADVIGaussianMixtureTrained {
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<usize>> {
        let probabilities = self.predict_proba(X)?;
        let mut predictions = Array1::zeros(X.nrows());
        for i in 0..X.nrows() {
            let mut max_prob = 0.0;
            let mut best_class = 0;
            for k in 0..self.n_components {
                if probabilities[[i, k]] > max_prob {
                    max_prob = probabilities[[i, k]];
                    best_class = k;
                }
            }
            predictions[i] = best_class;
        }
        Ok(predictions)
    }
}

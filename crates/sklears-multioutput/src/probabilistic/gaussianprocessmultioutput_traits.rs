//! # GaussianProcessMultiOutput - Trait Implementations
//!
//! This module contains trait implementations for `GaussianProcessMultiOutput`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Estimator`
//! - `Estimator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;

impl Default for GaussianProcessMultiOutput<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianProcessMultiOutput<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for GaussianProcessMultiOutput<GaussianProcessMultiOutputTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &()
    }
}

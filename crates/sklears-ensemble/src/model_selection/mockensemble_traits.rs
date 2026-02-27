//! # MockEnsemble - Trait Implementations
//!
//! This module contains trait implementations for `MockEnsemble`.
//!
//! ## Implemented Traits
//!
//! - `Predict`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, Axis};

use super::types::MockEnsemble;

impl Predict<Array2<Float>, Array1<Float>> for MockEnsemble {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        Ok(Array1::from_elem(x.nrows(), self.mean_prediction))
    }
}


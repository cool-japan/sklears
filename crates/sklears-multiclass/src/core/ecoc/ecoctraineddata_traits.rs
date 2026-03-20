//! # ECOCTrainedData - Trait Implementations
//!
//! This module contains trait implementations for `ECOCTrainedData`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ECOCTrainedData;

impl<T: Clone> Clone for ECOCTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            code_matrix: self.code_matrix.clone(),
            n_features: self.n_features,
        }
    }
}

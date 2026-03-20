//! # FeatureCountObjective - Trait Implementations
//!
//! This module contains trait implementations for `FeatureCountObjective`.
//!
//! ## Implemented Traits
//!
//! - `ObjectiveFunction`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::Result as SklResult;

impl ObjectiveFunction for FeatureCountObjective {
    fn evaluate(
        &self,
        _x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let n_selected = feature_mask.iter().filter(|&&selected| selected).count();
        let n_total = feature_mask.len();
        Ok(-(n_selected as f64) / (n_total as f64))
    }
    fn name(&self) -> &str {
        "feature_count"
    }
    fn is_minimization(&self) -> bool {
        true
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

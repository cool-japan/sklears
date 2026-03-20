//! # FeatureImportanceObjective - Trait Implementations
//!
//! This module contains trait implementations for `FeatureImportanceObjective`.
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

impl ObjectiveFunction for FeatureImportanceObjective {
    fn evaluate(
        &self,
        _x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let total_importance: f64 = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| {
                if selected {
                    Some(self.importance_scores[i])
                } else {
                    None
                }
            })
            .sum();
        let max_possible: f64 = self.importance_scores.iter().sum();
        Ok(total_importance / max_possible)
    }
    fn name(&self) -> &str {
        "feature_importance"
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

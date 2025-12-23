//! # FeatureDiversityObjective - Trait Implementations
//!
//! This module contains trait implementations for `FeatureDiversityObjective`.
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

impl ObjectiveFunction for FeatureDiversityObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        _y_classif: Option<&Array1<i32>>,
        _y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();
        if selected_indices.len() < 2 {
            return Ok(1.0);
        }
        let mut total_correlation = 0.0;
        let mut pair_count = 0;
        for i in 0..selected_indices.len() {
            for j in (i + 1)..selected_indices.len() {
                let feature_i = x.column(selected_indices[i]).to_owned();
                let feature_j = x.column(selected_indices[j]).to_owned();
                let correlation = compute_pearson_correlation(&feature_i, &feature_j).abs();
                total_correlation += correlation;
                pair_count += 1;
            }
        }
        let avg_correlation = if pair_count > 0 {
            total_correlation / pair_count as f64
        } else {
            0.0
        };
        Ok(1.0 - avg_correlation)
    }
    fn name(&self) -> &str {
        "feature_diversity"
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

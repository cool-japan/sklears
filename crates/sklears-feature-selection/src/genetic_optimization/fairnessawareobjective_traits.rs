//! # FairnessAwareObjective - Trait Implementations
//!
//! This module contains trait implementations for `FairnessAwareObjective`.
//!
//! ## Implemented Traits
//!
//! - `ObjectiveFunction`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::*;
use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

impl ObjectiveFunction for FairnessAwareObjective {
    fn evaluate(
        &self,
        x: &Array2<f64>,
        y_classif: Option<&Array1<i32>>,
        y_regression: Option<&Array1<f64>>,
        feature_mask: &[bool],
    ) -> SklResult<f64> {
        let selected_indices: Vec<usize> = feature_mask
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();
        if selected_indices.is_empty() {
            return Ok(1.0);
        }
        let mut total_performance = 0.0;
        if let Some(y) = y_classif {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let y_float = y.mapv(|v| v as f64);
                let correlation = compute_pearson_correlation(&feature, &y_float).abs();
                total_performance += correlation;
            }
        } else if let Some(y) = y_regression {
            for &idx in &selected_indices {
                let feature = x.column(idx).to_owned();
                let correlation = compute_pearson_correlation(&feature, y).abs();
                total_performance += correlation;
            }
        } else {
            return Err(SklearsError::InvalidInput(
                "Need either classification or regression target".to_string(),
            ));
        }
        let avg_performance = total_performance / selected_indices.len() as f64;
        let fairness_score = match self.fairness_metric {
            FairnessMetric::DemographicParity | FairnessMetric::StatisticalParityDifference => {
                let protected_float = self.protected_groups.mapv(|v| v as f64);
                let mut total_bias = 0.0;
                for &idx in &selected_indices {
                    let feature = x.column(idx).to_owned();
                    let correlation = compute_pearson_correlation(&feature, &protected_float).abs();
                    total_bias += correlation;
                }
                let avg_bias = total_bias / selected_indices.len() as f64;
                1.0 - avg_bias
            }
            _ => {
                let protected_float = self.protected_groups.mapv(|v| v as f64);
                let mut total_bias = 0.0;
                for &idx in &selected_indices {
                    let feature = x.column(idx).to_owned();
                    let correlation = compute_pearson_correlation(&feature, &protected_float).abs();
                    total_bias += correlation;
                }
                let avg_bias = total_bias / selected_indices.len() as f64;
                1.0 - avg_bias
            }
        };
        let fairness_aware_score =
            (1.0 - self.fairness_weight) * avg_performance + self.fairness_weight * fairness_score;
        Ok(fairness_aware_score)
    }
    fn name(&self) -> &str {
        "fairness_aware"
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

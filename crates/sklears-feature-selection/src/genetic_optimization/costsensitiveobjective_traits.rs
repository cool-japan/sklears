//! # CostSensitiveObjective - Trait Implementations
//!
//! This module contains trait implementations for `CostSensitiveObjective`.
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

impl ObjectiveFunction for CostSensitiveObjective {
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
            return Ok(0.0);
        }
        let total_cost: f64 = selected_indices
            .iter()
            .map(|&idx| self.feature_costs[idx])
            .sum();
        let max_cost: f64 = self.feature_costs.sum();
        let normalized_cost = if max_cost > 0.0 {
            total_cost / max_cost
        } else {
            0.0
        };
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
        let cost_sensitive_score =
            (1.0 - self.cost_weight) * avg_performance + self.cost_weight * (1.0 - normalized_cost);
        Ok(cost_sensitive_score)
    }
    fn name(&self) -> &str {
        "cost_sensitive"
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

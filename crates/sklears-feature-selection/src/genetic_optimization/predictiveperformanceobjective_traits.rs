//! # PredictivePerformanceObjective - Trait Implementations
//!
//! This module contains trait implementations for `PredictivePerformanceObjective`.
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

impl ObjectiveFunction for PredictivePerformanceObjective {
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
        Ok(total_performance / selected_indices.len() as f64)
    }
    fn name(&self) -> &str {
        "predictive_performance"
    }
    fn clone_box(&self) -> Box<dyn ObjectiveFunction> {
        Box::new(self.clone())
    }
}

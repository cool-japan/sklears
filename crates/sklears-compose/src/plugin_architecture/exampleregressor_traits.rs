//! # ExampleRegressor - Trait Implementations
//!
//! This module contains trait implementations for `ExampleRegressor`.
//!
//! ## Implemented Traits
//!
//! - `PluginComponent`
//! - `PluginEstimator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::any::Any;

use super::functions::{PluginComponent, PluginEstimator};
use super::types::{ComponentConfig, ComponentContext, ExampleRegressor};

impl PluginComponent for ExampleRegressor {
    fn component_type(&self) -> &'static str {
        "example_regressor"
    }
    fn config(&self) -> &ComponentConfig {
        &self.config
    }
    fn initialize(&mut self, _context: &ComponentContext) -> SklResult<()> {
        Ok(())
    }
    fn clone_component(&self) -> Box<dyn PluginComponent> {
        Box::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl PluginEstimator for ExampleRegressor {
    fn fit(&mut self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<()> {
        let n_features = x.ncols();
        let mut coefficients = Array1::zeros(n_features);
        let y_f64 = y.mapv(|v| v);
        for _ in 0..1000 {
            let predictions = x.dot(&coefficients);
            let errors = &predictions - &y_f64;
            let gradient = x.t().dot(&errors) / x.nrows() as f64;
            coefficients = coefficients - self.learning_rate * gradient;
        }
        self.coefficients = Some(coefficients);
        self.fitted = true;
        Ok(())
    }
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        if !self.fitted {
            return Err(SklearsError::InvalidOperation(
                "Estimator must be fitted before predict".to_string(),
            ));
        }
        let coefficients = self.coefficients.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("Coefficients not initialized".to_string())
        })?;
        Ok(x.dot(coefficients))
    }
    fn score(&self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<f64> {
        let predictions = self.predict(x)?;
        let y_f64 = y.mapv(|v: Float| v);
        let y_mean = y_f64.mean().ok_or_else(|| {
            SklearsError::InvalidOperation("Cannot compute mean of empty array".to_string())
        })?;
        let ss_res = (&predictions - &y_f64).mapv(|x: f64| x.powi(2)).sum();
        let ss_tot = y_f64.mapv(|x: f64| (x - y_mean).powi(2)).sum();
        Ok(1.0 - ss_res / ss_tot)
    }
    fn is_fitted(&self) -> bool {
        self.fitted
    }
    fn feature_importances(&self) -> Option<Array1<f64>> {
        self.coefficients
            .as_ref()
            .map(|coefs: &Array1<f64>| coefs.mapv(f64::abs))
    }
}

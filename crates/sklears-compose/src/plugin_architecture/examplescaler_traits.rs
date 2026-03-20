//! # ExampleScaler - Trait Implementations
//!
//! This module contains trait implementations for `ExampleScaler`.
//!
//! ## Implemented Traits
//!
//! - `PluginComponent`
//! - `PluginTransformer`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::any::Any;

use super::functions::{PluginComponent, PluginTransformer};
use super::types::{ComponentConfig, ComponentContext, ExampleScaler};

impl PluginComponent for ExampleScaler {
    fn component_type(&self) -> &'static str {
        "example_scaler"
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

impl PluginTransformer for ExampleScaler {
    fn fit(
        &mut self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<()> {
        self.fitted = true;
        Ok(())
    }
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::InvalidOperation(
                "Transformer must be fitted before transform".to_string(),
            ));
        }
        Ok(x.mapv(|v| v * self.scale_factor))
    }
    fn is_fitted(&self) -> bool {
        self.fitted
    }
    fn get_feature_names_out(&self, input_features: Option<&[String]>) -> Vec<String> {
        match input_features {
            Some(features) => features.iter().map(|f| format!("scaled_{f}")).collect(),
            None => (0..10).map(|i| format!("scaled_feature_{i}")).collect(),
        }
    }
}

//! # ExampleEstimatorPlugin - Trait Implementations
//!
//! This module contains trait implementations for `ExampleEstimatorPlugin`.
//!
//! ## Implemented Traits
//!
//! - `Plugin`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};

use super::functions::{Plugin, PluginComponent};
use super::types::{
    ComponentConfig, ComponentSchema, ConfigValue, ExampleEstimatorPlugin, ExampleRegressor,
    ParameterSchema, ParameterType, PluginCapability, PluginContext, PluginMetadata,
};

impl Plugin for ExampleEstimatorPlugin {
    fn metadata(&self) -> &PluginMetadata {
        &self.metadata
    }
    fn initialize(&mut self, _context: &PluginContext) -> SklResult<()> {
        Ok(())
    }
    fn shutdown(&mut self) -> SklResult<()> {
        Ok(())
    }
    fn capabilities(&self) -> Vec<PluginCapability> {
        vec![PluginCapability::Estimator]
    }
    fn create_component(
        &self,
        component_type: &str,
        config: &ComponentConfig,
    ) -> SklResult<Box<dyn PluginComponent>> {
        match component_type {
            "example_regressor" => Ok(Box::new(ExampleRegressor::new(config.clone()))),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown component type: {component_type}"
            ))),
        }
    }
    fn validate_config(&self, _config: &ComponentConfig) -> SklResult<()> {
        Ok(())
    }
    fn get_component_schema(&self, component_type: &str) -> Option<ComponentSchema> {
        match component_type {
            "example_regressor" => Some(ComponentSchema {
                name: "ExampleRegressor".to_string(),
                required_parameters: vec![],
                optional_parameters: vec![ParameterSchema {
                    name: "learning_rate".to_string(),
                    parameter_type: ParameterType::Float {
                        min_value: Some(0.0),
                        max_value: Some(1.0),
                    },
                    description: "Learning rate for the algorithm".to_string(),
                    default_value: Some(ConfigValue::Float(0.01)),
                }],
                constraints: vec![],
            }),
            _ => None,
        }
    }
}

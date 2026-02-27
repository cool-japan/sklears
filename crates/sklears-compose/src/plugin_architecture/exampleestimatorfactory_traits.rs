//! # ExampleEstimatorFactory - Trait Implementations
//!
//! This module contains trait implementations for `ExampleEstimatorFactory`.
//!
//! ## Implemented Traits
//!
//! - `ComponentFactory`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};

use super::functions::{ComponentFactory, PluginComponent};
use super::types::{
    ComponentConfig, ComponentSchema, ConfigValue, ExampleEstimatorFactory, ExampleRegressor,
    ParameterSchema, ParameterType,
};

impl ComponentFactory for ExampleEstimatorFactory {
    fn create(
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
    fn available_types(&self) -> Vec<String> {
        vec!["example_regressor".to_string()]
    }
    fn get_schema(&self, component_type: &str) -> Option<ComponentSchema> {
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

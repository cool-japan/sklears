//! # ExampleTransformerFactory - Trait Implementations
//!
//! This module contains trait implementations for `ExampleTransformerFactory`.
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
    ComponentConfig, ComponentSchema, ConfigValue, ExampleScaler, ExampleTransformerFactory,
    ParameterSchema, ParameterType,
};

impl ComponentFactory for ExampleTransformerFactory {
    fn create(
        &self,
        component_type: &str,
        config: &ComponentConfig,
    ) -> SklResult<Box<dyn PluginComponent>> {
        match component_type {
            "example_scaler" => Ok(Box::new(ExampleScaler::new(config.clone()))),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown component type: {component_type}"
            ))),
        }
    }
    fn available_types(&self) -> Vec<String> {
        vec!["example_scaler".to_string()]
    }
    fn get_schema(&self, component_type: &str) -> Option<ComponentSchema> {
        match component_type {
            "example_scaler" => Some(ComponentSchema {
                name: "ExampleScaler".to_string(),
                required_parameters: vec![],
                optional_parameters: vec![ParameterSchema {
                    name: "scale_factor".to_string(),
                    parameter_type: ParameterType::Float {
                        min_value: Some(0.0),
                        max_value: None,
                    },
                    description: "Factor to scale features by".to_string(),
                    default_value: Some(ConfigValue::Float(1.0)),
                }],
                constraints: vec![],
            }),
            _ => None,
        }
    }
}

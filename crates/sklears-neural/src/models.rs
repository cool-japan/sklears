//! Neural network model architectures.
//!
//! This module provides high-level model APIs including Sequential and Functional models
//! for easy composition of neural network layers.

use crate::{layers::Layer, NeuralResult};
use scirs2_core::ndarray::{Array2, Axis};
use sklears_core::types::FloatBounds;
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Sequential model for linear stack of layers
pub struct Sequential<T: FloatBounds> {
    layers: Vec<Box<dyn Layer<T>>>,
    name: String,
}

impl<T: FloatBounds> Sequential<T> {
    /// Create a new empty sequential model
    pub fn new() -> Self {
        Self {
            layers: Vec::new(),
            name: "sequential".to_string(),
        }
    }

    /// Create a new sequential model with a name
    pub fn with_name<S: Into<String>>(name: S) -> Self {
        Self {
            layers: Vec::new(),
            name: name.into(),
        }
    }

    /// Add a layer to the sequential model
    pub fn add(mut self, layer: Box<dyn Layer<T>>) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add a layer to the sequential model (mutable version)
    pub fn add_layer(&mut self, layer: Box<dyn Layer<T>>) -> &mut Self {
        self.layers.push(layer);
        self
    }

    /// Get the number of layers in the model
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Check if the model is empty
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the model name
    pub fn set_name<S: Into<String>>(&mut self, name: S) {
        self.name = name.into();
    }

    /// Get total number of parameters across all layers
    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|layer| layer.num_parameters()).sum()
    }

    /// Forward pass through all layers
    pub fn forward(&mut self, input: &Array2<T>, training: bool) -> NeuralResult<Array2<T>> {
        let mut output = input.clone();

        for layer in &mut self.layers {
            output = layer.forward(&output, training)?;
        }

        Ok(output)
    }

    /// Backward pass through all layers (in reverse order)
    pub fn backward(&mut self, grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        let mut grad = grad_output.clone();

        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        Ok(grad)
    }

    /// Reset all layer states
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }

    /// Get layer at specific index
    pub fn get_layer(&self, index: usize) -> Option<&dyn Layer<T>> {
        self.layers.get(index).map(|layer| layer.as_ref())
    }

    /// Get mutable layer at specific index
    pub fn get_layer_mut(&mut self, index: usize) -> Option<&mut (dyn Layer<T> + '_)> {
        match self.layers.get_mut(index) {
            Some(layer) => Some(layer.as_mut()),
            None => None,
        }
    }
}

impl<T: FloatBounds> std::fmt::Debug for Sequential<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Sequential")
            .field("layers", &format!("{} layers", self.layers.len()))
            .field("name", &self.name)
            .finish()
    }
}

impl<T: FloatBounds> Default for Sequential<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Functional model for DAG-style networks with named layers
pub struct Functional<T: FloatBounds> {
    layers: HashMap<String, Box<dyn Layer<T>>>,
    execution_order: Vec<String>,
    connections: HashMap<String, Vec<String>>, // layer_name -> input_layer_names
    outputs: HashMap<String, Array2<T>>,       // cached layer outputs during forward pass
    name: String,
}

impl<T: FloatBounds> Functional<T> {
    /// Create a new functional model
    pub fn new() -> Self {
        Self {
            layers: HashMap::new(),
            execution_order: Vec::new(),
            connections: HashMap::new(),
            outputs: HashMap::new(),
            name: "functional".to_string(),
        }
    }

    /// Create a new functional model with a name
    pub fn with_name<S: Into<String>>(name: S) -> Self {
        Self {
            layers: HashMap::new(),
            execution_order: Vec::new(),
            connections: HashMap::new(),
            outputs: HashMap::new(),
            name: name.into(),
        }
    }

    /// Add a layer with inputs from other layers
    pub fn add_layer<S: Into<String>>(
        &mut self,
        name: S,
        layer: Box<dyn Layer<T>>,
        inputs: Vec<String>,
    ) -> NeuralResult<()> {
        let name = name.into();

        // Validate that input layers exist (except for input layer)
        if !inputs.is_empty() {
            for input_name in &inputs {
                if !self.layers.contains_key(input_name) {
                    return Err(sklears_core::error::SklearsError::InvalidParameter {
                        name: "inputs".to_string(),
                        reason: format!("Input layer '{}' not found", input_name),
                    });
                }
            }
        }

        self.layers.insert(name.clone(), layer);
        self.connections.insert(name.clone(), inputs);

        // Update execution order (simple topological sort)
        self.update_execution_order()?;

        Ok(())
    }

    /// Add an input layer (layer with no inputs)
    pub fn add_input_layer<S: Into<String>>(
        &mut self,
        name: S,
        layer: Box<dyn Layer<T>>,
    ) -> NeuralResult<()> {
        self.add_layer(name, layer, vec![])
    }

    /// Update execution order using topological sort
    fn update_execution_order(&mut self) -> NeuralResult<()> {
        let mut order = Vec::new();
        let mut visited = HashMap::new();
        let mut temp_mark = HashMap::new();

        for layer_name in self.layers.keys() {
            if !visited.contains_key(layer_name) {
                self.dfs_visit(layer_name, &mut visited, &mut temp_mark, &mut order)?;
            }
        }

        order.reverse();
        self.execution_order = order;
        Ok(())
    }

    /// Depth-first search for topological sort
    fn dfs_visit(
        &self,
        node: &str,
        visited: &mut HashMap<String, bool>,
        temp_mark: &mut HashMap<String, bool>,
        order: &mut Vec<String>,
    ) -> NeuralResult<()> {
        if temp_mark.contains_key(node) {
            return Err(sklears_core::error::SklearsError::InvalidParameter {
                name: "graph".to_string(),
                reason: "Cycle detected in layer connections".to_string(),
            });
        }

        if visited.contains_key(node) {
            return Ok(());
        }

        temp_mark.insert(node.to_string(), true);

        if let Some(inputs) = self.connections.get(node) {
            for input_node in inputs {
                self.dfs_visit(input_node, visited, temp_mark, order)?;
            }
        }

        temp_mark.remove(node);
        visited.insert(node.to_string(), true);
        order.push(node.to_string());

        Ok(())
    }

    /// Forward pass through the network
    pub fn forward(
        &mut self,
        inputs: HashMap<String, Array2<T>>,
        training: bool,
    ) -> NeuralResult<HashMap<String, Array2<T>>> {
        self.outputs.clear();

        // Set input values
        for (name, value) in inputs {
            self.outputs.insert(name, value);
        }

        // Execute layers in topological order
        for layer_name in &self.execution_order.clone() {
            let input_data = self.get_layer_input(layer_name)?;
            if let Some(layer) = self.layers.get_mut(layer_name) {
                let output = layer.forward(&input_data, training)?;
                self.outputs.insert(layer_name.clone(), output);
            }
        }

        Ok(self.outputs.clone())
    }

    /// Get input data for a layer by combining inputs from connected layers
    fn get_layer_input(&self, layer_name: &str) -> NeuralResult<Array2<T>> {
        let empty_vec = vec![];
        let inputs = self.connections.get(layer_name).unwrap_or(&empty_vec);

        if inputs.is_empty() {
            // This should be an input layer, return stored input
            self.outputs.get(layer_name).cloned().ok_or_else(|| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "inputs".to_string(),
                    reason: format!("No input data found for layer '{}'", layer_name),
                }
            })
        } else if inputs.len() == 1 {
            // Single input
            self.outputs.get(&inputs[0]).cloned().ok_or_else(|| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "inputs".to_string(),
                    reason: format!("Input layer '{}' has no output", inputs[0]),
                }
            })
        } else {
            // Multiple inputs - concatenate along feature axis
            let mut combined_inputs = Vec::new();
            for input_name in inputs {
                if let Some(input_data) = self.outputs.get(input_name) {
                    combined_inputs.push(input_data.view());
                } else {
                    return Err(sklears_core::error::SklearsError::InvalidParameter {
                        name: "inputs".to_string(),
                        reason: format!("Input layer '{}' has no output", input_name),
                    });
                }
            }

            // Concatenate along the feature axis (axis 1)
            let concatenated =
                scirs2_core::ndarray::concatenate(scirs2_core::ndarray::Axis(1), &combined_inputs)
                    .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                        name: "inputs".to_string(),
                        reason: "Failed to concatenate layer inputs".to_string(),
                    })?;

            Ok(concatenated)
        }
    }

    /// Get the model name
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Set the model name
    pub fn set_name<S: Into<String>>(&mut self, name: S) {
        self.name = name.into();
    }

    /// Get total number of parameters across all layers
    pub fn num_parameters(&self) -> usize {
        self.layers
            .values()
            .map(|layer| layer.num_parameters())
            .sum()
    }

    /// Get layer by name
    pub fn get_layer(&self, name: &str) -> Option<&dyn Layer<T>> {
        self.layers.get(name).map(|layer| layer.as_ref())
    }

    /// Get mutable layer by name
    pub fn get_layer_mut(&mut self, name: &str) -> Option<&mut (dyn Layer<T> + '_)> {
        match self.layers.get_mut(name) {
            Some(layer) => Some(layer.as_mut()),
            None => None,
        }
    }

    /// Get list of all layer names
    pub fn layer_names(&self) -> Vec<&String> {
        self.execution_order.iter().collect()
    }

    /// Reset all layer states
    pub fn reset(&mut self) {
        for layer in self.layers.values_mut() {
            layer.reset();
        }
        self.outputs.clear();
    }
}

impl<T: FloatBounds> std::fmt::Debug for Functional<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Functional")
            .field("layers", &format!("{} layers", self.layers.len()))
            .field("execution_order", &self.execution_order)
            .field("connections", &self.connections)
            .field("name", &self.name)
            .finish()
    }
}

impl<T: FloatBounds> Default for Functional<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating Sequential models with a fluent API
pub struct SequentialBuilder<T: FloatBounds> {
    model: Sequential<T>,
}

impl<T: FloatBounds> SequentialBuilder<T> {
    /// Create a new sequential model builder
    pub fn new() -> Self {
        Self {
            model: Sequential::new(),
        }
    }

    /// Create a new sequential model builder with a name
    pub fn with_name<S: Into<String>>(name: S) -> Self {
        Self {
            model: Sequential::with_name(name),
        }
    }

    /// Add a layer to the model
    pub fn add(mut self, layer: Box<dyn Layer<T>>) -> Self {
        self.model = self.model.add(layer);
        self
    }

    /// Build the final Sequential model
    pub fn build(self) -> Sequential<T> {
        self.model
    }
}

impl<T: FloatBounds> Default for SequentialBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::dropout::Dropout;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_sequential_model_creation() {
        let model: Sequential<f32> = Sequential::new();
        assert_eq!(model.len(), 0);
        assert!(model.is_empty());
        assert_eq!(model.name(), "sequential");
    }

    #[test]
    fn test_sequential_builder() {
        let dropout = Dropout::new(0.5);
        let model: Sequential<f32> = SequentialBuilder::with_name("test_model")
            .add(Box::new(dropout))
            .build();

        assert_eq!(model.len(), 1);
        assert_eq!(model.name(), "test_model");
    }

    #[test]
    fn test_functional_model_creation() {
        let model: Functional<f32> = Functional::new();
        assert_eq!(model.name(), "functional");
        assert_eq!(model.num_parameters(), 0);
    }

    #[test]
    fn test_functional_model_layer_addition() -> NeuralResult<()> {
        let mut model: Functional<f32> = Functional::new();
        let dropout = Dropout::new(0.5);

        model.add_input_layer("input", Box::new(dropout))?;
        assert_eq!(model.layer_names().len(), 1);
        assert!(model.get_layer("input").is_some());

        Ok(())
    }
}

/// Serialization support for models
#[cfg(feature = "serde")]
pub mod serialization {
    use super::*;
    use serde::{Deserialize, Serialize};

    /// Serializable representation of a layer
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum SerializableLayer {
        /// Dense/Linear layer
        Dense {
            input_size: usize,
            output_size: usize,
            weights: Vec<Vec<f64>>,
            biases: Vec<f64>,
            activation: String,
        },
        /// Dropout layer
        Dropout { rate: f64 },
        /// Batch normalization layer
        BatchNorm {
            features: usize,
            momentum: f64,
            epsilon: f64,
            affine: bool,
            gamma: Option<Vec<f64>>,
            beta: Option<Vec<f64>>,
            running_mean: Option<Vec<f64>>,
            running_var: Option<Vec<f64>>,
        },
        /// Layer normalization
        LayerNorm {
            features: usize,
            epsilon: f64,
            affine: bool,
            gamma: Option<Vec<f64>>,
            beta: Option<Vec<f64>>,
        },
        /// Conv2D layer
        Conv2D {
            in_channels: usize,
            out_channels: usize,
            kernel_size: (usize, usize),
            stride: (usize, usize),
            padding: (usize, usize),
            dilation: (usize, usize),
            weights: Vec<Vec<Vec<Vec<f64>>>>,
            biases: Option<Vec<f64>>,
        },
        /// Other layer types can be added as needed
        Other {
            layer_type: String,
            config: HashMap<String, serde_json::Value>,
        },
    }

    /// Serializable representation of a Sequential model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableSequential {
        pub name: String,
        pub layers: Vec<SerializableLayer>,
    }

    /// Serializable representation of a Functional model
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SerializableFunctional {
        pub name: String,
        pub layers: HashMap<String, SerializableLayer>,
        pub connections: HashMap<String, Vec<String>>,
    }

    /// Configuration for model saving/loading
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct ModelConfig {
        pub model_type: String,
        pub version: String,
        pub created_at: String,
        pub metadata: HashMap<String, serde_json::Value>,
    }

    /// Complete model save format
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub struct SavedModel {
        pub config: ModelConfig,
        pub model: ModelData,
    }

    /// Model data union
    #[derive(Debug, Clone, Serialize, Deserialize)]
    #[serde(tag = "type")]
    pub enum ModelData {
        Sequential(SerializableSequential),
        Functional(SerializableFunctional),
    }

    impl SavedModel {
        /// Create a new saved model with metadata
        pub fn new(model: ModelData) -> Self {
            let model_type = match &model {
                ModelData::Sequential(_) => "Sequential",
                ModelData::Functional(_) => "Functional",
            };

            let config = ModelConfig {
                model_type: model_type.to_string(),
                version: "0.1.0".to_string(),
                created_at: chrono::Utc::now().to_rfc3339(),
                metadata: HashMap::new(),
            };

            Self { config, model }
        }

        /// Add metadata to the saved model
        pub fn with_metadata(mut self, key: String, value: serde_json::Value) -> Self {
            self.config.metadata.insert(key, value);
            self
        }

        /// Save model to JSON file
        pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> NeuralResult<()> {
            let json = serde_json::to_string_pretty(self).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "serialization".to_string(),
                    reason: format!("Failed to serialize model: {}", e),
                }
            })?;

            std::fs::write(path, json).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to write model file: {}", e),
                }
            })?;

            Ok(())
        }

        /// Load model from JSON file
        pub fn load_from_file<P: AsRef<std::path::Path>>(path: P) -> NeuralResult<Self> {
            let json = std::fs::read_to_string(path).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to read model file: {}", e),
                }
            })?;

            let model: Self = serde_json::from_str(&json).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "deserialization".to_string(),
                    reason: format!("Failed to deserialize model: {}", e),
                }
            })?;

            Ok(model)
        }

        /// Save model to binary format
        pub fn save_to_binary<P: AsRef<std::path::Path>>(&self, path: P) -> NeuralResult<()> {
            let binary =
                oxicode::serde::encode_to_vec(self, oxicode::config::standard()).map_err(|e| {
                    sklears_core::error::SklearsError::InvalidParameter {
                        name: "serialization".to_string(),
                        reason: format!("Failed to serialize model to binary: {}", e),
                    }
                })?;

            std::fs::write(path, binary).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to write binary model file: {}", e),
                }
            })?;

            Ok(())
        }

        /// Load model from binary format
        pub fn load_from_binary<P: AsRef<std::path::Path>>(path: P) -> NeuralResult<Self> {
            let binary = std::fs::read(path).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to read binary model file: {}", e),
                }
            })?;

            let (model, _bytes_read): (Self, usize) =
                oxicode::serde::decode_from_slice(&binary, oxicode::config::standard()).map_err(
                    |e| sklears_core::error::SklearsError::InvalidParameter {
                        name: "deserialization".to_string(),
                        reason: format!("Failed to deserialize binary model: {}", e),
                    },
                )?;

            Ok(model)
        }
    }

    /// Trait for converting models to/from serializable format
    pub trait ModelSerialization<T: FloatBounds> {
        type SerializableType;

        /// Convert to serializable format
        fn to_serializable(&self) -> NeuralResult<Self::SerializableType>;

        /// Convert from serializable format
        fn from_serializable(data: &Self::SerializableType) -> NeuralResult<Self>
        where
            Self: Sized;

        /// Save model to file
        fn save<P: AsRef<std::path::Path>>(&self, path: P) -> NeuralResult<()>
        where
            Self::SerializableType: Serialize,
        {
            let serializable = self.to_serializable()?;
            let json = serde_json::to_string_pretty(&serializable).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "serialization".to_string(),
                    reason: format!("Failed to serialize: {}", e),
                }
            })?;

            std::fs::write(path, json).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to write file: {}", e),
                }
            })?;

            Ok(())
        }

        /// Load model from file
        fn load<P: AsRef<std::path::Path>>(path: P) -> NeuralResult<Self>
        where
            Self: Sized,
            Self::SerializableType: for<'de> Deserialize<'de>,
        {
            let json = std::fs::read_to_string(path).map_err(|e| {
                sklears_core::error::SklearsError::InvalidParameter {
                    name: "file_io".to_string(),
                    reason: format!("Failed to read file: {}", e),
                }
            })?;

            let serializable: Self::SerializableType =
                serde_json::from_str(&json).map_err(|e| {
                    sklears_core::error::SklearsError::InvalidParameter {
                        name: "deserialization".to_string(),
                        reason: format!("Failed to deserialize: {}", e),
                    }
                })?;

            Self::from_serializable(&serializable)
        }
    }

    // Note: Actual implementations of ModelSerialization for Sequential and Functional
    // would require extensive layer-specific serialization logic which would be
    // implemented alongside each layer type.
}

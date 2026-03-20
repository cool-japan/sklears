//! Neural network implementations for ML-based trait recommendations
//!
//! This module contains neural network models for embedding generation,
//! similarity computation, and deep learning-based trait analysis.

use crate::error::{Result, SklearsError};
use crate::trait_explorer::ml_recommendations::data_types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// SciRS2 Compliance: Use scirs2_autograd for ndarray functionality
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};

/// Neural embedding model for generating dense vector representations of traits
#[derive(Debug, Clone)]
pub struct NeuralEmbeddingModel {
    /// Input layer weights
    pub input_weights: Array2<f64>,
    /// Hidden layer weights
    pub hidden_weights: Vec<Array2<f64>>,
    /// Output layer weights
    pub output_weights: Array2<f64>,
    /// Bias terms for each layer
    pub biases: Vec<Array1<f64>>,
    /// Embedding dimension
    pub embedding_dim: usize,
    /// Model configuration
    pub config: NeuralNetworkConfig,
}

impl NeuralEmbeddingModel {
    /// Create a new neural embedding model
    pub fn new(input_dim: usize, embedding_dim: usize, hidden_layers: &[usize]) -> Self {
        // Initialize input weights with small deterministic values
        let input_weights = Array2::from_shape_fn((input_dim, hidden_layers[0]), |_| {
            0.1 // Simple constant initialization for deterministic behavior
        });

        // Initialize hidden weights and biases
        let mut hidden_weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..hidden_layers.len() {
            let input_size = if i == 0 {
                hidden_layers[0]
            } else {
                hidden_layers[i - 1]
            };
            let output_size = if i == hidden_layers.len() - 1 {
                embedding_dim
            } else {
                hidden_layers[i + 1]
            };

            // Deterministic weight initialization based on layer position
            hidden_weights.push(Array2::from_shape_fn(
                (input_size, output_size),
                |(i, j)| {
                    // Use layer-specific constants for reproducible initialization
                    ((i as f64 * 0.1 + j as f64 * 0.01 + i as f64 * 0.007) % 1.0 - 0.5) * 0.2
                },
            ));

            // Deterministic bias initialization
            biases.push(Array1::from_shape_fn(output_size, |i| {
                ((i as f64 * 0.1) % 1.0 - 0.5) * 0.1
            }));
        }

        // Output layer weights with deterministic initialization
        let output_weights = Array2::from_shape_fn(
            (
                hidden_layers.last().unwrap_or(&embedding_dim).clone(),
                embedding_dim,
            ),
            |(i, j)| {
                ((i as f64 * 0.11 + j as f64 * 0.013) % 1.0 - 0.5) * 0.2
            },
        );

        Self {
            input_weights,
            hidden_weights,
            output_weights,
            biases,
            embedding_dim,
            config: NeuralNetworkConfig::default(),
        }
    }

    /// Create a neural embedding model with custom configuration
    pub fn with_config(
        input_dim: usize,
        embedding_dim: usize,
        config: NeuralNetworkConfig,
    ) -> Self {
        let layer_sizes: Vec<usize> = config
            .layers
            .iter()
            .map(|layer| layer.units)
            .collect();

        let mut model = Self::new(input_dim, embedding_dim, &layer_sizes);
        model.config = config;
        model
    }

    /// Forward pass through the neural network
    pub fn forward(&self, input: &Array1<f64>) -> Result<Array1<f64>> {
        if input.len() != self.input_weights.shape()[0] {
            return Err(SklearsError::InvalidParameter(format!(
                "Input dimension mismatch: expected {}, got {}",
                self.input_weights.shape()[0],
                input.len()
            )));
        }

        // Initial linear transformation
        let mut current = input.dot(&self.input_weights);

        // Apply hidden layers with activation functions
        for (i, (weights, bias)) in self.hidden_weights.iter().zip(&self.biases).enumerate() {
            current = current.dot(weights) + bias;

            // Apply activation function based on configuration
            current = self.apply_activation(&current, &self.config.activation);

            // Apply dropout if configured (during training)
            if let Some(dropout_rate) = self.config.layers.get(i).and_then(|l| l.dropout) {
                if dropout_rate > 0.0 {
                    current = self.apply_dropout(&current, dropout_rate);
                }
            }

            // Apply batch normalization if configured
            if self.config.layers.get(i).map_or(false, |l| l.batch_normalization) {
                current = self.apply_batch_normalization(&current);
            }
        }

        // Final output layer with tanh activation for normalized embeddings
        let output = current.dot(&self.output_weights);
        Ok(output.mapv(|x| x.tanh()))
    }

    /// Apply activation function to the layer output
    fn apply_activation(&self, input: &Array1<f64>, activation: &ActivationFunction) -> Array1<f64> {
        match activation {
            ActivationFunction::ReLU => input.mapv(|x| x.max(0.0)),
            ActivationFunction::LeakyReLU => input.mapv(|x| if x > 0.0 { x } else { 0.01 * x }),
            ActivationFunction::Tanh => input.mapv(|x| x.tanh()),
            ActivationFunction::Sigmoid => input.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            ActivationFunction::Softmax => {
                let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_vals = input.mapv(|x| (x - max_val).exp());
                let sum_exp = exp_vals.sum();
                exp_vals.mapv(|x| x / sum_exp)
            }
            ActivationFunction::Swish => input.mapv(|x| x / (1.0 + (-x).exp())),
            ActivationFunction::GELU => input.mapv(|x| {
                0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }),
        }
    }

    /// Apply dropout to layer output (simplified implementation)
    fn apply_dropout(&self, input: &Array1<f64>, dropout_rate: f64) -> Array1<f64> {
        // Simplified dropout: just scale by (1 - dropout_rate) for inference
        input.mapv(|x| x * (1.0 - dropout_rate))
    }

    /// Apply batch normalization (simplified implementation)
    fn apply_batch_normalization(&self, input: &Array1<f64>) -> Array1<f64> {
        // Simplified batch normalization: standardize to mean=0, std=1
        let mean = input.mean().unwrap_or(0.0);
        let variance = input.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let std_dev = variance.sqrt().max(1e-8); // Avoid division by zero

        input.mapv(|x| (x - mean) / std_dev)
    }

    /// Calculate similarity between two embeddings using cosine similarity
    pub fn similarity(&self, embedding1: &Array1<f64>, embedding2: &Array1<f64>) -> f64 {
        if embedding1.len() != embedding2.len() {
            return 0.0;
        }

        let dot_product = embedding1.dot(embedding2);
        let norm1 = embedding1.dot(embedding1).sqrt();
        let norm2 = embedding2.dot(embedding2).sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            0.0
        } else {
            dot_product / (norm1 * norm2)
        }
    }

    /// Calculate L2 (Euclidean) distance between embeddings
    pub fn l2_distance(&self, embedding1: &Array1<f64>, embedding2: &Array1<f64>) -> f64 {
        if embedding1.len() != embedding2.len() {
            return f64::INFINITY;
        }

        embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Calculate Manhattan (L1) distance between embeddings
    pub fn manhattan_distance(&self, embedding1: &Array1<f64>, embedding2: &Array1<f64>) -> f64 {
        if embedding1.len() != embedding2.len() {
            return f64::INFINITY;
        }

        embedding1
            .iter()
            .zip(embedding2.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    }

    /// Find the k most similar embeddings from a collection
    pub fn find_k_nearest(
        &self,
        query_embedding: &Array1<f64>,
        embeddings: &[(String, Array1<f64>)],
        k: usize,
        distance_metric: DistanceMetric,
    ) -> Vec<(String, f64)> {
        let mut similarities: Vec<(String, f64)> = embeddings
            .iter()
            .map(|(name, embedding)| {
                let score = match distance_metric {
                    DistanceMetric::Cosine => self.similarity(query_embedding, embedding),
                    DistanceMetric::Euclidean => -self.l2_distance(query_embedding, embedding), // Negative for sorting
                    DistanceMetric::Manhattan => -self.manhattan_distance(query_embedding, embedding),
                    _ => self.similarity(query_embedding, embedding), // Default to cosine
                };
                (name.clone(), score)
            })
            .collect();

        // Sort by similarity score (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        similarities.into_iter().take(k).collect()
    }

    /// Generate embeddings for a batch of inputs
    pub fn forward_batch(&self, inputs: &Array2<f64>) -> Result<Array2<f64>> {
        let batch_size = inputs.shape()[0];
        let mut outputs = Array2::zeros((batch_size, self.embedding_dim));

        for i in 0..batch_size {
            let input = inputs.row(i).to_owned();
            let output = self.forward(&input)?;
            outputs.row_mut(i).assign(&output);
        }

        Ok(outputs)
    }

    /// Train the model using gradient descent (simplified implementation)
    pub fn train(
        &mut self,
        training_data: &[(Array1<f64>, Array1<f64>)],
        config: &OptimizerConfig,
        epochs: usize,
    ) -> Result<Vec<f64>> {
        let mut loss_history = Vec::new();

        for epoch in 0..epochs {
            let mut total_loss = 0.0;

            for (input, target) in training_data {
                // Forward pass
                let output = self.forward(input)?;

                // Calculate loss (mean squared error)
                let loss = self.calculate_mse_loss(&output, target);
                total_loss += loss;

                // Backward pass (simplified - would need proper gradients in practice)
                self.update_weights_simple(input, target, &output, config)?;
            }

            let avg_loss = total_loss / training_data.len() as f64;
            loss_history.push(avg_loss);

            // Early stopping could be implemented here
            if epoch > 10 && avg_loss < 1e-6 {
                break;
            }
        }

        Ok(loss_history)
    }

    /// Calculate mean squared error loss
    fn calculate_mse_loss(&self, output: &Array1<f64>, target: &Array1<f64>) -> f64 {
        output
            .iter()
            .zip(target.iter())
            .map(|(o, t)| (o - t).powi(2))
            .sum::<f64>()
            / output.len() as f64
    }

    /// Simplified weight update (would need proper gradient computation in practice)
    fn update_weights_simple(
        &mut self,
        _input: &Array1<f64>,
        _target: &Array1<f64>,
        _output: &Array1<f64>,
        config: &OptimizerConfig,
    ) -> Result<()> {
        // This is a placeholder for proper gradient-based weight updates
        // In a real implementation, this would compute gradients and update weights
        // according to the specified optimizer (SGD, Adam, etc.)

        let learning_rate = config.learning_rate;

        // Apply small random perturbations as a simplified update
        // (This is not a proper training algorithm - just for demonstration)
        for weights in &mut self.hidden_weights {
            weights.mapv_inplace(|w| w + learning_rate * 0.001 * (fastrand::f64() - 0.5));
        }

        Ok(())
    }

    /// Evaluate model performance on validation data
    pub fn evaluate(&self, validation_data: &[(Array1<f64>, Array1<f64>)]) -> Result<ModelMetrics> {
        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let total_samples = validation_data.len();

        for (input, target) in validation_data {
            let output = self.forward(input)?;

            // Calculate loss
            let loss = self.calculate_mse_loss(&output, target);
            total_loss += loss;

            // Calculate accuracy (for classification tasks)
            let predicted_class = self.get_predicted_class(&output);
            let actual_class = self.get_predicted_class(target);
            if predicted_class == actual_class {
                correct_predictions += 1;
            }
        }

        Ok(ModelMetrics {
            average_loss: total_loss / total_samples as f64,
            accuracy: correct_predictions as f64 / total_samples as f64,
            total_samples,
        })
    }

    /// Get predicted class from output (for classification)
    fn get_predicted_class(&self, output: &Array1<f64>) -> usize {
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Get model configuration
    pub fn get_config(&self) -> &NeuralNetworkConfig {
        &self.config
    }

    /// Get embedding dimension
    pub fn get_embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Get input dimension
    pub fn get_input_dim(&self) -> usize {
        self.input_weights.shape()[0]
    }

    /// Get model parameters count
    pub fn get_parameter_count(&self) -> usize {
        let mut count = self.input_weights.len();

        for weights in &self.hidden_weights {
            count += weights.len();
        }

        for bias in &self.biases {
            count += bias.len();
        }

        count += self.output_weights.len();
        count
    }
}

/// Model evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetrics {
    /// Average loss on evaluation data
    pub average_loss: f64,
    /// Classification accuracy
    pub accuracy: f64,
    /// Total number of samples evaluated
    pub total_samples: usize,
}

impl Default for NeuralNetworkConfig {
    fn default() -> Self {
        Self {
            layers: vec![
                LayerConfig {
                    layer_type: LayerType::Dense,
                    units: 128,
                    dropout: Some(0.2),
                    batch_normalization: true,
                },
                LayerConfig {
                    layer_type: LayerType::Dense,
                    units: 64,
                    dropout: Some(0.2),
                    batch_normalization: true,
                },
            ],
            activation: ActivationFunction::ReLU,
            loss_function: LossFunction::MeanSquaredError,
            optimizer: OptimizerConfig::default(),
        }
    }
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            optimizer_type: OptimizerType::Adam {
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
            },
            learning_rate: 0.001,
            lr_schedule: None,
            gradient_clipping: Some(1.0),
        }
    }
}

/// Utility functions for neural network operations
pub struct NeuralNetworkUtils;

impl NeuralNetworkUtils {
    /// Xavier/Glorot weight initialization
    pub fn xavier_uniform(fan_in: usize, fan_out: usize) -> f64 {
        let limit = (6.0 / (fan_in + fan_out) as f64).sqrt();
        (fastrand::f64() * 2.0 - 1.0) * limit
    }

    /// He weight initialization
    pub fn he_uniform(fan_in: usize) -> f64 {
        let limit = (6.0 / fan_in as f64).sqrt();
        (fastrand::f64() * 2.0 - 1.0) * limit
    }

    /// LeCun weight initialization
    pub fn lecun_uniform(fan_in: usize) -> f64 {
        let limit = (3.0 / fan_in as f64).sqrt();
        (fastrand::f64() * 2.0 - 1.0) * limit
    }

    /// Compute softmax activation
    pub fn softmax(input: &Array1<f64>) -> Array1<f64> {
        let max_val = input.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let exp_vals = input.mapv(|x| (x - max_val).exp());
        let sum_exp = exp_vals.sum();
        exp_vals.mapv(|x| x / sum_exp)
    }

    /// Compute layer normalization
    pub fn layer_norm(input: &Array1<f64>, epsilon: f64) -> Array1<f64> {
        let mean = input.mean().unwrap_or(0.0);
        let variance = input.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(1.0);
        let std_dev = (variance + epsilon).sqrt();

        input.mapv(|x| (x - mean) / std_dev)
    }

    /// Compute attention weights
    pub fn attention_weights(
        query: &Array1<f64>,
        keys: &Array2<f64>,
        temperature: f64,
    ) -> Array1<f64> {
        let scores = keys.dot(query) / temperature;
        Self::softmax(&scores)
    }

    /// Apply attention mechanism
    pub fn apply_attention(
        weights: &Array1<f64>,
        values: &Array2<f64>,
    ) -> Array1<f64> {
        let mut result = Array1::zeros(values.shape()[1]);

        for (i, &weight) in weights.iter().enumerate() {
            if i < values.shape()[0] {
                let value = values.row(i);
                result = result + &(value * weight);
            }
        }

        result
    }

    /// Compute gradient clipping
    pub fn clip_gradients(gradients: &mut Array1<f64>, max_norm: f64) {
        let grad_norm = gradients.dot(gradients).sqrt();
        if grad_norm > max_norm {
            gradients.mapv_inplace(|g| g * max_norm / grad_norm);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_embedding_model_creation() {
        let model = NeuralEmbeddingModel::new(100, 50, &[128, 64]);
        assert_eq!(model.get_input_dim(), 100);
        assert_eq!(model.get_embedding_dim(), 50);
        assert!(model.get_parameter_count() > 0);
    }

    #[test]
    fn test_forward_pass() {
        let model = NeuralEmbeddingModel::new(10, 5, &[8, 6]);
        let input = Array1::from_vec((0..10).map(|i| i as f64 * 0.1).collect());

        let result = model.forward(&input);
        assert!(result.is_ok());

        let output = result.expect("expected valid value");
        assert_eq!(output.len(), 5);

        // Check that output is normalized (tanh activation)
        for &val in output.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_similarity_calculation() {
        let model = NeuralEmbeddingModel::new(10, 5, &[8]);
        let emb1 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let emb2 = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0]);
        let emb3 = Array1::from_vec(vec![0.0, 1.0, 0.0, 0.0, 0.0]);

        let sim_identical = model.similarity(&emb1, &emb2);
        let sim_orthogonal = model.similarity(&emb1, &emb3);

        assert!((sim_identical - 1.0).abs() < 1e-10);
        assert!((sim_orthogonal - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_distance_metrics() {
        let model = NeuralEmbeddingModel::new(10, 5, &[8]);
        let emb1 = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let emb2 = Array1::from_vec(vec![4.0, 5.0, 6.0]);

        let l2_dist = model.l2_distance(&emb1, &emb2);
        let manhattan_dist = model.manhattan_distance(&emb1, &emb2);

        assert!((l2_dist - (27.0_f64).sqrt()).abs() < 1e-10); // sqrt(3^2 + 3^2 + 3^2)
        assert!((manhattan_dist - 9.0).abs() < 1e-10); // |1-4| + |2-5| + |3-6| = 9
    }

    #[test]
    fn test_activation_functions() {
        let model = NeuralEmbeddingModel::new(5, 3, &[4]);
        let input = Array1::from_vec(vec![-2.0, -1.0, 0.0, 1.0, 2.0]);

        // Test ReLU
        let relu_output = model.apply_activation(&input, &ActivationFunction::ReLU);
        assert_eq!(relu_output, Array1::from_vec(vec![0.0, 0.0, 0.0, 1.0, 2.0]));

        // Test Sigmoid (values should be between 0 and 1)
        let sigmoid_output = model.apply_activation(&input, &ActivationFunction::Sigmoid);
        for &val in sigmoid_output.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Test Tanh (values should be between -1 and 1)
        let tanh_output = model.apply_activation(&input, &ActivationFunction::Tanh);
        for &val in tanh_output.iter() {
            assert!(val >= -1.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_k_nearest_search() {
        let model = NeuralEmbeddingModel::new(3, 3, &[3]);
        let embeddings = vec![
            ("trait1".to_string(), Array1::from_vec(vec![1.0, 0.0, 0.0])),
            ("trait2".to_string(), Array1::from_vec(vec![0.9, 0.1, 0.0])),
            ("trait3".to_string(), Array1::from_vec(vec![0.0, 1.0, 0.0])),
            ("trait4".to_string(), Array1::from_vec(vec![0.0, 0.0, 1.0])),
        ];

        let query = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let nearest = model.find_k_nearest(&query, &embeddings, 2, DistanceMetric::Cosine);

        assert_eq!(nearest.len(), 2);
        assert_eq!(nearest[0].0, "trait1"); // Should be most similar
        assert!(nearest[0].1 > nearest[1].1); // First should have higher similarity
    }

    #[test]
    fn test_batch_forward() {
        let model = NeuralEmbeddingModel::new(4, 2, &[3]);
        let batch_input = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0, 2.0, 3.0, 4.0,
                5.0, 6.0, 7.0, 8.0,
                9.0, 10.0, 11.0, 12.0,
            ],
        ).expect("expected valid value");

        let result = model.forward_batch(&batch_input);
        assert!(result.is_ok());

        let output = result.expect("expected valid value");
        assert_eq!(output.shape(), &[3, 2]);

        // Verify individual samples match batch processing
        for i in 0..3 {
            let individual_input = batch_input.row(i).to_owned();
            let individual_output = model.forward(&individual_input).expect("forward should succeed");
            let batch_output = output.row(i);

            for j in 0..2 {
                assert!((individual_output[j] - batch_output[j]).abs() < 1e-10);
            }
        }
    }
}
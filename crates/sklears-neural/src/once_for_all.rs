//! Once-for-All Networks: Train One Network and Specialize it for Efficient Deployment
//!
//! This module implements the Once-for-All (OFA) approach to neural architecture search
//! which trains a single large network (supernet) that supports diverse architectural
//! configurations. Smaller specialized networks can be extracted from the supernet
//! without retraining, enabling efficient deployment across different hardware platforms.
//!
//! # Key Features
//!
//! - **Elastic Depth**: Variable number of layers
//! - **Elastic Width**: Variable number of channels
//! - **Elastic Kernel Size**: Variable convolution kernel sizes
//! - **Progressive Shrinking**: Training strategy that gradually expands the search space
//! - **Subnet Sampling**: Extract specialized networks for specific constraints
//! - **Knowledge Distillation**: Transfer knowledge from larger to smaller subnets
//!
//! # References
//!
//! Cai, H., Gan, C., Wang, T., Zhang, Z., & Han, S. (2020).
//! "Once for All: Train One Network and Specialize it for Efficient Deployment"
//! ICLR 2020.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::collections::HashMap;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Elastic configuration for Once-for-All networks
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ElasticConfig {
    /// Supported depths (number of layers)
    pub depths: Vec<usize>,
    /// Supported widths (number of channels)
    pub widths: Vec<usize>,
    /// Supported kernel sizes
    pub kernel_sizes: Vec<usize>,
    /// Expansion ratios for inverted residual blocks
    pub expansion_ratios: Vec<f32>,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            depths: vec![2, 3, 4],
            widths: vec![4, 5, 6],
            kernel_sizes: vec![3, 5, 7],
            expansion_ratios: vec![3.0, 4.0, 6.0],
        }
    }
}

impl ElasticConfig {
    /// Create a new elastic configuration
    pub fn new(
        depths: Vec<usize>,
        widths: Vec<usize>,
        kernel_sizes: Vec<usize>,
        expansion_ratios: Vec<f32>,
    ) -> Self {
        Self {
            depths,
            widths,
            kernel_sizes,
            expansion_ratios,
        }
    }

    /// Get maximum depth
    pub fn max_depth(&self) -> usize {
        *self.depths.iter().max().unwrap_or(&4)
    }

    /// Get maximum width
    pub fn max_width(&self) -> usize {
        *self.widths.iter().max().unwrap_or(&6)
    }

    /// Get maximum kernel size
    pub fn max_kernel_size(&self) -> usize {
        *self.kernel_sizes.iter().max().unwrap_or(&7)
    }

    /// Get maximum expansion ratio
    pub fn max_expansion_ratio(&self) -> f32 {
        *self
            .expansion_ratios
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&6.0)
    }
}

/// Subnet configuration sampled from the supernet
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SubnetConfig {
    /// Depth (number of layers)
    pub depth: usize,
    /// Width (number of channels)
    pub width: usize,
    /// Kernel size
    pub kernel_size: usize,
    /// Expansion ratio
    pub expansion_ratio: f32,
    /// Resolution (input image size)
    pub resolution: usize,
}

impl SubnetConfig {
    /// Create a new subnet configuration
    pub fn new(
        depth: usize,
        width: usize,
        kernel_size: usize,
        expansion_ratio: f32,
        resolution: usize,
    ) -> Self {
        Self {
            depth,
            width,
            kernel_size,
            expansion_ratio,
            resolution,
        }
    }

    /// Sample a random subnet configuration
    pub fn sample_random(config: &ElasticConfig, resolutions: &[usize]) -> Self {
        let mut rng = thread_rng();
        Self {
            depth: *config
                .depths
                .get(rng.gen_range(0..config.depths.len()))
                .unwrap(),
            width: *config
                .widths
                .get(rng.gen_range(0..config.widths.len()))
                .unwrap(),
            kernel_size: *config
                .kernel_sizes
                .get(rng.gen_range(0..config.kernel_sizes.len()))
                .unwrap(),
            expansion_ratio: *config
                .expansion_ratios
                .get(rng.gen_range(0..config.expansion_ratios.len()))
                .unwrap(),
            resolution: *resolutions
                .get(rng.gen_range(0..resolutions.len()))
                .unwrap(),
        }
    }

    /// Estimate FLOPs for this configuration
    pub fn estimate_flops(&self) -> u64 {
        // Simplified FLOPs estimation for a basic architecture
        let conv_flops = (self.kernel_size
            * self.kernel_size
            * self.width
            * self.width
            * self.resolution
            * self.resolution) as u64;
        conv_flops * self.depth as u64
    }

    /// Estimate parameter count
    pub fn estimate_params(&self) -> u64 {
        // Simplified parameter count
        (self.kernel_size * self.kernel_size * self.width * self.width * self.depth) as u64
    }

    /// Estimate latency (in milliseconds, simplified model)
    pub fn estimate_latency_ms(&self, hardware_type: &HardwareType) -> f64 {
        let base_latency = match hardware_type {
            HardwareType::CPU => 10.0,
            HardwareType::GPU => 2.0,
            HardwareType::Mobile => 50.0,
            HardwareType::Edge => 100.0,
        };

        // Scale by complexity
        let complexity_factor = (self.estimate_flops() as f64) / 1e9;
        base_latency * complexity_factor
    }
}

/// Hardware platform types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum HardwareType {
    /// CPU deployment
    CPU,
    /// GPU deployment
    GPU,
    /// Mobile device deployment
    Mobile,
    /// Edge device deployment
    Edge,
}

/// Training phase for progressive shrinking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TrainingPhase {
    /// Train largest network only
    LargestOnly,
    /// Train largest network + elastic kernel
    ElasticKernel,
    /// Train largest network + elastic kernel + elastic depth
    ElasticDepth,
    /// Train all elastic dimensions (full OFA)
    FullElastic,
}

/// Once-for-All supernet
#[derive(Debug)]
pub struct OnceForAllNetwork<T: FloatBounds> {
    /// Elastic configuration
    config: ElasticConfig,
    /// Supported resolutions
    resolutions: Vec<usize>,
    /// Weights for the supernet (largest configuration)
    weights: Vec<Array4<T>>,
    /// Batch normalization parameters
    bn_weights: Vec<(Array1<T>, Array1<T>)>, // (gamma, beta) for each layer
    /// Current training phase
    training_phase: TrainingPhase,
    /// Accuracy predictor (maps subnet config to accuracy)
    accuracy_predictor: Option<AccuracyPredictor<T>>,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> OnceForAllNetwork<T> {
    /// Create a new Once-for-All network
    pub fn new(config: ElasticConfig, resolutions: Vec<usize>, in_channels: usize) -> Self {
        let mut rng = thread_rng();
        let max_depth = config.max_depth();
        let max_width = config.max_width();
        let max_kernel = config.max_kernel_size();

        // Initialize weights for maximum configuration
        let mut weights = Vec::new();
        let mut bn_weights = Vec::new();

        for _layer in 0..max_depth {
            // Conv weights: (out_channels, in_channels, kernel_h, kernel_w)
            let w_shape = (max_width, max_width, max_kernel, max_kernel);
            let w_size = max_width * max_width * max_kernel * max_kernel;
            let mut w = Array4::zeros(w_shape);

            // Xavier initialization
            let std_dev = (2.0 / (max_width + max_width) as f64).sqrt();
            for elem in w.iter_mut() {
                *elem = T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero());
            }
            weights.push(w);

            // BN parameters
            let gamma = Array1::from_elem(max_width, T::one());
            let beta = Array1::zeros(max_width);
            bn_weights.push((gamma, beta));
        }

        Self {
            config,
            resolutions,
            weights,
            bn_weights,
            training_phase: TrainingPhase::LargestOnly,
            accuracy_predictor: None,
        }
    }

    /// Set training phase for progressive shrinking
    pub fn set_training_phase(&mut self, phase: TrainingPhase) {
        self.training_phase = phase;
    }

    /// Get current training phase
    pub fn training_phase(&self) -> TrainingPhase {
        self.training_phase
    }

    /// Sample a subnet according to current training constraints
    pub fn sample_subnet(&self) -> SubnetConfig {
        let mut rng = thread_rng();

        let (depth, width, kernel_size) = match self.training_phase {
            TrainingPhase::LargestOnly => (
                self.config.max_depth(),
                self.config.max_width(),
                self.config.max_kernel_size(),
            ),
            TrainingPhase::ElasticKernel => {
                let ks = *self
                    .config
                    .kernel_sizes
                    .get(rng.gen_range(0..self.config.kernel_sizes.len()))
                    .unwrap();
                (self.config.max_depth(), self.config.max_width(), ks)
            }
            TrainingPhase::ElasticDepth => {
                let d = *self
                    .config
                    .depths
                    .get(rng.gen_range(0..self.config.depths.len()))
                    .unwrap();
                let ks = *self
                    .config
                    .kernel_sizes
                    .get(rng.gen_range(0..self.config.kernel_sizes.len()))
                    .unwrap();
                (d, self.config.max_width(), ks)
            }
            TrainingPhase::FullElastic => {
                let d = *self
                    .config
                    .depths
                    .get(rng.gen_range(0..self.config.depths.len()))
                    .unwrap();
                let w = *self
                    .config
                    .widths
                    .get(rng.gen_range(0..self.config.widths.len()))
                    .unwrap();
                let ks = *self
                    .config
                    .kernel_sizes
                    .get(rng.gen_range(0..self.config.kernel_sizes.len()))
                    .unwrap();
                (d, w, ks)
            }
        };

        let resolution = *self
            .resolutions
            .get(rng.gen_range(0..self.resolutions.len()))
            .unwrap();
        let expansion_ratio = *self
            .config
            .expansion_ratios
            .get(rng.gen_range(0..self.config.expansion_ratios.len()))
            .unwrap();

        SubnetConfig::new(depth, width, kernel_size, expansion_ratio, resolution)
    }

    /// Extract subnet weights for a specific configuration
    pub fn extract_subnet_weights(&self, subnet: &SubnetConfig) -> Vec<Array4<T>> {
        let mut subnet_weights = Vec::new();

        for layer_idx in 0..subnet.depth {
            if layer_idx < self.weights.len() {
                let full_weight = &self.weights[layer_idx];

                // Extract sub-kernel and sub-channels
                let kernel_offset = (self.config.max_kernel_size() - subnet.kernel_size) / 2;
                let sub_weight = full_weight
                    .slice(s![
                        ..subnet.width,
                        ..subnet.width,
                        kernel_offset..(kernel_offset + subnet.kernel_size),
                        kernel_offset..(kernel_offset + subnet.kernel_size)
                    ])
                    .to_owned();

                subnet_weights.push(sub_weight);
            }
        }

        subnet_weights
    }

    /// Train accuracy predictor for efficient search
    pub fn train_accuracy_predictor(
        &mut self,
        samples: &[(SubnetConfig, f64)],
    ) -> NeuralResult<()> {
        let mut predictor = AccuracyPredictor::new(5); // 5 input features

        // Prepare training data
        let mut features_vec = Vec::new();
        let mut targets_vec = Vec::new();

        for (config, accuracy) in samples {
            features_vec.push(Array1::from(vec![
                T::from(config.depth as f64 / self.config.max_depth() as f64).unwrap(),
                T::from(config.width as f64 / self.config.max_width() as f64).unwrap(),
                T::from(config.kernel_size as f64 / self.config.max_kernel_size() as f64).unwrap(),
                T::from(config.expansion_ratio as f64 / self.config.max_expansion_ratio() as f64)
                    .unwrap(),
                T::from(config.resolution as f64 / 224.0).unwrap(),
            ]));
            targets_vec.push(T::from(*accuracy).unwrap());
        }

        predictor.train(&features_vec, &targets_vec)?;
        self.accuracy_predictor = Some(predictor);

        Ok(())
    }

    /// Predict accuracy for a subnet configuration
    pub fn predict_accuracy(&self, subnet: &SubnetConfig) -> NeuralResult<f64> {
        if let Some(predictor) = &self.accuracy_predictor {
            let features = Array1::from(vec![
                T::from(subnet.depth as f64 / self.config.max_depth() as f64).unwrap(),
                T::from(subnet.width as f64 / self.config.max_width() as f64).unwrap(),
                T::from(subnet.kernel_size as f64 / self.config.max_kernel_size() as f64).unwrap(),
                T::from(subnet.expansion_ratio as f64 / self.config.max_expansion_ratio() as f64)
                    .unwrap(),
                T::from(subnet.resolution as f64 / 224.0).unwrap(),
            ]);
            predictor.predict(&features)
        } else {
            Err(SklearsError::InvalidInput(
                "Accuracy predictor not trained".to_string(),
            ))
        }
    }

    /// Search for optimal subnet under constraints
    pub fn search_subnet(
        &self,
        max_flops: u64,
        max_params: u64,
        max_latency_ms: f64,
        hardware: HardwareType,
        n_samples: usize,
    ) -> NeuralResult<SubnetConfig> {
        let mut best_subnet = None;
        let mut best_accuracy = 0.0;

        for _ in 0..n_samples {
            let subnet = SubnetConfig::sample_random(&self.config, &self.resolutions);

            // Check constraints
            if subnet.estimate_flops() > max_flops {
                continue;
            }
            if subnet.estimate_params() > max_params {
                continue;
            }
            if subnet.estimate_latency_ms(&hardware) > max_latency_ms {
                continue;
            }

            // Predict accuracy
            if let Ok(accuracy) = self.predict_accuracy(&subnet) {
                if accuracy > best_accuracy {
                    best_accuracy = accuracy;
                    best_subnet = Some(subnet);
                }
            }
        }

        best_subnet.ok_or_else(|| {
            SklearsError::InvalidInput("No subnet found satisfying constraints".to_string())
        })
    }
}

/// Simple accuracy predictor using a multi-layer perceptron
#[derive(Debug)]
pub struct AccuracyPredictor<T: FloatBounds> {
    /// Input dimension
    input_dim: usize,
    /// Hidden layer weights
    w1: Array2<T>,
    /// Hidden layer bias
    b1: Array1<T>,
    /// Output layer weights
    w2: Array1<T>,
    /// Output layer bias
    b2: T,
    /// Hidden dimension
    hidden_dim: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> AccuracyPredictor<T> {
    /// Create a new accuracy predictor
    pub fn new(input_dim: usize) -> Self {
        let hidden_dim = 64;
        let mut rng = thread_rng();

        // Xavier initialization
        let std_w1 = (2.0 / (input_dim + hidden_dim) as f64).sqrt();
        let w1 = Array2::from_shape_fn((hidden_dim, input_dim), |_| {
            T::from(rng.gen_range(-std_w1..std_w1)).unwrap_or(T::zero())
        });
        let b1 = Array1::zeros(hidden_dim);

        let std_w2 = (2.0 / (hidden_dim + 1) as f64).sqrt();
        let w2 = Array1::from_shape_fn(hidden_dim, |_| {
            T::from(rng.gen_range(-std_w2..std_w2)).unwrap_or(T::zero())
        });
        let b2 = T::zero();

        Self {
            input_dim,
            w1,
            b1,
            w2,
            b2,
            hidden_dim,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: &Array1<T>) -> T {
        // Hidden layer: ReLU(W1 * x + b1)
        let h = self.w1.dot(x) + &self.b1;
        let h_relu = h.mapv(|v| if v > T::zero() { v } else { T::zero() });

        // Output layer: W2 * h + b2
        self.w2.dot(&h_relu) + self.b2
    }

    /// Train the predictor using simple gradient descent
    pub fn train(&mut self, features: &[Array1<T>], targets: &[T]) -> NeuralResult<()> {
        if features.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Features and targets must have same length".to_string(),
            ));
        }

        let learning_rate = T::from(0.001).unwrap();
        let n_epochs = 100;

        for _epoch in 0..n_epochs {
            for (x, &target) in features.iter().zip(targets.iter()) {
                // Forward pass
                let h = self.w1.dot(x) + &self.b1;
                let h_relu = h.mapv(|v| if v > T::zero() { v } else { T::zero() });
                let pred = self.w2.dot(&h_relu) + self.b2;

                // Compute loss gradient
                let error = pred - target;

                // Backward pass (simplified)
                let grad_w2 = &h_relu * error;
                let grad_b2 = error;

                // Update weights
                self.w2 = &self.w2 - &(grad_w2 * learning_rate);
                self.b2 = self.b2 - grad_b2 * learning_rate;
            }
        }

        Ok(())
    }

    /// Predict accuracy for given features
    pub fn predict(&self, features: &Array1<T>) -> NeuralResult<f64> {
        let pred = self.forward(features);
        Ok(pred.to_f64().unwrap_or(0.0))
    }
}

/// Knowledge distillation for training smaller subnets
#[derive(Debug)]
pub struct KnowledgeDistillation<T: FloatBounds> {
    /// Temperature for softening probability distributions
    temperature: T,
    /// Weight for distillation loss vs. true label loss
    alpha: T,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> KnowledgeDistillation<T> {
    /// Create a new knowledge distillation trainer
    pub fn new(temperature: T, alpha: T) -> Self {
        Self { temperature, alpha }
    }

    /// Compute distillation loss
    pub fn distillation_loss(
        &self,
        student_logits: &Array1<T>,
        teacher_logits: &Array1<T>,
        true_labels: &Array1<T>,
    ) -> T {
        // Soft targets from teacher
        let teacher_soft = self.softmax_with_temperature(teacher_logits);
        let student_soft = self.softmax_with_temperature(student_logits);

        // KL divergence between distributions
        let kl_loss = self.kl_divergence(&student_soft, &teacher_soft);

        // Cross-entropy with true labels
        let ce_loss = self.cross_entropy(student_logits, true_labels);

        // Combined loss
        self.alpha * kl_loss + (T::one() - self.alpha) * ce_loss
    }

    /// Softmax with temperature
    fn softmax_with_temperature(&self, logits: &Array1<T>) -> Array1<T> {
        let scaled = logits.mapv(|x| x / self.temperature);
        let exp = scaled.mapv(|x| {
            // Safe exp computation
            let x_f64 = x.to_f64().unwrap_or(0.0);
            T::from(x_f64.exp()).unwrap_or(T::zero())
        });
        let sum: T = exp.iter().cloned().sum();
        if sum > T::zero() {
            exp.mapv(|x| x / sum)
        } else {
            Array1::from_elem(logits.len(), T::from(1.0 / logits.len() as f64).unwrap())
        }
    }

    /// KL divergence between two distributions
    fn kl_divergence(&self, p: &Array1<T>, q: &Array1<T>) -> T {
        let eps = T::from(1e-10).unwrap();
        p.iter()
            .zip(q.iter())
            .map(|(&p_i, &q_i)| {
                if p_i > eps {
                    let log_ratio = (p_i / (q_i + eps)).to_f64().unwrap_or(0.0).ln();
                    p_i * T::from(log_ratio).unwrap_or(T::zero())
                } else {
                    T::zero()
                }
            })
            .sum()
    }

    /// Cross-entropy loss
    fn cross_entropy(&self, logits: &Array1<T>, labels: &Array1<T>) -> T {
        let probs = self.softmax_with_temperature(logits);
        let eps = T::from(1e-10).unwrap();
        -labels
            .iter()
            .zip(probs.iter())
            .map(|(&label, &prob)| {
                if label > T::zero() {
                    let log_prob = (prob + eps).to_f64().unwrap_or(1e-10).ln();
                    label * T::from(log_prob).unwrap_or(T::zero())
                } else {
                    T::zero()
                }
            })
            .sum::<T>()
    }
}

// Helper macro for slicing
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ScientificNumber;

    #[test]
    fn test_elastic_config() {
        let config = ElasticConfig::default();
        assert_eq!(config.max_depth(), 4);
        assert_eq!(config.max_width(), 6);
        assert_eq!(config.max_kernel_size(), 7);
    }

    #[test]
    fn test_subnet_config() {
        let config = ElasticConfig::default();
        let resolutions = vec![128, 160, 192, 224];
        let subnet = SubnetConfig::sample_random(&config, &resolutions);

        assert!(config.depths.contains(&subnet.depth));
        assert!(config.widths.contains(&subnet.width));
        assert!(config.kernel_sizes.contains(&subnet.kernel_size));
        assert!(resolutions.contains(&subnet.resolution));
    }

    #[test]
    fn test_subnet_flops_estimation() {
        let subnet = SubnetConfig::new(3, 4, 3, 4.0, 224);
        let flops = subnet.estimate_flops();
        assert!(flops > 0);

        let params = subnet.estimate_params();
        assert!(params > 0);
    }

    #[test]
    fn test_once_for_all_creation() {
        let config = ElasticConfig::default();
        let resolutions = vec![192, 224];
        let ofa: OnceForAllNetwork<f64> = OnceForAllNetwork::new(config, resolutions, 3);

        assert_eq!(ofa.training_phase(), TrainingPhase::LargestOnly);
    }

    #[test]
    fn test_subnet_sampling() {
        let config = ElasticConfig::default();
        let resolutions = vec![192, 224];
        let mut ofa: OnceForAllNetwork<f64> =
            OnceForAllNetwork::new(config.clone(), resolutions, 3);

        // Test different training phases
        ofa.set_training_phase(TrainingPhase::LargestOnly);
        let subnet1 = ofa.sample_subnet();
        assert_eq!(subnet1.depth, config.max_depth());
        assert_eq!(subnet1.width, config.max_width());

        ofa.set_training_phase(TrainingPhase::FullElastic);
        let subnet2 = ofa.sample_subnet();
        assert!(config.depths.contains(&subnet2.depth));
        assert!(config.widths.contains(&subnet2.width));
    }

    #[test]
    fn test_subnet_weight_extraction() {
        let config = ElasticConfig::default();
        let resolutions = vec![224];
        let ofa: OnceForAllNetwork<f64> = OnceForAllNetwork::new(config.clone(), resolutions, 3);

        let subnet = SubnetConfig::new(2, 4, 3, 4.0, 224);
        let weights = ofa.extract_subnet_weights(&subnet);

        assert_eq!(weights.len(), subnet.depth);
    }

    #[test]
    fn test_accuracy_predictor() {
        let mut predictor: AccuracyPredictor<f64> = AccuracyPredictor::new(5);

        // Create some training data
        let features = vec![
            Array1::from(vec![0.8, 0.8, 0.8, 0.8, 0.8]),
            Array1::from(vec![0.5, 0.5, 0.5, 0.5, 0.5]),
            Array1::from(vec![0.3, 0.3, 0.3, 0.3, 0.3]),
        ];
        let targets = vec![0.95, 0.85, 0.75];

        // Train predictor
        predictor.train(&features, &targets).unwrap();

        // Test prediction
        let test_features = Array1::from(vec![0.8, 0.8, 0.8, 0.8, 0.8]);
        let pred = predictor.predict(&test_features).unwrap();
        assert!(pred > 0.0 && pred < 1.0);
    }

    #[test]
    fn test_knowledge_distillation() {
        let kd: KnowledgeDistillation<f64> = KnowledgeDistillation::new(2.0, 0.5);

        let student_logits = Array1::from(vec![2.0, 1.0, 0.5]);
        let teacher_logits = Array1::from(vec![2.5, 1.2, 0.3]);
        let true_labels = Array1::from(vec![1.0, 0.0, 0.0]);

        let loss = kd.distillation_loss(&student_logits, &teacher_logits, &true_labels);
        assert!(loss.to_f64().unwrap() >= 0.0);
    }

    #[test]
    fn test_hardware_latency_estimation() {
        let subnet = SubnetConfig::new(3, 4, 3, 4.0, 224);

        let cpu_latency = subnet.estimate_latency_ms(&HardwareType::CPU);
        let gpu_latency = subnet.estimate_latency_ms(&HardwareType::GPU);
        let mobile_latency = subnet.estimate_latency_ms(&HardwareType::Mobile);

        // Mobile should be slower than GPU
        assert!(mobile_latency > gpu_latency);
        // GPU should be faster than CPU
        assert!(gpu_latency < cpu_latency);
    }

    #[test]
    fn test_training_phase_progression() {
        let config = ElasticConfig::default();
        let resolutions = vec![224];
        let mut ofa: OnceForAllNetwork<f64> = OnceForAllNetwork::new(config, resolutions, 3);

        // Test phase progression
        ofa.set_training_phase(TrainingPhase::LargestOnly);
        assert_eq!(ofa.training_phase(), TrainingPhase::LargestOnly);

        ofa.set_training_phase(TrainingPhase::ElasticKernel);
        assert_eq!(ofa.training_phase(), TrainingPhase::ElasticKernel);

        ofa.set_training_phase(TrainingPhase::FullElastic);
        assert_eq!(ofa.training_phase(), TrainingPhase::FullElastic);
    }

    #[test]
    fn test_subnet_search_with_constraints() {
        let config = ElasticConfig::default();
        let resolutions = vec![192, 224];
        let mut ofa: OnceForAllNetwork<f64> =
            OnceForAllNetwork::new(config.clone(), resolutions, 3);

        // Train a simple accuracy predictor
        let samples = vec![
            (SubnetConfig::new(4, 6, 7, 6.0, 224), 0.95),
            (SubnetConfig::new(3, 5, 5, 4.0, 192), 0.90),
            (SubnetConfig::new(2, 4, 3, 3.0, 128), 0.85),
        ];
        ofa.train_accuracy_predictor(&samples).unwrap();

        // Search for subnet under constraints
        let result = ofa.search_subnet(
            1e9 as u64,        // max FLOPs
            1e6 as u64,        // max params
            100.0,             // max latency ms
            HardwareType::GPU, // hardware type
            50,                // n_samples
        );

        // Should find a valid subnet
        assert!(result.is_ok());
    }
}

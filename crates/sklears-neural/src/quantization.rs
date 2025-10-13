//! Model Quantization utilities for neural network compression
//!
//! This module provides comprehensive quantization techniques to reduce the precision
//! of neural network weights and activations, enabling deployment on resource-constrained
//! devices while maintaining model performance. Quantization reduces memory footprint
//! and improves inference speed, making it crucial for mobile and edge deployment.
//!
//! # Theory
//!
//! Quantization maps high-precision floating-point values to lower-precision
//! representations (typically 8-bit integers). The key equation is:
//!
//! quantized_value = round((float_value - zero_point) / scale)
//! dequantized_value = scale * quantized_value + zero_point
//!
//! Where:
//! - scale: Quantization scale factor
//! - zero_point: Zero point offset for asymmetric quantization
//!
//! # Supported Quantization Types
//!
//! - Post-Training Quantization (PTQ): Quantize pre-trained models
//! - Quantization-Aware Training (QAT): Train with quantization in mind
//! - Dynamic Quantization: Quantize weights statically, activations dynamically
//! - Static Quantization: Quantize both weights and activations statically
//!
//! # Example
//!
//! ```rust,ignore
//! use sklears_neural::quantization::{Quantizer, QuantizationConfig, QuantizationType};
//!
//! let config = QuantizationConfig::default()
//!     .quantization_type(QuantizationType::INT8)
//!     .symmetric(false);
//!
//! let quantizer = Quantizer::new(config);
//! ```

use crate::layers::Layer;
use crate::{NeuralResult, SklearsError};
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis};
use sklears_core::types::{Float, FloatBounds};
use std::collections::HashMap;
use std::marker::PhantomData;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Quantization data type
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QuantizationType {
    /// 8-bit integer quantization
    INT8,
    /// 16-bit integer quantization
    INT16,
    /// 4-bit integer quantization
    INT4,
    /// Binary quantization (1-bit)
    Binary,
    /// Ternary quantization (3 values: -1, 0, 1)
    Ternary,
}

impl Default for QuantizationType {
    fn default() -> Self {
        QuantizationType::INT8
    }
}

/// Quantization strategy
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum QuantizationStrategy {
    /// Post-training quantization
    PostTraining,
    /// Quantization-aware training
    QuantizationAware,
    /// Dynamic quantization
    Dynamic,
    /// Static quantization
    Static,
}

impl Default for QuantizationStrategy {
    fn default() -> Self {
        QuantizationStrategy::PostTraining
    }
}

/// Quantization granularity
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum Granularity {
    /// Per-tensor quantization (single scale/zero-point for entire tensor)
    PerTensor,
    /// Per-channel quantization (different scale/zero-point per output channel)
    PerChannel,
    /// Per-group quantization (different scale/zero-point per group of channels)
    PerGroup { group_size: usize },
}

impl Default for Granularity {
    fn default() -> Self {
        Granularity::PerTensor
    }
}

/// Calibration method for determining quantization parameters
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CalibrationMethod {
    /// Use min/max values from calibration data
    MinMax,
    /// Use percentile values (e.g., 99.9th percentile)
    Percentile { percentile: f64 },
    /// Use entropy-based calibration (KL divergence)
    Entropy,
    /// Use mean squared error for optimal quantization parameters
    MSE,
}

impl Default for CalibrationMethod {
    fn default() -> Self {
        CalibrationMethod::MinMax
    }
}

/// Quantization configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizationConfig {
    /// Type of quantization
    pub quantization_type: QuantizationType,
    /// Quantization strategy
    pub strategy: QuantizationStrategy,
    /// Granularity of quantization
    pub granularity: Granularity,
    /// Calibration method
    pub calibration_method: CalibrationMethod,
    /// Whether to use symmetric quantization (zero_point = 0)
    pub symmetric: bool,
    /// Whether to quantize weights
    pub quantize_weights: bool,
    /// Whether to quantize activations
    pub quantize_activations: bool,
    /// Layers to skip quantization
    pub skip_layers: Vec<String>,
    /// Quantization-aware training parameters
    pub qat_epochs: usize,
    pub qat_learning_rate: Float,
    /// Fake quantization during training
    pub fake_quantize: bool,
    /// Observer momentum for running statistics
    pub observer_momentum: Float,
    /// Number of calibration samples
    pub calibration_samples: usize,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self {
            quantization_type: QuantizationType::INT8,
            strategy: QuantizationStrategy::PostTraining,
            granularity: Granularity::PerTensor,
            calibration_method: CalibrationMethod::MinMax,
            symmetric: true,
            quantize_weights: true,
            quantize_activations: true,
            skip_layers: vec!["input".to_string(), "output".to_string()],
            qat_epochs: 10,
            qat_learning_rate: 0.0001,
            fake_quantize: false,
            observer_momentum: 0.1,
            calibration_samples: 1000,
            random_state: None,
        }
    }
}

impl QuantizationConfig {
    /// Set quantization type
    pub fn quantization_type(mut self, qtype: QuantizationType) -> Self {
        self.quantization_type = qtype;
        self
    }

    /// Set quantization strategy
    pub fn strategy(mut self, strategy: QuantizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set granularity
    pub fn granularity(mut self, granularity: Granularity) -> Self {
        self.granularity = granularity;
        self
    }

    /// Set calibration method
    pub fn calibration_method(mut self, method: CalibrationMethod) -> Self {
        self.calibration_method = method;
        self
    }

    /// Set symmetric quantization
    pub fn symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    /// Configure what to quantize
    pub fn quantize_components(mut self, weights: bool, activations: bool) -> Self {
        self.quantize_weights = weights;
        self.quantize_activations = activations;
        self
    }

    /// Set layers to skip
    pub fn skip_layers(mut self, layers: Vec<String>) -> Self {
        self.skip_layers = layers;
        self
    }

    /// Configure quantization-aware training
    pub fn qat_config(mut self, epochs: usize, lr: Float) -> Self {
        self.strategy = QuantizationStrategy::QuantizationAware;
        self.qat_epochs = epochs;
        self.qat_learning_rate = lr;
        self
    }
}

/// Quantization parameters
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizationParams {
    /// Quantization scale
    pub scale: f64,
    /// Zero point (for asymmetric quantization)
    pub zero_point: i32,
    /// Minimum value
    pub min_val: f64,
    /// Maximum value
    pub max_val: f64,
    /// Number of quantization levels
    pub n_levels: u32,
}

impl QuantizationParams {
    pub fn new(scale: f64, zero_point: i32, min_val: f64, max_val: f64, n_levels: u32) -> Self {
        Self {
            scale,
            zero_point,
            min_val,
            max_val,
            n_levels,
        }
    }

    /// Create symmetric quantization parameters
    pub fn symmetric(max_abs: f64, n_levels: u32) -> Self {
        let scale = 2.0 * max_abs / (n_levels - 1) as f64;
        Self {
            scale,
            zero_point: 0,
            min_val: -max_abs,
            max_val: max_abs,
            n_levels,
        }
    }

    /// Create asymmetric quantization parameters
    pub fn asymmetric(min_val: f64, max_val: f64, n_levels: u32) -> Self {
        let scale = (max_val - min_val) / (n_levels - 1) as f64;
        let zero_point = (-min_val / scale).round() as i32;
        Self {
            scale,
            zero_point,
            min_val,
            max_val,
            n_levels,
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized integer values
    pub data: Array2<i32>,
    /// Quantization parameters
    pub params: QuantizationParams,
    /// Original shape
    pub original_shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Create new quantized tensor
    pub fn new(data: Array2<i32>, params: QuantizationParams, original_shape: Vec<usize>) -> Self {
        Self {
            data,
            params,
            original_shape,
        }
    }

    /// Dequantize to floating point
    pub fn dequantize<T: FloatBounds>(&self) -> NeuralResult<Array2<T>> {
        let mut result = Array2::zeros(self.data.dim());

        for (i, &quantized_val) in self.data.iter().enumerate() {
            let dequantized = self.params.scale * (quantized_val - self.params.zero_point) as f64;
            result.as_slice_mut().unwrap()[i] = T::from(dequantized).unwrap();
        }

        Ok(result)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f64 {
        // Assuming original was 32-bit float and quantized is 8-bit int
        32.0 / 8.0
    }
}

/// Observer for collecting statistics during calibration
#[derive(Debug, Clone)]
pub struct Observer {
    /// Running minimum
    min_val: f64,
    /// Running maximum
    max_val: f64,
    /// Running count
    count: usize,
    /// Momentum for exponential moving average
    momentum: f64,
    /// Histogram for entropy calibration
    histogram: Vec<u64>,
    histogram_bins: usize,
}

impl Observer {
    pub fn new(momentum: f64, histogram_bins: usize) -> Self {
        Self {
            min_val: f64::INFINITY,
            max_val: f64::NEG_INFINITY,
            count: 0,
            momentum,
            histogram: vec![0; histogram_bins],
            histogram_bins,
        }
    }

    /// Update observer with new data
    pub fn update<T: FloatBounds>(&mut self, data: &Array2<T>) {
        let is_first_update = self.count == 0;

        for &val in data.iter() {
            let val_f64 = val.to_f64().unwrap_or(0.0);

            if is_first_update && self.min_val == f64::INFINITY {
                self.min_val = val_f64;
                self.max_val = val_f64;
            } else {
                // Update min/max directly
                self.min_val = self.min_val.min(val_f64);
                self.max_val = self.max_val.max(val_f64);
            }

            // Update histogram
            self.update_histogram(val_f64);
        }

        self.count += 1;
    }

    fn update_histogram(&mut self, val: f64) {
        if self.min_val < self.max_val {
            let range = self.max_val - self.min_val;
            let normalized = (val - self.min_val) / range;
            let bin_idx = ((normalized * (self.histogram_bins - 1) as f64).floor() as usize)
                .min(self.histogram_bins - 1);
            self.histogram[bin_idx] += 1;
        }
    }

    /// Get quantization parameters based on calibration method
    pub fn get_quantization_params(
        &self,
        method: &CalibrationMethod,
        symmetric: bool,
        n_levels: u32,
    ) -> QuantizationParams {
        match method {
            CalibrationMethod::MinMax => {
                if symmetric {
                    let max_abs = self.min_val.abs().max(self.max_val.abs());
                    QuantizationParams::symmetric(max_abs, n_levels)
                } else {
                    QuantizationParams::asymmetric(self.min_val, self.max_val, n_levels)
                }
            }
            CalibrationMethod::Percentile { percentile } => {
                // Use histogram to compute percentile
                let (min_p, max_p) = self.compute_percentile(*percentile);
                if symmetric {
                    let max_abs = min_p.abs().max(max_p.abs());
                    QuantizationParams::symmetric(max_abs, n_levels)
                } else {
                    QuantizationParams::asymmetric(min_p, max_p, n_levels)
                }
            }
            CalibrationMethod::Entropy => {
                // Use KL divergence to find optimal clipping values
                let (min_opt, max_opt) = self.compute_optimal_clipping(n_levels);
                QuantizationParams::asymmetric(min_opt, max_opt, n_levels)
            }
            CalibrationMethod::MSE => {
                // Use MSE optimization for quantization parameters
                let (min_mse, max_mse) = self.compute_mse_optimal();
                QuantizationParams::asymmetric(min_mse, max_mse, n_levels)
            }
        }
    }

    fn compute_percentile(&self, percentile: f64) -> (f64, f64) {
        let total_count: u64 = self.histogram.iter().sum();
        if total_count == 0 {
            return (self.min_val, self.max_val);
        }

        let target_low = ((100.0 - percentile) / 2.0 / 100.0 * total_count as f64) as u64;
        let target_high = (((100.0 + percentile) / 2.0 / 100.0) * total_count as f64) as u64;

        let mut cumsum = 0;
        let mut min_p = self.min_val;
        let mut max_p = self.max_val;

        let bin_width = (self.max_val - self.min_val) / self.histogram_bins as f64;

        for (i, &count) in self.histogram.iter().enumerate() {
            cumsum += count;
            if cumsum >= target_low && min_p == self.min_val {
                min_p = self.min_val + i as f64 * bin_width;
            }
            if cumsum >= target_high {
                max_p = self.min_val + (i + 1) as f64 * bin_width;
                break;
            }
        }

        (min_p, max_p)
    }

    fn compute_optimal_clipping(&self, n_levels: u32) -> (f64, f64) {
        // Simplified implementation - in practice would use KL divergence
        let (min_p, max_p) = self.compute_percentile(99.9);
        (min_p, max_p)
    }

    fn compute_mse_optimal(&self) -> (f64, f64) {
        // Simplified implementation - in practice would minimize MSE
        (self.min_val, self.max_val)
    }
}

/// Main quantization engine
pub struct Quantizer<T: FloatBounds> {
    config: QuantizationConfig,
    /// Layer-specific quantization parameters
    layer_params: HashMap<String, QuantizationParams>,
    /// Observers for calibration
    observers: HashMap<String, Observer>,
    /// Calibration data storage
    calibration_data: HashMap<String, Vec<Array2<T>>>,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds> Quantizer<T> {
    /// Create new quantizer
    pub fn new(config: QuantizationConfig) -> Self {
        Self {
            config,
            layer_params: HashMap::new(),
            observers: HashMap::new(),
            calibration_data: HashMap::new(),
            _phantom: PhantomData,
        }
    }

    /// Calibrate quantization parameters using sample data
    pub fn calibrate(&mut self, model_data: &HashMap<String, Array2<T>>) -> NeuralResult<()> {
        // Initialize observers
        for (layer_name, data) in model_data {
            if !self.config.skip_layers.contains(layer_name) {
                let observer = Observer::new(self.config.observer_momentum, 256);
                self.observers.insert(layer_name.clone(), observer);
            }
        }

        // Collect statistics from calibration data
        for (layer_name, data) in model_data {
            if let Some(observer) = self.observers.get_mut(layer_name) {
                observer.update(data);
            }
        }

        // Generate quantization parameters
        for (layer_name, observer) in &self.observers {
            let n_levels = self.get_quantization_levels();
            let params = observer.get_quantization_params(
                &self.config.calibration_method,
                self.config.symmetric,
                n_levels,
            );
            self.layer_params.insert(layer_name.clone(), params);
        }

        Ok(())
    }

    /// Quantize a tensor
    pub fn quantize_tensor(
        &self,
        tensor: &Array2<T>,
        layer_name: &str,
    ) -> NeuralResult<QuantizedTensor> {
        let params =
            self.layer_params
                .get(layer_name)
                .ok_or_else(|| SklearsError::InvalidParameter {
                    name: "layer_name".to_string(),
                    reason: format!("No quantization parameters found for layer {}", layer_name),
                })?;

        let quantized_data = self.apply_quantization(tensor, params)?;
        let original_shape = tensor.shape().to_vec();

        Ok(QuantizedTensor::new(
            quantized_data,
            params.clone(),
            original_shape,
        ))
    }

    /// Apply quantization to tensor data
    fn apply_quantization(
        &self,
        tensor: &Array2<T>,
        params: &QuantizationParams,
    ) -> NeuralResult<Array2<i32>> {
        let mut quantized = Array2::zeros(tensor.dim());

        for (i, &val) in tensor.iter().enumerate() {
            let val_f64 = val.to_f64().unwrap_or(0.0);

            // Apply quantization formula: q = round(x / scale) + zero_point
            let quantized_val = (val_f64 / params.scale).round() as i32 + params.zero_point;

            // Clamp to valid range
            let clamped = quantized_val
                .max(-(params.n_levels as i32 / 2))
                .min(params.n_levels as i32 / 2 - 1);

            quantized.as_slice_mut().unwrap()[i] = clamped;
        }

        Ok(quantized)
    }

    /// Dequantize a quantized tensor
    pub fn dequantize_tensor(&self, quantized: &QuantizedTensor) -> NeuralResult<Array2<T>> {
        quantized.dequantize()
    }

    /// Quantize model weights
    pub fn quantize_weights(
        &mut self,
        weights: &HashMap<String, Array2<T>>,
    ) -> NeuralResult<HashMap<String, QuantizedTensor>> {
        if !self.config.quantize_weights {
            return Err(SklearsError::InvalidParameter {
                name: "quantize_weights".to_string(),
                reason: "Weight quantization is disabled".to_string(),
            });
        }

        // Calibrate if not already done
        if self.layer_params.is_empty() {
            self.calibrate(weights)?;
        }

        let mut quantized_weights = HashMap::new();
        for (layer_name, weight) in weights {
            if !self.config.skip_layers.contains(layer_name) {
                let quantized = self.quantize_tensor(weight, layer_name)?;
                quantized_weights.insert(layer_name.clone(), quantized);
            }
        }

        Ok(quantized_weights)
    }

    /// Apply fake quantization (for QAT)
    pub fn fake_quantize_tensor(
        &self,
        tensor: &Array2<T>,
        layer_name: &str,
    ) -> NeuralResult<Array2<T>> {
        if !self.config.fake_quantize {
            return Ok(tensor.clone());
        }

        // Quantize then immediately dequantize
        let quantized = self.quantize_tensor(tensor, layer_name)?;
        self.dequantize_tensor(&quantized)
    }

    /// Get number of quantization levels based on type
    fn get_quantization_levels(&self) -> u32 {
        match self.config.quantization_type {
            QuantizationType::INT8 => 256,
            QuantizationType::INT16 => 65536,
            QuantizationType::INT4 => 16,
            QuantizationType::Binary => 2,
            QuantizationType::Ternary => 3,
        }
    }

    /// Compute quantization error
    pub fn compute_quantization_error(
        &self,
        original: &Array2<T>,
        quantized: &QuantizedTensor,
    ) -> NeuralResult<QuantizationMetrics> {
        let reconstructed = self.dequantize_tensor(quantized)?;

        // Mean Squared Error
        let diff = original - &reconstructed;
        let mse = diff.mapv(|x| x * x).mean().unwrap().to_f64().unwrap_or(0.0);

        // Signal-to-Noise Ratio
        let signal_power = original
            .mapv(|x| x * x)
            .mean()
            .unwrap()
            .to_f64()
            .unwrap_or(0.0);
        let snr_db = if mse > 0.0 {
            10.0 * (signal_power / mse).log10()
        } else {
            f64::INFINITY
        };

        // Peak Signal-to-Noise Ratio
        let max_val = original
            .iter()
            .map(|&x| x.abs())
            .fold(T::from(0.0).unwrap(), |a, b| a.max(b));
        let max_val_f64 = max_val.to_f64().unwrap_or(0.0);
        let psnr_db = if mse > 0.0 {
            20.0 * (max_val_f64 / mse.sqrt()).log10()
        } else {
            f64::INFINITY
        };

        // Compression ratio
        let compression_ratio = quantized.compression_ratio();

        Ok(QuantizationMetrics {
            mse,
            snr_db,
            psnr_db,
            compression_ratio,
            original_size: original.len() * std::mem::size_of::<T>(),
            quantized_size: quantized.data.len() * std::mem::size_of::<i32>(),
        })
    }

    /// Analyze quantization sensitivity
    pub fn analyze_layer_sensitivity(
        &mut self,
        layers_data: &HashMap<String, Array2<T>>,
    ) -> NeuralResult<HashMap<String, f64>> {
        let mut sensitivity_scores = HashMap::new();

        for (layer_name, data) in layers_data {
            if self.config.skip_layers.contains(layer_name) {
                continue;
            }

            // Compute quantization error for this layer
            let quantized = self.quantize_tensor(data, layer_name)?;
            let metrics = self.compute_quantization_error(data, &quantized)?;

            // Use MSE as sensitivity score
            sensitivity_scores.insert(layer_name.clone(), metrics.mse);
        }

        Ok(sensitivity_scores)
    }
}

/// Quantization performance metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct QuantizationMetrics {
    /// Mean Squared Error
    pub mse: f64,
    /// Signal-to-Noise Ratio (dB)
    pub snr_db: f64,
    /// Peak Signal-to-Noise Ratio (dB)
    pub psnr_db: f64,
    /// Compression ratio
    pub compression_ratio: f64,
    /// Original tensor size in bytes
    pub original_size: usize,
    /// Quantized tensor size in bytes
    pub quantized_size: usize,
}

impl QuantizationMetrics {
    /// Check if quantization quality is acceptable
    pub fn is_acceptable(&self, min_snr_db: f64, max_mse: f64) -> bool {
        self.snr_db >= min_snr_db && self.mse <= max_mse
    }

    /// Get memory savings ratio
    pub fn memory_savings(&self) -> f64 {
        1.0 - (self.quantized_size as f64 / self.original_size as f64)
    }
}

/// Utility functions for quantization
pub mod utils {
    use super::*;

    /// Create INT8 post-training quantization config
    pub fn int8_ptq_config() -> QuantizationConfig {
        QuantizationConfig::default()
            .quantization_type(QuantizationType::INT8)
            .strategy(QuantizationStrategy::PostTraining)
            .symmetric(true)
    }

    /// Create INT8 quantization-aware training config
    pub fn int8_qat_config(epochs: usize, lr: f64) -> QuantizationConfig {
        QuantizationConfig::default()
            .quantization_type(QuantizationType::INT8)
            .qat_config(epochs, lr)
    }

    /// Create dynamic quantization config
    pub fn dynamic_quantization_config() -> QuantizationConfig {
        QuantizationConfig::default()
            .strategy(QuantizationStrategy::Dynamic)
            .quantize_components(true, false) // Only quantize weights
    }

    /// Create per-channel quantization config
    pub fn per_channel_config() -> QuantizationConfig {
        QuantizationConfig::default()
            .granularity(Granularity::PerChannel)
            .symmetric(false)
    }

    /// Create binary quantization config
    pub fn binary_quantization_config() -> QuantizationConfig {
        QuantizationConfig::default()
            .quantization_type(QuantizationType::Binary)
            .symmetric(true)
    }

    /// Evaluate quantization impact on model accuracy
    pub fn evaluate_accuracy_impact<T: FloatBounds>(
        original_accuracy: f64,
        quantized_accuracy: f64,
    ) -> f64 {
        (original_accuracy - quantized_accuracy) / original_accuracy
    }

    /// Select optimal quantization parameters based on sensitivity analysis
    pub fn select_optimal_parameters(
        sensitivity_scores: &HashMap<String, f64>,
        target_compression: f64,
    ) -> Vec<String> {
        let mut layers: Vec<(String, f64)> = sensitivity_scores
            .iter()
            .map(|(name, &score)| (name.clone(), score))
            .collect();

        // Sort by sensitivity (ascending - less sensitive first)
        layers.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select layers to quantize based on target compression
        let n_layers_to_quantize = (layers.len() as f64 * target_compression) as usize;
        layers
            .into_iter()
            .take(n_layers_to_quantize)
            .map(|(name, _)| name)
            .collect()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx;
    use scirs2_core::ndarray::arr2;

    #[test]
    fn test_quantization_config() {
        let config = QuantizationConfig::default()
            .quantization_type(QuantizationType::INT8)
            .strategy(QuantizationStrategy::PostTraining)
            .symmetric(false);

        assert_eq!(config.quantization_type, QuantizationType::INT8);
        assert_eq!(config.strategy, QuantizationStrategy::PostTraining);
        assert!(!config.symmetric);
    }

    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::symmetric(1.0, 256);
        assert_eq!(params.zero_point, 0);
        approx::assert_abs_diff_eq!(params.scale, 2.0 / 255.0, epsilon = 1e-10);

        let params = QuantizationParams::asymmetric(-1.0, 2.0, 256);
        approx::assert_abs_diff_eq!(params.scale, 3.0 / 255.0, epsilon = 1e-10);
        assert!(params.zero_point > 0);
    }

    #[test]
    fn test_observer() {
        let mut observer = Observer::new(0.1, 100);

        let data = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        observer.update(&data);

        assert_eq!(observer.min_val, 1.0);
        assert_eq!(observer.max_val, 4.0);

        let params = observer.get_quantization_params(&CalibrationMethod::MinMax, true, 256);
        assert_eq!(params.zero_point, 0); // Symmetric
    }

    #[test]
    fn test_quantization_dequantization() {
        let config = QuantizationConfig::default();
        let mut quantizer = Quantizer::<f64>::new(config);

        let data = arr2(&[[1.0, -1.0], [0.5, -0.5]]);
        let mut layer_data = HashMap::new();
        layer_data.insert("test_layer".to_string(), data.clone());

        quantizer.calibrate(&layer_data).unwrap();

        let quantized = quantizer.quantize_tensor(&data, "test_layer").unwrap();
        let reconstructed = quantizer.dequantize_tensor(&quantized).unwrap();

        // Check that reconstruction is close to original
        for (orig, recon) in data.iter().zip(reconstructed.iter()) {
            approx::assert_abs_diff_eq!(orig, recon, epsilon = 0.1);
        }
    }

    #[test]
    fn test_quantization_metrics() {
        let original = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let reconstructed = arr2(&[[1.1, 1.9], [3.1, 3.9]]);

        let config = QuantizationConfig::default();
        let quantizer = Quantizer::<f64>::new(config);

        // Create a mock quantized tensor for testing
        let quantized_data = arr2(&[[110, 190], [310, 390]]);
        let params = QuantizationParams::symmetric(4.0, 256);
        let quantized = QuantizedTensor::new(quantized_data, params, vec![2, 2]);

        let metrics = quantizer
            .compute_quantization_error(&original, &quantized)
            .unwrap();
        assert!(metrics.mse >= 0.0);
        assert!(metrics.compression_ratio > 1.0);
    }

    #[test]
    fn test_utility_functions() {
        let config = utils::int8_ptq_config();
        assert_eq!(config.quantization_type, QuantizationType::INT8);
        assert_eq!(config.strategy, QuantizationStrategy::PostTraining);

        let qat_config = utils::int8_qat_config(10, 0.001);
        assert_eq!(qat_config.strategy, QuantizationStrategy::QuantizationAware);
        assert_eq!(qat_config.qat_epochs, 10);

        let accuracy_impact = utils::evaluate_accuracy_impact::<f64>(0.95, 0.92);
        approx::assert_abs_diff_eq!(accuracy_impact, 0.0315789, epsilon = 1e-6);
    }

    #[test]
    fn test_sensitivity_analysis() {
        let mut sensitivity_scores = HashMap::new();
        sensitivity_scores.insert("layer1".to_string(), 0.1);
        sensitivity_scores.insert("layer2".to_string(), 0.05);
        sensitivity_scores.insert("layer3".to_string(), 0.2);

        let selected = utils::select_optimal_parameters(&sensitivity_scores, 0.67);
        assert_eq!(selected.len(), 2); // 2 out of 3 layers
        assert!(selected.contains(&"layer2".to_string())); // Least sensitive should be first
    }

    #[test]
    fn test_quantized_tensor() {
        let data = arr2(&[[100, 150], [200, 250]]);
        let params = QuantizationParams::symmetric(2.0, 256);
        let quantized = QuantizedTensor::new(data, params, vec![2, 2]);

        let dequantized = quantized.dequantize::<f64>().unwrap();
        assert!(dequantized.dim() == (2, 2));

        let ratio = quantized.compression_ratio();
        assert_eq!(ratio, 4.0); // 32-bit to 8-bit
    }
}

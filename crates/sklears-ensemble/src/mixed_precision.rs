//! Mixed-precision training support for ensemble methods
//!
//! This module provides mixed-precision training capabilities using FP16 and FP32
//! to reduce memory usage and improve training speed while maintaining numerical stability.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

/// Half-precision floating point type (FP16)
pub type Half = half::f16;

/// Mixed precision configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    /// Enable mixed precision training
    pub enabled: bool,
    /// Loss scaling factor to prevent gradient underflow
    pub loss_scale: Float,
    /// Dynamic loss scaling
    pub dynamic_loss_scaling: bool,
    /// Initial loss scale for dynamic scaling
    pub initial_loss_scale: Float,
    /// Growth factor for loss scaling
    pub growth_factor: Float,
    /// Backoff factor when overflow is detected
    pub backoff_factor: Float,
    /// Number of steps without overflow before increasing scale
    pub growth_interval: usize,
    /// Operations to keep in FP32 (for numerical stability)
    pub fp32_operations: Vec<String>,
    /// Use automatic mixed precision (AMP)
    pub use_amp: bool,
    /// Gradient clipping threshold
    pub gradient_clip_threshold: Option<Float>,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            loss_scale: 65536.0, // 2^16
            dynamic_loss_scaling: true,
            initial_loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            fp32_operations: vec![
                "loss_computation".to_string(),
                "batch_norm".to_string(),
                "layer_norm".to_string(),
                "softmax".to_string(),
            ],
            use_amp: true,
            gradient_clip_threshold: Some(1.0),
        }
    }
}

/// Mixed precision trainer
pub struct MixedPrecisionTrainer {
    config: MixedPrecisionConfig,
    current_loss_scale: Float,
    overflow_count: usize,
    successful_steps: usize,
    scaler_state: ScalerState,
}

/// Loss scaler state
#[derive(Debug, Clone)]
pub struct ScalerState {
    pub scale: Float,
    pub growth_tracker: usize,
    pub overflow_detected: bool,
    pub should_skip_step: bool,
}

/// Mixed precision data types
#[derive(Debug, Clone)]
pub enum MixedPrecisionArray {
    /// Full precision (FP32)
    Full(Array2<Float>),
    /// Half precision (FP16)
    Half(Array2<Half>),
    /// Mixed arrays with different precisions per operation
    Mixed {
        fp32_data: Array2<Float>,
        fp16_data: Array2<Half>,
        precision_mask: Array2<bool>, // true = FP32, false = FP16
    },
}

/// Gradient accumulator with mixed precision
pub struct MixedPrecisionGradientAccumulator {
    fp32_gradients: HashMap<String, Array2<Float>>,
    fp16_gradients: HashMap<String, Array2<Half>>,
    accumulation_count: usize,
}

/// Automatic Mixed Precision (AMP) context
pub struct AMPContext {
    config: MixedPrecisionConfig,
    scaler: GradientScaler,
    autocast_enabled: bool,
}

/// Gradient scaler for mixed precision training
pub struct GradientScaler {
    scale: Float,
    growth_tracker: usize,
    growth_interval: usize,
    backoff_factor: Float,
    growth_factor: Float,
}

impl MixedPrecisionTrainer {
    /// Create new mixed precision trainer
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let current_loss_scale = if config.dynamic_loss_scaling {
            config.initial_loss_scale
        } else {
            config.loss_scale
        };

        Self {
            config,
            current_loss_scale,
            overflow_count: 0,
            successful_steps: 0,
            scaler_state: ScalerState {
                scale: current_loss_scale,
                growth_tracker: 0,
                overflow_detected: false,
                should_skip_step: false,
            },
        }
    }

    /// Enable mixed precision training
    pub fn enable() -> Self {
        Self::new(MixedPrecisionConfig {
            enabled: true,
            ..Default::default()
        })
    }

    /// Convert array to mixed precision format
    pub fn to_mixed_precision(
        &self,
        array: &Array2<Float>,
        operation_name: &str,
    ) -> MixedPrecisionArray {
        if !self.config.enabled {
            return MixedPrecisionArray::Full(array.clone());
        }

        if self
            .config
            .fp32_operations
            .contains(&operation_name.to_string())
        {
            // Keep in FP32 for numerical stability
            MixedPrecisionArray::Full(array.clone())
        } else {
            // Convert to FP16
            let half_array = array.map(|&x| Half::from_f32(x as f32));
            MixedPrecisionArray::Half(half_array)
        }
    }

    /// Convert mixed precision array back to FP32
    pub fn to_full_precision(&self, array: &MixedPrecisionArray) -> Array2<Float> {
        match array {
            MixedPrecisionArray::Full(arr) => arr.clone(),
            MixedPrecisionArray::Half(arr) => arr.map(|&x| x.to_f32() as Float),
            MixedPrecisionArray::Mixed {
                fp32_data,
                fp16_data,
                precision_mask,
            } => {
                let mut result = Array2::zeros(fp32_data.dim());
                for ((i, j), &use_fp32) in precision_mask.indexed_iter() {
                    result[[i, j]] = if use_fp32 {
                        fp32_data[[i, j]]
                    } else {
                        fp16_data[[i, j]].to_f32() as Float
                    };
                }
                result
            }
        }
    }

    /// Scale gradients to prevent underflow
    pub fn scale_gradients(&self, gradients: &mut Array2<Float>) {
        if self.config.enabled {
            *gradients *= self.current_loss_scale;
        }
    }

    /// Unscale gradients after backward pass
    pub fn unscale_gradients(&self, gradients: &mut Array2<Float>) -> bool {
        if !self.config.enabled {
            return false;
        }

        // Check for overflow/infinities
        let has_overflow = gradients.iter().any(|&x| !x.is_finite());

        if !has_overflow {
            *gradients /= self.current_loss_scale;
        }

        has_overflow
    }

    /// Update loss scale based on overflow detection
    pub fn update_scale(&mut self, overflow_detected: bool) {
        if !self.config.dynamic_loss_scaling {
            return;
        }

        self.scaler_state.overflow_detected = overflow_detected;

        if overflow_detected {
            // Reduce scale on overflow
            self.current_loss_scale *= self.config.backoff_factor;
            self.current_loss_scale = self.current_loss_scale.max(1.0);
            self.overflow_count += 1;
            self.successful_steps = 0;
            self.scaler_state.should_skip_step = true;
        } else {
            // Increase scale after successful steps
            self.successful_steps += 1;
            self.scaler_state.should_skip_step = false;

            if self.successful_steps >= self.config.growth_interval {
                self.current_loss_scale *= self.config.growth_factor;
                self.successful_steps = 0;
            }
        }

        self.scaler_state.scale = self.current_loss_scale;
    }

    /// Check if current step should be skipped due to overflow
    pub fn should_skip_step(&self) -> bool {
        self.scaler_state.should_skip_step
    }

    /// Get current loss scale
    pub fn get_loss_scale(&self) -> Float {
        self.current_loss_scale
    }

    /// Train ensemble with mixed precision
    pub fn train_ensemble_mixed_precision<F>(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Int>,
        n_estimators: usize,
        mut train_fn: F,
    ) -> Result<Vec<Array1<Float>>>
    where
        F: FnMut(&MixedPrecisionArray, &Array1<Int>) -> Result<Array1<Float>>,
    {
        let mut models = Vec::new();

        for i in 0..n_estimators {
            // Convert input to mixed precision
            let x_mixed = self.to_mixed_precision(x, "forward_pass");

            // Train single model
            let model = train_fn(&x_mixed, y)?;

            // Apply gradient scaling if needed
            // In a real implementation, this would involve the actual gradient computation

            models.push(model);

            // Update loss scale based on training stability
            let overflow_detected = false; // Would be detected during actual training
            self.update_scale(overflow_detected);
        }

        Ok(models)
    }

    /// Get scaler state
    pub fn scaler_state(&self) -> &ScalerState {
        &self.scaler_state
    }

    /// Reset scaler state
    pub fn reset_scaler(&mut self) {
        self.current_loss_scale = self.config.initial_loss_scale;
        self.overflow_count = 0;
        self.successful_steps = 0;
        self.scaler_state = ScalerState {
            scale: self.current_loss_scale,
            growth_tracker: 0,
            overflow_detected: false,
            should_skip_step: false,
        };
    }
}

impl MixedPrecisionArray {
    /// Get the shape of the array
    pub fn shape(&self) -> (usize, usize) {
        match self {
            MixedPrecisionArray::Full(arr) => arr.dim(),
            MixedPrecisionArray::Half(arr) => arr.dim(),
            MixedPrecisionArray::Mixed { fp32_data, .. } => fp32_data.dim(),
        }
    }

    /// Check if array uses mixed precision
    pub fn is_mixed_precision(&self) -> bool {
        matches!(
            self,
            MixedPrecisionArray::Half(_) | MixedPrecisionArray::Mixed { .. }
        )
    }

    /// Get memory usage in bytes
    pub fn memory_usage_bytes(&self) -> usize {
        match self {
            MixedPrecisionArray::Full(arr) => arr.len() * std::mem::size_of::<Float>(),
            MixedPrecisionArray::Half(arr) => arr.len() * std::mem::size_of::<Half>(),
            MixedPrecisionArray::Mixed {
                fp32_data,
                fp16_data,
                precision_mask,
            } => {
                let fp32_count = precision_mask.iter().filter(|&&x| x).count();
                let fp16_count = precision_mask.len() - fp32_count;
                fp32_count * std::mem::size_of::<Float>()
                    + fp16_count * std::mem::size_of::<Half>()
                    + precision_mask.len() * std::mem::size_of::<bool>()
            }
        }
    }

    /// Element-wise addition with automatic precision handling
    pub fn add(&self, other: &Self) -> Result<Self> {
        match (self, other) {
            (MixedPrecisionArray::Full(a), MixedPrecisionArray::Full(b)) => {
                Ok(MixedPrecisionArray::Full(a + b))
            }
            (MixedPrecisionArray::Half(a), MixedPrecisionArray::Half(b)) => {
                Ok(MixedPrecisionArray::Half(a + b))
            }
            _ => {
                // Convert both to full precision for mixed operations
                let a_full = match self {
                    MixedPrecisionArray::Full(arr) => arr.clone(),
                    MixedPrecisionArray::Half(arr) => arr.map(|&x| x.to_f32() as Float),
                    MixedPrecisionArray::Mixed {
                        fp32_data,
                        fp16_data,
                        precision_mask,
                    } => {
                        let mut result = Array2::zeros(fp32_data.dim());
                        for ((i, j), &use_fp32) in precision_mask.indexed_iter() {
                            result[[i, j]] = if use_fp32 {
                                fp32_data[[i, j]]
                            } else {
                                fp16_data[[i, j]].to_f32() as Float
                            };
                        }
                        result
                    }
                };

                let b_full = match other {
                    MixedPrecisionArray::Full(arr) => arr.clone(),
                    MixedPrecisionArray::Half(arr) => arr.map(|&x| x.to_f32() as Float),
                    MixedPrecisionArray::Mixed {
                        fp32_data,
                        fp16_data,
                        precision_mask,
                    } => {
                        let mut result = Array2::zeros(fp32_data.dim());
                        for ((i, j), &use_fp32) in precision_mask.indexed_iter() {
                            result[[i, j]] = if use_fp32 {
                                fp32_data[[i, j]]
                            } else {
                                fp16_data[[i, j]].to_f32() as Float
                            };
                        }
                        result
                    }
                };

                Ok(MixedPrecisionArray::Full(&a_full + &b_full))
            }
        }
    }
}

impl Default for MixedPrecisionGradientAccumulator {
    fn default() -> Self {
        Self::new()
    }
}

impl MixedPrecisionGradientAccumulator {
    /// Create new gradient accumulator
    pub fn new() -> Self {
        Self {
            fp32_gradients: HashMap::new(),
            fp16_gradients: HashMap::new(),
            accumulation_count: 0,
        }
    }

    /// Accumulate gradients with mixed precision
    pub fn accumulate(&mut self, name: &str, gradients: &MixedPrecisionArray) -> Result<()> {
        match gradients {
            MixedPrecisionArray::Full(grads) => {
                let entry = self
                    .fp32_gradients
                    .entry(name.to_string())
                    .or_insert_with(|| Array2::zeros(grads.dim()));
                *entry = entry.clone() + grads;
            }
            MixedPrecisionArray::Half(grads) => {
                let entry = self
                    .fp16_gradients
                    .entry(name.to_string())
                    .or_insert_with(|| Array2::zeros(grads.dim()));
                *entry = entry.clone() + grads;
            }
            MixedPrecisionArray::Mixed { .. } => {
                // Convert to full precision for accumulation
                let full_grads = match gradients {
                    MixedPrecisionArray::Mixed {
                        fp32_data,
                        fp16_data,
                        precision_mask,
                    } => {
                        let mut result = Array2::zeros(fp32_data.dim());
                        for ((i, j), &use_fp32) in precision_mask.indexed_iter() {
                            result[[i, j]] = if use_fp32 {
                                fp32_data[[i, j]]
                            } else {
                                fp16_data[[i, j]].to_f32() as Float
                            };
                        }
                        result
                    }
                    _ => unreachable!(),
                };

                let entry = self
                    .fp32_gradients
                    .entry(name.to_string())
                    .or_insert_with(|| Array2::zeros(full_grads.dim()));
                *entry = entry.clone() + &full_grads;
            }
        }

        self.accumulation_count += 1;
        Ok(())
    }

    /// Get averaged gradients
    pub fn get_averaged_gradients(&self) -> HashMap<String, Array2<Float>> {
        let mut result = HashMap::new();

        // Convert FP32 gradients
        for (name, grads) in &self.fp32_gradients {
            result.insert(
                name.clone(),
                grads.clone() / self.accumulation_count as Float,
            );
        }

        // Convert FP16 gradients to FP32
        for (name, grads) in &self.fp16_gradients {
            let fp32_grads = grads.map(|&x| x.to_f32() as Float);
            result.insert(name.clone(), fp32_grads / self.accumulation_count as Float);
        }

        result
    }

    /// Clear accumulated gradients
    pub fn clear(&mut self) {
        self.fp32_gradients.clear();
        self.fp16_gradients.clear();
        self.accumulation_count = 0;
    }
}

impl AMPContext {
    /// Create new AMP context
    pub fn new(config: MixedPrecisionConfig) -> Self {
        let scaler = GradientScaler::new(
            config.initial_loss_scale,
            config.growth_interval,
            config.backoff_factor,
            config.growth_factor,
        );

        Self {
            config,
            scaler,
            autocast_enabled: false,
        }
    }

    /// Enable autocast for current scope
    pub fn autocast<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let old_state = self.autocast_enabled;
        self.autocast_enabled = true;
        let result = f(self);
        self.autocast_enabled = old_state;
        result
    }

    /// Check if autocast is enabled
    pub fn is_autocast_enabled(&self) -> bool {
        self.autocast_enabled
    }

    /// Scale loss for backward pass
    pub fn scale_loss(&mut self, loss: Float) -> Float {
        self.scaler.scale(loss)
    }

    /// Step optimizer with gradient scaling
    pub fn step<F>(&mut self, optimizer_step: F) -> bool
    where
        F: FnOnce(),
    {
        if !self.scaler.should_skip_step() {
            optimizer_step();
            self.scaler.update(false); // No overflow
            true
        } else {
            self.scaler.update(true); // Overflow detected
            false
        }
    }
}

impl GradientScaler {
    /// Create new gradient scaler
    pub fn new(
        initial_scale: Float,
        growth_interval: usize,
        backoff_factor: Float,
        growth_factor: Float,
    ) -> Self {
        Self {
            scale: initial_scale,
            growth_tracker: 0,
            growth_interval,
            backoff_factor,
            growth_factor,
        }
    }

    /// Scale value
    pub fn scale(&self, value: Float) -> Float {
        value * self.scale
    }

    /// Unscale value
    pub fn unscale(&self, value: Float) -> Float {
        value / self.scale
    }

    /// Update scale based on overflow detection
    pub fn update(&mut self, overflow_detected: bool) {
        if overflow_detected {
            self.scale *= self.backoff_factor;
            self.scale = self.scale.max(1.0);
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.growth_interval {
                self.scale *= self.growth_factor;
                self.growth_tracker = 0;
            }
        }
    }

    /// Check if step should be skipped
    pub fn should_skip_step(&self) -> bool {
        self.scale < 1.0
    }

    /// Get current scale
    pub fn get_scale(&self) -> Float {
        self.scale
    }
}

/// Utility functions for mixed precision operations
pub mod utils {
    use super::*;

    /// Check if value is in FP16 range
    pub fn is_fp16_representable(value: Float) -> bool {
        let abs_val = value.abs();
        abs_val <= Half::MAX.to_f32() as Float && abs_val >= Half::MIN_POSITIVE.to_f32() as Float
    }

    /// Estimate memory savings from mixed precision
    pub fn estimate_memory_savings(
        fp32_arrays: &[Array2<Float>],
        mixed_precision_ratio: Float,
    ) -> (usize, usize, Float) {
        let fp32_memory = fp32_arrays
            .iter()
            .map(|arr| arr.len() * std::mem::size_of::<Float>())
            .sum::<usize>();

        let fp16_elements = (fp32_arrays.iter().map(|arr| arr.len()).sum::<usize>() as Float
            * mixed_precision_ratio) as usize;
        let fp32_elements = fp32_arrays.iter().map(|arr| arr.len()).sum::<usize>() - fp16_elements;

        let mixed_memory = fp32_elements * std::mem::size_of::<Float>()
            + fp16_elements * std::mem::size_of::<Half>();

        let savings_ratio = 1.0 - (mixed_memory as Float / fp32_memory as Float);

        (fp32_memory, mixed_memory, savings_ratio)
    }

    /// Convert Float to Half with overflow checking
    pub fn safe_float_to_half(value: Float) -> Result<Half> {
        if value.is_finite() && is_fp16_representable(value) {
            Ok(Half::from_f32(value as f32))
        } else {
            Err(SklearsError::InvalidInput(format!(
                "Value {} cannot be represented in FP16",
                value
            )))
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mixed_precision_config() {
        let config = MixedPrecisionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.loss_scale, 65536.0);
        assert!(config.dynamic_loss_scaling);
    }

    #[test]
    fn test_mixed_precision_trainer() {
        let config = MixedPrecisionConfig::default();
        let trainer = MixedPrecisionTrainer::new(config);
        assert_eq!(trainer.get_loss_scale(), 65536.0);
        assert!(!trainer.should_skip_step());
    }

    #[test]
    fn test_mixed_precision_array() {
        let full_array = array![[1.0, 2.0], [3.0, 4.0]];
        let mixed_array = MixedPrecisionArray::Full(full_array.clone());

        assert_eq!(mixed_array.shape(), (2, 2));
        assert!(!mixed_array.is_mixed_precision());
        assert_eq!(
            mixed_array.memory_usage_bytes(),
            4 * std::mem::size_of::<Float>()
        );
    }

    #[test]
    fn test_mixed_precision_array_addition() {
        let a = MixedPrecisionArray::Full(array![[1.0, 2.0], [3.0, 4.0]]);
        let b = MixedPrecisionArray::Full(array![[5.0, 6.0], [7.0, 8.0]]);

        let result = a.add(&b).unwrap();
        match result {
            MixedPrecisionArray::Full(arr) => {
                assert_eq!(arr, array![[6.0, 8.0], [10.0, 12.0]]);
            }
            _ => panic!("Expected full precision result"),
        }
    }

    #[test]
    fn test_gradient_accumulator() {
        let mut accumulator = MixedPrecisionGradientAccumulator::new();

        let grad1 = MixedPrecisionArray::Full(array![[1.0, 2.0], [3.0, 4.0]]);
        let grad2 = MixedPrecisionArray::Full(array![[2.0, 3.0], [4.0, 5.0]]);

        accumulator.accumulate("layer1", &grad1).unwrap();
        accumulator.accumulate("layer1", &grad2).unwrap();

        let averaged = accumulator.get_averaged_gradients();
        let layer1_grads = &averaged["layer1"];
        assert_eq!(*layer1_grads, array![[1.5, 2.5], [3.5, 4.5]]);
    }

    #[test]
    fn test_gradient_scaler() {
        let mut scaler = GradientScaler::new(1024.0, 2000, 0.5, 2.0);

        assert_eq!(scaler.scale(1.0), 1024.0);
        assert_eq!(scaler.unscale(1024.0), 1.0);

        // Test overflow handling
        scaler.update(true);
        assert_eq!(scaler.get_scale(), 512.0);

        // Test growth
        for _ in 0..2000 {
            scaler.update(false);
        }
        assert_eq!(scaler.get_scale(), 1024.0);
    }

    #[test]
    fn test_amp_context() {
        let config = MixedPrecisionConfig::default();
        let mut amp = AMPContext::new(config);

        assert!(!amp.is_autocast_enabled());

        amp.autocast(|ctx| {
            assert!(ctx.is_autocast_enabled());
        });

        assert!(!amp.is_autocast_enabled());
    }

    #[test]
    fn test_memory_savings_estimation() {
        let arrays = vec![
            array![[1.0, 2.0], [3.0, 4.0]],
            array![[5.0, 6.0], [7.0, 8.0]],
        ];

        let (fp32_mem, mixed_mem, savings) = utils::estimate_memory_savings(&arrays, 0.5);

        assert!(fp32_mem > mixed_mem);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_fp16_range_check() {
        assert!(utils::is_fp16_representable(1.0));
        assert!(utils::is_fp16_representable(-1.0));
        assert!(!utils::is_fp16_representable(Float::INFINITY));
        assert!(!utils::is_fp16_representable(Float::NAN));
    }
}

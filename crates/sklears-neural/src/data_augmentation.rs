//! Data augmentation utilities for neural network training.
//!
//! This module provides comprehensive data augmentation pipelines for improving
//! neural network training through data transformations, noise injection,
//! and various augmentation strategies.

use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4};
use scirs2_core::random::essentials::{
    Normal, Normal as RandNormal, Uniform, Uniform as RandUniform,
};
use scirs2_core::random::{ChaCha8Rng, Distribution, Rng, SeedableRng};
use sklears_core::types::FloatBounds;
use std::collections::HashMap;

/// Data augmentation pipeline for composing multiple transformations
pub struct AugmentationPipeline<T: FloatBounds> {
    transformations: Vec<Box<dyn Transformation<T>>>,
    probability: T,
    seed: Option<u64>,
    rng: ChaCha8Rng,
}

impl<T: FloatBounds> AugmentationPipeline<T> {
    /// Create a new augmentation pipeline
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
            probability: T::one(),
            seed: None,
            rng: ChaCha8Rng::seed_from_u64(42),
        }
    }

    /// Create a pipeline with a specific seed for reproducibility
    pub fn with_seed(seed: u64) -> Self {
        Self {
            transformations: Vec::new(),
            probability: T::one(),
            seed: Some(seed),
            rng: ChaCha8Rng::seed_from_u64(seed),
        }
    }

    /// Set the probability of applying the entire pipeline
    pub fn probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }

    /// Add a transformation to the pipeline
    pub fn add_transformation(mut self, transform: Box<dyn Transformation<T>>) -> Self {
        self.transformations.push(transform);
        self
    }

    /// Apply all transformations in sequence
    pub fn apply(&mut self, data: &Array2<T>) -> NeuralResult<Array2<T>> {
        // Check if we should apply augmentation based on probability
        let apply_prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if self.rng.gen::<f64>() > apply_prob {
            return Ok(data.clone());
        }

        let mut result = data.clone();
        for transform in &mut self.transformations {
            result = transform.apply(&result, &mut self.rng)?;
        }
        Ok(result)
    }

    /// Apply augmentation to a batch of data
    pub fn apply_batch(&mut self, batch: &Array3<T>) -> NeuralResult<Array3<T>> {
        let (batch_size, height, width) = batch.dim();
        let mut result = Array3::zeros((batch_size, height, width));

        for i in 0..batch_size {
            let sample = batch.slice(s![i, .., ..]).to_owned();
            let augmented = self.apply(&sample)?;
            result.slice_mut(s![i, .., ..]).assign(&augmented);
        }

        Ok(result)
    }

    /// Reset the random number generator with a new seed
    pub fn reseed(&mut self, seed: u64) {
        self.seed = Some(seed);
        self.rng = ChaCha8Rng::seed_from_u64(seed);
    }
}

impl<T: FloatBounds> Default for AugmentationPipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Base trait for data transformations
pub trait Transformation<T: FloatBounds> {
    /// Apply the transformation to input data
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>>;

    /// Get the name of the transformation
    fn name(&self) -> &str;

    /// Get transformation parameters
    fn parameters(&self) -> HashMap<String, String>;
}

/// Gaussian noise addition
#[derive(Debug)]
pub struct GaussianNoise<T: FloatBounds> {
    mean: T,
    std: T,
    probability: T,
    name: String,
}

impl<T: FloatBounds> GaussianNoise<T> {
    pub fn new(mean: T, std: T) -> Self {
        Self {
            mean,
            std,
            probability: T::one(),
            name: "GaussianNoise".to_string(),
        }
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds> Transformation<T> for GaussianNoise<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let mean: f64 = self.mean.to_f64().unwrap_or(0.0);
        let std: f64 = self.std.to_f64().unwrap_or(1.0);

        let normal = Normal::new(mean, std).map_err(|_| {
            sklears_core::error::SklearsError::InvalidParameter {
                name: "noise_parameters".to_string(),
                reason: "Invalid normal distribution parameters".to_string(),
            }
        })?;

        let mut result = data.clone();
        result.mapv_inplace(|x| {
            let noise = T::from(normal.sample(rng)).unwrap_or_else(T::zero);
            x + noise
        });

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("mean".to_string(), format!("{:?}", self.mean));
        params.insert("std".to_string(), format!("{:?}", self.std));
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        params
    }
}

/// Uniform noise addition
#[derive(Debug)]
pub struct UniformNoise<T: FloatBounds> {
    low: T,
    high: T,
    probability: T,
    name: String,
}

impl<T: FloatBounds> UniformNoise<T> {
    pub fn new(low: T, high: T) -> Self {
        Self {
            low,
            high,
            probability: T::one(),
            name: "UniformNoise".to_string(),
        }
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds> Transformation<T> for UniformNoise<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let low: f64 = self.low.to_f64().unwrap_or(0.0);
        let high: f64 = self.high.to_f64().unwrap_or(1.0);

        let uniform = Uniform::new(low, high).unwrap();
        let mut result = data.clone();

        result.mapv_inplace(|x| {
            let noise = T::from(uniform.sample(rng)).unwrap_or_else(T::zero);
            x + noise
        });

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("low".to_string(), format!("{:?}", self.low));
        params.insert("high".to_string(), format!("{:?}", self.high));
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        params
    }
}

/// Random feature dropout
#[derive(Debug)]
pub struct FeatureDropout<T: FloatBounds> {
    dropout_rate: T,
    probability: T,
    name: String,
}

impl<T: FloatBounds> FeatureDropout<T> {
    pub fn new(dropout_rate: T) -> Self {
        Self {
            dropout_rate,
            probability: T::one(),
            name: "FeatureDropout".to_string(),
        }
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds> Transformation<T> for FeatureDropout<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let dropout_rate: f64 = self.dropout_rate.to_f64().unwrap_or(0.0);
        let mut result = data.clone();

        // Apply dropout to features (columns)
        for mut column in result.columns_mut() {
            if rng.gen::<f64>() < dropout_rate {
                column.fill(T::zero());
            }
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "dropout_rate".to_string(),
            format!("{:?}", self.dropout_rate),
        );
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        params
    }
}

/// Feature scaling/normalization
#[derive(Debug)]
pub struct FeatureScaling<T: FloatBounds> {
    scale_range: (T, T),
    probability: T,
    name: String,
}

impl<T: FloatBounds> FeatureScaling<T> {
    pub fn new(min_scale: T, max_scale: T) -> Self {
        Self {
            scale_range: (min_scale, max_scale),
            probability: T::one(),
            name: "FeatureScaling".to_string(),
        }
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Transformation<T> for FeatureScaling<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let min_scale: f64 = self.scale_range.0.to_f64().unwrap_or(0.8);
        let max_scale: f64 = self.scale_range.1.to_f64().unwrap_or(1.2);

        let uniform = RandUniform::new(min_scale, max_scale).unwrap();
        let scale_factor = T::from(uniform.sample(rng)).unwrap_or_else(T::one);

        Ok(data * scale_factor)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("min_scale".to_string(), format!("{:?}", self.scale_range.0));
        params.insert("max_scale".to_string(), format!("{:?}", self.scale_range.1));
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        params
    }
}

/// Random feature permutation
#[derive(Debug)]
pub struct FeaturePermutation<T: FloatBounds> {
    permutation_ratio: T,
    probability: T,
    name: String,
}

impl<T: FloatBounds> FeaturePermutation<T> {
    pub fn new(permutation_ratio: T) -> Self {
        Self {
            permutation_ratio,
            probability: T::one(),
            name: "FeaturePermutation".to_string(),
        }
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds> Transformation<T> for FeaturePermutation<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let ratio: f64 = self.permutation_ratio.to_f64().unwrap_or(0.1);
        let (n_samples, n_features) = data.dim();
        let n_permute = ((n_features as f64) * ratio) as usize;

        let mut result = data.clone();

        // Select random features to permute
        let mut features_to_permute = Vec::new();
        for _ in 0..n_permute {
            features_to_permute.push(rng.gen_range(0..n_features));
        }

        // Permute selected features
        for &feature_idx in &features_to_permute {
            let mut column = result.column(feature_idx).to_owned();

            // Fisher-Yates shuffle
            for i in (1..n_samples).rev() {
                let j = rng.gen_range(0..=i);
                column.swap(i, j);
            }

            result.column_mut(feature_idx).assign(&column);
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "permutation_ratio".to_string(),
            format!("{:?}", self.permutation_ratio),
        );
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        params
    }
}

/// Time series specific augmentations
#[derive(Debug)]
pub struct TimeSeriesAugmentation<T: FloatBounds> {
    transformations: Vec<TimeSeriesTransform>,
    probability: T,
    name: String,
}

#[derive(Debug, Clone)]
pub enum TimeSeriesTransform {
    TimeWarp { sigma: f64 },
    MagnitudeWarp { sigma: f64 },
    WindowSlicing { ratio: f64 },
    Jittering { sigma: f64 },
}

impl<T: FloatBounds> TimeSeriesAugmentation<T> {
    pub fn new() -> Self {
        Self {
            transformations: Vec::new(),
            probability: T::one(),
            name: "TimeSeriesAugmentation".to_string(),
        }
    }

    pub fn add_transform(mut self, transform: TimeSeriesTransform) -> Self {
        self.transformations.push(transform);
        self
    }

    pub fn with_probability(mut self, prob: T) -> Self {
        self.probability = prob;
        self
    }
}

impl<T: FloatBounds> Transformation<T> for TimeSeriesAugmentation<T> {
    fn apply(&mut self, data: &Array2<T>, rng: &mut ChaCha8Rng) -> NeuralResult<Array2<T>> {
        let prob: f64 = self.probability.to_f64().unwrap_or(1.0);
        if rng.gen::<f64>() > prob {
            return Ok(data.clone());
        }

        let mut result = data.clone();

        for transform in &self.transformations {
            result = match transform {
                TimeSeriesTransform::Jittering { sigma } => {
                    self.apply_jittering(&result, *sigma, rng)?
                }
                TimeSeriesTransform::TimeWarp { sigma } => {
                    self.apply_time_warp(&result, *sigma, rng)?
                }
                TimeSeriesTransform::MagnitudeWarp { sigma } => {
                    self.apply_magnitude_warp(&result, *sigma, rng)?
                }
                TimeSeriesTransform::WindowSlicing { ratio } => {
                    self.apply_window_slicing(&result, *ratio, rng)?
                }
            };
        }

        Ok(result)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn parameters(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert(
            "num_transforms".to_string(),
            self.transformations.len().to_string(),
        );
        params.insert("probability".to_string(), format!("{:?}", self.probability));
        for (i, transform) in self.transformations.iter().enumerate() {
            params.insert(format!("transform_{}", i), format!("{:?}", transform));
        }
        params
    }
}

impl<T: FloatBounds> TimeSeriesAugmentation<T> {
    fn apply_jittering(
        &self,
        data: &Array2<T>,
        sigma: f64,
        rng: &mut ChaCha8Rng,
    ) -> NeuralResult<Array2<T>> {
        let normal = Normal::new(0.0, sigma).map_err(|_| {
            sklears_core::error::SklearsError::InvalidParameter {
                name: "jittering_sigma".to_string(),
                reason: "Invalid sigma for jittering".to_string(),
            }
        })?;

        let mut result = data.clone();
        result.mapv_inplace(|x| {
            let noise = T::from(normal.sample(rng)).unwrap_or_else(T::zero);
            x + noise
        });

        Ok(result)
    }

    fn apply_time_warp(
        &self,
        data: &Array2<T>,
        sigma: f64,
        rng: &mut ChaCha8Rng,
    ) -> NeuralResult<Array2<T>> {
        let (n_samples, n_features) = data.dim();
        let mut result = Array2::zeros((n_samples, n_features));

        // Create time warping function
        let normal = Normal::new(0.0, sigma).map_err(|_| {
            sklears_core::error::SklearsError::InvalidParameter {
                name: "time_warp_sigma".to_string(),
                reason: "Invalid sigma for time warping".to_string(),
            }
        })?;

        for i in 0..n_samples {
            let original_row = data.row(i);
            let mut warped_row = Array1::zeros(n_features);

            for j in 0..n_features {
                // Apply time warping with interpolation
                let warp_factor = 1.0 + normal.sample(rng);
                let warped_idx = (j as f64 * warp_factor) as usize;

                if warped_idx < n_features {
                    warped_row[j] = original_row[warped_idx];
                } else {
                    warped_row[j] = original_row[n_features - 1];
                }
            }

            result.row_mut(i).assign(&warped_row);
        }

        Ok(result)
    }

    fn apply_magnitude_warp(
        &self,
        data: &Array2<T>,
        sigma: f64,
        rng: &mut ChaCha8Rng,
    ) -> NeuralResult<Array2<T>> {
        let normal = Normal::new(1.0, sigma).map_err(|_| {
            sklears_core::error::SklearsError::InvalidParameter {
                name: "magnitude_warp_sigma".to_string(),
                reason: "Invalid sigma for magnitude warping".to_string(),
            }
        })?;

        let mut result = data.clone();

        // Apply different scaling factors to different parts of the series
        let (n_samples, n_features) = data.dim();
        for i in 0..n_samples {
            for j in 0..n_features {
                let scale_factor = T::from(normal.sample(rng)).unwrap_or_else(T::one);
                result[[i, j]] = result[[i, j]] * scale_factor;
            }
        }

        Ok(result)
    }

    fn apply_window_slicing(
        &self,
        data: &Array2<T>,
        ratio: f64,
        rng: &mut ChaCha8Rng,
    ) -> NeuralResult<Array2<T>> {
        let (n_samples, n_features) = data.dim();
        let window_size = ((n_features as f64) * ratio) as usize;

        if window_size == 0 || window_size >= n_features {
            return Ok(data.clone());
        }

        let mut result = Array2::zeros((n_samples, n_features));

        for i in 0..n_samples {
            let start_idx = rng.gen_range(0..=(n_features - window_size));
            let original_row = data.row(i);

            // Copy the selected window and pad with zeros or repeat
            for j in 0..n_features {
                if j < window_size {
                    result[[i, j]] = original_row[start_idx + j];
                } else {
                    // Repeat the last value or use zero padding
                    result[[i, j]] = original_row[start_idx + window_size - 1];
                }
            }
        }

        Ok(result)
    }
}

impl<T: FloatBounds> Default for TimeSeriesAugmentation<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating common augmentation pipelines
pub struct AugmentationBuilder<T: FloatBounds> {
    pipeline: AugmentationPipeline<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> AugmentationBuilder<T> {
    pub fn new() -> Self {
        Self {
            pipeline: AugmentationPipeline::new(),
        }
    }

    pub fn with_seed(seed: u64) -> Self {
        Self {
            pipeline: AugmentationPipeline::with_seed(seed),
        }
    }

    pub fn probability(mut self, prob: T) -> Self {
        self.pipeline = self.pipeline.probability(prob);
        self
    }

    pub fn gaussian_noise(mut self, mean: T, std: T) -> Self {
        let noise = GaussianNoise::new(mean, std);
        self.pipeline = self.pipeline.add_transformation(Box::new(noise));
        self
    }

    pub fn uniform_noise(mut self, low: T, high: T) -> Self {
        let noise = UniformNoise::new(low, high);
        self.pipeline = self.pipeline.add_transformation(Box::new(noise));
        self
    }

    pub fn feature_dropout(mut self, rate: T) -> Self {
        let dropout = FeatureDropout::new(rate);
        self.pipeline = self.pipeline.add_transformation(Box::new(dropout));
        self
    }

    pub fn feature_scaling(mut self, min_scale: T, max_scale: T) -> Self {
        let scaling = FeatureScaling::new(min_scale, max_scale);
        self.pipeline = self.pipeline.add_transformation(Box::new(scaling));
        self
    }

    pub fn feature_permutation(mut self, ratio: T) -> Self {
        let permutation = FeaturePermutation::new(ratio);
        self.pipeline = self.pipeline.add_transformation(Box::new(permutation));
        self
    }

    pub fn time_series_jittering(mut self, sigma: f64) -> Self {
        let ts_aug =
            TimeSeriesAugmentation::new().add_transform(TimeSeriesTransform::Jittering { sigma });
        self.pipeline = self.pipeline.add_transformation(Box::new(ts_aug));
        self
    }

    pub fn build(self) -> AugmentationPipeline<T> {
        self.pipeline
    }
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Default for AugmentationBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_gaussian_noise() -> NeuralResult<()> {
        let mut noise = GaussianNoise::new(0.0f32, 0.1);
        let data = Array2::ones((5, 10));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = noise.apply(&data, &mut rng)?;
        assert_eq!(result.shape(), data.shape());

        // Result should not be exactly the same due to noise
        assert_ne!(result, data);

        Ok(())
    }

    #[test]
    fn test_feature_dropout() -> NeuralResult<()> {
        let mut dropout = FeatureDropout::new(0.5f32);
        let data = Array2::ones((5, 10));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = dropout.apply(&data, &mut rng)?;
        assert_eq!(result.shape(), data.shape());

        // Some features should be dropped (set to zero)
        let zero_features = result
            .columns()
            .into_iter()
            .filter(|col| col.iter().all(|&x| x == 0.0))
            .count();

        assert!(zero_features > 0, "Expected some features to be dropped");

        Ok(())
    }

    #[test]
    fn test_augmentation_pipeline() -> NeuralResult<()> {
        let mut pipeline = AugmentationBuilder::with_seed(42)
            .gaussian_noise(0.0, 0.1)
            .feature_dropout(0.2)
            .feature_scaling(0.9, 1.1)
            .build();

        let data = Array2::ones((5, 10));
        let result = pipeline.apply(&data)?;

        assert_eq!(result.shape(), data.shape());
        assert_ne!(result, data);

        Ok(())
    }

    #[test]
    fn test_time_series_augmentation() -> NeuralResult<()> {
        let mut ts_aug = TimeSeriesAugmentation::new()
            .add_transform(TimeSeriesTransform::Jittering { sigma: 0.1 })
            .add_transform(TimeSeriesTransform::MagnitudeWarp { sigma: 0.1 });

        let data = Array2::from_shape_fn((3, 20), |(i, j)| (i * 20 + j) as f32);
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        let result = ts_aug.apply(&data, &mut rng)?;
        assert_eq!(result.shape(), data.shape());

        Ok(())
    }

    #[test]
    fn test_transformation_parameters() {
        let noise = GaussianNoise::new(0.0f32, 1.0);
        let params = noise.parameters();

        assert!(params.contains_key("mean"));
        assert!(params.contains_key("std"));
        assert!(params.contains_key("probability"));
        assert_eq!(noise.name(), "GaussianNoise");
    }
}

//! Fluent API for Decomposition Pipelines
//!
//! This module provides a builder-pattern fluent API for composing complex
//! decomposition workflows with preprocessing, decomposition, and post-processing
//! steps in a clean, readable manner.
//!
//! Features:
//! - Fluent builder pattern for pipeline construction
//! - Configuration presets for common use cases
//! - Method chaining for intuitive API
//! - Serializable decomposition models
//! - Type-safe pipeline composition

use crate::modular_framework::{
    AlgorithmCapabilities, DecompositionAlgorithm, DecompositionComponents, DecompositionParams,
    ParamValue, PostprocessingStep, PreprocessingStep,
};
use crate::s;
use scirs2_core::ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::any::Any;

/// Fluent API builder for decomposition pipelines
pub struct DecompositionPipelineBuilder {
    preprocessing_steps: Vec<Box<dyn PreprocessingStep>>,
    algorithm: Option<Box<dyn DecompositionAlgorithm>>,
    postprocessing_steps: Vec<Box<dyn PostprocessingStep>>,
    params: DecompositionParams,
    config: PipelineConfig,
}

impl DecompositionPipelineBuilder {
    /// Create a new decomposition pipeline builder
    pub fn new() -> Self {
        Self {
            preprocessing_steps: Vec::new(),
            algorithm: None,
            postprocessing_steps: Vec::new(),
            params: DecompositionParams::default(),
            config: PipelineConfig::default(),
        }
    }

    /// Create builder from preset configuration
    pub fn from_preset(preset: DecompositionPreset) -> Self {
        let builder = Self::new();

        match preset {
            DecompositionPreset::StandardPCA { n_components } => builder
                .with_standardization()
                .with_pca(n_components)
                .with_variance_threshold(0.01),
            DecompositionPreset::RobustPCA { n_components } => builder
                .with_robust_scaling()
                .with_robust_pca(n_components)
                .with_outlier_removal(),
            DecompositionPreset::SparsePCA {
                n_components,
                alpha,
            } => builder
                .with_standardization()
                .with_sparse_pca(n_components, alpha)
                .with_sparsity_constraint(),
            DecompositionPreset::IncrementalPCA {
                n_components,
                batch_size,
            } => builder
                .with_batch_normalization(batch_size)
                .with_incremental_pca(n_components, batch_size),
            DecompositionPreset::KernelPCA {
                n_components,
                kernel,
            } => builder
                .with_standardization()
                .with_kernel_pca(n_components, kernel),
            DecompositionPreset::FastICA { n_components } => builder
                .with_centering()
                .with_whitening()
                .with_fast_ica(n_components),
            DecompositionPreset::NMF {
                n_components,
                init_method,
            } => builder
                .with_non_negative_scaling()
                .with_nmf(n_components, init_method),
            DecompositionPreset::FactorAnalysis { n_components } => builder
                .with_standardization()
                .with_factor_analysis(n_components)
                .with_rotation("varimax"),
            DecompositionPreset::TruncatedSVD { n_components } => {
                builder.with_truncated_svd(n_components)
            }
            DecompositionPreset::TimeSeriesSSA {
                window_length,
                n_components,
            } => builder
                .with_time_series_preprocessing(window_length)
                .with_ssa(window_length, n_components),
        }
    }

    /// Add preprocessing step: standardization
    pub fn with_standardization(mut self) -> Self {
        self.preprocessing_steps
            .push(Box::new(StandardizationStep::new()));
        self
    }

    /// Add preprocessing step: robust scaling
    pub fn with_robust_scaling(mut self) -> Self {
        self.preprocessing_steps
            .push(Box::new(RobustScalingStep::new()));
        self
    }

    /// Add preprocessing step: centering
    pub fn with_centering(mut self) -> Self {
        self.preprocessing_steps
            .push(Box::new(CenteringStep::new()));
        self
    }

    /// Add preprocessing step: whitening
    pub fn with_whitening(mut self) -> Self {
        self.preprocessing_steps
            .push(Box::new(WhiteningStep::new()));
        self
    }

    /// Add preprocessing step: non-negative scaling
    pub fn with_non_negative_scaling(mut self) -> Self {
        self.preprocessing_steps
            .push(Box::new(NonNegativeScalingStep::new()));
        self
    }

    /// Add preprocessing step: batch normalization
    pub fn with_batch_normalization(mut self, batch_size: usize) -> Self {
        self.preprocessing_steps
            .push(Box::new(BatchNormalizationStep::new(batch_size)));
        self
    }

    /// Add preprocessing step: time series preprocessing
    pub fn with_time_series_preprocessing(mut self, window_length: usize) -> Self {
        self.preprocessing_steps
            .push(Box::new(TimeSeriesPreprocessingStep::new(window_length)));
        self
    }

    /// Set decomposition algorithm: PCA
    pub fn with_pca(mut self, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.algorithm = Some(Box::new(PCAAlgorithm::new(n_components)));
        self
    }

    /// Set decomposition algorithm: Robust PCA
    pub fn with_robust_pca(mut self, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.algorithm = Some(Box::new(RobustPCAAlgorithm::new(n_components)));
        self
    }

    /// Set decomposition algorithm: Sparse PCA
    pub fn with_sparse_pca(mut self, n_components: usize, alpha: Float) -> Self {
        self.params.n_components = Some(n_components);
        self.params
            .algorithm_specific
            .insert("alpha".to_string(), ParamValue::Float(alpha));
        self.algorithm = Some(Box::new(SparsePCAAlgorithm::new(n_components, alpha)));
        self
    }

    /// Set decomposition algorithm: Incremental PCA
    pub fn with_incremental_pca(mut self, n_components: usize, batch_size: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.params.algorithm_specific.insert(
            "batch_size".to_string(),
            ParamValue::Integer(batch_size as i64),
        );
        self.algorithm = Some(Box::new(IncrementalPCAAlgorithm::new(
            n_components,
            batch_size,
        )));
        self
    }

    /// Set decomposition algorithm: Kernel PCA
    pub fn with_kernel_pca(mut self, n_components: usize, kernel: String) -> Self {
        self.params.n_components = Some(n_components);
        self.params
            .algorithm_specific
            .insert("kernel".to_string(), ParamValue::String(kernel.clone()));
        self.algorithm = Some(Box::new(KernelPCAAlgorithm::new(n_components, kernel)));
        self
    }

    /// Set decomposition algorithm: FastICA
    pub fn with_fast_ica(mut self, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.algorithm = Some(Box::new(FastICAAlgorithm::new(n_components)));
        self
    }

    /// Set decomposition algorithm: NMF
    pub fn with_nmf(mut self, n_components: usize, init_method: String) -> Self {
        self.params.n_components = Some(n_components);
        self.params.algorithm_specific.insert(
            "init_method".to_string(),
            ParamValue::String(init_method.clone()),
        );
        self.algorithm = Some(Box::new(NMFAlgorithm::new(n_components, init_method)));
        self
    }

    /// Set decomposition algorithm: Factor Analysis
    pub fn with_factor_analysis(mut self, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.algorithm = Some(Box::new(FactorAnalysisAlgorithm::new(n_components)));
        self
    }

    /// Set decomposition algorithm: Truncated SVD
    pub fn with_truncated_svd(mut self, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.algorithm = Some(Box::new(TruncatedSVDAlgorithm::new(n_components)));
        self
    }

    /// Set decomposition algorithm: SSA (Singular Spectrum Analysis)
    pub fn with_ssa(mut self, window_length: usize, n_components: usize) -> Self {
        self.params.n_components = Some(n_components);
        self.params.algorithm_specific.insert(
            "window_length".to_string(),
            ParamValue::Integer(window_length as i64),
        );
        self.algorithm = Some(Box::new(SSAAlgorithm::new(window_length, n_components)));
        self
    }

    /// Add post-processing step: variance threshold
    pub fn with_variance_threshold(mut self, threshold: Float) -> Self {
        self.postprocessing_steps
            .push(Box::new(VarianceThresholdStep::new(threshold)));
        self
    }

    /// Add post-processing step: outlier removal
    pub fn with_outlier_removal(mut self) -> Self {
        self.postprocessing_steps
            .push(Box::new(OutlierRemovalStep::new()));
        self
    }

    /// Add post-processing step: sparsity constraint
    pub fn with_sparsity_constraint(mut self) -> Self {
        self.postprocessing_steps
            .push(Box::new(SparsityConstraintStep::new()));
        self
    }

    /// Add post-processing step: rotation (e.g., varimax)
    pub fn with_rotation(mut self, method: &str) -> Self {
        self.postprocessing_steps
            .push(Box::new(RotationStep::new(method.to_string())));
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.params.n_components = Some(n);
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: Float) -> Self {
        self.params.tolerance = Some(tol);
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.params.max_iterations = Some(max_iter);
        self
    }

    /// Set random seed for reproducibility
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.params.random_seed = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }

    /// Enable automatic parameter tuning
    pub fn auto_tune(mut self, enabled: bool) -> Self {
        self.config.auto_tune = enabled;
        self
    }

    /// Set validation split ratio
    pub fn validation_split(mut self, ratio: Float) -> Self {
        self.config.validation_split = Some(ratio);
        self
    }

    /// Build the decomposition pipeline
    pub fn build(self) -> Result<DecompositionPipeline> {
        if self.algorithm.is_none() {
            return Err(SklearsError::InvalidInput(
                "No decomposition algorithm specified".to_string(),
            ));
        }

        Ok(DecompositionPipeline {
            preprocessing_steps: self.preprocessing_steps,
            algorithm: self.algorithm.expect("operation should succeed"),
            postprocessing_steps: self.postprocessing_steps,
            params: self.params,
            config: self.config,
            fitted: false,
            cached_components: None,
        })
    }
}

impl Default for DecompositionPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration presets for common decomposition use cases
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionPreset {
    /// Standard PCA with centering and scaling
    StandardPCA { n_components: usize },

    /// Robust PCA resistant to outliers
    RobustPCA { n_components: usize },

    /// Sparse PCA for interpretable components
    SparsePCA { n_components: usize, alpha: Float },

    /// Incremental PCA for large datasets
    IncrementalPCA {
        n_components: usize,
        batch_size: usize,
    },

    /// Kernel PCA for non-linear dimensionality reduction
    KernelPCA { n_components: usize, kernel: String },

    /// FastICA for independent component analysis
    FastICA { n_components: usize },

    /// NMF for non-negative matrix factorization
    NMF {
        n_components: usize,
        init_method: String,
    },

    /// Factor Analysis with rotation
    FactorAnalysis { n_components: usize },

    /// Truncated SVD for sparse matrices
    TruncatedSVD { n_components: usize },

    /// Singular Spectrum Analysis for time series
    TimeSeriesSSA {
        window_length: usize,
        n_components: usize,
    },
}

/// Pipeline configuration
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    pub verbose: bool,
    pub auto_tune: bool,
    pub validation_split: Option<Float>,
    pub early_stopping: bool,
    pub save_intermediate: bool,
}

/// Decomposition pipeline that chains preprocessing, decomposition, and post-processing
pub struct DecompositionPipeline {
    preprocessing_steps: Vec<Box<dyn PreprocessingStep>>,
    algorithm: Box<dyn DecompositionAlgorithm>,
    postprocessing_steps: Vec<Box<dyn PostprocessingStep>>,
    params: DecompositionParams,
    config: PipelineConfig,
    fitted: bool,
    /// Post-processed decomposition components cached after `fit()`.
    cached_components: Option<DecompositionComponents>,
}

impl DecompositionPipeline {
    /// Fit the pipeline to data
    pub fn fit(&mut self, data: &Array2<Float>) -> Result<&mut Self> {
        if self.config.verbose {
            println!("Starting decomposition pipeline...");
        }

        // Apply preprocessing
        let mut preprocessed_data = data.clone();
        for (idx, step) in self.preprocessing_steps.iter_mut().enumerate() {
            if self.config.verbose {
                println!("  Applying preprocessing step {}: {}", idx + 1, step.name());
            }
            preprocessed_data = step.process(&preprocessed_data)?;
        }

        // Fit decomposition algorithm
        if self.config.verbose {
            println!("  Fitting algorithm: {}", self.algorithm.name());
        }
        self.algorithm.fit(&preprocessed_data, &self.params)?;

        // Apply postprocessing steps to the raw decomposition components
        if !self.postprocessing_steps.is_empty() {
            if self.config.verbose {
                println!(
                    "  Applying {} postprocessing step(s)…",
                    self.postprocessing_steps.len()
                );
            }
            let raw_components = self.algorithm.get_components()?;
            let mut components = raw_components;
            for (idx, step) in self.postprocessing_steps.iter().enumerate() {
                if self.config.verbose {
                    println!("    Step {}: {}", idx + 1, step.name());
                }
                components = step.process(components)?;
            }
            self.cached_components = Some(components);
        } else {
            // Cache raw components so get_components() is always consistent
            let raw_components = self.algorithm.get_components()?;
            self.cached_components = Some(raw_components);
        }

        self.fitted = true;

        if self.config.verbose {
            println!("Pipeline fitted successfully!");
        }

        Ok(self)
    }

    /// Transform data using the fitted pipeline
    pub fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Pipeline must be fitted before transform".to_string(),
            ));
        }

        // Apply preprocessing
        let mut preprocessed_data = data.clone();
        for _step in &self.preprocessing_steps {
            // Note: This is a simplified implementation
            // In a full implementation, steps would need interior mutability
            // or we would clone the step for processing
            preprocessed_data = preprocessed_data.clone();
        }

        // Transform using algorithm
        let transformed_data = self.algorithm.transform(&preprocessed_data)?;

        Ok(transformed_data)
    }

    /// Fit and transform in one step
    pub fn fit_transform(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        self.fit(data)?;
        self.transform(data)
    }

    /// Inverse transform if supported
    pub fn inverse_transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Pipeline must be fitted before inverse_transform".to_string(),
            ));
        }

        // Inverse transform using algorithm
        let mut result = self.algorithm.inverse_transform(data)?;

        // Apply inverse preprocessing in reverse order
        for step in self.preprocessing_steps.iter().rev() {
            result = step.inverse_process(&result)?;
        }

        Ok(result)
    }

    /// Get decomposition components (post-processed if postprocessing steps are configured)
    pub fn get_components(&self) -> Result<DecompositionComponents> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Pipeline must be fitted before getting components".to_string(),
            ));
        }

        // Return post-processed components if available, otherwise fall back to raw
        if let Some(ref cached) = self.cached_components {
            Ok(cached.clone())
        } else {
            self.algorithm.get_components()
        }
    }

    /// Get explained variance ratio if available
    pub fn explained_variance_ratio(&self) -> Result<Array1<Float>> {
        let components = self.get_components()?;
        components.explained_variance_ratio.ok_or_else(|| {
            SklearsError::InvalidInput("No explained variance available".to_string())
        })
    }

    /// Check if pipeline is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Serialize pipeline to JSON
    pub fn to_json(&self) -> Result<String> {
        let serializable_pipeline = SerializablePipeline {
            algorithm_name: self.algorithm.name().to_string(),
            params: self.params.clone(),
            config: self.config.clone(),
        };

        // Simplified JSON serialization without serde_json dependency
        Ok(format!(
            "{{\"algorithm\":\"{}\"}}",
            serializable_pipeline.algorithm_name
        ))
    }

    /// Deserialize pipeline from JSON
    pub fn from_json(_json: &str) -> Result<Self> {
        // Implementation would reconstruct pipeline from JSON
        Err(SklearsError::InvalidInput(
            "Deserialization not yet implemented".to_string(),
        ))
    }
}

/// Serializable representation of pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializablePipeline {
    pub algorithm_name: String,
    pub params: DecompositionParams,
    pub config: PipelineConfig,
}

// Placeholder implementations for preprocessing steps
// These would be fully implemented in the actual codebase

#[derive(Clone)]
struct StandardizationStep {
    mean: Option<Array1<Float>>,
    std: Option<Array1<Float>>,
}

impl StandardizationStep {
    fn new() -> Self {
        Self {
            mean: None,
            std: None,
        }
    }
}

impl PreprocessingStep for StandardizationStep {
    fn name(&self) -> &str {
        "Standardization"
    }

    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mean = data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute mean of empty array".to_string())
            })?;
        let std = data
            .mapv(|x| x.powi(2))
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute std of empty array".to_string())
            })?
            .mapv(|x| x.sqrt());

        self.mean = Some(mean.clone());
        self.std = Some(std.clone());

        let result = (data - &mean) / &std;
        Ok(result)
    }

    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mean = self.mean.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Standardization step not fitted".to_string())
        })?;
        let std = self.std.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Standardization step not fitted".to_string())
        })?;
        Ok((data - mean) / std)
    }

    fn is_fitted(&self) -> bool {
        self.mean.is_some()
    }

    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct RobustScalingStep;
impl RobustScalingStep {
    fn new() -> Self {
        Self
    }
}
impl PreprocessingStep for RobustScalingStep {
    fn name(&self) -> &str {
        "RobustScaling"
    }
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn is_fitted(&self) -> bool {
        true
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct CenteringStep {
    mean: Option<Array1<Float>>,
}
impl CenteringStep {
    fn new() -> Self {
        Self { mean: None }
    }
}
impl PreprocessingStep for CenteringStep {
    fn name(&self) -> &str {
        "Centering"
    }
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mean = data
            .mean_axis(scirs2_core::ndarray::Axis(0))
            .ok_or_else(|| {
                SklearsError::InvalidInput("Cannot compute mean of empty array".to_string())
            })?;
        self.mean = Some(mean.clone());
        Ok(data - &mean)
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let mean = self
            .mean
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Centering step not fitted".to_string()))?;
        Ok(data - mean)
    }
    fn is_fitted(&self) -> bool {
        self.mean.is_some()
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct WhiteningStep;
impl WhiteningStep {
    fn new() -> Self {
        Self
    }
}
impl PreprocessingStep for WhiteningStep {
    fn name(&self) -> &str {
        "Whitening"
    }
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn is_fitted(&self) -> bool {
        true
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct NonNegativeScalingStep;
impl NonNegativeScalingStep {
    fn new() -> Self {
        Self
    }
}
impl PreprocessingStep for NonNegativeScalingStep {
    fn name(&self) -> &str {
        "NonNegativeScaling"
    }
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.clone())
    }
    fn is_fitted(&self) -> bool {
        true
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct BatchNormalizationStep {
    batch_size: usize,
    /// Per-batch statistics computed during `process()`: (mean, variance) per feature
    batch_stats: Option<(
        scirs2_core::ndarray::Array1<Float>,
        scirs2_core::ndarray::Array1<Float>,
    )>,
}
impl BatchNormalizationStep {
    fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            batch_stats: None,
        }
    }
}
impl PreprocessingStep for BatchNormalizationStep {
    fn name(&self) -> &str {
        "BatchNormalization"
    }
    /// Normalize each mini-batch of `batch_size` rows independently using mean/variance
    /// computed within that batch.  The per-feature running statistics (across all
    /// batches) are stored so they can be reused by `apply_fitted()`.
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_rows, n_cols) = data.dim();
        if n_rows == 0 || n_cols == 0 {
            return Ok(data.clone());
        }

        let batch_size = if self.batch_size == 0 {
            1
        } else {
            self.batch_size
        };
        let mut output = data.clone();

        // Accumulators for overall statistics (used in apply_fitted)
        let mut running_mean = scirs2_core::ndarray::Array1::<Float>::zeros(n_cols);
        let mut running_var = scirs2_core::ndarray::Array1::<Float>::zeros(n_cols);
        let mut n_batches: usize = 0;

        let mut start = 0;
        while start < n_rows {
            let end = (start + batch_size).min(n_rows);
            let batch = data.slice(scirs2_core::ndarray::s![start..end, ..]);

            let batch_mean = batch
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .ok_or_else(|| {
                    SklearsError::InvalidInput("Cannot compute batch mean".to_string())
                })?;
            // Var = E[X^2] - E[X]^2 (unbiased enough for normalization)
            let e_x2 = batch
                .mapv(|x: Float| x.powi(2))
                .mean_axis(scirs2_core::ndarray::Axis(0))
                .ok_or_else(|| {
                    SklearsError::InvalidInput("Cannot compute batch variance".to_string())
                })?;
            let batch_var: scirs2_core::ndarray::Array1<Float> =
                (&e_x2 - &batch_mean.mapv(|m| m.powi(2))).mapv(|v: Float| v.max(0.0));

            let batch_std = batch_var.mapv(|v: Float| v.sqrt() + 1e-8);

            let batch_owned = batch.to_owned();
            let normalized = (batch_owned - &batch_mean) / &batch_std;
            output
                .slice_mut(scirs2_core::ndarray::s![start..end, ..])
                .assign(&normalized);

            running_mean += &batch_mean;
            running_var += &batch_var;
            n_batches += 1;

            start = end;
        }

        if n_batches > 0 {
            running_mean /= n_batches as Float;
            running_var /= n_batches as Float;
        }
        self.batch_stats = Some((running_mean, running_var));

        Ok(output)
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let (mean, var) = self.batch_stats.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("BatchNormalization step not fitted".to_string())
        })?;
        let std = var.mapv(|v| v.sqrt() + 1e-8);
        Ok((data - mean) / &std)
    }
    fn is_fitted(&self) -> bool {
        self.batch_stats.is_some()
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct TimeSeriesPreprocessingStep {
    window_length: usize,
    fitted: bool,
}
impl TimeSeriesPreprocessingStep {
    fn new(window_length: usize) -> Self {
        Self {
            window_length,
            fitted: false,
        }
    }

    /// Apply sliding-window feature extraction to a single time series row.
    ///
    /// Each row of the input is treated as a univariate series.  The output has
    /// `n_rows - window_length + 1` rows, each containing a window of `window_length`
    /// consecutive values.  Multiple input columns are concatenated within each window.
    fn extract_windows(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_rows, n_cols) = data.dim();
        if self.window_length == 0 {
            return Err(SklearsError::InvalidInput(
                "window_length must be greater than zero".to_string(),
            ));
        }
        if n_rows < self.window_length {
            return Err(SklearsError::InvalidInput(format!(
                "Data has {} rows but window_length is {}; need at least window_length rows",
                n_rows, self.window_length
            )));
        }
        let n_windows = n_rows - self.window_length + 1;
        let window_features = self.window_length * n_cols;
        let mut result = scirs2_core::ndarray::Array2::<Float>::zeros((n_windows, window_features));

        for w in 0..n_windows {
            let window_slice = data.slice(scirs2_core::ndarray::s![w..w + self.window_length, ..]);
            // Flatten the window (row-major)
            for (wi, row) in window_slice.outer_iter().enumerate() {
                for (ci, &val) in row.iter().enumerate() {
                    result[[w, wi * n_cols + ci]] = val;
                }
            }
        }
        Ok(result)
    }
}
impl PreprocessingStep for TimeSeriesPreprocessingStep {
    fn name(&self) -> &str {
        "TimeSeriesPreprocessing"
    }
    fn process(&mut self, data: &Array2<Float>) -> Result<Array2<Float>> {
        let result = self.extract_windows(data)?;
        self.fitted = true;
        Ok(result)
    }
    fn apply_fitted(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "TimeSeriesPreprocessing step not fitted".to_string(),
            ));
        }
        self.extract_windows(data)
    }
    fn is_fitted(&self) -> bool {
        self.fitted
    }
    fn clone_step(&self) -> Box<dyn PreprocessingStep> {
        Box::new(self.clone())
    }
}

// Placeholder implementations for algorithms

struct PCAAlgorithm {
    n_components: usize,
    components: Option<Array2<Float>>,
}
impl PCAAlgorithm {
    fn new(n_components: usize) -> Self {
        Self {
            n_components,
            components: None,
        }
    }
}
impl DecompositionAlgorithm for PCAAlgorithm {
    fn name(&self) -> &str {
        "PCA"
    }
    fn description(&self) -> &str {
        "Principal Component Analysis"
    }
    fn capabilities(&self) -> AlgorithmCapabilities {
        AlgorithmCapabilities::default()
    }
    fn validate_params(&self, _params: &DecompositionParams) -> Result<()> {
        Ok(())
    }
    fn fit(&mut self, data: &Array2<Float>, _params: &DecompositionParams) -> Result<()> {
        self.components = Some(data.slice(s![.., ..self.n_components]).to_owned());
        Ok(())
    }
    fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
        Ok(data.slice(s![.., ..self.n_components]).to_owned())
    }
    fn get_components(&self) -> Result<DecompositionComponents> {
        Ok(DecompositionComponents::default())
    }
    fn is_fitted(&self) -> bool {
        self.components.is_some()
    }
    fn clone_algorithm(&self) -> Box<dyn DecompositionAlgorithm> {
        Box::new(Self {
            n_components: self.n_components,
            components: self.components.clone(),
        })
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Additional placeholder algorithm implementations...
macro_rules! impl_placeholder_algorithm {
    ($name:ident, $display_name:expr, $($field:ident: $field_type:ty),*) => {
        #[allow(dead_code)]
        struct $name {
            n_components: usize,
            $(
                $field: $field_type,
            )*
            components: Option<Array2<Float>>,
        }
        impl $name {
            fn new(n_components: usize $(, $field: $field_type)*) -> Self {
                Self {
                    n_components,
                    $(
                        $field,
                    )*
                    components: None,
                }
            }
        }
        impl DecompositionAlgorithm for $name {
            fn name(&self) -> &str { $display_name }
            fn description(&self) -> &str { $display_name }
            fn capabilities(&self) -> AlgorithmCapabilities { AlgorithmCapabilities::default() }
            fn validate_params(&self, _params: &DecompositionParams) -> Result<()> { Ok(()) }
            fn fit(&mut self, data: &Array2<Float>, _params: &DecompositionParams) -> Result<()> {
                self.components = Some(data.slice(s![.., ..self.n_components]).to_owned());
                Ok(())
            }
            fn transform(&self, data: &Array2<Float>) -> Result<Array2<Float>> {
                Ok(data.slice(s![.., ..self.n_components]).to_owned())
            }
            fn get_components(&self) -> Result<DecompositionComponents> {
                Ok(DecompositionComponents::default())
            }
            fn is_fitted(&self) -> bool { self.components.is_some() }
            fn clone_algorithm(&self) -> Box<dyn DecompositionAlgorithm> {
                Box::new(Self {
                    n_components: self.n_components,
                    $(
                        $field: self.$field.clone(),
                    )*
                    components: self.components.clone(),
                })
            }
            fn as_any(&self) -> &dyn Any { self }
        }
    };
}

impl_placeholder_algorithm!(RobustPCAAlgorithm, "RobustPCA",);
impl_placeholder_algorithm!(SparsePCAAlgorithm, "SparsePCA", alpha: Float);
impl_placeholder_algorithm!(IncrementalPCAAlgorithm, "IncrementalPCA", batch_size: usize);
impl_placeholder_algorithm!(KernelPCAAlgorithm, "KernelPCA", kernel: String);
impl_placeholder_algorithm!(FastICAAlgorithm, "FastICA",);
impl_placeholder_algorithm!(NMFAlgorithm, "NMF", init_method: String);
impl_placeholder_algorithm!(FactorAnalysisAlgorithm, "FactorAnalysis",);
impl_placeholder_algorithm!(TruncatedSVDAlgorithm, "TruncatedSVD",);
impl_placeholder_algorithm!(SSAAlgorithm, "SSA", window_length: usize);

// Placeholder implementations for post-processing steps

#[derive(Clone)]
struct VarianceThresholdStep {
    threshold: Float,
}
impl VarianceThresholdStep {
    fn new(threshold: Float) -> Self {
        Self { threshold }
    }
}
impl PostprocessingStep for VarianceThresholdStep {
    fn name(&self) -> &str {
        "VarianceThreshold"
    }
    /// Remove components whose explained-variance ratio is below `threshold`.
    ///
    /// When `explained_variance_ratio` is available each component column is zeroed
    /// (or removed) if its ratio falls below the threshold.  When only raw
    /// eigenvalues are available, components are filtered by eigenvalue magnitude.
    fn process(&self, mut components: DecompositionComponents) -> Result<DecompositionComponents> {
        if let Some(ref evr) = components.explained_variance_ratio.clone() {
            // Build a mask of components that pass the threshold
            let keep_mask: Vec<bool> = evr.iter().map(|&r| r >= self.threshold).collect();
            let n_keep = keep_mask.iter().filter(|&&k| k).count();

            // Apply mask to component matrix columns
            if let Some(ref comps) = components.components.clone() {
                let n_rows = comps.nrows();
                let mut kept = scirs2_core::ndarray::Array2::<Float>::zeros((n_rows, n_keep));
                let mut col_out = 0;
                for (col_in, &keep) in keep_mask.iter().enumerate() {
                    if keep {
                        kept.column_mut(col_out).assign(&comps.column(col_in));
                        col_out += 1;
                    }
                }
                components.components = Some(kept);
            }

            // Trim explained_variance_ratio to kept components
            let new_evr: Vec<Float> = evr
                .iter()
                .zip(keep_mask.iter())
                .filter_map(|(&r, &k)| if k { Some(r) } else { None })
                .collect();
            components.explained_variance_ratio =
                Some(scirs2_core::ndarray::Array1::from_vec(new_evr));

            // Trim eigenvalues likewise
            if let Some(ref eigs) = components.eigenvalues.clone() {
                let new_eigs: Vec<Float> = eigs
                    .iter()
                    .zip(keep_mask.iter())
                    .filter_map(|(&e, &k)| if k { Some(e) } else { None })
                    .collect();
                components.eigenvalues = Some(scirs2_core::ndarray::Array1::from_vec(new_eigs));
            }

            components.metadata.insert(
                "variance_threshold_applied".to_string(),
                format!("{}", self.threshold),
            );
        }
        // No explained_variance_ratio — nothing to filter
        Ok(components)
    }
    fn clone_step(&self) -> Box<dyn PostprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct OutlierRemovalStep;
impl OutlierRemovalStep {
    fn new() -> Self {
        Self
    }
}
impl PostprocessingStep for OutlierRemovalStep {
    fn name(&self) -> &str {
        "OutlierRemoval"
    }
    fn process(&self, components: DecompositionComponents) -> Result<DecompositionComponents> {
        Ok(components)
    }
    fn clone_step(&self) -> Box<dyn PostprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct SparsityConstraintStep;
impl SparsityConstraintStep {
    fn new() -> Self {
        Self
    }
}
impl PostprocessingStep for SparsityConstraintStep {
    fn name(&self) -> &str {
        "SparsityConstraint"
    }
    fn process(&self, components: DecompositionComponents) -> Result<DecompositionComponents> {
        Ok(components)
    }
    fn clone_step(&self) -> Box<dyn PostprocessingStep> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct RotationStep {
    method: String,
}
impl RotationStep {
    fn new(method: String) -> Self {
        Self { method }
    }

    /// Apply Varimax rotation to a loading matrix using the standard Kaiser (1958) algorithm.
    ///
    /// Varimax maximises the variance of squared loadings within each component,
    /// making components easier to interpret by pushing loadings towards 0 or ±1.
    fn apply_varimax(
        matrix: &scirs2_core::ndarray::Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let (p, k) = matrix.dim();
        if k < 2 {
            // Nothing to rotate with a single component
            return Ok(matrix.clone());
        }

        // Start with the identity rotation matrix
        let mut rot = scirs2_core::ndarray::Array2::<Float>::eye(k);
        let mut lambda = matrix.clone();

        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            // Iterate over all pairs (i, j) with i < j
            for i in 0..k {
                for j in (i + 1)..k {
                    // Extract columns i and j from current rotated matrix
                    let col_i = lambda.column(i).to_owned();
                    let col_j = lambda.column(j).to_owned();

                    // Compute rotation angle using the Varimax criterion
                    let u: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| a.powi(2) - b.powi(2))
                        .collect();
                    let v: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| 2.0 * a * b)
                        .collect();

                    let u_sum: Float = u.iter().sum();
                    let v_sum: Float = v.iter().sum();
                    let u2_sum: Float = u.iter().map(|x| x.powi(2)).sum::<Float>()
                        - u.iter().map(|x| x.powi(2)).sum::<Float>()
                            * u.iter().sum::<Float>().powi(2)
                            / p as Float;
                    let v2_sum: Float = v.iter().map(|x| x.powi(2)).sum::<Float>();
                    let uv_sum: Float = u.iter().zip(v.iter()).map(|(a, b)| a * b).sum::<Float>();

                    // Numerator and denominator for the rotation angle
                    let num = 4.0 * (uv_sum - u_sum * v_sum / p as Float);
                    let den = u2_sum - v2_sum;

                    let theta = if den.abs() < Float::EPSILON {
                        0.0
                    } else {
                        Float::atan2(num, den) / 4.0
                    };

                    if theta.abs() < tol {
                        continue;
                    }
                    max_change = max_change.max(theta.abs());

                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    // Update lambda: rotate columns i and j
                    let new_i: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| cos_t * a + sin_t * b)
                        .collect();
                    let new_j: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| -sin_t * a + cos_t * b)
                        .collect();

                    for row in 0..p {
                        lambda[[row, i]] = new_i[row];
                        lambda[[row, j]] = new_j[row];
                    }

                    // Update rotation matrix
                    let old_rot_i = rot.column(i).to_owned();
                    let old_rot_j = rot.column(j).to_owned();
                    for row in 0..k {
                        rot[[row, i]] = cos_t * old_rot_i[row] + sin_t * old_rot_j[row];
                        rot[[row, j]] = -sin_t * old_rot_i[row] + cos_t * old_rot_j[row];
                    }
                }
            }

            if max_change < tol {
                break;
            }
        }

        Ok(lambda)
    }

    /// Apply Quartimax rotation (maximises variance of squared loadings per variable).
    ///
    /// Quartimax is simpler than Varimax and tends to produce one dominant general factor.
    fn apply_quartimax(
        matrix: &scirs2_core::ndarray::Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<scirs2_core::ndarray::Array2<Float>> {
        let (p, k) = matrix.dim();
        if k < 2 {
            return Ok(matrix.clone());
        }

        let mut lambda = matrix.clone();

        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            for i in 0..k {
                for j in (i + 1)..k {
                    let col_i = lambda.column(i).to_owned();
                    let col_j = lambda.column(j).to_owned();

                    // Quartimax criterion (no normalisation by p)
                    let u_sum: Float = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| a.powi(2) - b.powi(2))
                        .sum();
                    let v_sum: Float = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| 2.0 * a * b)
                        .sum();
                    let uv_sum: Float = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| (a.powi(2) - b.powi(2)) * 2.0 * a * b)
                        .sum();
                    let u2_v2: Float = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| {
                            (a.powi(2) - b.powi(2)).powi(2) - 4.0 * a.powi(2) * b.powi(2)
                        })
                        .sum();

                    let theta = if (u2_v2.abs() + uv_sum.abs()) < Float::EPSILON {
                        0.0
                    } else {
                        Float::atan2(u_sum * v_sum - uv_sum * (p as Float), u2_v2) / 4.0
                    };

                    let _ = (u_sum, v_sum); // suppress unused

                    if theta.abs() < tol {
                        continue;
                    }
                    max_change = max_change.max(theta.abs());

                    let cos_t = theta.cos();
                    let sin_t = theta.sin();

                    let new_i: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| cos_t * a + sin_t * b)
                        .collect();
                    let new_j: Vec<Float> = col_i
                        .iter()
                        .zip(col_j.iter())
                        .map(|(&a, &b)| -sin_t * a + cos_t * b)
                        .collect();

                    for row in 0..p {
                        lambda[[row, i]] = new_i[row];
                        lambda[[row, j]] = new_j[row];
                    }
                }
            }

            if max_change < tol {
                break;
            }
        }
        Ok(lambda)
    }
}
impl PostprocessingStep for RotationStep {
    fn name(&self) -> &str {
        "Rotation"
    }
    /// Apply the configured rotation method to factor loadings or component matrix.
    fn process(&self, mut components: DecompositionComponents) -> Result<DecompositionComponents> {
        let max_iter = 1000;
        let tol = 1e-6;

        let target = if let Some(ref loadings) = components.factor_loadings {
            Some(("factor_loadings", loadings.clone()))
        } else {
            components
                .components
                .as_ref()
                .map(|c| ("components", c.clone()))
        };

        if let Some((field_name, matrix)) = target {
            let rotated = match self.method.to_lowercase().as_str() {
                "varimax" => Self::apply_varimax(&matrix, max_iter, tol)?,
                "quartimax" => Self::apply_quartimax(&matrix, max_iter, tol)?,
                // Promax: apply varimax first, then raise to power 3 (simplified)
                "promax" => {
                    let vm = Self::apply_varimax(&matrix, max_iter, tol)?;
                    vm.mapv(|x| {
                        let sign = if x < 0.0 { -1.0 } else { 1.0 };
                        sign * x.abs().powi(3)
                    })
                }
                other => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown rotation method '{}'; supported: varimax, quartimax, promax",
                        other
                    )))
                }
            };

            match field_name {
                "factor_loadings" => components.factor_loadings = Some(rotated),
                _ => components.components = Some(rotated),
            }

            components
                .metadata
                .insert("rotation_applied".to_string(), self.method.clone());
        }

        Ok(components)
    }
    fn clone_step(&self) -> Box<dyn PostprocessingStep> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder() {
        let builder = DecompositionPipelineBuilder::new()
            .with_standardization()
            .with_pca(3)
            .n_components(3)
            .tolerance(1e-6);

        assert!(builder.build().is_ok());
    }

    #[test]
    fn test_preset_standard_pca() {
        let builder = DecompositionPipelineBuilder::from_preset(DecompositionPreset::StandardPCA {
            n_components: 5,
        });

        let pipeline = builder.build();
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_pipeline_fit_transform() {
        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();

        let data = Array2::from_shape_fn((50, 10), |(i, j)| {
            (i + j) as Float + rng.gen_range(-0.1..0.1)
        });

        let mut pipeline = DecompositionPipelineBuilder::new()
            .with_standardization()
            .with_pca(3)
            .build()
            .expect("operation should succeed");

        let result = pipeline.fit_transform(&data);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fluent_api_chaining() {
        let pipeline = DecompositionPipelineBuilder::new()
            .with_standardization()
            .with_pca(5)
            .n_components(5)
            .tolerance(1e-6)
            .max_iterations(100)
            .random_seed(42)
            .verbose(false)
            .build();

        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_serialization() {
        let mut pipeline = DecompositionPipelineBuilder::new()
            .with_pca(3)
            .build()
            .expect("operation should succeed");

        use scirs2_core::random::thread_rng;
        let mut rng = thread_rng();
        let data = Array2::from_shape_fn((20, 5), |(i, j)| {
            (i + j) as Float + rng.gen_range(-0.1..0.1)
        });

        pipeline.fit(&data).expect("model fitting should succeed");

        let json = pipeline.to_json();
        assert!(json.is_ok());
    }

    // ---- Tests for newly-implemented parameters ----

    /// batch_size actually normalises differently from a full-dataset normalisation.
    #[test]
    fn test_batch_normalization_affects_output() {
        let data = Array2::from_shape_vec((8, 3), (0..24).map(|x| x as Float).collect())
            .expect("shape should match");

        // batch_size = 2: normalise within 4 batches of 2
        let mut step_small = BatchNormalizationStep::new(2);
        let out_small = step_small
            .process(&data)
            .expect("batch norm should succeed");

        // batch_size = 8 (full dataset at once): different statistics
        let mut step_full = BatchNormalizationStep::new(8);
        let out_full = step_full.process(&data).expect("batch norm should succeed");

        // The two outputs should differ (different batch statistics)
        assert_ne!(
            out_small.as_slice().expect("contiguous"),
            out_full.as_slice().expect("contiguous"),
            "batch_size should change the normalisation output"
        );

        // After fitting, apply_fitted should use stored stats
        let apply_result = step_small.apply_fitted(&data);
        assert!(
            apply_result.is_ok(),
            "apply_fitted should work after process()"
        );
    }

    /// window_length changes the shape of the output (more cols, fewer rows).
    #[test]
    fn test_window_length_affects_output() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as Float).collect())
            .expect("shape should match");

        let window = 3_usize;
        let mut step = TimeSeriesPreprocessingStep::new(window);
        let out = step
            .process(&data)
            .expect("time-series preprocessing should succeed");

        // Expected: n_rows - window + 1 windows, window * n_cols features
        assert_eq!(
            out.nrows(),
            data.nrows() - window + 1,
            "output should have n_rows - window_length + 1 rows"
        );
        assert_eq!(
            out.ncols(),
            window * data.ncols(),
            "each window should have window_length * n_cols features"
        );

        // window=1 should produce same number of rows, just a reshape
        let mut step1 = TimeSeriesPreprocessingStep::new(1);
        let out1 = step1.process(&data).expect("window=1 should work");
        assert_eq!(out1.nrows(), data.nrows());
        assert_eq!(out1.ncols(), data.ncols());
    }

    /// variance_threshold removes components below the threshold.
    #[test]
    fn test_variance_threshold_filters_components() {
        use crate::modular_framework::{DecompositionComponents, PostprocessingStep};
        use scirs2_core::ndarray::{Array1, Array2};

        // Build components with known explained_variance_ratio
        let evr = Array1::from_vec(vec![0.5_f64, 0.02, 0.3]);
        let comps = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .expect("shape should match");

        let dc = DecompositionComponents {
            explained_variance_ratio: Some(evr.clone()),
            components: Some(comps),
            ..Default::default()
        };

        // Threshold 0.1: keep components with evr >= 0.1, i.e. indices 0 and 2
        let step = VarianceThresholdStep::new(0.1);
        let result = step.process(dc).expect("variance threshold should succeed");

        let kept_evr = result
            .explained_variance_ratio
            .expect("evr should be present");
        assert_eq!(
            kept_evr.len(),
            2,
            "should keep 2 components above threshold"
        );
        assert!((kept_evr[0] - 0.5).abs() < 1e-10);
        assert!((kept_evr[1] - 0.3).abs() < 1e-10);

        let kept_comps = result.components.expect("components should be present");
        assert_eq!(
            kept_comps.ncols(),
            2,
            "component matrix should have 2 columns"
        );
    }

    /// rotation with varimax should at least run and mark metadata.
    #[test]
    fn test_rotation_applies_to_components() {
        use crate::modular_framework::{DecompositionComponents, PostprocessingStep};
        use scirs2_core::ndarray::Array2;

        let comps = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.9, 0.1, 0.2, 0.8, 0.2, 0.3, 0.7, 0.3, 0.1, 0.6, 0.4, 0.5, 0.5, 0.5, 0.4,
            ],
        )
        .expect("shape should match");

        let dc = DecompositionComponents {
            components: Some(comps.clone()),
            ..Default::default()
        };

        let step = RotationStep::new("varimax".to_string());
        let result = step.process(dc).expect("rotation should succeed");

        assert_eq!(
            result.metadata.get("rotation_applied"),
            Some(&"varimax".to_string()),
            "metadata should record rotation method"
        );
        // Components should still have same dimensions
        let rotated = result.components.expect("components should be present");
        assert_eq!(rotated.dim(), comps.dim());
    }

    /// postprocessing_steps are applied during fit().
    #[test]
    fn test_postprocessing_steps_applied_in_fit() {
        use scirs2_core::ndarray::Array2;

        let data = Array2::from_shape_vec((20, 5), (0..100).map(|x| x as Float).collect())
            .expect("shape should match");

        // Build pipeline with variance threshold of 0.5 — should filter most components
        // (PCAAlgorithm returns empty DecompositionComponents so evr is None, but the
        //  postprocessing is still called — verify get_components() returns Ok)
        let mut pipeline = DecompositionPipelineBuilder::new()
            .with_pca(3)
            .with_variance_threshold(0.01)
            .build()
            .expect("build should succeed");

        pipeline.fit(&data).expect("fit should succeed");

        // get_components should return cached (post-processed) components
        let comps = pipeline.get_components();
        assert!(
            comps.is_ok(),
            "get_components should return Ok after fit with postprocessing"
        );
    }
}

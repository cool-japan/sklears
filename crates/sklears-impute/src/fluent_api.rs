//! Fluent API and builder patterns for easy imputation configuration
//!
//! This module provides a convenient, chainable API for configuring and using
//! imputation methods with sensible defaults and validation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::collections::HashMap;

use crate::{parallel::ParallelConfig, KNNImputer, ParallelKNNImputer, SimpleImputer};

/// Fluent API builder for imputation pipelines
#[derive(Debug, Clone)]
pub struct ImputationBuilder {
    method: ImputationMethod,
    validation: ValidationConfig,
    preprocessing: PreprocessingConfig,
    postprocessing: PostprocessingConfig,
    parallel_config: Option<ParallelConfig>,
}

/// Available imputation methods with their configurations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ImputationMethod {
    /// Simple
    Simple(SimpleImputationConfig),
    /// KNN
    KNN(KNNImputationConfig),
    /// Iterative
    Iterative(IterativeImputationConfig),
    /// GaussianProcess
    GaussianProcess(GaussianProcessConfig),
    /// MatrixFactorization
    MatrixFactorization(MatrixFactorizationConfig),
    /// Bayesian
    Bayesian(BayesianImputationConfig),
    /// Ensemble
    Ensemble(EnsembleImputationConfig),
    /// DeepLearning
    DeepLearning(DeepLearningConfig),
    /// Custom
    Custom(CustomImputationConfig),
}

/// Configuration for simple imputation methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SimpleImputationConfig {
    /// strategy
    pub strategy: String,
    /// fill_value
    pub fill_value: Option<f64>,
    /// copy
    pub copy: bool,
}

/// Configuration for KNN imputation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct KNNImputationConfig {
    /// n_neighbors
    pub n_neighbors: usize,
    /// weights
    pub weights: String,
    /// metric
    pub metric: String,
    /// add_indicator
    pub add_indicator: bool,
}

/// Configuration for iterative imputation (MICE)
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IterativeImputationConfig {
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
    /// n_nearest_features
    pub n_nearest_features: Option<usize>,
    /// sample_posterior
    pub sample_posterior: bool,
    /// random_state
    pub random_state: Option<u64>,
}

/// Configuration for Gaussian Process imputation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct GaussianProcessConfig {
    /// kernel
    pub kernel: String,
    /// alpha
    pub alpha: f64,
    /// n_restarts_optimizer
    pub n_restarts_optimizer: usize,
    /// normalize_y
    pub normalize_y: bool,
    /// random_state
    pub random_state: Option<u64>,
}

/// Configuration for Matrix Factorization imputation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct MatrixFactorizationConfig {
    /// n_components
    pub n_components: usize,
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
    /// regularization
    pub regularization: f64,
    /// random_state
    pub random_state: Option<u64>,
}

/// Configuration for Bayesian imputation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct BayesianImputationConfig {
    /// n_imputations
    pub n_imputations: usize,
    /// max_iter
    pub max_iter: usize,
    /// burn_in
    pub burn_in: usize,
    /// prior_variance
    pub prior_variance: f64,
    /// random_state
    pub random_state: Option<u64>,
}

/// Configuration for ensemble methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct EnsembleImputationConfig {
    /// method
    pub method: String, // "random_forest", "gradient_boosting", etc.
    /// n_estimators
    pub n_estimators: usize,
    /// max_depth
    pub max_depth: Option<usize>,
    /// random_state
    pub random_state: Option<u64>,
}

/// Configuration for deep learning methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DeepLearningConfig {
    /// method
    pub method: String, // "autoencoder", "vae", "gan"
    /// hidden_dims
    pub hidden_dims: Vec<usize>,
    /// learning_rate
    pub learning_rate: f64,
    /// epochs
    pub epochs: usize,
    /// batch_size
    pub batch_size: usize,
    /// device
    pub device: String, // "cpu", "cuda"
}

/// Configuration for custom imputation methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct CustomImputationConfig {
    /// name
    pub name: String,
    #[cfg(feature = "serde")]
    pub parameters: HashMap<String, serde_json::Value>,
    #[cfg(not(feature = "serde"))]
    pub parameters: HashMap<String, String>,
}

/// Validation configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationConfig {
    /// cross_validation
    pub cross_validation: bool,
    /// cv_folds
    pub cv_folds: usize,
    /// holdout_fraction
    pub holdout_fraction: Option<f64>,
    /// metrics
    pub metrics: Vec<String>,
    /// synthetic_missing_patterns
    pub synthetic_missing_patterns: Vec<String>,
}

/// Preprocessing configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PreprocessingConfig {
    /// normalize
    pub normalize: bool,
    /// scale
    pub scale: bool,
    /// remove_constant_features
    pub remove_constant_features: bool,
    /// handle_outliers
    pub handle_outliers: bool,
    /// outlier_method
    pub outlier_method: String,
}

/// Postprocessing configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PostprocessingConfig {
    /// clip_values
    pub clip_values: Option<(f64, f64)>,
    /// round_integers
    pub round_integers: bool,
    /// preserve_dtypes
    pub preserve_dtypes: bool,
    /// add_uncertainty_estimates
    pub add_uncertainty_estimates: bool,
}

/// Predefined configuration presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImputationPreset {
    /// Fast
    Fast,
    /// Balanced
    Balanced,
    /// HighQuality
    HighQuality,
    /// Memory
    Memory,
    /// Parallel
    Parallel,
    /// Academic
    Academic,
    /// Production
    Production,
}

impl Default for ImputationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImputationBuilder {
    /// Create a new imputation builder with default settings
    pub fn new() -> Self {
        Self {
            method: ImputationMethod::Simple(SimpleImputationConfig {
                strategy: "mean".to_string(),
                fill_value: None,
                copy: true,
            }),
            validation: ValidationConfig {
                cross_validation: false,
                cv_folds: 5,
                holdout_fraction: None,
                metrics: vec!["rmse".to_string()],
                synthetic_missing_patterns: vec!["mcar".to_string()],
            },
            preprocessing: PreprocessingConfig {
                normalize: false,
                scale: false,
                remove_constant_features: false,
                handle_outliers: false,
                outlier_method: "iqr".to_string(),
            },
            postprocessing: PostprocessingConfig {
                clip_values: None,
                round_integers: false,
                preserve_dtypes: true,
                add_uncertainty_estimates: false,
            },
            parallel_config: None,
        }
    }

    /// Apply a predefined configuration preset
    pub fn preset(mut self, preset: ImputationPreset) -> Self {
        match preset {
            ImputationPreset::Fast => {
                self.method = ImputationMethod::Simple(SimpleImputationConfig {
                    strategy: "mean".to_string(),
                    fill_value: None,
                    copy: true,
                });
            }
            ImputationPreset::Balanced => {
                self.method = ImputationMethod::KNN(KNNImputationConfig {
                    n_neighbors: 5,
                    weights: "uniform".to_string(),
                    metric: "euclidean".to_string(),
                    add_indicator: false,
                });
            }
            ImputationPreset::HighQuality => {
                self.method = ImputationMethod::Iterative(IterativeImputationConfig {
                    max_iter: 10,
                    tol: 1e-3,
                    n_nearest_features: None,
                    sample_posterior: true,
                    random_state: None,
                });
                self.validation.cross_validation = true;
                self.postprocessing.add_uncertainty_estimates = true;
            }
            ImputationPreset::Memory => {
                self.method = ImputationMethod::Simple(SimpleImputationConfig {
                    strategy: "median".to_string(),
                    fill_value: None,
                    copy: false,
                });
                self.preprocessing.remove_constant_features = true;
            }
            ImputationPreset::Parallel => {
                self.method = ImputationMethod::KNN(KNNImputationConfig {
                    n_neighbors: 3,
                    weights: "distance".to_string(),
                    metric: "euclidean".to_string(),
                    add_indicator: false,
                });
                self.parallel_config = Some(ParallelConfig::default());
            }
            ImputationPreset::Academic => {
                self.method = ImputationMethod::Bayesian(BayesianImputationConfig {
                    n_imputations: 5,
                    max_iter: 100,
                    burn_in: 20,
                    prior_variance: 1.0,
                    random_state: Some(42),
                });
                self.validation.cross_validation = true;
                self.validation.cv_folds = 10;
                self.validation.metrics = vec![
                    "rmse".to_string(),
                    "mae".to_string(),
                    "bias".to_string(),
                    "coverage".to_string(),
                ];
                self.postprocessing.add_uncertainty_estimates = true;
            }
            ImputationPreset::Production => {
                self.method = ImputationMethod::Ensemble(EnsembleImputationConfig {
                    method: "random_forest".to_string(),
                    n_estimators: 100,
                    max_depth: Some(10),
                    random_state: Some(42),
                });
                self.validation.cross_validation = true;
                self.preprocessing.handle_outliers = true;
                self.postprocessing.preserve_dtypes = true;
            }
        }
        self
    }

    /// Configure simple imputation
    pub fn simple(self) -> SimpleImputationBuilder {
        SimpleImputationBuilder::new(self)
    }

    /// Configure KNN imputation
    pub fn knn(self) -> KNNImputationBuilder {
        KNNImputationBuilder::new(self)
    }

    /// Configure iterative imputation
    pub fn iterative(self) -> IterativeImputationBuilder {
        IterativeImputationBuilder::new(self)
    }

    /// Configure Gaussian Process imputation
    pub fn gaussian_process(self) -> GaussianProcessBuilder {
        GaussianProcessBuilder::new(self)
    }

    /// Configure ensemble imputation
    pub fn ensemble(self) -> EnsembleImputationBuilder {
        EnsembleImputationBuilder::new(self)
    }

    /// Configure deep learning imputation
    pub fn deep_learning(self) -> DeepLearningBuilder {
        DeepLearningBuilder::new(self)
    }

    /// Enable parallel processing
    pub fn parallel(mut self, config: Option<ParallelConfig>) -> Self {
        self.parallel_config = config.or_else(|| Some(ParallelConfig::default()));
        self
    }

    /// Configure validation
    pub fn validation(mut self, config: ValidationConfig) -> Self {
        self.validation = config;
        self
    }

    /// Enable cross-validation
    pub fn cross_validate(mut self, folds: usize) -> Self {
        self.validation.cross_validation = true;
        self.validation.cv_folds = folds;
        self
    }

    /// Configure preprocessing
    pub fn preprocessing(mut self, config: PreprocessingConfig) -> Self {
        self.preprocessing = config;
        self
    }

    /// Enable normalization
    pub fn normalize(mut self) -> Self {
        self.preprocessing.normalize = true;
        self
    }

    /// Enable scaling
    pub fn scale(mut self) -> Self {
        self.preprocessing.scale = true;
        self
    }

    /// Configure postprocessing
    pub fn postprocessing(mut self, config: PostprocessingConfig) -> Self {
        self.postprocessing = config;
        self
    }

    /// Enable uncertainty estimation
    pub fn with_uncertainty(mut self) -> Self {
        self.postprocessing.add_uncertainty_estimates = true;
        self
    }

    /// Build the imputation pipeline
    pub fn build(self) -> SklResult<ImputationPipeline> {
        ImputationPipeline::new(
            self.method,
            self.validation,
            self.preprocessing,
            self.postprocessing,
            self.parallel_config,
        )
    }
}

/// Builder for simple imputation configuration
pub struct SimpleImputationBuilder {
    builder: ImputationBuilder,
    config: SimpleImputationConfig,
}

impl SimpleImputationBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: SimpleImputationConfig {
                strategy: "mean".to_string(),
                fill_value: None,
                copy: true,
            },
        }
    }

    pub fn strategy(mut self, strategy: &str) -> Self {
        self.config.strategy = strategy.to_string();
        self
    }

    pub fn mean(mut self) -> Self {
        self.config.strategy = "mean".to_string();
        self
    }

    pub fn median(mut self) -> Self {
        self.config.strategy = "median".to_string();
        self
    }

    pub fn mode(mut self) -> Self {
        self.config.strategy = "most_frequent".to_string();
        self
    }

    pub fn constant(mut self, value: f64) -> Self {
        self.config.strategy = "constant".to_string();
        self.config.fill_value = Some(value);
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::Simple(self.config);
        self.builder
    }
}

/// Builder for KNN imputation configuration
pub struct KNNImputationBuilder {
    builder: ImputationBuilder,
    config: KNNImputationConfig,
}

impl KNNImputationBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: KNNImputationConfig {
                n_neighbors: 5,
                weights: "uniform".to_string(),
                metric: "euclidean".to_string(),
                add_indicator: false,
            },
        }
    }

    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.config.n_neighbors = n_neighbors;
        self
    }

    pub fn weights(mut self, weights: &str) -> Self {
        self.config.weights = weights.to_string();
        self
    }

    pub fn uniform_weights(mut self) -> Self {
        self.config.weights = "uniform".to_string();
        self
    }

    pub fn distance_weights(mut self) -> Self {
        self.config.weights = "distance".to_string();
        self
    }

    pub fn metric(mut self, metric: &str) -> Self {
        self.config.metric = metric.to_string();
        self
    }

    pub fn euclidean(mut self) -> Self {
        self.config.metric = "euclidean".to_string();
        self
    }

    pub fn manhattan(mut self) -> Self {
        self.config.metric = "manhattan".to_string();
        self
    }

    pub fn add_indicator(mut self, add_indicator: bool) -> Self {
        self.config.add_indicator = add_indicator;
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::KNN(self.config);
        self.builder
    }
}

/// Builder for iterative imputation configuration
pub struct IterativeImputationBuilder {
    builder: ImputationBuilder,
    config: IterativeImputationConfig,
}

impl IterativeImputationBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: IterativeImputationConfig {
                max_iter: 10,
                tol: 1e-3,
                n_nearest_features: None,
                sample_posterior: false,
                random_state: None,
            },
        }
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    pub fn n_nearest_features(mut self, n_features: usize) -> Self {
        self.config.n_nearest_features = Some(n_features);
        self
    }

    pub fn sample_posterior(mut self, sample: bool) -> Self {
        self.config.sample_posterior = sample;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::Iterative(self.config);
        self.builder
    }
}

/// Builder for Gaussian Process imputation configuration
pub struct GaussianProcessBuilder {
    builder: ImputationBuilder,
    config: GaussianProcessConfig,
}

impl GaussianProcessBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: GaussianProcessConfig {
                kernel: "rbf".to_string(),
                alpha: 1e-6,
                n_restarts_optimizer: 0,
                normalize_y: false,
                random_state: None,
            },
        }
    }

    pub fn kernel(mut self, kernel: &str) -> Self {
        self.config.kernel = kernel.to_string();
        self
    }

    pub fn rbf_kernel(mut self) -> Self {
        self.config.kernel = "rbf".to_string();
        self
    }

    pub fn matern_kernel(mut self) -> Self {
        self.config.kernel = "matern".to_string();
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn n_restarts(mut self, n_restarts: usize) -> Self {
        self.config.n_restarts_optimizer = n_restarts;
        self
    }

    pub fn normalize_y(mut self, normalize: bool) -> Self {
        self.config.normalize_y = normalize;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::GaussianProcess(self.config);
        self.builder
    }
}

/// Builder for ensemble imputation configuration
pub struct EnsembleImputationBuilder {
    builder: ImputationBuilder,
    config: EnsembleImputationConfig,
}

impl EnsembleImputationBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: EnsembleImputationConfig {
                method: "random_forest".to_string(),
                n_estimators: 100,
                max_depth: None,
                random_state: None,
            },
        }
    }

    pub fn random_forest(mut self) -> Self {
        self.config.method = "random_forest".to_string();
        self
    }

    pub fn gradient_boosting(mut self) -> Self {
        self.config.method = "gradient_boosting".to_string();
        self
    }

    pub fn extra_trees(mut self) -> Self {
        self.config.method = "extra_trees".to_string();
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::Ensemble(self.config);
        self.builder
    }
}

/// Builder for deep learning imputation configuration
pub struct DeepLearningBuilder {
    builder: ImputationBuilder,
    config: DeepLearningConfig,
}

impl DeepLearningBuilder {
    fn new(builder: ImputationBuilder) -> Self {
        Self {
            builder,
            config: DeepLearningConfig {
                method: "autoencoder".to_string(),
                hidden_dims: vec![128, 64, 32],
                learning_rate: 0.001,
                epochs: 100,
                batch_size: 32,
                device: "cpu".to_string(),
            },
        }
    }

    pub fn autoencoder(mut self) -> Self {
        self.config.method = "autoencoder".to_string();
        self
    }

    pub fn vae(mut self) -> Self {
        self.config.method = "vae".to_string();
        self
    }

    pub fn gan(mut self) -> Self {
        self.config.method = "gan".to_string();
        self
    }

    pub fn hidden_dims(mut self, dims: Vec<usize>) -> Self {
        self.config.hidden_dims = dims;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.config.epochs = epochs;
        self
    }

    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    pub fn device(mut self, device: &str) -> Self {
        self.config.device = device.to_string();
        self
    }

    pub fn finish(mut self) -> ImputationBuilder {
        self.builder.method = ImputationMethod::DeepLearning(self.config);
        self.builder
    }
}

/// Main imputation pipeline that handles the complete workflow
pub struct ImputationPipeline {
    method: ImputationMethod,
    validation: ValidationConfig,
    preprocessing: PreprocessingConfig,
    postprocessing: PostprocessingConfig,
    parallel_config: Option<ParallelConfig>,
}

impl ImputationPipeline {
    fn new(
        method: ImputationMethod,
        validation: ValidationConfig,
        preprocessing: PreprocessingConfig,
        postprocessing: PostprocessingConfig,
        parallel_config: Option<ParallelConfig>,
    ) -> SklResult<Self> {
        Ok(Self {
            method,
            validation,
            preprocessing,
            postprocessing,
            parallel_config,
        })
    }

    /// Fit and transform in one step
    pub fn fit_transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // TODO: Implement the complete pipeline with preprocessing, imputation, and postprocessing
        // This is a simplified implementation for now

        match &self.method {
            ImputationMethod::Simple(config) => {
                let imputer = SimpleImputer::new().strategy(config.strategy.clone());
                let fitted = imputer.fit(X, &())?;
                fitted.transform(X)
            }
            ImputationMethod::KNN(config) => {
                if let Some(parallel_config) = &self.parallel_config {
                    let imputer = ParallelKNNImputer::new()
                        .n_neighbors(config.n_neighbors)
                        .weights(config.weights.clone())
                        .metric(config.metric.clone())
                        .parallel_config(parallel_config.clone());
                    let fitted = imputer.fit(X, &())?;
                    fitted.transform(X)
                } else {
                    let imputer = KNNImputer::new()
                        .n_neighbors(config.n_neighbors)
                        .weights(config.weights.clone())
                        .metric(config.metric.clone());
                    let fitted = imputer.fit(X, &())?;
                    fitted.transform(X)
                }
            }
            _ => {
                // For other methods, fall back to simple imputation for now
                let imputer = SimpleImputer::new().strategy("mean".to_string());
                let fitted = imputer.fit(X, &())?;
                fitted.transform(X)
            }
        }
    }

    /// Get configuration as JSON
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> SklResult<String> {
        #[derive(serde::Serialize)]
        struct PipelineConfig<'a> {
            method: &'a ImputationMethod,
            validation: &'a ValidationConfig,
            preprocessing: &'a PreprocessingConfig,
            postprocessing: &'a PostprocessingConfig,
            parallel_config: &'a Option<ParallelConfig>,
        }

        let config = PipelineConfig {
            method: &self.method,
            validation: &self.validation,
            preprocessing: &self.preprocessing,
            postprocessing: &self.postprocessing,
            parallel_config: &self.parallel_config,
        };

        serde_json::to_string_pretty(&config).map_err(|e| {
            SklearsError::SerializationError(format!("Failed to serialize config: {}", e))
        })
    }

    /// Get configuration as JSON (disabled without serde feature)
    #[cfg(not(feature = "serde"))]
    pub fn to_json(&self) -> SklResult<String> {
        Err(SklearsError::NotImplemented(
            "to_json requires serde feature".to_string(),
        ))
    }

    /// Load configuration from JSON
    pub fn from_json(json: &str) -> SklResult<Self> {
        // This would need a proper deserialization implementation
        Err(SklearsError::NotImplemented(
            "from_json not yet implemented".to_string(),
        ))
    }
}

/// Convenience functions for quick imputation
pub mod quick {
    use super::*;

    /// Quick mean imputation
    pub fn mean_impute(X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        ImputationBuilder::new()
            .simple()
            .mean()
            .finish()
            .build()?
            .fit_transform(X)
    }

    /// Quick KNN imputation
    pub fn knn_impute(X: &ArrayView2<'_, Float>, n_neighbors: usize) -> SklResult<Array2<Float>> {
        ImputationBuilder::new()
            .knn()
            .n_neighbors(n_neighbors)
            .finish()
            .build()?
            .fit_transform(X)
    }

    /// Quick parallel KNN imputation
    pub fn parallel_knn_impute(
        X: &ArrayView2<'_, Float>,
        n_neighbors: usize,
    ) -> SklResult<Array2<Float>> {
        ImputationBuilder::new()
            .knn()
            .n_neighbors(n_neighbors)
            .finish()
            .parallel(None)
            .build()?
            .fit_transform(X)
    }

    /// Quick iterative imputation
    pub fn iterative_impute(X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        ImputationBuilder::new()
            .iterative()
            .max_iter(10)
            .finish()
            .build()?
            .fit_transform(X)
    }

    /// Quick high-quality imputation with validation
    pub fn high_quality_impute(X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        ImputationBuilder::new()
            .preset(ImputationPreset::HighQuality)
            .build()?
            .fit_transform(X)
    }
}

/// Trait-based pluggable architecture for imputation modules
pub mod pluggable {
    use super::*;

    /// Core trait for all imputation modules
    pub trait ImputationModule: Send + Sync {
        /// Get the name of this imputation module
        fn name(&self) -> &str;

        /// Get the version of this module
        fn version(&self) -> &str;

        /// Check if this module can handle the given data characteristics
        fn can_handle(&self, data_info: &DataCharacteristics) -> bool;

        /// Get module-specific configuration schema
        fn config_schema(&self) -> ModuleConfigSchema;

        /// Create an instance of this module with given configuration
        fn create_instance(&self, config: &ModuleConfig) -> SklResult<Box<dyn ImputationInstance>>;

        /// Get module dependencies
        fn dependencies(&self) -> Vec<&str> {
            vec![]
        }

        /// Get module priority (higher = preferred)
        fn priority(&self) -> i32 {
            0
        }
    }

    /// Trait for actual imputation instances
    pub trait ImputationInstance: Send + Sync {
        /// Fit the imputation model
        fn fit(&mut self, X: &ArrayView2<Float>) -> SklResult<()>;

        /// Transform data using the fitted model
        fn transform(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>>;

        /// Fit and transform in one step
        fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            self.fit(X)?;
            self.transform(X)
        }

        /// Get uncertainty estimates if supported
        fn transform_with_uncertainty(
            &self,
            X: &ArrayView2<Float>,
        ) -> SklResult<(Array2<Float>, Option<Array2<Float>>)> {
            let result = self.transform(X)?;
            Ok((result, None))
        }

        /// Check if this instance supports uncertainty quantification
        fn supports_uncertainty(&self) -> bool {
            false
        }

        /// Get feature importance if supported
        fn feature_importance(&self) -> Option<Array1<Float>> {
            None
        }

        /// Partial fit for streaming data
        fn partial_fit(&mut self, X: &ArrayView2<Float>) -> SklResult<()> {
            Err(SklearsError::NotImplemented(
                "Partial fit not supported".to_string(),
            ))
        }

        /// Check if partial fit is supported
        fn supports_partial_fit(&self) -> bool {
            false
        }
    }

    /// Data characteristics for module selection
    #[derive(Debug, Clone)]
    pub struct DataCharacteristics {
        /// n_samples
        pub n_samples: usize,
        /// n_features
        pub n_features: usize,
        /// missing_rate
        pub missing_rate: f64,
        /// missing_pattern
        pub missing_pattern: MissingPatternType,
        /// data_types
        pub data_types: Vec<DataType>,
        /// has_categorical
        pub has_categorical: bool,
        /// has_temporal
        pub has_temporal: bool,
        /// is_sparse
        pub is_sparse: bool,
        /// memory_constraints
        pub memory_constraints: Option<usize>, // Max memory in bytes
    }

    /// Missing pattern types
    #[derive(Debug, Clone, PartialEq)]
    pub enum MissingPatternType {
        /// MCAR
        MCAR,
        /// MAR
        MAR,
        /// MNAR
        MNAR,
        /// Unknown
        Unknown,
        /// Block
        Block,
        /// Monotone
        Monotone,
    }

    /// Data types for features
    #[derive(Debug, Clone, PartialEq)]
    pub enum DataType {
        /// Continuous
        Continuous,
        /// Categorical
        Categorical,
        /// Ordinal
        Ordinal,
        /// Binary
        Binary,
        /// Count
        Count,
        /// Temporal
        Temporal,
        /// Text
        Text,
    }

    /// Module configuration schema
    #[derive(Debug, Clone)]
    pub struct ModuleConfigSchema {
        /// parameters
        pub parameters: HashMap<String, ParameterSchema>,
        /// required_parameters
        pub required_parameters: Vec<String>,
        /// parameter_groups
        pub parameter_groups: Vec<ParameterGroup>,
    }

    /// Parameter schema definition
    #[derive(Debug, Clone)]
    pub struct ParameterSchema {
        /// name
        pub name: String,
        /// parameter_type
        pub parameter_type: ParameterType,
        #[cfg(feature = "serde")]
        pub default_value: Option<serde_json::Value>,
        #[cfg(not(feature = "serde"))]
        pub default_value: Option<String>,
        /// valid_range
        pub valid_range: Option<ParameterRange>,
        /// description
        pub description: String,
        /// dependencies
        pub dependencies: Vec<String>,
    }

    /// Parameter types
    #[derive(Debug, Clone)]
    pub enum ParameterType {
        /// Integer
        Integer,
        /// Float
        Float,
        /// String
        String,
        /// Boolean
        Boolean,
        /// Array
        Array(Box<ParameterType>),
        /// Enum
        Enum(Vec<String>),
        /// Object
        Object(HashMap<String, ParameterType>),
    }

    /// Parameter value ranges
    #[derive(Debug, Clone)]
    pub enum ParameterRange {
        /// IntRange
        IntRange { min: Option<i64>, max: Option<i64> },
        /// FloatRange
        FloatRange { min: Option<f64>, max: Option<f64> },
        /// StringPattern
        StringPattern(String), // regex pattern
        /// ArrayLength
        ArrayLength {
            min: Option<usize>,
            max: Option<usize>,
        },
    }

    /// Parameter groups for UI organization
    #[derive(Debug, Clone)]
    pub struct ParameterGroup {
        /// name
        pub name: String,
        /// description
        pub description: String,
        /// parameters
        pub parameters: Vec<String>,
        /// optional
        pub optional: bool,
    }

    /// Module configuration
    #[derive(Debug, Clone)]
    pub struct ModuleConfig {
        #[cfg(feature = "serde")]
        pub parameters: HashMap<String, serde_json::Value>,
        #[cfg(not(feature = "serde"))]
        pub parameters: HashMap<String, String>,
    }

    /// Registry for managing imputation modules
    pub struct ModuleRegistry {
        modules: HashMap<String, Box<dyn ImputationModule>>,
        aliases: HashMap<String, String>,
    }

    impl ModuleRegistry {
        pub fn new() -> Self {
            Self {
                modules: HashMap::new(),
                aliases: HashMap::new(),
            }
        }

        /// Register a new imputation module
        pub fn register_module(&mut self, module: Box<dyn ImputationModule>) -> SklResult<()> {
            let name = module.name().to_string();
            if self.modules.contains_key(&name) {
                return Err(SklearsError::InvalidInput(format!(
                    "Module '{}' already registered",
                    name
                )));
            }
            self.modules.insert(name, module);
            Ok(())
        }

        /// Register an alias for a module
        pub fn register_alias(&mut self, alias: String, module_name: String) -> SklResult<()> {
            if !self.modules.contains_key(&module_name) {
                return Err(SklearsError::InvalidInput(format!(
                    "Module '{}' not found",
                    module_name
                )));
            }
            self.aliases.insert(alias, module_name);
            Ok(())
        }

        /// Get a module by name or alias
        pub fn get_module(&self, name: &str) -> Option<&dyn ImputationModule> {
            if let Some(actual_name) = self.aliases.get(name) {
                self.modules.get(actual_name).map(|m| m.as_ref())
            } else {
                self.modules.get(name).map(|m| m.as_ref())
            }
        }

        /// List all available modules
        pub fn list_modules(&self) -> Vec<&str> {
            self.modules.keys().map(|s| s.as_str()).collect()
        }

        /// Find suitable modules for given data characteristics
        pub fn find_suitable_modules(
            &self,
            data_info: &DataCharacteristics,
        ) -> Vec<&dyn ImputationModule> {
            let mut suitable: Vec<_> = self
                .modules
                .values()
                .filter(|m| m.can_handle(data_info))
                .map(|m| m.as_ref())
                .collect();

            // Sort by priority (descending)
            suitable.sort_by(|a, b| b.priority().cmp(&a.priority()));
            suitable
        }

        /// Get recommended module for data characteristics
        pub fn recommend_module(
            &self,
            data_info: &DataCharacteristics,
        ) -> Option<&dyn ImputationModule> {
            self.find_suitable_modules(data_info).into_iter().next()
        }
    }

    /// Pipeline composer for combining multiple modules
    pub struct PipelineComposer {
        stages: Vec<PipelineStage>,
        registry: ModuleRegistry,
    }

    /// A stage in the imputation pipeline
    #[derive(Debug, Clone)]
    pub struct PipelineStage {
        /// name
        pub name: String,
        /// module_name
        pub module_name: String,
        /// config
        pub config: ModuleConfig,
        /// condition
        pub condition: Option<StageCondition>,
    }

    /// Conditions for pipeline stage execution
    #[derive(Debug, Clone)]
    pub enum StageCondition {
        /// MissingRate
        MissingRate(f64), // Execute only if missing rate > threshold
        /// FeatureCount
        FeatureCount(usize), // Execute only if n_features > threshold
        /// DataType
        DataType(DataType), // Execute only for specific data types
        /// Custom
        Custom(String), // Custom condition expression
    }

    impl PipelineComposer {
        pub fn new(registry: ModuleRegistry) -> Self {
            Self {
                stages: Vec::new(),
                registry,
            }
        }

        /// Add a stage to the pipeline
        pub fn add_stage(&mut self, stage: PipelineStage) -> &mut Self {
            self.stages.push(stage);
            self
        }

        /// Add a conditional stage
        pub fn add_conditional_stage(
            &mut self,
            stage: PipelineStage,
            condition: StageCondition,
        ) -> &mut Self {
            let mut stage = stage;
            stage.condition = Some(condition);
            self.stages.push(stage);
            self
        }

        /// Build the complete pipeline
        pub fn build(&self) -> SklResult<ComposedPipeline> {
            let mut instances = Vec::new();

            for stage in &self.stages {
                let module = self
                    .registry
                    .get_module(&stage.module_name)
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(format!(
                            "Module '{}' not found",
                            stage.module_name
                        ))
                    })?;

                let instance = module.create_instance(&stage.config)?;
                instances.push((stage.clone(), instance));
            }

            Ok(ComposedPipeline { stages: instances })
        }
    }

    /// A composed pipeline of imputation modules
    pub struct ComposedPipeline {
        stages: Vec<(PipelineStage, Box<dyn ImputationInstance>)>,
    }

    impl ComposedPipeline {
        pub fn fit_transform(&mut self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            let mut data = X.to_owned();
            let data_info = self.analyze_data(&data.view())?;

            for (stage, instance) in &mut self.stages {
                // Check stage condition
                if let Some(condition) = &stage.condition {
                    if !Self::evaluate_condition_static(condition, &data_info)? {
                        continue;
                    }
                }

                data = instance.fit_transform(&data.view())?;
            }

            Ok(data)
        }

        fn analyze_data(&self, X: &ArrayView2<Float>) -> SklResult<DataCharacteristics> {
            let (n_samples, n_features) = X.dim();
            let missing_count = X.iter().filter(|&&x| (x as f64).is_nan()).count();
            let missing_rate = missing_count as f64 / (n_samples * n_features) as f64;

            Ok(DataCharacteristics {
                n_samples,
                n_features,
                missing_rate,
                missing_pattern: MissingPatternType::Unknown, // Would need more analysis
                data_types: vec![DataType::Continuous; n_features], // Default assumption
                has_categorical: false,
                has_temporal: false,
                is_sparse: missing_rate > 0.5,
                memory_constraints: None,
            })
        }

        fn evaluate_condition_static(
            condition: &StageCondition,
            data_info: &DataCharacteristics,
        ) -> SklResult<bool> {
            Ok(match condition {
                StageCondition::MissingRate(threshold) => data_info.missing_rate > *threshold,
                StageCondition::FeatureCount(threshold) => data_info.n_features > *threshold,
                StageCondition::DataType(data_type) => data_info.data_types.contains(data_type),
                StageCondition::Custom(_) => true, // Would need expression evaluator
            })
        }
    }

    /// Middleware for imputation pipelines
    pub trait ImputationMiddleware: Send + Sync {
        /// Process data before imputation
        fn before_imputation(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            Ok(X.to_owned())
        }

        /// Process data after imputation
        fn after_imputation(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            Ok(X.to_owned())
        }

        /// Handle errors during imputation
        fn on_error(&self, error: &SklearsError) -> SklResult<()> {
            Err(error.clone())
        }
    }

    /// Validation middleware
    pub struct ValidationMiddleware {
        /// validate_completeness
        pub validate_completeness: bool,
        /// validate_ranges
        pub validate_ranges: bool,
        /// expected_ranges
        pub expected_ranges: Option<HashMap<usize, (f64, f64)>>,
    }

    impl ImputationMiddleware for ValidationMiddleware {
        fn after_imputation(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            if self.validate_completeness {
                if X.iter().any(|&x| (x as f64).is_nan()) {
                    return Err(SklearsError::InvalidInput(
                        "Imputation failed: missing values remain".to_string(),
                    ));
                }
            }

            if self.validate_ranges {
                if let Some(ranges) = &self.expected_ranges {
                    for ((_, j), &value) in X.indexed_iter() {
                        if let Some((min_val, max_val)) = ranges.get(&j) {
                            let val = value as f64;
                            if val < *min_val || val > *max_val {
                                return Err(SklearsError::InvalidInput(
                                    format!("Imputed value {} out of expected range [{}, {}] for feature {}", 
                                           val, min_val, max_val, j)
                                ));
                            }
                        }
                    }
                }
            }

            Ok(X.to_owned())
        }
    }

    /// Logging middleware
    pub struct LoggingMiddleware {
        /// log_level
        pub log_level: LogLevel,
        /// log_performance
        pub log_performance: bool,
    }

    #[derive(Debug, Clone)]
    pub enum LogLevel {
        /// Debug
        Debug,
        /// Info
        Info,
        /// Warn
        Warn,
        /// Error
        Error,
    }

    impl ImputationMiddleware for LoggingMiddleware {
        fn before_imputation(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            if matches!(self.log_level, LogLevel::Debug | LogLevel::Info) {
                let missing_count = X.iter().filter(|&&x| (x as f64).is_nan()).count();
                println!(
                    "Starting imputation: {} missing values in {}x{} matrix",
                    missing_count,
                    X.nrows(),
                    X.ncols()
                );
            }
            Ok(X.to_owned())
        }

        fn after_imputation(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
            if matches!(self.log_level, LogLevel::Debug | LogLevel::Info) {
                let remaining_missing = X.iter().filter(|&&x| (x as f64).is_nan()).count();
                println!(
                    "Imputation completed: {} missing values remaining",
                    remaining_missing
                );
            }
            Ok(X.to_owned())
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_fluent_api_simple_imputation() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        let pipeline = ImputationBuilder::new()
            .simple()
            .mean()
            .finish()
            .build()
            .unwrap();

        let result = pipeline.fit_transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));

        // Non-missing values should be preserved
        assert_abs_diff_eq!(result[[0, 0]] as f64, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]] as f64, 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_fluent_api_knn_imputation() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0, 7.0, 8.0])
                .unwrap();

        let pipeline = ImputationBuilder::new()
            .knn()
            .n_neighbors(2)
            .distance_weights()
            .finish()
            .build()
            .unwrap();

        let result = pipeline.fit_transform(&data.view()).unwrap();

        // Should have no missing values
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));
    }

    #[test]
    fn test_preset_configurations() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        // Test fast preset
        let pipeline = ImputationBuilder::new()
            .preset(ImputationPreset::Fast)
            .build()
            .unwrap();

        let result = pipeline.fit_transform(&data.view()).unwrap();
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));

        // Test balanced preset
        let pipeline = ImputationBuilder::new()
            .preset(ImputationPreset::Balanced)
            .build()
            .unwrap();

        let result = pipeline.fit_transform(&data.view()).unwrap();
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));
    }

    #[test]
    fn test_quick_functions() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, f64::NAN, 4.0, 5.0, 6.0]).unwrap();

        // Test quick mean imputation
        let result = quick::mean_impute(&data.view()).unwrap();
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));

        // Test quick KNN imputation
        let result = quick::knn_impute(&data.view(), 2).unwrap();
        assert!(!result.iter().any(|&x| (x as f64).is_nan()));
    }

    #[test]
    fn test_method_chaining() {
        let builder = ImputationBuilder::new()
            .normalize()
            .cross_validate(5)
            .with_uncertainty()
            .parallel(None);

        // Should be able to chain methods without issues
        assert!(builder.validation.cross_validation);
        assert_eq!(builder.validation.cv_folds, 5);
        assert!(builder.preprocessing.normalize);
        assert!(builder.postprocessing.add_uncertainty_estimates);
        assert!(builder.parallel_config.is_some());
    }
}

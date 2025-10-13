//! Composable generation strategies for flexible dataset creation
//!
//! This module provides a framework for composing different dataset generation
//! strategies in a pipeline-style manner. It allows users to combine multiple
//! generation approaches to create complex, realistic datasets.

use crate::traits::{DatasetGenerator, DatasetTraitResult, InMemoryDataset};
use scirs2_core::ndarray::{Array1, Array2, Axis, s};
use scirs2_core::random::{Random, rng};
use std::collections::HashMap;
use std::sync::Arc;
use thiserror::Error;

/// Errors that can occur during composable generation
#[derive(Error, Debug)]
pub enum ComposableError {
    #[error("Strategy not found: {name}")]
    StrategyNotFound { name: String },
    #[error("Invalid composition: {reason}")]
    InvalidComposition { reason: String },
    #[error("Type mismatch in pipeline: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },
    #[error("Pipeline execution failed: {source}")]
    PipelineExecution { source: Box<dyn std::error::Error + Send + Sync> },
    #[error("Parameter error: {message}")]
    ParameterError { message: String },
}

/// Result type for composable generation operations
pub type ComposableResult<T> = Result<T, ComposableError>;

/// Configuration for generation strategies
#[derive(Debug, Clone)]
pub struct StrategyConfig {
    pub parameters: HashMap<String, StrategyValue>,
}

/// Values that can be used in strategy configuration
#[derive(Debug, Clone)]
pub enum StrategyValue {
    /// Integer

    Integer(i64),
    /// Float

    Float(f64),
    /// String

    String(String),
    /// Boolean

    Boolean(bool),
    /// Array

    Array(Vec<f64>),
    /// Nested

    Nested(HashMap<String, StrategyValue>),
}

impl StrategyConfig {
    /// Create a new empty configuration
    pub fn new() -> Self {
        Self {
            parameters: HashMap::new(),
        }
    }

    /// Add a parameter to the configuration
    pub fn with_param(mut self, key: &str, value: StrategyValue) -> Self {
        self.parameters.insert(key.to_string(), value);
        self
    }

    /// Get a parameter value
    pub fn get(&self, key: &str) -> Option<&StrategyValue> {
        self.parameters.get(key)
    }

    /// Get an integer parameter
    pub fn get_int(&self, key: &str) -> ComposableResult<i64> {
        match self.get(key) {
            Some(StrategyValue::Integer(value)) => Ok(*value),
            Some(_) => Err(ComposableError::ParameterError {
                message: format!("Parameter {} is not an integer", key),
            }),
            None => Err(ComposableError::ParameterError {
                message: format!("Parameter {} not found", key),
            }),
        }
    }

    /// Get a float parameter
    pub fn get_float(&self, key: &str) -> ComposableResult<f64> {
        match self.get(key) {
            Some(StrategyValue::Float(value)) => Ok(*value),
            Some(StrategyValue::Integer(value)) => Ok(*value as f64),
            Some(_) => Err(ComposableError::ParameterError {
                message: format!("Parameter {} is not a float", key),
            }),
            None => Err(ComposableError::ParameterError {
                message: format!("Parameter {} not found", key),
            }),
        }
    }

    /// Get a string parameter
    pub fn get_string(&self, key: &str) -> ComposableResult<&str> {
        match self.get(key) {
            Some(StrategyValue::String(value)) => Ok(value),
            Some(_) => Err(ComposableError::ParameterError {
                message: format!("Parameter {} is not a string", key),
            }),
            None => Err(ComposableError::ParameterError {
                message: format!("Parameter {} not found", key),
            }),
        }
    }

    /// Get a boolean parameter
    pub fn get_bool(&self, key: &str) -> ComposableResult<bool> {
        match self.get(key) {
            Some(StrategyValue::Boolean(value)) => Ok(*value),
            Some(_) => Err(ComposableError::ParameterError {
                message: format!("Parameter {} is not a boolean", key),
            }),
            None => Err(ComposableError::ParameterError {
                message: format!("Parameter {} not found", key),
            }),
        }
    }
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Trait for generation strategies that can be composed
pub trait GenerationStrategy: Send + Sync {
    /// Name of the strategy
    fn name(&self) -> &str;

    /// Generate data using this strategy
    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset>;

    /// Validate the configuration for this strategy
    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()>;

    /// Get required parameters for this strategy
    fn required_parameters(&self) -> Vec<String>;

    /// Get optional parameters with defaults for this strategy
    fn optional_parameters(&self) -> HashMap<String, StrategyValue>;

    /// Check if this strategy can be combined with another
    fn can_combine_with(&self, other: &dyn GenerationStrategy) -> bool {
        // Default implementation - most strategies can be combined
        true
    }
}

/// A strategy that applies transformations to existing datasets
pub trait TransformationStrategy: Send + Sync {
    /// Name of the transformation strategy
    fn name(&self) -> &str;

    /// Apply transformation to a dataset
    fn transform(&self,
                dataset: &InMemoryDataset,
                config: &StrategyConfig) -> ComposableResult<InMemoryDataset>;

    /// Validate the configuration for this transformation
    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()>;
}

/// Registry for generation strategies
pub struct StrategyRegistry {
    generators: HashMap<String, Arc<dyn GenerationStrategy>>,
    transformers: HashMap<String, Arc<dyn TransformationStrategy>>,
}

impl StrategyRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
            transformers: HashMap::new(),
        }
    }

    /// Register a generation strategy
    pub fn register_generator(&mut self, strategy: Arc<dyn GenerationStrategy>) {
        self.generators.insert(strategy.name().to_string(), strategy);
    }

    /// Register a transformation strategy
    pub fn register_transformer(&mut self, strategy: Arc<dyn TransformationStrategy>) {
        self.transformers.insert(strategy.name().to_string(), strategy);
    }

    /// Get a generation strategy by name
    pub fn get_generator(&self, name: &str) -> Option<&Arc<dyn GenerationStrategy>> {
        self.generators.get(name)
    }

    /// Get a transformation strategy by name
    pub fn get_transformer(&self, name: &str) -> Option<&Arc<dyn TransformationStrategy>> {
        self.transformers.get(name)
    }

    /// List all available generators
    pub fn list_generators(&self) -> Vec<&str> {
        self.generators.keys().map(|s| s.as_str()).collect()
    }

    /// List all available transformers
    pub fn list_transformers(&self) -> Vec<&str> {
        self.transformers.keys().map(|s| s.as_str()).collect()
    }
}

impl Default for StrategyRegistry {
    fn default() -> Self {
        let mut registry = Self::new();

        // Register built-in strategies
        registry.register_generator(Arc::new(BlobsStrategy::new()));
        registry.register_generator(Arc::new(ClassificationStrategy::new()));
        registry.register_generator(Arc::new(RegressionStrategy::new()));
        registry.register_generator(Arc::new(CirclesStrategy::new()));
        registry.register_generator(Arc::new(MoonsStrategy::new()));

        // Register built-in transformers
        registry.register_transformer(Arc::new(NoiseTransformer::new()));
        registry.register_transformer(Arc::new(ScaleTransformer::new()));
        registry.register_transformer(Arc::new(CorrelationTransformer::new()));
        registry.register_transformer(Arc::new(OutlierTransformer::new()));

        registry
    }
}

/// Step in a generation pipeline
#[derive(Debug, Clone)]
pub enum PipelineStep {
    /// Generate

    Generate {

        strategy: String,

        config: StrategyConfig,
    },
    /// Transform

    Transform {

        strategy: String,

        config: StrategyConfig,
    },
    /// Combine

    Combine {

        method: CombineMethod,
        weights: Option<Vec<f64>>,
    },
}

/// Methods for combining datasets
#[derive(Debug, Clone)]
pub enum CombineMethod {
    /// Concatenate datasets vertically (more samples)
    Concatenate,
    /// Merge datasets horizontally (more features)
    Merge,
    /// Weighted combination of datasets
    WeightedSum,
    /// Overlay datasets (for multi-modal data)
    Overlay,
}

/// Pipeline for composing generation strategies
pub struct GenerationPipeline {
    steps: Vec<PipelineStep>,
    registry: Arc<StrategyRegistry>,
    intermediate_results: Vec<InMemoryDataset>,
}

impl GenerationPipeline {
    /// Create a new pipeline
    pub fn new(registry: Arc<StrategyRegistry>) -> Self {
        Self {
            steps: Vec::new(),
            registry,
            intermediate_results: Vec::new(),
        }
    }

    /// Add a generation step
    pub fn generate(mut self, strategy: &str, config: StrategyConfig) -> Self {
        self.steps.push(PipelineStep::Generate {
            strategy: strategy.to_string(),
            config,
        });
        self
    }

    /// Add a transformation step
    pub fn transform(mut self, strategy: &str, config: StrategyConfig) -> Self {
        self.steps.push(PipelineStep::Transform {
            strategy: strategy.to_string(),
            config,
        });
        self
    }

    /// Add a combination step
    pub fn combine(mut self, method: CombineMethod, weights: Option<Vec<f64>>) -> Self {
        self.steps.push(PipelineStep::Combine { method, weights });
        self
    }

    /// Execute the pipeline
    pub fn execute(&mut self) -> ComposableResult<InMemoryDataset> {
        self.intermediate_results.clear();

        for step in &self.steps {
            match step {
                PipelineStep::Generate { strategy, config } => {
                    self.execute_generate_step(strategy, config)?;
                }
                PipelineStep::Transform { strategy, config } => {
                    self.execute_transform_step(strategy, config)?;
                }
                PipelineStep::Combine { method, weights } => {
                    self.execute_combine_step(method, weights)?;
                }
            }
        }

        self.intermediate_results.pop()
            .ok_or_else(|| ComposableError::InvalidComposition {
                reason: "Pipeline produced no results".to_string(),
            })
    }

    fn execute_generate_step(&mut self, strategy: &str, config: &StrategyConfig) -> ComposableResult<()> {
        let generator = self.registry.get_generator(strategy)
            .ok_or_else(|| ComposableError::StrategyNotFound {
                name: strategy.to_string(),
            })?;

        generator.validate_config(config)?;
        let dataset = generator.generate(config)?;
        self.intermediate_results.push(dataset);
        Ok(())
    }

    fn execute_transform_step(&mut self, strategy: &str, config: &StrategyConfig) -> ComposableResult<()> {
        let transformer = self.registry.get_transformer(strategy)
            .ok_or_else(|| ComposableError::StrategyNotFound {
                name: strategy.to_string(),
            })?;

        let dataset = self.intermediate_results.pop()
            .ok_or_else(|| ComposableError::InvalidComposition {
                reason: "No dataset to transform".to_string(),
            })?;

        transformer.validate_config(config)?;
        let transformed = transformer.transform(&dataset, config)?;
        self.intermediate_results.push(transformed);
        Ok(())
    }

    fn execute_combine_step(&mut self, method: &CombineMethod, weights: &Option<Vec<f64>>) -> ComposableResult<()> {
        if self.intermediate_results.len() < 2 {
            return Err(ComposableError::InvalidComposition {
                reason: "Need at least 2 datasets to combine".to_string(),
            });
        }

        let datasets = std::mem::take(&mut self.intermediate_results);
        let combined = self.combine_datasets(datasets, method, weights)?;
        self.intermediate_results.push(combined);
        Ok(())
    }

    fn combine_datasets(
        &self,
        datasets: Vec<InMemoryDataset>,
        method: &CombineMethod,
        weights: &Option<Vec<f64>>,
    ) -> ComposableResult<InMemoryDataset> {
        match method {
            CombineMethod::Concatenate => self.concatenate_datasets(datasets),
            CombineMethod::Merge => self.merge_datasets(datasets),
            CombineMethod::WeightedSum => self.weighted_sum_datasets(datasets, weights),
            CombineMethod::Overlay => self.overlay_datasets(datasets),
        }
    }

    fn concatenate_datasets(&self, datasets: Vec<InMemoryDataset>) -> ComposableResult<InMemoryDataset> {
        if datasets.is_empty() {
            return Err(ComposableError::InvalidComposition {
                reason: "No datasets to concatenate".to_string(),
            });
        }

        let n_features = datasets[0].features.ncols();
        let total_samples = datasets.iter().map(|d| d.features.nrows()).sum();

        // Verify all datasets have same number of features
        for dataset in &datasets {
            if dataset.features.ncols() != n_features {
                return Err(ComposableError::TypeMismatch {
                    expected: format!("{} features", n_features),
                    actual: format!("{} features", dataset.features.ncols()),
                });
            }
        }

        let mut combined_features = Array2::zeros((total_samples, n_features));
        let mut combined_targets = if datasets[0].targets.is_some() {
            Some(Array1::zeros(total_samples))
        } else {
            None
        };

        let mut row_offset = 0;
        for dataset in datasets {
            let n_rows = dataset.features.nrows();
            combined_features.slice_mut(s![row_offset..row_offset + n_rows, ..])
                .assign(&dataset.features);

            if let (Some(ref mut targets), Some(ref dataset_targets)) =
                (&mut combined_targets, &dataset.targets) {
                targets.slice_mut(s![row_offset..row_offset + n_rows])
                    .assign(dataset_targets);
            }

            row_offset += n_rows;
        }

        Ok(InMemoryDataset {
            features: combined_features,
            targets: combined_targets,
            feature_names: None,
            target_names: None,
        })
    }

    fn merge_datasets(&self, datasets: Vec<InMemoryDataset>) -> ComposableResult<InMemoryDataset> {
        if datasets.is_empty() {
            return Err(ComposableError::InvalidComposition {
                reason: "No datasets to merge".to_string(),
            });
        }

        let n_samples = datasets[0].features.nrows();
        let total_features: usize = datasets.iter().map(|d| d.features.ncols()).sum();

        // Verify all datasets have same number of samples
        for dataset in &datasets {
            if dataset.features.nrows() != n_samples {
                return Err(ComposableError::TypeMismatch {
                    expected: format!("{} samples", n_samples),
                    actual: format!("{} samples", dataset.features.nrows()),
                });
            }
        }

        let mut combined_features = Array2::zeros((n_samples, total_features));

        let mut col_offset = 0;
        for dataset in datasets {
            let n_cols = dataset.features.ncols();
            combined_features.slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(&dataset.features);
            col_offset += n_cols;
        }

        // Use targets from first dataset (if any)
        let targets = datasets.into_iter().next().unwrap().targets;

        Ok(InMemoryDataset {
            features: combined_features,
            targets,
            feature_names: None,
            target_names: None,
        })
    }

    fn weighted_sum_datasets(&self, datasets: Vec<InMemoryDataset>, weights: &Option<Vec<f64>>) -> ComposableResult<InMemoryDataset> {
        if datasets.is_empty() {
            return Err(ComposableError::InvalidComposition {
                reason: "No datasets to combine".to_string(),
            });
        }

        let weights = weights.as_ref().map(|w| w.clone()).unwrap_or_else(|| {
            vec![1.0 / datasets.len() as f64; datasets.len()]
        });

        if weights.len() != datasets.len() {
            return Err(ComposableError::ParameterError {
                message: format!("Number of weights ({}) doesn't match number of datasets ({})",
                               weights.len(), datasets.len()),
            });
        }

        let n_samples = datasets[0].features.nrows();
        let n_features = datasets[0].features.ncols();

        // Verify all datasets have same shape
        for dataset in &datasets {
            if dataset.features.nrows() != n_samples || dataset.features.ncols() != n_features {
                return Err(ComposableError::TypeMismatch {
                    expected: format!("{}x{}", n_samples, n_features),
                    actual: format!("{}x{}", dataset.features.nrows(), dataset.features.ncols()),
                });
            }
        }

        let mut combined_features = Array2::zeros((n_samples, n_features));
        let mut combined_targets = None;

        for (i, (dataset, weight)) in datasets.iter().zip(weights.iter()).enumerate() {
            if i == 0 {
                combined_features.assign(&(&dataset.features * *weight));
                if let Some(ref targets) = dataset.targets {
                    combined_targets = Some(targets * *weight);
                }
            } else {
                combined_features += &(&dataset.features * *weight);
                if let (Some(ref mut combined), Some(ref targets)) =
                    (&mut combined_targets, &dataset.targets) {
                    *combined += &(targets * *weight);
                }
            }
        }

        Ok(InMemoryDataset {
            features: combined_features,
            targets: combined_targets,
            feature_names: None,
            target_names: None,
        })
    }

    fn overlay_datasets(&self, datasets: Vec<InMemoryDataset>) -> ComposableResult<InMemoryDataset> {
        // For overlay, we take the maximum value at each position
        if datasets.is_empty() {
            return Err(ComposableError::InvalidComposition {
                reason: "No datasets to overlay".to_string(),
            });
        }

        let n_samples = datasets[0].features.nrows();
        let n_features = datasets[0].features.ncols();

        // Verify all datasets have same shape
        for dataset in &datasets {
            if dataset.features.nrows() != n_samples || dataset.features.ncols() != n_features {
                return Err(ComposableError::TypeMismatch {
                    expected: format!("{}x{}", n_samples, n_features),
                    actual: format!("{}x{}", dataset.features.nrows(), dataset.features.ncols()),
                });
            }
        }

        let mut combined_features = datasets[0].features.clone();
        let mut combined_targets = datasets[0].targets.clone();

        for dataset in datasets.iter().skip(1) {
            for ((i, j), value) in dataset.features.indexed_iter() {
                if *value > combined_features[(i, j)] {
                    combined_features[(i, j)] = *value;
                }
            }

            if let (Some(ref mut combined), Some(ref targets)) =
                (&mut combined_targets, &dataset.targets) {
                for (i, value) in targets.indexed_iter() {
                    if *value > combined[i] {
                        combined[i] = *value;
                    }
                }
            }
        }

        Ok(InMemoryDataset {
            features: combined_features,
            targets: combined_targets,
            feature_names: None,
            target_names: None,
        })
    }
}

// Built-in generation strategies

/// Blobs generation strategy
pub struct BlobsStrategy;

impl BlobsStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl GenerationStrategy for BlobsStrategy {
    fn name(&self) -> &str {
        "blobs"
    }

    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let n_samples = config.get_int("n_samples").unwrap_or(100) as usize;
        let n_features = config.get_int("n_features").unwrap_or(2) as usize;
        let centers = config.get_int("centers").unwrap_or(3) as usize;
        let cluster_std = config.get_float("cluster_std").unwrap_or(1.0);

        // Use make_blobs from the existing generators
        crate::make_blobs(n_samples, n_features, Some(centers), Some(cluster_std), None, Some(42))
            .map_err(|e| ComposableError::PipelineExecution {
                source: Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
            })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(n_samples) = config.get_int("n_samples") {
            if n_samples <= 0 {
                return Err(ComposableError::ParameterError {
                    message: "n_samples must be positive".to_string(),
                });
            }
        }
        Ok(())
    }

    fn required_parameters(&self) -> Vec<String> {
        vec![]
    }

    fn optional_parameters(&self) -> HashMap<String, StrategyValue> {
        let mut params = HashMap::new();
        params.insert("n_samples".to_string(), StrategyValue::Integer(100));
        params.insert("n_features".to_string(), StrategyValue::Integer(2));
        params.insert("centers".to_string(), StrategyValue::Integer(3));
        params.insert("cluster_std".to_string(), StrategyValue::Float(1.0));
        params
    }
}

/// Classification generation strategy
pub struct ClassificationStrategy;

impl ClassificationStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl GenerationStrategy for ClassificationStrategy {
    fn name(&self) -> &str {
        "classification"
    }

    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let n_samples = config.get_int("n_samples").unwrap_or(100) as usize;
        let n_features = config.get_int("n_features").unwrap_or(20) as usize;
        let n_informative = config.get_int("n_informative").unwrap_or(2) as usize;
        let n_redundant = config.get_int("n_redundant").unwrap_or(2) as usize;
        let n_classes = config.get_int("n_classes").unwrap_or(2) as usize;

        crate::make_classification(
            n_samples,
            n_features,
            Some(n_informative),
            Some(n_redundant),
            None,
            Some(n_classes),
            None,
            None,
            None,
            None,
            Some(42),
        ).map_err(|e| ComposableError::PipelineExecution {
            source: Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
        })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(n_classes) = config.get_int("n_classes") {
            if n_classes < 2 {
                return Err(ComposableError::ParameterError {
                    message: "n_classes must be at least 2".to_string(),
                });
            }
        }
        Ok(())
    }

    fn required_parameters(&self) -> Vec<String> {
        vec![]
    }

    fn optional_parameters(&self) -> HashMap<String, StrategyValue> {
        let mut params = HashMap::new();
        params.insert("n_samples".to_string(), StrategyValue::Integer(100));
        params.insert("n_features".to_string(), StrategyValue::Integer(20));
        params.insert("n_informative".to_string(), StrategyValue::Integer(2));
        params.insert("n_redundant".to_string(), StrategyValue::Integer(2));
        params.insert("n_classes".to_string(), StrategyValue::Integer(2));
        params
    }
}

/// Regression generation strategy
pub struct RegressionStrategy;

impl RegressionStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl GenerationStrategy for RegressionStrategy {
    fn name(&self) -> &str {
        "regression"
    }

    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let n_samples = config.get_int("n_samples").unwrap_or(100) as usize;
        let n_features = config.get_int("n_features").unwrap_or(20) as usize;
        let n_informative = config.get_int("n_informative").unwrap_or(10) as usize;
        let noise = config.get_float("noise").unwrap_or(0.0);

        crate::make_regression(
            n_samples,
            n_features,
            Some(n_informative),
            None,
            Some(noise),
            None,
            None,
            Some(42),
        ).map_err(|e| ComposableError::PipelineExecution {
            source: Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
        })
    }

    fn validate_config(&self, _config: &StrategyConfig) -> ComposableResult<()> {
        Ok(())
    }

    fn required_parameters(&self) -> Vec<String> {
        vec![]
    }

    fn optional_parameters(&self) -> HashMap<String, StrategyValue> {
        let mut params = HashMap::new();
        params.insert("n_samples".to_string(), StrategyValue::Integer(100));
        params.insert("n_features".to_string(), StrategyValue::Integer(20));
        params.insert("n_informative".to_string(), StrategyValue::Integer(10));
        params.insert("noise".to_string(), StrategyValue::Float(0.0));
        params
    }
}

/// Circles generation strategy
pub struct CirclesStrategy;

impl CirclesStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl GenerationStrategy for CirclesStrategy {
    fn name(&self) -> &str {
        "circles"
    }

    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let n_samples = config.get_int("n_samples").unwrap_or(100) as usize;
        let shuffle = config.get_bool("shuffle").unwrap_or(true);
        let noise = config.get_float("noise").unwrap_or(0.0);
        let factor = config.get_float("factor").unwrap_or(0.8);

        crate::make_circles(n_samples, Some(shuffle), Some(noise), Some(factor), Some(42))
            .map_err(|e| ComposableError::PipelineExecution {
                source: Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
            })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(factor) = config.get_float("factor") {
            if factor <= 0.0 || factor >= 1.0 {
                return Err(ComposableError::ParameterError {
                    message: "factor must be between 0 and 1".to_string(),
                });
            }
        }
        Ok(())
    }

    fn required_parameters(&self) -> Vec<String> {
        vec![]
    }

    fn optional_parameters(&self) -> HashMap<String, StrategyValue> {
        let mut params = HashMap::new();
        params.insert("n_samples".to_string(), StrategyValue::Integer(100));
        params.insert("shuffle".to_string(), StrategyValue::Boolean(true));
        params.insert("noise".to_string(), StrategyValue::Float(0.0));
        params.insert("factor".to_string(), StrategyValue::Float(0.8));
        params
    }
}

/// Moons generation strategy
pub struct MoonsStrategy;

impl MoonsStrategy {
    pub fn new() -> Self {
        Self
    }
}

impl GenerationStrategy for MoonsStrategy {
    fn name(&self) -> &str {
        "moons"
    }

    fn generate(&self, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let n_samples = config.get_int("n_samples").unwrap_or(100) as usize;
        let shuffle = config.get_bool("shuffle").unwrap_or(true);
        let noise = config.get_float("noise").unwrap_or(0.0);

        crate::make_moons(n_samples, Some(shuffle), Some(noise), Some(42))
            .map_err(|e| ComposableError::PipelineExecution {
                source: Box::new(e) as Box<dyn std::error::Error + Send + Sync>,
            })
    }

    fn validate_config(&self, _config: &StrategyConfig) -> ComposableResult<()> {
        Ok(())
    }

    fn required_parameters(&self) -> Vec<String> {
        vec![]
    }

    fn optional_parameters(&self) -> HashMap<String, StrategyValue> {
        let mut params = HashMap::new();
        params.insert("n_samples".to_string(), StrategyValue::Integer(100));
        params.insert("shuffle".to_string(), StrategyValue::Boolean(true));
        params.insert("noise".to_string(), StrategyValue::Float(0.0));
        params
    }
}

// Built-in transformation strategies

/// Noise addition transformer
pub struct NoiseTransformer;

impl NoiseTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl TransformationStrategy for NoiseTransformer {
    fn name(&self) -> &str {
        "noise"
    }

    fn transform(&self, dataset: &InMemoryDataset, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let std_dev = config.get_float("std_dev").unwrap_or(0.1);
        let feature_noise = config.get_bool("feature_noise").unwrap_or(true);
        let target_noise = config.get_bool("target_noise").unwrap_or(false);

        let mut rng = Random::new(Some(42));
        let mut new_features = dataset.features.clone();
        let mut new_targets = dataset.targets.clone();

        if feature_noise {
            for value in new_features.iter_mut() {
                *value += rng.sample_normal(0.0, std_dev);
            }
        }

        if target_noise {
            if let Some(ref mut targets) = new_targets {
                for value in targets.iter_mut() {
                    *value += rng.sample_normal(0.0, std_dev);
                }
            }
        }

        Ok(InMemoryDataset {
            features: new_features,
            targets: new_targets,
            feature_names: dataset.feature_names.clone(),
            target_names: dataset.target_names.clone(),
        })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(std_dev) = config.get_float("std_dev") {
            if std_dev < 0.0 {
                return Err(ComposableError::ParameterError {
                    message: "std_dev must be non-negative".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Scaling transformer
pub struct ScaleTransformer;

impl ScaleTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl TransformationStrategy for ScaleTransformer {
    fn name(&self) -> &str {
        "scale"
    }

    fn transform(&self, dataset: &InMemoryDataset, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let factor = config.get_float("factor").unwrap_or(1.0);
        let per_feature = config.get_bool("per_feature").unwrap_or(false);

        let mut new_features = dataset.features.clone();

        if per_feature {
            // Scale each feature independently
            for mut column in new_features.axis_iter_mut(Axis(1)) {
                let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
                let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let range = max_val - min_val;

                if range > 1e-10 {
                    for value in column.iter_mut() {
                        *value = (*value - min_val) / range * factor;
                    }
                }
            }
        } else {
            // Scale all features by the same factor
            new_features *= factor;
        }

        Ok(InMemoryDataset {
            features: new_features,
            targets: dataset.targets.clone(),
            feature_names: dataset.feature_names.clone(),
            target_names: dataset.target_names.clone(),
        })
    }

    fn validate_config(&self, _config: &StrategyConfig) -> ComposableResult<()> {
        Ok(())
    }
}

/// Correlation transformer
pub struct CorrelationTransformer;

impl CorrelationTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl TransformationStrategy for CorrelationTransformer {
    fn name(&self) -> &str {
        "correlation"
    }

    fn transform(&self, dataset: &InMemoryDataset, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let correlation = config.get_float("correlation").unwrap_or(0.5);

        if dataset.features.ncols() < 2 {
            return Ok(dataset.clone());
        }

        let mut new_features = dataset.features.clone();
        let mut rng = Random::new(Some(42));

        // Introduce correlation between first two features
        let first_col = new_features.column(0).to_owned();
        let mut second_col = new_features.column(1).to_owned();

        for (i, value) in second_col.iter_mut().enumerate() {
            let noise = rng.sample_normal(0.0, 1.0 - correlation.abs());
            *value = correlation * first_col[i] + (1.0 - correlation.abs()) * *value + noise;
        }

        new_features.column_mut(1).assign(&second_col);

        Ok(InMemoryDataset {
            features: new_features,
            targets: dataset.targets.clone(),
            feature_names: dataset.feature_names.clone(),
            target_names: dataset.target_names.clone(),
        })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(correlation) = config.get_float("correlation") {
            if correlation < -1.0 || correlation > 1.0 {
                return Err(ComposableError::ParameterError {
                    message: "correlation must be between -1 and 1".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Outlier transformer
pub struct OutlierTransformer;

impl OutlierTransformer {
    pub fn new() -> Self {
        Self
    }
}

impl TransformationStrategy for OutlierTransformer {
    fn name(&self) -> &str {
        "outliers"
    }

    fn transform(&self, dataset: &InMemoryDataset, config: &StrategyConfig) -> ComposableResult<InMemoryDataset> {
        let fraction = config.get_float("fraction").unwrap_or(0.05);
        let factor = config.get_float("factor").unwrap_or(3.0);

        let mut new_features = dataset.features.clone();
        let mut rng = Random::new(Some(42));

        let n_outliers = (dataset.features.nrows() as f64 * fraction) as usize;

        for _ in 0..n_outliers {
            let row = rng.sample_range(0..dataset.features.nrows());
            let col = rng.sample_range(0..dataset.features.ncols());

            // Get column statistics
            let column = new_features.column(col);
            let mean = column.sum() / column.len() as f64;
            let variance = column.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64;
            let std_dev = variance.sqrt();

            // Create outlier
            let sign = if rng.sample_range(0..2) == 0 { 1.0 } else { -1.0 };
            new_features[(row, col)] = mean + sign * factor * std_dev;
        }

        Ok(InMemoryDataset {
            features: new_features,
            targets: dataset.targets.clone(),
            feature_names: dataset.feature_names.clone(),
            target_names: dataset.target_names.clone(),
        })
    }

    fn validate_config(&self, config: &StrategyConfig) -> ComposableResult<()> {
        if let Ok(fraction) = config.get_float("fraction") {
            if fraction < 0.0 || fraction > 1.0 {
                return Err(ComposableError::ParameterError {
                    message: "fraction must be between 0 and 1".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// Builder for easy pipeline construction
pub struct PipelineBuilder {
    registry: Arc<StrategyRegistry>,
}

impl PipelineBuilder {
    /// Create a new pipeline builder
    pub fn new() -> Self {
        Self {
            registry: Arc::new(StrategyRegistry::default()),
        }
    }

    /// Create a pipeline builder with custom registry
    pub fn with_registry(registry: Arc<StrategyRegistry>) -> Self {
        Self { registry }
    }

    /// Start building a pipeline
    pub fn pipeline(&self) -> GenerationPipeline {
        GenerationPipeline::new(self.registry.clone())
    }

    /// Get a reference to the registry
    pub fn registry(&self) -> &StrategyRegistry {
        &self.registry
    }
}

impl Default for PipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_config() {
        let config = StrategyConfig::new()
            .with_param("n_samples", StrategyValue::Integer(100))
            .with_param("noise", StrategyValue::Float(0.1))
            .with_param("shuffle", StrategyValue::Boolean(true));

        assert_eq!(config.get_int("n_samples").unwrap(), 100);
        assert_eq!(config.get_float("noise").unwrap(), 0.1);
        assert_eq!(config.get_bool("shuffle").unwrap(), true);
    }

    #[test]
    fn test_blobs_strategy() {
        let strategy = BlobsStrategy::new();
        let config = StrategyConfig::new()
            .with_param("n_samples", StrategyValue::Integer(50))
            .with_param("n_features", StrategyValue::Integer(3));

        let dataset = strategy.generate(&config).unwrap();
        assert_eq!(dataset.features.nrows(), 50);
        assert_eq!(dataset.features.ncols(), 3);
        assert!(dataset.targets.is_some());
    }

    #[test]
    fn test_noise_transformer() {
        let transformer = NoiseTransformer::new();
        let config = StrategyConfig::new()
            .with_param("std_dev", StrategyValue::Float(0.1));

        // Create a simple dataset
        let features = Array2::zeros((10, 2));
        let targets = Some(Array1::zeros(10));
        let dataset = InMemoryDataset {
            features,
            targets,
            feature_names: None,
            target_names: None,
        };

        let transformed = transformer.transform(&dataset, &config).unwrap();

        // Check that the transformation was applied
        assert_eq!(transformed.features.shape(), dataset.features.shape());
        // Noise should have been added, so values shouldn't be exactly zero
        assert!(transformed.features.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_pipeline_execution() {
        let builder = PipelineBuilder::new();
        let mut pipeline = builder.pipeline()
            .generate("blobs", StrategyConfig::new()
                .with_param("n_samples", StrategyValue::Integer(100))
                .with_param("n_features", StrategyValue::Integer(2)))
            .transform("noise", StrategyConfig::new()
                .with_param("std_dev", StrategyValue::Float(0.1)));

        let result = pipeline.execute().unwrap();
        assert_eq!(result.features.nrows(), 100);
        assert_eq!(result.features.ncols(), 2);
    }

    #[test]
    fn test_dataset_concatenation() {
        let builder = PipelineBuilder::new();

        let mut pipeline1 = builder.pipeline()
            .generate("blobs", StrategyConfig::new()
                .with_param("n_samples", StrategyValue::Integer(50)));
        let dataset1 = pipeline1.execute().unwrap();

        let mut pipeline2 = builder.pipeline()
            .generate("blobs", StrategyConfig::new()
                .with_param("n_samples", StrategyValue::Integer(30)));
        let dataset2 = pipeline2.execute().unwrap();

        let pipeline = GenerationPipeline::new(builder.registry.clone());
        let combined = pipeline.concatenate_datasets(vec![dataset1, dataset2]).unwrap();

        assert_eq!(combined.features.nrows(), 80); // 50 + 30
    }

    #[test]
    fn test_dataset_merge() {
        let builder = PipelineBuilder::new();

        let mut pipeline1 = builder.pipeline()
            .generate("blobs", StrategyConfig::new()
                .with_param("n_samples", StrategyValue::Integer(50))
                .with_param("n_features", StrategyValue::Integer(2)));
        let dataset1 = pipeline1.execute().unwrap();

        let mut pipeline2 = builder.pipeline()
            .generate("classification", StrategyConfig::new()
                .with_param("n_samples", StrategyValue::Integer(50))
                .with_param("n_features", StrategyValue::Integer(3)));
        let dataset2 = pipeline2.execute().unwrap();

        let pipeline = GenerationPipeline::new(builder.registry.clone());
        let combined = pipeline.merge_datasets(vec![dataset1, dataset2]).unwrap();

        assert_eq!(combined.features.nrows(), 50);
        assert_eq!(combined.features.ncols(), 5); // 2 + 3
    }

    #[test]
    fn test_strategy_registry() {
        let registry = StrategyRegistry::default();

        assert!(registry.get_generator("blobs").is_some());
        assert!(registry.get_generator("classification").is_some());
        assert!(registry.get_transformer("noise").is_some());
        assert!(registry.get_transformer("scale").is_some());

        let generators = registry.list_generators();
        assert!(generators.contains(&"blobs"));
        assert!(generators.contains(&"classification"));
    }
}
//! Trait-based dataset framework for extensible dataset operations
//!
//! This module provides a comprehensive trait system for dataset generation,
//! loading, transformation, and validation. It enables pluggable generators
//! and composable generation strategies.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{Distribution, RandNormal, Random};
use std::collections::HashMap;
use thiserror::Error;

// Note: We inline the distribution sampling rather than using a helper function
// because Random and Random<StdRng> are different types and hard to make generic

/// Errors in the trait-based dataset framework
#[derive(Error, Debug)]
pub enum DatasetTraitError {
    #[error("Generation error: {0}")]
    Generation(String),
    #[error("Validation error: {0}")]
    Validation(String),
    #[error("Configuration error: {0}")]
    Configuration(String),
    #[error("IO error: {0}")]
    Io(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),
}

pub type DatasetTraitResult<T> = Result<T, DatasetTraitError>;

/// Core trait for any dataset representation
pub trait Dataset {
    /// Get the number of samples in the dataset
    fn n_samples(&self) -> usize;

    /// Get the number of features in the dataset
    fn n_features(&self) -> usize;

    /// Get the shape of the dataset as (n_samples, n_features)
    fn shape(&self) -> (usize, usize) {
        (self.n_samples(), self.n_features())
    }

    /// Get features as an array view
    fn features(&self) -> DatasetTraitResult<ArrayView2<'_, f64>>;

    /// Get a specific sample (row) by index
    fn sample(&self, index: usize) -> DatasetTraitResult<ArrayView1<'_, f64>>;

    /// Check if the dataset has target values
    fn has_targets(&self) -> bool;

    /// Get targets as an array view (if available)
    fn targets(&self) -> DatasetTraitResult<Option<ArrayView1<'_, f64>>>;

    /// Get dataset metadata
    fn metadata(&self) -> HashMap<String, String> {
        HashMap::new()
    }
}

/// Trait for datasets that can be generated
pub trait DatasetGenerator {
    type Config: Default + Clone;
    type Output: Dataset;

    /// Generate a dataset with the given configuration
    fn generate(&self, config: Self::Config) -> DatasetTraitResult<Self::Output>;

    /// Get the name of this generator
    fn name(&self) -> &'static str;

    /// Get a description of what this generator produces
    fn description(&self) -> &'static str;

    /// Validate the configuration before generation
    fn validate_config(&self, config: &Self::Config) -> DatasetTraitResult<()> {
        let _ = config;
        Ok(())
    }
}

/// Trait for loading datasets from external sources
pub trait DatasetLoader {
    type Config: Default + Clone;
    type Output: Dataset;

    /// Load a dataset with the given configuration
    fn load(&self, config: Self::Config) -> DatasetTraitResult<Self::Output>;

    /// Get the name of this loader
    fn name(&self) -> &'static str;

    /// Get available datasets that this loader can handle
    fn available_datasets(&self) -> Vec<String>;

    /// Check if a dataset is available
    fn has_dataset(&self, name: &str) -> bool {
        self.available_datasets().contains(&name.to_string())
    }
}

/// Trait for transforming datasets
pub trait DatasetTransformer {
    type Config: Default + Clone;
    type Input: Dataset;
    type Output: Dataset;

    /// Transform a dataset
    fn transform(
        &self,
        input: Self::Input,
        config: Self::Config,
    ) -> DatasetTraitResult<Self::Output>;

    /// Get the name of this transformer
    fn name(&self) -> &'static str;

    /// Check if this transformer can handle the given input
    fn can_transform(&self, input: &Self::Input) -> bool;
}

/// Trait for validating datasets
pub trait DatasetValidator {
    type Config: Default + Clone;
    type Report: Default;

    /// Validate a dataset and return a validation report
    fn validate(
        &self,
        dataset: &dyn Dataset,
        config: Self::Config,
    ) -> DatasetTraitResult<Self::Report>;

    /// Get the name of this validator
    fn name(&self) -> &'static str;

    /// Get validation criteria
    fn criteria(&self) -> Vec<String>;
}

/// Trait for streaming dataset access
pub trait StreamingDataset: Dataset {
    type Batch;

    /// Get a batch starting at the given index with the specified size
    fn batch(&self, start: usize, size: usize) -> DatasetTraitResult<Self::Batch>;

    /// Create an iterator over batches
    fn batches(
        &self,
        batch_size: usize,
    ) -> Box<dyn Iterator<Item = DatasetTraitResult<Self::Batch>>>;

    /// Get the preferred batch size for this dataset
    fn preferred_batch_size(&self) -> usize {
        1000
    }
}

/// Trait for mutable datasets
pub trait MutableDataset: Dataset {
    /// Set a specific sample
    fn set_sample(&mut self, index: usize, sample: ArrayView1<f64>) -> DatasetTraitResult<()>;

    /// Set targets (if supported)
    fn set_targets(&mut self, targets: ArrayView1<f64>) -> DatasetTraitResult<()>;

    /// Add a new sample to the dataset
    fn add_sample(
        &mut self,
        sample: ArrayView1<f64>,
        target: Option<f64>,
    ) -> DatasetTraitResult<()>;

    /// Remove a sample from the dataset
    fn remove_sample(&mut self, index: usize) -> DatasetTraitResult<()>;
}

/// Configuration for composable generation strategies
pub trait GenerationStrategy {
    type Config: Default + Clone;

    /// Apply this strategy to modify generation parameters
    fn apply(&self, config: &mut Self::Config, rng: &mut Random) -> DatasetTraitResult<()>;

    /// Get the name of this strategy
    fn name(&self) -> &'static str;

    /// Check if this strategy is applicable to the given configuration
    fn is_applicable(&self, config: &Self::Config) -> bool;
}

/// A concrete implementation of Dataset trait for in-memory datasets
#[derive(Debug, Clone)]
pub struct InMemoryDataset {
    features: Array2<f64>,
    targets: Option<Array1<f64>>,
    metadata: HashMap<String, String>,
}

impl InMemoryDataset {
    /// Create a new in-memory dataset
    pub fn new(features: Array2<f64>, targets: Option<Array1<f64>>) -> Self {
        Self {
            features,
            targets,
            metadata: HashMap::new(),
        }
    }

    /// Create with metadata
    pub fn with_metadata(
        features: Array2<f64>,
        targets: Option<Array1<f64>>,
        metadata: HashMap<String, String>,
    ) -> Self {
        Self {
            features,
            targets,
            metadata,
        }
    }

    /// Add metadata entry
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }
}

impl Dataset for InMemoryDataset {
    fn n_samples(&self) -> usize {
        self.features.nrows()
    }

    fn n_features(&self) -> usize {
        self.features.ncols()
    }

    fn features(&self) -> DatasetTraitResult<ArrayView2<'_, f64>> {
        Ok(self.features.view())
    }

    fn sample(&self, index: usize) -> DatasetTraitResult<ArrayView1<'_, f64>> {
        if index >= self.n_samples() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("index < {}", self.n_samples()),
                actual: format!("index = {}", index),
            });
        }
        Ok(self.features.row(index))
    }

    fn has_targets(&self) -> bool {
        self.targets.is_some()
    }

    fn targets(&self) -> DatasetTraitResult<Option<ArrayView1<'_, f64>>> {
        Ok(self.targets.as_ref().map(|t| t.view()))
    }

    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }
}

impl MutableDataset for InMemoryDataset {
    fn set_sample(&mut self, index: usize, sample: ArrayView1<f64>) -> DatasetTraitResult<()> {
        if index >= self.n_samples() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("index < {}", self.n_samples()),
                actual: format!("index = {}", index),
            });
        }
        if sample.len() != self.n_features() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                actual: format!("{} features", sample.len()),
            });
        }
        self.features.row_mut(index).assign(&sample);
        Ok(())
    }

    fn set_targets(&mut self, targets: ArrayView1<f64>) -> DatasetTraitResult<()> {
        if targets.len() != self.n_samples() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("{} targets", self.n_samples()),
                actual: format!("{} targets", targets.len()),
            });
        }
        self.targets = Some(targets.to_owned());
        Ok(())
    }

    fn add_sample(
        &mut self,
        sample: ArrayView1<f64>,
        _target: Option<f64>,
    ) -> DatasetTraitResult<()> {
        if sample.len() != self.n_features() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("{} features", self.n_features()),
                actual: format!("{} features", sample.len()),
            });
        }

        // This is a simplified implementation - in practice, you'd need to resize the arrays
        Err(DatasetTraitError::UnsupportedOperation(
            "Adding samples to fixed-size arrays not yet implemented".to_string(),
        ))
    }

    fn remove_sample(&mut self, index: usize) -> DatasetTraitResult<()> {
        if index >= self.n_samples() {
            return Err(DatasetTraitError::DimensionMismatch {
                expected: format!("index < {}", self.n_samples()),
                actual: format!("index = {}", index),
            });
        }

        // This is a simplified implementation - in practice, you'd need to resize the arrays
        Err(DatasetTraitError::UnsupportedOperation(
            "Removing samples from fixed-size arrays not yet implemented".to_string(),
        ))
    }
}

/// Registry for dataset generators
pub struct GeneratorRegistry {
    generators: HashMap<
        String,
        Box<dyn DatasetGenerator<Config = GeneratorConfig, Output = InMemoryDataset>>,
    >,
}

impl GeneratorRegistry {
    /// Create a new registry
    pub fn new() -> Self {
        Self {
            generators: HashMap::new(),
        }
    }

    /// Register a generator
    pub fn register<G>(&mut self, generator: G)
    where
        G: DatasetGenerator<Config = GeneratorConfig, Output = InMemoryDataset> + 'static,
    {
        self.generators
            .insert(generator.name().to_string(), Box::new(generator));
    }

    /// Get a generator by name
    pub fn get(
        &self,
        name: &str,
    ) -> Option<&dyn DatasetGenerator<Config = GeneratorConfig, Output = InMemoryDataset>> {
        self.generators.get(name).map(|g| g.as_ref())
    }

    /// List all available generators
    pub fn list(&self) -> Vec<String> {
        self.generators.keys().cloned().collect()
    }

    /// Generate a dataset using a named generator
    pub fn generate(
        &self,
        name: &str,
        config: GeneratorConfig,
    ) -> DatasetTraitResult<InMemoryDataset> {
        let generator = self.get(name).ok_or_else(|| {
            DatasetTraitError::Configuration(format!("Unknown generator: {}", name))
        })?;
        generator.generate(config)
    }
}

impl Default for GeneratorRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Universal configuration for generators
#[derive(Debug, Clone)]
pub struct GeneratorConfig {
    pub n_samples: usize,
    pub n_features: usize,
    pub random_state: Option<u64>,
    pub parameters: HashMap<String, ConfigValue>,
}

impl Default for GeneratorConfig {
    fn default() -> Self {
        Self {
            n_samples: 100,
            n_features: 2,
            random_state: None,
            parameters: HashMap::new(),
        }
    }
}

impl GeneratorConfig {
    /// Create a new configuration
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        Self {
            n_samples,
            n_features,
            random_state: None,
            parameters: HashMap::new(),
        }
    }

    /// Set a parameter
    pub fn set_parameter<T: Into<ConfigValue>>(&mut self, key: String, value: T) {
        self.parameters.insert(key, value.into());
    }

    /// Get a parameter
    pub fn get_parameter(&self, key: &str) -> Option<&ConfigValue> {
        self.parameters.get(key)
    }

    /// Set random state
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Configuration value types
#[derive(Debug, Clone)]
pub enum ConfigValue {
    /// Int
    Int(i64),
    /// Float
    Float(f64),
    /// String
    String(String),
    /// Bool
    Bool(bool),
    /// IntArray
    IntArray(Vec<i64>),
    /// FloatArray
    FloatArray(Vec<f64>),
}

impl From<i64> for ConfigValue {
    fn from(value: i64) -> Self {
        ConfigValue::Int(value)
    }
}

impl From<f64> for ConfigValue {
    fn from(value: f64) -> Self {
        ConfigValue::Float(value)
    }
}

impl From<String> for ConfigValue {
    fn from(value: String) -> Self {
        ConfigValue::String(value)
    }
}

impl From<bool> for ConfigValue {
    fn from(value: bool) -> Self {
        ConfigValue::Bool(value)
    }
}

impl From<Vec<i64>> for ConfigValue {
    fn from(value: Vec<i64>) -> Self {
        ConfigValue::IntArray(value)
    }
}

impl From<Vec<f64>> for ConfigValue {
    fn from(value: Vec<f64>) -> Self {
        ConfigValue::FloatArray(value)
    }
}

/// Example implementation: Classification generator
pub struct ClassificationGenerator;

impl DatasetGenerator for ClassificationGenerator {
    type Config = GeneratorConfig;
    type Output = InMemoryDataset;

    fn generate(&self, config: Self::Config) -> DatasetTraitResult<Self::Output> {
        let mut rng = match config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Get number of classes from parameters
        let n_classes = config
            .get_parameter("n_classes")
            .and_then(|v| match v {
                ConfigValue::Int(n) => Some(*n as usize),
                _ => None,
            })
            .unwrap_or(2);

        // Generate features
        let mut features = Array2::<f64>::zeros((config.n_samples, config.n_features));
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        for mut row in features.rows_mut() {
            for val in row.iter_mut() {
                *val = normal_dist.sample(&mut rng);
            }
        }

        // Generate targets
        let targets: Array1<f64> =
            Array1::from_shape_fn(config.n_samples, |_| rng.gen_range(0..n_classes) as f64);

        let mut metadata = HashMap::new();
        metadata.insert("generator".to_string(), "classification".to_string());
        metadata.insert("n_classes".to_string(), n_classes.to_string());

        Ok(InMemoryDataset::with_metadata(
            features,
            Some(targets),
            metadata,
        ))
    }

    fn name(&self) -> &'static str {
        "classification"
    }

    fn description(&self) -> &'static str {
        "Generates a classification dataset with Gaussian features"
    }

    fn validate_config(&self, config: &Self::Config) -> DatasetTraitResult<()> {
        if config.n_samples == 0 {
            return Err(DatasetTraitError::Configuration(
                "n_samples must be > 0".to_string(),
            ));
        }
        if config.n_features == 0 {
            return Err(DatasetTraitError::Configuration(
                "n_features must be > 0".to_string(),
            ));
        }

        // Validate n_classes parameter
        if let Some(ConfigValue::Int(n_classes)) = config.get_parameter("n_classes") {
            if *n_classes <= 0 {
                return Err(DatasetTraitError::Configuration(
                    "n_classes must be > 0".to_string(),
                ));
            }
        }

        Ok(())
    }
}

/// Example implementation: Regression generator
pub struct RegressionGenerator;

impl DatasetGenerator for RegressionGenerator {
    type Config = GeneratorConfig;
    type Output = InMemoryDataset;

    fn generate(&self, config: Self::Config) -> DatasetTraitResult<Self::Output> {
        let mut rng = match config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Get noise level from parameters
        let noise = config
            .get_parameter("noise")
            .and_then(|v| match v {
                ConfigValue::Float(n) => Some(*n),
                _ => None,
            })
            .unwrap_or(0.1);

        // Generate features
        let mut features = Array2::<f64>::zeros((config.n_samples, config.n_features));
        let normal_dist = RandNormal::new(0.0, 1.0).unwrap();
        for mut row in features.rows_mut() {
            for val in row.iter_mut() {
                *val = normal_dist.sample(&mut rng);
            }
        }

        // Generate random coefficients
        let coefficients: Array1<f64> =
            Array1::from_shape_fn(config.n_features, |_| rng.random_range(-1.0, 1.0));

        // Generate targets
        let mut targets = Array1::<f64>::zeros(config.n_samples);
        for (i, target) in targets.iter_mut().enumerate() {
            let feature_row = features.row(i);
            let noise_dist = RandNormal::new(0.0, noise).unwrap();
            *target = feature_row.dot(&coefficients) + noise_dist.sample(&mut rng);
        }

        let mut metadata = HashMap::new();
        metadata.insert("generator".to_string(), "regression".to_string());
        metadata.insert("noise".to_string(), noise.to_string());

        Ok(InMemoryDataset::with_metadata(
            features,
            Some(targets),
            metadata,
        ))
    }

    fn name(&self) -> &'static str {
        "regression"
    }

    fn description(&self) -> &'static str {
        "Generates a regression dataset with linear relationship and noise"
    }
}

/// Factory function to create a default registry with standard generators
pub fn create_default_registry() -> GeneratorRegistry {
    let mut registry = GeneratorRegistry::new();
    registry.register(ClassificationGenerator);
    registry.register(RegressionGenerator);
    registry
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array;

    #[test]
    fn test_in_memory_dataset() {
        let features = Array::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let targets = Array1::from_shape_vec(10, (0..10).map(|x| x as f64).collect()).unwrap();

        let dataset = InMemoryDataset::new(features, Some(targets));

        assert_eq!(dataset.n_samples(), 10);
        assert_eq!(dataset.n_features(), 3);
        assert_eq!(dataset.shape(), (10, 3));
        assert!(dataset.has_targets());

        let features_view = dataset.features().unwrap();
        assert_eq!(features_view.dim(), (10, 3));

        let sample = dataset.sample(5).unwrap();
        assert_eq!(sample.len(), 3);
        assert_eq!(sample[0], 15.0); // 5 * 3 + 0

        let targets_view = dataset.targets().unwrap().unwrap();
        assert_eq!(targets_view.len(), 10);
        assert_eq!(targets_view[5], 5.0);
    }

    #[test]
    fn test_generator_registry() {
        let mut registry = GeneratorRegistry::new();
        registry.register(ClassificationGenerator);
        registry.register(RegressionGenerator);

        let generators = registry.list();
        assert!(generators.contains(&"classification".to_string()));
        assert!(generators.contains(&"regression".to_string()));

        let config = GeneratorConfig::new(50, 4);
        let dataset = registry.generate("classification", config).unwrap();

        assert_eq!(dataset.n_samples(), 50);
        assert_eq!(dataset.n_features(), 4);
        assert!(dataset.has_targets());
    }

    #[test]
    fn test_classification_generator() {
        let generator = ClassificationGenerator;
        let mut config = GeneratorConfig::new(100, 5);
        config.set_parameter("n_classes".to_string(), 3i64);
        config.random_state = Some(42);

        let dataset = generator.generate(config).unwrap();

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 5);
        assert!(dataset.has_targets());

        let targets = dataset.targets().unwrap().unwrap();
        assert!(targets.iter().all(|&t| t >= 0.0 && t < 3.0));

        let metadata = dataset.metadata();
        assert_eq!(
            metadata.get("generator"),
            Some(&"classification".to_string())
        );
        assert_eq!(metadata.get("n_classes"), Some(&"3".to_string()));
    }

    #[test]
    fn test_regression_generator() {
        let generator = RegressionGenerator;
        let mut config = GeneratorConfig::new(100, 3);
        config.set_parameter("noise".to_string(), 0.05);
        config.random_state = Some(42);

        let dataset = generator.generate(config).unwrap();

        assert_eq!(dataset.n_samples(), 100);
        assert_eq!(dataset.n_features(), 3);
        assert!(dataset.has_targets());

        let metadata = dataset.metadata();
        assert_eq!(metadata.get("generator"), Some(&"regression".to_string()));
        assert_eq!(metadata.get("noise"), Some(&"0.05".to_string()));
    }

    #[test]
    fn test_config_validation() {
        let generator = ClassificationGenerator;

        // Valid config
        let valid_config = GeneratorConfig::new(100, 5);
        assert!(generator.validate_config(&valid_config).is_ok());

        // Invalid configs
        let invalid_config = GeneratorConfig::new(0, 5);
        assert!(generator.validate_config(&invalid_config).is_err());

        let invalid_config = GeneratorConfig::new(100, 0);
        assert!(generator.validate_config(&invalid_config).is_err());
    }

    #[test]
    fn test_config_parameters() {
        let mut config = GeneratorConfig::new(100, 5);

        config.set_parameter("n_classes".to_string(), 3i64);
        config.set_parameter("noise".to_string(), 0.1);
        config.set_parameter("seed".to_string(), "test".to_string());
        config.set_parameter("enabled".to_string(), true);

        assert!(matches!(
            config.get_parameter("n_classes"),
            Some(ConfigValue::Int(3))
        ));
        assert!(matches!(
            config.get_parameter("noise"),
            Some(ConfigValue::Float(0.1))
        ));
        assert!(matches!(
            config.get_parameter("seed"),
            Some(ConfigValue::String(_))
        ));
        assert!(matches!(
            config.get_parameter("enabled"),
            Some(ConfigValue::Bool(true))
        ));
    }

    #[test]
    fn test_default_registry() {
        let registry = create_default_registry();
        let generators = registry.list();

        assert!(generators.contains(&"classification".to_string()));
        assert!(generators.contains(&"regression".to_string()));
        assert_eq!(generators.len(), 2);
    }

    #[test]
    fn test_mutable_dataset() {
        let features = Array::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let targets = Array1::from_shape_vec(3, vec![10.0, 20.0, 30.0]).unwrap();

        let mut dataset = InMemoryDataset::new(features, Some(targets));

        // Test setting a sample
        let new_sample = Array1::from_vec(vec![99.0, 88.0]);
        assert!(dataset.set_sample(1, new_sample.view()).is_ok());

        let updated_sample = dataset.sample(1).unwrap();
        assert_eq!(updated_sample[0], 99.0);
        assert_eq!(updated_sample[1], 88.0);

        // Test dimension mismatch
        let wrong_sample = Array1::from_vec(vec![1.0, 2.0, 3.0]); // Wrong size
        assert!(dataset.set_sample(0, wrong_sample.view()).is_err());

        // Test index out of bounds
        let sample = Array1::from_vec(vec![1.0, 2.0]);
        assert!(dataset.set_sample(10, sample.view()).is_err());
    }
}

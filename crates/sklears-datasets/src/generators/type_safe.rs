//! Type-safe dataset abstractions with phantom types and const generics
//!
//! This module provides compile-time safety for dataset generation using
//! phantom types and const generics to prevent common errors and ensure
//! dataset consistency.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::marker::PhantomData;

/// Phantom types for dataset characteristics
pub struct Classification;
pub struct Regression;
pub struct Clustering;
pub struct TimeSeries;
pub struct Spatial;

/// Dataset configuration with compile-time validation
#[derive(Debug, Clone)]
pub struct DatasetConfig<T, const N_SAMPLES: usize, const N_FEATURES: usize> {
    _phantom: PhantomData<T>,
    pub random_state: Option<u64>,
}

impl<T, const N_SAMPLES: usize, const N_FEATURES: usize> DatasetConfig<T, N_SAMPLES, N_FEATURES> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
            random_state: None,
        }
    }

    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

/// Type-safe dataset container with guaranteed dimensions
#[derive(Debug, Clone)]
pub struct TypeSafeDataset<T, const N_SAMPLES: usize, const N_FEATURES: usize> {
    pub features: Array2<f64>,
    pub targets: DatasetTargets<T>,
    _phantom: PhantomData<T>,
}

impl<T, const N_SAMPLES: usize, const N_FEATURES: usize> TypeSafeDataset<T, N_SAMPLES, N_FEATURES> {
    /// Create a new type-safe dataset with compile-time dimension validation
    pub fn new(features: Array2<f64>, targets: DatasetTargets<T>) -> Result<Self> {
        if features.shape() != &[N_SAMPLES, N_FEATURES] {
            return Err(SklearsError::InvalidInput(format!(
                "Expected features shape [{}, {}], got {:?}",
                N_SAMPLES,
                N_FEATURES,
                features.shape()
            )));
        }

        Ok(Self {
            features,
            targets,
            _phantom: PhantomData,
        })
    }

    /// Get the number of samples (compile-time constant)
    pub const fn n_samples() -> usize {
        N_SAMPLES
    }

    /// Get the number of features (compile-time constant)
    pub const fn n_features() -> usize {
        N_FEATURES
    }

    /// Get feature matrix
    pub fn features(&self) -> &Array2<f64> {
        &self.features
    }

    /// Get targets
    pub fn targets(&self) -> &DatasetTargets<T> {
        &self.targets
    }
}

/// Type-safe target values for different dataset types
#[derive(Debug, Clone)]
pub enum DatasetTargets<T> {
    /// Classification

    Classification(Array1<i32>),
    /// Regression

    Regression(Array1<f64>),
    /// Clustering

    Clustering(Array1<i32>),
    /// TimeSeries

    TimeSeries(Array1<f64>),
    /// Spatial

    Spatial(Array1<f64>),

    _Phantom(PhantomData<T>),
}

/// Type-safe classification dataset generator with const generics
pub fn make_typed_classification<
    const N_SAMPLES: usize,
    const N_FEATURES: usize,
    const N_CLASSES: usize,
>(
    config: DatasetConfig<Classification, N_SAMPLES, N_FEATURES>,
) -> Result<TypeSafeDataset<Classification, N_SAMPLES, N_FEATURES>> {
    // Compile-time validation
    const { assert!(N_SAMPLES > 0, "N_SAMPLES must be positive") };
    const { assert!(N_FEATURES > 0, "N_FEATURES must be positive") };
    const { assert!(N_CLASSES >= 2, "N_CLASSES must be at least 2") };

    // Use the basic classification generator
    let (features, targets) = crate::generators::basic::make_classification(
        N_SAMPLES,
        N_FEATURES,
        N_FEATURES / 2, // n_informative
        N_FEATURES / 4, // n_redundant
        N_CLASSES,
        config.random_state,
    )?;

    let dataset = TypeSafeDataset::new(features, DatasetTargets::Classification(targets))?;

    Ok(dataset)
}

/// Type-safe regression dataset generator with const generics
pub fn make_typed_regression<const N_SAMPLES: usize, const N_FEATURES: usize>(
    config: DatasetConfig<Regression, N_SAMPLES, N_FEATURES>,
    noise: f64,
) -> Result<TypeSafeDataset<Regression, N_SAMPLES, N_FEATURES>> {
    // Compile-time validation
    const { assert!(N_SAMPLES > 0, "N_SAMPLES must be positive") };
    const { assert!(N_FEATURES > 0, "N_FEATURES must be positive") };

    // Use the basic regression generator
    let (features, targets) = crate::generators::basic::make_regression(
        N_SAMPLES,
        N_FEATURES,
        N_FEATURES / 2, // n_informative
        noise,
        config.random_state,
    )?;

    let dataset = TypeSafeDataset::new(features, DatasetTargets::Regression(targets))?;

    Ok(dataset)
}

/// Type-safe clustering dataset generator with const generics
pub fn make_typed_blobs<const N_SAMPLES: usize, const N_FEATURES: usize, const N_CENTERS: usize>(
    config: DatasetConfig<Clustering, N_SAMPLES, N_FEATURES>,
    cluster_std: f64,
) -> Result<TypeSafeDataset<Clustering, N_SAMPLES, N_FEATURES>> {
    // Compile-time validation
    const { assert!(N_SAMPLES > 0, "N_SAMPLES must be positive") };
    const { assert!(N_FEATURES > 0, "N_FEATURES must be positive") };
    const { assert!(N_CENTERS > 0, "N_CENTERS must be positive") };

    // Use the basic blob generator
    let (features, targets) = crate::generators::basic::make_blobs(
        N_SAMPLES,
        N_FEATURES,
        N_CENTERS,
        cluster_std,
        config.random_state,
    )?;

    let dataset = TypeSafeDataset::new(features, DatasetTargets::Clustering(targets))?;

    Ok(dataset)
}

/// Builder pattern for type-safe dataset configuration
pub struct DatasetBuilder<T> {
    random_state: Option<u64>,
    _phantom: PhantomData<T>,
}

impl<T> DatasetBuilder<T> {
    pub fn new() -> Self {
        Self {
            random_state: None,
            _phantom: PhantomData,
        }
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn build<const N_SAMPLES: usize, const N_FEATURES: usize>(
        self,
    ) -> DatasetConfig<T, N_SAMPLES, N_FEATURES> {
        DatasetConfig {
            _phantom: PhantomData,
            random_state: self.random_state,
        }
    }
}

/// Convenience functions for creating builders
pub fn classification_builder() -> DatasetBuilder<Classification> {
    DatasetBuilder::new()
}

pub fn regression_builder() -> DatasetBuilder<Regression> {
    DatasetBuilder::new()
}

pub fn clustering_builder() -> DatasetBuilder<Clustering> {
    DatasetBuilder::new()
}

/// Type-safe dataset validation at compile time
pub trait ValidateDataset<const N_SAMPLES: usize, const N_FEATURES: usize> {
    const VALID: bool = N_SAMPLES > 0 && N_FEATURES > 0;

    fn validate() -> Result<()> {
        if !Self::VALID {
            return Err(SklearsError::InvalidInput(
                "Invalid dataset dimensions".to_string(),
            ));
        }
        Ok(())
    }
}

impl<T, const N_SAMPLES: usize, const N_FEATURES: usize> ValidateDataset<N_SAMPLES, N_FEATURES>
    for TypeSafeDataset<T, N_SAMPLES, N_FEATURES>
{
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_classification() {
        let config = classification_builder().random_state(42).build::<100, 10>();

        let dataset = make_typed_classification::<100, 10, 3>(config).unwrap();

        assert_eq!(dataset.features().shape(), &[100, 10]);
        assert_eq!(TypeSafeDataset::<Classification, 100, 10>::n_samples(), 100);
        assert_eq!(TypeSafeDataset::<Classification, 100, 10>::n_features(), 10);

        match dataset.targets() {
            DatasetTargets::Classification(targets) => {
                assert_eq!(targets.len(), 100);
            }
            _ => panic!("Wrong target type"),
        }
    }

    #[test]
    fn test_typed_regression() {
        let config = regression_builder().random_state(42).build::<50, 5>();

        let dataset = make_typed_regression::<50, 5>(config, 0.1).unwrap();

        assert_eq!(dataset.features().shape(), &[50, 5]);
        assert_eq!(TypeSafeDataset::<Regression, 50, 5>::n_samples(), 50);
        assert_eq!(TypeSafeDataset::<Regression, 50, 5>::n_features(), 5);

        match dataset.targets() {
            DatasetTargets::Regression(targets) => {
                assert_eq!(targets.len(), 50);
            }
            _ => panic!("Wrong target type"),
        }
    }

    #[test]
    fn test_typed_blobs() {
        let config = clustering_builder().random_state(42).build::<80, 4>();

        let dataset = make_typed_blobs::<80, 4, 3>(config, 1.0).unwrap();

        assert_eq!(dataset.features().shape(), &[80, 4]);
        assert_eq!(TypeSafeDataset::<Clustering, 80, 4>::n_samples(), 80);
        assert_eq!(TypeSafeDataset::<Clustering, 80, 4>::n_features(), 4);

        match dataset.targets() {
            DatasetTargets::Clustering(targets) => {
                assert_eq!(targets.len(), 80);
            }
            _ => panic!("Wrong target type"),
        }
    }

    #[test]
    fn test_dataset_validation() {
        type ValidDataset = TypeSafeDataset<Classification, 100, 10>;
        assert!(ValidDataset::validate().is_ok());
    }

    #[test]
    fn test_wrong_dimensions() {
        let config = classification_builder().build::<10, 5>();
        let wrong_features = Array2::zeros((5, 5)); // Wrong shape
        let targets = DatasetTargets::Classification(Array1::zeros(10));

        let result = TypeSafeDataset::<Classification, 10, 5>::new(wrong_features, targets);
        assert!(result.is_err());
    }
}

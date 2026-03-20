pub mod builder;
/// Dataset functionality for sklears machine learning library
///
/// This module provides comprehensive dataset handling capabilities organized
/// into focused sub-modules:
///
/// - `core`: Core Dataset struct and fundamental operations
/// - `builder`: Type-safe builder pattern with compile-time validation
/// - `synthetic`: Synthetic dataset generation for testing and prototyping
/// - `mmap`: Memory-mapped datasets for handling large files
///
/// # Quick Start
///
/// ## Creating a simple dataset
///
/// ```rust
/// use sklears_core::dataset::{Dataset, synthetic};
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// // Create from arrays
/// let features = Array2::<f64>::zeros((100, 4));
/// let targets = Array1::<f64>::zeros(100);
/// let dataset = Dataset::new(features, targets);
///
/// // Or generate synthetic data
/// let dataset = synthetic::make_regression(100, 4, 0.1).unwrap();
/// ```
///
/// ## Using the builder pattern
///
/// ```rust
/// use sklears_core::dataset::Dataset;
/// use scirs2_core::ndarray::{Array1, Array2};
///
/// let features = Array2::<f64>::ones((50, 3));
/// let targets = Array1::<f64>::ones(50);
///
/// let dataset = Dataset::builder()
///     .data(features)
///     .target(targets)
///     .feature_names(vec!["f1".to_string(), "f2".to_string(), "f3".to_string()])
///     .description("Example dataset".to_string())
///     .build();
/// ```
///
/// ## Memory-mapped datasets for large files
///
/// ```rust,ignore
/// #[cfg(feature = "mmap")]
/// use sklears_core::dataset::mmap::{MmapDataset, make_large_regression};
/// use std::path::Path;
///
/// // Create a large dataset file
/// make_large_regression(
///     Path::new("large_data.skl"),
///     1_000_000,  // 1M samples
///     100,        // 100 features
///     0.1,        // noise level
///     Some(5000)  // chunk size
/// ).unwrap();
///
/// // Load and process in batches
/// let dataset = MmapDataset::from_file("large_data.skl").unwrap();
/// for batch in dataset.batch_iter(1000) {
///     let (features, targets) = batch.unwrap();
///     // Process batch...
/// }
/// ```
pub mod core;
pub mod synthetic;

#[cfg(feature = "mmap")]
pub mod mmap;

// Re-export commonly used types for convenience
pub use builder::{DatasetBuilder, HasData, HasTarget, NoData, NoTarget};
pub use core::{Dataset, HasShape};

// Re-export synthetic data generation functions
pub use synthetic::{load_iris, make_blobs, make_classification, make_regression};

// Re-export memory-mapped functionality when available
#[cfg(feature = "mmap")]
pub use mmap::{
    make_large_regression, MmapDataset, MmapDatasetBuilder, MmapDatasetBuilderConfig,
    MmapSerializable,
};

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Array1, Array2};

    #[test]
    fn test_module_integration() {
        // Test that all modules work together
        let synthetic_dataset =
            synthetic::make_regression(20, 3, 0.1).expect("expected valid value");

        assert_eq!(synthetic_dataset.data.dim(), (20, 3));
        assert_eq!(synthetic_dataset.target.len(), 20);

        // Test builder pattern
        let builder_dataset = Dataset::builder()
            .data(Array2::<f64>::zeros((10, 2)))
            .target(Array1::<f64>::zeros(10))
            .description("Integration test".to_string())
            .build();

        assert_eq!(builder_dataset.description, "Integration test");
        assert_eq!(builder_dataset.data.dim(), (10, 2));
    }

    #[test]
    fn test_iris_dataset() {
        let iris = synthetic::load_iris().expect("expected valid value");

        assert_eq!(iris.data.dim(), (6, 4));
        assert_eq!(iris.target.len(), 6);
        assert_eq!(iris.feature_names.len(), 4);
        assert!(iris.target_names.is_some());
        assert_eq!(
            iris.target_names
                .as_ref()
                .expect("value should be present")
                .len(),
            3
        );
    }

    #[test]
    fn test_blob_generation() {
        let blobs = synthetic::make_blobs(30, 2, 3, 1.0).expect("expected valid value");

        assert_eq!(blobs.data.dim(), (30, 2));
        assert_eq!(blobs.target.len(), 30);

        // Should have 3 different target values
        let mut unique_targets: Vec<_> = blobs.target.iter().cloned().collect();
        unique_targets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_targets.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        assert!(unique_targets.len() <= 3);
    }

    #[test]
    fn test_classification_generation() {
        let classification =
            synthetic::make_classification(40, 3, 2.0).expect("expected valid value");

        assert_eq!(classification.data.dim(), (40, 3));
        assert_eq!(classification.target.len(), 40);

        // Should be binary classification
        let mut unique_targets: Vec<_> = classification.target.iter().cloned().collect();
        unique_targets.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        unique_targets.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        assert!(unique_targets.len() <= 2);
    }

    #[test]
    fn test_dataset_metadata() {
        let dataset = Dataset::new(Array2::<f64>::ones((5, 2)), Array1::<f64>::ones(5))
            .with_feature_names(vec!["x".to_string(), "y".to_string()])
            .with_target_names(vec!["class_a".to_string(), "class_b".to_string()])
            .with_description("Test metadata".to_string());

        assert_eq!(dataset.feature_names.len(), 2);
        assert_eq!(
            dataset
                .target_names
                .as_ref()
                .expect("value should be present")
                .len(),
            2
        );
        assert_eq!(dataset.description, "Test metadata");
        assert_eq!(dataset.n_samples(), Some(5));
        assert_eq!(dataset.n_features(), Some(2));
    }
}

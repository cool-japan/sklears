use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;


#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::generators::basic::{make_blobs, make_classification, make_regression};
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_export_import_classification_csv() {
        let (features, targets) = make_classification(100, 4, 3, 1, 3, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_classification.csv");

        // Export
        let feature_names = vec![
            "feat1".to_string(),
            "feat2".to_string(),
            "feat3".to_string(),
            "feat4".to_string(),
        ];
        export_classification_csv(&file_path, &features, &targets, Some(&feature_names), None)
            .unwrap();

        // Verify file exists and has content
        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("feat1"));
        assert!(content.contains("target"));

        // Import back
        let (imported_features, imported_targets, imported_names) =
            import_classification_csv(&file_path, None).unwrap();

        assert_eq!(features.dim(), imported_features.dim());
        assert_eq!(targets.len(), imported_targets.len());
        assert!(imported_names.is_some());
        assert_eq!(imported_names.unwrap(), feature_names);
    }

    #[test]
    fn test_export_import_regression_csv() {
        let (features, targets) = make_regression(100, 4, 3, 0.1, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_regression.csv");

        // Export
        export_regression_csv(&file_path, &features, &targets, None, None).unwrap();

        // Verify file exists
        assert!(file_path.exists());

        // Import back
        let (imported_features, imported_targets, _) =
            import_regression_csv(&file_path, None).unwrap();

        assert_eq!(features.dim(), imported_features.dim());
        assert_eq!(targets.len(), imported_targets.len());
    }

    #[test]
    #[cfg(feature = "serde")]
    fn test_export_classification_json() {
        let (features, targets) = make_blobs(50, 3, 2, 1.0, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_classification.json");

        let metadata = serde_json::json!({
            "description": "Test classification dataset",
            "created_by": "sklears-datasets",
            "n_samples": 50,
            "n_features": 3
        });

        export_classification_json(&file_path, &features, &targets, None, Some(metadata)).unwrap();

        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains("features"));
        assert!(content.contains("targets"));
        assert!(content.contains("description"));
    }

    #[test]
    fn test_export_tsv() {
        let (features, targets) = make_classification(50, 3, 2, 1, 2, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.tsv");

        export_classification_tsv(&file_path, &features, &targets, None).unwrap();

        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains('\t')); // Should contain tabs
    }

    #[test]
    fn test_export_jsonl() {
        let (features, targets) = make_blobs(20, 2, 2, 1.0, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test.jsonl");

        export_classification_jsonl(&file_path, &features, &targets, None).unwrap();

        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        let lines: Vec<&str> = content.trim().split('\n').collect();
        assert_eq!(lines.len(), 20); // Should have 20 lines

        // Each line should be valid JSON
        for line in lines {
            let _: serde_json::Value = serde_json::from_str(line).unwrap();
        }
    }

    #[test]
    fn test_csv_config() {
        let (features, targets) = make_blobs(10, 2, 2, 1.0, Some(42)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_custom.csv");

        let config = CsvConfig {
            delimiter: ';',
            has_header: false,
            quote_char: '\'',
            escape_char: None,
        };

        export_classification_csv(&file_path, &features, &targets, None, Some(config)).unwrap();

        assert!(file_path.exists());
        let content = fs::read_to_string(&file_path).unwrap();
        assert!(content.contains(';')); // Should use semicolon delimiter
        assert!(!content.starts_with("feature_")); // Should not have header
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_export_import_classification_parquet() {
        let (features, targets) = make_classification(100, 4, 3, 1, 3, Some(42)).unwrap();
        let feature_names = vec![
            "sepal_length".to_string(),
            "sepal_width".to_string(),
            "petal_length".to_string(),
            "petal_width".to_string(),
        ];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_classification.parquet");

        // Export to parquet
        export_classification_parquet(&file_path, &features, &targets, Some(&feature_names))
            .unwrap();
        assert!(file_path.exists());

        // Import from parquet
        let (imported_features, imported_targets, imported_names) =
            import_classification_parquet(&file_path).unwrap();

        // Verify dimensions
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());

        // Verify feature names
        assert!(imported_names.is_some());
        let imported_names = imported_names.unwrap();
        assert_eq!(imported_names, feature_names);

        // Verify data (within floating point precision)
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert_eq!(targets[i], imported_targets[i]);
        }
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_export_import_regression_parquet() {
        let (features, targets) = make_regression(50, 3, 2, 0.1, Some(42)).unwrap();
        let feature_names = vec![
            "feature_0".to_string(),
            "feature_1".to_string(),
            "feature_2".to_string(),
        ];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_regression.parquet");

        // Export to parquet
        export_regression_parquet(&file_path, &features, &targets, Some(&feature_names)).unwrap();
        assert!(file_path.exists());

        // Import from parquet
        let (imported_features, imported_targets, imported_names) =
            import_regression_parquet(&file_path).unwrap();

        // Verify dimensions
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());

        // Verify feature names
        assert!(imported_names.is_some());
        let imported_names = imported_names.unwrap();
        assert_eq!(imported_names, feature_names);

        // Verify data (within floating point precision)
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert!((targets[i] - imported_targets[i]).abs() < 1e-10);
        }
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_parquet_without_feature_names() {
        let (features, targets) = make_classification(30, 2, 2, 0, 2, Some(123)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_no_names.parquet");

        // Export without feature names
        export_classification_parquet(&file_path, &features, &targets, None).unwrap();
        assert!(file_path.exists());

        // Import and check default feature names
        let (imported_features, imported_targets, imported_names) =
            import_classification_parquet(&file_path).unwrap();

        assert!(imported_names.is_some());
        let imported_names = imported_names.unwrap();
        assert_eq!(
            imported_names,
            vec!["feature_0".to_string(), "feature_1".to_string()]
        );

        // Verify data integrity
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());
    }

    #[cfg(feature = "parquet")]
    #[test]
    fn test_parquet_large_dataset() {
        // Test with a larger dataset to ensure scalability
        let (features, targets) = make_regression(1000, 10, 5, 0.05, Some(999)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_large.parquet");

        // Export large dataset
        export_regression_parquet(&file_path, &features, &targets, None).unwrap();
        assert!(file_path.exists());

        // Import and verify
        let (imported_features, imported_targets, _) =
            import_regression_parquet(&file_path).unwrap();

        assert_eq!(imported_features.dim(), (1000, 10));
        assert_eq!(imported_targets.dim(), 1000);

        // Spot check a few values
        for i in [0, 100, 500, 999] {
            for j in 0..10 {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert!((targets[i] - imported_targets[i]).abs() < 1e-10);
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_export_import_classification_hdf5() {
        let (features, targets) = make_classification(100, 4, 3, 1, 3, Some(42)).unwrap();
        let feature_names = vec![
            "sepal_length".to_string(),
            "sepal_width".to_string(),
            "petal_length".to_string(),
            "petal_width".to_string(),
        ];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_classification.h5");

        // Export to HDF5
        export_classification_hdf5(&file_path, &features, &targets, Some(&feature_names)).unwrap();
        assert!(file_path.exists());

        // Import from HDF5
        let (imported_features, imported_targets, imported_names) =
            import_classification_hdf5(&file_path).unwrap();

        // Verify dimensions
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());

        // Verify feature names
        assert!(imported_names.is_some());
        let imported_names = imported_names.unwrap();
        assert_eq!(imported_names, feature_names);

        // Verify data (within floating point precision)
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert_eq!(targets[i], imported_targets[i]);
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_export_import_regression_hdf5() {
        let (features, targets) = make_regression(50, 3, 2, 0.1, Some(42)).unwrap();
        let feature_names = vec![
            "feature_0".to_string(),
            "feature_1".to_string(),
            "feature_2".to_string(),
        ];

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_regression.h5");

        // Export to HDF5
        export_regression_hdf5(&file_path, &features, &targets, Some(&feature_names)).unwrap();
        assert!(file_path.exists());

        // Import from HDF5
        let (imported_features, imported_targets, imported_names) =
            import_regression_hdf5(&file_path).unwrap();

        // Verify dimensions
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());

        // Verify feature names
        assert!(imported_names.is_some());
        let imported_names = imported_names.unwrap();
        assert_eq!(imported_names, feature_names);

        // Verify data (within floating point precision)
        for i in 0..features.nrows() {
            for j in 0..features.ncols() {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert!((targets[i] - imported_targets[i]).abs() < 1e-10);
        }
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_hdf5_without_feature_names() {
        let (features, targets) = make_classification(30, 2, 2, 0, 2, Some(123)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_no_names.h5");

        // Export without feature names
        export_classification_hdf5(&file_path, &features, &targets, None).unwrap();
        assert!(file_path.exists());

        // Import and check no feature names
        let (imported_features, imported_targets, imported_names) =
            import_classification_hdf5(&file_path).unwrap();

        assert!(imported_names.is_none());

        // Verify data integrity
        assert_eq!(imported_features.dim(), features.dim());
        assert_eq!(imported_targets.dim(), targets.dim());
    }

    #[cfg(feature = "hdf5")]
    #[test]
    fn test_hdf5_large_dataset() {
        // Test with a larger dataset to ensure scalability
        let (features, targets) = make_regression(1000, 10, 5, 0.05, Some(999)).unwrap();

        let dir = tempdir().unwrap();
        let file_path = dir.path().join("test_large.h5");

        // Export large dataset
        export_regression_hdf5(&file_path, &features, &targets, None).unwrap();
        assert!(file_path.exists());

        // Import and verify
        let (imported_features, imported_targets, _) = import_regression_hdf5(&file_path).unwrap();

        assert_eq!(imported_features.dim(), (1000, 10));
        assert_eq!(imported_targets.dim(), 1000);

        // Spot check a few values
        for i in [0, 100, 500, 999] {
            for j in 0..10 {
                assert!((features[[i, j]] - imported_features[[i, j]]).abs() < 1e-10);
            }
            assert!((targets[i] - imported_targets[i]).abs() < 1e-10);
        }
    }
}

use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;


/// Export classification dataset to HDF5 format
#[cfg(feature = "hdf5")]
pub fn export_classification_hdf5<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    let (n_samples, n_features) = features.dim();

    let file = H5File::create(path)?;

    // Create groups for organization
    let dataset_group = file.create_group("dataset")?;
    let metadata_group = file.create_group("metadata")?;

    // Write features as 2D array
    let features_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples, n_features])
        .create("features")?;

    // Convert ndarray to Vec for HDF5
    let features_vec: Vec<f64> = features.iter().cloned().collect();
    features_dataset.write(&features_vec)?;

    // Write targets as 1D array
    let targets_dataset = dataset_group
        .new_dataset::<i32>()
        .shape([n_samples])
        .create("targets")?;

    targets_dataset.write(targets.as_slice().unwrap())?;

    // Write feature names if provided
    if let Some(names) = feature_names {
        let names_dataset = metadata_group
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([names.len()])
            .create("feature_names")?;

        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names
            .iter()
            .map(|s| hdf5::types::VarLenUnicode::from_str(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FormatError::Parse(format!("String conversion error: {}", e)))?;

        names_dataset.write(&names_utf8)?;
    }

    // Write dataset metadata
    metadata_group
        .new_attr::<u64>()
        .create("n_samples")?
        .write_scalar(&(n_samples as u64))?;
    metadata_group
        .new_attr::<u64>()
        .create("n_features")?
        .write_scalar(&(n_features as u64))?;
    metadata_group
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("dataset_type")?
        .write_scalar(&hdf5::types::VarLenUnicode::from_str("classification").unwrap())?;

    Ok(())
}

/// Export regression dataset to HDF5 format
#[cfg(feature = "hdf5")]
pub fn export_regression_hdf5<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    let (n_samples, n_features) = features.dim();

    let file = H5File::create(path)?;

    // Create groups for organization
    let dataset_group = file.create_group("dataset")?;
    let metadata_group = file.create_group("metadata")?;

    // Write features as 2D array
    let features_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples, n_features])
        .create("features")?;

    // Convert ndarray to Vec for HDF5
    let features_vec: Vec<f64> = features.iter().cloned().collect();
    features_dataset.write(&features_vec)?;

    // Write targets as 1D array
    let targets_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples])
        .create("targets")?;

    targets_dataset.write(targets.as_slice().unwrap())?;

    // Write feature names if provided
    if let Some(names) = feature_names {
        let names_dataset = metadata_group
            .new_dataset::<hdf5::types::VarLenUnicode>()
            .shape([names.len()])
            .create("feature_names")?;

        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names
            .iter()
            .map(|s| hdf5::types::VarLenUnicode::from_str(s))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| FormatError::Parse(format!("String conversion error: {}", e)))?;

        names_dataset.write(&names_utf8)?;
    }

    // Write dataset metadata
    metadata_group
        .new_attr::<u64>()
        .create("n_samples")?
        .write_scalar(&(n_samples as u64))?;
    metadata_group
        .new_attr::<u64>()
        .create("n_features")?
        .write_scalar(&(n_features as u64))?;
    metadata_group
        .new_attr::<hdf5::types::VarLenUnicode>()
        .create("dataset_type")?
        .write_scalar(&hdf5::types::VarLenUnicode::from_str("regression").unwrap())?;

    Ok(())
}

/// Import classification dataset from HDF5 format
#[cfg(feature = "hdf5")]
pub fn import_classification_hdf5<P: AsRef<Path>>(
    path: P,
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let file = H5File::open(path)?;

    let dataset_group = file.group("dataset")?;
    let metadata_group = file.group("metadata")?;

    // Read metadata to get dimensions
    let n_samples = metadata_group.attr("n_samples")?.read_scalar::<u64>()? as usize;
    let n_features = metadata_group.attr("n_features")?.read_scalar::<u64>()? as usize;

    // Verify this is a classification dataset
    let dataset_type: hdf5::types::VarLenUnicode =
        metadata_group.attr("dataset_type")?.read_scalar()?;
    if dataset_type.as_str() != "classification" {
        return Err(FormatError::InvalidFormat(
            "Expected classification dataset".to_string(),
        ));
    }

    // Read features
    let features_dataset = dataset_group.dataset("features")?;
    let features_vec: Vec<f64> = features_dataset.read_1d()?;

    if features_vec.len() != n_samples * n_features {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples * n_features,
            actual: features_vec.len(),
        });
    }

    let mut features = Array2::zeros((n_samples, n_features));
    for (i, chunk) in features_vec.chunks(n_features).enumerate() {
        for (j, &value) in chunk.iter().enumerate() {
            features[[i, j]] = value;
        }
    }

    // Read targets
    let targets_dataset = dataset_group.dataset("targets")?;
    let targets_vec: Vec<i32> = targets_dataset.read_1d()?;

    if targets_vec.len() != n_samples {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples,
            actual: targets_vec.len(),
        });
    }

    let targets = Array1::from_vec(targets_vec);

    // Read feature names if available
    let feature_names = if metadata_group.link_exists("feature_names") {
        let names_dataset = metadata_group.dataset("feature_names")?;
        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names_dataset.read_1d()?;
        Some(
            names_utf8
                .into_iter()
                .map(|s| s.into_string())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| FormatError::Parse(format!("String conversion error: {}", e)))?,
        )
    } else {
        None
    };

    Ok((features, targets, feature_names))
}

/// Import regression dataset from HDF5 format
#[cfg(feature = "hdf5")]
pub fn import_regression_hdf5<P: AsRef<Path>>(
    path: P,
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    let file = H5File::open(path)?;

    let dataset_group = file.group("dataset")?;
    let metadata_group = file.group("metadata")?;

    // Read metadata to get dimensions
    let n_samples = metadata_group.attr("n_samples")?.read_scalar::<u64>()? as usize;
    let n_features = metadata_group.attr("n_features")?.read_scalar::<u64>()? as usize;

    // Verify this is a regression dataset
    let dataset_type: hdf5::types::VarLenUnicode =
        metadata_group.attr("dataset_type")?.read_scalar()?;
    if dataset_type.as_str() != "regression" {
        return Err(FormatError::InvalidFormat(
            "Expected regression dataset".to_string(),
        ));
    }

    // Read features
    let features_dataset = dataset_group.dataset("features")?;
    let features_vec: Vec<f64> = features_dataset.read_1d()?;

    if features_vec.len() != n_samples * n_features {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples * n_features,
            actual: features_vec.len(),
        });
    }

    let mut features = Array2::zeros((n_samples, n_features));
    for (i, chunk) in features_vec.chunks(n_features).enumerate() {
        for (j, &value) in chunk.iter().enumerate() {
            features[[i, j]] = value;
        }
    }

    // Read targets
    let targets_dataset = dataset_group.dataset("targets")?;
    let targets_vec: Vec<f64> = targets_dataset.read_1d()?;

    if targets_vec.len() != n_samples {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples,
            actual: targets_vec.len(),
        });
    }

    let targets = Array1::from_vec(targets_vec);

    // Read feature names if available
    let feature_names = if metadata_group.link_exists("feature_names") {
        let names_dataset = metadata_group.dataset("feature_names")?;
        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names_dataset.read_1d()?;
        Some(
            names_utf8
                .into_iter()
                .map(|s| s.into_string())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| FormatError::Parse(format!("String conversion error: {}", e)))?,
        )
    } else {
        None
    };

    Ok((features, targets, feature_names))
}


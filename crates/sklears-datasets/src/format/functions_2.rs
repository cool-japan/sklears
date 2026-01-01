//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::path::Path;
#[cfg(feature = "parquet")]
use arrow::array::{Float64Array, Int32Array, StringArray};
#[cfg(feature = "hdf5")]
use hdf5::{Dataset, File as H5File, Group, H5Type};
#[cfg(feature = "cloud-s3")]
use aws_sdk_s3::{Client as S3Client, Config as S3Config};

use super::types::{CloudStorageProvider, FormatError};
use super::functions::FormatResult;

/// Import classification dataset from Parquet format
#[cfg(feature = "parquet")]
pub fn import_classification_parquet<P: AsRef<Path>>(
    path: P,
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    use std::sync::Arc;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    let batch = reader
        .next()
        .ok_or_else(|| FormatError::Parse("No data found in Parquet file".to_string()))?
        .map_err(|e| FormatError::Parse(format!("Failed to read batch: {}", e)))?;
    let schema = batch.schema();
    let n_samples = batch.num_rows();
    let n_fields = batch.num_columns();
    if n_fields < 2 {
        return Err(
            FormatError::Parse(
                "Parquet file must have at least 2 columns (features + target)"
                    .to_string(),
            ),
        );
    }
    let n_features = n_fields - 1;
    let feature_names: Vec<String> = schema
        .fields()[..n_features]
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut features = Array2::zeros((n_samples, n_features));
    for j in 0..n_features {
        let column = batch.column(j);
        let float_array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| FormatError::Parse(format!("Column {} is not float64", j)))?;
        for i in 0..n_samples {
            features[[i, j]] = float_array.value(i);
        }
    }
    let mut targets = Array1::zeros(n_samples);
    let target_column = batch.column(n_features);
    let int_array = target_column
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| FormatError::Parse("Target column is not int32".to_string()))?;
    for i in 0..n_samples {
        targets[i] = int_array.value(i);
    }
    Ok((features, targets, Some(feature_names)))
}
/// Import regression dataset from Parquet format
#[cfg(feature = "parquet")]
pub fn import_regression_parquet<P: AsRef<Path>>(
    path: P,
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    use std::sync::Arc;
    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let mut reader = builder.build()?;
    let batch = reader
        .next()
        .ok_or_else(|| FormatError::Parse("No data found in Parquet file".to_string()))?
        .map_err(|e| FormatError::Parse(format!("Failed to read batch: {}", e)))?;
    let schema = batch.schema();
    let n_samples = batch.num_rows();
    let n_fields = batch.num_columns();
    if n_fields < 2 {
        return Err(
            FormatError::Parse(
                "Parquet file must have at least 2 columns (features + target)"
                    .to_string(),
            ),
        );
    }
    let n_features = n_fields - 1;
    let feature_names: Vec<String> = schema
        .fields()[..n_features]
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let mut features = Array2::zeros((n_samples, n_features));
    for j in 0..n_features {
        let column = batch.column(j);
        let float_array = column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| FormatError::Parse(format!("Column {} is not float64", j)))?;
        for i in 0..n_samples {
            features[[i, j]] = float_array.value(i);
        }
    }
    let mut targets = Array1::zeros(n_samples);
    let target_column = batch.column(n_features);
    let float_array = target_column
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| FormatError::Parse("Target column is not float64".to_string()))?;
    for i in 0..n_samples {
        targets[i] = float_array.value(i);
    }
    Ok((features, targets, Some(feature_names)))
}
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
    let dataset_group = file.create_group("dataset")?;
    let metadata_group = file.create_group("metadata")?;
    let features_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples, n_features])
        .create("features")?;
    let features_vec: Vec<f64> = features.iter().cloned().collect();
    features_dataset.write(&features_vec)?;
    let targets_dataset = dataset_group
        .new_dataset::<i32>()
        .shape([n_samples])
        .create("targets")?;
    targets_dataset.write(targets.as_slice().unwrap())?;
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
    let dataset_group = file.create_group("dataset")?;
    let metadata_group = file.create_group("metadata")?;
    let features_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples, n_features])
        .create("features")?;
    let features_vec: Vec<f64> = features.iter().cloned().collect();
    features_dataset.write(&features_vec)?;
    let targets_dataset = dataset_group
        .new_dataset::<f64>()
        .shape([n_samples])
        .create("targets")?;
    targets_dataset.write(targets.as_slice().unwrap())?;
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
    let n_samples = metadata_group.attr("n_samples")?.read_scalar::<u64>()? as usize;
    let n_features = metadata_group.attr("n_features")?.read_scalar::<u64>()? as usize;
    let dataset_type: hdf5::types::VarLenUnicode = metadata_group
        .attr("dataset_type")?
        .read_scalar()?;
    if dataset_type.as_str() != "classification" {
        return Err(
            FormatError::InvalidFormat("Expected classification dataset".to_string()),
        );
    }
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
    let targets_dataset = dataset_group.dataset("targets")?;
    let targets_vec: Vec<i32> = targets_dataset.read_1d()?;
    if targets_vec.len() != n_samples {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples,
            actual: targets_vec.len(),
        });
    }
    let targets = Array1::from_vec(targets_vec);
    let feature_names = if metadata_group.link_exists("feature_names") {
        let names_dataset = metadata_group.dataset("feature_names")?;
        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names_dataset.read_1d()?;
        Some(
            names_utf8
                .into_iter()
                .map(|s| s.into_string())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| FormatError::Parse(
                    format!("String conversion error: {}", e),
                ))?,
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
    let n_samples = metadata_group.attr("n_samples")?.read_scalar::<u64>()? as usize;
    let n_features = metadata_group.attr("n_features")?.read_scalar::<u64>()? as usize;
    let dataset_type: hdf5::types::VarLenUnicode = metadata_group
        .attr("dataset_type")?
        .read_scalar()?;
    if dataset_type.as_str() != "regression" {
        return Err(
            FormatError::InvalidFormat("Expected regression dataset".to_string()),
        );
    }
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
    let targets_dataset = dataset_group.dataset("targets")?;
    let targets_vec: Vec<f64> = targets_dataset.read_1d()?;
    if targets_vec.len() != n_samples {
        return Err(FormatError::DimensionMismatch {
            expected: n_samples,
            actual: targets_vec.len(),
        });
    }
    let targets = Array1::from_vec(targets_vec);
    let feature_names = if metadata_group.link_exists("feature_names") {
        let names_dataset = metadata_group.dataset("feature_names")?;
        let names_utf8: Vec<hdf5::types::VarLenUnicode> = names_dataset.read_1d()?;
        Some(
            names_utf8
                .into_iter()
                .map(|s| s.into_string())
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| FormatError::Parse(
                    format!("String conversion error: {}", e),
                ))?,
        )
    } else {
        None
    };
    Ok((features, targets, feature_names))
}
#[cfg(feature = "cloud-s3")]
/// Upload classification dataset to AWS S3
pub async fn upload_classification_to_s3(
    bucket: &str,
    key: &str,
    region: &str,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
    format: &str,
) -> FormatResult<()> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));
    match format {
        "csv" => {
            export_classification_csv(
                &temp_path,
                features,
                targets,
                feature_names,
                None,
            )?
        }
        #[cfg(feature = "serde")]
        "json" => {
            export_classification_json(
                &temp_path,
                features,
                targets,
                feature_names,
                None,
            )?
        }
        #[cfg(feature = "parquet")]
        "parquet" => {
            export_classification_parquet(&temp_path, features, targets, feature_names)?
        }
        #[cfg(feature = "hdf5")]
        "hdf5" => {
            export_classification_hdf5(&temp_path, features, targets, feature_names)?
        }
        _ => {
            return Err(
                FormatError::InvalidFormat(format!("Unsupported format: {}", format)),
            );
        }
    }
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;
    let _result = client
        .put_object()
        .bucket(bucket)
        .key(key)
        .body(body.into())
        .send()
        .await
        .map_err(|e| FormatError::S3(format!("S3 upload failed: {}", e)))?;
    Ok(())
}
#[cfg(feature = "cloud-s3")]
/// Download classification dataset from AWS S3
pub async fn download_classification_from_s3(
    bucket: &str,
    key: &str,
    region: &str,
    format: &str,
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);
    let result = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| FormatError::S3(format!("S3 download failed: {}", e)))?;
    let body = result
        .body
        .collect()
        .await
        .map_err(|e| FormatError::S3(
            format!("Failed to read S3 response body: {}", e),
        ))?;
    let data = body.into_bytes();
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));
    tokio::fs::write(&temp_path, data).await.map_err(FormatError::Io)?;
    match format {
        "csv" => import_classification_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_classification_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_classification_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!("Unsupported format: {}", format))),
    }
}
#[cfg(feature = "cloud-storage")]
/// Convenience function to upload classification dataset to cloud storage from URL
pub async fn upload_classification_to_cloud(
    url: &str,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
    format: &str,
) -> FormatResult<()> {
    let provider = CloudStorageProvider::from_url(url)?;
    match provider {
        #[cfg(feature = "cloud-s3")]
        CloudStorageProvider::S3 { bucket, region, key } => {
            upload_classification_to_s3(
                    &bucket,
                    &key,
                    &region,
                    features,
                    targets,
                    feature_names,
                    format,
                )
                .await
        }
        #[cfg(feature = "cloud-gcs")]
        CloudStorageProvider::GoogleCloudStorage { bucket, object_name, project_id } => {
            upload_classification_to_gcs(
                    &bucket,
                    &object_name,
                    &project_id,
                    features,
                    targets,
                    feature_names,
                    format,
                )
                .await
        }
    }
}

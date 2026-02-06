use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "cloud-s3")]
use aws_sdk_s3::{Client as S3Client, Config as S3Config};
#[cfg(feature = "cloud-gcs")]
use google_cloud_storage::client::{Client as GcsClient, ClientConfig as GcsConfig};

use super::types::{CloudStorageProvider, FormatError};
use super::functions::FormatResult;

#[cfg(feature = "cloud-s3")]
/// Upload regression dataset to AWS S3
pub async fn upload_regression_to_s3(
    bucket: &str,
    key: &str,
    region: &str,
    features: &Array2<f64>,
    targets: &Array1<f64>,
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
            export_regression_csv(&temp_path, features, targets, feature_names, None)?
        }
        #[cfg(feature = "serde")]
        "json" => {
            export_regression_json(&temp_path, features, targets, feature_names, None)?
        }
        #[cfg(feature = "parquet")]
        "parquet" => {
            export_regression_parquet(&temp_path, features, targets, feature_names)?
        }
        #[cfg(feature = "hdf5")]
        "hdf5" => export_regression_hdf5(&temp_path, features, targets, feature_names)?,
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
/// Download regression dataset from AWS S3
pub async fn download_regression_from_s3(
    bucket: &str,
    key: &str,
    region: &str,
    format: &str,
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
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
        "csv" => import_regression_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_regression_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_regression_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!("Unsupported format: {}", format))),
    }
}
#[cfg(feature = "cloud-gcs")]
/// Upload classification dataset to Google Cloud Storage
pub async fn upload_classification_to_gcs(
    bucket: &str,
    object_name: &str,
    project_id: &str,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
    format: &str,
) -> FormatResult<()> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);
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
        .upload_object(
            &google_cloud_storage::client::UploadObjectRequest {
                bucket: bucket.to_string(),
                name: object_name.to_string(),
                data: body,
                ..Default::default()
            },
        )
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS upload failed: {}", e)))?;
    Ok(())
}
#[cfg(feature = "cloud-gcs")]
/// Download classification dataset from Google Cloud Storage
pub async fn download_classification_from_gcs(
    bucket: &str,
    object_name: &str,
    project_id: &str,
    format: &str,
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);
    let body = client
        .download_object(
            &google_cloud_storage::client::GetObjectRequest {
                bucket: bucket.to_string(),
                object: object_name.to_string(),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS download failed: {}", e)))?;
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));
    tokio::fs::write(&temp_path, body).await.map_err(FormatError::Io)?;
    match format {
        "csv" => import_classification_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_classification_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_classification_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!("Unsupported format: {}", format))),
    }
}
#[cfg(feature = "cloud-gcs")]
/// Upload regression dataset to Google Cloud Storage
pub async fn upload_regression_to_gcs(
    bucket: &str,
    object_name: &str,
    project_id: &str,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
    format: &str,
) -> FormatResult<()> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));
    match format {
        "csv" => {
            export_regression_csv(&temp_path, features, targets, feature_names, None)?
        }
        #[cfg(feature = "serde")]
        "json" => {
            export_regression_json(&temp_path, features, targets, feature_names, None)?
        }
        #[cfg(feature = "parquet")]
        "parquet" => {
            export_regression_parquet(&temp_path, features, targets, feature_names)?
        }
        #[cfg(feature = "hdf5")]
        "hdf5" => export_regression_hdf5(&temp_path, features, targets, feature_names)?,
        _ => {
            return Err(
                FormatError::InvalidFormat(format!("Unsupported format: {}", format)),
            );
        }
    }
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;
    let _result = client
        .upload_object(
            &google_cloud_storage::client::UploadObjectRequest {
                bucket: bucket.to_string(),
                name: object_name.to_string(),
                data: body,
                ..Default::default()
            },
        )
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS upload failed: {}", e)))?;
    Ok(())
}
#[cfg(feature = "cloud-gcs")]
/// Download regression dataset from Google Cloud Storage
pub async fn download_regression_from_gcs(
    bucket: &str,
    object_name: &str,
    project_id: &str,
    format: &str,
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);
    let body = client
        .download_object(
            &google_cloud_storage::client::GetObjectRequest {
                bucket: bucket.to_string(),
                object: object_name.to_string(),
                ..Default::default()
            },
        )
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS download failed: {}", e)))?;
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));
    tokio::fs::write(&temp_path, body).await.map_err(FormatError::Io)?;
    match format {
        "csv" => import_regression_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_regression_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_regression_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!("Unsupported format: {}", format))),
    }
}
#[cfg(feature = "cloud-storage")]
/// Convenience function to upload regression dataset to cloud storage from URL
pub async fn upload_regression_to_cloud(
    url: &str,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
    format: &str,
) -> FormatResult<()> {
    let provider = CloudStorageProvider::from_url(url)?;
    match provider {
        #[cfg(feature = "cloud-s3")]
        CloudStorageProvider::S3 { bucket, region, key } => {
            upload_regression_to_s3(
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
            upload_regression_to_gcs(
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

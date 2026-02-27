use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

// Cloud Storage Integration
#[cfg(feature = "cloud-storage")]
/// Configuration for cloud storage providers
#[derive(Debug, Clone)]
pub enum CloudStorageProvider {
    #[cfg(feature = "cloud-s3")]
    S3 {

        bucket: String,

        region: String,

        key: String,
    },
    #[cfg(feature = "cloud-gcs")]
    GoogleCloudStorage {

        bucket: String,

        object_name: String,
        project_id: String,
    },
}

#[cfg(feature = "cloud-storage")]
impl CloudStorageProvider {
    /// Parse a cloud storage URL into a provider configuration
    pub fn from_url(url: &str) -> FormatResult<Self> {
        let parsed_url = Url::parse(url)?;

        match parsed_url.scheme() {
            #[cfg(feature = "cloud-s3")]
            "s3" => {
                let bucket = parsed_url
                    .host_str()
                    .ok_or_else(|| FormatError::UrlParse(url::ParseError::EmptyHost))?;
                let key = parsed_url.path().trim_start_matches('/');
                let region = parsed_url
                    .query_pairs()
                    .find(|(name, _)| name == "region")
                    .map(|(_, value)| value.to_string())
                    .unwrap_or_else(|| "us-east-1".to_string());

                Ok(CloudStorageProvider::S3 {
                    bucket: bucket.to_string(),
                    region,
                    key: key.to_string(),
                })
            }
            #[cfg(feature = "cloud-gcs")]
            "gs" => {
                let bucket = parsed_url
                    .host_str()
                    .ok_or_else(|| FormatError::UrlParse(url::ParseError::EmptyHost))?;
                let object_name = parsed_url.path().trim_start_matches('/');
                let project_id = parsed_url
                    .query_pairs()
                    .find(|(name, _)| name == "project")
                    .map(|(_, value)| value.to_string())
                    .ok_or_else(|| {
                        FormatError::CloudStorage("Missing project_id parameter".to_string())
                    })?;

                Ok(CloudStorageProvider::GoogleCloudStorage {
                    bucket: bucket.to_string(),
                    object_name: object_name.to_string(),
                    project_id,
                })
            }
            _ => Err(FormatError::CloudStorage(format!(
                "Unsupported URL scheme: {}",
                parsed_url.scheme()
            ))),
        }
    }
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<()> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);

    // Create temporary file with the dataset
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Export to temporary file based on format
    match format {
        "csv" => export_classification_csv(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "serde")]
        "json" => export_classification_json(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "parquet")]
        "parquet" => export_classification_parquet(&temp_path, features, targets, feature_names)?,
        #[cfg(feature = "hdf5")]
        "hdf5" => export_classification_hdf5(&temp_path, features, targets, feature_names)?,
        _ => {
            return Err(FormatError::InvalidFormat(format!(
                "Unsupported format: {}",
                format
            )))
        }
    }

    // Read file content
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;

    // Upload to S3
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);

    // Download from S3
    let result = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| FormatError::S3(format!("S3 download failed: {}", e)))?;

    // Read body
    let body = result
        .body
        .collect()
        .await
        .map_err(|e| FormatError::S3(format!("Failed to read S3 response body: {}", e)))?;
    let data = body.into_bytes();

    // Create temporary file
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Write data to temporary file
    tokio::fs::write(&temp_path, data)
        .await
        .map_err(FormatError::Io)?;

    // Import from temporary file based on format
    match format {
        "csv" => import_classification_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_classification_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_classification_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!(
            "Unsupported format: {}",
            format
        ))),
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
        CloudStorageProvider::S3 {
            bucket,
            region,
            key,
        } => {
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
        CloudStorageProvider::GoogleCloudStorage {
            bucket,
            object_name,
            project_id,
        } => {
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

#[cfg(feature = "cloud-s3")]
/// Upload regression dataset to AWS S3
pub async fn upload_regression_to_s3(
    bucket: &str,
    key: &str,
    region: &str,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<()> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);

    // Create temporary file with the dataset
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Export to temporary file based on format
    match format {
        "csv" => export_regression_csv(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "serde")]
        "json" => export_regression_json(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "parquet")]
        "parquet" => export_regression_parquet(&temp_path, features, targets, feature_names)?,
        #[cfg(feature = "hdf5")]
        "hdf5" => export_regression_hdf5(&temp_path, features, targets, feature_names)?,
        _ => {
            return Err(FormatError::InvalidFormat(format!(
                "Unsupported format: {}",
                format
            )))
        }
    }

    // Read file content
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;

    // Upload to S3
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    let config = aws_config::defaults(aws_config::BehaviorVersion::latest())
        .region(aws_config::Region::new(region.to_string()))
        .load()
        .await;
    let client = S3Client::new(&config);

    // Download from S3
    let result = client
        .get_object()
        .bucket(bucket)
        .key(key)
        .send()
        .await
        .map_err(|e| FormatError::S3(format!("S3 download failed: {}", e)))?;

    // Read body
    let body = result
        .body
        .collect()
        .await
        .map_err(|e| FormatError::S3(format!("Failed to read S3 response body: {}", e)))?;
    let data = body.into_bytes();

    // Create temporary file
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Write data to temporary file
    tokio::fs::write(&temp_path, data)
        .await
        .map_err(FormatError::Io)?;

    // Import from temporary file based on format
    match format {
        "csv" => import_regression_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_regression_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_regression_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!(
            "Unsupported format: {}",
            format
        ))),
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<()> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);

    // Create temporary file with the dataset
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Export to temporary file based on format
    match format {
        "csv" => export_classification_csv(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "serde")]
        "json" => export_classification_json(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "parquet")]
        "parquet" => export_classification_parquet(&temp_path, features, targets, feature_names)?,
        #[cfg(feature = "hdf5")]
        "hdf5" => export_classification_hdf5(&temp_path, features, targets, feature_names)?,
        _ => {
            return Err(FormatError::InvalidFormat(format!(
                "Unsupported format: {}",
                format
            )))
        }
    }

    // Read file content
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;

    // Upload to GCS
    let _result = client
        .upload_object(&google_cloud_storage::client::UploadObjectRequest {
            bucket: bucket.to_string(),
            name: object_name.to_string(),
            data: body,
            ..Default::default()
        })
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);

    // Download from GCS
    let body = client
        .download_object(&google_cloud_storage::client::GetObjectRequest {
            bucket: bucket.to_string(),
            object: object_name.to_string(),
            ..Default::default()
        })
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS download failed: {}", e)))?;

    // Create temporary file
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Write data to temporary file
    tokio::fs::write(&temp_path, body)
        .await
        .map_err(FormatError::Io)?;

    // Import from temporary file based on format
    match format {
        "csv" => import_classification_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_classification_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_classification_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!(
            "Unsupported format: {}",
            format
        ))),
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<()> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);

    // Create temporary file with the dataset
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Export to temporary file based on format
    match format {
        "csv" => export_regression_csv(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "serde")]
        "json" => export_regression_json(&temp_path, features, targets, feature_names, None)?,
        #[cfg(feature = "parquet")]
        "parquet" => export_regression_parquet(&temp_path, features, targets, feature_names)?,
        #[cfg(feature = "hdf5")]
        "hdf5" => export_regression_hdf5(&temp_path, features, targets, feature_names)?,
        _ => {
            return Err(FormatError::InvalidFormat(format!(
                "Unsupported format: {}",
                format
            )))
        }
    }

    // Read file content
    let body = tokio::fs::read(&temp_path).await.map_err(FormatError::Io)?;

    // Upload to GCS
    let _result = client
        .upload_object(&google_cloud_storage::client::UploadObjectRequest {
            bucket: bucket.to_string(),
            name: object_name.to_string(),
            data: body,
            ..Default::default()
        })
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
    format: &str, // "csv", "json", "parquet", "hdf5"
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    let config = GcsConfig::default()
        .with_auth()
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS authentication failed: {}", e)))?;
    let client = GcsClient::new(config);

    // Download from GCS
    let body = client
        .download_object(&google_cloud_storage::client::GetObjectRequest {
            bucket: bucket.to_string(),
            object: object_name.to_string(),
            ..Default::default()
        })
        .await
        .map_err(|e| FormatError::Gcs(format!("GCS download failed: {}", e)))?;

    // Create temporary file
    let temp_dir = tempfile::tempdir().map_err(|e| FormatError::Io(e))?;
    let temp_path = temp_dir.path().join(format!("dataset.{}", format));

    // Write data to temporary file
    tokio::fs::write(&temp_path, body)
        .await
        .map_err(FormatError::Io)?;

    // Import from temporary file based on format
    match format {
        "csv" => import_regression_csv(&temp_path, None),
        #[cfg(feature = "parquet")]
        "parquet" => import_regression_parquet(&temp_path),
        #[cfg(feature = "hdf5")]
        "hdf5" => import_regression_hdf5(&temp_path),
        _ => Err(FormatError::InvalidFormat(format!(
            "Unsupported format: {}",
            format
        ))),
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
        CloudStorageProvider::S3 {
            bucket,
            region,
            key,
        } => {
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
        CloudStorageProvider::GoogleCloudStorage {
            bucket,
            object_name,
            project_id,
        } => {
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


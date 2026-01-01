//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use thiserror::Error;
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "cloud-storage")]
use url::Url;

use super::functions::FormatResult;


/// Configuration for CSV format
#[derive(Debug, Clone)]
pub struct CsvConfig {
    /// Field delimiter (default: ',')
    pub delimiter: char,
    /// Include header row
    pub has_header: bool,
    /// Quote character for fields containing delimiter
    pub quote_char: char,
    /// Escape character for quotes within quoted fields
    pub escape_char: Option<char>,
}
#[cfg(feature = "cloud-storage")]
/// Configuration for cloud storage providers
#[derive(Debug, Clone)]
pub enum CloudStorageProvider {
    #[cfg(feature = "cloud-s3")]
    S3 { bucket: String, region: String, key: String },
    #[cfg(feature = "cloud-gcs")]
    GoogleCloudStorage { bucket: String, object_name: String, project_id: String },
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
                        FormatError::CloudStorage(
                            "Missing project_id parameter".to_string(),
                        )
                    })?;
                Ok(CloudStorageProvider::GoogleCloudStorage {
                    bucket: bucket.to_string(),
                    object_name: object_name.to_string(),
                    project_id,
                })
            }
            _ => {
                Err(
                    FormatError::CloudStorage(
                        format!("Unsupported URL scheme: {}", parsed_url.scheme()),
                    ),
                )
            }
        }
    }
}
/// Serializable dataset for JSON export (fallback when serde is not available)
#[cfg(not(feature = "serde"))]
#[derive(Debug)]
pub struct SerializableDataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<serde_json::Value>,
    pub feature_names: Option<Vec<String>>,
    pub target_names: Option<Vec<String>>,
    pub metadata: Option<serde_json::Value>,
}
/// Error types for format operations
#[derive(Error, Debug)]
pub enum FormatError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Parse error: {0}")]
    Parse(String),
    #[error("CSV error: {0}")]
    Csv(String),
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Invalid format: {0}")]
    InvalidFormat(String),
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[cfg(feature = "parquet")]
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    #[cfg(feature = "parquet")]
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    #[cfg(feature = "hdf5")]
    #[error("HDF5 error: {0}")]
    Hdf5(#[from] hdf5::Error),
    #[cfg(feature = "cloud-storage")]
    #[error("Cloud storage error: {0}")]
    CloudStorage(String),
    #[cfg(feature = "cloud-storage")]
    #[error("URL parse error: {0}")]
    UrlParse(#[from] url::ParseError),
    #[cfg(feature = "cloud-s3")]
    #[error("S3 error: {0}")]
    S3(String),
    #[cfg(feature = "cloud-gcs")]
    #[error("Google Cloud Storage error: {0}")]
    Gcs(String),
}

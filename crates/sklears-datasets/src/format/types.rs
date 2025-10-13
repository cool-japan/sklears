//! Core types and error handling for dataset format operations
//!
//! This module defines the core types, error enums, and configuration
//! structures used across all format modules.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use thiserror::Error;

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

pub type FormatResult<T> = Result<T, FormatError>;

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

impl Default for CsvConfig {
    fn default() -> Self {
        Self {
            delimiter: ',',
            has_header: true,
            quote_char: '"',
            escape_char: Some('\\'),
        }
    }
}

/// Serializable dataset for JSON export
#[cfg(feature = "serde")]
#[derive(Debug, Serialize, Deserialize)]
pub struct SerializableDataset {
    pub features: Vec<Vec<f64>>,
    pub targets: Vec<serde_json::Value>,
    pub feature_names: Option<Vec<String>>,
    pub target_names: Option<Vec<String>>,
    pub metadata: Option<serde_json::Value>,
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
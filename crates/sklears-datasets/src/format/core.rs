use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

//! Dataset format support for import/export
//!
//! This module provides functionality to export and import datasets in various formats:
//! - CSV (Comma-Separated Values)
//! - JSON (JavaScript Object Notation)
//! - TSV (Tab-Separated Values)
//! - JSONL (JSON Lines)
//! - Parquet (Apache Parquet columnar storage)
//! - HDF5 (Hierarchical Data Format version 5)

use scirs2_core::ndarray::{Array1, Array2};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;
use thiserror::Error;

#[cfg(feature = "parquet")]
use arrow::array::{Float64Array, Int32Array, StringArray};
#[cfg(feature = "parquet")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "parquet")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "parquet")]
use parquet::arrow::{arrow_reader::ParquetRecordBatchReaderBuilder, ArrowWriter};

#[cfg(feature = "hdf5")]
use hdf5::{Dataset, File as H5File, Group, H5Type};

#[cfg(feature = "cloud-s3")]
use aws_config;
#[cfg(feature = "cloud-s3")]
use aws_sdk_s3::{Client as S3Client, Config as S3Config};
#[cfg(feature = "cloud-storage")]
use futures::{stream, StreamExt};
#[cfg(feature = "cloud-gcs")]
use google_cloud_storage::client::{Client as GcsClient, ClientConfig as GcsConfig};
#[cfg(feature = "cloud-storage")]
use tokio;
#[cfg(feature = "cloud-storage")]
use url::Url;

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

/// Export classification dataset to CSV

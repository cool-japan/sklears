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
pub fn export_classification_csv<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
    config: Option<CsvConfig>,
) -> FormatResult<()> {
    let config = config.unwrap_or_default();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (n_samples, n_features) = features.dim();

    // Write header if requested
    if config.has_header {
        if let Some(names) = feature_names {
            if names.len() != n_features {
                return Err(FormatError::DimensionMismatch {
                    expected: n_features,
                    actual: names.len(),
                });
            }
            for (i, name) in names.iter().enumerate() {
                if i > 0 {
                    write!(writer, "{}", config.delimiter)?;
                }
                write_csv_field(&mut writer, name, &config)?;
            }
        } else {
            for i in 0..n_features {
                if i > 0 {
                    write!(writer, "{}", config.delimiter)?;
                }
                write!(writer, "feature_{}", i)?;
            }
        }
        write!(writer, "{}target\n", config.delimiter)?;
    }

    // Write data rows
    for i in 0..n_samples {
        for j in 0..n_features {
            if j > 0 {
                write!(writer, "{}", config.delimiter)?;
            }
            write!(writer, "{}", features[[i, j]])?;
        }
        write!(writer, "{}{}\n", config.delimiter, targets[i])?;
    }

    writer.flush()?;
    Ok(())
}

/// Export regression dataset to CSV
pub fn export_regression_csv<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
    config: Option<CsvConfig>,
) -> FormatResult<()> {
    let config = config.unwrap_or_default();
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);

    let (n_samples, n_features) = features.dim();

    // Write header if requested
    if config.has_header {
        if let Some(names) = feature_names {
            if names.len() != n_features {
                return Err(FormatError::DimensionMismatch {
                    expected: n_features,
                    actual: names.len(),
                });
            }
            for (i, name) in names.iter().enumerate() {
                if i > 0 {
                    write!(writer, "{}", config.delimiter)?;
                }
                write_csv_field(&mut writer, name, &config)?;
            }
        } else {
            for i in 0..n_features {
                if i > 0 {
                    write!(writer, "{}", config.delimiter)?;
                }
                write!(writer, "feature_{}", i)?;
            }
        }
        write!(writer, "{}target\n", config.delimiter)?;
    }

    // Write data rows
    for i in 0..n_samples {
        for j in 0..n_features {
            if j > 0 {
                write!(writer, "{}", config.delimiter)?;
            }
            write!(writer, "{}", features[[i, j]])?;
        }
        write!(writer, "{}{}\n", config.delimiter, targets[i])?;
    }

    writer.flush()?;
    Ok(())
}

/// Import classification dataset from CSV
pub fn import_classification_csv<P: AsRef<Path>>(
    path: P,
    config: Option<CsvConfig>,
) -> FormatResult<(Array2<f64>, Array1<i32>, Option<Vec<String>>)> {
    let config = config.unwrap_or_default();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut feature_names = None;
    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();

    // Handle header if present
    if config.has_header {
        if let Some(header_line) = lines.next() {
            let header = header_line?;
            let fields: Vec<&str> = split_csv_line(&header, &config);
            if !fields.is_empty() {
                feature_names = Some(
                    fields[..fields.len() - 1]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                );
            }
        }
    }

    // Parse data lines
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = split_csv_line(&line, &config);
        if fields.is_empty() {
            continue;
        }

        // Parse features
        let features: Result<Vec<f64>, _> = fields[..fields.len() - 1]
            .iter()
            .map(|s| s.trim().parse::<f64>())
            .collect();

        let features =
            features.map_err(|e| FormatError::Parse(format!("Feature parse error: {}", e)))?;

        // Parse target
        let target = fields[fields.len() - 1]
            .trim()
            .parse::<i32>()
            .map_err(|e| FormatError::Parse(format!("Target parse error: {}", e)))?;

        all_features.push(features);
        all_targets.push(target);
    }

    if all_features.is_empty() {
        return Err(FormatError::Parse("No data found in CSV file".to_string()));
    }

    let n_samples = all_features.len();
    let n_features = all_features[0].len();

    // Verify consistent number of features
    for (i, row) in all_features.iter().enumerate() {
        if row.len() != n_features {
            return Err(FormatError::DimensionMismatch {
                expected: n_features,
                actual: row.len(),
            });
        }
    }

    // Convert to ndarray
    let mut features = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);

    for (i, row) in all_features.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            features[[i, j]] = value;
        }
        targets[i] = all_targets[i];
    }

    Ok((features, targets, feature_names))
}

/// Import regression dataset from CSV
pub fn import_regression_csv<P: AsRef<Path>>(
    path: P,
    config: Option<CsvConfig>,
) -> FormatResult<(Array2<f64>, Array1<f64>, Option<Vec<String>>)> {
    let config = config.unwrap_or_default();
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    let mut feature_names = None;
    let mut all_features = Vec::new();
    let mut all_targets = Vec::new();

    // Handle header if present
    if config.has_header {
        if let Some(header_line) = lines.next() {
            let header = header_line?;
            let fields: Vec<&str> = split_csv_line(&header, &config);
            if !fields.is_empty() {
                feature_names = Some(
                    fields[..fields.len() - 1]
                        .iter()
                        .map(|s| s.to_string())
                        .collect(),
                );
            }
        }
    }

    // Parse data lines
    for line in lines {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = split_csv_line(&line, &config);
        if fields.is_empty() {
            continue;
        }

        // Parse features
        let features: Result<Vec<f64>, _> = fields[..fields.len() - 1]
            .iter()
            .map(|s| s.trim().parse::<f64>())
            .collect();

        let features =
            features.map_err(|e| FormatError::Parse(format!("Feature parse error: {}", e)))?;

        // Parse target
        let target = fields[fields.len() - 1]
            .trim()
            .parse::<f64>()
            .map_err(|e| FormatError::Parse(format!("Target parse error: {}", e)))?;

        all_features.push(features);
        all_targets.push(target);
    }

    if all_features.is_empty() {
        return Err(FormatError::Parse("No data found in CSV file".to_string()));
    }

    let n_samples = all_features.len();
    let n_features = all_features[0].len();

    // Verify consistent number of features
    for (i, row) in all_features.iter().enumerate() {
        if row.len() != n_features {
            return Err(FormatError::DimensionMismatch {
                expected: n_features,
                actual: row.len(),
            });
        }
    }

    // Convert to ndarray
    let mut features = Array2::zeros((n_samples, n_features));
    let mut targets = Array1::zeros(n_samples);

    for (i, row) in all_features.iter().enumerate() {
        for (j, &value) in row.iter().enumerate() {
            features[[i, j]] = value;
        }
        targets[i] = all_targets[i];
    }

    Ok((features, targets, feature_names))
}

/// Export classification dataset to JSON
#[cfg(feature = "serde")]
pub fn export_classification_json<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
    metadata: Option<serde_json::Value>,
) -> FormatResult<()> {
    let (n_samples, n_features) = features.dim();

    let feature_data: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| features[[i, j]]).collect())
        .collect();

    let target_data: Vec<serde_json::Value> = targets
        .iter()
        .map(|&t| serde_json::Value::Number(serde_json::Number::from(t)))
        .collect();

    let dataset = SerializableDataset {
        features: feature_data,
        targets: target_data,
        feature_names: feature_names.map(|names| names.to_vec()),
        target_names: None,
        metadata,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &dataset)?;
    Ok(())
}

/// Export regression dataset to JSON
#[cfg(feature = "serde")]
pub fn export_regression_json<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
    metadata: Option<serde_json::Value>,
) -> FormatResult<()> {
    let (n_samples, n_features) = features.dim();

    let feature_data: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| (0..n_features).map(|j| features[[i, j]]).collect())
        .collect();

    let target_data: Vec<serde_json::Value> = targets
        .iter()
        .map(|&t| {
            serde_json::Value::Number(
                serde_json::Number::from_f64(t).unwrap_or(serde_json::Number::from(0)),
            )
        })
        .collect();

    let dataset = SerializableDataset {
        features: feature_data,
        targets: target_data,
        feature_names: feature_names.map(|names| names.to_vec()),
        target_names: None,
        metadata,
    };

    let file = File::create(path)?;
    let writer = BufWriter::new(file);
    serde_json::to_writer_pretty(writer, &dataset)?;
    Ok(())
}

/// Helper function to write a CSV field with proper quoting
fn write_csv_field<W: Write>(writer: &mut W, field: &str, config: &CsvConfig) -> FormatResult<()> {
    let needs_quotes = field.contains(config.delimiter)
        || field.contains('\n')
        || field.contains('\r')
        || field.contains(config.quote_char);

    if needs_quotes {
        write!(writer, "{}", config.quote_char)?;
        for ch in field.chars() {
            if ch == config.quote_char {
                if let Some(escape) = config.escape_char {
                    write!(writer, "{}", escape)?;
                }
                write!(writer, "{}", config.quote_char)?;
            } else {
                write!(writer, "{}", ch)?;
            }
        }
        write!(writer, "{}", config.quote_char)?;
    } else {
        write!(writer, "{}", field)?;
    }
    Ok(())
}

/// Helper function to split a CSV line respecting quotes
fn split_csv_line<'a>(line: &'a str, config: &CsvConfig) -> Vec<&'a str> {
    // Simplified CSV parsing - in production, use a proper CSV library
    line.split(config.delimiter).map(|s| s.trim()).collect()
}

/// Export dataset to TSV (Tab-Separated Values)
pub fn export_classification_tsv<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    let config = CsvConfig {
        delimiter: '\t',
        has_header: true,
        quote_char: '"',
        escape_char: Some('\\'),
    };
    export_classification_csv(path, features, targets, feature_names, Some(config))
}

/// Export dataset to TSV (Tab-Separated Values)
pub fn export_regression_tsv<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    let config = CsvConfig {
        delimiter: '\t',
        has_header: true,
        quote_char: '"',
        escape_char: Some('\\'),
    };
    export_regression_csv(path, features, targets, feature_names, Some(config))
}

/// Export classification dataset to JSONL (JSON Lines)
pub fn export_classification_jsonl<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    let (n_samples, n_features) = features.dim();

    for i in 0..n_samples {
        let mut record = serde_json::Map::new();

        // Add features
        if let Some(names) = feature_names {
            for (j, name) in names.iter().enumerate() {
                record.insert(
                    name.clone(),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(features[[i, j]])
                            .unwrap_or(serde_json::Number::from(0)),
                    ),
                );
            }
        } else {
            for j in 0..n_features {
                record.insert(
                    format!("feature_{}", j),
                    serde_json::Value::Number(
                        serde_json::Number::from_f64(features[[i, j]])
                            .unwrap_or(serde_json::Number::from(0)),
                    ),
                );
            }
        }

        // Add target
        record.insert(
            "target".to_string(),
            serde_json::Value::Number(serde_json::Number::from(targets[i])),
        );

        writeln!(writer, "{}", serde_json::to_string(&record)?)?;
    }

    writer.flush()?;
    Ok(())
}

/// Export classification dataset to Parquet format
#[cfg(feature = "parquet")]
pub fn export_classification_parquet<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<i32>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    use std::sync::Arc;

    let (n_samples, n_features) = features.dim();

    // Create schema
    let mut fields = Vec::new();

    // Add feature columns
    for i in 0..n_features {
        let field_name = if let Some(names) = feature_names {
            names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", i))
        } else {
            format!("feature_{}", i)
        };
        fields.push(Field::new(field_name, DataType::Float64, false));
    }

    // Add target column
    fields.push(Field::new("target".to_string(), DataType::Int32, false));

    let schema = Schema::new(fields);

    // Create arrays
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::new();

    // Feature arrays
    for j in 0..n_features {
        let column_data: Vec<f64> = (0..n_samples).map(|i| features[[i, j]]).collect();
        arrays.push(Arc::new(Float64Array::from(column_data)));
    }

    // Target array
    let target_data: Vec<i32> = targets.to_vec();
    arrays.push(Arc::new(Int32Array::from(target_data)));

    // Create record batch
    let batch = RecordBatch::try_new(Arc::new(schema), arrays)?;

    // Write to parquet file
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

/// Export regression dataset to Parquet format
#[cfg(feature = "parquet")]
pub fn export_regression_parquet<P: AsRef<Path>>(
    path: P,
    features: &Array2<f64>,
    targets: &Array1<f64>,
    feature_names: Option<&[String]>,
) -> FormatResult<()> {
    use std::sync::Arc;

    let (n_samples, n_features) = features.dim();

    // Create schema
    let mut fields = Vec::new();

    // Add feature columns
    for i in 0..n_features {
        let field_name = if let Some(names) = feature_names {
            names
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("feature_{}", i))
        } else {
            format!("feature_{}", i)
        };
        fields.push(Field::new(field_name, DataType::Float64, false));
    }

    // Add target column
    fields.push(Field::new("target".to_string(), DataType::Float64, false));

    let schema = Schema::new(fields);

    // Create arrays
    let mut arrays: Vec<Arc<dyn arrow::array::Array>> = Vec::new();

    // Feature arrays
    for j in 0..n_features {
        let column_data: Vec<f64> = (0..n_samples).map(|i| features[[i, j]]).collect();
        arrays.push(Arc::new(Float64Array::from(column_data)));
    }

    // Target array
    let target_data: Vec<f64> = targets.to_vec();
    arrays.push(Arc::new(Float64Array::from(target_data)));

    // Create record batch
    let batch = RecordBatch::try_new(Arc::new(schema), arrays)?;

    // Write to parquet file
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None)?;
    writer.write(&batch)?;
    writer.close()?;

    Ok(())
}

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
        return Err(FormatError::Parse(
            "Parquet file must have at least 2 columns (features + target)".to_string(),
        ));
    }

    let n_features = n_fields - 1; // Last column is target

    // Extract feature names
    let feature_names: Vec<String> = schema.fields()[..n_features]
        .iter()
        .map(|field| field.name().clone())
        .collect();

    // Extract features
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

    // Extract targets
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
        return Err(FormatError::Parse(
            "Parquet file must have at least 2 columns (features + target)".to_string(),
        ));
    }

    let n_features = n_fields - 1; // Last column is target

    // Extract feature names
    let feature_names: Vec<String> = schema.fields()[..n_features]
        .iter()
        .map(|field| field.name().clone())
        .collect();

    // Extract features
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

    // Extract targets
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

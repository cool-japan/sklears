use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

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


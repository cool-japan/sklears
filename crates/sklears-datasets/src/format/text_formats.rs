use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

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


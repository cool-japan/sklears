use scirs2_core::ndarray::{Array1, Array2};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

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

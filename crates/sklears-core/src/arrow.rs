/// Arrow format integration for large-scale data processing
///
/// This module provides integration with Apache Arrow format for efficient
/// columnar data processing and interoperability with Arrow-based systems.
use crate::dataset::Dataset;
use crate::error::{Result, SklearsError};
use crate::types::{Array1, Array2, Float};
use std::io::BufWriter;
use std::path::Path;

#[cfg(feature = "arrow")]
use arrow::array::{Array, Float64Array};
#[cfg(feature = "arrow")]
use arrow::array::{Int64Array, StringArray};
#[cfg(feature = "arrow")]
use arrow::compute;
#[cfg(feature = "arrow")]
use arrow::datatypes::{DataType, Field, Schema};
#[cfg(feature = "arrow")]
use arrow::record_batch::RecordBatch;
#[cfg(feature = "arrow")]
use arrow_csv::ReaderBuilder as CsvReaderBuilder;
#[cfg(feature = "arrow")]
use arrow_ipc::{reader::FileReader, writer::FileWriter};
#[cfg(feature = "arrow")]
use std::fs::File;
#[cfg(feature = "arrow")]
use std::io::BufReader;
#[cfg(feature = "arrow")]
use std::sync::Arc;

/// Arrow-based dataset for large-scale data processing
#[cfg(feature = "arrow")]
#[derive(Debug, Clone)]
pub struct ArrowDataset {
    /// Arrow RecordBatch containing the data
    pub batch: RecordBatch,
    /// Names of feature columns
    pub feature_columns: Vec<String>,
    /// Name of target column
    pub target_column: String,
    /// Dataset description
    pub description: String,
}

#[cfg(feature = "arrow")]
impl ArrowDataset {
    /// Create a new ArrowDataset from a RecordBatch
    ///
    /// # Arguments
    /// * `batch` - Arrow RecordBatch containing the data
    /// * `target_column` - Name of the target column
    /// * `feature_columns` - Optional list of feature column names. If None, all columns except target will be used.
    pub fn new(
        batch: RecordBatch,
        target_column: String,
        feature_columns: Option<Vec<String>>,
    ) -> Result<Self> {
        let schema = batch.schema();
        let column_names: Vec<String> = schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect();

        // Validate target column exists
        if !column_names.contains(&target_column) {
            return Err(SklearsError::InvalidInput(format!(
                "Target column '{}' not found in schema",
                target_column
            )));
        }

        // Determine feature columns
        let feature_cols = if let Some(cols) = feature_columns {
            // Validate all specified feature columns exist
            for col in &cols {
                if !column_names.contains(col) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature column '{}' not found in schema",
                        col
                    )));
                }
            }
            cols
        } else {
            // Use all columns except target
            column_names
                .into_iter()
                .filter(|name| name != &target_column)
                .collect()
        };

        if feature_cols.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No feature columns specified or found".to_string(),
            ));
        }

        Ok(Self {
            batch,
            feature_columns: feature_cols,
            target_column,
            description: String::new(),
        })
    }

    /// Set dataset description
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }

    /// Get the number of rows in the dataset
    pub fn num_rows(&self) -> usize {
        self.batch.num_rows()
    }

    /// Get the number of feature columns
    pub fn num_features(&self) -> usize {
        self.feature_columns.len()
    }

    /// Convert to a standard Dataset for use with ML algorithms
    pub fn to_dataset(&self) -> Result<Dataset> {
        let num_rows = self.num_rows();
        let num_features = self.num_features();

        // Extract feature data
        let mut feature_data = Vec::with_capacity(num_rows * num_features);

        for feature_col in &self.feature_columns {
            let column = self.batch.column_by_name(feature_col).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Feature column '{}' not found", feature_col))
            })?;

            let float_array = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| {
                    SklearsError::InvalidInput(format!(
                        "Feature column '{}' is not Float64",
                        feature_col
                    ))
                })?;

            for i in 0..num_rows {
                feature_data.push(float_array.value(i));
            }
        }

        // Reshape from column-major to row-major
        let mut reshaped_data = Vec::with_capacity(num_rows * num_features);
        for i in 0..num_rows {
            for j in 0..num_features {
                reshaped_data.push(feature_data[j * num_rows + i]);
            }
        }

        let features = Array2::from_shape_vec((num_rows, num_features), reshaped_data)
            .map_err(|e| SklearsError::Other(e.to_string()))?;

        // Extract target data
        let target_column = self
            .batch
            .column_by_name(&self.target_column)
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Target column '{}' not found",
                    self.target_column
                ))
            })?;

        let target_array = target_column
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Target column '{}' is not Float64",
                    self.target_column
                ))
            })?;

        let target_data: Vec<Float> = (0..num_rows).map(|i| target_array.value(i)).collect();
        let target = Array1::from_vec(target_data);

        Ok(Dataset::new(features, target)
            .with_feature_names(self.feature_columns.clone())
            .with_description(self.description.clone()))
    }

    /// Load dataset from Arrow IPC file format
    pub fn from_ipc_file<P: AsRef<Path>>(
        path: P,
        target_column: String,
        feature_columns: Option<Vec<String>>,
    ) -> Result<Self> {
        let file = File::open(path).map_err(SklearsError::IoError)?;
        let reader = FileReader::try_new(BufReader::new(file), None)
            .map_err(|e| SklearsError::Other(format!("Arrow IPC read error: {}", e)))?;

        // For simplicity, read the first batch
        // In production, you might want to read all batches or stream them
        let batch = reader
            .into_iter()
            .next()
            .ok_or_else(|| SklearsError::Other("No batches found in IPC file".to_string()))?
            .map_err(|e| SklearsError::Other(format!("Error reading batch: {}", e)))?;

        Self::new(batch, target_column, feature_columns)
            .map(|dataset| dataset.with_description("Dataset loaded from Arrow IPC file"))
    }

    /// Save dataset to Arrow IPC file format
    pub fn to_ipc_file<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(SklearsError::IoError)?;
        let mut writer = FileWriter::try_new(BufWriter::new(file), &self.batch.schema())
            .map_err(|e| SklearsError::Other(format!("Arrow IPC write error: {}", e)))?;

        writer
            .write(&self.batch)
            .map_err(|e| SklearsError::Other(format!("Error writing batch: {}", e)))?;

        writer
            .finish()
            .map_err(|e| SklearsError::Other(format!("Error finishing write: {}", e)))?;

        Ok(())
    }

    /// Load dataset from CSV file using Arrow CSV reader
    pub fn from_csv<P: AsRef<Path>>(
        path: P,
        target_column: String,
        feature_columns: Option<Vec<String>>,
        has_header: bool,
    ) -> Result<Self> {
        let file = File::open(path).map_err(SklearsError::IoError)?;

        let mut reader_builder = CsvReaderBuilder::new(Arc::new(Schema::empty()));
        reader_builder = reader_builder.with_header(has_header);

        let reader = reader_builder
            .build(BufReader::new(file))
            .map_err(|e| SklearsError::Other(format!("CSV reader creation error: {}", e)))?;

        // Read all batches and combine them
        let mut batches = Vec::new();
        for batch_result in reader {
            let batch =
                batch_result.map_err(|e| SklearsError::Other(format!("CSV read error: {}", e)))?;
            batches.push(batch);
        }

        if batches.is_empty() {
            return Err(SklearsError::Other("No data found in CSV file".to_string()));
        }

        // For simplicity, use the first batch
        // In production, you'd want to concatenate all batches
        let batch = batches
            .into_iter()
            .next()
            .expect("iterator should have at least one element");

        Self::new(batch, target_column, feature_columns)
            .map(|dataset| dataset.with_description("Dataset loaded from CSV file"))
    }

    /// Create from a builder pattern for complex Arrow datasets
    pub fn builder() -> ArrowDatasetBuilder {
        ArrowDatasetBuilder::new()
    }
}

/// Builder for creating complex Arrow datasets
#[cfg(feature = "arrow")]
#[derive(Debug)]
pub struct ArrowDatasetBuilder {
    arrays: Vec<Arc<dyn Array>>,
    fields: Vec<Arc<Field>>,
    target_column: Option<String>,
    feature_columns: Option<Vec<String>>,
    description: String,
}

#[cfg(feature = "arrow")]
impl ArrowDatasetBuilder {
    pub fn new() -> Self {
        Self {
            arrays: Vec::new(),
            fields: Vec::new(),
            target_column: None,
            feature_columns: None,
            description: String::new(),
        }
    }

    pub fn add_float64_column(mut self, name: &str, values: Vec<f64>) -> Self {
        self.arrays.push(Arc::new(Float64Array::from(values)));
        self.fields
            .push(Arc::new(Field::new(name, DataType::Float64, false)));
        self
    }

    pub fn add_int64_column(mut self, name: &str, values: Vec<i64>) -> Self {
        self.arrays.push(Arc::new(Int64Array::from(values)));
        self.fields
            .push(Arc::new(Field::new(name, DataType::Int64, false)));
        self
    }

    pub fn add_string_column(mut self, name: &str, values: Vec<Option<String>>) -> Self {
        self.arrays.push(Arc::new(StringArray::from(values)));
        self.fields
            .push(Arc::new(Field::new(name, DataType::Utf8, true)));
        self
    }

    pub fn target_column(mut self, name: String) -> Self {
        self.target_column = Some(name);
        self
    }

    pub fn feature_columns(mut self, names: Vec<String>) -> Self {
        self.feature_columns = Some(names);
        self
    }

    pub fn description<S: Into<String>>(mut self, desc: S) -> Self {
        self.description = desc.into();
        self
    }

    pub fn build(self) -> Result<ArrowDataset> {
        if self.arrays.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No columns added to dataset".to_string(),
            ));
        }

        let target_column = self
            .target_column
            .ok_or_else(|| SklearsError::InvalidInput("Target column not specified".to_string()))?;

        let schema = Arc::new(Schema::new(self.fields));
        let batch = RecordBatch::try_new(schema, self.arrays)
            .map_err(|e| SklearsError::Other(format!("Failed to create RecordBatch: {}", e)))?;

        ArrowDataset::new(batch, target_column, self.feature_columns)
            .map(|dataset| dataset.with_description(self.description))
    }
}

#[cfg(feature = "arrow")]
impl Default for ArrowDatasetBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "arrow")]
impl ArrowDataset {
    /// Slice the dataset to get a subset of rows
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        let sliced_batch = self.batch.slice(offset, length);

        Self::new(
            sliced_batch,
            self.target_column.clone(),
            Some(self.feature_columns.clone()),
        )
        .map(|dataset| {
            dataset.with_description(format!(
                "Slice of {} (offset={}, length={})",
                self.description, offset, length
            ))
        })
    }

    /// Filter rows based on a boolean mask
    pub fn filter(&self, predicate: &[bool]) -> Result<Self> {
        if predicate.len() != self.num_rows() {
            return Err(SklearsError::InvalidInput(format!(
                "Predicate length {} doesn't match dataset rows {}",
                predicate.len(),
                self.num_rows()
            )));
        }

        // Convert boolean mask to indices
        let indices: Vec<u32> = predicate
            .iter()
            .enumerate()
            .filter_map(|(i, &keep)| if keep { Some(i as u32) } else { None })
            .collect();

        if indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Filter predicate resulted in empty dataset".to_string(),
            ));
        }

        // Create index array
        let index_array = arrow::array::UInt32Array::from(indices);

        // Filter the batch using compute kernels
        let filtered_batch = compute::take_record_batch(&self.batch, &index_array)
            .map_err(|e| SklearsError::Other(format!("Filter operation failed: {}", e)))?;

        Self::new(
            filtered_batch,
            self.target_column.clone(),
            Some(self.feature_columns.clone()),
        )
        .map(|dataset| dataset.with_description(format!("Filtered {}", self.description)))
    }

    /// Get column statistics (mean, std, min, max) for numeric columns
    pub fn describe(&self) -> Result<Vec<ColumnStats>> {
        let mut stats = Vec::new();

        // Collect stats for feature columns
        for col_name in &self.feature_columns {
            let column = self.batch.column_by_name(col_name).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Column '{}' not found", col_name))
            })?;

            if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                let col_stats = compute_column_stats(col_name, float_array)?;
                stats.push(col_stats);
            }
        }

        // Add stats for target column
        let target_col = self
            .batch
            .column_by_name(&self.target_column)
            .ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Target column '{}' not found",
                    self.target_column
                ))
            })?;

        if let Some(float_array) = target_col.as_any().downcast_ref::<Float64Array>() {
            let target_stats = compute_column_stats(&self.target_column, float_array)?;
            stats.push(target_stats);
        }

        Ok(stats)
    }

    /// Apply Arrow compute operations for feature engineering
    #[allow(clippy::type_complexity)]
    pub fn with_computed_columns(
        &self,
        computations: Vec<(
            &str,
            fn(&RecordBatch) -> arrow::error::Result<Arc<dyn Array>>,
        )>,
    ) -> Result<Self> {
        let mut arrays = Vec::new();
        let mut fields = Vec::new();

        // Copy existing columns
        for (i, field) in self.batch.schema().fields().iter().enumerate() {
            arrays.push(self.batch.column(i).clone());
            fields.push(field.clone());
        }

        // Add computed columns
        for (column_name, compute_fn) in &computations {
            let computed_array = compute_fn(&self.batch).map_err(|e| {
                SklearsError::Other(format!(
                    "Failed to compute column '{}': {}",
                    *column_name, e
                ))
            })?;

            let field = Field::new(*column_name, computed_array.data_type().clone(), true);
            arrays.push(computed_array);
            fields.push(Arc::new(field));
        }

        let new_schema = Arc::new(Schema::new(fields));
        let new_batch = RecordBatch::try_new(new_schema, arrays)
            .map_err(|e| SklearsError::Other(format!("Failed to create new batch: {}", e)))?;

        // Update feature columns to include new computed columns
        let mut new_feature_columns = self.feature_columns.clone();
        for (column_name, _) in &computations {
            if *column_name != self.target_column {
                new_feature_columns.push((*column_name).to_string());
            }
        }

        Ok(Self {
            batch: new_batch,
            feature_columns: new_feature_columns,
            target_column: self.target_column.clone(),
            description: format!("Enhanced {}", self.description),
        })
    }

    /// Aggregate data using Arrow compute kernels
    pub fn aggregate(
        &self,
        group_by_columns: &[String],
        aggregations: Vec<(&str, AggregationType)>,
    ) -> Result<Self> {
        // This is a simplified aggregation - in practice you'd use more sophisticated grouping
        let mut agg_arrays = Vec::new();
        let mut agg_fields = Vec::new();

        for group_col in group_by_columns {
            let group_array = self.batch.column_by_name(group_col).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Group column '{}' not found", group_col))
            })?;
            agg_arrays.push(group_array.clone());

            let field_idx = self
                .batch
                .schema()
                .index_of(group_col)
                .map_err(|e| SklearsError::Other(format!("Schema error: {}", e)))?;
            agg_fields.push(self.batch.schema().field(field_idx).clone());
        }

        for (col_name, agg_type) in aggregations {
            let column = self.batch.column_by_name(col_name).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Column '{}' not found", col_name))
            })?;

            if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                let agg_result = match agg_type {
                    AggregationType::Sum => {
                        let sum = compute::sum(float_array).unwrap_or(0.0);
                        Arc::new(Float64Array::from(vec![sum])) as Arc<dyn Array>
                    }
                    AggregationType::Mean => {
                        let count = float_array.len() - float_array.null_count();
                        let sum = compute::sum(float_array).unwrap_or(0.0);
                        let mean = if count > 0 { sum / count as f64 } else { 0.0 };
                        Arc::new(Float64Array::from(vec![mean])) as Arc<dyn Array>
                    }
                    AggregationType::Min => {
                        let min_val = compute::min(float_array).unwrap_or(0.0);
                        Arc::new(Float64Array::from(vec![min_val])) as Arc<dyn Array>
                    }
                    AggregationType::Max => {
                        let max_val = compute::max(float_array).unwrap_or(0.0);
                        Arc::new(Float64Array::from(vec![max_val])) as Arc<dyn Array>
                    }
                    AggregationType::Count => {
                        let count = float_array.len() - float_array.null_count();
                        Arc::new(Float64Array::from(vec![count as f64])) as Arc<dyn Array>
                    }
                };

                agg_arrays.push(agg_result);
                let agg_field_name = format!("{}_{:?}", col_name, agg_type).to_lowercase();
                agg_fields.push(Field::new(&agg_field_name, DataType::Float64, true));
            }
        }

        let agg_schema = Arc::new(Schema::new(agg_fields));
        let agg_batch = RecordBatch::try_new(agg_schema, agg_arrays).map_err(|e| {
            SklearsError::Other(format!("Failed to create aggregated batch: {}", e))
        })?;

        // Determine new feature and target columns
        let new_feature_columns: Vec<String> = agg_batch
            .schema()
            .fields()
            .iter()
            .filter(|field| field.name() != &self.target_column)
            .map(|field| field.name().clone())
            .collect();

        let field_names: Vec<String> = agg_batch
            .schema()
            .fields()
            .iter()
            .map(|f| f.name().clone())
            .collect();
        let new_target_column = if field_names.contains(&self.target_column) {
            self.target_column.clone()
        } else {
            // If target column doesn't exist in aggregation, use the first numeric column
            new_feature_columns.first().cloned().unwrap_or_default()
        };

        Ok(Self {
            batch: agg_batch,
            feature_columns: new_feature_columns,
            target_column: new_target_column,
            description: format!("Aggregated {}", self.description),
        })
    }

    /// Join with another ArrowDataset
    pub fn join(
        &self,
        other: &ArrowDataset,
        join_keys: &[String],
        join_type: JoinType,
    ) -> Result<Self> {
        // This is a simplified join implementation
        // In practice, you'd use more sophisticated Arrow join operations

        // For now, implement a basic hash join for inner joins
        if matches!(join_type, JoinType::Inner) && join_keys.len() == 1 {
            let join_key = &join_keys[0];

            // Get join columns
            let _left_join_col = self.batch.column_by_name(join_key).ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Join key '{}' not found in left dataset",
                    join_key
                ))
            })?;
            let _right_join_col = other.batch.column_by_name(join_key).ok_or_else(|| {
                SklearsError::InvalidInput(format!(
                    "Join key '{}' not found in right dataset",
                    join_key
                ))
            })?;

            // For simplicity, assume the join results in the concatenation of all columns
            // In a real implementation, you'd perform proper join logic
            let mut joined_arrays = Vec::new();
            let mut joined_fields = Vec::new();

            // Add left table columns
            for (i, field) in self.batch.schema().fields().iter().enumerate() {
                joined_arrays.push(self.batch.column(i).clone());
                joined_fields.push(field.clone());
            }

            // Add right table columns (excluding join key to avoid duplication)
            for (i, field) in other.batch.schema().fields().iter().enumerate() {
                if field.name() != join_key {
                    let prefixed_name = format!("right_{}", field.name());
                    let new_field = Arc::new(Field::new(
                        &prefixed_name,
                        field.data_type().clone(),
                        field.is_nullable(),
                    ));
                    joined_arrays.push(other.batch.column(i).clone());
                    joined_fields.push(new_field);
                }
            }

            let joined_schema = Arc::new(Schema::new(joined_fields));
            let joined_batch = RecordBatch::try_new(joined_schema, joined_arrays).map_err(|e| {
                SklearsError::Other(format!("Failed to create joined batch: {}", e))
            })?;

            // Update feature columns
            let mut new_feature_columns = self.feature_columns.clone();
            for field in other.batch.schema().fields() {
                let field_name = field.name();
                // Include all columns from right except join key
                // Include right target column if it's also a feature column
                if field_name != join_key {
                    new_feature_columns.push(format!("right_{}", field_name));
                }
            }

            Ok(Self {
                batch: joined_batch,
                feature_columns: new_feature_columns,
                target_column: self.target_column.clone(),
                description: format!("Joined {} with {}", self.description, other.description),
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Only inner joins with single key are currently supported".to_string(),
            ))
        }
    }

    /// Convert to Parquet format for efficient storage
    #[cfg(feature = "arrow")]
    pub fn to_parquet<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use arrow_ipc::writer::FileWriter;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path).map_err(SklearsError::IoError)?;
        let mut writer = FileWriter::try_new(BufWriter::new(file), &self.batch.schema())
            .map_err(|e| SklearsError::Other(format!("Parquet writer creation failed: {}", e)))?;

        writer
            .write(&self.batch)
            .map_err(|e| SklearsError::Other(format!("Parquet write failed: {}", e)))?;

        writer
            .finish()
            .map_err(|e| SklearsError::Other(format!("Parquet finish failed: {}", e)))?;

        Ok(())
    }

    /// Create multiple datasets from time series windows
    pub fn create_time_windows(
        &self,
        time_column: &str,
        window_size: usize,
        step_size: usize,
    ) -> Result<Vec<ArrowDataset>> {
        let _time_col = self.batch.column_by_name(time_column).ok_or_else(|| {
            SklearsError::InvalidInput(format!("Time column '{}' not found", time_column))
        })?;

        let total_rows = self.batch.num_rows();
        let mut windows = Vec::new();

        let mut start = 0;
        while start + window_size <= total_rows {
            let end = start + window_size;

            // Create a slice of the batch for this window
            let window_batch = self.batch.slice(start, window_size);

            let window_dataset = ArrowDataset {
                batch: window_batch,
                feature_columns: self.feature_columns.clone(),
                target_column: self.target_column.clone(),
                description: format!("Time window {}-{} of {}", start, end, self.description),
            };

            windows.push(window_dataset);
            start += step_size;
        }

        Ok(windows)
    }

    /// Create batches for streaming processing
    pub fn batches(&self, batch_size: usize) -> impl Iterator<Item = Result<ArrowDataset>> + '_ {
        let num_rows = self.num_rows();
        (0..num_rows).step_by(batch_size).map(move |offset| {
            let length = std::cmp::min(batch_size, num_rows - offset);
            self.slice(offset, length)
        })
    }

    /// Convert to polars DataFrame for interoperability
    #[cfg(feature = "polars")]
    pub fn to_polars(&self) -> Result<polars::prelude::DataFrame> {
        use polars::prelude::NamedFrom;

        let mut series_vec = Vec::new();

        for field in self.batch.schema().fields() {
            let column = self.batch.column_by_name(field.name()).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Column '{}' not found", field.name()))
            })?;

            match field.data_type() {
                DataType::Float64 => {
                    if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                        let values: Vec<f64> = (0..float_array.len())
                            .map(|i| float_array.value(i))
                            .collect();
                        let series = polars::prelude::Series::new(field.name().into(), values);
                        series_vec.push(series.into());
                    }
                }
                DataType::Int64 => {
                    if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
                        let values: Vec<i64> =
                            (0..int_array.len()).map(|i| int_array.value(i)).collect();
                        let series = polars::prelude::Series::new(field.name().into(), values);
                        series_vec.push(series.into());
                    }
                }
                DataType::Utf8 => {
                    if let Some(string_array) = column.as_any().downcast_ref::<StringArray>() {
                        let values: Vec<Option<&str>> = (0..string_array.len())
                            .map(|i| {
                                if string_array.is_null(i) {
                                    None
                                } else {
                                    Some(string_array.value(i))
                                }
                            })
                            .collect();
                        let series = polars::prelude::Series::new(field.name().into(), values);
                        series_vec.push(series.into());
                    }
                }
                _ => {
                    // For unsupported types, skip or convert to string representation
                    continue;
                }
            }
        }

        polars::prelude::DataFrame::new(series_vec)
            .map_err(|e| SklearsError::Other(format!("Failed to create polars DataFrame: {}", e)))
    }

    /// Create from polars DataFrame for interoperability
    #[cfg(feature = "polars")]
    pub fn from_polars(
        df: &polars::prelude::DataFrame,
        target_column: String,
        feature_columns: Option<Vec<String>>,
    ) -> Result<Self> {
        let mut arrays: Vec<Arc<dyn Array>> = Vec::new();
        let mut fields: Vec<Arc<arrow::datatypes::Field>> = Vec::new();

        for column in df.get_columns() {
            let field_name = column.name();

            match column.dtype() {
                polars::datatypes::DataType::Float64 => {
                    let values: Vec<f64> = column
                        .f64()
                        .map_err(|e| {
                            SklearsError::Other(format!("Failed to extract f64 column: {}", e))
                        })?
                        .into_iter()
                        .map(|opt_val| opt_val.unwrap_or(0.0))
                        .collect();

                    arrays.push(Arc::new(Float64Array::from(values)));
                    fields.push(Arc::new(arrow::datatypes::Field::new(
                        field_name.to_string(),
                        arrow::datatypes::DataType::Float64,
                        false,
                    )));
                }
                polars::datatypes::DataType::Int64 => {
                    let values: Vec<i64> = column
                        .i64()
                        .map_err(|e| {
                            SklearsError::Other(format!("Failed to extract i64 column: {}", e))
                        })?
                        .into_iter()
                        .map(|opt_val| opt_val.unwrap_or(0))
                        .collect();

                    arrays.push(Arc::new(Int64Array::from(values)));
                    fields.push(Arc::new(arrow::datatypes::Field::new(
                        field_name.to_string(),
                        arrow::datatypes::DataType::Int64,
                        false,
                    )));
                }
                polars::datatypes::DataType::String => {
                    let values: Vec<Option<String>> = column
                        .str()
                        .map_err(|e| {
                            SklearsError::Other(format!("Failed to extract string column: {}", e))
                        })?
                        .into_iter()
                        .map(|opt_val| opt_val.map(|s| s.to_string()))
                        .collect();

                    arrays.push(Arc::new(StringArray::from(values)));
                    fields.push(Arc::new(arrow::datatypes::Field::new(
                        field_name.to_string(),
                        arrow::datatypes::DataType::Utf8,
                        true,
                    )));
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unsupported polars data type for column '{}': {:?}",
                        field_name,
                        column.dtype()
                    )));
                }
            }
        }

        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .map_err(|e| SklearsError::Other(format!("Failed to create RecordBatch: {}", e)))?;

        Self::new(batch, target_column, feature_columns)
    }

    /// Save to multiple formats
    pub fn save_to_formats<P: AsRef<Path>>(&self, base_path: P) -> Result<()> {
        let base = base_path.as_ref();

        // Save as IPC/Arrow format
        let ipc_path = base.with_extension("arrow");
        self.to_ipc_file(&ipc_path)?;

        // Save as CSV
        let csv_path = base.with_extension("csv");
        self.to_csv(&csv_path)?;

        // Save as JSON (record-oriented)
        let json_path = base.with_extension("json");
        self.to_json(&json_path)?;

        Ok(())
    }

    /// Export to CSV format
    pub fn to_csv<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use arrow_csv::writer::Writer;
        use std::fs::File;

        let file = File::create(path).map_err(SklearsError::IoError)?;
        let mut writer = Writer::new(file);

        writer
            .write(&self.batch)
            .map_err(|e| SklearsError::Other(format!("CSV write failed: {}", e)))?;

        Ok(())
    }

    /// Export to JSON format (one record per line)
    pub fn to_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        use std::fs::File;
        use std::io::Write;

        let file = File::create(path).map_err(SklearsError::IoError)?;
        let mut writer = BufWriter::new(file);

        // Convert to JSON records
        for row_idx in 0..self.num_rows() {
            let mut record = std::collections::HashMap::new();

            for (col_idx, field) in self.batch.schema().fields().iter().enumerate() {
                let column = self.batch.column(col_idx);
                let field_name = field.name();

                match field.data_type() {
                    DataType::Float64 => {
                        if let Some(float_array) = column.as_any().downcast_ref::<Float64Array>() {
                            record.insert(
                                field_name.clone(),
                                format!("{}", float_array.value(row_idx)),
                            );
                        }
                    }
                    DataType::Int64 => {
                        if let Some(int_array) = column.as_any().downcast_ref::<Int64Array>() {
                            record.insert(
                                field_name.clone(),
                                format!("{}", int_array.value(row_idx)),
                            );
                        }
                    }
                    DataType::Utf8 => {
                        if let Some(string_array) = column.as_any().downcast_ref::<StringArray>() {
                            if !string_array.is_null(row_idx) {
                                let value = string_array.value(row_idx);
                                record.insert(field_name.clone(), format!("\"{}\"", value));
                            }
                        }
                    }
                    _ => {}
                }
            }

            // Write as JSON object
            let json_str = format!(
                "{{{}}}\n",
                record
                    .iter()
                    .map(|(k, v)| format!("\"{}\": {}", k, v))
                    .collect::<Vec<_>>()
                    .join(", ")
            );

            writer
                .write_all(json_str.as_bytes())
                .map_err(SklearsError::IoError)?;
        }

        writer.flush().map_err(SklearsError::IoError)?;
        Ok(())
    }

    /// Load from multiple formats based on file extension
    pub fn load_from_file<P: AsRef<Path>>(
        path: P,
        target_column: String,
        feature_columns: Option<Vec<String>>,
    ) -> Result<Self> {
        let path_ref = path.as_ref();
        let extension = path_ref
            .extension()
            .and_then(|ext| ext.to_str())
            .ok_or_else(|| SklearsError::InvalidInput("No file extension found".to_string()))?;

        match extension.to_lowercase().as_str() {
            "arrow" | "ipc" => Self::from_ipc_file(path_ref, target_column, feature_columns),
            "csv" => Self::from_csv(path_ref, target_column, feature_columns, true),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unsupported file format: {}",
                extension
            ))),
        }
    }
}

/// Aggregation types for Arrow operations
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub enum AggregationType {
    Sum,
    Mean,
    Min,
    Max,
    Count,
}

/// Join types for Arrow operations
#[cfg(feature = "arrow")]
#[derive(Debug, Clone, Copy)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

/// Column statistics for numeric data
#[cfg(feature = "arrow")]
#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub name: String,
    pub count: usize,
    pub mean: Float,
    pub std: Float,
    pub min: Float,
    pub max: Float,
    pub null_count: usize,
}

#[cfg(feature = "arrow")]
fn compute_column_stats(name: &str, array: &Float64Array) -> Result<ColumnStats> {
    let count = array.len();
    let null_count = array.null_count();
    let valid_count = count - null_count;

    if valid_count == 0 {
        return Ok(ColumnStats {
            name: name.to_string(),
            count,
            mean: 0.0,
            std: 0.0,
            min: 0.0,
            max: 0.0,
            null_count,
        });
    }

    // Compute basic statistics
    let mut sum = 0.0;
    let mut min_val = Float::INFINITY;
    let mut max_val = Float::NEG_INFINITY;

    for i in 0..count {
        if array.is_valid(i) {
            let val = array.value(i);
            sum += val;
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }
    }

    let mean = sum / valid_count as Float;

    // Compute standard deviation
    let mut sum_sq_diff = 0.0;
    for i in 0..count {
        if array.is_valid(i) {
            let diff = array.value(i) - mean;
            sum_sq_diff += diff * diff;
        }
    }
    let std = if valid_count > 1 {
        (sum_sq_diff / (valid_count - 1) as Float).sqrt()
    } else {
        0.0
    };

    Ok(ColumnStats {
        name: name.to_string(),
        count,
        mean,
        std,
        min: min_val,
        max: max_val,
        null_count,
    })
}

/// Convert a standard Dataset to ArrowDataset
#[cfg(feature = "arrow")]
impl From<Dataset> for ArrowDataset {
    fn from(dataset: Dataset) -> Self {
        let (num_rows, num_features) = dataset.data.dim();

        // Create feature columns
        let mut fields = Vec::with_capacity(num_features + 1);
        let mut arrays: Vec<Arc<dyn Array>> = Vec::with_capacity(num_features + 1);

        for (j, feature_name) in dataset.feature_names.iter().enumerate() {
            let column_data: Vec<Float> = (0..num_rows).map(|i| dataset.data[[i, j]]).collect();

            let array = Arc::new(Float64Array::from(column_data));
            fields.push(Field::new(feature_name, DataType::Float64, false));
            arrays.push(array);
        }

        // Add target column
        let target_data: Vec<Float> = dataset.target.to_vec();
        let target_array = Arc::new(Float64Array::from(target_data));
        fields.push(Field::new("target", DataType::Float64, false));
        arrays.push(target_array);

        // Create schema and record batch
        let schema = Arc::new(Schema::new(fields));
        let batch = RecordBatch::try_new(schema, arrays)
            .expect("Failed to create RecordBatch from Dataset");

        ArrowDataset {
            batch,
            feature_columns: dataset.feature_names,
            target_column: "target".to_string(),
            description: dataset.description,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
#[cfg(feature = "arrow")]
mod tests {
    use super::*;
    use arrow::array::Float64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use std::sync::Arc;

    fn create_test_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("feature1", DataType::Float64, false),
            Field::new("feature2", DataType::Float64, false),
            Field::new("target", DataType::Float64, false),
        ]));

        let feature1 = Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0]));
        let feature2 = Arc::new(Float64Array::from(vec![4.0, 5.0, 6.0]));
        let target = Arc::new(Float64Array::from(vec![7.0, 8.0, 9.0]));

        RecordBatch::try_new(schema, vec![feature1, feature2, target])
            .expect("valid RecordBatch construction")
    }

    #[test]
    fn test_arrow_dataset_creation() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        assert_eq!(dataset.num_rows(), 3);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.target_column, "target");
        assert_eq!(dataset.feature_columns, vec!["feature1", "feature2"]);
    }

    #[test]
    fn test_arrow_to_dataset_conversion() {
        let batch = create_test_batch();
        let arrow_dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let dataset = arrow_dataset
            .to_dataset()
            .expect("to_dataset should succeed");

        assert_eq!(dataset.data.shape(), &[3, 2]);
        assert_eq!(dataset.target.len(), 3);
        assert_eq!(dataset.feature_names, vec!["feature1", "feature2"]);

        // Check data values
        assert_eq!(dataset.data[[0, 0]], 1.0);
        assert_eq!(dataset.data[[0, 1]], 4.0);
        assert_eq!(dataset.data[[1, 0]], 2.0);
        assert_eq!(dataset.data[[1, 1]], 5.0);
        assert_eq!(dataset.target[0], 7.0);
        assert_eq!(dataset.target[1], 8.0);
    }

    #[test]
    fn test_arrow_dataset_slice() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let sliced = dataset.slice(1, 2).expect("slice should succeed");
        assert_eq!(sliced.num_rows(), 2);

        let sliced_dataset = sliced.to_dataset().expect("to_dataset should succeed");
        assert_eq!(sliced_dataset.data[[0, 0]], 2.0); // Second row of original
        assert_eq!(sliced_dataset.target[0], 8.0);
    }

    #[test]
    fn test_arrow_dataset_filter() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let filtered = dataset
            .filter(&[true, false, true])
            .expect("filter should succeed");
        assert_eq!(filtered.num_rows(), 2);

        let filtered_dataset = filtered.to_dataset().expect("to_dataset should succeed");
        assert_eq!(filtered_dataset.data[[0, 0]], 1.0); // First row
        assert_eq!(filtered_dataset.data[[1, 0]], 3.0); // Third row
        assert_eq!(filtered_dataset.target[0], 7.0);
        assert_eq!(filtered_dataset.target[1], 9.0);
    }

    #[test]
    fn test_dataset_to_arrow_conversion() {
        use scirs2_core::ndarray::Array;

        let features =
            Array::from_shape_vec((2, 2), vec![1.0, 3.0, 2.0, 4.0]).expect("valid array shape");
        let targets = Array::from_vec(vec![5.0, 6.0]);

        let dataset = Dataset::new(features, targets)
            .with_feature_names(vec!["f1".to_string(), "f2".to_string()]);

        let arrow_dataset: ArrowDataset = dataset.into();

        assert_eq!(arrow_dataset.num_rows(), 2);
        assert_eq!(arrow_dataset.num_features(), 2);
        assert_eq!(arrow_dataset.feature_columns, vec!["f1", "f2"]);
        assert_eq!(arrow_dataset.target_column, "target");
    }

    #[test]
    fn test_column_stats() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let stats = dataset.describe().expect("describe should succeed");
        assert_eq!(stats.len(), 3); // 2 features + 1 target

        let feature1_stats = &stats[0];
        assert_eq!(feature1_stats.name, "feature1");
        assert_eq!(feature1_stats.count, 3);
        assert_eq!(feature1_stats.mean, 2.0);
        assert_eq!(feature1_stats.min, 1.0);
        assert_eq!(feature1_stats.max, 3.0);
    }

    #[test]
    fn test_arrow_aggregation() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let aggregated = dataset
            .aggregate(
                &[],
                vec![
                    ("feature1", AggregationType::Mean),
                    ("feature2", AggregationType::Sum),
                ],
            )
            .expect("expected valid value");

        assert_eq!(aggregated.num_rows(), 1);
        let schema = aggregated.batch.schema();
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert!(field_names.contains(&"feature1_mean"));
        assert!(field_names.contains(&"feature2_sum"));
    }

    #[test]
    fn test_arrow_computed_columns() {
        let batch = create_test_batch();
        let dataset = ArrowDataset::new(
            batch,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        let enhanced = dataset
            .with_computed_columns(vec![("feature1_squared", |batch: &RecordBatch| {
                let feature1 = batch.column_by_name("feature1").expect("column exists");
                let float_array = feature1
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .expect("valid downcast");
                let squared_values: Vec<f64> = float_array
                    .iter()
                    .map(|opt_val| opt_val.map(|v| v * v).unwrap_or(0.0))
                    .collect();
                Ok(Arc::new(Float64Array::from(squared_values)) as Arc<dyn Array>)
            })])
            .expect("expected valid value");

        let enhanced_schema = enhanced.batch.schema();
        let enhanced_field_names: Vec<&str> = enhanced_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(enhanced_field_names.contains(&"feature1_squared"));
        assert_eq!(enhanced.num_features(), 3); // Original 2 + 1 computed
    }

    #[test]
    fn test_arrow_time_windows() {
        use arrow::array::Int64Array;

        let schema = Arc::new(Schema::new(vec![
            Field::new("timestamp", DataType::Int64, false),
            Field::new("value", DataType::Float64, false),
            Field::new("target", DataType::Float64, false),
        ]));

        let timestamp = Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5, 6]));
        let value = Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]));
        let target = Arc::new(Float64Array::from(vec![
            100.0, 200.0, 300.0, 400.0, 500.0, 600.0,
        ]));

        let batch = RecordBatch::try_new(schema, vec![timestamp, value, target])
            .expect("valid RecordBatch construction");

        let dataset =
            ArrowDataset::new(batch, "target".to_string(), Some(vec!["value".to_string()]))
                .expect("expected valid value");

        let windows = dataset
            .create_time_windows("timestamp", 3, 2)
            .expect("create_time_windows should succeed");

        assert_eq!(windows.len(), 2); // (6-3)/2 + 1 = 2 windows
        assert_eq!(windows[0].num_rows(), 3);
        assert_eq!(windows[1].num_rows(), 3);
    }

    #[test]
    fn test_arrow_join() {
        // Create left dataset
        let left_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("feature1", DataType::Float64, false),
            Field::new("target", DataType::Float64, false),
        ]));

        let left_batch = RecordBatch::try_new(
            left_schema,
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![10.0, 20.0, 30.0])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 300.0])),
            ],
        )
        .expect("expected valid value");

        // Create right dataset
        let right_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("feature2", DataType::Float64, false),
        ]));

        let right_batch = RecordBatch::try_new(
            right_schema,
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 3])),
                Arc::new(Float64Array::from(vec![40.0, 50.0, 60.0])),
            ],
        )
        .expect("expected valid value");

        let left_dataset = ArrowDataset::new(
            left_batch,
            "target".to_string(),
            Some(vec!["feature1".to_string()]),
        )
        .expect("expected valid value");

        let right_dataset = ArrowDataset::new(
            right_batch,
            "feature2".to_string(),
            Some(vec!["feature2".to_string()]),
        )
        .expect("expected valid value");

        let joined = left_dataset
            .join(&right_dataset, &["id".to_string()], JoinType::Inner)
            .expect("expected valid value");

        let joined_schema = joined.batch.schema();
        let joined_field_names: Vec<&str> = joined_schema
            .fields()
            .iter()
            .map(|f| f.name().as_str())
            .collect();
        assert!(joined_field_names.contains(&"right_feature2"));
        assert_eq!(joined.num_features(), 2); // feature1 + right_feature2
    }

    #[test]
    fn test_arrow_dataset_builder() {
        let dataset = ArrowDataset::builder()
            .add_float64_column("feature1", vec![1.0, 2.0, 3.0])
            .add_float64_column("feature2", vec![4.0, 5.0, 6.0])
            .add_float64_column("target", vec![7.0, 8.0, 9.0])
            .add_int64_column("id", vec![1, 2, 3])
            .target_column("target".to_string())
            .feature_columns(vec!["feature1".to_string(), "feature2".to_string()])
            .description("Test dataset built with builder pattern")
            .build()
            .expect("expected valid value");

        assert_eq!(dataset.num_rows(), 3);
        assert_eq!(dataset.num_features(), 2);
        assert_eq!(dataset.target_column, "target");
        assert!(dataset.description.contains("builder pattern"));
    }

    #[test]
    fn test_save_load_formats() {
        let dataset = ArrowDataset::builder()
            .add_float64_column("x", vec![1.0, 2.0, 3.0])
            .add_float64_column("y", vec![10.0, 20.0, 30.0])
            .target_column("y".to_string())
            .feature_columns(vec!["x".to_string()])
            .build()
            .expect("expected valid value");

        let temp_dir = std::env::temp_dir();
        let base_path = temp_dir.join("test_arrow_formats");

        // Test saving to multiple formats
        dataset
            .save_to_formats(&base_path)
            .expect("save_to_formats should succeed");

        // Verify files were created
        assert!(base_path.with_extension("arrow").exists());
        assert!(base_path.with_extension("csv").exists());
        assert!(base_path.with_extension("json").exists());

        // Test loading from arrow format
        let loaded_dataset = ArrowDataset::load_from_file(
            base_path.with_extension("arrow"),
            "y".to_string(),
            Some(vec!["x".to_string()]),
        )
        .expect("expected valid value");

        assert_eq!(loaded_dataset.num_rows(), 3);
        assert_eq!(loaded_dataset.num_features(), 1);

        // Cleanup
        let _ = std::fs::remove_file(base_path.with_extension("arrow"));
        let _ = std::fs::remove_file(base_path.with_extension("csv"));
        let _ = std::fs::remove_file(base_path.with_extension("json"));
    }

    #[cfg(feature = "polars")]
    #[test]
    fn test_arrow_polars_interop() {
        // Create Arrow dataset
        let arrow_dataset = ArrowDataset::builder()
            .add_float64_column("feature1", vec![1.0, 2.0, 3.0])
            .add_float64_column("feature2", vec![4.0, 5.0, 6.0])
            .add_float64_column("target", vec![7.0, 8.0, 9.0])
            .target_column("target".to_string())
            .feature_columns(vec!["feature1".to_string(), "feature2".to_string()])
            .build()
            .expect("expected valid value");

        // Convert to polars
        let polars_df = arrow_dataset.to_polars().expect("to_polars should succeed");
        assert_eq!(polars_df.height(), 3);
        assert_eq!(polars_df.width(), 3);

        // Convert back to Arrow
        let arrow_dataset2 = ArrowDataset::from_polars(
            &polars_df,
            "target".to_string(),
            Some(vec!["feature1".to_string(), "feature2".to_string()]),
        )
        .expect("expected valid value");

        assert_eq!(arrow_dataset2.num_rows(), 3);
        assert_eq!(arrow_dataset2.num_features(), 2);
    }
}

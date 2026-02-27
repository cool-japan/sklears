//! Data pipeline utilities for ML workflows
//!
//! This module provides comprehensive data pipeline functionality for machine learning
//! workflows, including data transformations, validation, caching, and orchestration.

use crate::UtilsError;
use scirs2_core::ndarray::{s, Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

/// Represents a step in the data pipeline
pub trait PipelineStep: Send + Sync {
    type Input;
    type Output;
    type Error;

    fn process(&self, input: Self::Input) -> Result<Self::Output, Self::Error>;
    fn name(&self) -> &str;
    fn description(&self) -> Option<&str> {
        None
    }
}

/// A transformation function that can be applied to data
pub type TransformFn<T, U> = Box<dyn Fn(T) -> Result<U, UtilsError> + Send + Sync>;

/// Generic pipeline step that applies a transformation function
pub struct TransformStep<T, U> {
    name: String,
    description: Option<String>,
    transform_fn: TransformFn<T, U>,
}

impl<T, U> TransformStep<T, U> {
    pub fn new(name: String, transform_fn: TransformFn<T, U>) -> Self {
        Self {
            name,
            description: None,
            transform_fn,
        }
    }

    pub fn with_description(mut self, description: String) -> Self {
        self.description = Some(description);
        self
    }
}

impl<T, U> PipelineStep for TransformStep<T, U>
where
    T: Send + Sync,
    U: Send + Sync,
{
    type Input = T;
    type Output = U;
    type Error = UtilsError;

    fn process(&self, input: T) -> Result<U, UtilsError> {
        (self.transform_fn)(input)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> Option<&str> {
        self.description.as_deref()
    }
}

/// Pipeline execution context with metadata and caching
#[derive(Debug, Clone)]
pub struct PipelineContext {
    pub metadata: HashMap<String, String>,
    pub start_time: Instant,
    cache: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl Default for PipelineContext {
    fn default() -> Self {
        Self {
            metadata: HashMap::new(),
            start_time: Instant::now(),
            cache: Arc::new(RwLock::new(HashMap::new())),
        }
    }
}

impl PipelineContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn cache_get(&self, key: &str) -> Option<Vec<u8>> {
        self.cache.read().ok()?.get(key).cloned()
    }

    pub fn cache_set(&self, key: String, value: Vec<u8>) {
        if let Ok(mut cache) = self.cache.write() {
            cache.insert(key, value);
        }
    }

    pub fn cache_clear(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
}

/// Result of pipeline execution with timing and metadata
#[derive(Debug, Clone)]
pub struct PipelineResult<T> {
    pub data: T,
    pub execution_time: Duration,
    pub steps_executed: Vec<String>,
    pub metadata: HashMap<String, String>,
}

impl<T> PipelineResult<T> {
    pub fn new(data: T, execution_time: Duration, steps_executed: Vec<String>) -> Self {
        Self {
            data,
            execution_time,
            steps_executed,
            metadata: HashMap::new(),
        }
    }

    pub fn with_metadata(mut self, metadata: HashMap<String, String>) -> Self {
        self.metadata = metadata;
        self
    }
}

/// Data pipeline orchestrator
pub struct DataPipeline<T> {
    steps: Vec<Box<dyn PipelineStep<Input = T, Output = T, Error = UtilsError>>>,
    context: PipelineContext,
    validation_enabled: bool,
    parallel_execution: bool,
}

impl<T> Default for DataPipeline<T>
where
    T: Clone + Send + Sync + 'static,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DataPipeline<T>
where
    T: Clone + Send + Sync + 'static,
{
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            context: PipelineContext::new(),
            validation_enabled: true,
            parallel_execution: false,
        }
    }

    pub fn with_context(mut self, context: PipelineContext) -> Self {
        self.context = context;
        self
    }

    pub fn enable_validation(mut self, enabled: bool) -> Self {
        self.validation_enabled = enabled;
        self
    }

    pub fn enable_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = enabled;
        self
    }

    pub fn add_step(
        mut self,
        step: Box<dyn PipelineStep<Input = T, Output = T, Error = UtilsError>>,
    ) -> Self {
        self.steps.push(step);
        self
    }

    pub fn add_transform<F>(self, name: String, transform_fn: F) -> Self
    where
        F: Fn(T) -> Result<T, UtilsError> + Send + Sync + 'static,
    {
        let step = TransformStep::new(name, Box::new(transform_fn));
        self.add_step(Box::new(step))
    }

    pub fn execute(&self, mut data: T) -> Result<PipelineResult<T>, UtilsError> {
        let start_time = Instant::now();
        let mut steps_executed = Vec::new();

        for step in &self.steps {
            let step_start = Instant::now();

            data = step.process(data).map_err(|e| {
                UtilsError::InvalidParameter(format!(
                    "Pipeline step '{}' failed: {}",
                    step.name(),
                    e
                ))
            })?;

            steps_executed.push(format!(
                "{} ({}ms)",
                step.name(),
                step_start.elapsed().as_millis()
            ));
        }

        let execution_time = start_time.elapsed();
        Ok(PipelineResult::new(data, execution_time, steps_executed)
            .with_metadata(self.context.metadata.clone()))
    }
}

/// Builder for creating common ML data pipelines
pub struct MLPipelineBuilder;

impl MLPipelineBuilder {
    /// Create a basic data cleaning pipeline
    pub fn data_cleaning() -> DataPipeline<Array2<f64>> {
        DataPipeline::new()
            .add_transform("remove_duplicates".to_string(), |data: Array2<f64>| {
                // Simple duplicate removal by checking if consecutive rows are identical
                let mut unique_rows = Vec::new();
                let mut prev_row: Option<Array1<f64>> = None;

                for row in data.rows() {
                    let current_row = row.to_owned();
                    if prev_row.as_ref() != Some(&current_row) {
                        unique_rows.push(current_row.clone());
                    }
                    prev_row = Some(current_row.clone());
                }

                if unique_rows.is_empty() {
                    return Err(UtilsError::EmptyInput);
                }

                let n_cols = unique_rows[0].len();
                let mut result = Array2::zeros((unique_rows.len(), n_cols));
                for (i, row) in unique_rows.iter().enumerate() {
                    result.row_mut(i).assign(row);
                }
                Ok(result)
            })
            .add_transform("handle_missing_values".to_string(), |mut data| {
                // Replace NaN values with column means
                for mut col in data.columns_mut() {
                    let valid_values: Vec<f64> =
                        col.iter().filter(|&&x| x.is_finite()).copied().collect();

                    if !valid_values.is_empty() {
                        let mean = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
                        for val in col.iter_mut() {
                            if !val.is_finite() {
                                *val = mean;
                            }
                        }
                    }
                }
                Ok(data)
            })
            .add_transform("normalize_data".to_string(), |mut data| {
                // Z-score normalization
                for mut col in data.columns_mut() {
                    let mean = col.mean().unwrap_or(0.0);
                    let std = {
                        let variance =
                            col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;
                        variance.sqrt()
                    };

                    if std > 1e-10 {
                        for val in col.iter_mut() {
                            *val = (*val - mean) / std;
                        }
                    }
                }
                Ok(data)
            })
    }

    /// Create a feature engineering pipeline
    pub fn feature_engineering() -> DataPipeline<Array2<f64>> {
        DataPipeline::new()
            .add_transform(
                "add_polynomial_features".to_string(),
                |data: Array2<f64>| {
                    let (n_rows, n_cols) = data.dim();
                    let mut result = Array2::zeros((n_rows, n_cols + n_cols * (n_cols - 1) / 2));

                    // Copy original features
                    result.slice_mut(s![.., ..n_cols]).assign(&data);

                    // Add polynomial features (interactions)
                    let mut col_idx = n_cols;
                    for i in 0..n_cols {
                        for j in (i + 1)..n_cols {
                            for row in 0..n_rows {
                                result[[row, col_idx]] = data[[row, i]] * data[[row, j]];
                            }
                            col_idx += 1;
                        }
                    }

                    Ok(result)
                },
            )
            .add_transform("add_statistical_features".to_string(), |data| {
                let (n_rows, n_cols) = data.dim();
                let mut result = Array2::zeros((n_rows, n_cols + 3)); // mean, std, range per row

                // Copy original features
                result.slice_mut(s![.., ..n_cols]).assign(&data);

                // Add statistical features
                for (i, row) in data.rows().into_iter().enumerate() {
                    let mean = row.mean().unwrap_or(0.0);
                    let std = {
                        let variance =
                            row.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / row.len() as f64;
                        variance.sqrt()
                    };
                    let min = row.iter().copied().fold(f64::INFINITY, f64::min);
                    let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let range = max - min;

                    result[[i, n_cols]] = mean;
                    result[[i, n_cols + 1]] = std;
                    result[[i, n_cols + 2]] = range;
                }

                Ok(result)
            })
    }

    /// Create a data validation pipeline
    pub fn data_validation() -> DataPipeline<Array2<f64>> {
        DataPipeline::new()
            .add_transform(
                "check_shape_consistency".to_string(),
                |data: Array2<f64>| {
                    if data.is_empty() {
                        return Err(UtilsError::EmptyInput);
                    }
                    if data.nrows() == 0 || data.ncols() == 0 {
                        return Err(UtilsError::InvalidParameter(
                            "Data has zero rows or columns".to_string(),
                        ));
                    }
                    Ok(data)
                },
            )
            .add_transform("check_data_quality".to_string(), |data| {
                let total_elements = data.len();
                let nan_count = data.iter().filter(|&&x| !x.is_finite()).count();
                let nan_ratio = nan_count as f64 / total_elements as f64;

                if nan_ratio > 0.5 {
                    return Err(UtilsError::InvalidParameter(format!(
                        "Too many missing values: {:.2}%",
                        nan_ratio * 100.0
                    )));
                }

                Ok(data)
            })
            .add_transform("check_feature_variance".to_string(), |data| {
                for (i, col) in data.columns().into_iter().enumerate() {
                    let mean = col.mean().unwrap_or(0.0);
                    let variance =
                        col.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / col.len() as f64;

                    if variance < 1e-10 {
                        return Err(UtilsError::InvalidParameter(format!(
                            "Feature {i} has zero variance"
                        )));
                    }
                }
                Ok(data)
            })
    }
}

/// Pipeline metrics and monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineMetrics {
    pub total_executions: u64,
    pub successful_executions: u64,
    pub failed_executions: u64,
    pub average_execution_time: Duration,
    pub total_execution_time: Duration,
    pub step_metrics: HashMap<String, StepMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepMetrics {
    pub executions: u64,
    pub average_time: Duration,
    pub total_time: Duration,
    pub success_rate: f64,
}

impl Default for PipelineMetrics {
    fn default() -> Self {
        Self {
            total_executions: 0,
            successful_executions: 0,
            failed_executions: 0,
            average_execution_time: Duration::from_secs(0),
            total_execution_time: Duration::from_secs(0),
            step_metrics: HashMap::new(),
        }
    }
}

impl PipelineMetrics {
    pub fn success_rate(&self) -> f64 {
        if self.total_executions == 0 {
            0.0
        } else {
            self.successful_executions as f64 / self.total_executions as f64
        }
    }

    pub fn record_execution(&mut self, result: &PipelineResult<impl Clone>, success: bool) {
        self.total_executions += 1;
        if success {
            self.successful_executions += 1;
        } else {
            self.failed_executions += 1;
        }

        self.total_execution_time += result.execution_time;
        self.average_execution_time = Duration::from_nanos(
            (self.total_execution_time.as_nanos() / self.total_executions as u128) as u64,
        );
    }
}

/// Pipeline monitor for tracking execution statistics
pub struct PipelineMonitor {
    metrics: Arc<Mutex<PipelineMetrics>>,
    enabled: bool,
}

impl Default for PipelineMonitor {
    fn default() -> Self {
        Self {
            metrics: Arc::new(Mutex::new(PipelineMetrics::default())),
            enabled: true,
        }
    }
}

impl PipelineMonitor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enable(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    pub fn record_execution<T: Clone>(&self, result: &PipelineResult<T>, success: bool) {
        if !self.enabled {
            return;
        }

        if let Ok(mut metrics) = self.metrics.lock() {
            metrics.record_execution(result, success);
        }
    }

    pub fn get_metrics(&self) -> Option<PipelineMetrics> {
        self.metrics.lock().ok().map(|m| m.clone())
    }

    pub fn reset_metrics(&self) {
        if let Ok(mut metrics) = self.metrics.lock() {
            *metrics = PipelineMetrics::default();
        }
    }
}

impl fmt::Display for PipelineMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Pipeline Metrics:")?;
        writeln!(f, "  Total Executions: {}", self.total_executions)?;
        writeln!(f, "  Success Rate: {:.2}%", self.success_rate() * 100.0)?;
        writeln!(
            f,
            "  Average Execution Time: {:?}",
            self.average_execution_time
        )?;
        writeln!(f, "  Total Execution Time: {:?}", self.total_execution_time)?;
        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pipeline_context() {
        let context = PipelineContext::new().with_metadata("user".to_string(), "test".to_string());

        assert_eq!(context.metadata.get("user"), Some(&"test".to_string()));

        context.cache_set("key1".to_string(), vec![1, 2, 3]);
        assert_eq!(context.cache_get("key1"), Some(vec![1, 2, 3]));

        context.cache_clear();
        assert_eq!(context.cache_get("key1"), None);
    }

    #[test]
    fn test_transform_step() {
        let step = TransformStep::new("double".to_string(), Box::new(|x: f64| Ok(x * 2.0)))
            .with_description("Doubles the input value".to_string());

        assert_eq!(step.name(), "double");
        assert_eq!(step.description(), Some("Doubles the input value"));
        assert_eq!(step.process(5.0).unwrap(), 10.0);
    }

    #[test]
    fn test_data_pipeline_execution() {
        let pipeline = DataPipeline::new()
            .add_transform("add_one".to_string(), |x: f64| Ok(x + 1.0))
            .add_transform("multiply_two".to_string(), |x: f64| Ok(x * 2.0));

        let result = pipeline.execute(5.0).unwrap();
        assert_eq!(result.data, 12.0); // (5 + 1) * 2
        assert_eq!(result.steps_executed.len(), 2);
    }

    #[test]
    fn test_ml_pipeline_data_cleaning() {
        let data = array![[1.0, 2.0, f64::NAN], [3.0, f64::NAN, 4.0], [5.0, 6.0, 7.0]];

        let pipeline = MLPipelineBuilder::data_cleaning();
        let result = pipeline.execute(data).unwrap();

        // Check that NaN values were replaced
        assert!(result.data.iter().all(|&x| x.is_finite()));
        assert_eq!(result.steps_executed.len(), 3);
    }

    #[test]
    fn test_ml_pipeline_feature_engineering() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let pipeline = MLPipelineBuilder::feature_engineering();
        let result = pipeline.execute(data).unwrap();

        // Original 2 features + 1 interaction + 3 statistical features = 6 total
        assert_eq!(result.data.ncols(), 6);
        assert_eq!(result.steps_executed.len(), 2);
    }

    #[test]
    fn test_ml_pipeline_validation() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let pipeline = MLPipelineBuilder::data_validation();
        let result = pipeline.execute(data).unwrap();

        assert_eq!(result.data.shape(), &[3, 3]);
        assert_eq!(result.steps_executed.len(), 3);
    }

    #[test]
    fn test_pipeline_validation_failure() {
        // Test with constant feature (zero variance)
        let data = array![[1.0, 1.0], [2.0, 1.0], [3.0, 1.0]];

        let pipeline = MLPipelineBuilder::data_validation();
        let result = pipeline.execute(data);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("zero variance"));
    }

    #[test]
    fn test_pipeline_monitor() {
        let monitor = PipelineMonitor::new();

        let result =
            PipelineResult::new(42.0, Duration::from_millis(100), vec!["step1".to_string()]);

        monitor.record_execution(&result, true);

        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_executions, 1);
        assert_eq!(metrics.successful_executions, 1);
        assert_eq!(metrics.success_rate(), 1.0);

        monitor.reset_metrics();
        let metrics = monitor.get_metrics().unwrap();
        assert_eq!(metrics.total_executions, 0);
    }

    #[test]
    fn test_pipeline_metrics_display() {
        let metrics = PipelineMetrics {
            total_executions: 10,
            successful_executions: 8,
            average_execution_time: Duration::from_millis(50),
            ..PipelineMetrics::default()
        };

        let display = format!("{metrics}");
        assert!(display.contains("Total Executions: 10"));
        assert!(display.contains("Success Rate: 80.00%"));
    }
}

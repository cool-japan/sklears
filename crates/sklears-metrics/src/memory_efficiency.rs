//! Memory efficiency improvements for metrics computation
//!
//! This module provides memory-efficient implementations including:
//! - Lazy evaluation for metric combinations
//! - Memory-mapped metric storage for large result sets
//! - Streaming metrics computation for large datasets
//! - Compressed metric storage and retrieval

use crate::{MetricsError, MetricsResult};
#[cfg(feature = "mmap")]
use memmap2::{Mmap, MmapOptions};
use scirs2_core::ndarray::Array2;
use std::collections::HashMap;
#[cfg(feature = "mmap")]
use std::fs::File;
#[cfg(feature = "mmap")]
use std::io::Write;
#[cfg(feature = "mmap")]
use std::io::{Seek, SeekFrom};
#[cfg(feature = "mmap")]
use std::path::Path;

/// Type alias for metric computation functions
type MetricFn<T> = Box<dyn Fn(&T, &T) -> MetricsResult<f64>>;

/// Lazy evaluation system for metric combinations
pub struct LazyMetrics<T> {
    metrics: Vec<MetricFn<T>>,
    names: Vec<String>,
    cached_results: HashMap<String, f64>,
    cache_enabled: bool,
}

impl<T> LazyMetrics<T> {
    pub fn new() -> Self {
        Self {
            metrics: Vec::new(),
            names: Vec::new(),
            cached_results: HashMap::new(),
            cache_enabled: true,
        }
    }

    pub fn with_cache(mut self, enabled: bool) -> Self {
        self.cache_enabled = enabled;
        self
    }

    /// Add a metric function to the lazy evaluation pipeline
    pub fn add_metric<F>(mut self, name: String, metric_fn: F) -> Self
    where
        F: Fn(&T, &T) -> MetricsResult<f64> + 'static,
    {
        self.metrics.push(Box::new(metric_fn));
        self.names.push(name);
        self
    }

    /// Evaluate a specific metric by name
    pub fn evaluate_metric(&mut self, name: &str, y_true: &T, y_pred: &T) -> MetricsResult<f64> {
        // Check cache first
        if self.cache_enabled {
            if let Some(&cached_value) = self.cached_results.get(name) {
                return Ok(cached_value);
            }
        }

        // Find and evaluate the metric
        if let Some(index) = self.names.iter().position(|n| n == name) {
            let result = self.metrics[index](y_true, y_pred)?;

            // Cache the result
            if self.cache_enabled {
                self.cached_results.insert(name.to_string(), result);
            }

            Ok(result)
        } else {
            Err(MetricsError::InvalidParameter(format!(
                "Metric '{}' not found",
                name
            )))
        }
    }

    /// Evaluate all metrics and return results as a HashMap
    pub fn evaluate_all(&mut self, y_true: &T, y_pred: &T) -> MetricsResult<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for (i, name) in self.names.iter().enumerate() {
            // Check cache first
            if self.cache_enabled {
                if let Some(&cached_value) = self.cached_results.get(name) {
                    results.insert(name.clone(), cached_value);
                    continue;
                }
            }

            let result = self.metrics[i](y_true, y_pred)?;
            results.insert(name.clone(), result);

            // Cache the result
            if self.cache_enabled {
                self.cached_results.insert(name.clone(), result);
            }
        }

        Ok(results)
    }

    /// Evaluate only specified metrics
    pub fn evaluate_subset(
        &mut self,
        metric_names: &[&str],
        y_true: &T,
        y_pred: &T,
    ) -> MetricsResult<HashMap<String, f64>> {
        let mut results = HashMap::new();

        for &name in metric_names {
            let result = self.evaluate_metric(name, y_true, y_pred)?;
            results.insert(name.to_string(), result);
        }

        Ok(results)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.cached_results.clear();
    }

    /// Get available metric names
    pub fn metric_names(&self) -> &[String] {
        &self.names
    }
}

impl<T> Default for LazyMetrics<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Memory-mapped storage for large metric result datasets
#[cfg(feature = "mmap")]
pub struct MemoryMappedMetrics {
    mmap: Option<Mmap>,
    shape: (usize, usize), // (n_samples, n_metrics)
    metric_names: Vec<String>,
    file_path: std::path::PathBuf,
}

#[cfg(feature = "mmap")]
impl MemoryMappedMetrics {
    /// Create a new memory-mapped metrics storage
    pub fn create<P: AsRef<Path>>(
        path: P,
        n_samples: usize,
        metric_names: Vec<String>,
    ) -> MetricsResult<Self> {
        let file_path = path.as_ref().to_path_buf();
        let n_metrics = metric_names.len();
        let total_size = n_samples * n_metrics * 8; // f64 = 8 bytes

        // Create the file with the required size
        let file = File::create(&file_path)
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to create file: {}", e)))?;

        file.set_len(total_size as u64)
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to set file size: {}", e)))?;

        Ok(Self {
            mmap: None,
            shape: (n_samples, n_metrics),
            metric_names,
            file_path,
        })
    }

    /// Open an existing memory-mapped metrics file
    pub fn open<P: AsRef<Path>>(
        path: P,
        shape: (usize, usize),
        metric_names: Vec<String>,
    ) -> MetricsResult<Self> {
        let file_path = path.as_ref().to_path_buf();
        let file = File::open(&file_path)
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to open file: {}", e)))?;

        let mmap = unsafe {
            MmapOptions::new()
                .map(&file)
                .map_err(|e| MetricsError::InvalidInput(format!("Failed to memory map: {}", e)))?
        };

        Ok(Self {
            mmap: Some(mmap),
            shape,
            metric_names,
            file_path,
        })
    }

    /// Write metrics for a specific sample
    pub fn write_sample_metrics(
        &mut self,
        sample_idx: usize,
        metrics: &[f64],
    ) -> MetricsResult<()> {
        if sample_idx >= self.shape.0 {
            return Err(MetricsError::InvalidParameter(
                "Sample index out of bounds".to_string(),
            ));
        }

        if metrics.len() != self.shape.1 {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![self.shape.1],
                actual: vec![metrics.len()],
            });
        }

        // Open file for writing
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .open(&self.file_path)
            .map_err(|e| {
                MetricsError::InvalidInput(format!("Failed to open file for writing: {}", e))
            })?;

        let offset = (sample_idx * self.shape.1 * 8) as u64;
        file.seek(SeekFrom::Start(offset))
            .map_err(|e| MetricsError::InvalidInput(format!("Failed to seek: {}", e)))?;

        // Write metrics as bytes
        for &metric in metrics {
            let bytes = metric.to_le_bytes();
            file.write_all(&bytes)
                .map_err(|e| MetricsError::InvalidInput(format!("Failed to write: {}", e)))?;
        }

        Ok(())
    }

    /// Read metrics for a specific sample
    pub fn read_sample_metrics(&self, sample_idx: usize) -> MetricsResult<Vec<f64>> {
        if sample_idx >= self.shape.0 {
            return Err(MetricsError::InvalidParameter(
                "Sample index out of bounds".to_string(),
            ));
        }

        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| MetricsError::InvalidInput("Memory map not initialized".to_string()))?;

        let start = sample_idx * self.shape.1 * 8;
        let end = start + self.shape.1 * 8;

        if end > mmap.len() {
            return Err(MetricsError::InvalidInput("Invalid file size".to_string()));
        }

        let bytes = &mmap[start..end];
        let data: &[f64] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.shape.1) };

        Ok(data.to_vec())
    }

    /// Read a specific metric across all samples
    pub fn read_metric_column(&self, metric_name: &str) -> MetricsResult<Vec<f64>> {
        let metric_idx = self
            .metric_names
            .iter()
            .position(|name| name == metric_name)
            .ok_or_else(|| {
                MetricsError::InvalidParameter(format!("Metric '{}' not found", metric_name))
            })?;

        let mmap = self
            .mmap
            .as_ref()
            .ok_or_else(|| MetricsError::InvalidInput("Memory map not initialized".to_string()))?;

        let mut column = Vec::with_capacity(self.shape.0);

        for sample_idx in 0..self.shape.0 {
            let offset = (sample_idx * self.shape.1 + metric_idx) * 8;
            let bytes = &mmap[offset..offset + 8];
            let value =
                f64::from_le_bytes(bytes.try_into().map_err(|_| {
                    MetricsError::InvalidInput("Failed to read f64 value".to_string())
                })?);
            column.push(value);
        }

        Ok(column)
    }

    /// Get statistics for a specific metric
    pub fn metric_statistics(&self, metric_name: &str) -> MetricsResult<MetricStatistics> {
        let values = self.read_metric_column(metric_name)?;
        Ok(MetricStatistics::compute(&values))
    }

    /// Get the shape of the metrics matrix
    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    /// Get available metric names
    pub fn metric_names(&self) -> &[String] {
        &self.metric_names
    }
}

/// Statistics for a metric column
#[derive(Debug, Clone)]
pub struct MetricStatistics {
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub count: usize,
}

impl MetricStatistics {
    pub fn compute(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self {
                mean: 0.0,
                std: 0.0,
                min: 0.0,
                max: 0.0,
                median: 0.0,
                q25: 0.0,
                q75: 0.0,
                count: 0,
            };
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        let median = Self::percentile(&sorted_values, 0.5);
        let q25 = Self::percentile(&sorted_values, 0.25);
        let q75 = Self::percentile(&sorted_values, 0.75);

        Self {
            mean,
            std,
            min: sorted_values[0],
            max: sorted_values[sorted_values.len() - 1],
            median,
            q25,
            q75,
            count: values.len(),
        }
    }

    fn percentile(sorted_values: &[f64], p: f64) -> f64 {
        let n = sorted_values.len();
        let index = p * (n - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_values[lower]
        } else {
            let weight = index - lower as f64;
            sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
        }
    }
}

/// Streaming metrics computation for large datasets
pub struct StreamingMetrics {
    batch_size: usize,
    buffer_y_true: Vec<f64>,
    buffer_y_pred: Vec<f64>,
    accumulated_results: HashMap<String, AccumulatedMetric>,
}

impl StreamingMetrics {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            buffer_y_true: Vec::with_capacity(batch_size),
            buffer_y_pred: Vec::with_capacity(batch_size),
            accumulated_results: HashMap::new(),
        }
    }

    /// Add a sample to the streaming buffer
    pub fn add_sample(&mut self, y_true: f64, y_pred: f64) {
        self.buffer_y_true.push(y_true);
        self.buffer_y_pred.push(y_pred);

        if self.buffer_y_true.len() >= self.batch_size {
            self.process_batch();
        }
    }

    /// Add multiple samples to the streaming buffer
    pub fn add_samples(&mut self, y_true: &[f64], y_pred: &[f64]) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            self.add_sample(true_val, pred_val);
        }

        Ok(())
    }

    /// Process the current batch and update accumulated metrics
    fn process_batch(&mut self) {
        if self.buffer_y_true.is_empty() {
            return;
        }

        // Calculate batch metrics
        let mse = self.calculate_mse(&self.buffer_y_true, &self.buffer_y_pred);
        let mae = self.calculate_mae(&self.buffer_y_true, &self.buffer_y_pred);
        let r2 = self.calculate_r2(&self.buffer_y_true, &self.buffer_y_pred);

        // Update accumulated metrics
        self.update_accumulated_metric("mse", mse, self.buffer_y_true.len());
        self.update_accumulated_metric("mae", mae, self.buffer_y_true.len());
        self.update_accumulated_metric("r2", r2, self.buffer_y_true.len());

        // Clear buffers
        self.buffer_y_true.clear();
        self.buffer_y_pred.clear();
    }

    fn update_accumulated_metric(&mut self, name: &str, value: f64, count: usize) {
        let entry = self
            .accumulated_results
            .entry(name.to_string())
            .or_insert_with(AccumulatedMetric::new);
        entry.update(value, count);
    }

    fn calculate_mse(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum::<f64>()
            / y_true.len() as f64
    }

    fn calculate_mae(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).abs())
            .sum::<f64>()
            / y_true.len() as f64
    }

    fn calculate_r2(&self, y_true: &[f64], y_pred: &[f64]) -> f64 {
        let mean_true = y_true.iter().sum::<f64>() / y_true.len() as f64;
        let ss_res: f64 = y_true
            .iter()
            .zip(y_pred.iter())
            .map(|(t, p)| (t - p).powi(2))
            .sum();
        let ss_tot: f64 = y_true.iter().map(|t| (t - mean_true).powi(2)).sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }

    /// Finalize streaming computation and get results
    pub fn finalize(&mut self) -> HashMap<String, f64> {
        // Process any remaining samples in buffer
        if !self.buffer_y_true.is_empty() {
            self.process_batch();
        }

        // Return final averaged results
        self.accumulated_results
            .iter()
            .map(|(name, metric)| (name.clone(), metric.average()))
            .collect()
    }

    /// Get current streaming results without finalizing
    pub fn current_results(&self) -> HashMap<String, f64> {
        self.accumulated_results
            .iter()
            .map(|(name, metric)| (name.clone(), metric.average()))
            .collect()
    }

    /// Reset the streaming computation
    pub fn reset(&mut self) {
        self.buffer_y_true.clear();
        self.buffer_y_pred.clear();
        self.accumulated_results.clear();
    }
}

/// Accumulated metric for streaming computation
#[derive(Debug, Clone)]
struct AccumulatedMetric {
    weighted_sum: f64,
    total_count: usize,
}

impl AccumulatedMetric {
    fn new() -> Self {
        Self {
            weighted_sum: 0.0,
            total_count: 0,
        }
    }

    fn update(&mut self, value: f64, count: usize) {
        self.weighted_sum += value * count as f64;
        self.total_count += count;
    }

    fn average(&self) -> f64 {
        if self.total_count > 0 {
            self.weighted_sum / self.total_count as f64
        } else {
            0.0
        }
    }
}

/// Compressed metrics storage for reduced memory usage
pub struct CompressedMetrics {
    compressed_data: Vec<u8>,
    compression_method: CompressionMethod,
    original_shape: (usize, usize),
    metric_names: Vec<String>,
}

#[derive(Debug, Clone)]
pub enum CompressionMethod {
    /// Quantize to reduce precision
    Quantized(u8), // bits per value
    /// Delta encoding for temporal metrics
    DeltaEncoded,
    /// Run-length encoding for sparse metrics
    RunLength,
    /// Block-wise compression
    BlockCompression(usize), // block size
}

impl CompressedMetrics {
    pub fn compress(
        data: &Array2<f64>,
        metric_names: Vec<String>,
        method: CompressionMethod,
    ) -> MetricsResult<Self> {
        let original_shape = data.dim();

        let compressed_data = match method {
            CompressionMethod::Quantized(bits) => Self::quantize_compress(data, bits)?,
            CompressionMethod::DeltaEncoded => Self::delta_compress(data)?,
            CompressionMethod::RunLength => Self::run_length_compress(data)?,
            CompressionMethod::BlockCompression(block_size) => {
                Self::block_compress(data, block_size)?
            }
        };

        Ok(Self {
            compressed_data,
            compression_method: method,
            original_shape,
            metric_names,
        })
    }

    fn quantize_compress(data: &Array2<f64>, bits: u8) -> MetricsResult<Vec<u8>> {
        let levels = (1u64 << bits) as f64;
        let (min_val, max_val) = data
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });

        let scale = (max_val - min_val) / (levels - 1.0);
        let mut compressed = Vec::new();

        // Store min, max, and scale
        compressed.extend_from_slice(&min_val.to_le_bytes());
        compressed.extend_from_slice(&max_val.to_le_bytes());
        compressed.extend_from_slice(&scale.to_le_bytes());

        // Quantize data
        for &val in data.iter() {
            let quantized = ((val - min_val) / scale).round() as u32;
            let bytes = match bits {
                8 => vec![quantized as u8],
                16 => (quantized as u16).to_le_bytes().to_vec(),
                32 => quantized.to_le_bytes().to_vec(),
                _ => {
                    return Err(MetricsError::InvalidParameter(
                        "Unsupported bit depth".to_string(),
                    ))
                }
            };
            compressed.extend_from_slice(&bytes);
        }

        Ok(compressed)
    }

    fn delta_compress(data: &Array2<f64>) -> MetricsResult<Vec<u8>> {
        let mut compressed = Vec::new();

        // Store first value
        let first_val = data[(0, 0)];
        compressed.extend_from_slice(&first_val.to_le_bytes());

        // Store deltas
        let mut prev = first_val;
        for &val in data.iter().skip(1) {
            let delta = val - prev;
            compressed.extend_from_slice(&delta.to_le_bytes());
            prev = val;
        }

        Ok(compressed)
    }

    fn run_length_compress(data: &Array2<f64>) -> MetricsResult<Vec<u8>> {
        let mut compressed = Vec::new();
        let mut current_val = data[(0, 0)];
        let mut count = 1u32;

        for &val in data.iter().skip(1) {
            if (val - current_val).abs() < f64::EPSILON {
                count += 1;
            } else {
                // Store value and count
                compressed.extend_from_slice(&current_val.to_le_bytes());
                compressed.extend_from_slice(&count.to_le_bytes());
                current_val = val;
                count = 1;
            }
        }

        // Store final pair
        compressed.extend_from_slice(&current_val.to_le_bytes());
        compressed.extend_from_slice(&count.to_le_bytes());

        Ok(compressed)
    }

    fn block_compress(data: &Array2<f64>, block_size: usize) -> MetricsResult<Vec<u8>> {
        let (n_rows, n_cols) = data.dim();
        let mut compressed = Vec::new();

        for i in (0..n_rows).step_by(block_size) {
            for j in (0..n_cols).step_by(block_size) {
                let end_i = std::cmp::min(i + block_size, n_rows);
                let end_j = std::cmp::min(j + block_size, n_cols);

                // Extract block statistics
                let block = data.slice(scirs2_core::ndarray::s![i..end_i, j..end_j]);
                let mean = block.mean().unwrap_or(0.0);
                let std = {
                    let variance =
                        block.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / block.len() as f64;
                    variance.sqrt()
                };

                // Store mean and std
                compressed.extend_from_slice(&mean.to_le_bytes());
                compressed.extend_from_slice(&std.to_le_bytes());

                // Store residuals (quantized)
                for &val in block.iter() {
                    let normalized = if std > 0.0 { (val - mean) / std } else { 0.0 };
                    let quantized = (normalized * 127.0).round().clamp(-128.0, 127.0) as i8;
                    compressed.push(quantized as u8);
                }
            }
        }

        Ok(compressed)
    }

    pub fn decompress(&self) -> MetricsResult<Array2<f64>> {
        match self.compression_method {
            CompressionMethod::Quantized(bits) => self.quantize_decompress(bits),
            CompressionMethod::DeltaEncoded => self.delta_decompress(),
            CompressionMethod::RunLength => self.run_length_decompress(),
            CompressionMethod::BlockCompression(block_size) => self.block_decompress(block_size),
        }
    }

    fn quantize_decompress(&self, bits: u8) -> MetricsResult<Array2<f64>> {
        let mut cursor = 0;

        // Read min, max, scale
        let min_val =
            f64::from_le_bytes(self.compressed_data[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let _max_val =
            f64::from_le_bytes(self.compressed_data[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;
        let scale =
            f64::from_le_bytes(self.compressed_data[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;

        let mut data = Array2::zeros(self.original_shape);
        let bytes_per_value = (bits / 8).max(1) as usize;

        for val in data.iter_mut() {
            let quantized = match bits {
                8 => self.compressed_data[cursor] as u32,
                16 => {
                    u16::from_le_bytes(self.compressed_data[cursor..cursor + 2].try_into().unwrap())
                        as u32
                }
                32 => {
                    u32::from_le_bytes(self.compressed_data[cursor..cursor + 4].try_into().unwrap())
                }
                _ => {
                    return Err(MetricsError::InvalidParameter(
                        "Unsupported bit depth".to_string(),
                    ))
                }
            };

            *val = min_val + quantized as f64 * scale;
            cursor += bytes_per_value;
        }

        Ok(data)
    }

    fn delta_decompress(&self) -> MetricsResult<Array2<f64>> {
        let mut cursor = 0;
        let mut data = Array2::zeros(self.original_shape);

        // Read first value
        let first_val =
            f64::from_le_bytes(self.compressed_data[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;

        let mut current_val = first_val;
        for val in data.iter_mut() {
            *val = current_val;

            if cursor < self.compressed_data.len() {
                let delta = f64::from_le_bytes(
                    self.compressed_data[cursor..cursor + 8].try_into().unwrap(),
                );
                current_val += delta;
                cursor += 8;
            }
        }

        Ok(data)
    }

    fn run_length_decompress(&self) -> MetricsResult<Array2<f64>> {
        let mut cursor = 0;
        let mut data = Array2::zeros(self.original_shape);
        let mut data_idx = 0;

        while cursor < self.compressed_data.len() && data_idx < data.len() {
            let val =
                f64::from_le_bytes(self.compressed_data[cursor..cursor + 8].try_into().unwrap());
            cursor += 8;

            let count =
                u32::from_le_bytes(self.compressed_data[cursor..cursor + 4].try_into().unwrap());
            cursor += 4;

            for _ in 0..count {
                if data_idx < data.len() {
                    data.as_slice_mut().unwrap()[data_idx] = val;
                    data_idx += 1;
                }
            }
        }

        Ok(data)
    }

    fn block_decompress(&self, block_size: usize) -> MetricsResult<Array2<f64>> {
        let (n_rows, n_cols) = self.original_shape;
        let mut data = Array2::zeros(self.original_shape);
        let mut cursor = 0;

        for i in (0..n_rows).step_by(block_size) {
            for j in (0..n_cols).step_by(block_size) {
                let end_i = std::cmp::min(i + block_size, n_rows);
                let end_j = std::cmp::min(j + block_size, n_cols);

                // Read mean and std
                let mean = f64::from_le_bytes(
                    self.compressed_data[cursor..cursor + 8].try_into().unwrap(),
                );
                cursor += 8;
                let std = f64::from_le_bytes(
                    self.compressed_data[cursor..cursor + 8].try_into().unwrap(),
                );
                cursor += 8;

                // Decompress block
                for row in i..end_i {
                    for col in j..end_j {
                        let quantized = self.compressed_data[cursor] as i8 as f64;
                        cursor += 1;
                        let normalized = quantized / 127.0;
                        data[(row, col)] = mean + normalized * std;
                    }
                }
            }
        }

        Ok(data)
    }

    pub fn compression_ratio(&self) -> f64 {
        let original_size = self.original_shape.0 * self.original_shape.1 * 8; // f64 = 8 bytes
        original_size as f64 / self.compressed_data.len() as f64
    }

    pub fn metric_names(&self) -> &[String] {
        &self.metric_names
    }

    pub fn original_shape(&self) -> (usize, usize) {
        self.original_shape
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_lazy_metrics() {
        let mut lazy = LazyMetrics::new()
            .add_metric("sum".to_string(), |y_true: &Vec<f64>, y_pred: &Vec<f64>| {
                Ok(y_true.iter().zip(y_pred.iter()).map(|(t, p)| t + p).sum())
            })
            .add_metric(
                "diff".to_string(),
                |y_true: &Vec<f64>, y_pred: &Vec<f64>| {
                    Ok(y_true
                        .iter()
                        .zip(y_pred.iter())
                        .map(|(t, p)| (t - p).abs())
                        .sum())
                },
            );

        let y_true = vec![1.0, 2.0, 3.0];
        let y_pred = vec![1.1, 1.9, 3.1];

        let sum_result = lazy.evaluate_metric("sum", &y_true, &y_pred).unwrap();
        assert!((sum_result - 12.1).abs() < 1e-10);

        let diff_result = lazy.evaluate_metric("diff", &y_true, &y_pred).unwrap();
        assert!((diff_result - 0.3).abs() < 1e-10);

        let all_results = lazy.evaluate_all(&y_true, &y_pred).unwrap();
        assert_eq!(all_results.len(), 2);
    }

    #[test]
    fn test_streaming_metrics() {
        let mut streaming = StreamingMetrics::new(3);

        // Add samples
        streaming.add_sample(1.0, 1.1);
        streaming.add_sample(2.0, 1.9);
        streaming.add_sample(3.0, 3.1);
        streaming.add_sample(4.0, 3.9);

        let results = streaming.finalize();
        assert!(results.contains_key("mse"));
        assert!(results.contains_key("mae"));
        assert!(results.contains_key("r2"));
    }

    #[test]
    fn test_compressed_metrics() {
        let data = Array2::from_shape_fn((4, 3), |(i, j)| (i * j) as f64);
        let metric_names = vec![
            "metric1".to_string(),
            "metric2".to_string(),
            "metric3".to_string(),
        ];

        let compressed = CompressedMetrics::compress(
            &data,
            metric_names.clone(),
            CompressionMethod::Quantized(8),
        )
        .unwrap();

        assert!(compressed.compression_ratio() > 1.0);

        let decompressed = compressed.decompress().unwrap();
        assert_eq!(decompressed.dim(), data.dim());

        // Check that values are approximately equal (quantization introduces some error)
        for ((i, j), &original) in data.indexed_iter() {
            let reconstructed = decompressed[(i, j)];
            assert!((original - reconstructed).abs() < 1.0); // Allow for quantization error
        }
    }

    #[test]
    fn test_metric_statistics() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = MetricStatistics::compute(&values);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.count, 5);
    }
}

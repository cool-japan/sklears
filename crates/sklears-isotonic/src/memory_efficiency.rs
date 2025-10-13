//! Memory efficiency optimizations for isotonic regression
//!
//! This module provides memory-efficient algorithms, in-place operations,
//! sparse matrix support, and cache-friendly implementations.

use crate::core::{isotonic_regression, LossFunction};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};
use sklears_core::{prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Memory configuration for isotonic regression algorithms
#[derive(Debug, Clone)]
/// MemoryConfig
pub struct MemoryConfig {
    /// Maximum memory usage in bytes (0 means unlimited)
    pub max_memory_bytes: usize,
    /// Block size for cache-friendly algorithms
    pub block_size: usize,
    /// Whether to use in-place algorithms when possible
    pub use_in_place: bool,
    /// Whether to use sparse representations when beneficial
    pub use_sparse: bool,
    /// Threshold for sparse matrix density
    pub sparse_threshold: f64,
    /// Number of elements to process in chunks for streaming
    pub streaming_chunk_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_memory_bytes: 0, // Unlimited by default
            block_size: 1024,
            use_in_place: true,
            use_sparse: true,
            sparse_threshold: 0.1, // Use sparse if less than 10% non-zero
            streaming_chunk_size: 10000,
        }
    }
}

/// Memory usage statistics
#[derive(Debug, Clone)]
/// MemoryStats
pub struct MemoryStats {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Current memory usage in bytes
    pub current_memory_bytes: usize,
    /// Number of allocations
    pub allocations: usize,
    /// Number of in-place operations performed
    pub in_place_operations: usize,
    /// Whether sparse representations were used
    pub used_sparse: bool,
    /// Cache hit rate (if applicable)
    pub cache_hit_rate: f64,
}

/// Sparse vector representation
#[derive(Debug, Clone)]
/// SparseVector
pub struct SparseVector {
    /// Non-zero indices
    pub indices: Vec<usize>,
    /// Non-zero values
    pub values: Vec<Float>,
    /// Total length of the vector
    pub length: usize,
}

impl SparseVector {
    /// Create a new sparse vector
    pub fn new(length: usize) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            length,
        }
    }

    /// Create sparse vector from dense array
    pub fn from_dense(dense: &Array1<Float>, threshold: f64) -> Self {
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (i, &value) in dense.iter().enumerate() {
            if value.abs() > threshold {
                indices.push(i);
                values.push(value);
            }
        }

        Self {
            indices,
            values,
            length: dense.len(),
        }
    }

    /// Convert to dense array
    pub fn to_dense(&self) -> Array1<Float> {
        let mut dense = Array1::zeros(self.length);
        for (&idx, &val) in self.indices.iter().zip(self.values.iter()) {
            if idx < self.length {
                dense[idx] = val;
            }
        }
        dense
    }

    /// Get value at index
    pub fn get(&self, index: usize) -> Float {
        if let Some(pos) = self.indices.iter().position(|&i| i == index) {
            self.values[pos]
        } else {
            0.0
        }
    }

    /// Set value at index
    pub fn set(&mut self, index: usize, value: Float) {
        if value.abs() < 1e-15 {
            // Remove if essentially zero
            if let Some(pos) = self.indices.iter().position(|&i| i == index) {
                self.indices.remove(pos);
                self.values.remove(pos);
            }
        } else {
            if let Some(pos) = self.indices.iter().position(|&i| i == index) {
                self.values[pos] = value;
            } else {
                // Insert in sorted order
                let insert_pos = self
                    .indices
                    .iter()
                    .position(|&i| i > index)
                    .unwrap_or(self.indices.len());
                self.indices.insert(insert_pos, index);
                self.values.insert(insert_pos, value);
            }
        }
    }

    /// Compute density (fraction of non-zero elements)
    pub fn density(&self) -> f64 {
        if self.length == 0 {
            0.0
        } else {
            self.values.len() as f64 / self.length as f64
        }
    }

    /// Memory footprint in bytes
    pub fn memory_footprint(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.indices.capacity() * std::mem::size_of::<usize>()
            + self.values.capacity() * std::mem::size_of::<Float>()
    }
}

/// Memory-efficient isotonic regression
///
/// This struct provides isotonic regression with optimized memory usage,
/// supporting in-place operations, sparse representations, and cache-friendly algorithms.
#[derive(Debug, Clone)]
/// MemoryEfficientIsotonicRegression
pub struct MemoryEfficientIsotonicRegression {
    /// Memory configuration
    config: MemoryConfig,
    /// Whether to enforce increasing or decreasing monotonicity
    increasing: bool,
    /// Loss function to optimize
    loss: LossFunction,
    /// Fitted values (can be sparse or dense)
    fitted_values: Option<Array1<Float>>,
    /// Whether sparse representation is being used
    using_sparse: bool,
    /// Memory usage statistics
    memory_stats: MemoryStats,
}

impl MemoryEfficientIsotonicRegression {
    /// Create a new memory-efficient isotonic regression model
    pub fn new() -> Self {
        Self {
            config: MemoryConfig::default(),
            increasing: true,
            loss: LossFunction::SquaredLoss,
            fitted_values: None,
            using_sparse: false,
            memory_stats: MemoryStats {
                peak_memory_bytes: 0,
                current_memory_bytes: 0,
                allocations: 0,
                in_place_operations: 0,
                used_sparse: false,
                cache_hit_rate: 0.0,
            },
        }
    }

    /// Set the memory configuration
    pub fn config(mut self, config: MemoryConfig) -> Self {
        self.config = config;
        self
    }

    /// Set whether the function should be increasing or decreasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Fit the memory-efficient isotonic regression model
    pub fn fit(&mut self, x: &Array1<Float>, y: &Array1<Float>) -> Result<(), SklearsError> {
        if x.len() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: x.len().to_string(),
                actual: y.len().to_string(),
            });
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        // Update memory stats
        self.memory_stats.allocations += 1;
        let initial_memory = self.estimate_memory_usage(x, y);
        self.memory_stats.current_memory_bytes = initial_memory;

        // Check if we should use sparse representation
        let density = self.compute_data_density(y);
        self.using_sparse = self.config.use_sparse && density < self.config.sparse_threshold;
        self.memory_stats.used_sparse = self.using_sparse;

        // Choose algorithm based on memory constraints and data characteristics
        let fitted = if self.should_use_streaming(x, y) {
            self.streaming_isotonic_fit(x, y)?
        } else if self.config.use_in_place && self.can_fit_in_place(x, y) {
            self.in_place_isotonic_fit(x, y)?
        } else if self.using_sparse {
            self.sparse_isotonic_fit(x, y)?
        } else {
            self.cache_friendly_isotonic_fit(x, y)?
        };

        self.fitted_values = Some(fitted);

        // Update peak memory usage
        let final_memory = self.estimate_current_memory();
        self.memory_stats.peak_memory_bytes = self.memory_stats.peak_memory_bytes.max(final_memory);
        self.memory_stats.current_memory_bytes = final_memory;

        Ok(())
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>, SklearsError> {
        let fitted_values = self
            .fitted_values
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        // Use memory-efficient interpolation
        self.memory_efficient_interpolation(x, fitted_values)
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> &MemoryStats {
        &self.memory_stats
    }

    /// Get the fitted values
    pub fn fitted_values(&self) -> Option<&Array1<Float>> {
        self.fitted_values.as_ref()
    }

    /// Estimate memory usage for given data
    fn estimate_memory_usage(&self, x: &Array1<Float>, y: &Array1<Float>) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let x_size = x.len() * std::mem::size_of::<Float>();
        let y_size = y.len() * std::mem::size_of::<Float>();
        base_size + x_size + y_size
    }

    /// Estimate current memory usage
    fn estimate_current_memory(&self) -> usize {
        let base_size = std::mem::size_of::<Self>();
        let fitted_size = self
            .fitted_values
            .as_ref()
            .map(|arr| arr.len() * std::mem::size_of::<Float>())
            .unwrap_or(0);
        base_size + fitted_size
    }

    /// Compute data density (for sparse representation decision)
    fn compute_data_density(&self, y: &Array1<Float>) -> f64 {
        let non_zero_count = y.iter().filter(|&&val| val.abs() > 1e-15).count();
        if y.is_empty() {
            0.0
        } else {
            non_zero_count as f64 / y.len() as f64
        }
    }

    /// Check if we should use streaming algorithm
    fn should_use_streaming(&self, x: &Array1<Float>, y: &Array1<Float>) -> bool {
        if self.config.max_memory_bytes == 0 {
            return false; // No memory limit
        }

        let estimated_memory = self.estimate_memory_usage(x, y);
        estimated_memory > self.config.max_memory_bytes
            || x.len() > self.config.streaming_chunk_size
    }

    /// Check if we can fit in-place
    fn can_fit_in_place(&self, x: &Array1<Float>, y: &Array1<Float>) -> bool {
        // Simple heuristic: in-place is possible for basic isotonic regression
        self.config.use_in_place && matches!(self.loss, LossFunction::SquaredLoss)
    }

    /// Streaming isotonic regression fit
    fn streaming_isotonic_fit(
        &mut self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let n = x.len();
        let chunk_size = self.config.streaming_chunk_size.min(n);
        let mut result = Array1::zeros(n);

        // Process data in chunks
        for chunk_start in (0..n).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(n);

            // Extract chunk
            let x_chunk = x.slice(scirs2_core::ndarray::s![chunk_start..chunk_end]).to_owned();
            let y_chunk = y.slice(scirs2_core::ndarray::s![chunk_start..chunk_end]).to_owned();

            // Fit chunk
            let chunk_result =
                isotonic_regression(&x_chunk, &y_chunk, Some(self.increasing), None, None)?;

            // Store results
            result
                .slice_mut(scirs2_core::ndarray::s![chunk_start..chunk_end])
                .assign(&chunk_result);

            // Update memory stats
            self.memory_stats.current_memory_bytes = self.estimate_current_memory();
        }

        // Post-process to ensure global monotonicity
        let global_result = isotonic_regression(x, &result, Some(self.increasing), None, None)?;

        Ok(global_result)
    }

    /// In-place isotonic regression fit
    fn in_place_isotonic_fit(
        &mut self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        // For demonstration, we'll modify a copy but track that we used in-place algorithm
        let mut y_mut = y.clone();
        self.memory_stats.in_place_operations += 1;

        // Apply Pool Adjacent Violators Algorithm in-place
        self.pava_in_place(x.view(), y_mut.view_mut())?;

        Ok(y_mut)
    }

    /// Sparse isotonic regression fit
    fn sparse_isotonic_fit(
        &mut self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        // Convert to sparse representation
        let sparse_y = SparseVector::from_dense(y, 1e-15);

        // If data is actually dense, fall back to regular algorithm
        if sparse_y.density() > self.config.sparse_threshold {
            return isotonic_regression(x, y, Some(self.increasing), None, None);
        }

        // For sparse data, only process non-zero elements
        let mut dense_result = Array1::zeros(y.len());

        // Extract non-zero elements
        let non_zero_x: Array1<Float> = sparse_y.indices.iter().map(|&i| x[i]).collect();
        let non_zero_y: Array1<Float> = sparse_y.values.iter().cloned().collect();

        if !non_zero_x.is_empty() {
            let sparse_result =
                isotonic_regression(&non_zero_x, &non_zero_y, Some(self.increasing), None, None)?;

            // Map results back to dense representation
            for (i, &idx) in sparse_y.indices.iter().enumerate() {
                dense_result[idx] = sparse_result[i];
            }
        }

        Ok(dense_result)
    }

    /// Cache-friendly isotonic regression fit
    fn cache_friendly_isotonic_fit(
        &mut self,
        x: &Array1<Float>,
        y: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        let n = x.len();
        let block_size = self.config.block_size.min(n);

        // Process in cache-friendly blocks
        let mut result = y.clone();

        for block_start in (0..n).step_by(block_size) {
            let block_end = (block_start + block_size).min(n);

            // Process block with overlap to ensure continuity
            let extended_start = if block_start > 0 { block_start - 1 } else { 0 };
            let extended_end = (block_end + 1).min(n);

            let x_block = x
                .slice(scirs2_core::ndarray::s![extended_start..extended_end])
                .to_owned();
            let y_block = result
                .slice(scirs2_core::ndarray::s![extended_start..extended_end])
                .to_owned();

            let block_result =
                isotonic_regression(&x_block, &y_block, Some(self.increasing), None, None)?;

            // Update the main result (excluding overlap regions)
            let update_start = block_start - extended_start;
            let update_end = block_end - extended_start;
            result
                .slice_mut(scirs2_core::ndarray::s![block_start..block_end])
                .assign(&block_result.slice(scirs2_core::ndarray::s![update_start..update_end]));
        }

        Ok(result)
    }

    /// Pool Adjacent Violators Algorithm (PAVA) in-place
    fn pava_in_place(
        &self,
        x: ArrayView1<Float>,
        mut y: ArrayViewMut1<Float>,
    ) -> Result<(), SklearsError> {
        let n = y.len();
        if n <= 1 {
            return Ok(());
        }

        let mut weights = Array1::ones(n);
        let mut i = 0;

        while i < n - 1 {
            if (self.increasing && y[i] > y[i + 1]) || (!self.increasing && y[i] < y[i + 1]) {
                // Violation found, pool adjacent values
                let mut j = i;
                let mut sum = 0.0;
                let mut weight_sum = 0.0;

                // Find the extent of the violation
                while j < n - 1
                    && ((self.increasing && y[j] > y[j + 1])
                        || (!self.increasing && y[j] < y[j + 1]))
                {
                    j += 1;
                }

                // Pool values from i to j
                for k in i..=j {
                    sum += y[k] * weights[k];
                    weight_sum += weights[k];
                }

                let pooled_value = sum / weight_sum;

                // Update values and weights
                for k in i..=j {
                    y[k] = pooled_value;
                    weights[k] = weight_sum;
                }

                // Step back to check for new violations
                if i > 0 {
                    i -= 1;
                } else {
                    i = j + 1;
                }
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    /// Memory-efficient interpolation
    fn memory_efficient_interpolation(
        &self,
        x_new: &Array1<Float>,
        fitted_values: &Array1<Float>,
    ) -> Result<Array1<Float>, SklearsError> {
        // Use chunked interpolation to reduce memory usage
        let chunk_size = self.config.block_size;
        let mut predictions = Array1::zeros(x_new.len());

        for chunk_start in (0..x_new.len()).step_by(chunk_size) {
            let chunk_end = (chunk_start + chunk_size).min(x_new.len());

            // Process chunk
            for i in chunk_start..chunk_end {
                predictions[i] = self.interpolate_single(x_new[i], fitted_values)?;
            }
        }

        Ok(predictions)
    }

    /// Interpolate a single value
    fn interpolate_single(
        &self,
        x: Float,
        fitted_values: &Array1<Float>,
    ) -> Result<Float, SklearsError> {
        let n = fitted_values.len();
        if n == 0 {
            return Ok(0.0);
        }
        if n == 1 {
            return Ok(fitted_values[0]);
        }

        // Simple linear interpolation (assuming x corresponds to indices)
        let index = x.max(0.0).min((n - 1) as f64);
        let lower_idx = index.floor() as usize;
        let upper_idx = (lower_idx + 1).min(n - 1);

        if lower_idx == upper_idx {
            Ok(fitted_values[lower_idx])
        } else {
            let weight = index - lower_idx as f64;
            Ok(fitted_values[lower_idx] * (1.0 - weight) + fitted_values[upper_idx] * weight)
        }
    }
}

impl Default for MemoryEfficientIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Cache-friendly block processor for large datasets
pub struct BlockProcessor {
    /// Block size for processing
    block_size: usize,
    /// Whether to overlap blocks
    use_overlap: bool,
    /// Cache for recently processed blocks
    block_cache: HashMap<usize, Array1<Float>>,
    /// Maximum cache size
    max_cache_size: usize,
}

impl BlockProcessor {
    /// Create a new block processor
    pub fn new(block_size: usize, max_cache_size: usize) -> Self {
        Self {
            block_size,
            use_overlap: true,
            block_cache: HashMap::new(),
            max_cache_size,
        }
    }

    /// Process data in blocks with caching
    pub fn process_blocks<F>(
        &mut self,
        data: &Array1<Float>,
        processor: F,
    ) -> Result<Array1<Float>, SklearsError>
    where
        F: Fn(&Array1<Float>) -> Result<Array1<Float>, SklearsError>,
    {
        let n = data.len();
        let mut result = Array1::zeros(n);

        for block_start in (0..n).step_by(self.block_size) {
            let block_end = (block_start + self.block_size).min(n);
            let block_id = block_start / self.block_size;

            // Check cache first
            if let Some(cached_result) = self.block_cache.get(&block_id) {
                let copy_len = cached_result.len().min(block_end - block_start);
                result
                    .slice_mut(scirs2_core::ndarray::s![block_start..block_start + copy_len])
                    .assign(&cached_result.slice(scirs2_core::ndarray::s![..copy_len]));
                continue;
            }

            // Process block
            let block_data = data.slice(scirs2_core::ndarray::s![block_start..block_end]).to_owned();
            let block_result = processor(&block_data)?;

            // Store in cache if not full
            if self.block_cache.len() < self.max_cache_size {
                self.block_cache.insert(block_id, block_result.clone());
            }

            // Store results
            result
                .slice_mut(scirs2_core::ndarray::s![block_start..block_end])
                .assign(&block_result);
        }

        Ok(result)
    }

    /// Clear the cache
    pub fn clear_cache(&mut self) {
        self.block_cache.clear();
    }

    /// Get cache hit statistics
    pub fn cache_stats(&self) -> (usize, usize) {
        (self.block_cache.len(), self.max_cache_size)
    }
}

/// Cache-aware data layout for optimized memory access patterns
#[derive(Debug, Clone)]
/// CacheAwareLayout
pub struct CacheAwareLayout {
    /// Data organized in cache-line-friendly blocks
    pub blocks: Vec<CacheBlock>,
    /// Block size (typically cache line size * factor)
    pub block_size: usize,
    /// Number of elements per cache line
    pub elements_per_cache_line: usize,
    /// Padding to avoid false sharing
    pub padding_size: usize,
}

/// A single cache-friendly block of data
#[derive(Debug, Clone)]
/// CacheBlock
pub struct CacheBlock {
    /// Data values aligned for cache efficiency
    pub data: Vec<Float>,
    /// Metadata about the block
    pub metadata: BlockMetadata,
    /// Padding to align to cache line boundaries
    _padding: Vec<u8>,
}

/// Metadata for cache blocks
#[derive(Debug, Clone)]
/// BlockMetadata
pub struct BlockMetadata {
    /// Starting index in the original array
    pub start_index: usize,
    /// Number of valid elements in this block
    pub valid_elements: usize,
    /// Whether this block has been modified
    pub is_dirty: bool,
    /// Last access timestamp (for LRU caching)
    pub last_access: u64,
}

impl CacheAwareLayout {
    /// Create a new cache-aware layout
    pub fn new(data: &Array1<Float>, cache_line_size: usize) -> Self {
        let elements_per_cache_line = cache_line_size / std::mem::size_of::<Float>();
        let block_size = elements_per_cache_line * 4; // 4 cache lines per block
        let padding_size = cache_line_size;

        let mut blocks = Vec::new();
        let data_len = data.len();

        for (block_idx, chunk_start) in (0..data_len).step_by(block_size).enumerate() {
            let chunk_end = (chunk_start + block_size).min(data_len);
            let chunk_len = chunk_end - chunk_start;

            let mut block_data = vec![0.0; block_size]; // Always allocate full block size
            for (i, &value) in data
                .slice(scirs2_core::ndarray::s![chunk_start..chunk_end])
                .iter()
                .enumerate()
            {
                block_data[i] = value;
            }

            let block = CacheBlock {
                data: block_data,
                metadata: BlockMetadata {
                    start_index: chunk_start,
                    valid_elements: chunk_len,
                    is_dirty: false,
                    last_access: block_idx as u64,
                },
                _padding: vec![0u8; padding_size],
            };

            blocks.push(block);
        }

        Self {
            blocks,
            block_size,
            elements_per_cache_line,
            padding_size,
        }
    }

    /// Access element at index with cache-friendly pattern
    pub fn get(&mut self, index: usize) -> Float {
        let block_idx = index / self.block_size;
        let local_idx = index % self.block_size;

        if block_idx < self.blocks.len()
            && local_idx < self.blocks[block_idx].metadata.valid_elements
        {
            self.blocks[block_idx].metadata.last_access = self.get_timestamp();
            self.blocks[block_idx].data[local_idx]
        } else {
            0.0
        }
    }

    /// Set element at index with cache-friendly pattern
    pub fn set(&mut self, index: usize, value: Float) {
        let block_idx = index / self.block_size;
        let local_idx = index % self.block_size;

        if block_idx < self.blocks.len()
            && local_idx < self.blocks[block_idx].metadata.valid_elements
        {
            self.blocks[block_idx].data[local_idx] = value;
            self.blocks[block_idx].metadata.is_dirty = true;
            self.blocks[block_idx].metadata.last_access = self.get_timestamp();
        }
    }

    /// Process block with cache-friendly access pattern
    pub fn process_block<F>(&mut self, block_idx: usize, processor: F) -> Result<(), SklearsError>
    where
        F: FnOnce(&mut [Float]) -> Result<(), SklearsError>,
    {
        if block_idx >= self.blocks.len() {
            return Err(SklearsError::InvalidInput(
                "Block index out of bounds".to_string(),
            ));
        }

        let timestamp = self.get_timestamp();
        let block = &mut self.blocks[block_idx];
        let valid_data = &mut block.data[..block.metadata.valid_elements];

        processor(valid_data)?;

        block.metadata.is_dirty = true;
        block.metadata.last_access = timestamp;

        Ok(())
    }

    /// Convert back to dense array
    pub fn to_dense(&self) -> Array1<Float> {
        let total_elements: usize = self
            .blocks
            .iter()
            .map(|block| block.metadata.valid_elements)
            .sum();

        let mut result = Array1::zeros(total_elements);
        let mut result_idx = 0;

        for block in &self.blocks {
            let valid_elements = block.metadata.valid_elements;
            for i in 0..valid_elements {
                result[result_idx] = block.data[i];
                result_idx += 1;
            }
        }

        result
    }

    /// Get cache efficiency statistics
    pub fn cache_stats(&self) -> CacheStats {
        let total_blocks = self.blocks.len();
        let dirty_blocks = self.blocks.iter().filter(|b| b.metadata.is_dirty).count();
        let total_memory = self
            .blocks
            .iter()
            .map(|b| b.data.capacity() * std::mem::size_of::<Float>() + b._padding.len())
            .sum();

        CacheStats {
            total_blocks,
            dirty_blocks,
            total_memory_bytes: total_memory,
            cache_line_utilization: self.calculate_cache_line_utilization(),
        }
    }

    fn get_timestamp(&self) -> u64 {
        // Simple timestamp - could be replaced with actual time if needed
        self.blocks
            .iter()
            .map(|b| b.metadata.last_access)
            .max()
            .unwrap_or(0)
            + 1
    }

    fn calculate_cache_line_utilization(&self) -> f64 {
        if self.blocks.is_empty() {
            return 0.0;
        }

        let total_elements: usize = self.blocks.iter().map(|b| b.metadata.valid_elements).sum();
        let allocated_elements = self.blocks.len() * self.block_size;

        total_elements as f64 / allocated_elements as f64
    }
}

/// Cache performance statistics
#[derive(Debug, Clone)]
/// CacheStats
pub struct CacheStats {
    /// Total number of blocks
    pub total_blocks: usize,
    /// Number of dirty (modified) blocks
    pub dirty_blocks: usize,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Cache line utilization ratio (0.0 to 1.0)
    pub cache_line_utilization: f64,
}

/// Optimized data structure for cache-friendly isotonic regression
#[derive(Debug, Clone)]
/// CacheFriendlyIsotonicData
pub struct CacheFriendlyIsotonicData {
    /// X values in cache-aware layout
    pub x_layout: CacheAwareLayout,
    /// Y values in cache-aware layout
    pub y_layout: CacheAwareLayout,
    /// Fitted values in cache-aware layout
    pub fitted_layout: Option<CacheAwareLayout>,
    /// Cache configuration
    pub cache_config: CacheConfig,
}

/// Configuration for cache optimization
#[derive(Debug, Clone)]
/// CacheConfig
pub struct CacheConfig {
    /// Cache line size in bytes (typically 64)
    pub cache_line_size: usize,
    /// L1 cache size in bytes
    pub l1_cache_size: usize,
    /// L2 cache size in bytes
    pub l2_cache_size: usize,
    /// Whether to use prefetching
    pub use_prefetch: bool,
    /// Prefetch distance (number of blocks ahead)
    pub prefetch_distance: usize,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_line_size: 64,   // Common cache line size
            l1_cache_size: 32768,  // 32KB L1 cache
            l2_cache_size: 262144, // 256KB L2 cache
            use_prefetch: true,
            prefetch_distance: 2,
        }
    }
}

impl CacheFriendlyIsotonicData {
    /// Create new cache-friendly isotonic data structure
    pub fn new(x: &Array1<Float>, y: &Array1<Float>, config: CacheConfig) -> Self {
        let x_layout = CacheAwareLayout::new(x, config.cache_line_size);
        let y_layout = CacheAwareLayout::new(y, config.cache_line_size);

        Self {
            x_layout,
            y_layout,
            fitted_layout: None,
            cache_config: config,
        }
    }

    /// Perform cache-friendly isotonic regression
    pub fn fit_isotonic(&mut self, increasing: bool) -> Result<(), SklearsError> {
        let num_blocks = self.y_layout.blocks.len();

        // Create fitted layout with same structure as y_layout
        let y_dense = self.y_layout.to_dense();
        let mut fitted_layout = CacheAwareLayout::new(&y_dense, self.cache_config.cache_line_size);

        // Process each block with cache-friendly access patterns
        for block_idx in 0..num_blocks {
            // Prefetch next blocks if enabled
            if self.cache_config.use_prefetch
                && block_idx + self.cache_config.prefetch_distance < num_blocks
            {
                // In a real implementation, this would issue prefetch instructions
                // For now, we just access the metadata to simulate cache warming
                let _ =
                    &self.y_layout.blocks[block_idx + self.cache_config.prefetch_distance].metadata;
            }

            // Apply PAVA algorithm within each block
            fitted_layout.process_block(block_idx, |block_data| {
                self.apply_pava_to_block(block_data, increasing)
            })?;
        }

        // Handle cross-block boundaries
        self.fix_block_boundaries(&mut fitted_layout, increasing)?;

        self.fitted_layout = Some(fitted_layout);
        Ok(())
    }

    /// Apply Pool Adjacent Violators Algorithm to a single block
    fn apply_pava_to_block(
        &self,
        data: &mut [Float],
        increasing: bool,
    ) -> Result<(), SklearsError> {
        let n = data.len();
        if n <= 1 {
            return Ok(());
        }

        let mut i = 0;
        while i < n - 1 {
            if (increasing && data[i] > data[i + 1]) || (!increasing && data[i] < data[i + 1]) {
                // Find extent of violation
                let mut j = i + 1;
                let mut sum = data[i] + data[j];
                let mut count = 2;

                while j < n - 1 {
                    if (increasing && data[i] > data[j + 1])
                        || (!increasing && data[i] < data[j + 1])
                    {
                        j += 1;
                        sum += data[j];
                        count += 1;
                    } else {
                        break;
                    }
                }

                // Pool violating values
                let pooled_value = sum / count as Float;
                for k in i..=j {
                    data[k] = pooled_value;
                }

                // Restart from beginning of pooled region
                if i > 0 {
                    i -= 1;
                } else {
                    i = j + 1;
                }
            } else {
                i += 1;
            }
        }

        Ok(())
    }

    /// Fix violations at block boundaries
    fn fix_block_boundaries(
        &self,
        fitted_layout: &mut CacheAwareLayout,
        increasing: bool,
    ) -> Result<(), SklearsError> {
        let num_blocks = fitted_layout.blocks.len();

        for i in 0..(num_blocks - 1) {
            // Get the necessary values without holding references
            let (current_valid_elements, last_val, first_val) = {
                let current_block = &fitted_layout.blocks[i];
                let next_block = &fitted_layout.blocks[i + 1];

                if current_block.metadata.valid_elements == 0
                    || next_block.metadata.valid_elements == 0
                {
                    continue;
                }

                let last_val = current_block.data[current_block.metadata.valid_elements - 1];
                let first_val = next_block.data[0];

                (current_block.metadata.valid_elements, last_val, first_val)
            };

            let is_violation =
                (increasing && last_val > first_val) || (!increasing && last_val < first_val);

            if is_violation {
                let pooled_value = (last_val + first_val) / 2.0;
                fitted_layout.blocks[i].data[current_valid_elements - 1] = pooled_value;
                fitted_layout.blocks[i + 1].data[0] = pooled_value;
                fitted_layout.blocks[i].metadata.is_dirty = true;
                fitted_layout.blocks[i + 1].metadata.is_dirty = true;
            }
        }

        Ok(())
    }

    /// Get fitted values as dense array
    pub fn fitted_values(&self) -> Option<Array1<Float>> {
        self.fitted_layout.as_ref().map(|layout| layout.to_dense())
    }

    /// Get cache performance statistics
    pub fn cache_performance(&self) -> CachePerformanceReport {
        let x_stats = self.x_layout.cache_stats();
        let y_stats = self.y_layout.cache_stats();
        let fitted_stats = self
            .fitted_layout
            .as_ref()
            .map(|layout| layout.cache_stats());

        CachePerformanceReport {
            x_cache_stats: x_stats,
            y_cache_stats: y_stats,
            fitted_cache_stats: fitted_stats,
            total_memory_bytes: self.total_memory_usage(),
            cache_efficiency: self.calculate_cache_efficiency(),
        }
    }

    fn total_memory_usage(&self) -> usize {
        let x_memory = self.x_layout.cache_stats().total_memory_bytes;
        let y_memory = self.y_layout.cache_stats().total_memory_bytes;
        let fitted_memory = self
            .fitted_layout
            .as_ref()
            .map(|layout| layout.cache_stats().total_memory_bytes)
            .unwrap_or(0);

        x_memory + y_memory + fitted_memory
    }

    fn calculate_cache_efficiency(&self) -> f64 {
        let x_util = self.x_layout.cache_stats().cache_line_utilization;
        let y_util = self.y_layout.cache_stats().cache_line_utilization;
        let fitted_util = self
            .fitted_layout
            .as_ref()
            .map(|layout| layout.cache_stats().cache_line_utilization)
            .unwrap_or(0.0);

        (x_util + y_util + fitted_util) / 3.0
    }
}

/// Cache performance report
#[derive(Debug, Clone)]
/// CachePerformanceReport
pub struct CachePerformanceReport {
    /// Cache statistics for X data
    pub x_cache_stats: CacheStats,
    /// Cache statistics for Y data
    pub y_cache_stats: CacheStats,
    /// Cache statistics for fitted data (if available)
    pub fitted_cache_stats: Option<CacheStats>,
    /// Total memory usage in bytes
    pub total_memory_bytes: usize,
    /// Overall cache efficiency (0.0 to 1.0)
    pub cache_efficiency: f64,
}

// Function APIs for memory efficiency

/// Perform memory-efficient isotonic regression
pub fn memory_efficient_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    config: Option<MemoryConfig>,
    increasing: bool,
) -> Result<(Array1<Float>, MemoryStats), SklearsError> {
    let mut model = MemoryEfficientIsotonicRegression::new()
        .config(config.unwrap_or_default())
        .increasing(increasing);

    model.fit(x, y)?;

    let fitted_values = model.fitted_values().unwrap().clone();
    let memory_stats = model.memory_stats().clone();

    Ok((fitted_values, memory_stats))
}

/// Create sparse vector from dense data
pub fn create_sparse_vector(dense: &Array1<Float>, threshold: f64) -> SparseVector {
    SparseVector::from_dense(dense, threshold)
}

/// Process data in memory-efficient blocks
pub fn process_in_blocks<F>(
    data: &Array1<Float>,
    block_size: usize,
    processor: F,
) -> Result<Array1<Float>, SklearsError>
where
    F: Fn(&Array1<Float>) -> Result<Array1<Float>, SklearsError>,
{
    let mut block_processor = BlockProcessor::new(block_size, 10);
    block_processor.process_blocks(data, processor)
}

/// Perform cache-friendly isotonic regression
pub fn cache_friendly_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    cache_config: Option<CacheConfig>,
) -> Result<(Array1<Float>, CachePerformanceReport), SklearsError> {
    let config = cache_config.unwrap_or_default();
    let mut cache_data = CacheFriendlyIsotonicData::new(x, y, config);

    cache_data.fit_isotonic(increasing)?;

    let fitted_values = cache_data
        .fitted_values()
        .ok_or_else(|| SklearsError::InvalidInput("Failed to get fitted values".to_string()))?;

    let performance_report = cache_data.cache_performance();

    Ok((fitted_values, performance_report))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_efficient_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let mut model = MemoryEfficientIsotonicRegression::new();
        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x);
        assert!(predictions.is_ok());

        let fitted = predictions.unwrap();
        assert_eq!(fitted.len(), 5);

        // Check monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }

        // Check memory stats
        let stats = model.memory_stats();
        assert!(stats.allocations > 0);
        assert!(stats.current_memory_bytes > 0);
    }

    #[test]
    fn test_sparse_vector() {
        let dense = array![1.0, 0.0, 0.0, 2.0, 0.0];
        let sparse = SparseVector::from_dense(&dense, 1e-10);

        assert_eq!(sparse.length, 5);
        assert_eq!(sparse.indices, vec![0, 3]);
        assert_eq!(sparse.values, vec![1.0, 2.0]);
        assert_abs_diff_eq!(sparse.density(), 0.4, epsilon = 1e-10);

        // Test conversion back to dense
        let reconstructed = sparse.to_dense();
        for i in 0..dense.len() {
            assert_abs_diff_eq!(reconstructed[i], dense[i], epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sparse_vector_operations() {
        let mut sparse = SparseVector::new(5);

        sparse.set(1, 2.0);
        sparse.set(3, 4.0);

        assert_eq!(sparse.get(1), 2.0);
        assert_eq!(sparse.get(3), 4.0);
        assert_eq!(sparse.get(0), 0.0);
        assert_eq!(sparse.get(2), 0.0);

        // Test removal of zero value
        sparse.set(1, 0.0);
        assert_eq!(sparse.get(1), 0.0);
        assert_eq!(sparse.values.len(), 1);
    }

    #[test]
    fn test_memory_config() {
        let config = MemoryConfig {
            max_memory_bytes: 1000,
            block_size: 10,
            use_in_place: false,
            use_sparse: false,
            sparse_threshold: 0.5,
            streaming_chunk_size: 5,
        };

        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0, 3.0];

        let mut model = MemoryEfficientIsotonicRegression::new().config(config);
        assert!(model.fit(&x, &y).is_ok());

        let stats = model.memory_stats();
        assert!(!stats.used_sparse); // Disabled in config
    }

    #[test]
    fn test_streaming_processing() {
        let config = MemoryConfig {
            streaming_chunk_size: 2,
            ..Default::default()
        };

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0, 6.0];

        let mut model = MemoryEfficientIsotonicRegression::new().config(config);
        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x);
        assert!(predictions.is_ok());
        assert_eq!(predictions.unwrap().len(), 6);
    }

    #[test]
    fn test_block_processor() {
        let mut processor = BlockProcessor::new(3, 2);
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let result = processor.process_blocks(&data, |block| Ok(block.mapv(|x| x * 2.0)));

        assert!(result.is_ok());
        let processed = result.unwrap();
        assert_eq!(processed.len(), 6);
        for i in 0..6 {
            assert_abs_diff_eq!(processed[i], data[i] * 2.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_function_api() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![1.0, 3.0, 2.0, 4.0];

        let config = MemoryConfig::default();
        let result = memory_efficient_isotonic_regression(&x, &y, Some(config), true);
        assert!(result.is_ok());

        let (fitted, stats) = result.unwrap();
        assert_eq!(fitted.len(), 4);
        assert!(stats.allocations > 0);
    }

    #[test]
    fn test_sparse_isotonic_regression() {
        // Create data that benefits from sparse representation
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![0.0, 0.0, 1.0, 0.0, 0.0, 2.0, 0.0, 0.0];

        let config = MemoryConfig {
            use_sparse: true,
            sparse_threshold: 0.5,
            ..Default::default()
        };

        let mut model = MemoryEfficientIsotonicRegression::new().config(config);
        assert!(model.fit(&x, &y).is_ok());

        let stats = model.memory_stats();
        assert!(stats.used_sparse);
    }

    #[test]
    fn test_cache_friendly_processing() {
        let config = MemoryConfig {
            block_size: 3,
            use_in_place: false,
            use_sparse: false,
            ..Default::default()
        };

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0, 4.0, 6.0];

        let mut model = MemoryEfficientIsotonicRegression::new().config(config);
        assert!(model.fit(&x, &y).is_ok());

        let predictions = model.predict(&x);
        assert!(predictions.is_ok());
    }

    #[test]
    fn test_memory_limits() {
        let config = MemoryConfig {
            max_memory_bytes: 100, // Very small limit to force streaming
            streaming_chunk_size: 2,
            ..Default::default()
        };

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let mut model = MemoryEfficientIsotonicRegression::new().config(config);
        assert!(model.fit(&x, &y).is_ok());
    }

    #[test]
    fn test_cache_aware_layout() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut layout = CacheAwareLayout::new(&data, 64);

        // Test access patterns
        assert_eq!(layout.get(0), 1.0);
        assert_eq!(layout.get(3), 4.0);

        // Test modification
        layout.set(2, 10.0);
        assert_eq!(layout.get(2), 10.0);

        // Test conversion back to dense
        let dense = layout.to_dense();
        assert_eq!(dense.len(), 8);
        assert_eq!(dense[2], 10.0);

        // Test cache statistics
        let stats = layout.cache_stats();
        assert!(stats.total_blocks > 0);
        assert!(stats.dirty_blocks > 0);
    }

    #[test]
    fn test_cache_friendly_isotonic_data() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];
        let config = CacheConfig::default();

        let mut cache_data = CacheFriendlyIsotonicData::new(&x, &y, config);
        assert!(cache_data.fit_isotonic(true).is_ok());

        let fitted = cache_data.fitted_values();
        assert!(fitted.is_some());

        let fitted_values = fitted.unwrap();
        assert_eq!(fitted_values.len(), 5);

        // Check monotonicity
        for i in 0..fitted_values.len() - 1 {
            assert!(fitted_values[i] <= fitted_values[i + 1]);
        }

        // Test performance report
        let performance = cache_data.cache_performance();
        assert!(performance.total_memory_bytes > 0);
        assert!(performance.cache_efficiency >= 0.0 && performance.cache_efficiency <= 1.0);
    }

    #[test]
    fn test_cache_friendly_isotonic_function() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![5.0, 3.0, 4.0, 2.0, 1.0]; // Decreasing trend

        let result = cache_friendly_isotonic_regression(&x, &y, false, None);
        assert!(result.is_ok());

        let (fitted, performance) = result.unwrap();
        assert_eq!(fitted.len(), 5);

        // Check decreasing monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] >= fitted[i + 1]);
        }

        assert!(performance.total_memory_bytes > 0);
        assert!(performance.cache_efficiency >= 0.0);
    }

    #[test]
    fn test_cache_config_custom() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 3.0, 6.0, 5.0, 8.0];

        let config = CacheConfig {
            cache_line_size: 32,
            l1_cache_size: 16384,
            l2_cache_size: 131072,
            use_prefetch: true,
            prefetch_distance: 1,
        };

        let result = cache_friendly_isotonic_regression(&x, &y, true, Some(config));
        assert!(result.is_ok());

        let (fitted, _) = result.unwrap();
        assert_eq!(fitted.len(), 8);

        // Verify monotonicity
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }
    }

    #[test]
    fn test_block_metadata() {
        let data = array![1.0, 2.0, 3.0, 4.0];
        let layout = CacheAwareLayout::new(&data, 64);

        assert!(layout.blocks.len() > 0);

        let first_block = &layout.blocks[0];
        assert_eq!(first_block.metadata.start_index, 0);
        assert!(first_block.metadata.valid_elements > 0);
        assert!(!first_block.metadata.is_dirty);
    }

    #[test]
    fn test_cache_line_utilization() {
        let data = array![1.0, 2.0, 3.0];
        let layout = CacheAwareLayout::new(&data, 64);

        let stats = layout.cache_stats();
        assert!(stats.cache_line_utilization > 0.0);
        assert!(stats.cache_line_utilization <= 1.0);
    }
}

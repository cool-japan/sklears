//! Cache-friendly algorithms and data structures for high-performance dummy estimator operations

use rayon::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView2};

/// Cache-aligned data structure for frequent access patterns
#[repr(align(64))] // Cache line alignment
pub struct CacheAlignedStats {
    /// count
    pub count: usize,
    /// sum
    pub sum: f64,
    /// sum_squares
    pub sum_squares: f64,
    /// min
    pub min: f64,
    /// max
    pub max: f64,
    _padding: [u8; 32], // Padding to prevent false sharing
}

impl Default for CacheAlignedStats {
    fn default() -> Self {
        Self::new()
    }
}

impl CacheAlignedStats {
    pub fn new() -> Self {
        Self {
            count: 0,
            sum: 0.0,
            sum_squares: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            _padding: [0; 32],
        }
    }

    pub fn update(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.sum_squares += value * value;
        self.min = self.min.min(value);
        self.max = self.max.max(value);
    }

    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn variance(&self) -> f64 {
        if self.count <= 1 {
            0.0
        } else {
            let mean = self.mean();
            (self.sum_squares / self.count as f64) - (mean * mean)
        }
    }
}

/// Memory-efficient prediction storage
pub struct PredictionBuffer<T> {
    data: Vec<T>,
    capacity: usize,
    write_index: usize,
}

impl<T: Clone + Default> PredictionBuffer<T> {
    pub fn new(capacity: usize) -> Self {
        Self {
            data: vec![T::default(); capacity],
            capacity,
            write_index: 0,
        }
    }

    pub fn push(&mut self, value: T) {
        self.data[self.write_index % self.capacity] = value;
        self.write_index += 1;
    }

    pub fn get_latest(&self, n: usize) -> Vec<T> {
        let n = n.min(self.capacity).min(self.write_index);
        let start_index = if self.write_index >= n {
            (self.write_index - n) % self.capacity
        } else {
            0
        };

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let index = (start_index + i) % self.capacity;
            result.push(self.data[index].clone());
        }
        result
    }
}

/// Blocked matrix operations for better cache utilization
pub fn blocked_matrix_operation<F>(
    matrix: &Array2<f64>,
    block_size: usize,
    operation: F,
) -> Vec<f64>
where
    F: Fn(ArrayView2<f64>) -> f64 + Sync + Send,
{
    let (n_rows, n_cols) = matrix.dim();
    let mut results = Vec::with_capacity((n_rows / block_size + 1) * (n_cols / block_size + 1));

    for row_start in (0..n_rows).step_by(block_size) {
        for col_start in (0..n_cols).step_by(block_size) {
            let row_end = (row_start + block_size).min(n_rows);
            let col_end = (col_start + block_size).min(n_cols);

            let block = matrix.slice(scirs2_core::ndarray::s![
                row_start..row_end,
                col_start..col_end
            ]);
            results.push(operation(block));
        }
    }

    results
}

/// Block-wise processing for better cache utilization
pub fn block_mean(data: &[f64], block_size: usize) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    let total_sum: f64 = data
        .chunks(block_size)
        .map(|chunk| chunk.iter().sum::<f64>())
        .sum();

    total_sum / data.len() as f64
}

/// Cache-friendly data prefetching
pub fn prefetch_data<T>(data: &[T], access_pattern: &[usize]) {
    for &index in access_pattern {
        if index < data.len() {
            // Compiler hint for prefetching
            let _prefetch = unsafe { std::ptr::read_volatile(&data[index] as *const T) };
        }
    }
}

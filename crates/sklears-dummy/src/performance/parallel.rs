//! Parallel computation utilities for high-performance dummy estimator operations

use rayon::prelude::*;
use scirs2_core::ndarray::{Array2, ArrayView1, Axis};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Parallel prediction for large datasets
pub fn parallel_predict_classification<F>(
    x: &Array2<f64>,
    predict_fn: F,
    chunk_size: Option<usize>,
) -> Vec<i32>
where
    F: Fn(ArrayView1<f64>) -> i32 + Sync + Send,
{
    let chunk_size = chunk_size.unwrap_or(1000);

    if x.nrows() <= chunk_size {
        // Small dataset - use sequential processing
        return x.rows().into_iter().map(predict_fn).collect();
    }

    // Large dataset - use parallel processing with order preservation
    let mut results: Vec<(usize, Vec<i32>)> = x
        .axis_chunks_iter(Axis(0), chunk_size)
        .enumerate()
        .par_bridge()
        .map(|(chunk_idx, chunk)| {
            let chunk_results: Vec<i32> = chunk.rows().into_iter().map(&predict_fn).collect();
            (chunk_idx, chunk_results)
        })
        .collect();

    // Sort by chunk index to preserve order
    results.sort_by_key(|(idx, _)| *idx);

    // Flatten results
    results
        .into_iter()
        .flat_map(|(_, chunk_results)| chunk_results)
        .collect()
}

/// Parallel prediction for regression
pub fn parallel_predict_regression<F>(
    x: &Array2<f64>,
    predict_fn: F,
    chunk_size: Option<usize>,
) -> Vec<f64>
where
    F: Fn(ArrayView1<f64>) -> f64 + Sync + Send,
{
    let chunk_size = chunk_size.unwrap_or(1000);

    if x.nrows() <= chunk_size {
        return x.rows().into_iter().map(predict_fn).collect();
    }

    // Large dataset - use parallel processing with order preservation
    let mut results: Vec<(usize, Vec<f64>)> = x
        .axis_chunks_iter(Axis(0), chunk_size)
        .enumerate()
        .par_bridge()
        .map(|(chunk_idx, chunk)| {
            let chunk_results: Vec<f64> = chunk.rows().into_iter().map(&predict_fn).collect();
            (chunk_idx, chunk_results)
        })
        .collect();

    // Sort by chunk index to preserve order
    results.sort_by_key(|(idx, _)| *idx);

    // Flatten results
    results
        .into_iter()
        .flat_map(|(_, chunk_results)| chunk_results)
        .collect()
}

/// Parallel statistical computation
pub fn parallel_statistics(data: &Array2<f64>) -> ParallelStats {
    let chunk_size = (data.nrows() / rayon::current_num_threads()).max(1000);

    let results: Vec<_> = data
        .axis_chunks_iter(Axis(0), chunk_size)
        .par_bridge()
        .map(|chunk| {
            let mut local_stats = ParallelStats::new(data.ncols());
            for row in chunk.rows() {
                local_stats.update_row(row);
            }
            local_stats
        })
        .collect();

    // Combine results
    let mut combined = ParallelStats::new(data.ncols());
    for stats in results {
        combined.merge(stats);
    }
    combined.finalize();
    combined
}

/// Thread-safe statistics accumulator
#[derive(Debug)]
pub struct ParallelStats {
    /// count
    pub count: AtomicUsize,
    /// sums
    pub sums: Vec<AtomicU64>, // Use atomic for thread-safe f64 accumulation
    /// sum_squares
    pub sum_squares: Vec<AtomicU64>,
    /// mins
    pub mins: Vec<f64>,
    /// maxs
    pub maxs: Vec<f64>,
}

impl ParallelStats {
    pub fn new(n_features: usize) -> Self {
        Self {
            count: AtomicUsize::new(0),
            sums: (0..n_features).map(|_| AtomicU64::new(0)).collect(),
            sum_squares: (0..n_features).map(|_| AtomicU64::new(0)).collect(),
            mins: vec![f64::INFINITY; n_features],
            maxs: vec![f64::NEG_INFINITY; n_features],
        }
    }

    pub fn update_row(&mut self, row: ArrayView1<f64>) {
        self.count.fetch_add(1, Ordering::Relaxed);

        for (i, &value) in row.iter().enumerate() {
            // Use bit manipulation for atomic f64 operations
            let value_bits = value.to_bits();
            let squared_bits = (value * value).to_bits();

            self.sums[i].fetch_add(value_bits, Ordering::Relaxed);
            self.sum_squares[i].fetch_add(squared_bits, Ordering::Relaxed);

            // Note: mins/maxs are not thread-safe in this simple implementation
            // For production use, consider using atomic operations or locks
            if value < self.mins[i] {
                self.mins[i] = value;
            }
            if value > self.maxs[i] {
                self.maxs[i] = value;
            }
        }
    }

    pub fn merge(&mut self, other: ParallelStats) {
        self.count
            .fetch_add(other.count.load(Ordering::Relaxed), Ordering::Relaxed);

        for i in 0..self.sums.len() {
            self.sums[i].fetch_add(other.sums[i].load(Ordering::Relaxed), Ordering::Relaxed);
            self.sum_squares[i].fetch_add(
                other.sum_squares[i].load(Ordering::Relaxed),
                Ordering::Relaxed,
            );

            self.mins[i] = self.mins[i].min(other.mins[i]);
            self.maxs[i] = self.maxs[i].max(other.maxs[i]);
        }
    }

    pub fn finalize(&self) {
        // Convert back from bit representation
        // This is where you'd implement proper finalization
    }

    pub fn get_means(&self) -> Vec<f64> {
        let count = self.count.load(Ordering::Relaxed) as f64;
        self.sums
            .iter()
            .map(|sum| f64::from_bits(sum.load(Ordering::Relaxed)) / count)
            .collect()
    }

    pub fn get_variances(&self) -> Vec<f64> {
        let count = self.count.load(Ordering::Relaxed) as f64;
        let means = self.get_means();

        self.sum_squares
            .iter()
            .zip(means.iter())
            .map(|(sum_sq, &mean)| {
                let sum_sq_val = f64::from_bits(sum_sq.load(Ordering::Relaxed));
                (sum_sq_val / count) - (mean * mean)
            })
            .collect()
    }
}

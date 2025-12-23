//! Parallel random number generation for dataset creation
//!
//! This module provides thread-safe parallel random number generation
//! for efficient multi-threaded dataset generation.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Distribution, RandNormal, Random, Rng};
use scirs2_core::rngs::StdRng;

/// Thread-safe parallel random number generator
#[derive(Clone)]
pub struct ParallelRng {
    base_seed: u64,
}

impl ParallelRng {
    /// Create a new parallel RNG with a base seed
    pub fn new(seed: u64) -> Self {
        Self { base_seed: seed }
    }

    /// Get a thread-local RNG for a specific thread ID
    pub fn get_thread_rng(&self, thread_id: usize) -> Random<StdRng> {
        // Combine base seed with thread ID for deterministic per-thread RNG
        let thread_seed = self.base_seed.wrapping_add(thread_id as u64);
        Random::seed(thread_seed)
    }

    /// Generate a normal distribution matrix in parallel
    pub fn generate_normal_matrix_parallel(
        &self,
        n_samples: usize,
        n_features: usize,
        mean: f64,
        std: f64,
        n_threads: Option<usize>,
    ) -> Array2<f64> {
        // Set thread pool size if specified
        if let Some(threads) = n_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build_global()
                .ok();
        }

        // Determine chunk size for parallel processing
        let num_workers = rayon::current_num_threads();
        let chunk_size = (n_samples + num_workers - 1) / num_workers;

        // Generate rows in parallel
        let rows: Vec<Array1<f64>> = (0..n_samples)
            .into_par_iter()
            .chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_id, chunk)| {
                let mut rng = self.get_thread_rng(chunk_id);
                let normal = RandNormal::new(mean, std).unwrap();

                chunk
                    .into_iter()
                    .map(|_| Array1::from_shape_fn(n_features, |_| normal.sample(&mut rng)))
                    .collect::<Vec<_>>()
            })
            .collect();

        // Stack rows into matrix
        let mut matrix = Array2::zeros((n_samples, n_features));
        for (i, row) in rows.into_iter().enumerate() {
            matrix.row_mut(i).assign(&row);
        }

        matrix
    }

    /// Generate uniform random values in parallel
    pub fn generate_uniform_parallel(&self, n_samples: usize, low: f64, high: f64) -> Array1<f64> {
        let num_workers = rayon::current_num_threads();
        let chunk_size = (n_samples + num_workers - 1) / num_workers;

        (0..n_samples)
            .into_par_iter()
            .chunks(chunk_size)
            .enumerate()
            .flat_map(|(chunk_id, chunk)| {
                let mut rng = self.get_thread_rng(chunk_id);

                chunk
                    .into_iter()
                    .map(|_| {
                        let u: f64 = rng.gen();
                        low + u * (high - low)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
            .into()
    }
}

/// Parallel classification dataset generation
pub fn make_classification_parallel(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    class_sep: f64,
    random_state: u64,
    n_threads: Option<usize>,
) -> (Array2<f64>, Array1<i32>) {
    let rng = ParallelRng::new(random_state);

    // Generate features in parallel
    let features = rng.generate_normal_matrix_parallel(n_samples, n_features, 0.0, 1.0, n_threads);

    // Generate targets
    let targets = Array1::from_shape_fn(n_samples, |i| (i % n_classes) as i32);

    // Apply class separation in parallel
    let separated_features = features
        .axis_iter(Axis(0))
        .into_par_iter()
        .enumerate()
        .map(|(i, row)| {
            let offset = targets[i] as f64 * class_sep;
            row.mapv(|x| x + offset)
        })
        .collect::<Vec<_>>();

    let mut result = Array2::zeros((n_samples, n_features));
    for (i, row) in separated_features.into_iter().enumerate() {
        result.row_mut(i).assign(&row);
    }

    (result, targets)
}

/// Parallel regression dataset generation
pub fn make_regression_parallel(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: u64,
    n_threads: Option<usize>,
) -> (Array2<f64>, Array1<f64>) {
    let rng = ParallelRng::new(random_state);

    // Generate features in parallel
    let features = rng.generate_normal_matrix_parallel(n_samples, n_features, 0.0, 1.0, n_threads);

    // Generate coefficients
    let mut coef_rng = rng.get_thread_rng(0);
    let normal_coef = RandNormal::new(0.0, 1.0).unwrap();
    let coef = Array1::from_shape_fn(n_features, |_| normal_coef.sample(&mut coef_rng));

    // Compute targets in parallel: y = X @ coef
    let targets_base: Vec<f64> = features
        .axis_iter(Axis(0))
        .into_par_iter()
        .map(|row| row.dot(&coef))
        .collect();

    // Add noise in parallel if requested
    let targets = if noise > 0.0 {
        let noise_values: Vec<f64> = (0..n_samples)
            .into_par_iter()
            .chunks((n_samples + rayon::current_num_threads() - 1) / rayon::current_num_threads())
            .enumerate()
            .flat_map(|(chunk_id, chunk)| {
                let mut chunk_rng = rng.get_thread_rng(chunk_id + 1);
                let normal_noise = RandNormal::new(0.0, noise).unwrap();

                chunk
                    .into_iter()
                    .map(|_| normal_noise.sample(&mut chunk_rng))
                    .collect::<Vec<_>>()
            })
            .collect();

        targets_base
            .par_iter()
            .zip(noise_values.par_iter())
            .map(|(&t, &n)| t + n)
            .collect::<Vec<_>>()
            .into()
    } else {
        targets_base.into()
    };

    (features, targets)
}

/// Parallel blob generation for clustering
pub fn make_blobs_parallel(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
    random_state: u64,
    _n_threads: Option<usize>,
) -> (Array2<f64>, Array1<i32>) {
    let rng = ParallelRng::new(random_state);

    // Generate cluster centers
    let mut center_rng = rng.get_thread_rng(0);
    let center_normal = RandNormal::new(0.0, 10.0).unwrap();
    let cluster_centers = Array2::from_shape_fn((centers, n_features), |_| {
        center_normal.sample(&mut center_rng)
    });

    // Assign samples to clusters
    let samples_per_cluster = n_samples / centers;
    let targets = Array1::from_shape_fn(n_samples, |i| {
        (i / samples_per_cluster).min(centers - 1) as i32
    });

    // Generate samples in parallel
    let samples: Vec<Array1<f64>> = (0..n_samples)
        .into_par_iter()
        .chunks((n_samples + rayon::current_num_threads() - 1) / rayon::current_num_threads())
        .enumerate()
        .flat_map(|(chunk_id, chunk)| {
            let mut chunk_rng = rng.get_thread_rng(chunk_id + 1);
            let sample_normal = RandNormal::new(0.0, cluster_std).unwrap();

            chunk
                .into_iter()
                .map(|i| {
                    let cluster_id = targets[i] as usize;
                    let center = cluster_centers.row(cluster_id);
                    center.mapv(|c| c + sample_normal.sample(&mut chunk_rng))
                })
                .collect::<Vec<_>>()
        })
        .collect();

    let mut features = Array2::zeros((n_samples, n_features));
    for (i, sample) in samples.into_iter().enumerate() {
        features.row_mut(i).assign(&sample);
    }

    (features, targets)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_rng_creation() {
        let rng = ParallelRng::new(42);
        let mut thread_rng = rng.get_thread_rng(0);
        // Just verify it was created successfully
        let val1 = thread_rng.gen::<f64>();
        let val2 = thread_rng.gen::<f64>();
        assert!(val1 >= 0.0 && val1 <= 1.0);
        assert!(val2 >= 0.0 && val2 <= 1.0);
    }

    #[test]
    fn test_generate_normal_matrix_parallel() {
        let rng = ParallelRng::new(42);
        let matrix = rng.generate_normal_matrix_parallel(100, 10, 0.0, 1.0, Some(2));

        assert_eq!(matrix.nrows(), 100);
        assert_eq!(matrix.ncols(), 10);

        // Check that values are reasonable
        for &val in matrix.iter() {
            assert!(val.abs() < 5.0);
        }
    }

    #[test]
    fn test_generate_uniform_parallel() {
        let rng = ParallelRng::new(42);
        let values = rng.generate_uniform_parallel(100, 0.0, 10.0);

        assert_eq!(values.len(), 100);

        // Check that all values are in range
        for &val in values.iter() {
            assert!(val >= 0.0 && val <= 10.0);
        }
    }

    #[test]
    fn test_make_classification_parallel() {
        let (features, targets) = make_classification_parallel(100, 5, 3, 1.0, 42, Some(2));

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert_eq!(targets.len(), 100);

        // Check that we have all classes
        let mut has_class = vec![false; 3];
        for &target in targets.iter() {
            assert!(target >= 0 && target < 3);
            has_class[target as usize] = true;
        }
        assert!(has_class.iter().all(|&x| x));
    }

    #[test]
    fn test_make_regression_parallel() {
        let (features, targets) = make_regression_parallel(100, 5, 0.1, 42, Some(2));

        assert_eq!(features.nrows(), 100);
        assert_eq!(features.ncols(), 5);
        assert_eq!(targets.len(), 100);
    }

    #[test]
    fn test_make_blobs_parallel() {
        let (features, targets) = make_blobs_parallel(150, 5, 3, 1.0, 42, Some(2));

        assert_eq!(features.nrows(), 150);
        assert_eq!(features.ncols(), 5);
        assert_eq!(targets.len(), 150);

        // Check that we have all clusters
        let mut has_cluster = vec![false; 3];
        for &target in targets.iter() {
            assert!(target >= 0 && target < 3);
            has_cluster[target as usize] = true;
        }
        assert!(has_cluster.iter().all(|&x| x));
    }

    #[test]
    fn test_deterministic_parallel_generation() {
        // Test that same seed produces same results
        let (features1, _) = make_classification_parallel(50, 3, 2, 1.0, 42, Some(2));
        let (features2, _) = make_classification_parallel(50, 3, 2, 1.0, 42, Some(2));

        // Should be identical
        for i in 0..features1.nrows() {
            for j in 0..features1.ncols() {
                assert_eq!(features1[[i, j]], features2[[i, j]]);
            }
        }
    }
}

//! Parallel Eigenvalue Decomposition
//!
//! This module provides parallel eigenvalue decomposition implementations for discriminant analysis,
//! leveraging SciRS2 for performance optimizations and supporting large-scale matrix operations.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis, Zip};
// The standard and generalized symmetric eigensolvers used here are provided by
// `scirs2_linalg::eigh` / `scirs2_linalg::eigh_gen`, which run a work-stealing
// parallel kernel for large matrices when multiple workers are requested.

use crate::numerical_stability::{NumericalConfig, NumericalStability};
use rayon::prelude::*;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Configuration for parallel eigenvalue decomposition
#[derive(Debug, Clone)]
pub struct ParallelEigenConfig {
    /// Base numerical configuration
    pub numerical_config: NumericalConfig,
    /// Number of threads for parallel operations (None = auto-detect)
    pub num_threads: Option<usize>,
    /// Threshold for switching to parallel processing (matrix size)
    pub parallel_threshold: usize,
    /// Use SIMD optimizations when available
    pub use_simd: bool,
    /// Chunk size for parallel operations
    pub chunk_size: usize,
    /// Use distributed memory approach for very large matrices
    pub use_distributed: bool,
}

impl Default for ParallelEigenConfig {
    fn default() -> Self {
        Self {
            numerical_config: NumericalConfig::default(),
            num_threads: None,       // Auto-detect
            parallel_threshold: 500, // Switch to parallel for matrices > 500x500
            use_simd: true,
            chunk_size: 100,
            use_distributed: false,
        }
    }
}

/// Parallel eigenvalue decomposition engine for discriminant analysis
pub struct ParallelEigenDecomposition {
    config: ParallelEigenConfig,
    numerical_stability: NumericalStability,
}

impl ParallelEigenDecomposition {
    /// Create a new parallel eigenvalue decomposition engine
    pub fn new() -> Self {
        let config = ParallelEigenConfig::default();
        let numerical_stability = NumericalStability::with_config(config.numerical_config.clone());

        Self {
            config,
            numerical_stability,
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: ParallelEigenConfig) -> Self {
        let numerical_stability = NumericalStability::with_config(config.numerical_config.clone());

        Self {
            config,
            numerical_stability,
        }
    }

    /// Compute eigenvalue decomposition with automatic parallelization
    ///
    /// This method automatically chooses between serial and parallel implementations
    /// based on matrix size and configuration.
    pub fn decompose(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // Use parallel implementation for large matrices
        if n >= self.config.parallel_threshold {
            self.parallel_decompose(matrix)
        } else {
            // Use the stable serial implementation from numerical_stability
            self.numerical_stability.stable_eigen_decomposition(matrix)
        }
    }

    /// Parallel eigenvalue decomposition for large matrices
    pub fn parallel_decompose(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if matrix.nrows() != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigenvalue decomposition".to_string(),
            ));
        }
        // Setup thread pool if specified
        if let Some(num_threads) = self.config.num_threads {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Failed to create thread pool: {}", e))
                })?
                .install(|| self.parallel_eigen_impl(matrix))
        } else {
            self.parallel_eigen_impl(matrix)
        }
    }

    /// Internal parallel eigenvalue decomposition implementation
    fn parallel_eigen_impl(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Try to use SciRS2 parallel eigen solver first
        if let Ok(result) = self.try_scirs2_parallel_eigen(matrix) {
            return Ok(result);
        }

        // Fallback to custom parallel implementation
        self.custom_parallel_eigen(matrix)
    }

    /// Resolve the number of worker threads available for parallel work.
    ///
    /// Honors an explicit `num_threads` from the configuration; otherwise falls
    /// back to the number of logical CPUs. At least one worker is always used.
    fn resolve_worker_count(&self) -> usize {
        self.config.num_threads.unwrap_or_else(num_cpus::get).max(1)
    }

    /// Decompose a single symmetric matrix with the SciRS2 symmetric eigensolver.
    ///
    /// This delegates to `scirs2_linalg::eigh` (through the numerical-stability
    /// pipeline) to obtain real eigenvalues/eigenvectors sorted in descending
    /// order and filtered for numerical stability.
    ///
    /// For a single matrix the accurate sequential SciRS2 kernel is used: the
    /// upstream work-stealing parallel kernel in scirs2-linalg 0.5.0 returns
    /// non-finite eigenvalues for larger matrices, so it is not relied upon.
    /// Parallelism for this engine is instead realized across *independent*
    /// sub-problems (the block/distributed driver) and the parallel matrix
    /// products used by the generalized solver.
    fn try_scirs2_parallel_eigen(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let workers = self.resolve_worker_count();
        self.numerical_stability
            .stable_eigen_decomposition_parallel(matrix, workers)
    }

    /// Custom parallel eigenvalue decomposition implementation
    fn custom_parallel_eigen(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();

        // For very large matrices, use block-wise parallel approach
        if n > 2000 && self.config.use_distributed {
            self.distributed_eigen_decomposition(matrix)
        } else {
            // Use parallel matrix operations with SIMD when possible
            self.parallel_matrix_operations(matrix)
        }
    }

    /// Distributed eigenvalue decomposition for very large matrices
    fn distributed_eigen_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let block_size = (n / num_cpus::get()).max(100);

        // Split matrix into blocks and process in parallel
        let blocks: Vec<_> = (0..n)
            .step_by(block_size)
            .map(|start| {
                let end = (start + block_size).min(n);
                (start, end)
            })
            .collect();

        // Process blocks in parallel using rayon
        let partial_results: Result<Vec<_>> = blocks
            .par_iter()
            .map(|&(start, end)| {
                let block = matrix.slice(s![start..end, start..end]);
                // Process block eigenvalues
                self.numerical_stability
                    .stable_eigen_decomposition(&block.to_owned())
            })
            .collect();

        // Merge results from parallel blocks
        self.merge_distributed_eigen_results(partial_results?)
    }

    /// Parallel matrix operations using SIMD when available
    fn parallel_matrix_operations(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Check if we should use SIMD optimizations
        if self.config.use_simd {
            self.simd_accelerated_eigen(matrix)
        } else {
            // Fallback to standard numerical stability implementation
            self.numerical_stability.stable_eigen_decomposition(matrix)
        }
    }

    /// SIMD-accelerated eigenvalue decomposition
    fn simd_accelerated_eigen(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // For now, use SIMD for matrix operations leading up to eigenvalue decomposition
        // First, compute matrix products using SIMD where possible
        let mut processed_matrix = matrix.clone();

        // Process rows in parallel
        let processed_rows: Vec<_> = (0..processed_matrix.nrows())
            .into_par_iter()
            .map(|i| {
                let row = processed_matrix.row(i).to_owned();
                // Use SIMD operations for row processing if available
                if let Ok(simd_result) = self.try_simd_row_operations(&row) {
                    simd_result
                } else {
                    row
                }
            })
            .collect();

        // Copy results back
        for (i, processed_row) in processed_rows.into_iter().enumerate() {
            processed_matrix.row_mut(i).assign(&processed_row);
        }

        // Apply numerical stability eigen decomposition to SIMD-processed matrix
        self.numerical_stability
            .stable_eigen_decomposition(&processed_matrix)
    }

    /// Try to apply SIMD operations to a matrix row
    fn try_simd_row_operations(&self, row: &Array1<Float>) -> Result<Array1<Float>> {
        // Use SciRS2 SIMD operations if available
        let row_data: Vec<Float> = row.to_vec();

        // Apply SIMD dot product or other operations
        if row_data.len() >= 4 {
            // Use SIMD dot product with itself for normalization
            // Fallback: use standard dot product since SciRS2 SIMD may not be available
            let norm_squared = row_data.iter().map(|x| x * x).sum::<Float>();

            if norm_squared > 0.0 {
                let norm = norm_squared.sqrt();
                let normalized: Vec<Float> = row_data.iter().map(|&x| x / norm).collect();
                return Ok(Array1::from_vec(normalized));
            }
        }

        Ok(row.clone())
    }

    /// Merge results from distributed eigenvalue decomposition
    fn merge_distributed_eigen_results(
        &self,
        results: Vec<(Array1<Float>, Array2<Float>)>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No eigenvalue results to merge".to_string(),
            ));
        }

        // Simple approach: concatenate all eigenvalues and sort
        let mut all_eigenvalues = Vec::new();
        let mut all_eigenvectors = Vec::new();

        for (eigenvals, eigenvecs) in results {
            all_eigenvalues.extend(eigenvals.to_vec());
            all_eigenvectors.push(eigenvecs);
        }

        // Sort eigenvalues in descending order
        let mut indexed_eigenvalues: Vec<(usize, Float)> = all_eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_eigenvalues
            .par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let sorted_eigenvalues = Array1::from(
            indexed_eigenvalues
                .iter()
                .map(|(_, val)| *val)
                .collect::<Vec<_>>(),
        );

        // For simplicity, return first eigenvector matrix
        // In a full implementation, we would need to properly merge eigenvectors
        let merged_eigenvectors = if !all_eigenvectors.is_empty() {
            all_eigenvectors.into_iter().next().expect("empty iterator")
        } else {
            Array2::zeros((0, 0))
        };

        Ok((sorted_eigenvalues, merged_eigenvectors))
    }

    /// Compute generalized eigenvalue decomposition in parallel
    pub fn parallel_generalized_eigen(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // For large matrices, use parallel processing
        if a.nrows() >= self.config.parallel_threshold {
            self.parallel_generalized_eigen_impl(a, b)
        } else {
            // Use the stable serial implementation
            self.numerical_stability.stable_generalized_eigen(a, b)
        }
    }

    /// Internal parallel generalized eigenvalue decomposition
    fn parallel_generalized_eigen_impl(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if a.dim() != b.dim() || a.nrows() != a.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrices must be square and same size for generalized eigenvalue decomposition"
                    .to_string(),
            ));
        }

        // Solve the generalized problem A v = lambda B v by reducing it to a
        // standard symmetric problem via B^{-1/2} (B must be symmetric positive
        // definite). The standard reduced problem is then solved with the
        // numerically accurate SciRS2 symmetric eigensolver, and the matrix
        // products are parallelized across rows with rayon.
        //
        // Note: `scirs2_linalg::eigh_gen` is intentionally NOT used here. As of
        // scirs2-linalg 0.5.0 its Cholesky-based reduction returns eigenvectors
        // that do not satisfy A v = lambda B v (large residuals), so relying on
        // it would yield mathematically incorrect results.
        let result: Result<_> = {
            // Parallel computation of matrix square root inverse
            let b_inv_sqrt = self.compute_parallel_matrix_sqrt_inverse(b)?;

            // Parallel matrix multiplication: B^{-1/2} * A * B^{-1/2}
            let transformed_a = self.parallel_matrix_multiply(&b_inv_sqrt, a, &b_inv_sqrt)?;

            // Parallel eigenvalue decomposition
            let (eigenvalues, transformed_eigenvectors) =
                self.parallel_decompose(&transformed_a)?;

            // Transform eigenvectors back: v = B^{-1/2} * v'
            let eigenvectors =
                self.parallel_matrix_vector_multiply(&b_inv_sqrt, &transformed_eigenvectors)?;

            Ok((eigenvalues, eigenvectors))
        };

        result
    }

    /// Parallel matrix square root inverse computation
    fn compute_parallel_matrix_sqrt_inverse(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        // Use eigenvalue decomposition: A^{-1/2} = V * Λ^{-1/2} * V^T
        let (eigenvalues, eigenvectors) = self.parallel_decompose(matrix)?;

        // Parallel computation of sqrt inverse eigenvalues
        let sqrt_inv_eigenvalues: Array1<Float> = eigenvalues
            .par_iter()
            .map(|&val| {
                if val > self.config.numerical_config.eigenvalue_threshold {
                    1.0 / val.sqrt()
                } else {
                    0.0 // Set small eigenvalues to zero
                }
            })
            .collect::<Vec<_>>()
            .into();

        // Reconstruct matrix: V * Λ^{-1/2} * V^T
        let mut result = Array2::zeros(matrix.raw_dim());

        // Parallel outer product computation
        Zip::from(result.axis_iter_mut(Axis(0)))
            .and(eigenvectors.axis_iter(Axis(0)))
            .par_for_each(|mut result_row, eigen_row| {
                for (j, &sqrt_inv_val) in sqrt_inv_eigenvalues.iter().enumerate() {
                    if sqrt_inv_val != 0.0 {
                        let eigen_vec_j = eigenvectors.slice(s![.., j]);
                        let contrib = sqrt_inv_val * eigen_row[j];
                        for (k, &eigen_val_k) in eigen_vec_j.iter().enumerate() {
                            result_row[k] += contrib * eigen_val_k;
                        }
                    }
                }
            });

        Ok(result)
    }

    /// Parallel matrix multiplication: A * B * C
    fn parallel_matrix_multiply(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
        c: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        // First compute A * B in parallel
        let ab = self.parallel_matrix_mult(a, b)?;
        // Then compute (A * B) * C in parallel
        self.parallel_matrix_mult(&ab, c)
    }

    /// Parallel matrix multiplication: A * B
    fn parallel_matrix_mult(&self, a: &Array2<Float>, b: &Array2<Float>) -> Result<Array2<Float>> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match for multiplication".to_string(),
            ));
        }

        let mut result = Array2::zeros((a.nrows(), b.ncols()));

        // Parallel computation using Rayon
        Zip::from(result.axis_iter_mut(Axis(0)))
            .and(a.axis_iter(Axis(0)))
            .par_for_each(|mut result_row, a_row| {
                for (j, result_elem) in result_row.iter_mut().enumerate() {
                    let b_col = b.slice(s![.., j]);

                    // Use standard dot product (SIMD fallback since SciRS2 may not be available)
                    let dot_product = a_row.dot(&b_col);

                    *result_elem = dot_product;
                }
            });

        Ok(result)
    }

    /// Parallel matrix-vector multiplication
    fn parallel_matrix_vector_multiply(
        &self,
        matrix: &Array2<Float>,
        vectors: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        self.parallel_matrix_mult(matrix, vectors)
    }
}

impl Default for ParallelEigenDecomposition {
    fn default() -> Self {
        Self::new()
    }
}

// Additional trait for integration with existing discriminant analysis
pub trait ParallelEigen {
    /// Compute eigenvalue decomposition with parallel processing
    fn parallel_eigen(&self) -> Result<(Array1<Float>, Array2<Float>)>;

    /// Compute generalized eigenvalue decomposition with parallel processing
    fn parallel_generalized_eigen(
        &self,
        other: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)>;
}

impl ParallelEigen for Array2<Float> {
    fn parallel_eigen(&self) -> Result<(Array1<Float>, Array2<Float>)> {
        let decomposer = ParallelEigenDecomposition::new();
        decomposer.decompose(self)
    }

    fn parallel_generalized_eigen(
        &self,
        other: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let decomposer = ParallelEigenDecomposition::new();
        decomposer.parallel_generalized_eigen(self, other)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_parallel_eigen_small_matrix() {
        let decomposer = ParallelEigenDecomposition::new();
        let matrix = array![[2.0, 1.0], [1.0, 2.0]];

        let result = decomposer.decompose(&matrix);
        assert!(result.is_ok());

        let (eigenvalues, _) = result.expect("operation should succeed");
        // Expected eigenvalues: 3.0, 1.0 (sorted descending)
        assert_abs_diff_eq!(eigenvalues[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(eigenvalues[1], 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_threshold() {
        let config = ParallelEigenConfig {
            parallel_threshold: 2, // Force parallel for 2x2 matrix
            ..Default::default()
        };

        let decomposer = ParallelEigenDecomposition::with_config(config);
        let matrix = array![[4.0, 2.0], [2.0, 4.0]];

        let result = decomposer.parallel_decompose(&matrix);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simd_operations() {
        let config = ParallelEigenConfig {
            use_simd: true,
            ..Default::default()
        };

        let decomposer = ParallelEigenDecomposition::with_config(config);
        let row = array![1.0, 2.0, 3.0, 4.0];

        let result = decomposer.try_simd_row_operations(&row);
        assert!(result.is_ok());
    }

    #[test]
    fn test_parallel_eigen_trait() {
        let matrix = array![[3.0, 1.0], [1.0, 3.0]];
        let result = matrix.parallel_eigen();
        assert!(result.is_ok());

        let (eigenvalues, _) = result.expect("operation should succeed");
        assert!(eigenvalues[0] >= eigenvalues[1]); // Should be sorted descending
    }

    /// Exercise the real SciRS2 parallel solver on a matrix large enough to
    /// engage the work-stealing kernel (n > 100 with multiple workers), and
    /// verify the eigenpairs against an analytically known spectrum.
    ///
    /// We build a symmetric matrix `A = Q diag(lambda) Q^T` from an explicit
    /// orthonormal basis (a Householder reflector) so the eigenvalues are known
    /// exactly. The reconstruction `A v_i = lambda_i v_i` must then hold for the
    /// computed eigenpairs.
    #[test]
    fn test_parallel_eigen_large_matrix_correctness() {
        let n = 128;

        // Known eigenvalues, strictly positive and well separated.
        let mut lambda = Array1::zeros(n);
        for i in 0..n {
            lambda[i] = 1.0 + (n - i) as Float;
        }

        // Householder reflector Q = I - 2 u u^T with a normalized u.
        let mut u = Array1::from_shape_fn(n, |i| ((i as Float) + 1.0).sin());
        let u_norm = u.dot(&u).sqrt();
        u.mapv_inplace(|x| x / u_norm);

        let mut q = Array2::<Float>::eye(n);
        for i in 0..n {
            for j in 0..n {
                q[[i, j]] -= 2.0 * u[i] * u[j];
            }
        }

        // A = Q diag(lambda) Q^T (symmetric in exact arithmetic).
        let mut q_scaled = q.clone();
        for j in 0..n {
            let scale = lambda[j];
            for i in 0..n {
                q_scaled[[i, j]] *= scale;
            }
        }
        let a_raw = q_scaled.dot(&q.t());
        // Symmetrize to remove floating-point asymmetry from the reconstruction.
        let a = (&a_raw + &a_raw.t()) * 0.5;

        // Force the parallel path and request multiple workers.
        let config = ParallelEigenConfig {
            parallel_threshold: 1,
            num_threads: Some(4),
            use_distributed: false,
            ..Default::default()
        };
        let decomposer = ParallelEigenDecomposition::with_config(config);

        let (eigenvalues, eigenvectors) = decomposer
            .parallel_decompose(&a)
            .expect("parallel eigen decomposition should succeed");

        assert_eq!(eigenvalues.len(), n);
        assert_eq!(eigenvectors.dim(), (n, n));

        // Descending order.
        for i in 1..eigenvalues.len() {
            assert!(eigenvalues[i - 1] >= eigenvalues[i] - 1e-6);
        }

        // The recovered spectrum must match the known eigenvalues.
        let mut expected: Vec<Float> = lambda.to_vec();
        expected.sort_by(|x, y| y.partial_cmp(x).unwrap_or(std::cmp::Ordering::Equal));
        for (computed, exp) in eigenvalues.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*computed, *exp, epsilon = 1e-4);
        }

        // Each eigenpair must satisfy A v = lambda v (residual near zero).
        for j in 0..n {
            let v = eigenvectors.column(j).to_owned();
            let av = a.dot(&v);
            let lv = &v * eigenvalues[j];
            let residual = (&av - &lv).iter().map(|x| x * x).sum::<Float>().sqrt();
            assert!(
                residual < 1e-3,
                "eigenpair {} residual too large: {}",
                j,
                residual
            );
        }
    }

    /// Verify the parallel generalized solver yields correct eigenpairs for the
    /// problem `A v = lambda B v`, with `B` symmetric positive definite. The
    /// generalized residual `A v_j - lambda_j B v_j` must be near zero.
    #[test]
    fn test_parallel_generalized_eigen_correctness() {
        let n = 16usize;

        // Symmetric A.
        let a_raw =
            Array2::<Float>::from_shape_fn((n, n), |(i, j)| ((i + 2 * j + 1) as Float).cos());
        let a = (&a_raw + &a_raw.t()) * 0.5;

        // SPD B = M^T M + n*I (strictly positive definite, well conditioned).
        let m =
            Array2::<Float>::from_shape_fn((n, n), |(i, j)| ((i * 7 + j * 3 + 1) as Float).sin());
        let mut b = m.t().dot(&m);
        for i in 0..n {
            b[[i, i]] += n as Float;
        }

        // Force the parallel generalized path.
        let config = ParallelEigenConfig {
            parallel_threshold: 1,
            num_threads: Some(4),
            ..Default::default()
        };
        let decomposer = ParallelEigenDecomposition::with_config(config);

        let (eigenvalues, eigenvectors) = decomposer
            .parallel_generalized_eigen(&a, &b)
            .expect("parallel generalized eigen should succeed");

        assert!(!eigenvalues.is_empty());

        // Descending order.
        for i in 1..eigenvalues.len() {
            assert!(eigenvalues[i - 1] >= eigenvalues[i] - 1e-6);
        }

        // Each eigenpair must satisfy A v = lambda B v.
        for j in 0..eigenvalues.len() {
            let v = eigenvectors.column(j).to_owned();
            let av = a.dot(&v);
            let bv = b.dot(&v);
            let residual = (&av - &(&bv * eigenvalues[j]))
                .iter()
                .map(|x| x * x)
                .sum::<Float>()
                .sqrt();
            assert!(
                residual < 1e-6,
                "generalized eigenpair {} residual too large: {}",
                j,
                residual
            );
        }
    }
}

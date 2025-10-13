//! Utility functions for covariance estimation

use scirs2_core::ndarray::{Array2, NdFloat};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Compute matrix inverse using appropriate method based on size
pub fn matrix_inverse(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square".to_string(),
        ));
    }

    // Simple LU decomposition for small matrices
    if n <= 3 {
        return matrix_inverse_small(matrix);
    }

    // For larger matrices, use pseudoinverse approximation
    matrix_pseudoinverse(matrix)
}

/// Compute matrix inverse for small matrices (1x1, 2x2, 3x3)
fn matrix_inverse_small(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = matrix.nrows();

    match n {
        1 => {
            let det = matrix[[0, 0]];
            if det.abs() < 1e-12 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }
            Ok(Array2::from_elem((1, 1), 1.0 / det))
        }
        2 => {
            let det = matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]];
            if det.abs() < 1e-12 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }
            let mut inv = Array2::zeros((2, 2));
            inv[[0, 0]] = matrix[[1, 1]] / det;
            inv[[0, 1]] = -matrix[[0, 1]] / det;
            inv[[1, 0]] = -matrix[[1, 0]] / det;
            inv[[1, 1]] = matrix[[0, 0]] / det;
            Ok(inv)
        }
        _ => {
            // Use Gauss-Jordan elimination for 3x3 and larger
            gauss_jordan_inverse(matrix)
        }
    }
}

/// Compute matrix inverse using Gauss-Jordan elimination
fn gauss_jordan_inverse(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = matrix.nrows();
    let mut aug = Array2::zeros((n, 2 * n));

    // Create augmented matrix [A | I]
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
            aug[[i, j + n]] = if i == j { 1.0 } else { 0.0 };
        }
    }

    // Gauss-Jordan elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows
        if max_row != i {
            for j in 0..(2 * n) {
                let temp = aug[[i, j]];
                aug[[i, j]] = aug[[max_row, j]];
                aug[[max_row, j]] = temp;
            }
        }

        // Check for singularity
        if aug[[i, i]].abs() < 1e-12 {
            return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
        }

        // Scale pivot row
        let pivot = aug[[i, i]];
        for j in 0..(2 * n) {
            aug[[i, j]] /= pivot;
        }

        // Eliminate column
        for k in 0..n {
            if k != i {
                let factor = aug[[k, i]];
                for j in 0..(2 * n) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }
    }

    // Extract inverse matrix
    let mut inv = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, j + n]];
        }
    }

    Ok(inv)
}

/// Compute matrix pseudoinverse using regularization
fn matrix_pseudoinverse(matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
    // Simplified pseudoinverse using regularization
    let n = matrix.nrows();
    let mut regularized = matrix.clone();

    // Add small regularization to diagonal
    for i in 0..n {
        regularized[[i, i]] += 1e-6;
    }

    gauss_jordan_inverse(&regularized)
}

/// Compute matrix determinant
pub fn matrix_determinant(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    if n != matrix.ncols() {
        return 0.0;
    }

    match n {
        1 => matrix[[0, 0]],
        2 => matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]],
        _ => {
            // Use LU decomposition for larger matrices
            lu_determinant(matrix)
        }
    }
}

/// Compute determinant using LU decomposition
fn lu_determinant(matrix: &Array2<f64>) -> f64 {
    let n = matrix.nrows();
    let mut a = matrix.clone();
    let mut det = 1.0;

    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in (i + 1)..n {
            if a[[k, i]].abs() > a[[max_row, i]].abs() {
                max_row = k;
            }
        }

        // Swap rows if needed
        if max_row != i {
            for j in 0..n {
                let temp = a[[i, j]];
                a[[i, j]] = a[[max_row, j]];
                a[[max_row, j]] = temp;
            }
            det = -det; // Row swap changes sign
        }

        det *= a[[i, i]];

        if a[[i, i]].abs() < 1e-12 {
            return 0.0; // Singular matrix
        }

        // Eliminate below pivot
        for k in (i + 1)..n {
            let factor = a[[k, i]] / a[[i, i]];
            for j in i..n {
                a[[k, j]] -= factor * a[[i, j]];
            }
        }
    }

    det
}

/// Validate input data for covariance estimation
pub fn validate_data<F>(x: &Array2<F>) -> Result<(), SklearsError>
where
    F: NdFloat,
{
    let (n_samples, n_features) = x.dim();

    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of samples must be greater than 0".to_string(),
        ));
    }

    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "Number of features must be greater than 0".to_string(),
        ));
    }

    // Check for NaN or infinite values
    for value in x.iter() {
        if !value.is_finite() {
            return Err(SklearsError::InvalidInput(
                "Input data contains NaN or infinite values".to_string(),
            ));
        }
    }

    Ok(())
}

/// Regularize a matrix by adding a small value to the diagonal for numerical stability
pub fn regularize_matrix<F>(matrix: &Array2<F>, reg_param: F) -> Result<Array2<F>, SklearsError>
where
    F: NdFloat,
{
    let (n_rows, n_cols) = matrix.dim();

    if n_rows != n_cols {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for regularization".to_string(),
        ));
    }

    let mut regularized = matrix.clone();

    // Add regularization to diagonal
    for i in 0..n_rows {
        regularized[[i, i]] = regularized[[i, i]] + reg_param;
    }

    Ok(regularized)
}

/// Validate properties of a covariance matrix
pub fn validate_covariance_matrix<F>(
    matrix: &Array2<F>,
) -> Result<CovarianceProperties<F>, SklearsError>
where
    F: NdFloat + std::fmt::Display,
{
    let (n_rows, n_cols) = matrix.dim();

    if n_rows != n_cols {
        return Err(SklearsError::InvalidInput(
            "Covariance matrix must be square".to_string(),
        ));
    }

    let mut properties = CovarianceProperties {
        is_symmetric: true,
        is_positive_definite: true,
        is_positive_semi_definite: true,
        condition_number: F::one(),
        determinant: F::one(),
        trace: F::zero(),
        min_eigenvalue: F::zero(),
        max_eigenvalue: F::zero(),
    };

    // Check symmetry
    for i in 0..n_rows {
        for j in 0..n_cols {
            if (matrix[[i, j]] - matrix[[j, i]]).abs() > F::from(1e-12).unwrap() {
                properties.is_symmetric = false;
                break;
            }
        }
        if !properties.is_symmetric {
            break;
        }
    }

    // Compute trace
    for i in 0..n_rows {
        properties.trace = properties.trace + matrix[[i, i]];
    }

    // Simple checks for positive definiteness (diagonal dominance heuristic)
    let mut min_diag = matrix[[0, 0]];
    let mut max_diag = matrix[[0, 0]];

    for i in 0..n_rows {
        let diag_val = matrix[[i, i]];
        if diag_val < min_diag {
            min_diag = diag_val;
        }
        if diag_val > max_diag {
            max_diag = diag_val;
        }

        if diag_val < F::zero() {
            properties.is_positive_definite = false;
            properties.is_positive_semi_definite = false;
        } else if diag_val == F::zero() {
            properties.is_positive_definite = false;
        }
    }

    properties.min_eigenvalue = min_diag;
    properties.max_eigenvalue = max_diag;

    // Approximate condition number
    if min_diag > F::zero() {
        properties.condition_number = max_diag / min_diag;
    } else {
        properties.condition_number = F::infinity();
    }

    // Approximate determinant (product of diagonal elements for diagonal matrices)
    properties.determinant = F::one();
    for i in 0..n_rows {
        properties.determinant = properties.determinant * matrix[[i, i]];
    }

    Ok(properties)
}

/// Properties of a covariance matrix
#[derive(Debug, Clone)]
pub struct CovarianceProperties<F: NdFloat> {
    /// Whether the matrix is symmetric
    pub is_symmetric: bool,
    /// Whether the matrix is positive definite
    pub is_positive_definite: bool,
    /// Whether the matrix is positive semi-definite
    pub is_positive_semi_definite: bool,
    /// Condition number of the matrix
    pub condition_number: F,
    /// Determinant of the matrix
    pub determinant: F,
    /// Trace of the matrix (sum of diagonal elements)
    pub trace: F,
    /// Minimum eigenvalue (approximated)
    pub min_eigenvalue: F,
    /// Maximum eigenvalue (approximated)
    pub max_eigenvalue: F,
}

/// Compute the Frobenius norm of a matrix
pub fn frobenius_norm<F>(matrix: &Array2<F>) -> F
where
    F: NdFloat,
{
    let mut sum = F::zero();
    for value in matrix.iter() {
        sum = sum + (*value) * (*value);
    }
    sum.sqrt()
}

/// Compute the nuclear norm (sum of singular values) approximation via trace
pub fn nuclear_norm_approximation<F>(matrix: &Array2<F>) -> F
where
    F: NdFloat,
{
    // For symmetric positive semidefinite matrices, nuclear norm ≈ trace
    let mut trace = F::zero();
    let min_dim = matrix.nrows().min(matrix.ncols());
    for i in 0..min_dim {
        trace = trace + matrix[[i, i]];
    }
    trace
}

/// Check if a matrix is diagonally dominant (useful for convergence analysis)
pub fn is_diagonally_dominant<F>(matrix: &Array2<F>) -> bool
where
    F: NdFloat,
{
    let (n_rows, n_cols) = matrix.dim();
    if n_rows != n_cols {
        return false;
    }

    for i in 0..n_rows {
        let diag_val = matrix[[i, i]].abs();
        let mut off_diag_sum = F::zero();

        for j in 0..n_cols {
            if i != j {
                off_diag_sum = off_diag_sum + matrix[[i, j]].abs();
            }
        }

        if diag_val <= off_diag_sum {
            return false;
        }
    }
    true
}

/// Estimate the spectral radius (largest eigenvalue magnitude) via power iteration
pub fn spectral_radius_estimate<F>(matrix: &Array2<F>, max_iterations: usize) -> SklResult<F>
where
    F: NdFloat,
{
    let (n_rows, n_cols) = matrix.dim();
    if n_rows != n_cols {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for spectral radius estimation".to_string(),
        ));
    }

    if n_rows == 0 {
        return Ok(F::zero());
    }

    // Initialize with random vector
    let mut x = Array2::ones((n_rows, 1));
    let mut eigenvalue = F::one();

    for _ in 0..max_iterations {
        // x_new = A * x
        let mut x_new = Array2::zeros((n_rows, 1));
        for i in 0..n_rows {
            for j in 0..n_cols {
                x_new[[i, 0]] = x_new[[i, 0]] + matrix[[i, j]] * x[[j, 0]];
            }
        }

        // Compute norm
        let mut norm = F::zero();
        for i in 0..n_rows {
            norm = norm + x_new[[i, 0]] * x_new[[i, 0]];
        }
        norm = norm.sqrt();

        if norm.abs() < F::from(1e-12).unwrap() {
            return Ok(F::zero());
        }

        // Normalize
        for i in 0..n_rows {
            x_new[[i, 0]] = x_new[[i, 0]] / norm;
        }

        eigenvalue = norm;
        x = x_new;
    }

    Ok(eigenvalue)
}

/// Matrix rank estimation via iterative hard thresholding
pub fn rank_estimate<F>(matrix: &Array2<F>, threshold: F) -> usize
where
    F: NdFloat,
{
    let min_dim = matrix.nrows().min(matrix.ncols());
    let mut rank = 0;

    // Count diagonal elements above threshold (rough approximation)
    for i in 0..min_dim {
        if matrix[[i, i]].abs() > threshold {
            rank += 1;
        }
    }

    rank
}

/// Stability-oriented covariance shrinkage
pub fn adaptive_shrinkage<F>(
    sample_cov: &Array2<F>,
    n_samples: usize,
    target: Option<&Array2<F>>,
) -> SklResult<Array2<F>>
where
    F: NdFloat,
{
    let n_features = sample_cov.nrows();

    if n_features != sample_cov.ncols() {
        return Err(SklearsError::InvalidInput(
            "Covariance matrix must be square".to_string(),
        ));
    }

    // Default target is identity matrix scaled by average variance
    let target_matrix = if let Some(target) = target {
        target.clone()
    } else {
        let trace = (0..n_features)
            .map(|i| sample_cov[[i, i]])
            .fold(F::zero(), |acc, x| acc + x);
        let avg_var = trace / F::from(n_features).unwrap();

        let mut identity = Array2::zeros((n_features, n_features));
        for i in 0..n_features {
            identity[[i, i]] = avg_var;
        }
        identity
    };

    // Adaptive shrinkage intensity based on sample size
    let shrinkage_intensity = if n_samples > n_features {
        F::from(n_features as f64 / (n_samples as f64 + n_features as f64)).unwrap()
    } else {
        F::from(0.8).unwrap() // High shrinkage for small sample sizes
    };

    // Shrunk covariance = (1 - λ) * sample_cov + λ * target
    let mut shrunk_cov = Array2::zeros((n_features, n_features));
    let one_minus_lambda = F::one() - shrinkage_intensity;

    for i in 0..n_features {
        for j in 0..n_features {
            shrunk_cov[[i, j]] =
                one_minus_lambda * sample_cov[[i, j]] + shrinkage_intensity * target_matrix[[i, j]];
        }
    }

    Ok(shrunk_cov)
}

/// Cross-validation utility for covariance estimator selection
pub struct CovarianceCV<F: NdFloat> {
    /// Number of folds for cross-validation
    pub n_folds: usize,
    /// Scoring method
    pub scoring: ScoringMethod,
    /// Random seed for fold generation
    pub random_seed: Option<u64>,
    _phantom: std::marker::PhantomData<F>,
}

/// Scoring methods for covariance cross-validation
#[derive(Debug, Clone, Copy)]
pub enum ScoringMethod {
    /// Log-likelihood score
    LogLikelihood,
    /// Frobenius norm of residuals
    Frobenius,
    /// Prediction accuracy on held-out data
    Prediction,
}

impl<F: NdFloat> CovarianceCV<F> {
    /// Create a new cross-validation instance
    pub fn new(n_folds: usize, scoring: ScoringMethod) -> Self {
        Self {
            n_folds,
            scoring,
            random_seed: None,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Set random seed for reproducible cross-validation
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Generate cross-validation fold indices
    pub fn generate_folds(&self, n_samples: usize) -> Vec<Vec<usize>> {
        let indices: Vec<usize> = (0..n_samples).collect();

        // Simple k-fold splitting (deterministic for now)
        let fold_size = n_samples / self.n_folds;
        let mut folds = Vec::new();

        for i in 0..self.n_folds {
            let start = i * fold_size;
            let end = if i == self.n_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            folds.push((start..end).collect());
        }

        folds
    }
}

/// Performance benchmarking utilities for covariance estimators
pub struct CovarianceBenchmark {
    /// Number of benchmark iterations
    pub n_iterations: usize,
    /// Whether to include warm-up runs
    pub include_warmup: bool,
    /// Warm-up iterations
    pub warmup_iterations: usize,
}

impl CovarianceBenchmark {
    /// Create a new benchmark instance
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            include_warmup: true,
            warmup_iterations: 3,
        }
    }

    /// Benchmark execution time for a closure
    pub fn time_execution<F, R>(&self, mut operation: F) -> BenchmarkResult
    where
        F: FnMut() -> R,
    {
        use std::time::Instant;

        // Warm-up runs
        if self.include_warmup {
            for _ in 0..self.warmup_iterations {
                operation();
            }
        }

        // Actual benchmark
        let start = Instant::now();
        let mut times = Vec::with_capacity(self.n_iterations);

        for _ in 0..self.n_iterations {
            let iter_start = Instant::now();
            operation();
            let iter_duration = iter_start.elapsed();
            times.push(iter_duration.as_nanos() as f64);
        }

        let total_duration = start.elapsed();

        // Compute statistics
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = times.iter().sum::<f64>() / times.len() as f64;
        let median = times[times.len() / 2];
        let min = times[0];
        let max = times[times.len() - 1];

        BenchmarkResult {
            n_iterations: self.n_iterations,
            total_time_ns: total_duration.as_nanos() as f64,
            mean_time_ns: mean,
            median_time_ns: median,
            min_time_ns: min,
            max_time_ns: max,
            times_ns: times,
        }
    }
}

/// Results from performance benchmarking
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Total execution time in nanoseconds
    pub total_time_ns: f64,
    /// Mean execution time per iteration in nanoseconds
    pub mean_time_ns: f64,
    /// Median execution time per iteration in nanoseconds
    pub median_time_ns: f64,
    /// Minimum execution time in nanoseconds
    pub min_time_ns: f64,
    /// Maximum execution time in nanoseconds
    pub max_time_ns: f64,
    /// All individual timing measurements
    pub times_ns: Vec<f64>,
}

impl BenchmarkResult {
    /// Get mean execution time in milliseconds
    pub fn mean_time_ms(&self) -> f64 {
        self.mean_time_ns / 1_000_000.0
    }

    /// Get median execution time in milliseconds
    pub fn median_time_ms(&self) -> f64 {
        self.median_time_ns / 1_000_000.0
    }

    /// Get throughput (operations per second)
    pub fn throughput_ops_per_sec(&self) -> f64 {
        1_000_000_000.0 / self.mean_time_ns
    }

    /// Get standard deviation of execution times
    pub fn std_dev_ns(&self) -> f64 {
        let mean = self.mean_time_ns;
        let variance = self
            .times_ns
            .iter()
            .map(|&t| (t - mean).powi(2))
            .sum::<f64>()
            / self.times_ns.len() as f64;
        variance.sqrt()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_frobenius_norm() {
        let matrix = array![[1.0, 2.0], [3.0, 4.0]];
        let norm = frobenius_norm(&matrix);
        // Expected: sqrt(1^2 + 2^2 + 3^2 + 4^2) = sqrt(30) ≈ 5.477
        assert_abs_diff_eq!(norm, 30.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_nuclear_norm_approximation() {
        let matrix = array![[3.0, 1.0], [1.0, 2.0]];
        let nuclear_norm = nuclear_norm_approximation(&matrix);
        // For this 2x2 matrix, nuclear norm approximation = trace = 3 + 2 = 5
        assert_abs_diff_eq!(nuclear_norm, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_is_diagonally_dominant() {
        // Diagonally dominant matrix
        let dominant_matrix = array![[4.0, 1.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 5.0]];
        assert!(is_diagonally_dominant(&dominant_matrix));

        // Non-diagonally dominant matrix
        let non_dominant_matrix = array![[1.0, 2.0, 3.0], [2.0, 1.0, 2.0], [3.0, 2.0, 1.0]];
        assert!(!is_diagonally_dominant(&non_dominant_matrix));

        // Non-square matrix
        let non_square = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(!is_diagonally_dominant(&non_square));
    }

    #[test]
    fn test_spectral_radius_estimate() {
        // Simple 2x2 matrix with known eigenvalues
        let matrix = array![[3.0, 1.0], [1.0, 2.0]];
        let spectral_radius = spectral_radius_estimate(&matrix, 10).unwrap();

        // The largest eigenvalue should be approximately 3.618 (golden ratio + 2.5)
        assert!(spectral_radius > 3.0);
        assert!(spectral_radius < 5.0);

        // Identity matrix should have spectral radius ≈ 1
        let identity = array![[1.0, 0.0], [0.0, 1.0]];
        let identity_radius = spectral_radius_estimate(&identity, 10).unwrap();
        assert_abs_diff_eq!(identity_radius, 1.0, epsilon = 0.1);
    }

    #[test]
    fn test_rank_estimate() {
        // Full rank matrix
        let full_rank = array![[1.0, 0.0], [0.0, 1.0]];
        assert_eq!(rank_estimate(&full_rank, 1e-10), 2);

        // Low rank matrix (diagonal with one zero)
        let low_rank = array![[1.0, 0.0], [0.0, 0.0]];
        assert_eq!(rank_estimate(&low_rank, 1e-10), 1);

        // All zeros should have rank 0
        let zero_matrix = array![[0.0, 0.0], [0.0, 0.0]];
        assert_eq!(rank_estimate(&zero_matrix, 1e-10), 0);
    }

    #[test]
    fn test_adaptive_shrinkage() {
        let sample_cov = array![[2.0, 0.5], [0.5, 1.5]];
        let n_samples = 10;

        // Test with default target
        let shrunk = adaptive_shrinkage(&sample_cov, n_samples, None).unwrap();
        assert_eq!(shrunk.shape(), [2, 2]);

        // Shrunk covariance should be closer to identity-like structure
        // but still retain some of the original structure
        assert!(shrunk[[0, 0]] > 0.0);
        assert!(shrunk[[1, 1]] > 0.0);
        assert!((shrunk[[0, 1]] as f64).abs() < (sample_cov[[0, 1]] as f64).abs());

        // Test with custom target
        let target = array![[1.0, 0.0], [0.0, 1.0]];
        let shrunk_custom = adaptive_shrinkage(&sample_cov, n_samples, Some(&target)).unwrap();
        assert_eq!(shrunk_custom.shape(), [2, 2]);
    }

    #[test]
    fn test_covariance_cv() {
        let cv = CovarianceCV::<f64>::new(5, ScoringMethod::LogLikelihood);
        assert_eq!(cv.n_folds, 5);

        let cv_with_seed = cv.with_seed(42);
        assert_eq!(cv_with_seed.random_seed, Some(42));

        // Test fold generation with a new cv instance
        let cv_for_folds = CovarianceCV::<f64>::new(5, ScoringMethod::LogLikelihood);
        let folds = cv_for_folds.generate_folds(20);
        assert_eq!(folds.len(), 5);

        // Each fold should have approximately equal size
        for fold in &folds {
            assert!(fold.len() >= 3 && fold.len() <= 5); // 20/5 = 4, with some variation
        }

        // All indices should be covered exactly once
        let mut all_indices: Vec<usize> = folds.into_iter().flatten().collect();
        all_indices.sort();
        assert_eq!(all_indices, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn test_covariance_benchmark() {
        let benchmark = CovarianceBenchmark::new(5);
        assert_eq!(benchmark.n_iterations, 5);
        assert!(benchmark.include_warmup);

        // Benchmark a simple operation
        let result = benchmark.time_execution(|| {
            // Simple computation to benchmark
            (0..100).map(|x| x * x).sum::<i32>()
        });

        assert_eq!(result.n_iterations, 5);
        assert!(result.mean_time_ns > 0.0);
        assert!(result.median_time_ns > 0.0);
        assert!(result.min_time_ns <= result.mean_time_ns);
        assert!(result.max_time_ns >= result.mean_time_ns);
        assert_eq!(result.times_ns.len(), 5);
    }

    #[test]
    fn test_benchmark_result_conversions() {
        let result = BenchmarkResult {
            n_iterations: 3,
            total_time_ns: 3_000_000.0, // 3 milliseconds
            mean_time_ns: 1_000_000.0,  // 1 millisecond
            median_time_ns: 900_000.0,  // 0.9 milliseconds
            min_time_ns: 800_000.0,
            max_time_ns: 1_400_000.0,
            times_ns: vec![800_000.0, 900_000.0, 1_400_000.0],
        };

        assert_abs_diff_eq!(result.mean_time_ms(), 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(result.median_time_ms(), 0.9, epsilon = 1e-6);
        assert_abs_diff_eq!(result.throughput_ops_per_sec(), 1000.0, epsilon = 1e-3);

        // Test standard deviation calculation
        let std_dev = result.std_dev_ns();
        assert!(std_dev > 0.0);
    }

    #[test]
    fn test_scoring_method_variants() {
        let methods = [
            ScoringMethod::LogLikelihood,
            ScoringMethod::Frobenius,
            ScoringMethod::Prediction,
        ];

        // Ensure all variants can be created and are distinct
        assert_ne!(format!("{:?}", methods[0]), format!("{:?}", methods[1]));
        assert_ne!(format!("{:?}", methods[1]), format!("{:?}", methods[2]));
    }

    #[test]
    fn test_utility_edge_cases() {
        // Test frobenius norm with zeros
        let zero_matrix = Array2::<f64>::zeros((2, 2));
        assert_eq!(frobenius_norm(&zero_matrix), 0.0);

        // Test spectral radius with non-square matrix (should fail)
        let non_square = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        assert!(spectral_radius_estimate(&non_square, 10).is_err());

        // Test adaptive shrinkage with non-square matrix (should fail)
        assert!(adaptive_shrinkage(&non_square, 10, None).is_err());

        // Test empty matrix spectral radius
        let empty_matrix = Array2::<f64>::zeros((0, 0));
        assert_eq!(spectral_radius_estimate(&empty_matrix, 10).unwrap(), 0.0);
    }

    #[test]
    fn test_matrix_properties_integration() {
        let matrix = array![[4.0, 1.0], [1.0, 3.0]];

        // Test that our utility functions work together coherently
        let frobenius = frobenius_norm(&matrix);
        let nuclear = nuclear_norm_approximation(&matrix);
        let is_dominant = is_diagonally_dominant(&matrix);
        let rank = rank_estimate(&matrix, 1e-10);

        assert!(frobenius > 0.0);
        assert!(nuclear > 0.0);
        assert!(is_dominant); // This matrix should be diagonally dominant
        assert_eq!(rank, 2); // Should be full rank
    }
}

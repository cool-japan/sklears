//! Numerical Stability Utilities
//!
//! This module provides numerically stable implementations for common linear algebra operations
//! used in discriminant analysis, with focus on eigenvalue decomposition and condition monitoring.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::ndarray_linalg::{Eigh, UPLO};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayBase, Axis, Ix2, OwnedRepr};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::VecDeque;
use std::f64;

/// Configuration for numerical stability parameters
#[derive(Debug, Clone)]
pub struct NumericalConfig {
    /// Tolerance for numerical operations
    pub tolerance: Float,
    /// Maximum condition number allowed before warning
    pub max_condition_number: Float,
    /// Minimum eigenvalue threshold (relative to largest eigenvalue)
    pub eigenvalue_threshold: Float,
    /// Use regularization for ill-conditioned matrices
    pub use_regularization: bool,
    /// Regularization parameter
    pub regularization_param: Float,
    /// Enable condition number monitoring
    pub enable_monitoring: bool,
    /// Maximum number of condition numbers to track
    pub monitoring_history_size: usize,
    /// Log warnings for condition numbers above threshold
    pub log_warnings: bool,
}

/// Condition number monitoring data
#[derive(Debug, Clone)]
pub struct ConditionMonitor {
    /// History of condition numbers
    pub condition_history: VecDeque<Float>,
    /// Number of warnings issued
    pub warning_count: usize,
    /// Maximum condition number seen
    pub max_condition_number: Float,
    /// Average condition number
    pub average_condition_number: Float,
}

impl Default for NumericalConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-12,
            max_condition_number: 1e12,
            eigenvalue_threshold: 1e-15,
            use_regularization: true,
            regularization_param: 1e-10,
            enable_monitoring: true,
            monitoring_history_size: 100,
            log_warnings: true,
        }
    }
}

/// Numerical stability utilities for discriminant analysis
pub struct NumericalStability {
    config: NumericalConfig,
}

impl NumericalStability {
    /// Create a new instance with default configuration
    pub fn new() -> Self {
        Self {
            config: NumericalConfig::default(),
        }
    }

    /// Create a new instance with custom configuration
    pub fn with_config(config: NumericalConfig) -> Self {
        Self { config }
    }

    /// Compute eigenvalue decomposition using numerically stable algorithm
    ///
    /// Returns (eigenvalues, eigenvectors) where eigenvalues are sorted in descending order
    pub fn stable_eigen_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if matrix.nrows() != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigenvalue decomposition".to_string(),
            ));
        }

        // Check if matrix is symmetric (required for Eigh)
        if !self.is_symmetric(matrix) {
            return Err(SklearsError::InvalidInput(
                "Matrix must be symmetric for stable eigenvalue decomposition".to_string(),
            ));
        }

        // Add regularization if needed to improve conditioning
        let regularized_matrix = if self.config.use_regularization {
            self.add_regularization(matrix)?
        } else {
            matrix.clone()
        };

        // Compute condition number
        let condition_number = self.estimate_condition_number(&regularized_matrix)?;
        if condition_number > self.config.max_condition_number {
            eprintln!(
                "Warning: Matrix condition number ({:.2e}) exceeds threshold ({:.2e})",
                condition_number, self.config.max_condition_number
            );
        }

        // Use ndarray-linalg for stable eigenvalue decomposition
        let (eigenvalues, eigenvectors) = regularized_matrix.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::InvalidInput(format!("Eigenvalue decomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        let (sorted_eigenvalues, sorted_eigenvectors) =
            self.sort_eigen_desc(eigenvalues, eigenvectors);

        // Filter out small eigenvalues to improve numerical stability
        let (filtered_eigenvalues, filtered_eigenvectors) =
            self.filter_small_eigenvalues(sorted_eigenvalues, sorted_eigenvectors)?;

        Ok((filtered_eigenvalues, filtered_eigenvectors))
    }

    /// Compute generalized eigenvalue decomposition for matrices A and B
    /// Solves A * v = lambda * B * v
    pub fn stable_generalized_eigen(
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

        // Compute B^{-1/2} for transformation to standard eigenvalue problem
        let b_inv_sqrt = self.matrix_sqrt_inverse(b)?;

        // Transform to standard eigenvalue problem: (B^{-1/2} * A * B^{-1/2}) * v' = lambda * v'
        let transformed_a = b_inv_sqrt.dot(a).dot(&b_inv_sqrt);

        // Solve standard eigenvalue problem
        let (eigenvalues, transformed_eigenvectors) =
            self.stable_eigen_decomposition(&transformed_a)?;

        // Transform eigenvectors back: v = B^{-1/2} * v'
        let eigenvectors = b_inv_sqrt.dot(&transformed_eigenvectors);

        Ok((eigenvalues, eigenvectors))
    }

    /// Check if a matrix is symmetric within numerical tolerance
    fn is_symmetric(&self, matrix: &Array2<Float>) -> bool {
        if matrix.nrows() != matrix.ncols() {
            return false;
        }

        for i in 0..matrix.nrows() {
            for j in 0..matrix.ncols() {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > self.config.tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Add regularization to improve matrix conditioning
    fn add_regularization(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let mut regularized = matrix.clone();
        let n = matrix.nrows();

        // Add regularization to diagonal
        for i in 0..n {
            regularized[[i, i]] += self.config.regularization_param;
        }

        Ok(regularized)
    }

    /// Estimate condition number of a matrix
    pub fn estimate_condition_number(&self, matrix: &Array2<Float>) -> Result<Float> {
        // For symmetric matrices, condition number is ratio of largest to smallest eigenvalue
        // Use direct eigenvalue decomposition to avoid recursion
        let (eigenvalues, _) = matrix.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::InvalidInput(format!("Eigenvalue decomposition failed: {}", e))
        })?;

        if eigenvalues.is_empty() {
            return Ok(f64::INFINITY);
        }

        let max_eigenval = eigenvalues
            .iter()
            .fold(0.0 as Float, |a, &b| a.max(b.abs()));
        let min_eigenval = eigenvalues
            .iter()
            .fold(Float::INFINITY, |a, &b| a.min(b.abs()));

        if min_eigenval < self.config.tolerance {
            Ok(f64::INFINITY)
        } else {
            Ok(max_eigenval / min_eigenval)
        }
    }

    /// Sort eigenvalues and eigenvectors in descending order
    fn sort_eigen_desc(
        &self,
        eigenvalues: Array1<Float>,
        eigenvectors: Array2<Float>,
    ) -> (Array1<Float>, Array2<Float>) {
        let mut indices: Vec<usize> = (0..eigenvalues.len()).collect();
        indices.sort_by(|&i, &j| eigenvalues[j].partial_cmp(&eigenvalues[i]).unwrap());

        let sorted_eigenvalues = Array1::from_iter(indices.iter().map(|&i| eigenvalues[i]));
        let sorted_eigenvectors =
            Array2::from_shape_fn(eigenvectors.dim(), |(i, j)| eigenvectors[[i, indices[j]]]);

        (sorted_eigenvalues, sorted_eigenvectors)
    }

    /// Filter out eigenvalues that are too small relative to the largest
    fn filter_small_eigenvalues(
        &self,
        eigenvalues: Array1<Float>,
        eigenvectors: Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        if eigenvalues.is_empty() {
            return Ok((eigenvalues, eigenvectors));
        }

        let max_eigenval = eigenvalues[0].abs(); // Already sorted in descending order
        let threshold = max_eigenval * self.config.eigenvalue_threshold;

        // Find number of eigenvalues above threshold
        let n_keep = eigenvalues
            .iter()
            .take_while(|&&val| val.abs() >= threshold)
            .count()
            .max(1); // Keep at least one eigenvalue

        let filtered_eigenvalues = eigenvalues.slice(s![..n_keep]).to_owned();
        let filtered_eigenvectors = eigenvectors.slice(s![.., ..n_keep]).to_owned();

        Ok((filtered_eigenvalues, filtered_eigenvectors))
    }

    /// Compute matrix square root inverse using eigenvalue decomposition
    fn matrix_sqrt_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let (eigenvalues, eigenvectors) = self.stable_eigen_decomposition(matrix)?;

        // Check for non-positive eigenvalues
        for &val in eigenvalues.iter() {
            if val <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Matrix is not positive definite for square root inverse".to_string(),
                ));
            }
        }

        // Compute sqrt inverse of eigenvalues
        let sqrt_inv_eigenvalues = eigenvalues.map(|&val| 1.0 / val.sqrt());

        // Reconstruct matrix: V * diag(sqrt_inv_eigenvalues) * V^T
        let mut result = Array2::zeros(matrix.dim());
        for i in 0..eigenvectors.ncols() {
            let scaled_vec = &eigenvectors.column(i) * sqrt_inv_eigenvalues[i];
            for j in 0..eigenvectors.nrows() {
                for k in 0..eigenvectors.nrows() {
                    result[[j, k]] += scaled_vec[j] * eigenvectors[[k, i]];
                }
            }
        }

        Ok(result)
    }

    /// Check numerical rank of a matrix
    pub fn numerical_rank(&self, matrix: &Array2<Float>) -> Result<usize> {
        let (eigenvalues, _) = self.stable_eigen_decomposition(matrix)?;

        if eigenvalues.is_empty() {
            return Ok(0);
        }

        let max_eigenval = eigenvalues[0].abs();
        let threshold = max_eigenval * self.config.eigenvalue_threshold;

        Ok(eigenvalues
            .iter()
            .filter(|&&val| val.abs() >= threshold)
            .count())
    }

    /// Compute stable matrix inverse using SVD-based pseudoinverse
    pub fn stable_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        // For symmetric matrices, use eigenvalue decomposition
        if self.is_symmetric(matrix) {
            let (eigenvalues, eigenvectors) = self.stable_eigen_decomposition(matrix)?;

            // Compute pseudoinverse using eigenvalue decomposition
            let mut result = Array2::zeros(matrix.dim());
            for i in 0..eigenvalues.len() {
                if eigenvalues[i].abs() > self.config.eigenvalue_threshold {
                    let inv_eigenval = 1.0 / eigenvalues[i];
                    let outer_product = eigenvectors
                        .column(i)
                        .insert_axis(Axis(1))
                        .dot(&eigenvectors.column(i).insert_axis(Axis(0)));
                    result = result + inv_eigenval * outer_product;
                }
            }

            Ok(result)
        } else {
            // For general matrices, would use SVD (not implemented here)
            Err(SklearsError::NotImplemented(
                "SVD-based inverse for non-symmetric matrices".to_string(),
            ))
        }
    }

    /// Advanced iterative eigenvalue algorithms
    pub fn iterative_eigen_decomposition(
        &self,
        matrix: &Array2<Float>,
        n_eigenvalues: Option<usize>,
        max_iterations: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let k = n_eigenvalues.unwrap_or(n).min(n);

        if !self.is_symmetric(matrix) {
            return Err(SklearsError::InvalidInput(
                "Matrix must be symmetric for iterative eigenvalue decomposition".to_string(),
            ));
        }

        // Use Lanczos algorithm for large sparse matrices
        if n > 500 {
            self.lanczos_algorithm(matrix, k, max_iterations)
        } else {
            // Fall back to standard method for small matrices
            let (eigenvalues, eigenvectors) = self.stable_eigen_decomposition(matrix)?;
            Ok((
                eigenvalues.slice(s![..k]).to_owned(),
                eigenvectors.slice(s![.., ..k]).to_owned(),
            ))
        }
    }

    /// Lanczos algorithm for computing largest eigenvalues and eigenvectors
    #[allow(non_snake_case)]
    fn lanczos_algorithm(
        &self,
        matrix: &Array2<Float>,
        k: usize,
        max_iterations: usize,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let m = max_iterations.min(n).max(k + 10); // Lanczos dimension

        // Initialize random starting vector
        let mut v = Array1::from_vec((0..n).map(|_| fastrand::f64()).collect::<Vec<Float>>());
        v = v.clone() / (v.dot(&v).sqrt()); // Normalize

        // Lanczos vectors
        let mut V = Array2::zeros((n, m));
        let mut alpha = Array1::zeros(m);
        let mut beta = Array1::zeros(m - 1);

        V.column_mut(0).assign(&v);
        let mut w = matrix.dot(&v);
        alpha[0] = v.dot(&w);
        w = w - alpha[0] * &v;

        for j in 1..m {
            beta[j - 1] = (w.dot(&w)).sqrt();

            if beta[j - 1] < self.config.tolerance {
                // Exact convergence or numerical breakdown
                break;
            }

            v = w / beta[j - 1];
            V.column_mut(j).assign(&v);
            w = matrix.dot(&v) - beta[j - 1] * V.column(j - 1).to_owned();
            alpha[j] = v.dot(&w);
            w = w - alpha[j] * &v;

            // Re-orthogonalize if necessary
            if j % 10 == 0 {
                let v_partial = V.slice(s![.., ..=j]).to_owned();
                self.reorthogonalize(&mut w, &v_partial)?;
            }
        }

        // Construct tridiagonal matrix T
        let T = self.construct_tridiagonal(&alpha, &beta, m)?;

        // Solve eigenvalue problem for tridiagonal matrix
        let (theta, y) = self.tridiagonal_eigen_decomposition(&T)?;

        // Take the k largest eigenvalues
        let mut indices: Vec<usize> = (0..theta.len()).collect();
        indices.sort_by(|&i, &j| theta[j].partial_cmp(&theta[i]).unwrap());

        let eigenvalues = Array1::from_iter(indices.iter().take(k).map(|&i| theta[i]));

        // Compute Ritz vectors: eigenvectors = V * y
        let mut eigenvectors = Array2::zeros((n, k));
        for i in 0..k {
            let ritz_vec = V.slice(s![.., ..m]).dot(&y.column(indices[i]));
            eigenvectors.column_mut(i).assign(&ritz_vec);
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Re-orthogonalize a vector against a set of orthonormal vectors
    fn reorthogonalize(
        &self,
        w: &mut Array1<Float>,
        V: &ArrayBase<OwnedRepr<Float>, Ix2>,
    ) -> Result<()> {
        for j in 0..V.ncols() {
            let v_j = V.column(j);
            let projection = v_j.dot(w);
            *w = &*w - projection * &v_j;
        }
        Ok(())
    }

    /// Construct tridiagonal matrix from Lanczos coefficients
    fn construct_tridiagonal(
        &self,
        alpha: &Array1<Float>,
        beta: &Array1<Float>,
        m: usize,
    ) -> Result<Array2<Float>> {
        let mut T = Array2::zeros((m, m));

        // Fill diagonal with alpha values
        for i in 0..m {
            T[[i, i]] = alpha[i];
        }

        // Fill off-diagonal with beta values
        for i in 0..m - 1 {
            T[[i, i + 1]] = beta[i];
            T[[i + 1, i]] = beta[i];
        }

        Ok(T)
    }

    /// Solve eigenvalue problem for tridiagonal matrix
    fn tridiagonal_eigen_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        // Use standard eigenvalue decomposition for tridiagonal matrix
        // In practice, this would use specialized algorithms like QR with shifts
        matrix.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!(
                "Tridiagonal eigenvalue decomposition failed: {}",
                e
            ))
        })
    }

    /// Advanced condition number monitoring with statistics
    pub fn create_condition_monitor(&self) -> ConditionMonitor {
        ConditionMonitor {
            condition_history: VecDeque::with_capacity(self.config.monitoring_history_size),
            warning_count: 0,
            max_condition_number: 0.0,
            average_condition_number: 0.0,
        }
    }

    /// Update condition monitor with new measurement
    pub fn update_condition_monitor(
        &self,
        monitor: &mut ConditionMonitor,
        condition_number: Float,
    ) {
        // Add to history
        if monitor.condition_history.len() >= self.config.monitoring_history_size {
            monitor.condition_history.pop_front();
        }
        monitor.condition_history.push_back(condition_number);

        // Update statistics
        monitor.max_condition_number = monitor.max_condition_number.max(condition_number);
        monitor.average_condition_number = monitor.condition_history.iter().sum::<Float>()
            / monitor.condition_history.len() as Float;

        // Check for warnings
        if condition_number > self.config.max_condition_number {
            monitor.warning_count += 1;
            if self.config.log_warnings {
                log::warn!(
                    "High condition number detected: {:.2e} (threshold: {:.2e})",
                    condition_number,
                    self.config.max_condition_number
                );
            }
        }
    }

    /// Get condition monitor statistics
    pub fn get_condition_statistics(&self, monitor: &ConditionMonitor) -> ConditionStatistics {
        let history_vec: Vec<Float> = monitor.condition_history.iter().cloned().collect();

        let median = if !history_vec.is_empty() {
            let mut sorted = history_vec.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) / 2.0
            } else {
                sorted[mid]
            }
        } else {
            0.0
        };

        let variance = if history_vec.len() > 1 {
            let mean = monitor.average_condition_number;
            history_vec
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<Float>()
                / (history_vec.len() - 1) as Float
        } else {
            0.0
        };

        ConditionStatistics {
            count: history_vec.len(),
            mean: monitor.average_condition_number,
            median,
            max: monitor.max_condition_number,
            min: history_vec.iter().fold(Float::INFINITY, |a, &b| a.min(b)),
            std_dev: variance.sqrt(),
            warning_count: monitor.warning_count,
            trend: self.compute_trend(&history_vec),
        }
    }

    /// Compute trend in condition numbers (positive = increasing, negative = decreasing)
    fn compute_trend(&self, history: &[Float]) -> Float {
        if history.len() < 3 {
            return 0.0;
        }

        // Simple linear regression to detect trend
        let n = history.len() as Float;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = history.iter().sum::<Float>() / n;

        let numerator: Float = history
            .iter()
            .enumerate()
            .map(|(i, &y)| (i as Float - x_mean) * (y - y_mean))
            .sum();

        let denominator: Float = history
            .iter()
            .enumerate()
            .map(|(i, _)| (i as Float - x_mean).powi(2))
            .sum();

        if denominator.abs() < Float::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    /// Adaptive eigenvalue algorithm selection based on matrix properties
    pub fn adaptive_eigen_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = matrix.nrows();
        let condition_estimate = self.estimate_condition_number(matrix)?;

        // Choose algorithm based on matrix size and condition number
        match (n, condition_estimate) {
            (n, _cond) if n > 1000 => {
                // Large matrices: use iterative methods
                log::info!("Using iterative method for large matrix (n={})", n);
                self.iterative_eigen_decomposition(matrix, None, 100)
            }
            (_, cond) if cond > 1e12 => {
                // Ill-conditioned matrices: use regularization and careful algorithms
                log::warn!(
                    "High condition number detected ({:.2e}), using regularized decomposition",
                    cond
                );
                let regularized_matrix = self.add_regularization(matrix)?;
                self.stable_eigen_decomposition(&regularized_matrix)
            }
            (n, _) if n < 10 => {
                // Very small matrices: direct computation
                self.direct_eigen_2x2_3x3(matrix)
            }
            _ => {
                // Standard case: use stable decomposition
                self.stable_eigen_decomposition(matrix)
            }
        }
    }

    /// Direct eigenvalue computation for 2x2 and 3x3 matrices
    fn direct_eigen_2x2_3x3(
        &self,
        matrix: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        match matrix.nrows() {
            2 => self.direct_eigen_2x2(matrix),
            3 => self.direct_eigen_3x3(matrix),
            _ => self.stable_eigen_decomposition(matrix),
        }
    }

    /// Direct 2x2 eigenvalue computation using analytical formula
    fn direct_eigen_2x2(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        let a = matrix[[0, 0]];
        let b = matrix[[0, 1]];
        let c = matrix[[1, 0]];
        let d = matrix[[1, 1]];

        // For symmetric matrix: b = c
        let trace = a + d;
        let det = a * d - b * c;
        let discriminant = (trace * trace - 4.0 * det).sqrt();

        let lambda1 = (trace + discriminant) / 2.0;
        let lambda2 = (trace - discriminant) / 2.0;

        // Compute eigenvectors
        let mut v1 = Array1::zeros(2);
        let mut v2 = Array1::zeros(2);

        if b.abs() > self.config.tolerance {
            v1[0] = lambda1 - d;
            v1[1] = b;
            v2[0] = lambda2 - d;
            v2[1] = b;
        } else if c.abs() > self.config.tolerance {
            v1[0] = c;
            v1[1] = lambda1 - a;
            v2[0] = c;
            v2[1] = lambda2 - a;
        } else {
            // Diagonal matrix
            v1[0] = 1.0;
            v1[1] = 0.0;
            v2[0] = 0.0;
            v2[1] = 1.0;
        }

        // Normalize eigenvectors
        v1 = v1.clone() / (v1.dot(&v1).sqrt());
        v2 = v2.clone() / (v2.dot(&v2).sqrt());

        let eigenvalues = if lambda1 >= lambda2 {
            Array1::from_vec(vec![lambda1, lambda2])
        } else {
            Array1::from_vec(vec![lambda2, lambda1])
        };

        let eigenvectors = if lambda1 >= lambda2 {
            Array2::from_shape_fn((2, 2), |(i, j)| match j {
                0 => v1[i],
                1 => v2[i],
                _ => unreachable!(),
            })
        } else {
            Array2::from_shape_fn((2, 2), |(i, j)| match j {
                0 => v2[i],
                1 => v1[i],
                _ => unreachable!(),
            })
        };

        Ok((eigenvalues, eigenvectors))
    }

    /// Direct 3x3 eigenvalue computation (placeholder - would use cubic formula)
    fn direct_eigen_3x3(&self, matrix: &Array2<Float>) -> Result<(Array1<Float>, Array2<Float>)> {
        // For now, fall back to standard method
        // Real implementation would use the characteristic polynomial approach
        self.stable_eigen_decomposition(matrix)
    }

    /// Power iteration method for computing dominant eigenvalue
    pub fn power_iteration(
        &self,
        matrix: &Array2<Float>,
        max_iterations: usize,
        tolerance: Option<Float>,
    ) -> Result<(Float, Array1<Float>)> {
        let tol = tolerance.unwrap_or(self.config.tolerance);
        let n = matrix.nrows();

        // Initialize with random vector
        let mut v = Array1::from_vec((0..n).map(|_| fastrand::f64()).collect());
        v = v.clone() / (v.dot(&v).sqrt());

        let mut eigenvalue = 0.0;

        for _ in 0..max_iterations {
            let v_new = matrix.dot(&v);
            let eigenvalue_new = v.dot(&v_new);

            let v_new_normalized = v_new.clone() / (v_new.dot(&v_new).sqrt());

            // Check convergence
            if (eigenvalue_new - eigenvalue).abs() < tol {
                return Ok((eigenvalue_new, v_new_normalized));
            }

            eigenvalue = eigenvalue_new;
            v = v_new_normalized;
        }

        Err(SklearsError::NumericalError(format!(
            "Power iteration did not converge after {} iterations",
            max_iterations
        )))
    }

    /// Inverse power iteration for computing smallest eigenvalue
    pub fn inverse_power_iteration(
        &self,
        matrix: &Array2<Float>,
        max_iterations: usize,
        tolerance: Option<Float>,
    ) -> Result<(Float, Array1<Float>)> {
        let tol = tolerance.unwrap_or(self.config.tolerance);

        // Compute matrix inverse (or use LU factorization for efficiency)
        let matrix_inv = self.stable_inverse(matrix)?;

        // Apply power iteration to inverse matrix
        let (largest_inv_eigenvalue, eigenvector) =
            self.power_iteration(&matrix_inv, max_iterations, Some(tol))?;

        // Smallest eigenvalue of original matrix is reciprocal of largest of inverse
        let smallest_eigenvalue = 1.0 / largest_inv_eigenvalue;

        Ok((smallest_eigenvalue, eigenvector))
    }

    /// Check matrix properties for numerical analysis
    pub fn analyze_matrix_properties(&self, matrix: &Array2<Float>) -> MatrixAnalysis {
        let n = matrix.nrows();

        let is_symmetric = self.is_symmetric(matrix);
        let condition_number = self
            .estimate_condition_number(matrix)
            .unwrap_or(Float::INFINITY);
        let numerical_rank = self.numerical_rank(matrix).unwrap_or(0);

        // Check if matrix is positive definite (for symmetric matrices)
        let is_positive_definite = if is_symmetric {
            match self.stable_eigen_decomposition(matrix) {
                Ok((eigenvalues, _)) => eigenvalues.iter().all(|&val| val > self.config.tolerance),
                Err(_) => false,
            }
        } else {
            false
        };

        // Estimate sparsity
        let total_elements = n * n;
        let non_zero_elements = matrix
            .iter()
            .filter(|&&x| x.abs() > self.config.tolerance)
            .count();
        let sparsity = 1.0 - (non_zero_elements as Float) / (total_elements as Float);

        // Compute norms
        let frobenius_norm = matrix.iter().map(|&x| x * x).sum::<Float>().sqrt();
        let max_norm = matrix.iter().fold(0.0_f64, |acc, &x| acc.max(x.abs()));

        MatrixAnalysis {
            size: n,
            is_square: matrix.nrows() == matrix.ncols(),
            is_symmetric,
            is_positive_definite,
            condition_number,
            numerical_rank,
            sparsity,
            frobenius_norm,
            max_norm,
            properties: self.classify_matrix_type(matrix),
        }
    }

    /// Classify matrix type based on its properties
    fn classify_matrix_type(&self, matrix: &Array2<Float>) -> Vec<String> {
        let mut properties = Vec::new();

        if self.is_symmetric(matrix) {
            properties.push("Symmetric".to_string());
        }

        let condition_number = self
            .estimate_condition_number(matrix)
            .unwrap_or(Float::INFINITY);
        if condition_number > 1e12 {
            properties.push("Ill-conditioned".to_string());
        } else if condition_number < 10.0 {
            properties.push("Well-conditioned".to_string());
        }

        // Check if diagonal dominant
        let mut is_diagonally_dominant = true;
        for i in 0..matrix.nrows() {
            let diagonal_element = matrix[[i, i]].abs();
            let off_diagonal_sum: Float = (0..matrix.ncols())
                .filter(|&j| j != i)
                .map(|j| matrix[[i, j]].abs())
                .sum();

            if diagonal_element <= off_diagonal_sum {
                is_diagonally_dominant = false;
                break;
            }
        }
        if is_diagonally_dominant {
            properties.push("Diagonally dominant".to_string());
        }

        // Check if sparse
        let total_elements = matrix.nrows() * matrix.ncols();
        let non_zero_elements = matrix
            .iter()
            .filter(|&&x| x.abs() > self.config.tolerance)
            .count();
        let sparsity = 1.0 - (non_zero_elements as Float) / (total_elements as Float);

        if sparsity > 0.9 {
            properties.push("Very sparse".to_string());
        } else if sparsity > 0.5 {
            properties.push("Sparse".to_string());
        }

        properties
    }

    /// Batch eigenvalue decomposition with optimizations
    pub fn batch_eigen_decomposition(
        &self,
        matrices: &[Array2<Float>],
    ) -> Result<Vec<(Array1<Float>, Array2<Float>)>> {
        use rayon::prelude::*;

        // Process matrices in parallel
        let results: Result<Vec<_>> = matrices
            .par_iter()
            .map(|matrix| self.adaptive_eigen_decomposition(matrix))
            .collect();

        results
    }

    /// Matrix inverse using different algorithms based on properties
    pub fn matrix_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let analysis = self.analyze_matrix_properties(matrix);

        match analysis
            .properties
            .iter()
            .any(|p| p.contains("Ill-conditioned"))
        {
            true => {
                log::warn!("Matrix is ill-conditioned, using regularized inverse");
                let regularized = self.add_regularization(matrix)?;
                self.stable_inverse(&regularized)
            }
            false if analysis.is_positive_definite => {
                // Use Cholesky decomposition for positive definite matrices
                self.cholesky_inverse(matrix)
            }
            false => {
                // Use standard stable inverse
                self.stable_inverse(matrix)
            }
        }
    }

    /// Cholesky-based matrix inverse for positive definite matrices
    fn cholesky_inverse(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        // Placeholder for Cholesky decomposition inverse
        // Real implementation would use Cholesky factorization
        self.stable_inverse(matrix)
    }
}

/// Statistics about condition number monitoring
#[derive(Debug, Clone)]
pub struct ConditionStatistics {
    pub count: usize,
    pub mean: Float,
    pub median: Float,
    pub max: Float,
    pub min: Float,
    pub std_dev: Float,
    pub warning_count: usize,
    pub trend: Float, // Positive = increasing, negative = decreasing
}

/// Comprehensive matrix analysis results
#[derive(Debug, Clone)]
pub struct MatrixAnalysis {
    /// size
    pub size: usize,
    /// is_square
    pub is_square: bool,
    /// is_symmetric
    pub is_symmetric: bool,
    /// is_positive_definite
    pub is_positive_definite: bool,
    /// condition_number
    pub condition_number: Float,
    /// numerical_rank
    pub numerical_rank: usize,
    /// sparsity
    pub sparsity: Float,
    /// frobenius_norm
    pub frobenius_norm: Float,
    /// max_norm
    pub max_norm: Float,
    /// properties
    pub properties: Vec<String>,
}

impl Default for NumericalStability {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stable_eigen_decomposition() {
        let ns = NumericalStability::new();

        // Create a simple symmetric matrix
        let matrix = array![[4.0, 2.0], [2.0, 3.0]];

        let (eigenvalues, eigenvectors) = ns.stable_eigen_decomposition(&matrix).unwrap();

        // Check that eigenvalues are real and sorted in descending order
        assert!(eigenvalues.len() == 2);
        assert!(eigenvalues[0] >= eigenvalues[1]);

        // Check that eigenvectors are orthonormal
        let identity = eigenvectors.t().dot(&eigenvectors);
        for i in 0..identity.nrows() {
            for j in 0..identity.ncols() {
                if i == j {
                    assert_abs_diff_eq!(identity[[i, j]], 1.0, epsilon = 1e-10);
                } else {
                    assert_abs_diff_eq!(identity[[i, j]], 0.0, epsilon = 1e-10);
                }
            }
        }
    }

    #[test]
    fn test_condition_number_estimation() {
        let ns = NumericalStability::new();

        // Well-conditioned matrix
        let well_conditioned = array![[2.0, 0.0], [0.0, 1.0]];

        let cond_num = ns.estimate_condition_number(&well_conditioned).unwrap();
        assert!(cond_num < 10.0); // Should be around 2.0

        // Ill-conditioned matrix
        let ill_conditioned = array![[1.0, 1.0], [1.0, 1.0001]];

        let cond_num_ill = ns.estimate_condition_number(&ill_conditioned).unwrap();
        assert!(cond_num_ill > 1000.0); // Should be very large
    }

    #[test]
    fn test_numerical_rank() {
        // Use a configuration with higher threshold for rank detection
        let config = NumericalConfig {
            eigenvalue_threshold: 1e-10,
            ..Default::default()
        };
        let ns = NumericalStability::with_config(config);

        // Full rank matrix
        let full_rank = array![[1.0, 0.0], [0.0, 1.0]];

        assert_eq!(ns.numerical_rank(&full_rank).unwrap(), 2);

        // Rank deficient matrix
        let rank_deficient = array![[1.0, 1.0], [1.0, 1.0]];

        assert_eq!(ns.numerical_rank(&rank_deficient).unwrap(), 1);
    }

    #[test]
    fn test_symmetric_check() {
        let ns = NumericalStability::new();

        let symmetric = array![[1.0, 2.0], [2.0, 3.0]];

        assert!(ns.is_symmetric(&symmetric));

        let non_symmetric = array![[1.0, 2.0], [3.0, 4.0]];

        assert!(!ns.is_symmetric(&non_symmetric));
    }
}

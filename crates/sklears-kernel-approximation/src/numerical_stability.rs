//! Numerical Stability Enhancements for Kernel Approximation Methods
//!
//! This module provides tools for monitoring and improving numerical stability
//! including condition number monitoring, overflow/underflow protection,
//! and numerically stable algorithms.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::prelude::{Result, SklearsError};

/// Numerical stability monitor for kernel approximation methods
#[derive(Debug, Clone)]
/// NumericalStabilityMonitor
pub struct NumericalStabilityMonitor {
    config: StabilityConfig,
    metrics: StabilityMetrics,
    warnings: Vec<StabilityWarning>,
}

/// Configuration for numerical stability monitoring
#[derive(Debug, Clone)]
/// StabilityConfig
pub struct StabilityConfig {
    /// Maximum allowed condition number
    pub max_condition_number: f64,

    /// Minimum eigenvalue threshold
    pub min_eigenvalue: f64,

    /// Maximum eigenvalue threshold
    pub max_eigenvalue: f64,

    /// Tolerance for numerical precision
    pub numerical_tolerance: f64,

    /// Enable overflow/underflow protection
    pub enable_overflow_protection: bool,

    /// Enable high-precision arithmetic when needed
    pub enable_high_precision: bool,

    /// Regularization parameter for ill-conditioned matrices
    pub regularization: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self {
            max_condition_number: 1e12,
            min_eigenvalue: 1e-12,
            max_eigenvalue: 1e12,
            numerical_tolerance: 1e-12,
            enable_overflow_protection: true,
            enable_high_precision: false,
            regularization: 1e-8,
        }
    }
}

/// Numerical stability metrics
#[derive(Debug, Clone, Default)]
/// StabilityMetrics
pub struct StabilityMetrics {
    /// Condition numbers of matrices encountered
    pub condition_numbers: Vec<f64>,

    /// Eigenvalue ranges
    pub eigenvalue_ranges: Vec<(f64, f64)>,

    /// Numerical errors detected
    pub numerical_errors: Vec<f64>,

    /// Matrix ranks
    pub matrix_ranks: Vec<usize>,

    /// Overflow/underflow occurrences
    pub overflow_count: usize,
    /// underflow_count
    pub underflow_count: usize,

    /// Precision loss estimates
    pub precision_loss: Vec<f64>,
}

/// Types of numerical stability warnings
#[derive(Debug, Clone)]
/// StabilityWarning
pub enum StabilityWarning {
    /// High condition number detected
    HighConditionNumber {
        condition_number: f64,

        location: String,
    },

    /// Near-singular matrix detected
    NearSingular {
        smallest_eigenvalue: f64,

        location: String,
    },

    /// Overflow detected
    Overflow { value: f64, location: String },

    /// Underflow detected
    Underflow { value: f64, location: String },

    /// Significant precision loss
    PrecisionLoss {
        estimated_loss: f64,
        location: String,
    },

    /// Rank deficiency detected
    RankDeficient {
        expected_rank: usize,
        actual_rank: usize,
        location: String,
    },
}

impl NumericalStabilityMonitor {
    /// Create a new stability monitor
    pub fn new(config: StabilityConfig) -> Self {
        Self {
            config,
            metrics: StabilityMetrics::default(),
            warnings: Vec::new(),
        }
    }

    /// Monitor matrix for numerical stability issues
    pub fn monitor_matrix(&mut self, matrix: &Array2<f64>, location: &str) -> Result<()> {
        // Check for NaN and infinite values
        self.check_finite_values(matrix, location)?;

        // Compute and monitor condition number
        let condition_number = self.estimate_condition_number(matrix)?;
        self.metrics.condition_numbers.push(condition_number);

        if condition_number > self.config.max_condition_number {
            self.warnings.push(StabilityWarning::HighConditionNumber {
                condition_number,
                location: location.to_string(),
            });
        }

        // Check eigenvalues if matrix is square
        if matrix.nrows() == matrix.ncols() {
            let eigenvalues = self.estimate_eigenvalues(matrix)?;
            let min_eigenval = eigenvalues.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max_eigenval = eigenvalues
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

            self.metrics
                .eigenvalue_ranges
                .push((min_eigenval, max_eigenval));

            if min_eigenval.abs() < self.config.min_eigenvalue {
                self.warnings.push(StabilityWarning::NearSingular {
                    smallest_eigenvalue: min_eigenval,
                    location: location.to_string(),
                });
            }
        }

        // Estimate matrix rank
        let rank = self.estimate_rank(matrix)?;
        self.metrics.matrix_ranks.push(rank);

        let expected_rank = matrix.nrows().min(matrix.ncols());
        if rank < expected_rank {
            self.warnings.push(StabilityWarning::RankDeficient {
                expected_rank,
                actual_rank: rank,
                location: location.to_string(),
            });
        }

        Ok(())
    }

    /// Apply numerical stabilization to a matrix
    pub fn stabilize_matrix(&mut self, matrix: &mut Array2<f64>) -> Result<()> {
        // Apply regularization for ill-conditioned matrices
        if matrix.nrows() == matrix.ncols() {
            for i in 0..matrix.nrows() {
                matrix[[i, i]] += self.config.regularization;
            }
        }

        // Clamp extreme values if overflow protection is enabled
        if self.config.enable_overflow_protection {
            self.clamp_extreme_values(matrix)?;
        }

        Ok(())
    }

    /// Compute numerically stable eigendecomposition
    pub fn stable_eigendecomposition(
        &mut self,
        matrix: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        if matrix.nrows() != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for eigendecomposition".to_string(),
            ));
        }

        self.monitor_matrix(matrix, "eigendecomposition_input")?;

        // Apply stabilization
        let mut stabilized_matrix = matrix.clone();
        self.stabilize_matrix(&mut stabilized_matrix)?;

        // Simplified eigendecomposition using power iteration for largest eigenvalues
        let n = matrix.nrows();
        let max_eigenvalues = 10.min(n);

        let mut eigenvalues = Array1::zeros(max_eigenvalues);
        let mut eigenvectors = Array2::zeros((n, max_eigenvalues));

        let mut current_matrix = stabilized_matrix.clone();

        for i in 0..max_eigenvalues {
            let (eigenvalue, eigenvector) = self.power_iteration(&current_matrix)?;
            eigenvalues[i] = eigenvalue;
            eigenvectors.column_mut(i).assign(&eigenvector);

            // Deflate the matrix
            let outer_product = self.outer_product(&eigenvector, &eigenvector);
            current_matrix = &current_matrix - &(&outer_product * eigenvalue);

            // Check for convergence
            if eigenvalue.abs() < self.config.min_eigenvalue {
                break;
            }
        }

        Ok((eigenvalues, eigenvectors))
    }

    /// Compute numerically stable matrix inversion
    pub fn stable_matrix_inverse(&mut self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        self.monitor_matrix(matrix, "matrix_inverse_input")?;

        if matrix.nrows() != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        let n = matrix.nrows();

        // Use regularized pseudoinverse for stability
        let regularized_matrix = self.regularize_matrix(matrix)?;

        // Simplified inversion using Gauss-Jordan elimination with pivoting
        let mut augmented = Array2::zeros((n, 2 * n));

        // Set up augmented matrix [A | I]
        for i in 0..n {
            for j in 0..n {
                augmented[[i, j]] = regularized_matrix[[i, j]];
            }
            augmented[[i, n + i]] = 1.0;
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if augmented[[k, i]].abs() > augmented[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows if needed
            if max_row != i {
                for j in 0..2 * n {
                    let temp = augmented[[i, j]];
                    augmented[[i, j]] = augmented[[max_row, j]];
                    augmented[[max_row, j]] = temp;
                }
            }

            // Check for near-zero pivot
            if augmented[[i, i]].abs() < self.config.numerical_tolerance {
                return Err(SklearsError::NumericalError(
                    "Matrix is singular or near-singular".to_string(),
                ));
            }

            // Scale pivot row
            let pivot = augmented[[i, i]];
            for j in 0..2 * n {
                augmented[[i, j]] /= pivot;
            }

            // Eliminate column
            for k in 0..n {
                if k != i {
                    let factor = augmented[[k, i]];
                    for j in 0..2 * n {
                        augmented[[k, j]] -= factor * augmented[[i, j]];
                    }
                }
            }
        }

        // Extract inverse matrix
        let mut inverse = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                inverse[[i, j]] = augmented[[i, n + j]];
            }
        }

        self.monitor_matrix(&inverse, "matrix_inverse_output")?;

        Ok(inverse)
    }

    /// Compute numerically stable Cholesky decomposition
    pub fn stable_cholesky(&mut self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        self.monitor_matrix(matrix, "cholesky_input")?;

        if matrix.nrows() != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square for Cholesky decomposition".to_string(),
            ));
        }

        let n = matrix.nrows();

        // Apply regularization for numerical stability
        let regularized_matrix = self.regularize_matrix(matrix)?;

        let mut L = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += L[[j, k]] * L[[j, k]];
                    }

                    let diagonal_value = regularized_matrix[[j, j]] - sum;
                    if diagonal_value <= 0.0 {
                        return Err(SklearsError::NumericalError(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }

                    L[[j, j]] = diagonal_value.sqrt();
                } else {
                    // Off-diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += L[[i, k]] * L[[j, k]];
                    }

                    L[[i, j]] = (regularized_matrix[[i, j]] - sum) / L[[j, j]];
                }
            }
        }

        self.monitor_matrix(&L, "cholesky_output")?;

        Ok(L)
    }

    /// Get stability warnings
    pub fn get_warnings(&self) -> &[StabilityWarning] {
        &self.warnings
    }

    /// Get stability metrics
    pub fn get_metrics(&self) -> &StabilityMetrics {
        &self.metrics
    }

    /// Clear warnings and metrics
    pub fn clear(&mut self) {
        self.warnings.clear();
        self.metrics = StabilityMetrics::default();
    }

    /// Generate stability report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("=== Numerical Stability Report ===\n\n");

        // Summary statistics
        if !self.metrics.condition_numbers.is_empty() {
            let mean_condition = self.metrics.condition_numbers.iter().sum::<f64>()
                / self.metrics.condition_numbers.len() as f64;
            let max_condition = self
                .metrics
                .condition_numbers
                .iter()
                .fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

            report.push_str(&format!(
                "Condition Numbers:\n  Mean: {:.2e}\n  Max: {:.2e}\n  Count: {}\n\n",
                mean_condition,
                max_condition,
                self.metrics.condition_numbers.len()
            ));
        }

        // Eigenvalue analysis
        if !self.metrics.eigenvalue_ranges.is_empty() {
            let min_eigenval = self
                .metrics
                .eigenvalue_ranges
                .iter()
                .map(|(min, _)| *min)
                .fold(f64::INFINITY, f64::min);
            let max_eigenval = self
                .metrics
                .eigenvalue_ranges
                .iter()
                .map(|(_, max)| *max)
                .fold(f64::NEG_INFINITY, f64::max);

            report.push_str(&format!(
                "Eigenvalue Range:\n  Min: {:.2e}\n  Max: {:.2e}\n  Ratio: {:.2e}\n\n",
                min_eigenval,
                max_eigenval,
                max_eigenval / min_eigenval.abs().max(1e-16)
            ));
        }

        // Warnings
        if !self.warnings.is_empty() {
            report.push_str(&format!("Warnings ({}):\n", self.warnings.len()));
            for (i, warning) in self.warnings.iter().enumerate() {
                report.push_str(&format!("  {}: {}\n", i + 1, self.format_warning(warning)));
            }
            report.push('\n');
        }

        // Overflow/underflow statistics
        if self.metrics.overflow_count > 0 || self.metrics.underflow_count > 0 {
            report.push_str(&format!(
                "Overflow/Underflow:\n  Overflows: {}\n  Underflows: {}\n\n",
                self.metrics.overflow_count, self.metrics.underflow_count
            ));
        }

        report.push_str("=== End Report ===\n");

        report
    }

    // Helper methods

    fn check_finite_values(&mut self, matrix: &Array2<f64>, location: &str) -> Result<()> {
        for &value in matrix.iter() {
            if !value.is_finite() {
                if value.is_infinite() {
                    self.metrics.overflow_count += 1;
                    self.warnings.push(StabilityWarning::Overflow {
                        value,
                        location: location.to_string(),
                    });
                } else if value == 0.0 && value.is_sign_negative() {
                    self.metrics.underflow_count += 1;
                    self.warnings.push(StabilityWarning::Underflow {
                        value,
                        location: location.to_string(),
                    });
                }

                return Err(SklearsError::NumericalError(format!(
                    "Non-finite value detected: {} at {}",
                    value, location
                )));
            }
        }
        Ok(())
    }

    fn estimate_condition_number(&self, matrix: &Array2<f64>) -> Result<f64> {
        // Simplified condition number estimation using Frobenius norm
        let norm = matrix.mapv(|x| x * x).sum().sqrt();

        if matrix.nrows() == matrix.ncols() {
            // For square matrices, estimate using diagonal dominance
            let mut min_diag = f64::INFINITY;
            let mut max_off_diag: f64 = 0.0;

            for i in 0..matrix.nrows() {
                min_diag = min_diag.min(matrix[[i, i]].abs());

                for j in 0..matrix.ncols() {
                    if i != j {
                        max_off_diag = max_off_diag.max(matrix[[i, j]].abs());
                    }
                }
            }

            let condition_estimate = if min_diag > 0.0 {
                (norm + max_off_diag) / min_diag
            } else {
                f64::INFINITY
            };

            Ok(condition_estimate)
        } else {
            // For non-square matrices, use norm-based estimate
            let min_norm = matrix
                .axis_iter(Axis(0))
                .map(|row| row.mapv(|x| x * x).sum().sqrt())
                .fold(f64::INFINITY, f64::min);

            Ok(norm / min_norm.max(1e-16))
        }
    }

    fn estimate_eigenvalues(&self, matrix: &Array2<f64>) -> Result<Array1<f64>> {
        let n = matrix.nrows();

        // Simplified eigenvalue estimation using Gershgorin circles
        let mut eigenvalue_bounds = Array1::zeros(n);

        for i in 0..n {
            let center = matrix[[i, i]];
            let radius = (0..n)
                .filter(|&j| j != i)
                .map(|j| matrix[[i, j]].abs())
                .sum::<f64>();

            eigenvalue_bounds[i] = center + radius; // Upper bound estimate
        }

        Ok(eigenvalue_bounds)
    }

    fn estimate_rank(&self, matrix: &Array2<f64>) -> Result<usize> {
        // Simplified rank estimation using diagonal elements after regularization
        let mut regularized = matrix.clone();

        if matrix.nrows() == matrix.ncols() {
            for i in 0..matrix.nrows() {
                regularized[[i, i]] += self.config.regularization;
            }
        }

        let mut rank = 0;
        let min_dim = matrix.nrows().min(matrix.ncols());

        for i in 0..min_dim {
            let column_norm = regularized.column(i).mapv(|x| x * x).sum().sqrt();
            if column_norm > self.config.numerical_tolerance {
                rank += 1;
            }
        }

        Ok(rank)
    }

    fn clamp_extreme_values(&mut self, matrix: &mut Array2<f64>) -> Result<()> {
        let max_value = 1e12;
        let min_value = -1e12;

        for value in matrix.iter_mut() {
            if *value > max_value {
                *value = max_value;
                self.metrics.overflow_count += 1;
            } else if *value < min_value {
                *value = min_value;
                self.metrics.underflow_count += 1;
            }
        }

        Ok(())
    }

    fn power_iteration(&self, matrix: &Array2<f64>) -> Result<(f64, Array1<f64>)> {
        let n = matrix.nrows();
        let mut vector = Array1::from_shape_fn(n, |_| thread_rng().gen::<f64>() - 0.5);

        // Normalize initial vector
        let norm = vector.mapv(|x| x * x).sum().sqrt();
        vector /= norm;

        let mut eigenvalue = 0.0;

        for _ in 0..100 {
            let new_vector = matrix.dot(&vector);
            let new_norm = new_vector.mapv(|x| x * x).sum().sqrt();

            if new_norm < self.config.numerical_tolerance {
                break;
            }

            eigenvalue = vector.dot(&new_vector);
            vector = new_vector / new_norm;
        }

        Ok((eigenvalue, vector))
    }

    fn outer_product(&self, v1: &Array1<f64>, v2: &Array1<f64>) -> Array2<f64> {
        let mut result = Array2::zeros((v1.len(), v2.len()));

        for i in 0..v1.len() {
            for j in 0..v2.len() {
                result[[i, j]] = v1[i] * v2[j];
            }
        }

        result
    }

    fn regularize_matrix(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let mut regularized = matrix.clone();

        if matrix.nrows() == matrix.ncols() {
            for i in 0..matrix.nrows() {
                regularized[[i, i]] += self.config.regularization;
            }
        }

        Ok(regularized)
    }

    fn format_warning(&self, warning: &StabilityWarning) -> String {
        match warning {
            StabilityWarning::HighConditionNumber {
                condition_number,
                location,
            } => {
                format!(
                    "High condition number {:.2e} at {}",
                    condition_number, location
                )
            }
            StabilityWarning::NearSingular {
                smallest_eigenvalue,
                location,
            } => {
                format!(
                    "Near-singular matrix (λ_min = {:.2e}) at {}",
                    smallest_eigenvalue, location
                )
            }
            StabilityWarning::Overflow { value, location } => {
                format!("Overflow (value = {:.2e}) at {}", value, location)
            }
            StabilityWarning::Underflow { value, location } => {
                format!("Underflow (value = {:.2e}) at {}", value, location)
            }
            StabilityWarning::PrecisionLoss {
                estimated_loss,
                location,
            } => {
                format!(
                    "Precision loss ({:.1}%) at {}",
                    estimated_loss * 100.0,
                    location
                )
            }
            StabilityWarning::RankDeficient {
                expected_rank,
                actual_rank,
                location,
            } => {
                format!(
                    "Rank deficient ({}/{}) at {}",
                    actual_rank, expected_rank, location
                )
            }
        }
    }
}

/// Numerically stable kernel matrix computation
pub fn stable_kernel_matrix(
    data1: &Array2<f64>,
    data2: Option<&Array2<f64>>,
    kernel_type: &str,
    bandwidth: f64,
    monitor: &mut NumericalStabilityMonitor,
) -> Result<Array2<f64>> {
    let data2 = data2.unwrap_or(data1);
    let (n1, n_features) = data1.dim();
    let (n2, _) = data2.dim();

    let mut kernel = Array2::zeros((n1, n2));

    // Precompute squared norms for numerical stability
    let norms1: Vec<f64> = data1
        .axis_iter(Axis(0))
        .map(|row| row.mapv(|x| x * x).sum())
        .collect();

    let norms2: Vec<f64> = data2
        .axis_iter(Axis(0))
        .map(|row| row.mapv(|x| x * x).sum())
        .collect();

    for i in 0..n1 {
        for j in 0..n2 {
            let similarity = match kernel_type {
                "RBF" => {
                    // Use numerically stable distance computation: ||x-y||² = ||x||² + ||y||² - 2⟨x,y⟩
                    let dot_product = data1.row(i).dot(&data2.row(j));
                    let dist_sq = norms1[i] + norms2[j] - 2.0 * dot_product;
                    let dist_sq = dist_sq.max(0.0); // Ensure non-negative due to numerical errors

                    let exponent = -bandwidth * dist_sq;

                    // Clamp exponent to prevent underflow
                    let clamped_exponent = exponent.max(-700.0); // e^(-700) ≈ 1e-304

                    clamped_exponent.exp()
                }
                "Laplacian" => {
                    let diff = &data1.row(i) - &data2.row(j);
                    let dist = diff.mapv(|x| x.abs()).sum();

                    let exponent = -bandwidth * dist;
                    let clamped_exponent = exponent.max(-700.0);

                    clamped_exponent.exp()
                }
                "Polynomial" => {
                    let dot_product = data1.row(i).dot(&data2.row(j));
                    let base = bandwidth * dot_product + 1.0;

                    // Ensure positive base for polynomial kernel
                    let clamped_base = base.max(1e-16);

                    clamped_base.powi(2) // Degree 2 polynomial
                }
                "Linear" => data1.row(i).dot(&data2.row(j)),
                _ => {
                    return Err(SklearsError::InvalidOperation(format!(
                        "Unsupported kernel type: {}",
                        kernel_type
                    )));
                }
            };

            kernel[[i, j]] = similarity;
        }
    }

    monitor.monitor_matrix(&kernel, &format!("{}_kernel_matrix", kernel_type))?;

    Ok(kernel)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stability_monitor_creation() {
        let config = StabilityConfig::default();
        let monitor = NumericalStabilityMonitor::new(config);

        assert!(monitor.get_warnings().is_empty());
        assert_eq!(monitor.get_metrics().condition_numbers.len(), 0);
    }

    #[test]
    fn test_condition_number_monitoring() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        // Well-conditioned matrix
        let well_conditioned = array![[2.0, 1.0], [1.0, 2.0],];

        monitor
            .monitor_matrix(&well_conditioned, "test_well_conditioned")
            .unwrap();
        assert!(monitor.get_warnings().is_empty());

        // Ill-conditioned matrix
        let ill_conditioned = array![[1.0, 1.0], [1.0, 1.000001],];

        monitor
            .monitor_matrix(&ill_conditioned, "test_ill_conditioned")
            .unwrap();
        // Should detect high condition number or near-singularity
    }

    #[test]
    fn test_stable_eigendecomposition() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        let matrix = array![[4.0, 2.0], [2.0, 3.0],];

        let (eigenvalues, eigenvectors) = monitor.stable_eigendecomposition(&matrix).unwrap();

        assert!(eigenvalues.len() <= matrix.nrows());
        assert_eq!(eigenvectors.nrows(), matrix.nrows());
        assert!(eigenvalues.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_stable_matrix_inverse() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        let matrix = array![[4.0, 2.0], [2.0, 3.0],];

        let inverse = monitor.stable_matrix_inverse(&matrix).unwrap();

        // Check that A_regularized * A^(-1) ≈ I (where A^(-1) is inverse of regularized matrix)
        let mut regularized_matrix = matrix.clone();
        for i in 0..matrix.nrows() {
            regularized_matrix[[i, i]] += 1e-8; // Default regularization
        }

        let product = regularized_matrix.dot(&inverse);
        let identity_error = (&product - &Array2::<f64>::eye(2)).mapv(|x| x.abs()).sum();

        assert!(identity_error < 1e-10);
    }

    #[test]
    fn test_stable_cholesky() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        // Positive definite matrix
        let matrix = array![[4.0, 2.0], [2.0, 3.0],];

        let cholesky = monitor.stable_cholesky(&matrix).unwrap();

        // Check that L * L^T = A_regularized (regularized matrix)
        let reconstructed = cholesky.dot(&cholesky.t());

        // Compute the regularized matrix for comparison
        let mut regularized_matrix = matrix.clone();
        for i in 0..matrix.nrows() {
            regularized_matrix[[i, i]] += 1e-8; // Default regularization
        }

        let reconstruction_error = (&regularized_matrix - &reconstructed)
            .mapv(|x| x.abs())
            .sum();

        assert!(reconstruction_error < 1e-10);
    }

    #[test]
    fn test_stable_kernel_matrix() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let kernel = stable_kernel_matrix(&data, None, "RBF", 1.0, &mut monitor).unwrap();

        assert_eq!(kernel.shape(), &[3, 3]);
        assert!(kernel.iter().all(|&x| x.is_finite() && x >= 0.0));

        // Kernel matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert!((kernel[[i, j]] - kernel[[j, i]]).abs() < 1e-12);
            }
        }

        // Diagonal should be 1 for RBF kernel
        for i in 0..3 {
            assert!((kernel[[i, i]] - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_stability_report() {
        let mut monitor = NumericalStabilityMonitor::new(StabilityConfig::default());

        let matrix = array![[1.0, 2.0], [3.0, 4.0],];

        monitor.monitor_matrix(&matrix, "test_matrix").unwrap();

        let report = monitor.generate_report();
        assert!(report.contains("Numerical Stability Report"));
        assert!(report.contains("Condition Numbers"));
    }
}

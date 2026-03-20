//! Diagnostic utilities for covariance estimation
//!
//! This module provides comprehensive diagnostic tools for analyzing
//! covariance matrices, identifying issues, and validating estimation quality.

use crate::utils::{
    frobenius_norm, is_diagonally_dominant, rank_estimate, spectral_radius_estimate,
    validate_covariance_matrix, CovarianceProperties,
};
use scirs2_core::ndarray::{Array1, Array2, Axis, NdFloat};
use sklears_core::error::{Result, SklearsError};

/// Comprehensive diagnostic report for a covariance matrix
#[derive(Debug, Clone)]
pub struct CovarianceDiagnostics<F: NdFloat> {
    /// Basic matrix properties
    pub properties: CovarianceProperties<F>,
    /// Frobenius norm
    pub frobenius_norm: F,
    /// Spectral radius (largest eigenvalue magnitude)
    pub spectral_radius: F,
    /// Estimated rank
    pub rank: usize,
    /// Whether matrix is diagonally dominant
    pub is_diagonally_dominant: bool,
    /// Diagonal element statistics
    pub diagonal_stats: DiagonalStats<F>,
    /// Off-diagonal element statistics
    pub off_diagonal_stats: OffDiagonalStats<F>,
    /// Correlation-based diagnostics
    pub correlation_diagnostics: CorrelationDiagnostics<F>,
    /// Quality assessment
    pub quality_assessment: QualityAssessment,
}

/// Statistics for diagonal elements (variances)
#[derive(Debug, Clone)]
pub struct DiagonalStats<F: NdFloat> {
    pub mean: F,
    pub std_dev: F,
    pub min: F,
    pub max: F,
    pub median: F,
    pub range: F,
}

/// Statistics for off-diagonal elements (covariances)
#[derive(Debug, Clone)]
pub struct OffDiagonalStats<F: NdFloat> {
    pub mean: F,
    pub std_dev: F,
    pub min: F,
    pub max: F,
    pub median: F,
    pub mean_abs: F,
}

/// Correlation-based diagnostics
#[derive(Debug, Clone)]
pub struct CorrelationDiagnostics<F: NdFloat> {
    /// Maximum correlation coefficient
    pub max_correlation: F,
    /// Minimum correlation coefficient
    pub min_correlation: F,
    /// Mean absolute correlation
    pub mean_abs_correlation: F,
    /// Number of high correlations (>0.8)
    pub n_high_correlations: usize,
}

/// Overall quality assessment
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityAssessment {
    /// Excellent quality - all checks passed
    Excellent,
    /// Good quality - minor issues detected
    Good,
    /// Acceptable quality - some concerns
    Acceptable,
    /// Poor quality - significant issues
    Poor,
    /// Failed - critical issues detected
    Failed,
}

impl<F: NdFloat + std::fmt::Display> CovarianceDiagnostics<F> {
    /// Compute comprehensive diagnostics for a covariance matrix
    pub fn analyze(covariance: &Array2<F>) -> Result<Self> {
        let (n_rows, n_cols) = covariance.dim();

        if n_rows != n_cols {
            return Err(SklearsError::InvalidInput(
                "Covariance matrix must be square".to_string(),
            ));
        }

        // Validate basic properties
        let properties = validate_covariance_matrix(covariance)?;

        // Compute norms and other metrics
        let frobenius_norm = frobenius_norm(covariance);
        let spectral_radius = spectral_radius_estimate(covariance, 100)?;
        let rank = rank_estimate(
            covariance,
            F::from(1e-6)
                .ok_or_else(|| SklearsError::NumericalError("numeric conversion failed".into()))?,
        );
        let is_diagonally_dominant = is_diagonally_dominant(covariance);

        // Analyze diagonal elements
        let diagonal_stats = Self::compute_diagonal_stats(covariance);

        // Analyze off-diagonal elements
        let off_diagonal_stats = Self::compute_off_diagonal_stats(covariance);

        // Compute correlation diagnostics
        let correlation_diagnostics = Self::compute_correlation_diagnostics(covariance);

        // Assess overall quality
        let quality_assessment =
            Self::assess_quality(&properties, frobenius_norm, rank, n_rows, &diagonal_stats);

        Ok(Self {
            properties,
            frobenius_norm,
            spectral_radius,
            rank,
            is_diagonally_dominant,
            diagonal_stats,
            off_diagonal_stats,
            correlation_diagnostics,
            quality_assessment,
        })
    }

    fn compute_diagonal_stats(covariance: &Array2<F>) -> DiagonalStats<F> {
        let n = covariance.nrows();
        let mut diag_elements: Vec<F> = (0..n).map(|i| covariance[[i, i]]).collect();

        diag_elements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: F = diag_elements
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x);
        let mean = sum / F::from(n).expect("operation should succeed");

        let variance = diag_elements
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n).expect("operation should succeed");
        let std_dev = variance.sqrt();

        let min = diag_elements[0];
        let max = diag_elements[n - 1];
        let median = if n % 2 == 0 {
            (diag_elements[n / 2 - 1] + diag_elements[n / 2])
                / F::from(2.0).expect("operation should succeed")
        } else {
            diag_elements[n / 2]
        };
        let range = max - min;

        DiagonalStats {
            mean,
            std_dev,
            min,
            max,
            median,
            range,
        }
    }

    fn compute_off_diagonal_stats(covariance: &Array2<F>) -> OffDiagonalStats<F> {
        let n = covariance.nrows();
        let mut off_diag_elements = Vec::new();

        for i in 0..n {
            for j in 0..n {
                if i != j {
                    off_diag_elements.push(covariance[[i, j]]);
                }
            }
        }

        off_diag_elements.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n_elements = off_diag_elements.len();
        let sum: F = off_diag_elements
            .iter()
            .copied()
            .fold(F::zero(), |acc, x| acc + x);
        let mean = sum / F::from(n_elements).expect("operation should succeed");

        let variance = off_diag_elements
            .iter()
            .map(|&x| (x - mean) * (x - mean))
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n_elements).expect("operation should succeed");
        let std_dev = variance.sqrt();

        let min = off_diag_elements[0];
        let max = off_diag_elements[n_elements - 1];
        let median = if n_elements % 2 == 0 {
            (off_diag_elements[n_elements / 2 - 1] + off_diag_elements[n_elements / 2])
                / F::from(2.0).expect("operation should succeed")
        } else {
            off_diag_elements[n_elements / 2]
        };

        let mean_abs: F = off_diag_elements
            .iter()
            .map(|x| x.abs())
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(n_elements).expect("operation should succeed");

        OffDiagonalStats {
            mean,
            std_dev,
            min,
            max,
            median,
            mean_abs,
        }
    }

    fn compute_correlation_diagnostics(covariance: &Array2<F>) -> CorrelationDiagnostics<F> {
        let n = covariance.nrows();
        let mut correlations = Vec::new();
        let mut n_high = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let cov_ij = covariance[[i, j]];
                let var_i = covariance[[i, i]];
                let var_j = covariance[[j, j]];

                if var_i > F::zero() && var_j > F::zero() {
                    let corr = cov_ij / (var_i * var_j).sqrt();
                    correlations.push(corr);

                    if corr.abs() > F::from(0.8).expect("operation should succeed") {
                        n_high += 1;
                    }
                }
            }
        }

        if correlations.is_empty() {
            return CorrelationDiagnostics {
                max_correlation: F::zero(),
                min_correlation: F::zero(),
                mean_abs_correlation: F::zero(),
                n_high_correlations: 0,
            };
        }

        correlations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let max_correlation = correlations[correlations.len() - 1];
        let min_correlation = correlations[0];

        let mean_abs_correlation: F = correlations
            .iter()
            .map(|x| x.abs())
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(correlations.len()).expect("operation should succeed");

        CorrelationDiagnostics {
            max_correlation,
            min_correlation,
            mean_abs_correlation,
            n_high_correlations: n_high,
        }
    }

    fn assess_quality(
        properties: &CovarianceProperties<F>,
        frobenius_norm: F,
        rank: usize,
        n: usize,
        diagonal_stats: &DiagonalStats<F>,
    ) -> QualityAssessment {
        let mut issues = 0;
        let mut warnings = 0;

        // Check symmetry
        if !properties.is_symmetric {
            issues += 1;
        }

        // Check positive definiteness
        if !properties.is_positive_semi_definite {
            issues += 1;
        }

        // Check condition number
        if properties.condition_number > F::from(1e10).expect("operation should succeed") {
            warnings += 1;
        } else if properties.condition_number > F::from(1e15).expect("operation should succeed") {
            issues += 1;
        }

        // Check for numerical issues
        if !frobenius_norm.is_finite() {
            issues += 1;
        }

        // Check rank
        if rank < n / 2 {
            warnings += 1;
        }

        // Check diagonal elements
        if diagonal_stats.min <= F::zero() {
            issues += 1;
        }

        // Assess quality based on issues and warnings
        if issues >= 2 {
            QualityAssessment::Failed
        } else if issues == 1 {
            QualityAssessment::Poor
        } else if warnings >= 2 {
            QualityAssessment::Acceptable
        } else if warnings == 1 {
            QualityAssessment::Good
        } else {
            QualityAssessment::Excellent
        }
    }

    /// Print a formatted diagnostic report
    pub fn print_report(&self) {
        println!("╔══════════════════════════════════════════════════════════════╗");
        println!("║           Covariance Matrix Diagnostic Report               ║");
        println!("╚══════════════════════════════════════════════════════════════╝");

        println!("\n📊 OVERALL ASSESSMENT: {:?}", self.quality_assessment);

        println!("\n─── Basic Properties ───");
        println!("  Symmetric: {}", self.properties.is_symmetric);
        println!(
            "  Positive Definite: {}",
            self.properties.is_positive_definite
        );
        println!(
            "  Positive Semi-Definite: {}",
            self.properties.is_positive_semi_definite
        );
        println!("  Diagonally Dominant: {}", self.is_diagonally_dominant);

        println!("\n─── Matrix Metrics ───");
        println!(
            "  Condition Number: {:.2e}",
            self.properties.condition_number
        );
        println!("  Trace: {}", self.properties.trace);
        println!("  Determinant: {:.4e}", self.properties.determinant);
        println!("  Frobenius Norm: {}", self.frobenius_norm);
        println!("  Spectral Radius: {}", self.spectral_radius);
        println!("  Estimated Rank: {}", self.rank);

        println!("\n─── Eigenvalue Range ───");
        println!("  Min Eigenvalue: {}", self.properties.min_eigenvalue);
        println!("  Max Eigenvalue: {}", self.properties.max_eigenvalue);

        println!("\n─── Diagonal Statistics (Variances) ───");
        println!("  Mean: {}", self.diagonal_stats.mean);
        println!("  Std Dev: {}", self.diagonal_stats.std_dev);
        println!("  Min: {}", self.diagonal_stats.min);
        println!("  Max: {}", self.diagonal_stats.max);
        println!("  Median: {}", self.diagonal_stats.median);
        println!("  Range: {}", self.diagonal_stats.range);

        println!("\n─── Off-Diagonal Statistics (Covariances) ───");
        println!("  Mean: {}", self.off_diagonal_stats.mean);
        println!("  Std Dev: {}", self.off_diagonal_stats.std_dev);
        println!("  Min: {}", self.off_diagonal_stats.min);
        println!("  Max: {}", self.off_diagonal_stats.max);
        println!("  Median: {}", self.off_diagonal_stats.median);
        println!("  Mean Abs: {}", self.off_diagonal_stats.mean_abs);

        println!("\n─── Correlation Diagnostics ───");
        println!(
            "  Max Correlation: {}",
            self.correlation_diagnostics.max_correlation
        );
        println!(
            "  Min Correlation: {}",
            self.correlation_diagnostics.min_correlation
        );
        println!(
            "  Mean Abs Correlation: {}",
            self.correlation_diagnostics.mean_abs_correlation
        );
        println!(
            "  High Correlations (>0.8): {}",
            self.correlation_diagnostics.n_high_correlations
        );

        println!("\n─── Quality Indicators ───");
        self.print_quality_indicators();

        println!();
    }

    fn print_quality_indicators(&self) {
        let condition_threshold = F::from(1e10).expect("operation should succeed");
        if self.properties.condition_number > condition_threshold {
            println!("  ⚠️  High condition number - matrix may be ill-conditioned");
        } else {
            println!("  ✓  Condition number is acceptable");
        }

        if self.properties.is_symmetric {
            println!("  ✓  Matrix is symmetric");
        } else {
            println!("  ❌ Matrix is NOT symmetric");
        }

        if self.properties.is_positive_semi_definite {
            println!("  ✓  Matrix is positive semi-definite");
        } else {
            println!("  ❌ Matrix is NOT positive semi-definite");
        }

        if self.diagonal_stats.min > F::zero() {
            println!("  ✓  All variances are positive");
        } else {
            println!("  ❌ Some variances are non-positive");
        }

        if self.correlation_diagnostics.n_high_correlations > 0 {
            println!(
                "  ⚠️  {} pairs have high correlation (>0.8) - possible multicollinearity",
                self.correlation_diagnostics.n_high_correlations
            );
        }
    }
}

/// Compare two covariance matrices
pub fn compare_covariance_matrices<F: NdFloat + std::fmt::Display>(
    cov1: &Array2<F>,
    cov2: &Array2<F>,
    name1: &str,
    name2: &str,
) -> Result<()> {
    if cov1.shape() != cov2.shape() {
        return Err(SklearsError::InvalidInput(
            "Covariance matrices must have the same shape".to_string(),
        ));
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!(
        "║   Comparing: {} vs {}{}║",
        name1,
        name2,
        " ".repeat(52 - name1.len() - name2.len())
    );
    println!("╚══════════════════════════════════════════════════════════════╝");

    // Compute difference
    let diff = cov1 - cov2;
    let diff_norm = frobenius_norm(&diff);
    let cov1_norm = frobenius_norm(cov1);

    println!("\n─── Difference Metrics ───");
    println!("  Frobenius Norm of Difference: {}", diff_norm);
    println!(
        "  Relative Difference: {:.2}%",
        (diff_norm / cov1_norm)
            * F::from(100.0)
                .ok_or_else(|| SklearsError::NumericalError("numeric conversion failed".into()))?
    );

    // Element-wise statistics
    let n = cov1.nrows();
    let mut max_abs_diff = F::zero();
    let mut max_rel_diff = F::zero();

    for i in 0..n {
        for j in 0..n {
            let abs_diff = (cov1[[i, j]] - cov2[[i, j]]).abs();
            if abs_diff > max_abs_diff {
                max_abs_diff = abs_diff;
            }

            if cov1[[i, j]].abs()
                > F::from(1e-10).ok_or_else(|| {
                    SklearsError::NumericalError("numeric conversion failed".into())
                })?
            {
                let rel_diff = abs_diff / cov1[[i, j]].abs();
                if rel_diff > max_rel_diff {
                    max_rel_diff = rel_diff;
                }
            }
        }
    }

    println!("  Max Absolute Difference: {}", max_abs_diff);
    println!(
        "  Max Relative Difference: {:.2}%",
        max_rel_diff
            * F::from(100.0)
                .ok_or_else(|| SklearsError::NumericalError("numeric conversion failed".into()))?
    );

    println!();
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_diagnostics_valid_matrix() {
        let cov = array![[2.0, 0.5], [0.5, 3.0]];
        let diagnostics = CovarianceDiagnostics::analyze(&cov).expect("operation should succeed");

        assert!(diagnostics.properties.is_symmetric);
        assert!(diagnostics.properties.is_positive_semi_definite);
        assert_eq!(diagnostics.quality_assessment, QualityAssessment::Excellent);
    }

    #[test]
    fn test_diagnostics_identity_matrix() {
        let cov = Array2::<f64>::eye(5);
        let diagnostics = CovarianceDiagnostics::analyze(&cov).expect("operation should succeed");

        assert_eq!(diagnostics.rank, 5);
        assert!(diagnostics.is_diagonally_dominant);
        assert_eq!(diagnostics.quality_assessment, QualityAssessment::Excellent);
    }

    #[test]
    fn test_compare_matrices() {
        let cov1 = array![[1.0, 0.2], [0.2, 1.5]];
        let cov2 = array![[1.1, 0.25], [0.25, 1.6]];

        // Should not panic
        compare_covariance_matrices(&cov1, &cov2, "Matrix1", "Matrix2")
            .expect("operation should succeed");
    }
}

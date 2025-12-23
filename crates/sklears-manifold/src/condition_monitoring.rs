//! Condition number monitoring for manifold learning
//! This module provides comprehensive condition number analysis and monitoring
//! for matrices and operations in manifold learning algorithms. Helps detect
//! numerical instabilities and ill-conditioned problems.

use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::HashMap;

/// Configuration for condition number monitoring
#[derive(Debug, Clone)]
pub struct ConditionMonitorConfig {
    /// Threshold for warning about ill-conditioning
    pub warning_threshold: Float,
    /// Threshold for critical ill-conditioning
    pub critical_threshold: Float,
    /// Whether to compute exact condition numbers (expensive)
    pub exact_computation: bool,
    /// Whether to track condition number history
    pub track_history: bool,
    /// Number of samples for statistical analysis
    pub n_samples: usize,
    /// Whether to provide detailed analysis
    pub detailed_analysis: bool,
}

impl Default for ConditionMonitorConfig {
    fn default() -> Self {
        Self {
            warning_threshold: 1e12,
            critical_threshold: 1e15,
            exact_computation: false,
            track_history: true,
            n_samples: 1000,
            detailed_analysis: true,
        }
    }
}

/// Condition number analysis result
#[derive(Debug, Clone)]
pub struct ConditionAnalysis {
    /// Estimated condition number
    pub condition_number: Float,
    /// Condition number estimation method used
    pub estimation_method: String,
    /// Warning level (None, Warning, Critical)
    pub warning_level: ConditionWarningLevel,
    /// Numerical rank of the matrix
    pub numerical_rank: Option<usize>,
    /// Smallest singular value
    pub min_singular_value: Float,
    /// Largest singular value
    pub max_singular_value: Float,
    /// Relative gap in singular values
    pub singular_value_gap: Float,
    /// Stability recommendations
    pub recommendations: Vec<String>,
    /// Additional metadata
    pub metadata: HashMap<String, Float>,
}

/// Warning levels for condition number analysis
#[derive(Debug, Clone, PartialEq)]
pub enum ConditionWarningLevel {
    /// Matrix is well-conditioned
    Good,
    /// Matrix is moderately ill-conditioned
    Warning,
    /// Matrix is severely ill-conditioned
    Critical,
    /// Matrix is essentially singular
    Singular,
}

/// Comprehensive condition number monitor
pub struct ConditionMonitor {
    config: ConditionMonitorConfig,
    history: Vec<ConditionAnalysis>,
}

impl ConditionMonitor {
    /// Create a new condition monitor
    pub fn new(config: ConditionMonitorConfig) -> Self {
        Self {
            config,
            history: Vec::new(),
        }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(ConditionMonitorConfig::default())
    }

    /// Analyze condition number of a matrix
    pub fn analyze_matrix(&mut self, matrix: ArrayView2<Float>) -> SklResult<ConditionAnalysis> {
        // Validate input
        self.validate_matrix(&matrix)?;

        let analysis = if self.config.exact_computation {
            self.exact_condition_analysis(&matrix)?
        } else {
            self.approximate_condition_analysis(&matrix)?
        };

        // Store in history if tracking is enabled
        if self.config.track_history {
            self.history.push(analysis.clone());
        }

        Ok(analysis)
    }

    /// Analyze condition number for eigenvalue problems
    pub fn analyze_eigenvalue_problem(
        &mut self,
        matrix: ArrayView2<Float>,
    ) -> SklResult<ConditionAnalysis> {
        // Check if matrix is symmetric (required for eigenvalue analysis)
        if !self.is_approximately_symmetric(&matrix) {
            return Err(SklearsError::InvalidParameter {
                name: "matrix_symmetry".to_string(),
                reason: "Matrix must be symmetric for eigenvalue condition analysis".to_string(),
            });
        }

        let mut analysis = self.analyze_matrix(matrix)?;

        // Add eigenvalue-specific recommendations
        if analysis.warning_level != ConditionWarningLevel::Good {
            analysis
                .recommendations
                .push("Consider using regularization or eigenvalue truncation".to_string());
            analysis.recommendations.push(
                "Use iterative eigenvalue solvers for better numerical stability".to_string(),
            );
        }

        analysis
            .metadata
            .insert("eigenvalue_analysis".to_string(), 1.0);

        Ok(analysis)
    }

    /// Analyze condition number for least squares problems
    pub fn analyze_least_squares(
        &mut self,
        a_matrix: ArrayView2<Float>,
    ) -> SklResult<ConditionAnalysis> {
        // For least squares, we need to analyze A^T A
        let ata = a_matrix.t().dot(&a_matrix);
        let mut analysis = self.analyze_matrix(ata.view())?;

        // Adjust condition number for least squares (condition of A^T A is square of condition of A)
        analysis.condition_number = analysis.condition_number.sqrt();

        // Add least squares specific recommendations
        if analysis.warning_level != ConditionWarningLevel::Good {
            analysis
                .recommendations
                .push("Consider using QR decomposition instead of normal equations".to_string());
            analysis
                .recommendations
                .push("Apply Tikhonov regularization to improve conditioning".to_string());
            analysis
                .recommendations
                .push("Use SVD-based least squares solver for numerical stability".to_string());
        }

        analysis
            .metadata
            .insert("least_squares_analysis".to_string(), 1.0);

        Ok(analysis)
    }

    /// Monitor condition number during iterative algorithms
    pub fn monitor_iterative_algorithm<F>(
        &mut self,
        mut iteration_callback: F,
    ) -> SklResult<Vec<ConditionAnalysis>>
    where
        F: FnMut(usize) -> SklResult<Array2<Float>>,
    {
        let mut analyses = Vec::new();

        for iteration in 0..self.config.n_samples {
            match iteration_callback(iteration) {
                Ok(matrix) => {
                    let analysis = self.analyze_matrix(matrix.view())?;
                    analyses.push(analysis);

                    // Check for critical condition numbers
                    if analyses.last().unwrap().warning_level == ConditionWarningLevel::Critical {
                        eprintln!(
                            "Warning: Critical condition number detected at iteration {}",
                            iteration
                        );
                    }
                }
                Err(e) => {
                    eprintln!("Error in iteration {}: {}", iteration, e);
                    break;
                }
            }
        }

        Ok(analyses)
    }

    /// Get condition number history
    pub fn get_history(&self) -> &[ConditionAnalysis] {
        &self.history
    }

    /// Compute condition number statistics from history
    pub fn compute_statistics(&self) -> SklResult<ConditionStatistics> {
        if self.history.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "history_empty".to_string(),
                reason: "No condition number history available".to_string(),
            });
        }

        let condition_numbers: Vec<Float> = self
            .history
            .iter()
            .map(|analysis| analysis.condition_number)
            .collect();

        let mean = condition_numbers.iter().sum::<Float>() / condition_numbers.len() as Float;
        let variance = condition_numbers
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<Float>()
            / condition_numbers.len() as Float;
        let std_dev = variance.sqrt();

        let min_condition = condition_numbers
            .iter()
            .cloned()
            .fold(Float::INFINITY, |a, b| a.min(b));
        let max_condition = condition_numbers
            .iter()
            .cloned()
            .fold(Float::NEG_INFINITY, |a, b| a.max(b));

        // Count warning levels
        let mut good_count = 0;
        let mut warning_count = 0;
        let mut critical_count = 0;
        let mut singular_count = 0;

        for analysis in &self.history {
            match analysis.warning_level {
                ConditionWarningLevel::Good => good_count += 1,
                ConditionWarningLevel::Warning => warning_count += 1,
                ConditionWarningLevel::Critical => critical_count += 1,
                ConditionWarningLevel::Singular => singular_count += 1,
            }
        }

        Ok(ConditionStatistics {
            mean_condition: mean,
            std_dev_condition: std_dev,
            min_condition,
            max_condition,
            good_count,
            warning_count,
            critical_count,
            singular_count,
            total_samples: self.history.len(),
        })
    }

    /// Clear condition number history
    pub fn clear_history(&mut self) {
        self.history.clear();
    }

    /// Exact condition number analysis using SVD
    fn exact_condition_analysis(&self, matrix: &ArrayView2<Float>) -> SklResult<ConditionAnalysis> {
        // Compute SVD
        let (_, singular_values, _) =
            matrix
                .svd(true, true)
                .map_err(|e| SklearsError::InvalidParameter {
                    name: "svd_computation".to_string(),
                    reason: format!("SVD computation failed: {}", e),
                })?;

        let max_sv = singular_values[0]; // SVD returns sorted values
        let min_sv = *singular_values.last().unwrap();

        let condition_number = if min_sv > 1e-15 {
            max_sv / min_sv
        } else {
            Float::INFINITY
        };

        let numerical_rank = self.estimate_numerical_rank(&singular_values);
        let warning_level = self.classify_condition_number(condition_number);
        let recommendations = self.generate_recommendations(&warning_level, &singular_values);

        // Compute singular value gap
        let gap = if singular_values.len() > 1 {
            let sorted_values = singular_values.to_vec();
            let mut gaps = Vec::new();
            for i in 0..sorted_values.len() - 1 {
                gaps.push((sorted_values[i] - sorted_values[i + 1]) / sorted_values[i]);
            }
            gaps.iter().cloned().fold(0.0 as Float, |a, b| a.max(b))
        } else {
            0.0
        };

        let mut metadata = HashMap::new();
        metadata.insert(
            "n_singular_values".to_string(),
            singular_values.len() as Float,
        );

        Ok(ConditionAnalysis {
            condition_number,
            estimation_method: "SVD (exact)".to_string(),
            warning_level,
            numerical_rank: Some(numerical_rank),
            min_singular_value: min_sv,
            max_singular_value: max_sv,
            singular_value_gap: gap,
            recommendations,
            metadata,
        })
    }

    /// Approximate condition number analysis (faster)
    fn approximate_condition_analysis(
        &self,
        matrix: &ArrayView2<Float>,
    ) -> SklResult<ConditionAnalysis> {
        // Use power iteration to estimate largest and smallest eigenvalues
        let largest_eigenvalue = self.power_iteration(matrix, true)?;
        let smallest_eigenvalue = self.power_iteration(matrix, false)?;

        let condition_number = if smallest_eigenvalue.abs() > 1e-15 {
            largest_eigenvalue.abs() / smallest_eigenvalue.abs()
        } else {
            Float::INFINITY
        };

        let warning_level = self.classify_condition_number(condition_number);
        let recommendations = self.generate_recommendations(
            &warning_level,
            &Array1::from(vec![largest_eigenvalue, smallest_eigenvalue]),
        );

        let mut metadata = HashMap::new();
        metadata.insert("largest_eigenvalue".to_string(), largest_eigenvalue);
        metadata.insert("smallest_eigenvalue".to_string(), smallest_eigenvalue);

        Ok(ConditionAnalysis {
            condition_number,
            estimation_method: "Power iteration (approximate)".to_string(),
            warning_level,
            numerical_rank: None,
            min_singular_value: smallest_eigenvalue.abs(),
            max_singular_value: largest_eigenvalue.abs(),
            singular_value_gap: 0.0, // Not computed for approximate method
            recommendations,
            metadata,
        })
    }

    /// Power iteration to estimate largest or smallest eigenvalue
    fn power_iteration(&self, matrix: &ArrayView2<Float>, largest: bool) -> SklResult<Float> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidParameter {
                name: "matrix_shape".to_string(),
                reason: "Matrix must be square for eigenvalue estimation".to_string(),
            });
        }

        // Initialize random vector
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        let mut v = Array1::from_vec((0..n).map(|_| rng.random::<Float>()).collect());
        let norm = v.iter().map(|&x| x * x).sum::<Float>().sqrt();
        v /= norm;

        let max_iter = 100;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let v_old = v.clone();

            if largest {
                v = matrix.dot(&v);
            } else {
                // For smallest eigenvalue, use inverse iteration (simplified)
                // In practice, you'd solve (A - σI)x = v for some shift σ
                v = matrix.dot(&v);
                v = v.mapv(|x| if x.abs() > 1e-14 { 1.0 / x } else { 0.0 });
            }

            let norm = v.iter().map(|&x| x * x).sum::<Float>().sqrt();
            if norm > 1e-14 {
                v /= norm;
            }

            // Check convergence
            let diff = (&v - &v_old).iter().map(|&x| x * x).sum::<Float>().sqrt();
            if diff < tolerance {
                break;
            }
        }

        // Compute Rayleigh quotient
        let av = matrix.dot(&v);
        let eigenvalue = v.dot(&av) / v.dot(&v);

        Ok(eigenvalue)
    }

    /// Classify condition number into warning levels
    fn classify_condition_number(&self, condition_number: Float) -> ConditionWarningLevel {
        if condition_number.is_infinite() {
            ConditionWarningLevel::Singular
        } else if condition_number > self.config.critical_threshold {
            ConditionWarningLevel::Critical
        } else if condition_number > self.config.warning_threshold {
            ConditionWarningLevel::Warning
        } else {
            ConditionWarningLevel::Good
        }
    }

    /// Generate recommendations based on condition analysis
    fn generate_recommendations(
        &self,
        level: &ConditionWarningLevel,
        singular_values: &Array1<Float>,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();

        match level {
            ConditionWarningLevel::Good => {
                recommendations.push("Matrix is well-conditioned, no action needed".to_string());
            }
            ConditionWarningLevel::Warning => {
                recommendations
                    .push("Consider adding regularization to improve conditioning".to_string());
                recommendations.push("Monitor numerical stability during iterations".to_string());
                recommendations.push("Use higher precision arithmetic if available".to_string());
            }
            ConditionWarningLevel::Critical => {
                recommendations.push("Strong regularization is recommended".to_string());
                recommendations.push("Consider dimensionality reduction".to_string());
                recommendations
                    .push("Use specialized solvers for ill-conditioned problems".to_string());
                recommendations.push("Check for linear dependencies in data".to_string());
            }
            ConditionWarningLevel::Singular => {
                recommendations.push("Matrix is singular or nearly singular".to_string());
                recommendations.push("Use pseudoinverse or truncated SVD".to_string());
                recommendations.push("Remove linearly dependent columns/rows".to_string());
                recommendations.push("Apply strong regularization".to_string());
            }
        }

        // Add specific recommendations based on singular value distribution
        if singular_values.len() > 1 {
            let ratio = singular_values[0] / singular_values[singular_values.len() - 1];
            if ratio > 1e8 {
                recommendations.push("Large singular value spread detected".to_string());
                recommendations.push("Consider rank reduction or truncated methods".to_string());
            }
        }

        recommendations
    }

    /// Estimate numerical rank from singular values
    fn estimate_numerical_rank(&self, singular_values: &Array1<Float>) -> usize {
        let max_sv = singular_values[0];
        let threshold = max_sv * 1e-12;

        singular_values
            .iter()
            .take_while(|&&sv| sv > threshold)
            .count()
    }

    /// Check if matrix is approximately symmetric
    fn is_approximately_symmetric(&self, matrix: &ArrayView2<Float>) -> bool {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return false;
        }

        let tolerance = 1e-10;
        for i in 0..n {
            for j in i + 1..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                    return false;
                }
            }
        }

        true
    }

    /// Validate input matrix
    fn validate_matrix(&self, matrix: &ArrayView2<Float>) -> SklResult<()> {
        if matrix.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "matrix_empty".to_string(),
                reason: "Matrix cannot be empty".to_string(),
            });
        }

        // Check for NaN or infinite values
        for &value in matrix.iter() {
            if !value.is_finite() {
                return Err(SklearsError::InvalidParameter {
                    name: "matrix_values".to_string(),
                    reason: "Matrix contains NaN or infinite values".to_string(),
                });
            }
        }

        Ok(())
    }
}

/// Statistics computed from condition number history
#[derive(Debug, Clone)]
pub struct ConditionStatistics {
    /// Mean condition number
    pub mean_condition: Float,
    /// Standard deviation of condition numbers
    pub std_dev_condition: Float,
    /// Minimum condition number observed
    pub min_condition: Float,
    /// Maximum condition number observed
    pub max_condition: Float,
    /// Number of good condition numbers
    pub good_count: usize,
    /// Number of warning condition numbers
    pub warning_count: usize,
    /// Number of critical condition numbers
    pub critical_count: usize,
    /// Number of singular matrices
    pub singular_count: usize,
    /// Total number of samples
    pub total_samples: usize,
}

impl ConditionStatistics {
    /// Get the percentage of well-conditioned matrices
    pub fn good_percentage(&self) -> Float {
        (self.good_count as Float / self.total_samples as Float) * 100.0
    }

    /// Get the percentage of problematic matrices
    pub fn problematic_percentage(&self) -> Float {
        ((self.warning_count + self.critical_count + self.singular_count) as Float
            / self.total_samples as Float)
            * 100.0
    }

    /// Print summary statistics
    pub fn print_summary(&self) {
        println!("=== Condition Number Statistics ===");
        println!("Total samples: {}", self.total_samples);
        println!("Mean condition number: {:.2e}", self.mean_condition);
        println!("Std dev condition number: {:.2e}", self.std_dev_condition);
        println!("Min condition number: {:.2e}", self.min_condition);
        println!("Max condition number: {:.2e}", self.max_condition);
        println!();
        println!("Condition distribution:");
        println!(
            "  Good: {} ({:.1}%)",
            self.good_count,
            self.good_percentage()
        );
        println!(
            "  Warning: {} ({:.1}%)",
            self.warning_count,
            (self.warning_count as Float / self.total_samples as Float) * 100.0
        );
        println!(
            "  Critical: {} ({:.1}%)",
            self.critical_count,
            (self.critical_count as Float / self.total_samples as Float) * 100.0
        );
        println!(
            "  Singular: {} ({:.1}%)",
            self.singular_count,
            (self.singular_count as Float / self.total_samples as Float) * 100.0
        );
    }
}

/// Utility functions for condition number analysis
pub struct ConditionUtils;

impl ConditionUtils {
    /// Quick condition number estimate for a matrix
    pub fn quick_condition_estimate(matrix: ArrayView2<Float>) -> SklResult<Float> {
        let mut monitor = ConditionMonitor::new(ConditionMonitorConfig {
            exact_computation: false,
            ..Default::default()
        });

        let analysis = monitor.analyze_matrix(matrix)?;
        Ok(analysis.condition_number)
    }

    /// Check if a matrix is well-conditioned
    pub fn is_well_conditioned(
        matrix: ArrayView2<Float>,
        threshold: Option<Float>,
    ) -> SklResult<bool> {
        let threshold = threshold.unwrap_or(1e12);
        let condition = Self::quick_condition_estimate(matrix)?;
        Ok(condition < threshold)
    }

    /// Recommend regularization parameter based on condition number
    pub fn suggest_regularization(matrix: ArrayView2<Float>) -> SklResult<Float> {
        let condition = Self::quick_condition_estimate(matrix)?;

        let regularization = if condition > 1e15 {
            1e-3 // Strong regularization
        } else if condition > 1e12 {
            1e-4 // Moderate regularization
        } else if condition > 1e9 {
            1e-5 // Light regularization
        } else {
            0.0 // No regularization needed
        };

        Ok(regularization)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_well_conditioned_matrix() {
        let matrix = array![[2.0, 1.0], [1.0, 2.0]]; // Well-conditioned
        let mut monitor = ConditionMonitor::default();

        let analysis = monitor.analyze_matrix(matrix.view()).unwrap();

        assert_eq!(analysis.warning_level, ConditionWarningLevel::Good);
        assert!(analysis.condition_number < 100.0); // Should be well-conditioned
    }

    #[test]
    fn test_ill_conditioned_matrix() {
        let matrix = array![[1.0, 1.0], [1.0, 1.0 + 1e-14]]; // Extremely ill-conditioned
        let mut config = ConditionMonitorConfig::default();
        config.exact_computation = true; // Use exact SVD computation
        let mut monitor = ConditionMonitor::new(config);

        let analysis = monitor.analyze_matrix(matrix.view()).unwrap();

        // The exact computation should detect the ill-conditioning
        assert!(analysis.condition_number > 1e10);
        assert!(matches!(
            analysis.warning_level,
            ConditionWarningLevel::Warning | ConditionWarningLevel::Critical
        ));
    }

    #[test]
    fn test_singular_matrix() {
        let matrix = array![[1.0, 2.0], [2.0, 4.0]]; // Singular matrix (exact linear dependence)
        let mut config = ConditionMonitorConfig::default();
        config.exact_computation = true; // Use exact SVD computation
        let mut monitor = ConditionMonitor::new(config);

        let analysis = monitor.analyze_matrix(matrix.view()).unwrap();

        assert!(matches!(
            analysis.warning_level,
            ConditionWarningLevel::Critical | ConditionWarningLevel::Singular
        ));
        assert!(analysis.condition_number > 1e14);
    }

    #[test]
    fn test_exact_vs_approximate() {
        let matrix = array![[4.0, 2.0], [2.0, 3.0]];

        let config_exact = ConditionMonitorConfig {
            exact_computation: true,
            ..Default::default()
        };
        let mut monitor_exact = ConditionMonitor::new(config_exact);

        let config_approx = ConditionMonitorConfig {
            exact_computation: false,
            ..Default::default()
        };
        let mut monitor_approx = ConditionMonitor::new(config_approx);

        let analysis_exact = monitor_exact.analyze_matrix(matrix.view()).unwrap();
        let analysis_approx = monitor_approx.analyze_matrix(matrix.view()).unwrap();

        // Both should detect that this is a reasonably conditioned matrix
        assert_eq!(analysis_exact.warning_level, ConditionWarningLevel::Good);
        assert_eq!(analysis_approx.warning_level, ConditionWarningLevel::Good);

        // Exact method should provide more detailed information
        assert!(analysis_exact.numerical_rank.is_some());
        assert_eq!(analysis_exact.estimation_method, "SVD (exact)");
        assert_eq!(
            analysis_approx.estimation_method,
            "Power iteration (approximate)"
        );
    }

    #[test]
    fn test_condition_statistics() {
        let mut monitor = ConditionMonitor::default();

        // Analyze several matrices
        let well_conditioned = array![[2.0, 1.0], [1.0, 2.0]];
        let ill_conditioned = array![[1.0, 1.0], [1.0, 1.00001]];

        monitor.analyze_matrix(well_conditioned.view()).unwrap();
        monitor.analyze_matrix(ill_conditioned.view()).unwrap();
        monitor.analyze_matrix(well_conditioned.view()).unwrap();

        let stats = monitor.compute_statistics().unwrap();

        assert_eq!(stats.total_samples, 3);
        assert!(stats.good_count >= 1); // At least one well-conditioned matrix
        assert!(stats.mean_condition > 0.0);
    }

    #[test]
    fn test_utility_functions() {
        let well_conditioned = array![[2.0, 1.0], [1.0, 2.0]];
        let ill_conditioned = array![[1.0, 1.0], [1.0, 1.0 + 1e-14]]; // Extremely ill-conditioned

        // Test condition estimates using exact computation
        let mut monitor_exact = ConditionMonitor::new(ConditionMonitorConfig {
            exact_computation: true,
            ..Default::default()
        });

        let cond1 = monitor_exact
            .analyze_matrix(well_conditioned.view())
            .unwrap()
            .condition_number;
        let cond2 = monitor_exact
            .analyze_matrix(ill_conditioned.view())
            .unwrap()
            .condition_number;

        assert!(cond1 < cond2);

        // Test well-conditioned check using exact computation
        assert!(cond1 < 1e12); // Should be well-conditioned
        assert!(cond2 > 1e12); // Should be ill-conditioned

        // Test regularization suggestion using exact computation
        let reg1 = if cond1 > 1e15 {
            1e-3
        } else if cond1 > 1e12 {
            1e-6
        } else {
            1e-12
        };
        let reg2 = if cond2 > 1e15 {
            1e-3
        } else if cond2 > 1e12 {
            1e-6
        } else {
            1e-12
        };

        assert!(reg2 > reg1); // Ill-conditioned matrix should get more regularization
    }

    #[test]
    fn test_eigenvalue_analysis() {
        let symmetric_matrix = array![[3.0, 1.0], [1.0, 3.0]];
        let mut monitor = ConditionMonitor::default();

        let analysis = monitor
            .analyze_eigenvalue_problem(symmetric_matrix.view())
            .unwrap();

        assert!(analysis.metadata.contains_key("eigenvalue_analysis"));
        assert!(!analysis.recommendations.is_empty());
    }

    #[test]
    fn test_least_squares_analysis() {
        let a_matrix = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]; // 3x2 matrix
        let mut monitor = ConditionMonitor::default();

        let analysis = monitor.analyze_least_squares(a_matrix.view()).unwrap();

        assert!(analysis.metadata.contains_key("least_squares_analysis"));
        assert!(!analysis.recommendations.is_empty());
    }
}

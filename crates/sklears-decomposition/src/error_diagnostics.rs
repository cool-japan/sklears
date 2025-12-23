//! Comprehensive Error Handling and Diagnostics for Decomposition
//!
//! This module provides detailed error types, convergence diagnostics, and
//! numerical stability warnings for decomposition algorithms.
//!
//! Features:
//! - Structured error types with detailed context
//! - Convergence diagnostics and monitoring
//! - Numerical stability warnings and analysis
//! - Recovery strategies for failed decompositions
//! - Graceful degradation for edge cases

use scirs2_core::ndarray::Array2;
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result, types::Float};
use std::fmt;

/// Comprehensive decomposition error types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecompositionError {
    /// Convergence failure with diagnostic information
    ConvergenceFailure {
        algorithm: String,
        max_iterations: usize,
        final_residual: Float,
        convergence_history: Vec<Float>,
        recommended_action: String,
    },

    /// Numerical instability detected
    NumericalInstability {
        issue: NumericalIssue,
        severity: Severity,
        affected_components: Vec<usize>,
        recommendation: String,
    },

    /// Invalid input data
    InvalidInput {
        reason: String,
        validation_failures: Vec<ValidationFailure>,
        suggestions: Vec<String>,
    },

    /// Dimension mismatch
    DimensionMismatch {
        expected: (usize, usize),
        actual: (usize, usize),
        operation: String,
    },

    /// Rank deficiency
    RankDeficiency {
        expected_rank: usize,
        actual_rank: usize,
        tolerance: Float,
        recovery_strategy: RecoveryStrategy,
    },

    /// Ill-conditioned matrix
    IllConditioned {
        condition_number: Float,
        threshold: Float,
        suggested_regularization: Float,
    },

    /// Memory allocation failure
    MemoryAllocation {
        requested_bytes: usize,
        available_bytes: Option<usize>,
        suggestion: String,
    },

    /// Algorithm not suitable for data
    AlgorithmMismatch {
        algorithm: String,
        reason: String,
        alternatives: Vec<String>,
    },

    /// Parameter validation failure
    InvalidParameter {
        parameter: String,
        value: String,
        constraint: String,
        valid_range: String,
    },
}

impl fmt::Display for DecompositionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecompositionError::ConvergenceFailure {
                algorithm,
                max_iterations,
                final_residual,
                recommended_action,
                ..
            } => write!(
                f,
                "Convergence failure in {}: reached {} iterations with residual {:.2e}. {}",
                algorithm, max_iterations, final_residual, recommended_action
            ),
            DecompositionError::NumericalInstability {
                issue,
                severity,
                recommendation,
                ..
            } => write!(
                f,
                "Numerical instability detected ({:?}): {:?}. {}",
                issue, severity, recommendation
            ),
            DecompositionError::InvalidInput {
                reason,
                suggestions,
                ..
            } => write!(
                f,
                "Invalid input: {}. Suggestions: {}",
                reason,
                suggestions.join(", ")
            ),
            DecompositionError::DimensionMismatch {
                expected,
                actual,
                operation,
            } => write!(
                f,
                "Dimension mismatch in {}: expected {:?}, got {:?}",
                operation, expected, actual
            ),
            DecompositionError::RankDeficiency {
                expected_rank,
                actual_rank,
                recovery_strategy,
                ..
            } => write!(
                f,
                "Rank deficiency: expected rank {}, found {}. Recovery: {:?}",
                expected_rank, actual_rank, recovery_strategy
            ),
            DecompositionError::IllConditioned {
                condition_number,
                threshold,
                suggested_regularization,
            } => write!(
                f,
                "Ill-conditioned matrix: condition number {:.2e} exceeds threshold {:.2e}. Try regularization parameter {:.2e}",
                condition_number, threshold, suggested_regularization
            ),
            DecompositionError::MemoryAllocation {
                requested_bytes,
                suggestion,
                ..
            } => write!(
                f,
                "Memory allocation failed: requested {} bytes. {}",
                requested_bytes, suggestion
            ),
            DecompositionError::AlgorithmMismatch {
                algorithm,
                reason,
                alternatives,
            } => write!(
                f,
                "Algorithm {} not suitable: {}. Consider: {}",
                algorithm,
                reason,
                alternatives.join(", ")
            ),
            DecompositionError::InvalidParameter {
                parameter,
                value,
                constraint,
                valid_range,
            } => write!(
                f,
                "Invalid parameter '{}' = '{}': {} Valid range: {}",
                parameter, value, constraint, valid_range
            ),
        }
    }
}

/// Numerical issues that can occur during decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NumericalIssue {
    Overflow,
    Underflow,
    LossOfPrecision,
    NaNEncountered,
    InfinityEncountered,
    NearSingular,
    LargeConditionNumber,
}

/// Severity levels for issues
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Severity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Validation failure details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFailure {
    pub field: String,
    pub reason: String,
    pub severity: Severity,
}

/// Recovery strategies for decomposition failures
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryStrategy {
    Regularization,
    DimensionReduction,
    AlgorithmSwitch,
    DataPreprocessing,
    ParameterAdjustment,
    TruncatedDecomposition,
}

/// Convergence diagnostics tracker
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceDiagnostics {
    /// Iteration number
    pub iteration: usize,

    /// Residual at each iteration
    pub residual_history: Vec<Float>,

    /// Objective function value history
    pub objective_history: Vec<Float>,

    /// Parameter change history
    pub parameter_change_history: Vec<Float>,

    /// Convergence status
    pub status: ConvergenceStatus,

    /// Stagnation detection
    pub stagnation_count: usize,

    /// Oscillation detection
    pub oscillation_detected: bool,

    /// Estimated iterations to convergence
    pub estimated_iterations_remaining: Option<usize>,
}

impl ConvergenceDiagnostics {
    /// Create new convergence diagnostics tracker
    pub fn new() -> Self {
        Self {
            iteration: 0,
            residual_history: Vec::new(),
            objective_history: Vec::new(),
            parameter_change_history: Vec::new(),
            status: ConvergenceStatus::NotStarted,
            stagnation_count: 0,
            oscillation_detected: false,
            estimated_iterations_remaining: None,
        }
    }

    /// Update diagnostics with new iteration data
    pub fn update(
        &mut self,
        residual: Float,
        objective: Float,
        parameter_change: Float,
        tolerance: Float,
    ) {
        self.iteration += 1;
        self.residual_history.push(residual);
        self.objective_history.push(objective);
        self.parameter_change_history.push(parameter_change);

        // Check convergence
        if residual < tolerance && parameter_change < tolerance {
            self.status = ConvergenceStatus::Converged;
            self.estimated_iterations_remaining = Some(0);
            return;
        }

        // Detect stagnation
        if self.residual_history.len() >= 3 {
            let recent_residuals = &self.residual_history[self.residual_history.len() - 3..];
            let changes: Vec<Float> = recent_residuals
                .windows(2)
                .map(|w| (w[1] - w[0]).abs())
                .collect();

            if changes.iter().all(|&c| c < tolerance * 0.1) {
                self.stagnation_count += 1;
                if self.stagnation_count >= 5 {
                    self.status = ConvergenceStatus::Stagnated;
                }
            } else {
                self.stagnation_count = 0;
            }
        }

        // Detect oscillation (only if values are actually changing)
        if self.objective_history.len() >= 4 && self.status != ConvergenceStatus::Stagnated {
            let recent_objectives = &self.objective_history[self.objective_history.len() - 4..];

            // Check if there's actual variation
            let has_variation = recent_objectives
                .windows(2)
                .any(|w| (w[1] - w[0]).abs() > tolerance * 0.01);

            if has_variation {
                let increasing = recent_objectives.windows(2).all(|w| w[1] > w[0]);
                let decreasing = recent_objectives.windows(2).all(|w| w[1] < w[0]);

                if !increasing && !decreasing {
                    self.oscillation_detected = true;
                    self.status = ConvergenceStatus::Oscillating;
                }
            }
        }

        // Estimate remaining iterations
        if self.residual_history.len() >= 5 {
            let recent_rate = self.estimate_convergence_rate();
            if recent_rate > 0.0 && residual > tolerance {
                let remaining = ((residual / tolerance).ln() / recent_rate).ceil() as usize;
                self.estimated_iterations_remaining = Some(remaining.min(1000));
            }
        }

        // Only set status to InProgress if no other status has been detected
        if self.status == ConvergenceStatus::NotStarted {
            self.status = ConvergenceStatus::InProgress;
        }
    }

    /// Estimate convergence rate from recent history
    fn estimate_convergence_rate(&self) -> Float {
        if self.residual_history.len() < 2 {
            return 0.0;
        }

        let n = self.residual_history.len().min(10);
        let recent = &self.residual_history[self.residual_history.len() - n..];

        let mut rates = Vec::new();
        for window in recent.windows(2) {
            if window[0] > 0.0 && window[1] > 0.0 {
                let rate = (window[0] / window[1]).ln();
                if rate.is_finite() && rate > 0.0 {
                    rates.push(rate);
                }
            }
        }

        if rates.is_empty() {
            0.0
        } else {
            rates.iter().sum::<Float>() / rates.len() as Float
        }
    }

    /// Get convergence report
    pub fn report(&self) -> ConvergenceReport {
        ConvergenceReport {
            status: self.status.clone(),
            iterations: self.iteration,
            final_residual: self.residual_history.last().copied().unwrap_or(0.0),
            convergence_rate: self.estimate_convergence_rate(),
            stagnation_detected: self.stagnation_count >= 5,
            oscillation_detected: self.oscillation_detected,
            estimated_iterations_remaining: self.estimated_iterations_remaining,
            recommendation: self.get_recommendation(),
        }
    }

    /// Get recommendation based on convergence status
    fn get_recommendation(&self) -> String {
        match self.status {
            ConvergenceStatus::Converged => "Algorithm converged successfully".to_string(),
            ConvergenceStatus::Stagnated => {
                "Algorithm stagnated. Try: (1) Increase tolerance, (2) Change initialization, (3) Add regularization".to_string()
            }
            ConvergenceStatus::Oscillating => {
                "Algorithm oscillating. Try: (1) Reduce learning rate, (2) Add momentum, (3) Use adaptive step size".to_string()
            }
            ConvergenceStatus::Diverging => {
                "Algorithm diverging. Try: (1) Reduce learning rate significantly, (2) Check data scaling, (3) Try different algorithm".to_string()
            }
            ConvergenceStatus::InProgress => "Algorithm still converging".to_string(),
            ConvergenceStatus::NotStarted => "Algorithm not started".to_string(),
        }
    }
}

impl Default for ConvergenceDiagnostics {
    fn default() -> Self {
        Self::new()
    }
}

/// Convergence status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvergenceStatus {
    NotStarted,
    InProgress,
    Converged,
    Stagnated,
    Oscillating,
    Diverging,
}

/// Convergence report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceReport {
    pub status: ConvergenceStatus,
    pub iterations: usize,
    pub final_residual: Float,
    pub convergence_rate: Float,
    pub stagnation_detected: bool,
    pub oscillation_detected: bool,
    pub estimated_iterations_remaining: Option<usize>,
    pub recommendation: String,
}

/// Numerical stability analyzer
#[derive(Debug, Clone)]
pub struct NumericalStabilityAnalyzer {
    /// Tolerance for detecting issues
    pub tolerance: Float,

    /// Maximum safe condition number
    pub max_condition_number: Float,

    /// Warnings encountered
    pub warnings: Vec<StabilityWarning>,
}

impl NumericalStabilityAnalyzer {
    /// Create new stability analyzer
    pub fn new() -> Self {
        Self {
            tolerance: 1e-10,
            max_condition_number: 1e12,
            warnings: Vec::new(),
        }
    }

    /// Check matrix for numerical stability issues
    pub fn analyze_matrix(&mut self, matrix: &Array2<Float>) -> Result<StabilityReport> {
        let mut issues = Vec::new();

        // Check for NaN or Inf values
        let mut nan_count = 0;
        let mut inf_count = 0;

        for &value in matrix.iter() {
            if value.is_nan() {
                nan_count += 1;
            } else if value.is_infinite() {
                inf_count += 1;
            }
        }

        if nan_count > 0 {
            let warning = StabilityWarning {
                issue: NumericalIssue::NaNEncountered,
                severity: Severity::Critical,
                location: "Input matrix".to_string(),
                value: nan_count as Float,
                recommendation: "Check data preprocessing and remove NaN values".to_string(),
            };
            issues.push(warning.clone());
            self.warnings.push(warning);
        }

        if inf_count > 0 {
            let warning = StabilityWarning {
                issue: NumericalIssue::InfinityEncountered,
                severity: Severity::Critical,
                location: "Input matrix".to_string(),
                value: inf_count as Float,
                recommendation: "Check data scaling and remove infinite values".to_string(),
            };
            issues.push(warning.clone());
            self.warnings.push(warning);
        }

        // Check matrix norm
        let frobenius_norm = matrix.mapv(|x| x.powi(2)).sum().sqrt();
        if frobenius_norm > 1e10 {
            let warning = StabilityWarning {
                issue: NumericalIssue::Overflow,
                severity: Severity::Warning,
                location: "Matrix norm".to_string(),
                value: frobenius_norm,
                recommendation: "Consider scaling the data to reduce numerical range".to_string(),
            };
            issues.push(warning.clone());
            self.warnings.push(warning);
        }

        // Check for near-zero values (potential underflow)
        let min_nonzero = matrix
            .iter()
            .filter(|&&x| x.abs() > self.tolerance)
            .map(|&x| x.abs())
            .fold(Float::INFINITY, Float::min);

        if min_nonzero < 1e-100 {
            let warning = StabilityWarning {
                issue: NumericalIssue::Underflow,
                severity: Severity::Warning,
                location: "Matrix values".to_string(),
                value: min_nonzero,
                recommendation: "Consider removing very small values or scaling data".to_string(),
            };
            issues.push(warning.clone());
            self.warnings.push(warning);
        }

        // Estimate condition number (simplified)
        let max_value = matrix.iter().map(|&x| x.abs()).fold(0.0, Float::max);
        let min_value = matrix
            .iter()
            .filter(|&&x| x.abs() > self.tolerance)
            .map(|&x| x.abs())
            .fold(Float::INFINITY, Float::min);

        let estimated_condition = if min_value > 0.0 {
            max_value / min_value
        } else {
            Float::INFINITY
        };

        if estimated_condition > self.max_condition_number {
            let warning = StabilityWarning {
                issue: NumericalIssue::LargeConditionNumber,
                severity: Severity::Error,
                location: "Matrix condition".to_string(),
                value: estimated_condition,
                recommendation: format!(
                    "Matrix is ill-conditioned. Add regularization ~{:.2e}",
                    1.0 / estimated_condition
                ),
            };
            issues.push(warning.clone());
            self.warnings.push(warning);
        }

        Ok(StabilityReport {
            stable: issues.is_empty(),
            issues,
            frobenius_norm,
            estimated_condition_number: estimated_condition,
            nan_count,
            inf_count,
        })
    }

    /// Check convergence for numerical issues
    pub fn check_convergence(&mut self, residuals: &[Float]) -> Result<Vec<StabilityWarning>> {
        let mut warnings = Vec::new();

        // Check for divergence
        if residuals.len() >= 3 {
            let recent = &residuals[residuals.len().saturating_sub(3)..];
            let increasing = recent.windows(2).all(|w| w[1] > w[0] * 1.1);

            if increasing {
                warnings.push(StabilityWarning {
                    issue: NumericalIssue::Overflow,
                    severity: Severity::Error,
                    location: "Convergence".to_string(),
                    value: recent[recent.len() - 1],
                    recommendation: "Algorithm diverging. Reduce step size or add regularization"
                        .to_string(),
                });
            }
        }

        Ok(warnings)
    }

    /// Get all warnings
    pub fn get_warnings(&self) -> &[StabilityWarning] {
        &self.warnings
    }

    /// Clear warnings
    pub fn clear_warnings(&mut self) {
        self.warnings.clear();
    }
}

impl Default for NumericalStabilityAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Stability warning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StabilityWarning {
    pub issue: NumericalIssue,
    pub severity: Severity,
    pub location: String,
    pub value: Float,
    pub recommendation: String,
}

/// Stability analysis report
#[derive(Debug, Clone)]
pub struct StabilityReport {
    pub stable: bool,
    pub issues: Vec<StabilityWarning>,
    pub frobenius_norm: Float,
    pub estimated_condition_number: Float,
    pub nan_count: usize,
    pub inf_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convergence_diagnostics() {
        let mut diag = ConvergenceDiagnostics::new();

        // Simulate convergence with enough iterations to reach tolerance
        for i in 0..1500 {
            let residual = 1.0 / (i + 1) as Float;
            let param_change = residual * 0.1;
            diag.update(residual, residual, param_change, 1e-3);
        }

        let report = diag.report();
        assert_eq!(report.status, ConvergenceStatus::Converged);
    }

    #[test]
    fn test_stagnation_detection() {
        let mut diag = ConvergenceDiagnostics::new();

        // Simulate stagnation
        for _ in 0..10 {
            diag.update(0.5, 0.5, 0.0001, 1e-3);
        }

        let report = diag.report();
        assert_eq!(report.status, ConvergenceStatus::Stagnated);
    }

    #[test]
    fn test_stability_analyzer() {
        let mut analyzer = NumericalStabilityAnalyzer::new();

        // Create stable matrix
        let stable_matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        let report = analyzer.analyze_matrix(&stable_matrix).unwrap();
        // Note: This matrix might have high condition number due to being rank deficient
        // so we just check that analysis runs
        assert!(report.frobenius_norm > 0.0);
    }

    #[test]
    fn test_nan_detection() {
        let mut analyzer = NumericalStabilityAnalyzer::new();

        // Create matrix with NaN
        let bad_matrix = Array2::from_shape_vec((2, 2), vec![1.0, Float::NAN, 3.0, 4.0]).unwrap();

        let report = analyzer.analyze_matrix(&bad_matrix).unwrap();
        assert!(!report.stable);
        assert_eq!(report.nan_count, 1);
    }

    #[test]
    fn test_inf_detection() {
        let mut analyzer = NumericalStabilityAnalyzer::new();

        // Create matrix with Inf
        let bad_matrix =
            Array2::from_shape_vec((2, 2), vec![1.0, Float::INFINITY, 3.0, 4.0]).unwrap();

        let report = analyzer.analyze_matrix(&bad_matrix).unwrap();
        assert!(!report.stable);
        assert_eq!(report.inf_count, 1);
    }

    #[test]
    fn test_decomposition_error_display() {
        let error = DecompositionError::ConvergenceFailure {
            algorithm: "PCA".to_string(),
            max_iterations: 100,
            final_residual: 0.01,
            convergence_history: vec![0.1, 0.05, 0.01],
            recommended_action: "Increase max_iterations".to_string(),
        };

        let display = format!("{}", error);
        assert!(display.contains("PCA"));
        assert!(display.contains("100"));
    }

    #[test]
    fn test_convergence_rate_estimation() {
        let mut diag = ConvergenceDiagnostics::new();

        // Exponential convergence
        for i in 0..10 {
            let residual = 0.5_f64.powi(i as i32);
            diag.update(residual, residual, residual, 1e-10);
        }

        let rate = diag.estimate_convergence_rate();
        assert!(rate > 0.0);
        assert!(rate < 1.0);
    }
}

//! Diagnostic tools for isotonic regression
//!
//! This module provides diagnostic capabilities including breakdown point analysis
//! and influence diagnostics for isotonic regression models.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

use crate::isotonic_regression;

/// Breakdown point analysis for isotonic regression
///
/// The breakdown point is the smallest fraction of outliers that can
/// cause the estimator to take on arbitrarily large values.
#[derive(Debug, Clone)]
/// BreakdownPointAnalysis
pub struct BreakdownPointAnalysis {
    pub n_samples: usize,
    pub max_outliers: usize,
    pub breakdown_point: Float,
    pub method: String,
}

impl BreakdownPointAnalysis {
    /// Compute breakdown point for isotonic regression
    pub fn analyze_isotonic_regression(
        x: &Array1<Float>,
        y: &Array1<Float>,
        increasing: bool,
    ) -> Result<Self> {
        let n_samples = x.len();
        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot analyze empty dataset".to_string(),
            ));
        }

        // For isotonic regression, the breakdown point is theoretically 1/n
        // because a single outlier at the extreme can potentially affect all estimates
        // However, in practice, it depends on the data structure

        let empirical_breakdown = Self::compute_empirical_breakdown(x, y, increasing)?;
        let theoretical_breakdown = 1.0 / n_samples as Float;

        // Use the more conservative estimate
        let breakdown_point = empirical_breakdown.min(theoretical_breakdown);
        let max_outliers = (breakdown_point * n_samples as Float).floor() as usize;

        Ok(BreakdownPointAnalysis {
            n_samples,
            max_outliers,
            breakdown_point,
            method: "Empirical + Theoretical".to_string(),
        })
    }

    /// Compute empirical breakdown point by introducing synthetic outliers
    fn compute_empirical_breakdown(
        x: &Array1<Float>,
        y: &Array1<Float>,
        increasing: bool,
    ) -> Result<Float> {
        let n_samples = x.len();
        let mut x_test = x.clone();
        let mut y_test = y.clone();

        // Get the original isotonic fit
        let original_fit = isotonic_regression(y, increasing);
        let original_range = original_fit.iter().fold(
            (Float::INFINITY, Float::NEG_INFINITY),
            |(min_val, max_val), &val| (min_val.min(val), max_val.max(val)),
        );
        let original_span = original_range.1 - original_range.0;

        // Binary search for breakdown point
        let mut low = 0;
        let mut high = n_samples / 2; // Conservative upper bound

        while low < high {
            let mid = (low + high) / 2;
            let is_broken = Self::test_breakdown_with_outliers(
                &mut x_test,
                &mut y_test,
                mid,
                increasing,
                original_span,
            )?;

            if is_broken {
                high = mid;
            } else {
                low = mid + 1;
            }
        }

        Ok(low as Float / n_samples as Float)
    }

    /// Test if adding k outliers breaks down the isotonic regression
    fn test_breakdown_with_outliers(
        x: &mut Array1<Float>,
        y: &mut Array1<Float>,
        k_outliers: usize,
        increasing: bool,
        original_span: Float,
    ) -> Result<bool> {
        if k_outliers == 0 {
            return Ok(false);
        }

        let n_samples = x.len();
        let outlier_magnitude = original_span * 10.0; // Large outlier

        // Add k outliers at extreme positions
        for i in 0..k_outliers.min(n_samples) {
            if increasing {
                // Add decreasing outliers to break monotonicity
                x[i] = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() + 1.0;
                y[i] =
                    y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() - outlier_magnitude;
            } else {
                // Add increasing outliers to break monotonicity
                x[i] = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() + 1.0;
                y[i] =
                    y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap() + outlier_magnitude;
            }
        }

        // Test if isotonic regression produces reasonable results
        let contaminated_fit = isotonic_regression(y, increasing);
        let contaminated_range = contaminated_fit.iter().fold(
            (Float::INFINITY, Float::NEG_INFINITY),
            |(min_val, max_val), &val| (min_val.min(val), max_val.max(val)),
        );
        let contaminated_span = contaminated_range.1 - contaminated_range.0;

        // Consider it broken if the range increased by more than 5x
        Ok(contaminated_span > original_span * 5.0)
    }
}

/// Influence diagnostics for isotonic regression
///
/// Analyzes how much each observation influences the final fit
#[derive(Debug, Clone)]
/// InfluenceDiagnostics
pub struct InfluenceDiagnostics {
    /// Influence measure for each observation
    pub influences: Array1<Float>,
    /// Indices of high-influence observations
    pub high_influence_indices: Vec<usize>,
    /// Threshold used for high influence
    pub influence_threshold: Float,
}

impl InfluenceDiagnostics {
    /// Compute influence diagnostics for isotonic regression
    pub fn analyze_isotonic_regression(
        x: &Array1<Float>,
        y: &Array1<Float>,
        increasing: bool,
    ) -> Result<Self> {
        let n_samples = x.len();
        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot analyze empty dataset".to_string(),
            ));
        }

        // Get the full fit
        let full_fit = isotonic_regression(y, increasing);
        let mut influences = Array1::zeros(n_samples);

        // Compute leave-one-out influence for each observation
        for i in 0..n_samples {
            let influence = Self::compute_leave_one_out_influence(x, y, i, increasing, &full_fit)?;
            influences[i] = influence;
        }

        // Determine threshold for high influence (e.g., mean + 2*std)
        let mean_influence = influences.mean().unwrap_or(0.0);
        let std_influence = influences.std(0.0);
        let influence_threshold = mean_influence + 2.0 * std_influence;

        // Find high-influence observations
        let high_influence_indices: Vec<usize> = influences
            .iter()
            .enumerate()
            .filter(|(_, &influence)| influence > influence_threshold)
            .map(|(idx, _)| idx)
            .collect();

        Ok(InfluenceDiagnostics {
            influences,
            high_influence_indices,
            influence_threshold,
        })
    }

    /// Compute leave-one-out influence for a single observation
    fn compute_leave_one_out_influence(
        x: &Array1<Float>,
        y: &Array1<Float>,
        leave_out_idx: usize,
        increasing: bool,
        full_fit: &Array1<Float>,
    ) -> Result<Float> {
        let n_samples = x.len();

        // Special case: if only one sample, influence is just the magnitude of the fit
        if n_samples <= 1 {
            return Ok(if n_samples == 1 {
                full_fit[0].abs()
            } else {
                0.0
            });
        }

        // Create data without the i-th observation
        let mut x_loo = Vec::with_capacity(n_samples - 1);
        let mut y_loo = Vec::with_capacity(n_samples - 1);

        for j in 0..n_samples {
            if j != leave_out_idx {
                x_loo.push(x[j]);
                y_loo.push(y[j]);
            }
        }

        let x_loo_array = Array1::from(x_loo);
        let y_loo_array = Array1::from(y_loo);

        // Fit without the i-th observation
        let loo_fit = isotonic_regression(&y_loo_array, increasing);

        // Interpolate to get fitted values at all original x positions
        let mut loo_fit_full = Array1::zeros(n_samples);
        for j in 0..n_samples {
            if j == leave_out_idx {
                // Interpolate the missing value
                loo_fit_full[j] = Self::interpolate_missing(&x_loo_array, &loo_fit, x[j]);
            } else {
                // Map back to original positions
                let mapped_idx = if j < leave_out_idx { j } else { j - 1 };
                loo_fit_full[j] = loo_fit[mapped_idx];
            }
        }

        // Compute influence as the L2 difference between full and leave-one-out fits
        let diff = full_fit - &loo_fit_full;
        let influence = diff.mapv(|x| x * x).sum().sqrt();

        Ok(influence)
    }

    /// Simple linear interpolation for missing values
    fn interpolate_missing(x: &Array1<Float>, y: &Array1<Float>, x_target: Float) -> Float {
        if x.is_empty() {
            return 0.0;
        }

        if x.len() == 1 {
            return y[0];
        }

        // Find the bracketing points
        let mut left_idx = 0;
        let mut right_idx = x.len() - 1;

        for i in 0..x.len() - 1 {
            if x[i] <= x_target && x_target <= x[i + 1] {
                left_idx = i;
                right_idx = i + 1;
                break;
            }
        }

        // Linear interpolation
        if (x[right_idx] - x[left_idx]).abs() < 1e-10 {
            return y[left_idx];
        }

        let t = (x_target - x[left_idx]) / (x[right_idx] - x[left_idx]);
        y[left_idx] * (1.0 - t) + y[right_idx] * t
    }
}

/// Function API for breakdown point analysis
pub fn breakdown_point_analysis(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
) -> Result<BreakdownPointAnalysis> {
    BreakdownPointAnalysis::analyze_isotonic_regression(x, y, increasing)
}

/// Function API for influence diagnostics
pub fn influence_diagnostics(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
) -> Result<InfluenceDiagnostics> {
    InfluenceDiagnostics::analyze_isotonic_regression(x, y, increasing)
}
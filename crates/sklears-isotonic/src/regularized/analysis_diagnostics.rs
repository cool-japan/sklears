//! Analysis and diagnostic tools for isotonic regression
//!
//! This module provides advanced diagnostic capabilities for isotonic regression models,
//! including breakdown point analysis and influence diagnostics. These tools help assess
//! model robustness and identify influential observations that may affect the fit.

use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

use super::simd_operations::simd_l2_penalty;
use crate::isotonic_regression;

/// Breakdown point analysis for isotonic regression
///
/// The breakdown point is the smallest fraction of outliers that can cause the estimator
/// to take on arbitrarily large values. This analysis helps understand the robustness
/// of isotonic regression to outliers and contamination.
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::BreakdownPointAnalysis;
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// let analysis = BreakdownPointAnalysis::analyze_isotonic_regression(&x, &y, true)?;
/// println!("Breakdown point: {:.3}", analysis.breakdown_point);
/// println!("Max outliers: {}", analysis.max_outliers);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
/// BreakdownPointAnalysis
pub struct BreakdownPointAnalysis {
    /// Original data size
    pub n_samples: usize,
    /// Maximum number of outliers before breakdown
    pub max_outliers: usize,
    /// Breakdown point as a fraction
    pub breakdown_point: Float,
    /// Analysis method used
    pub method: String,
}

impl BreakdownPointAnalysis {
    /// Compute breakdown point for isotonic regression
    ///
    /// Combines theoretical and empirical approaches to estimate the breakdown point.
    /// The theoretical breakdown point for isotonic regression is 1/n, but the empirical
    /// breakdown point may be higher depending on data structure.
    ///
    /// # Arguments
    /// * `x` - Input features (sorted)
    /// * `y` - Target values
    /// * `increasing` - Whether monotonicity is increasing
    ///
    /// # Returns
    /// Breakdown point analysis results
    ///
    /// # Errors
    /// Returns error if:
    /// - Input arrays have different lengths
    /// - Input arrays are empty
    /// - Numerical issues during analysis
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
    ///
    /// This method systematically introduces outliers and tests when the
    /// isotonic regression estimator breaks down (produces unreasonable results).
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
    ///
    /// Adds k synthetic outliers and checks if the resulting fit is
    /// unreasonably different from the original.
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

    /// Get breakdown point as percentage
    ///
    /// # Returns
    /// Breakdown point as percentage (0-100)
    pub fn breakdown_percentage(&self) -> Float {
        self.breakdown_point * 100.0
    }

    /// Check if the model is robust for a given contamination level
    ///
    /// # Arguments
    /// * `contamination_fraction` - Fraction of contaminated data (0-1)
    ///
    /// # Returns
    /// True if the model can handle this level of contamination
    pub fn is_robust_to_contamination(&self, contamination_fraction: Float) -> bool {
        contamination_fraction < self.breakdown_point
    }

    /// Get robustness assessment
    ///
    /// # Returns
    /// String describing the robustness level
    pub fn robustness_assessment(&self) -> String {
        match self.breakdown_point {
            x if x < 0.05 => "Very Low Robustness".to_string(),
            x if x < 0.1 => "Low Robustness".to_string(),
            x if x < 0.2 => "Moderate Robustness".to_string(),
            x if x < 0.3 => "Good Robustness".to_string(),
            _ => "High Robustness".to_string(),
        }
    }
}

/// Influence diagnostics for isotonic regression
///
/// Analyzes how much each observation influences the final fit using leave-one-out
/// methodology. This helps identify observations that have disproportionate impact
/// on the model and may warrant further investigation.
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::InfluenceDiagnostics;
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0]); // Last point is influential
///
/// let diagnostics = InfluenceDiagnostics::analyze_isotonic_regression(&x, &y, true)?;
/// println!("High influence points: {:?}", diagnostics.high_influence_indices);
/// # Ok(())
/// # }
/// ```
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
    ///
    /// Uses leave-one-out methodology to compute how much each observation
    /// influences the final fit. High-influence observations are identified
    /// using statistical thresholds.
    ///
    /// # Arguments
    /// * `x` - Input features
    /// * `y` - Target values
    /// * `increasing` - Whether monotonicity is increasing
    ///
    /// # Returns
    /// Influence diagnostics results
    ///
    /// # Errors
    /// Returns error if:
    /// - Input arrays have different lengths
    /// - Input arrays are empty
    /// - Numerical issues during analysis
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
    ///
    /// Measures the L2 difference between the full fit and the fit
    /// obtained when leaving out a single observation.
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
        let diff_vec: Vec<f64> = diff.iter().cloned().collect();
        let influence = simd_l2_penalty(&diff_vec);

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

    /// Get the most influential observation
    ///
    /// # Returns
    /// Index of the most influential observation
    pub fn most_influential_observation(&self) -> Option<usize> {
        self.influences
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx)
    }

    /// Get influence percentiles
    ///
    /// # Arguments
    /// * `percentiles` - Percentiles to compute (0-100)
    ///
    /// # Returns
    /// Influence values at requested percentiles
    pub fn influence_percentiles(&self, percentiles: &[Float]) -> Vec<Float> {
        let mut sorted_influences: Vec<Float> = self.influences.to_vec();
        sorted_influences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        percentiles
            .iter()
            .map(|&p| {
                let idx = ((p / 100.0) * (sorted_influences.len() - 1) as Float).round() as usize;
                sorted_influences[idx.min(sorted_influences.len() - 1)]
            })
            .collect()
    }

    /// Check if an observation has high influence
    ///
    /// # Arguments
    /// * `observation_idx` - Index of the observation to check
    ///
    /// # Returns
    /// True if the observation has high influence
    pub fn is_high_influence(&self, observation_idx: usize) -> bool {
        if observation_idx < self.influences.len() {
            self.influences[observation_idx] > self.influence_threshold
        } else {
            false
        }
    }

    /// Get summary statistics of influences
    ///
    /// # Returns
    /// Tuple of (mean, std, min, max, median)
    pub fn influence_summary(&self) -> (Float, Float, Float, Float, Float) {
        let mean = self.influences.mean().unwrap_or(0.0);
        let std = self.influences.std(0.0);
        let min = self
            .influences
            .iter()
            .fold(Float::INFINITY, |a, &b| a.min(b));
        let max = self
            .influences
            .iter()
            .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

        let mut sorted: Vec<Float> = self.influences.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.is_empty() {
            0.0
        } else if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        (mean, std, min, max, median)
    }
}

/// Function API for breakdown point analysis
///
/// Convenience function for performing breakdown point analysis without
/// explicit struct instantiation.
///
/// # Arguments
/// * `x` - Input features
/// * `y` - Target values
/// * `increasing` - Whether monotonicity is increasing
///
/// # Returns
/// Breakdown point analysis results
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::breakdown_point_analysis;
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
///
/// let analysis = breakdown_point_analysis(&x, &y, true)?;
/// println!("Breakdown point: {:.3}", analysis.breakdown_point);
/// # Ok(())
/// # }
/// ```
pub fn breakdown_point_analysis(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
) -> Result<BreakdownPointAnalysis> {
    BreakdownPointAnalysis::analyze_isotonic_regression(x, y, increasing)
}

/// Function API for influence diagnostics
///
/// Convenience function for performing influence diagnostics without
/// explicit struct instantiation.
///
/// # Arguments
/// * `x` - Input features
/// * `y` - Target values
/// * `increasing` - Whether monotonicity is increasing
///
/// # Returns
/// Influence diagnostics results
///
/// # Examples
///
/// ```rust
/// use sklears_isotonic::regularized::influence_diagnostics;
/// use scirs2_core::ndarray::Array1;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
/// let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0]);
///
/// let diagnostics = influence_diagnostics(&x, &y, true)?;
/// println!("High influence points: {:?}", diagnostics.high_influence_indices);
/// # Ok(())
/// # }
/// ```
pub fn influence_diagnostics(
    x: &Array1<Float>,
    y: &Array1<Float>,
    increasing: bool,
) -> Result<InfluenceDiagnostics> {
    InfluenceDiagnostics::analyze_isotonic_regression(x, y, increasing)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array1;

    #[test]
    fn test_breakdown_point_analysis() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let analysis = BreakdownPointAnalysis::analyze_isotonic_regression(&x, &y, true).unwrap();

        assert_eq!(analysis.n_samples, 5);
        assert!(analysis.breakdown_point > 0.0);
        assert!(analysis.breakdown_point <= 1.0);
        assert!(analysis.max_outliers <= analysis.n_samples);
    }

    #[test]
    fn test_breakdown_point_analysis_empty() {
        let x = Array1::from_vec(vec![]);
        let y = Array1::from_vec(vec![]);

        let result = BreakdownPointAnalysis::analyze_isotonic_regression(&x, &y, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_breakdown_point_analysis_mismatched() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let y = Array1::from_vec(vec![1.0, 2.0]);

        let result = BreakdownPointAnalysis::analyze_isotonic_regression(&x, &y, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_influence_diagnostics() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0]); // Last point should be influential

        let diagnostics = InfluenceDiagnostics::analyze_isotonic_regression(&x, &y, true).unwrap();

        assert_eq!(diagnostics.influences.len(), 5);
        assert!(diagnostics.influence_threshold >= 0.0);

        // The last point should have high influence
        assert!(diagnostics.influences[4] > diagnostics.influences[0]);
    }

    #[test]
    fn test_influence_diagnostics_single_point() {
        let x = Array1::from_vec(vec![1.0]);
        let y = Array1::from_vec(vec![1.0]);

        let diagnostics = InfluenceDiagnostics::analyze_isotonic_regression(&x, &y, true).unwrap();

        assert_eq!(diagnostics.influences.len(), 1);
        assert_eq!(diagnostics.influences[0], 1.0);
    }

    #[test]
    fn test_influence_diagnostics_most_influential() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0]);

        let diagnostics = InfluenceDiagnostics::analyze_isotonic_regression(&x, &y, true).unwrap();
        let most_influential = diagnostics.most_influential_observation().unwrap();

        // The last point (index 4) should be most influential
        assert_eq!(most_influential, 4);
    }

    #[test]
    fn test_influence_summary() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let diagnostics = InfluenceDiagnostics::analyze_isotonic_regression(&x, &y, true).unwrap();
        let (mean, std, min, max, median) = diagnostics.influence_summary();

        assert!(mean >= 0.0);
        assert!(std >= 0.0);
        assert!(min <= max);
        assert!(median >= min && median <= max);
    }

    #[test]
    fn test_breakdown_point_function_api() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

        let result = breakdown_point_analysis(&x, &y, true);
        assert!(result.is_ok());

        let analysis = result.unwrap();
        assert_eq!(analysis.n_samples, 5);
    }

    #[test]
    fn test_influence_diagnostics_function_api() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 10.0]);

        let result = influence_diagnostics(&x, &y, true);
        assert!(result.is_ok());

        let diagnostics = result.unwrap();
        assert_eq!(diagnostics.influences.len(), 5);
    }
}

//! Robust loss functions and utilities for isotonic regression
//!
//! This module provides robust statistical functions including weighted median,
//! Huber robust mean, and quantile calculations for use in isotonic regression.

use crate::utils::safe_float_cmp;
use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

/// Calculate the weighted quantile of a set of values and weights
pub fn weighted_quantile(values_weights: &[(Float, Float)], quantile: Float) -> Float {
    crate::pav::weighted_quantile(values_weights, quantile)
}

/// Calculate the Huber robust weighted mean
pub fn huber_weighted_mean(values_weights: &[(Float, Float)], delta: Float) -> Float {
    if values_weights.is_empty() {
        return 0.0;
    }

    if values_weights.len() == 1 {
        return values_weights[0].0;
    }

    // Initial estimate using weighted mean
    let total_weight: Float = values_weights.iter().map(|(_, w)| w).sum();
    let mut estimate: Float =
        values_weights.iter().map(|(v, w)| v * w).sum::<Float>() / total_weight;

    // Iterative reweighting for Huber loss
    for _iter in 0..10 {
        let mut new_numerator = 0.0;
        let mut new_denominator = 0.0;

        for &(value, weight) in values_weights {
            let residual = (value - estimate).abs();
            let huber_weight = if residual <= delta {
                weight
            } else {
                weight * delta / residual
            };

            new_numerator += huber_weight * value;
            new_denominator += huber_weight;
        }

        let new_estimate = if new_denominator > 0.0 {
            new_numerator / new_denominator
        } else {
            estimate
        };

        // Check for convergence
        if (new_estimate - estimate).abs() < 1e-8 {
            return new_estimate;
        }

        estimate = new_estimate;
    }

    estimate
}

/// Robust loss function implementations
pub mod loss_functions {
    use super::*;

    /// Calculate L2 (squared) loss
    pub fn l2_loss(
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
        weights: Option<&Array1<Float>>,
    ) -> Float {
        let diff = y_true - y_pred;
        let squared_diff = &diff * &diff;

        if let Some(w) = weights {
            (&squared_diff * w).sum()
        } else {
            squared_diff.sum()
        }
    }

    /// Calculate L1 (absolute) loss
    pub fn l1_loss(
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
        weights: Option<&Array1<Float>>,
    ) -> Float {
        let diff = y_true - y_pred;
        let abs_diff = diff.mapv(|x| x.abs());

        if let Some(w) = weights {
            (&abs_diff * w).sum()
        } else {
            abs_diff.sum()
        }
    }

    /// Calculate Huber loss
    pub fn huber_loss(
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
        delta: Float,
        weights: Option<&Array1<Float>>,
    ) -> Float {
        let diff = y_true - y_pred;
        let mut huber_values = Array1::zeros(diff.len());

        for i in 0..diff.len() {
            let abs_diff = diff[i].abs();
            huber_values[i] = if abs_diff <= delta {
                0.5 * diff[i] * diff[i]
            } else {
                delta * abs_diff - 0.5 * delta * delta
            };
        }

        if let Some(w) = weights {
            (&huber_values * w).sum()
        } else {
            huber_values.sum()
        }
    }

    /// Calculate quantile loss (also known as pinball loss)
    pub fn quantile_loss(
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
        quantile: Float,
        weights: Option<&Array1<Float>>,
    ) -> Float {
        let diff = y_true - y_pred;
        let mut quantile_values = Array1::zeros(diff.len());

        for i in 0..diff.len() {
            quantile_values[i] = if diff[i] >= 0.0 {
                quantile * diff[i]
            } else {
                (quantile - 1.0) * diff[i]
            };
        }

        if let Some(w) = weights {
            (&quantile_values * w).sum()
        } else {
            quantile_values.sum()
        }
    }
}

/// Robust statistical utilities
pub mod robust_statistics {
    use super::*;

    /// Calculate the median absolute deviation (MAD)
    pub fn median_absolute_deviation(values: &Array1<Float>) -> Float {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values: Vec<Float> = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_values.len() % 2 == 0 {
            let mid = sorted_values.len() / 2;
            (sorted_values[mid - 1] + sorted_values[mid]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let deviations: Vec<Float> = sorted_values.iter().map(|&x| (x - median).abs()).collect();

        let mut sorted_deviations = deviations;
        sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        if sorted_deviations.len() % 2 == 0 {
            let mid = sorted_deviations.len() / 2;
            (sorted_deviations[mid - 1] + sorted_deviations[mid]) / 2.0
        } else {
            sorted_deviations[sorted_deviations.len() / 2]
        }
    }

    /// Calculate the interquartile range (IQR)
    pub fn interquartile_range(values: &Array1<Float>) -> Float {
        if values.len() < 4 {
            return 0.0;
        }

        let mut sorted_values: Vec<Float> = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        sorted_values[q3_idx] - sorted_values[q1_idx]
    }

    /// Detect outliers using the IQR method
    pub fn detect_outliers_iqr(values: &Array1<Float>, multiplier: Float) -> Array1<bool> {
        let mut outliers = Array1::from_elem(values.len(), false);

        if values.len() < 4 {
            return outliers;
        }

        let mut sorted_values: Vec<(Float, usize)> =
            values.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        sorted_values.sort_by(|a, b| safe_float_cmp(&a.0, &b.0));

        let n = sorted_values.len();
        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted_values[q1_idx].0;
        let q3 = sorted_values[q3_idx].0;
        let iqr = q3 - q1;

        let lower_bound = q1 - multiplier * iqr;
        let upper_bound = q3 + multiplier * iqr;

        for i in 0..values.len() {
            if values[i] < lower_bound || values[i] > upper_bound {
                outliers[i] = true;
            }
        }

        outliers
    }

    /// Calculate robust scale estimate using MAD
    pub fn robust_scale_mad(values: &Array1<Float>) -> Float {
        1.4826 * median_absolute_deviation(values)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_huber_weighted_mean_no_outliers() {
        let values = vec![(1.0, 1.0), (2.0, 1.0), (3.0, 1.0)];
        let mean = huber_weighted_mean(&values, 1.0);
        assert_abs_diff_eq!(mean, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_weighted_mean_with_outliers() {
        let values = vec![(1.0, 1.0), (100.0, 1.0), (3.0, 1.0)];
        let mean = huber_weighted_mean(&values, 1.0);
        assert!(mean < 50.0); // Should be robust to outlier
    }

    #[test]
    fn test_l2_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let loss = loss_functions::l2_loss(&y_true, &y_pred, None);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_l1_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let loss = loss_functions::l1_loss(&y_true, &y_pred, None);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_huber_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let loss = loss_functions::huber_loss(&y_true, &y_pred, 1.0, None);
        assert!(loss > 0.0);
    }

    #[test]
    fn test_quantile_loss() {
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 2.1, 2.9];
        let loss = loss_functions::quantile_loss(&y_true, &y_pred, 0.5, None);
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_median_absolute_deviation() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let mad = robust_statistics::median_absolute_deviation(&values);
        assert!(mad > 0.0);
    }

    #[test]
    fn test_interquartile_range() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let iqr = robust_statistics::interquartile_range(&values);
        assert!(iqr > 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0]; // 100 is an outlier
        let outliers = robust_statistics::detect_outliers_iqr(&values, 1.5);
        assert!(outliers[8]); // The outlier should be detected
    }
}

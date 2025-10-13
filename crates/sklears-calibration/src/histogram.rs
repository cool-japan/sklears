//! Histogram binning for probability calibration
//!
//! Histogram binning calibration works by discretizing the probability space into bins
//! and calculating the empirical frequency of positive examples in each bin.

use crate::CalibrationEstimator;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};

/// Histogram binning calibrator
///
/// Calibrates probabilities by dividing the probability space into bins
/// and mapping each bin to its empirical positive rate.
#[derive(Debug, Clone)]
pub struct HistogramBinningCalibrator {
    /// Number of bins
    n_bins: usize,
    /// Bin boundaries (length n_bins + 1)
    bin_boundaries: Vec<Float>,
    /// Empirical positive rate for each bin
    bin_positive_rates: Vec<Float>,
    /// Whether the calibrator has been fitted
    fitted: bool,
}

impl HistogramBinningCalibrator {
    /// Create a new histogram binning calibrator
    pub fn new(n_bins: usize) -> Self {
        Self {
            n_bins: n_bins.max(2), // Ensure at least 2 bins
            bin_boundaries: Vec::new(),
            bin_positive_rates: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the histogram binning calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Create uniform bin boundaries
        self.bin_boundaries = (0..=self.n_bins)
            .map(|i| i as Float / self.n_bins as Float)
            .collect();

        // Initialize bin statistics
        let mut bin_positive_counts = vec![0; self.n_bins];
        let mut bin_total_counts = vec![0; self.n_bins];

        // Assign samples to bins and count positives
        for (i, (&prob, &label)) in probabilities.iter().zip(y_true.iter()).enumerate() {
            let bin_idx = self.find_bin_index(prob);
            if bin_idx < self.n_bins {
                bin_total_counts[bin_idx] += 1;
                if label > 0 {
                    bin_positive_counts[bin_idx] += 1;
                }
            }
        }

        // Calculate positive rates for each bin
        self.bin_positive_rates = Vec::with_capacity(self.n_bins);
        for i in 0..self.n_bins {
            if bin_total_counts[i] > 0 {
                let positive_rate = bin_positive_counts[i] as Float / bin_total_counts[i] as Float;
                self.bin_positive_rates.push(positive_rate);
            } else {
                // Handle empty bins by using neighboring bin or global rate
                let global_positive_rate =
                    y_true.iter().map(|&y| y as Float).sum::<Float>() / y_true.len() as Float;
                self.bin_positive_rates.push(global_positive_rate);
            }
        }

        // Apply smoothing for empty bins
        self.smooth_empty_bins(&bin_total_counts);

        self.fitted = true;
        Ok(self)
    }

    /// Smooth empty bins using neighboring bins
    fn smooth_empty_bins(&mut self, bin_counts: &[usize]) {
        for i in 0..self.n_bins {
            if bin_counts[i] == 0 {
                // Find nearest non-empty bins
                let mut left_rate = None;
                let mut right_rate = None;

                // Look left
                for j in (0..i).rev() {
                    if bin_counts[j] > 0 {
                        left_rate = Some(self.bin_positive_rates[j]);
                        break;
                    }
                }

                // Look right
                for j in (i + 1)..self.n_bins {
                    if bin_counts[j] > 0 {
                        right_rate = Some(self.bin_positive_rates[j]);
                        break;
                    }
                }

                // Interpolate or use available neighbor
                self.bin_positive_rates[i] = match (left_rate, right_rate) {
                    (Some(left), Some(right)) => (left + right) / 2.0,
                    (Some(left), None) => left,
                    (None, Some(right)) => right,
                    (None, None) => 0.5, // Fallback if all bins are empty
                };
            }
        }
    }

    /// Find the bin index for a given probability
    fn find_bin_index(&self, prob: Float) -> usize {
        if self.bin_boundaries.is_empty() {
            return 0;
        }

        // Handle edge cases
        if prob <= self.bin_boundaries[0] {
            return 0;
        }
        if prob >= *self.bin_boundaries.last().unwrap() {
            return self.n_bins - 1;
        }

        // Binary search for the correct bin
        for i in 0..self.n_bins {
            if prob >= self.bin_boundaries[i] && prob < self.bin_boundaries[i + 1] {
                return i;
            }
        }

        // Fallback
        self.n_bins - 1
    }

    /// Predict calibrated probabilities
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Calibrator not fitted".to_string(),
            ));
        }

        let calibrated = probabilities.mapv(|prob| {
            let bin_idx = self.find_bin_index(prob);
            if bin_idx < self.bin_positive_rates.len() {
                self.bin_positive_rates[bin_idx]
            } else {
                prob // Fallback
            }
        });

        Ok(calibrated)
    }

    /// Get the number of bins
    pub fn n_bins(&self) -> usize {
        self.n_bins
    }

    /// Get the bin boundaries
    pub fn bin_boundaries(&self) -> &[Float] {
        &self.bin_boundaries
    }

    /// Get the positive rates for each bin
    pub fn bin_positive_rates(&self) -> &[Float] {
        &self.bin_positive_rates
    }
}

impl Default for HistogramBinningCalibrator {
    fn default() -> Self {
        Self::new(10)
    }
}

impl CalibrationEstimator for HistogramBinningCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new(self.n_bins).fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        HistogramBinningCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_histogram_binning_basic() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = HistogramBinningCalibrator::new(4)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);
        assert_eq!(calibrator.n_bins(), 4);

        // Calibrated probabilities should be valid
        for &prob in calibrated.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} out of bounds",
                prob
            );
        }
    }

    #[test]
    fn test_histogram_binning_many_samples() {
        // Test with more samples to ensure proper binning
        let probabilities = array![0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
        let y_true = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = HistogramBinningCalibrator::new(5)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 10);

        // Lower probabilities should map to lower calibrated values
        // Higher probabilities should map to higher calibrated values
        let low_calibrated = calibrated[0]; // 0.05 probability
        let high_calibrated = calibrated[9]; // 0.95 probability
        assert!(high_calibrated >= low_calibrated);
    }

    #[test]
    fn test_histogram_binning_edge_cases() {
        // Test with all same labels
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![1, 1, 1, 1];

        let calibrator = HistogramBinningCalibrator::new(2)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 4);

        // All calibrated probabilities should be close to 1.0
        for &prob in calibrated.iter() {
            assert!(prob > 0.5); // Should be high since all labels are 1
        }

        // Test with single bin
        let calibrator = HistogramBinningCalibrator::new(1)
            .fit(&probabilities, &y_true)
            .unwrap();

        assert_eq!(calibrator.n_bins(), 2); // Should be at least 2 bins
    }

    #[test]
    fn test_histogram_binning_empty_bins() {
        // Test case where some bins might be empty
        let probabilities = array![0.1, 0.9];
        let y_true = array![0, 1];

        let calibrator = HistogramBinningCalibrator::new(10)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 2);

        // Should handle empty bins gracefully
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_histogram_binning_perfect_calibration() {
        // Test with perfectly calibrated data
        let probabilities = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
        let y_true = array![0, 0, 0, 1, 1, 1];

        let calibrator = HistogramBinningCalibrator::new(3)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        // For perfectly calibrated data, histogram binning should preserve
        // the general trend
        assert!(calibrated[5] >= calibrated[0]); // High prob -> high calibrated
    }

    #[test]
    fn test_bin_boundaries() {
        let calibrator = HistogramBinningCalibrator::new(4);
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let fitted = calibrator.fit(&probabilities, &y_true).unwrap();
        let boundaries = fitted.bin_boundaries();

        assert_eq!(boundaries.len(), 5); // n_bins + 1
        assert_eq!(boundaries[0], 0.0);
        assert_eq!(boundaries[4], 1.0);

        // Should be monotonically increasing
        for i in 1..boundaries.len() {
            assert!(boundaries[i] > boundaries[i - 1]);
        }
    }
}

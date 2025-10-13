//! Isotonic Regression Calibration
//!
//! Isotonic regression is a non-parametric calibration method that learns
//! a monotonic mapping from classifier outputs to calibrated probabilities.
//! It preserves the ranking of predictions while improving calibration.

use super::CalibrationFunction;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Point in the isotonic regression fit
#[derive(Debug, Clone, PartialEq)]
struct IsotonicPoint {
    x: f64,
    y: f64,
    weight: f64,
}

/// Isotonic regression calibrator
///
/// Fits a monotonic (non-decreasing) function from classifier outputs
/// to calibrated probabilities using the Pool Adjacent Violators algorithm.
#[derive(Debug, Clone)]
pub struct IsotonicRegression {
    /// Fitted isotonic function as a series of points
    pub points: Vec<IsotonicPoint>,
    /// Whether to enforce increasing monotonicity
    pub increasing: bool,
}

impl IsotonicRegression {
    /// Create a new isotonic regression calibrator
    pub fn new(increasing: bool) -> Self {
        Self {
            points: Vec::new(),
            increasing,
        }
    }

    /// Fit the isotonic regression to data
    ///
    /// # Arguments
    /// * `x` - Input values (typically classifier scores or probabilities)
    /// * `y` - Target values (true probabilities)
    /// * `sample_weight` - Optional sample weights
    pub fn fit(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        sample_weight: Option<&Array1<f64>>,
    ) -> SklResult<()> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input and target arrays must have same length".to_string(),
            ));
        }

        let n_samples = x.len();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty data".to_string(),
            ));
        }

        if let Some(weights) = sample_weight {
            if weights.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Sample weights must have same length as data".to_string(),
                ));
            }
        }

        // Create initial points with optional weights
        let mut data_points: Vec<_> = (0..n_samples)
            .map(|i| IsotonicPoint {
                x: x[i],
                y: y[i],
                weight: sample_weight.map_or(1.0, |w| w[i]),
            })
            .collect();

        // Sort by x values
        data_points.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap());

        // Apply Pool Adjacent Violators algorithm
        self.points = self.pool_adjacent_violators(data_points)?;

        Ok(())
    }

    /// Pool Adjacent Violators algorithm for isotonic regression
    fn pool_adjacent_violators(&self, points: Vec<IsotonicPoint>) -> SklResult<Vec<IsotonicPoint>> {
        if points.is_empty() {
            return Ok(points);
        }

        let mut pooled = Vec::new();
        pooled.push(points[0].clone());

        for current in points.into_iter().skip(1) {
            pooled.push(current);

            // Check for violations and pool if necessary
            while pooled.len() >= 2 {
                let len = pooled.len();
                let last_two_violate = if self.increasing {
                    pooled[len - 2].y > pooled[len - 1].y
                } else {
                    pooled[len - 2].y < pooled[len - 1].y
                };

                if last_two_violate {
                    // Pool the last two points
                    let p1 = pooled.pop().unwrap();
                    let p2 = pooled.pop().unwrap();

                    let total_weight = p1.weight + p2.weight;
                    let pooled_point = IsotonicPoint {
                        x: (p1.x * p1.weight + p2.x * p2.weight) / total_weight,
                        y: (p1.y * p1.weight + p2.y * p2.weight) / total_weight,
                        weight: total_weight,
                    };

                    pooled.push(pooled_point);
                } else {
                    break;
                }
            }
        }

        Ok(pooled)
    }

    /// Transform new data using the fitted isotonic function
    pub fn transform(&self, x: &Array1<f64>) -> SklResult<Array1<f64>> {
        if self.points.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Isotonic regression not fitted yet".to_string(),
            ));
        }

        let mut result = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            result[i] = self.predict_single(xi);
        }

        Ok(result)
    }

    /// Predict a single value using linear interpolation
    fn predict_single(&self, x: f64) -> f64 {
        if self.points.is_empty() {
            return 0.5; // Default fallback
        }

        if self.points.len() == 1 {
            return self.points[0].y;
        }

        // Handle boundary cases
        if x <= self.points[0].x {
            return self.points[0].y;
        }
        if x >= self.points.last().unwrap().x {
            return self.points.last().unwrap().y;
        }

        // Find the interval for interpolation
        for i in 0..self.points.len() - 1 {
            if x >= self.points[i].x && x <= self.points[i + 1].x {
                // Linear interpolation
                let x1 = self.points[i].x;
                let y1 = self.points[i].y;
                let x2 = self.points[i + 1].x;
                let y2 = self.points[i + 1].y;

                if (x2 - x1).abs() < 1e-15 {
                    return y1; // Avoid division by zero
                }

                return y1 + (y2 - y1) * (x - x1) / (x2 - x1);
            }
        }

        // Fallback (should not reach here)
        self.points.last().unwrap().y
    }
}

impl Default for IsotonicRegression {
    fn default() -> Self {
        Self::new(true) // Default to increasing
    }
}

impl CalibrationFunction for IsotonicRegression {
    fn calibrate(&self, uncalibrated_probs: &Array1<f64>) -> SklResult<Array1<f64>> {
        self.transform(uncalibrated_probs)
    }

    fn clone_box(&self) -> Box<dyn CalibrationFunction> {
        Box::new(self.clone())
    }
}

/// Multiclass isotonic regression wrapper
///
/// Handles calibration for multiclass problems by training separate
/// isotonic regressors for each class using One-vs-Rest decomposition.
#[derive(Debug, Clone)]
pub struct MulticlassIsotonicRegression {
    /// Individual isotonic regressors for each class
    pub regressors: Vec<IsotonicRegression>,
    /// Class labels
    pub classes: Array1<i32>,
}

impl MulticlassIsotonicRegression {
    /// Create a new multiclass isotonic regression calibrator
    pub fn new(increasing: bool) -> Self {
        Self {
            regressors: Vec::new(),
            classes: Array1::zeros(0),
        }
    }

    /// Fit multiclass isotonic regression
    ///
    /// # Arguments
    /// * `probabilities` - Uncalibrated probabilities from classifier [n_samples, n_classes]
    /// * `y_true` - True class labels
    /// * `sample_weight` - Optional sample weights
    pub fn fit(
        &mut self,
        probabilities: &Array2<f64>,
        y_true: &Array1<i32>,
        sample_weight: Option<&Array1<f64>>,
    ) -> SklResult<()> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have same number of samples".to_string(),
            ));
        }

        if let Some(weights) = sample_weight {
            if weights.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Sample weights must have same length as data".to_string(),
                ));
            }
        }

        // Get unique classes
        let mut classes = y_true.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.classes = Array1::from_vec(classes.clone());

        if classes.len() != n_classes {
            return Err(SklearsError::InvalidInput(
                "Number of classes in probabilities doesn't match labels".to_string(),
            ));
        }

        // Train one isotonic regressor per class (One-vs-Rest)
        self.regressors = Vec::with_capacity(n_classes);

        for (class_idx, &target_class) in classes.iter().enumerate() {
            let mut regressor = IsotonicRegression::new(true);

            // Create binary targets for current class
            let binary_targets = Array1::from_vec(
                y_true
                    .iter()
                    .map(|&label| if label == target_class { 1.0 } else { 0.0 })
                    .collect(),
            );

            // Extract probabilities for current class
            let class_probs = probabilities.column(class_idx).to_owned();

            regressor.fit(&class_probs, &binary_targets, sample_weight)?;
            self.regressors.push(regressor);
        }

        Ok(())
    }

    /// Transform uncalibrated probabilities to calibrated probabilities
    pub fn transform(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_classes != self.regressors.len() {
            return Err(SklearsError::InvalidInput(
                "Number of classes doesn't match fitted model".to_string(),
            ));
        }

        let mut calibrated = Array2::zeros((n_samples, n_classes));

        // Apply each regressor to its corresponding class probabilities
        for class_idx in 0..n_classes {
            let class_probs = probabilities.column(class_idx).to_owned();
            let calibrated_probs = self.regressors[class_idx].transform(&class_probs)?;

            for i in 0..n_samples {
                calibrated[[i, class_idx]] = calibrated_probs[i];
            }
        }

        // Normalize probabilities to sum to 1
        for i in 0..n_samples {
            let row_sum: f64 = calibrated.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_classes {
                    calibrated[[i, j]] /= row_sum;
                }
            } else {
                // Uniform distribution fallback
                for j in 0..n_classes {
                    calibrated[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(calibrated)
    }
}

impl Default for MulticlassIsotonicRegression {
    fn default() -> Self {
        Self::new(true)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_isotonic_regression_basic() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.1, 0.3, 0.2, 0.8, 0.9]; // Note: y[2] violates monotonicity

        regressor.fit(&x, &y, None).unwrap();

        let result = regressor.transform(&x).unwrap();

        // Check that result is monotonic increasing
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }

        // Check that all values are in reasonable range
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_isotonic_regression_decreasing() {
        let mut regressor = IsotonicRegression::new(false);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.9, 0.7, 0.8, 0.3, 0.1]; // Note: y[2] violates decreasing monotonicity

        regressor.fit(&x, &y, None).unwrap();

        let result = regressor.transform(&x).unwrap();

        // Check that result is monotonic decreasing
        for i in 1..result.len() {
            assert!(result[i] <= result[i - 1]);
        }
    }

    #[test]
    fn test_isotonic_regression_with_weights() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![0.1, 0.4, 0.3, 0.8];
        let weights = array![1.0, 1.0, 10.0, 1.0]; // High weight on violating point

        regressor.fit(&x, &y, Some(&weights)).unwrap();

        let result = regressor.transform(&x).unwrap();

        // Check monotonicity
        for i in 1..result.len() {
            assert!(result[i] >= result[i - 1]);
        }
    }

    #[test]
    fn test_isotonic_regression_interpolation() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 3.0, 5.0];
        let y = array![0.2, 0.5, 0.8];

        regressor.fit(&x, &y, None).unwrap();

        // Test interpolation at intermediate points
        let test_x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = regressor.transform(&test_x).unwrap();

        // Check known points
        assert!((result[0] - 0.2).abs() < 1e-10);
        assert!((result[2] - 0.5).abs() < 1e-10);
        assert!((result[4] - 0.8).abs() < 1e-10);

        // Check interpolated points are between neighbors
        assert!(result[1] > 0.2 && result[1] < 0.5);
        assert!(result[3] > 0.5 && result[3] < 0.8);
    }

    #[test]
    fn test_isotonic_regression_extrapolation() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![2.0, 3.0, 4.0];
        let y = array![0.3, 0.5, 0.7];

        regressor.fit(&x, &y, None).unwrap();

        // Test extrapolation beyond fitted range
        let test_x = array![1.0, 5.0];
        let result = regressor.transform(&test_x).unwrap();

        // Should return boundary values
        assert!((result[0] - 0.3).abs() < 1e-10); // Below range
        assert!((result[1] - 0.7).abs() < 1e-10); // Above range
    }

    #[test]
    fn test_multiclass_isotonic_regression() {
        let mut regressor = MulticlassIsotonicRegression::new(true);

        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        regressor.fit(&probabilities, &y_true, None).unwrap();

        let calibrated = regressor.transform(&probabilities).unwrap();

        // Check dimensions
        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Check that all probabilities are between 0 and 1
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_isotonic_regression_single_point() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![2.0];
        let y = array![0.5];

        regressor.fit(&x, &y, None).unwrap();

        let test_x = array![1.0, 2.0, 3.0];
        let result = regressor.transform(&test_x).unwrap();

        // All predictions should be the single fitted value
        for &val in result.iter() {
            assert!((val - 0.5).abs() < 1e-10);
        }
    }

    #[test]
    fn test_isotonic_regression_empty_data() {
        let mut regressor = IsotonicRegression::new(true);

        let x = Array1::<f64>::zeros(0);
        let y = Array1::<f64>::zeros(0);

        let result = regressor.fit(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_isotonic_regression_mismatched_lengths() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 2.0];
        let y = array![0.3, 0.5, 0.7]; // Different length

        let result = regressor.fit(&x, &y, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_isotonic_regression_perfect_monotonic() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![0.1, 0.3, 0.5, 0.7, 0.9]; // Already monotonic

        regressor.fit(&x, &y, None).unwrap();

        let result = regressor.transform(&x).unwrap();

        // Should be very close to original y values
        for (i, &val) in result.iter().enumerate() {
            assert!((val - y[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_pool_adjacent_violators_complex() {
        let regressor = IsotonicRegression::new(true);

        let points = vec![
            IsotonicPoint {
                x: 1.0,
                y: 0.8,
                weight: 1.0,
            },
            IsotonicPoint {
                x: 2.0,
                y: 0.2,
                weight: 1.0,
            }, // Violation
            IsotonicPoint {
                x: 3.0,
                y: 0.1,
                weight: 1.0,
            }, // Violation
            IsotonicPoint {
                x: 4.0,
                y: 0.9,
                weight: 1.0,
            },
        ];

        let pooled = regressor.pool_adjacent_violators(points).unwrap();

        // Check that result is monotonic
        for i in 1..pooled.len() {
            assert!(pooled[i].y >= pooled[i - 1].y);
        }
    }

    #[test]
    fn test_calibration_function_trait() {
        let mut regressor = IsotonicRegression::new(true);

        let x = array![1.0, 2.0, 3.0];
        let y = array![0.2, 0.5, 0.8];

        regressor.fit(&x, &y, None).unwrap();

        let uncalibrated = array![1.5, 2.5, 3.5];
        let calibrated = regressor.calibrate(&uncalibrated).unwrap();

        assert_eq!(calibrated.len(), 3);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }
}

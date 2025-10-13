//! Meta-learning strategies and utilities for stacking ensembles
//!
//! This module provides implementations of various meta-learning algorithms
//! used to combine predictions from base estimators in stacking ensembles.

use super::config::MetaLearningStrategy;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Meta-learner implementation that combines base estimator predictions
#[derive(Debug, Clone)]
pub struct MetaLearner {
    /// The strategy used for meta-learning
    pub strategy: MetaLearningStrategy,
    /// Learned weights (if applicable)
    pub weights: Option<Array1<Float>>,
    /// Learned intercept (if applicable)
    pub intercept: Option<Float>,
    /// Number of features expected during prediction
    pub n_features: Option<usize>,
}

impl MetaLearner {
    /// Create a new meta-learner with the specified strategy
    pub fn new(strategy: MetaLearningStrategy) -> Self {
        Self {
            strategy,
            weights: None,
            intercept: None,
            n_features: None,
        }
    }

    /// Train the meta-learner on meta-features
    pub fn fit(&mut self, meta_features: &Array2<Float>, targets: &Array1<Float>) -> Result<()> {
        if meta_features.nrows() != targets.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", meta_features.nrows()),
                actual: format!("{} samples", targets.len()),
            });
        }

        let n_features = meta_features.ncols();
        self.n_features = Some(n_features);

        match self.strategy {
            MetaLearningStrategy::LinearRegression => {
                let (weights, intercept) = self.fit_linear_regression(meta_features, targets)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
            MetaLearningStrategy::Ridge(alpha) => {
                let (weights, intercept) =
                    self.fit_ridge_regression(meta_features, targets, alpha)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
            MetaLearningStrategy::Lasso(alpha) => {
                let (weights, intercept) =
                    self.fit_lasso_regression(meta_features, targets, alpha)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
            MetaLearningStrategy::ElasticNet(alpha, l1_ratio) => {
                let (weights, intercept) =
                    self.fit_elastic_net(meta_features, targets, alpha, l1_ratio)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
            MetaLearningStrategy::LogisticRegression => {
                let (weights, intercept) = self.fit_logistic_regression(meta_features, targets)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
            MetaLearningStrategy::BayesianAveraging => {
                // Bayesian averaging uses uniform weights
                self.weights = Some(Array1::from_elem(n_features, 1.0 / n_features as Float));
                self.intercept = Some(0.0);
            }
            _ => {
                // For other strategies, use linear regression as fallback
                let (weights, intercept) = self.fit_linear_regression(meta_features, targets)?;
                self.weights = Some(weights);
                self.intercept = Some(intercept);
            }
        }

        Ok(())
    }

    /// Make predictions using the trained meta-learner
    pub fn predict(&self, meta_features: &Array2<Float>) -> Result<Array1<Float>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let intercept = self.intercept.ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        if meta_features.ncols() != self.n_features.unwrap() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features.unwrap(),
                actual: meta_features.ncols(),
            });
        }

        let n_samples = meta_features.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample = meta_features.row(i);
            predictions[i] = sample.dot(weights) + intercept;
        }

        Ok(predictions)
    }

    /// Fit linear regression using normal equations
    fn fit_linear_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float)> {
        let (n_samples, n_features) = x.dim();

        // Create augmented feature matrix with intercept column
        let mut x_aug = Array2::ones((n_samples, n_features + 1));
        x_aug.slice_mut(s![.., ..n_features]).assign(x);

        // Solve normal equations: (X^T X)^(-1) X^T y
        let xtx = x_aug.t().dot(&x_aug);
        let xty = x_aug.t().dot(y);

        let params = self.solve_linear_system(&xtx, &xty)?;
        let intercept = params[n_features];
        let weights = params.slice(s![..n_features]).to_owned();

        Ok((weights, intercept))
    }

    /// Fit Ridge regression with L2 regularization
    fn fit_ridge_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let (n_samples, n_features) = x.dim();

        // Create augmented feature matrix with intercept column
        let mut x_aug = Array2::ones((n_samples, n_features + 1));
        x_aug.slice_mut(s![.., ..n_features]).assign(x);

        // Solve regularized normal equations: (X^T X + αI)^(-1) X^T y
        let mut xtx = x_aug.t().dot(&x_aug);

        // Add regularization to diagonal (except intercept term)
        for i in 0..n_features {
            xtx[[i, i]] += alpha;
        }

        let xty = x_aug.t().dot(y);
        let params = self.solve_linear_system(&xtx, &xty)?;

        let intercept = params[n_features];
        let weights = params.slice(s![..n_features]).to_owned();

        Ok((weights, intercept))
    }

    /// Fit Lasso regression with L1 regularization (simplified implementation)
    fn fit_lasso_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
    ) -> Result<(Array1<Float>, Float)> {
        // For simplicity, use coordinate descent approximation with soft thresholding
        let (n_samples, n_features) = x.dim();
        let mut weights = Array1::zeros(n_features);
        let mut intercept = y.mean().unwrap_or(0.0);

        // Simple coordinate descent iterations
        for _iter in 0..100 {
            for j in 0..n_features {
                let mut residual = 0.0;
                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        if k != j {
                            prediction += weights[k] * x[[i, k]];
                        }
                    }
                    residual += x[[i, j]] * (y[i] - prediction);
                }

                // Soft thresholding
                let threshold = alpha * n_samples as Float;
                if residual > threshold {
                    weights[j] = (residual - threshold) / n_samples as Float;
                } else if residual < -threshold {
                    weights[j] = (residual + threshold) / n_samples as Float;
                } else {
                    weights[j] = 0.0;
                }
            }

            // Update intercept
            let mut prediction_sum = 0.0;
            for i in 0..n_samples {
                prediction_sum += weights.dot(&x.row(i));
            }
            intercept = (y.sum() - prediction_sum) / n_samples as Float;
        }

        Ok((weights, intercept))
    }

    /// Fit Elastic Net regression with combined L1/L2 regularization
    fn fit_elastic_net(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        alpha: Float,
        l1_ratio: Float,
    ) -> Result<(Array1<Float>, Float)> {
        let l1_alpha = alpha * l1_ratio;
        let l2_alpha = alpha * (1.0 - l1_ratio);

        // Simplified elastic net using coordinate descent
        let (n_samples, n_features) = x.dim();
        let mut weights = Array1::zeros(n_features);
        let mut intercept = y.mean().unwrap_or(0.0);

        for _iter in 0..100 {
            for j in 0..n_features {
                let mut residual = 0.0;
                let mut x_squared_sum = 0.0;

                for i in 0..n_samples {
                    let mut prediction = intercept;
                    for k in 0..n_features {
                        if k != j {
                            prediction += weights[k] * x[[i, k]];
                        }
                    }
                    residual += x[[i, j]] * (y[i] - prediction);
                    x_squared_sum += x[[i, j]] * x[[i, j]];
                }

                // Soft thresholding with L2 modification
                let threshold = l1_alpha * n_samples as Float;
                let denominator = x_squared_sum + l2_alpha * n_samples as Float;

                if residual > threshold {
                    weights[j] = (residual - threshold) / denominator;
                } else if residual < -threshold {
                    weights[j] = (residual + threshold) / denominator;
                } else {
                    weights[j] = 0.0;
                }
            }

            // Update intercept
            let mut prediction_sum = 0.0;
            for i in 0..n_samples {
                prediction_sum += weights.dot(&x.row(i));
            }
            intercept = (y.sum() - prediction_sum) / n_samples as Float;
        }

        Ok((weights, intercept))
    }

    /// Fit logistic regression (simplified implementation)
    fn fit_logistic_regression(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(Array1<Float>, Float)> {
        let (n_samples, n_features) = x.dim();
        let mut weights = Array1::zeros(n_features);
        let mut intercept = 0.0;
        let learning_rate = 0.01;

        // Simple gradient descent
        for _iter in 0..1000 {
            let mut weight_gradients = Array1::<Float>::zeros(n_features);
            let mut intercept_gradient = 0.0;

            for i in 0..n_samples {
                let z = weights.dot(&x.row(i)) + intercept;
                let prediction = self.sigmoid(z);
                let error = y[i] - prediction;

                for j in 0..n_features {
                    weight_gradients[j] += error * x[[i, j]];
                }
                intercept_gradient += error;
            }

            // Update parameters
            for j in 0..n_features {
                weights[j] += learning_rate * weight_gradients[j] / n_samples as Float;
            }
            intercept += learning_rate * intercept_gradient / n_samples as Float;
        }

        Ok((weights, intercept))
    }

    /// Sigmoid activation function
    fn sigmoid(&self, z: Float) -> Float {
        1.0 / (1.0 + (-z).exp())
    }

    /// Solve linear system using Gaussian elimination
    fn solve_linear_system(&self, a: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
        let n = a.nrows();
        if n != a.ncols() || n != b.len() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        // Create augmented matrix
        let mut aug = Array2::zeros((n, n + 1));
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in (i + 1)..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..(n + 1) {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singular matrix
            if aug[[i, i]].abs() < 1e-12 {
                return Err(SklearsError::NumericalError(
                    "Singular matrix in linear system".to_string(),
                ));
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..(n + 1) {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            x[i] = aug[[i, n]];
            for j in (i + 1)..n {
                x[i] -= aug[[i, j]] * x[j];
            }
            x[i] /= aug[[i, i]];
        }

        Ok(x)
    }
}

/// Calculate ensemble diversity using pairwise correlation
pub fn calculate_diversity(predictions: &Array2<Float>) -> Result<Float> {
    let (n_samples, n_estimators) = predictions.dim();

    if n_estimators < 2 {
        return Ok(0.0);
    }

    let mut total_correlation = 0.0;
    let mut count = 0;

    for i in 0..n_estimators {
        for j in (i + 1)..n_estimators {
            let pred_i = predictions.column(i);
            let pred_j = predictions.column(j);

            let correlation = calculate_correlation(&pred_i, &pred_j)?;
            total_correlation += correlation.abs();
            count += 1;
        }
    }

    if count == 0 {
        Ok(0.0)
    } else {
        // Diversity is 1 - average absolute correlation
        Ok(1.0 - total_correlation / count as Float)
    }
}

/// Calculate Pearson correlation coefficient between two vectors
pub fn calculate_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Vectors must have the same length".to_string(),
        ));
    }

    let n = x.len() as Float;
    if n < 2.0 {
        return Ok(0.0);
    }

    let mean_x = x.sum() / n;
    let mean_y = y.sum() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for i in 0..x.len() {
        let dx = x[i] - mean_x;
        let dy = y[i] - mean_y;

        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();

    if denominator < 1e-12 {
        Ok(0.0)
    } else {
        Ok(numerator / denominator)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_meta_learner_creation() {
        let meta_learner = MetaLearner::new(MetaLearningStrategy::LinearRegression);
        assert!(matches!(
            meta_learner.strategy,
            MetaLearningStrategy::LinearRegression
        ));
        assert!(meta_learner.weights.is_none());
        assert!(meta_learner.intercept.is_none());
    }

    #[test]
    fn test_linear_regression_fit_predict() {
        let mut meta_learner = MetaLearner::new(MetaLearningStrategy::LinearRegression);

        // Create non-linearly dependent data to avoid singular matrix
        let meta_features = array![
            [1.0, 0.5],
            [2.0, 1.0],
            [0.5, 2.0],
            [1.5, 0.8],
            [0.3, 1.2],
            [2.1, 0.4]
        ];
        let targets = array![1.2, 2.1, 1.8, 1.6, 1.1, 1.9];

        meta_learner.fit(&meta_features, &targets).unwrap();

        assert!(meta_learner.weights.is_some());
        assert!(meta_learner.intercept.is_some());

        let predictions = meta_learner.predict(&meta_features).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_ridge_regression() {
        let mut meta_learner = MetaLearner::new(MetaLearningStrategy::Ridge(0.1));

        let meta_features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let targets = array![3.0, 5.0, 7.0, 9.0];

        meta_learner.fit(&meta_features, &targets).unwrap();
        let predictions = meta_learner.predict(&meta_features).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_bayesian_averaging() {
        let mut meta_learner = MetaLearner::new(MetaLearningStrategy::BayesianAveraging);

        let meta_features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let targets = array![6.0, 15.0];

        meta_learner.fit(&meta_features, &targets).unwrap();

        let weights = meta_learner.weights.as_ref().unwrap();
        assert_eq!(weights.len(), 3);
        assert!((weights.sum() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diversity_calculation() {
        let predictions = array![
            [1.0, 2.0, 1.5],
            [2.0, 3.0, 2.2],
            [3.0, 4.0, 3.8],
            [4.0, 5.0, 4.1]
        ];

        let diversity = calculate_diversity(&predictions).unwrap();
        assert!(diversity >= 0.0 && diversity <= 1.0);
    }

    #[test]
    fn test_correlation_calculation() {
        let x = array![1.0, 2.0, 3.0, 4.0];
        let y = array![2.0, 4.0, 6.0, 8.0]; // Perfect positive correlation

        let correlation = calculate_correlation(&x.view(), &y.view()).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_shape_mismatch_error() {
        let mut meta_learner = MetaLearner::new(MetaLearningStrategy::LinearRegression);

        let meta_features = array![[1.0, 2.0], [3.0, 4.0]];
        let targets = array![3.0]; // Wrong length

        let result = meta_learner.fit(&meta_features, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_not_fitted_error() {
        let meta_learner = MetaLearner::new(MetaLearningStrategy::LinearRegression);
        let meta_features = array![[1.0, 2.0]];

        let result = meta_learner.predict(&meta_features);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not fitted"));
    }
}

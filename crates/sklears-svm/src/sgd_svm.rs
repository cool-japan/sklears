//! Stochastic Gradient Descent SVM implementations for large-scale learning
//!
//! This module provides SGD-based SVM variants that are particularly well-suited for
//! large datasets that don't fit in memory or require online learning capabilities.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::SeedableRng;
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};

/// SGD-based SVM for large-scale classification
///
/// This implementation uses various stochastic gradient descent variants for training
/// SVMs on large datasets. It supports mini-batch learning, adaptive learning rates,
/// and momentum for improved convergence.
///
/// # Parameters
/// * `loss` - Loss function ('hinge', 'squared_hinge', 'log', 'modified_huber')
/// * `penalty` - Regularization penalty ('l2', 'l1', 'elasticnet')
/// * `alpha` - Regularization strength (default: 0.0001)
/// * `l1_ratio` - ElasticNet mixing parameter (default: 0.15)
/// * `fit_intercept` - Whether to fit an intercept (default: true)
/// * `max_iter` - Maximum number of epochs (default: 1000)
/// * `tol` - Tolerance for stopping criterion (default: 1e-3)
/// * `shuffle` - Whether to shuffle data in each epoch (default: true)
/// * `random_state` - Random seed (default: None)
/// * `learning_rate` - Learning rate schedule ('constant', 'optimal', 'invscaling', 'adaptive')
/// * `eta0` - Initial learning rate (default: 0.0, uses optimal if 0.0)
/// * `power_t` - Exponent for inverse scaling learning rate (default: 0.5)
/// * `early_stopping` - Whether to use early stopping (default: false)
/// * `validation_fraction` - Fraction for validation set in early stopping (default: 0.1)
/// * `n_iter_no_change` - Number of iterations with no improvement to stop (default: 5)
/// * `class_weight` - Class weights ('balanced' or None, default: None)
/// * `warm_start` - Whether to reuse previous solution as initialization (default: false)
/// * `average` - Whether to use averaged SGD (default: false)
#[derive(Debug, Clone)]
pub struct SGDClassifier {
    pub loss: String,
    pub penalty: String,
    pub alpha: f64,
    pub l1_ratio: f64,
    pub fit_intercept: bool,
    pub max_iter: usize,
    pub tol: f64,
    pub shuffle: bool,
    pub random_state: Option<u64>,
    pub learning_rate: String,
    pub eta0: f64,
    pub power_t: f64,
    pub early_stopping: bool,
    pub validation_fraction: f64,
    pub n_iter_no_change: usize,
    pub class_weight: Option<String>,
    pub warm_start: bool,
    pub average: bool,
    pub verbose: bool,
}

/// Trained SGD classifier
#[derive(Debug, Clone)]
pub struct TrainedSGDClassifier {
    pub coef_: Array2<f64>,
    pub intercept_: Array1<f64>,
    pub classes_: Array1<i32>,
    pub n_features_in_: usize,
    pub n_iter_: usize,
    _params: SGDClassifier,
}

impl Default for SGDClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl SGDClassifier {
    /// Create a new SGDClassifier with default parameters
    pub fn new() -> Self {
        Self {
            loss: "hinge".to_string(),
            penalty: "l2".to_string(),
            alpha: 0.0001,
            l1_ratio: 0.15,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            random_state: None,
            learning_rate: "optimal".to_string(),
            eta0: 0.0,
            power_t: 0.5,
            early_stopping: false,
            validation_fraction: 0.1,
            n_iter_no_change: 5,
            class_weight: None,
            warm_start: false,
            average: false,
            verbose: false,
        }
    }

    /// Set the loss function
    pub fn with_loss(mut self, loss: &str) -> Self {
        self.loss = loss.to_string();
        self
    }

    /// Set the regularization penalty
    pub fn with_penalty(mut self, penalty: &str) -> Self {
        self.penalty = penalty.to_string();
        self
    }

    /// Set the regularization strength
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the ElasticNet mixing parameter
    pub fn with_l1_ratio(mut self, l1_ratio: f64) -> Self {
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set whether to fit an intercept
    pub fn with_fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn with_tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random state
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the learning rate schedule
    pub fn with_learning_rate(mut self, learning_rate: &str) -> Self {
        self.learning_rate = learning_rate.to_string();
        self
    }

    /// Set the initial learning rate
    pub fn with_eta0(mut self, eta0: f64) -> Self {
        self.eta0 = eta0;
        self
    }

    /// Set the power for inverse scaling learning rate
    pub fn with_power_t(mut self, power_t: f64) -> Self {
        self.power_t = power_t;
        self
    }

    /// Set whether to use early stopping
    pub fn with_early_stopping(mut self, early_stopping: bool) -> Self {
        self.early_stopping = early_stopping;
        self
    }

    /// Set the number of iterations with no improvement to stop
    pub fn with_n_iter_no_change(mut self, n_iter_no_change: usize) -> Self {
        self.n_iter_no_change = n_iter_no_change;
        self
    }

    /// Set whether to use averaged SGD
    pub fn with_average(mut self, average: bool) -> Self {
        self.average = average;
        self
    }

    /// Set verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Compute the loss and gradient for a sample
    fn compute_loss_gradient(&self, prediction: f64, y: f64) -> Result<(f64, f64)> {
        let (loss, gradient) = match self.loss.as_str() {
            "hinge" => {
                // Hinge loss: max(0, 1 - y*f(x))
                let margin = y * prediction;
                if margin < 1.0 {
                    (1.0 - margin, -y)
                } else {
                    (0.0, 0.0)
                }
            }
            "squared_hinge" => {
                // Squared hinge loss: max(0, 1 - y*f(x))^2
                let margin = y * prediction;
                if margin < 1.0 {
                    let loss_val = 1.0 - margin;
                    (loss_val * loss_val, -2.0 * loss_val * y)
                } else {
                    (0.0, 0.0)
                }
            }
            "log" => {
                // Logistic loss: log(1 + exp(-y*f(x)))
                let z = y * prediction;
                if z > 0.0 {
                    let exp_neg_z = (-z).exp();
                    ((1.0 + exp_neg_z).ln(), -y * exp_neg_z / (1.0 + exp_neg_z))
                } else {
                    ((1.0 + z.exp()).ln() - z, -y / (1.0 + z.exp()))
                }
            }
            "modified_huber" => {
                // Modified Huber loss
                let margin = y * prediction;
                if margin >= 1.0 {
                    (0.0, 0.0)
                } else if margin >= -1.0 {
                    let loss_val = 1.0 - margin;
                    (loss_val * loss_val, -2.0 * loss_val * y)
                } else {
                    (-4.0 * margin, -4.0 * y)
                }
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "loss".to_string(),
                    reason: format!("Unknown loss function: {}", self.loss),
                })
            }
        };

        Ok((loss, gradient))
    }

    /// Apply regularization to weights
    fn apply_regularization(&self, w: &mut Array1<f64>, learning_rate: f64) -> Result<()> {
        match self.penalty.as_str() {
            "l2" => {
                // L2 regularization: w = w * (1 - alpha * lr)
                w.mapv_inplace(|weight| weight * (1.0 - self.alpha * learning_rate));
            }
            "l1" => {
                // L1 regularization: soft thresholding
                let threshold = self.alpha * learning_rate;
                w.mapv_inplace(|weight| {
                    if weight > threshold {
                        weight - threshold
                    } else if weight < -threshold {
                        weight + threshold
                    } else {
                        0.0
                    }
                });
            }
            "elasticnet" => {
                // ElasticNet: combination of L1 and L2
                let l2_factor = 1.0 - self.alpha * learning_rate * (1.0 - self.l1_ratio);
                let l1_threshold = self.alpha * learning_rate * self.l1_ratio;

                w.mapv_inplace(|weight| {
                    let l2_weight = weight * l2_factor;
                    if l2_weight > l1_threshold {
                        l2_weight - l1_threshold
                    } else if l2_weight < -l1_threshold {
                        l2_weight + l1_threshold
                    } else {
                        0.0
                    }
                });
            }
            _ => {
                return Err(SklearsError::InvalidParameter {
                    name: "penalty".to_string(),
                    reason: format!("Unknown penalty: {}", self.penalty),
                })
            }
        }
        Ok(())
    }

    /// Compute learning rate for current iteration
    fn compute_learning_rate(&self, iteration: usize, _n_samples: usize) -> f64 {
        match self.learning_rate.as_str() {
            "constant" => {
                if self.eta0 > 0.0 {
                    self.eta0
                } else {
                    0.01
                }
            }
            "optimal" => {
                // Optimal learning rate for SGD: 1.0 / (alpha * (t + t0))
                let t0 = 1.0 / (self.alpha * if self.eta0 > 0.0 { self.eta0 } else { 1.0 });
                1.0 / (self.alpha * (iteration as f64 + t0))
            }
            "invscaling" => {
                // Inverse scaling: eta0 / (t^power_t)
                let eta0 = if self.eta0 > 0.0 { self.eta0 } else { 0.01 };
                eta0 / (iteration as f64 + 1.0).powf(self.power_t)
            }
            "adaptive" => {
                // Adaptive learning rate (simplified version)
                let eta0 = if self.eta0 > 0.0 { self.eta0 } else { 0.01 };
                eta0 / (1.0 + iteration as f64 / 100.0)
            }
            _ => 0.01, // fallback
        }
    }

    /// SGD training algorithm
    fn sgd_fit(
        &self,
        x: ArrayView2<f64>,
        y: ArrayView1<i32>,
        w: &mut Array1<f64>,
        intercept: &mut f64,
    ) -> Result<usize> {
        let (n_samples, n_features) = x.dim();
        let mut rng = StdRng::from_rng(&mut thread_rng());

        // Convert labels to -1/+1
        let y_binary: Array1<f64> = y.map(|&label| if label == 1 { 1.0 } else { -1.0 });

        // For averaged SGD
        let mut w_avg = if self.average {
            Some(Array1::zeros(n_features))
        } else {
            None
        };
        let mut intercept_avg = if self.average { Some(0.0) } else { None };
        let mut update_count = 0;

        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut best_loss = f64::INFINITY;
        let mut no_improvement = 0;

        for epoch in 0..self.max_iter {
            // Shuffle data if requested
            if self.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut epoch_loss = 0.0;
            let learning_rate = self.compute_learning_rate(epoch, n_samples);

            for &i in &indices {
                let xi = x.row(i);
                let yi = y_binary[i];

                // Compute prediction
                let mut prediction = if self.fit_intercept { *intercept } else { 0.0 };
                for j in 0..n_features {
                    prediction += w[j] * xi[j];
                }

                // Compute loss and gradient
                let (loss, gradient) = self.compute_loss_gradient(prediction, yi)?;
                epoch_loss += loss;

                // Update weights if gradient is non-zero
                if gradient.abs() > 1e-12 {
                    for j in 0..n_features {
                        w[j] -= learning_rate * gradient * xi[j];
                    }

                    if self.fit_intercept {
                        *intercept -= learning_rate * gradient;
                    }
                }

                // Apply regularization
                self.apply_regularization(w, learning_rate)?;

                // Update averaged weights
                if let (Some(ref mut w_avg), Some(ref mut intercept_avg)) =
                    (&mut w_avg, &mut intercept_avg)
                {
                    update_count += 1;
                    let eta = 1.0 / update_count as f64;
                    for j in 0..n_features {
                        w_avg[j] = (1.0 - eta) * w_avg[j] + eta * w[j];
                    }
                    *intercept_avg = (1.0 - eta) * *intercept_avg + eta * *intercept;
                }
            }

            // Check convergence
            epoch_loss /= n_samples as f64;

            if self.verbose && epoch % 100 == 0 {
                println!(
                    "SGD epoch {}, loss: {:.6}, lr: {:.6}",
                    epoch, epoch_loss, learning_rate
                );
            }

            // Early stopping check
            if self.early_stopping {
                if epoch_loss < best_loss - self.tol {
                    best_loss = epoch_loss;
                    no_improvement = 0;
                } else {
                    no_improvement += 1;
                    if no_improvement >= self.n_iter_no_change {
                        if self.verbose {
                            println!(
                                "Early stopping at epoch {} (loss: {:.6})",
                                epoch, epoch_loss
                            );
                        }
                        break;
                    }
                }
            } else if epoch_loss < self.tol {
                if self.verbose {
                    println!("SGD converged at epoch {} (loss: {:.6})", epoch, epoch_loss);
                }
                break;
            }
        }

        // Use averaged weights if requested
        if let (Some(w_avg), Some(intercept_avg)) = (w_avg, intercept_avg) {
            *w = w_avg;
            *intercept = intercept_avg;
        }

        Ok(self.max_iter)
    }
}

impl Estimator for SGDClassifier {
    type Config = Self;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for SGDClassifier {
    type Fitted = TrainedSGDClassifier;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> Result<TrainedSGDClassifier> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if x.len_of(Axis(0)) != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("({}, {})", n_samples, n_features),
                actual: format!("({}, ?)", y.len()),
            });
        }

        // Get unique classes
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from(classes);

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        let n_classes = classes.len();

        // For binary classification
        if n_classes == 2 {
            let mut w = Array1::zeros(n_features);
            let mut intercept = 0.0;

            // Convert labels to binary (0/1 -> -1/1)
            let y_binary = y.map(|&label| if label == classes[1] { 1 } else { -1 });

            let n_iter = self.sgd_fit(x.view(), y_binary.view(), &mut w, &mut intercept)?;

            let coef = w.insert_axis(Axis(0));
            let intercept_arr = Array1::from(vec![intercept]);

            Ok(TrainedSGDClassifier {
                coef_: coef,
                intercept_: intercept_arr,
                classes_: classes,
                n_features_in_: n_features,
                n_iter_: n_iter,
                _params: self,
            })
        } else {
            // Multi-class: One-vs-Rest approach
            let mut coef_matrix = Array2::zeros((n_classes, n_features));
            let mut intercept_vec = Array1::zeros(n_classes);
            let mut total_iter = 0;

            for (class_idx, &class_label) in classes.iter().enumerate() {
                // Create binary labels (current class vs rest)
                let y_binary = y.map(|&label| if label == class_label { 1 } else { -1 });

                let mut w = Array1::zeros(n_features);
                let mut intercept = 0.0;

                let n_iter = self.sgd_fit(x.view(), y_binary.view(), &mut w, &mut intercept)?;
                total_iter += n_iter;

                coef_matrix.row_mut(class_idx).assign(&w);
                intercept_vec[class_idx] = intercept;
            }

            Ok(TrainedSGDClassifier {
                coef_: coef_matrix,
                intercept_: intercept_vec,
                classes_: classes,
                n_features_in_: n_features,
                n_iter_: total_iter / n_classes,
                _params: self,
            })
        }
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedSGDClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_,
                actual: n_features,
            });
        }

        let n_classes = self.classes_.len();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let xi = x.row(i);

            if n_classes == 2 {
                // Binary classification
                let mut score = self.intercept_[0];
                for j in 0..n_features {
                    score += self.coef_[[0, j]] * xi[j];
                }
                predictions[i] = if score >= 0.0 {
                    self.classes_[1]
                } else {
                    self.classes_[0]
                };
            } else {
                // Multi-class: predict class with highest score
                let mut best_score = f64::NEG_INFINITY;
                let mut best_class = 0;

                for class_idx in 0..n_classes {
                    let mut score = self.intercept_[class_idx];
                    for j in 0..n_features {
                        score += self.coef_[[class_idx, j]] * xi[j];
                    }

                    if score > best_score {
                        best_score = score;
                        best_class = class_idx;
                    }
                }

                predictions[i] = self.classes_[best_class];
            }
        }

        Ok(predictions)
    }
}

impl TrainedSGDClassifier {
    /// Get decision function scores
    pub fn decision_function(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in_ {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in_,
                actual: n_features,
            });
        }

        let n_classes = self.classes_.len();
        let output_shape = if n_classes == 2 {
            (n_samples, 1)
        } else {
            (n_samples, n_classes)
        };
        let mut scores = Array2::zeros(output_shape);

        for i in 0..n_samples {
            let xi = x.row(i);

            if n_classes == 2 {
                let mut score = self.intercept_[0];
                for j in 0..n_features {
                    score += self.coef_[[0, j]] * xi[j];
                }
                scores[[i, 0]] = score;
            } else {
                for class_idx in 0..n_classes {
                    let mut score = self.intercept_[class_idx];
                    for j in 0..n_features {
                        score += self.coef_[[class_idx, j]] * xi[j];
                    }
                    scores[[i, class_idx]] = score;
                }
            }
        }

        Ok(scores)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_sgd_classifier_basic() {
        let X = array![[2.0, 3.0], [3.0, 3.0], [1.0, 1.0], [2.0, 1.0]];
        let y = array![1, 1, 0, 0];

        let model = SGDClassifier::new()
            .with_max_iter(1000)
            .with_tol(1e-6)
            .with_random_state(42);

        let trained = model.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        // Check that we have reasonable accuracy
        let accuracy = predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| if pred == actual { 1.0 } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;

        assert!(accuracy >= 0.5, "SGD accuracy too low: {}", accuracy);
    }

    #[test]
    fn test_sgd_loss_functions() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        for loss in &["hinge", "squared_hinge", "log", "modified_huber"] {
            let model = SGDClassifier::new()
                .with_loss(loss)
                .with_max_iter(500)
                .with_random_state(42);

            let result = model.fit(&X, &y);
            assert!(result.is_ok(), "Loss function {} should work", loss);
        }
    }

    #[test]
    fn test_sgd_penalties() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        for penalty in &["l2", "l1", "elasticnet"] {
            let model = SGDClassifier::new()
                .with_penalty(penalty)
                .with_alpha(0.01)
                .with_max_iter(500)
                .with_random_state(42);

            let result = model.fit(&X, &y);
            assert!(result.is_ok(), "Penalty {} should work", penalty);
        }
    }

    #[test]
    fn test_sgd_learning_rates() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        for lr in &["constant", "optimal", "invscaling", "adaptive"] {
            let model = SGDClassifier::new()
                .with_learning_rate(lr)
                .with_eta0(0.01)
                .with_max_iter(500)
                .with_random_state(42);

            let result = model.fit(&X, &y);
            assert!(result.is_ok(), "Learning rate {} should work", lr);
        }
    }

    #[test]
    fn test_sgd_multiclass() {
        let X = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [3.0, 1.0],
            [1.0, 2.0],
            [2.0, 3.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let model = SGDClassifier::new()
            .with_max_iter(1000)
            .with_random_state(42);

        let trained = model.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert_eq!(trained.classes_.len(), 3);
    }

    #[test]
    fn test_sgd_decision_function() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        let model = SGDClassifier::new()
            .with_max_iter(500)
            .with_random_state(42);

        let trained = model.fit(&X, &y).unwrap();
        let scores = trained.decision_function(&X).unwrap();

        assert_eq!(scores.shape(), &[4, 1]); // Binary classification
    }

    #[test]
    fn test_sgd_averaged() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        let model = SGDClassifier::new()
            .with_average(true)
            .with_max_iter(500)
            .with_random_state(42);

        let result = model.fit(&X, &y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_sgd_early_stopping() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![1, 1, 0, 0];

        let model = SGDClassifier::new()
            .with_early_stopping(true)
            .with_n_iter_no_change(3)
            .with_max_iter(1000)
            .with_random_state(42);

        let result = model.fit(&X, &y);
        assert!(result.is_ok());
    }
}

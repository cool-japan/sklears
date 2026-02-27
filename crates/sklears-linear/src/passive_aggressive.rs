//! Passive Aggressive algorithms for online learning
//!
//! The Passive-Aggressive algorithms are a family of algorithms for large-scale learning.
//! They are especially suited for learning from data streams.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::SliceRandomExt;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Float, Int},
};

// Helper function for NaN-safe float comparison
fn compare_floats(a: &Float, b: &Float) -> Result<std::cmp::Ordering> {
    a.partial_cmp(b)
        .ok_or_else(|| SklearsError::InvalidInput("NaN encountered in comparison".to_string()))
}

// Helper function for safe mean computation
fn safe_mean(arr: &Array1<Float>) -> Result<Float> {
    arr.mean().ok_or_else(|| {
        SklearsError::InvalidInput("Mean computation failed (empty array)".to_string())
    })
}

/// Configuration for PassiveAggressiveClassifier
#[derive(Debug, Clone)]
pub struct PassiveAggressiveClassifierConfig {
    /// The regularization parameter
    pub c: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of passes over the training data
    pub max_iter: usize,
    /// The stopping criterion tolerance
    pub tol: Float,
    /// Whether to shuffle the training data before each epoch
    pub shuffle: bool,
    /// The loss function: "hinge" or "squared_hinge"
    pub loss: PassiveAggressiveLoss,
    /// Number of samples to use for validation
    pub n_iter_no_change: usize,
    /// Random state for shuffling
    pub random_state: Option<u64>,
    /// Learning rate
    pub learning_rate: Float,
    /// Number of classes (will be set during fit)
    pub n_classes: Option<usize>,
}

/// Loss functions for Passive Aggressive algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PassiveAggressiveLoss {
    Hinge,
    SquaredHinge,
    Epsilon(Float),        // For regression
    SquaredEpsilon(Float), // For regression
}

impl Default for PassiveAggressiveClassifierConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            loss: PassiveAggressiveLoss::Hinge,
            n_iter_no_change: 5,
            random_state: None,
            learning_rate: 1.0,
            n_classes: None,
        }
    }
}

/// Passive Aggressive Classifier
pub struct PassiveAggressiveClassifier<State = Untrained> {
    config: PassiveAggressiveClassifierConfig,
    state: PhantomData<State>,
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    classes_: Option<Array1<Int>>,
    n_iter_: Option<usize>,
}

impl PassiveAggressiveClassifier<Untrained> {
    /// Create a new PassiveAggressiveClassifier with default configuration
    pub fn new() -> Self {
        Self {
            config: PassiveAggressiveClassifierConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            classes_: None,
            n_iter_: None,
        }
    }

    /// Set the regularization parameter
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: PassiveAggressiveLoss) -> Self {
        self.config.loss = loss;
        self
    }
}

impl Default for PassiveAggressiveClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PassiveAggressiveClassifier<Untrained> {
    type Float = Float;
    type Config = PassiveAggressiveClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for PassiveAggressiveClassifier<Trained> {
    type Float = Float;
    type Config = PassiveAggressiveClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute the hinge loss
fn hinge_loss(y_true: Float, y_pred: Float) -> Float {
    (1.0 - y_true * y_pred).max(0.0)
}

/// Compute the squared hinge loss
fn squared_hinge_loss(y_true: Float, y_pred: Float) -> Float {
    let loss = hinge_loss(y_true, y_pred);
    loss * loss
}

impl Fit<Array2<Float>, Array1<Int>> for PassiveAggressiveClassifier<Untrained> {
    type Fitted = PassiveAggressiveClassifier<Trained>;

    fn fit(mut self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<Int> = y.iter().copied().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();
        self.config.n_classes = Some(n_classes);

        // Initialize weights and intercept
        let mut coef = Array2::zeros((n_classes, n_features));
        let mut intercept = Array1::zeros(n_classes);

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = thread_rng();

        let mut n_iter = 0;
        let mut no_improvement_count = 0;
        let mut best_loss = Float::INFINITY;

        // Training loop
        for epoch in 0..self.config.max_iter {
            n_iter = epoch + 1;

            // Shuffle indices if requested
            if self.config.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut epoch_loss = 0.0;
            let mut n_updates = 0;

            // Process each sample
            for &idx in &indices {
                let x_i = x.row(idx);
                let y_i = y[idx] as usize;

                // Compute scores for all classes
                let scores = coef.dot(&x_i) + &intercept;

                // For multi-class, we use one-vs-all approach
                let mut max_loss: Float = 0.0;
                let mut update_needed = false;

                for k in 0..n_classes {
                    let y_k = if k == y_i { 1.0 } else { -1.0 };
                    let score_k = scores[k];

                    let loss = match self.config.loss {
                        PassiveAggressiveLoss::Hinge => hinge_loss(y_k, score_k),
                        PassiveAggressiveLoss::SquaredHinge => squared_hinge_loss(y_k, score_k),
                        _ => {
                            return Err(SklearsError::InvalidInput(
                                "Invalid loss for classification".to_string(),
                            ))
                        }
                    };

                    if loss > 0.0 {
                        update_needed = true;
                        max_loss = Float::max(max_loss, loss);

                        // Compute update step size
                        let x_norm_sq = x_i.dot(&x_i);
                        let tau = match self.config.loss {
                            PassiveAggressiveLoss::Hinge => {
                                loss / (x_norm_sq + 1.0 / (2.0 * self.config.c))
                            }
                            PassiveAggressiveLoss::SquaredHinge => {
                                loss / (x_norm_sq + 1.0 / (2.0 * self.config.c) + 0.5 * loss)
                            }
                            _ => 0.0,
                        };

                        // Update weights
                        let update = tau * y_k * self.config.learning_rate;
                        coef.row_mut(k).scaled_add(update, &x_i);

                        if self.config.fit_intercept {
                            intercept[k] += update;
                        }
                    }
                }

                if update_needed {
                    n_updates += 1;
                    epoch_loss += max_loss;
                }
            }

            // Check for convergence
            let avg_loss = epoch_loss / n_samples as Float;
            if avg_loss < best_loss - self.config.tol {
                best_loss = avg_loss;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            if no_improvement_count >= self.config.n_iter_no_change {
                break;
            }

            if n_updates == 0 {
                break;
            }
        }

        Ok(PassiveAggressiveClassifier {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            classes_: Some(Array1::from_vec(classes)),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array1<Int>> for PassiveAggressiveClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ must be Some in Trained state");
        let intercept = self
            .intercept_
            .as_ref()
            .expect("intercept_ must be Some in Trained state");
        let classes = self
            .classes_
            .as_ref()
            .expect("classes_ must be Some in Trained state");

        let scores = x.dot(&coef.t()) + intercept;
        let mut predictions = Vec::with_capacity(scores.nrows());
        for row in scores.axis_iter(Axis(0)) {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| compare_floats(a, b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .ok_or_else(|| SklearsError::InvalidInput("Empty row in scores".to_string()))?;
            predictions.push(classes[max_idx]);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl Score<Array2<Float>, Array1<Int>> for PassiveAggressiveClassifier<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Float> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, true_val)| pred == true_val)
            .count();

        Ok(correct as Float / y.len() as Float)
    }
}

impl PassiveAggressiveClassifier<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_
            .as_ref()
            .expect("coef_ must be Some in Trained state")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<&Array1<Float>> {
        self.intercept_.as_ref()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_
            .as_ref()
            .expect("classes_ must be Some in Trained state")
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("n_iter_ must be Some in Trained state")
    }
}

/// Configuration for PassiveAggressiveRegressor
#[derive(Debug, Clone)]
pub struct PassiveAggressiveRegressorConfig {
    /// The regularization parameter
    pub c: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of passes over the training data
    pub max_iter: usize,
    /// The stopping criterion tolerance
    pub tol: Float,
    /// Whether to shuffle the training data before each epoch
    pub shuffle: bool,
    /// The loss function
    pub loss: PassiveAggressiveLoss,
    /// The epsilon parameter for epsilon-insensitive loss
    pub epsilon: Float,
    /// Number of samples to use for validation
    pub n_iter_no_change: usize,
    /// Random state for shuffling
    pub random_state: Option<u64>,
    /// Learning rate
    pub learning_rate: Float,
}

impl Default for PassiveAggressiveRegressorConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            loss: PassiveAggressiveLoss::Epsilon(0.1),
            epsilon: 0.1,
            n_iter_no_change: 5,
            random_state: None,
            learning_rate: 1.0,
        }
    }
}

/// Passive Aggressive Regressor
pub struct PassiveAggressiveRegressor<State = Untrained> {
    config: PassiveAggressiveRegressorConfig,
    state: PhantomData<State>,
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_iter_: Option<usize>,
}

impl PassiveAggressiveRegressor<Untrained> {
    /// Create a new PassiveAggressiveRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: PassiveAggressiveRegressorConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_iter_: None,
        }
    }

    /// Set the regularization parameter
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set whether to fit the intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self.config.loss = PassiveAggressiveLoss::Epsilon(epsilon);
        self
    }
}

impl Default for PassiveAggressiveRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PassiveAggressiveRegressor<Untrained> {
    type Float = Float;
    type Config = PassiveAggressiveRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for PassiveAggressiveRegressor<Trained> {
    type Float = Float;
    type Config = PassiveAggressiveRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute epsilon-insensitive loss
fn epsilon_loss(y_true: Float, y_pred: Float, epsilon: Float) -> Float {
    ((y_true - y_pred).abs() - epsilon).max(0.0)
}

/// Compute squared epsilon-insensitive loss
fn squared_epsilon_loss(y_true: Float, y_pred: Float, epsilon: Float) -> Float {
    let loss = epsilon_loss(y_true, y_pred, epsilon);
    loss * loss
}

impl Fit<Array2<Float>, Array1<Float>> for PassiveAggressiveRegressor<Untrained> {
    type Fitted = PassiveAggressiveRegressor<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize weights and intercept
        let mut coef = Array1::zeros(n_features);
        let mut intercept = 0.0;

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = thread_rng();

        let mut n_iter = 0;
        let mut no_improvement_count = 0;
        let mut best_loss = Float::INFINITY;

        // Training loop
        for epoch in 0..self.config.max_iter {
            n_iter = epoch + 1;

            // Shuffle indices if requested
            if self.config.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut epoch_loss = 0.0;
            let mut n_updates = 0;

            // Process each sample
            for &idx in &indices {
                let x_i = x.row(idx);
                let y_i = y[idx];

                // Compute prediction
                let y_pred = coef.dot(&x_i) + intercept;

                // Compute loss
                let (loss, _epsilon) = match self.config.loss {
                    PassiveAggressiveLoss::Epsilon(eps) => (epsilon_loss(y_i, y_pred, eps), eps),
                    PassiveAggressiveLoss::SquaredEpsilon(eps) => {
                        (squared_epsilon_loss(y_i, y_pred, eps), eps)
                    }
                    _ => (
                        epsilon_loss(y_i, y_pred, self.config.epsilon),
                        self.config.epsilon,
                    ),
                };

                if loss > 0.0 {
                    n_updates += 1;
                    epoch_loss += loss;

                    // Compute update step size
                    let x_norm_sq = x_i.dot(&x_i);
                    let tau = match self.config.loss {
                        PassiveAggressiveLoss::Epsilon(_) => {
                            loss / (x_norm_sq + 1.0 / (2.0 * self.config.c))
                        }
                        PassiveAggressiveLoss::SquaredEpsilon(_) => {
                            loss / (x_norm_sq + 1.0 / (2.0 * self.config.c) + 0.5 * loss)
                        }
                        _ => loss / (x_norm_sq + 1.0 / (2.0 * self.config.c)),
                    };

                    // Determine sign of update
                    let sign = if y_i > y_pred { 1.0 } else { -1.0 };

                    // Update weights
                    let update = tau * sign * self.config.learning_rate;
                    coef.scaled_add(update, &x_i);

                    if self.config.fit_intercept {
                        intercept += update;
                    }
                }
            }

            // Check for convergence
            let avg_loss = epoch_loss / n_samples as Float;
            if avg_loss < best_loss - self.config.tol {
                best_loss = avg_loss;
                no_improvement_count = 0;
            } else {
                no_improvement_count += 1;
            }

            if no_improvement_count >= self.config.n_iter_no_change {
                break;
            }

            if n_updates == 0 {
                break;
            }
        }

        Ok(PassiveAggressiveRegressor {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for PassiveAggressiveRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let coef = self
            .coef_
            .as_ref()
            .expect("coef_ must be Some in Trained state");
        let intercept = self
            .intercept_
            .expect("intercept_ must be Some in Trained state");

        Ok(x.dot(coef) + intercept)
    }
}

impl Score<Array2<Float>, Array1<Float>> for PassiveAggressiveRegressor<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        let predictions = self.predict(x)?;
        let ss_res = (y - &predictions).mapv(|e| e * e).sum();
        let y_mean = safe_mean(y)?;
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl PassiveAggressiveRegressor<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_
            .as_ref()
            .expect("coef_ must be Some in Trained state")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        Some(
            self.intercept_
                .expect("intercept_ must be Some in Trained state"),
        )
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("n_iter_ must be Some in Trained state")
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_passive_aggressive_classifier_binary() {
        // Simple linearly separable data
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 3.0],
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-3.0, -3.0],
        ];
        let y = array![1, 1, 1, 0, 0, 0];

        let model = PassiveAggressiveClassifier::new()
            .c(1.0)
            .max_iter(100)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        let accuracy = model.score(&x, &y).unwrap();

        // Should achieve perfect classification on this simple data
        assert!(accuracy > 0.9);
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_passive_aggressive_classifier_multiclass() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = PassiveAggressiveClassifier::new()
            .c(1.0)
            .max_iter(200)
            .fit(&x, &y)
            .unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy > 0.8);

        // Check that we have the right number of classes
        assert_eq!(model.classes().len(), 3);
        assert_eq!(model.coef().nrows(), 3);
    }

    #[test]
    fn test_passive_aggressive_regressor() {
        // Simple linear relationship with some noise
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
        let y = array![2.1, 4.2, 5.9, 8.1, 9.8, 12.1, 14.2, 15.9];

        let model = PassiveAggressiveRegressor::new()
            .c(1.0)
            .epsilon(0.1)
            .max_iter(100)
            .fit(&x, &y)
            .unwrap();

        let _predictions = model.predict(&x).unwrap();
        let r2 = model.score(&x, &y).unwrap();

        assert!(r2 > 0.9);

        // Test prediction on new data
        let x_test = array![[2.5], [4.5]];
        let predictions_test = model.predict(&x_test).unwrap();
        assert!(predictions_test[0] > 4.0 && predictions_test[0] < 6.0);
        assert!(predictions_test[1] > 8.0 && predictions_test[1] < 10.0);
    }

    #[test]
    fn test_passive_aggressive_no_intercept() {
        let x = array![[1.0], [2.0], [3.0]];
        let y = array![2.0, 4.0, 6.0];

        let model = PassiveAggressiveRegressor::new()
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.intercept().unwrap(), 0.0);
    }

    #[test]
    fn test_squared_hinge_loss() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1, 1, 0, 0];

        let model = PassiveAggressiveClassifier::new()
            .loss(PassiveAggressiveLoss::SquaredHinge)
            .fit(&x, &y)
            .unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy > 0.8);
    }
}

//! Stochastic Gradient Descent (SGD) linear models
//!
//! This module provides SGD-based linear classifiers and regressors that can handle
//! very large datasets efficiently by processing one sample at a time.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{seq::SliceRandom, thread_rng};
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Float, Int},
};

/// Loss functions for SGD
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SGDLoss {
    // Classification losses
    Hinge,         // Linear SVM
    Log,           // Logistic Regression
    ModifiedHuber, // Smooth hinge loss
    SquaredHinge,  // Squared hinge
    Perceptron,    // Perceptron loss
    // Regression losses
    SquaredError,              // Ordinary least squares
    Huber,                     // Huber loss
    EpsilonInsensitive,        // Linear SVM regression
    SquaredEpsilonInsensitive, // Squared epsilon-insensitive
}

/// Penalty types for SGD
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SGDPenalty {
    None,
    L2,
    L1,
    ElasticNet(Float), // l1_ratio parameter
}

/// Learning rate schedules
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LearningRateSchedule {
    Constant,
    Optimal,
    InvScaling(Float), // power parameter
    Adaptive,
}

/// Configuration for SGDClassifier
#[derive(Debug, Clone)]
pub struct SGDClassifierConfig {
    /// The loss function to use
    pub loss: SGDLoss,
    /// The penalty (regularization term) to be used
    pub penalty: SGDPenalty,
    /// Constant that multiplies the regularization term
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of passes over the training data
    pub max_iter: usize,
    /// The stopping criterion tolerance
    pub tol: Float,
    /// Whether to shuffle the training data before each epoch
    pub shuffle: bool,
    /// The learning rate schedule
    pub learning_rate: LearningRateSchedule,
    /// The initial learning rate
    pub eta0: Float,
    /// Random state for shuffling
    pub random_state: Option<u64>,
    /// Number of iterations with no improvement to wait before early stopping
    pub n_iter_no_change: usize,
    /// Whether to use averaging
    pub average: bool,
    /// Epsilon for epsilon-insensitive loss
    pub epsilon: Float,
}

impl Default for SGDClassifierConfig {
    fn default() -> Self {
        Self {
            loss: SGDLoss::Hinge,
            penalty: SGDPenalty::L2,
            alpha: 0.0001,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            learning_rate: LearningRateSchedule::Optimal,
            eta0: 0.0,
            random_state: None,
            n_iter_no_change: 5,
            average: false,
            epsilon: 0.1,
        }
    }
}

/// SGD Classifier
pub struct SGDClassifier<State = Untrained> {
    config: SGDClassifierConfig,
    state: PhantomData<State>,
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    classes_: Option<Array1<Int>>,
    n_iter_: Option<usize>,
    t_: Option<Float>,
}

impl SGDClassifier<Untrained> {
    /// Create a new SGDClassifier with default configuration
    pub fn new() -> Self {
        Self {
            config: SGDClassifierConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            classes_: None,
            n_iter_: None,
            t_: None,
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: SGDLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the penalty type
    pub fn penalty(mut self, penalty: SGDPenalty) -> Self {
        self.config.penalty = penalty;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
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

    /// Set the learning rate schedule
    pub fn learning_rate(mut self, learning_rate: LearningRateSchedule) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the initial learning rate
    pub fn eta0(mut self, eta0: Float) -> Self {
        self.config.eta0 = eta0;
        self
    }

    /// Set whether to use averaging
    pub fn average(mut self, average: bool) -> Self {
        self.config.average = average;
        self
    }
}

impl Default for SGDClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SGDClassifier<Untrained> {
    type Float = Float;
    type Config = SGDClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for SGDClassifier<Trained> {
    type Float = Float;
    type Config = SGDClassifierConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute the learning rate for the current iteration
fn compute_learning_rate(
    schedule: LearningRateSchedule,
    eta0: Float,
    alpha: Float,
    t: Float,
) -> Float {
    match schedule {
        LearningRateSchedule::Constant => eta0,
        LearningRateSchedule::Optimal => eta0 / (1.0 + eta0 * alpha * t),
        LearningRateSchedule::InvScaling(power) => eta0 / t.powf(power),
        LearningRateSchedule::Adaptive => eta0, // Simplified adaptive
    }
}

/// Apply penalty to weights
fn apply_penalty(
    weights: &mut Array1<Float>,
    penalty: SGDPenalty,
    alpha: Float,
    learning_rate: Float,
) {
    match penalty {
        SGDPenalty::None => {}
        SGDPenalty::L2 => {
            *weights *= 1.0 - alpha * learning_rate;
        }
        SGDPenalty::L1 => {
            let threshold = alpha * learning_rate;
            weights.mapv_inplace(|w| {
                if w > threshold {
                    w - threshold
                } else if w < -threshold {
                    w + threshold
                } else {
                    0.0
                }
            });
        }
        SGDPenalty::ElasticNet(l1_ratio) => {
            // Apply L2 part
            *weights *= 1.0 - alpha * (1.0 - l1_ratio) * learning_rate;

            // Apply L1 part
            let threshold = alpha * l1_ratio * learning_rate;
            weights.mapv_inplace(|w| {
                if w > threshold {
                    w - threshold
                } else if w < -threshold {
                    w + threshold
                } else {
                    0.0
                }
            });
        }
    }
}

/// Compute classification loss and gradient
fn compute_classification_loss_gradient(
    loss: SGDLoss,
    y_true: Float,
    y_pred: Float,
    epsilon: Float,
) -> (Float, Float) {
    match loss {
        SGDLoss::Hinge => {
            let margin = y_true * y_pred;
            if margin < 1.0 {
                (1.0 - margin, -y_true)
            } else {
                (0.0, 0.0)
            }
        }
        SGDLoss::Log => {
            // Logistic loss: log(1 + exp(-y * f(x)))
            let z = y_true * y_pred;
            let loss = if z > 0.0 {
                (1.0 + (-z).exp()).ln()
            } else {
                -z + (1.0 + z.exp()).ln()
            };
            let grad = -y_true / (1.0 + (y_true * y_pred).exp());
            (loss, grad)
        }
        SGDLoss::ModifiedHuber => {
            let margin = y_true * y_pred;
            if margin >= 1.0 {
                (0.0, 0.0)
            } else if margin >= -1.0 {
                let diff = 1.0 - margin;
                (diff * diff, -2.0 * y_true * diff)
            } else {
                (-4.0 * margin, -4.0 * y_true)
            }
        }
        SGDLoss::SquaredHinge => {
            let margin = y_true * y_pred;
            if margin < 1.0 {
                let diff = 1.0 - margin;
                (diff * diff, -2.0 * y_true * diff)
            } else {
                (0.0, 0.0)
            }
        }
        SGDLoss::Perceptron => {
            if y_true * y_pred <= 0.0 {
                (1.0, -y_true)
            } else {
                (0.0, 0.0)
            }
        }
        _ => panic!("Invalid loss function for classification"),
    }
}

impl Fit<Array2<Float>, Array1<Int>> for SGDClassifier<Untrained> {
    type Fitted = SGDClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Self::Fitted> {
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

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "At least two classes are required".to_string(),
            ));
        }

        // Initialize weights and intercept
        let mut coef = Array2::<Float>::zeros((n_classes, n_features));
        let mut intercept = Array1::<Float>::zeros(n_classes);

        // For averaging
        let mut coef_avg = if self.config.average {
            Some(Array2::zeros((n_classes, n_features)))
        } else {
            None
        };
        let mut intercept_avg = if self.config.average {
            Some(Array1::zeros(n_classes))
        } else {
            None
        };

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = thread_rng();

        // Set initial learning rate
        let mut eta0 = self.config.eta0;
        if eta0 == 0.0 {
            match self.config.learning_rate {
                LearningRateSchedule::Optimal => {
                    // Use a heuristic for optimal learning rate
                    eta0 = 1.0 / (self.config.alpha * n_samples as Float).max(1.0);
                }
                _ => eta0 = 0.01,
            }
        }

        let mut t = 1.0;
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

            // Process each sample
            for &idx in &indices {
                let x_i = x.row(idx);
                let y_i = y[idx];
                let class_idx = classes.iter().position(|&c| c == y_i).unwrap();

                // Compute learning rate
                let learning_rate =
                    compute_learning_rate(self.config.learning_rate, eta0, self.config.alpha, t);

                if n_classes == 2 {
                    // Binary classification: use single weight vector
                    let y_binary = if class_idx == 1 { 1.0 } else { -1.0 };
                    let prediction = coef.row(1).dot(&x_i) + intercept[1];

                    let (loss, grad) = compute_classification_loss_gradient(
                        self.config.loss,
                        y_binary,
                        prediction,
                        self.config.epsilon,
                    );

                    epoch_loss += loss;

                    if grad != 0.0 {
                        // Update weights
                        coef.row_mut(1).scaled_add(-learning_rate * grad, &x_i);
                        if self.config.fit_intercept {
                            intercept[1] -= learning_rate * grad;
                        }
                    }

                    // Apply penalty
                    apply_penalty(
                        &mut coef.row_mut(1).to_owned(),
                        self.config.penalty,
                        self.config.alpha,
                        learning_rate,
                    );
                } else {
                    // Multi-class: one-vs-all
                    for k in 0..n_classes {
                        let y_k = if k == class_idx { 1.0 } else { -1.0 };
                        let prediction = coef.row(k).dot(&x_i) + intercept[k];

                        let (loss, grad) = compute_classification_loss_gradient(
                            self.config.loss,
                            y_k,
                            prediction,
                            self.config.epsilon,
                        );

                        if k == class_idx {
                            epoch_loss += loss;
                        }

                        if grad != 0.0 {
                            // Update weights
                            coef.row_mut(k).scaled_add(-learning_rate * grad, &x_i);
                            if self.config.fit_intercept {
                                intercept[k] -= learning_rate * grad;
                            }
                        }

                        // Apply penalty
                        let mut w = coef.row(k).to_owned();
                        apply_penalty(
                            &mut w,
                            self.config.penalty,
                            self.config.alpha,
                            learning_rate,
                        );
                        coef.row_mut(k).assign(&w);
                    }
                }

                // Update averaging
                if self.config.average {
                    if let (Some(ref mut coef_avg), Some(ref mut intercept_avg)) =
                        (&mut coef_avg, &mut intercept_avg)
                    {
                        *coef_avg = &*coef_avg * ((t - 1.0) / t) + &coef * (1.0 / t);
                        *intercept_avg = &*intercept_avg * ((t - 1.0) / t) + &intercept * (1.0 / t);
                    }
                }

                t += 1.0;
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
        }

        // Use averaged weights if requested
        if self.config.average {
            if let (Some(coef_avg), Some(intercept_avg)) = (coef_avg, intercept_avg) {
                coef = coef_avg;
                intercept = intercept_avg;
            }
        }

        // For binary classification, ensure coef[0] = -coef[1]
        if n_classes == 2 {
            let neg_coef1 = -&coef.row(1).to_owned();
            coef.row_mut(0).assign(&neg_coef1);
            intercept[0] = -intercept[1];
        }

        Ok(SGDClassifier {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            classes_: Some(Array1::from_vec(classes)),
            n_iter_: Some(n_iter),
            t_: Some(t),
        })
    }
}

impl Predict<Array2<Float>, Array1<Int>> for SGDClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let scores = x.dot(&coef.t()) + intercept;
        let predictions = scores
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }
}

impl Score<Array2<Float>, Array1<Int>> for SGDClassifier<Trained> {
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

impl SGDClassifier<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<&Array1<Float>> {
        self.intercept_.as_ref()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }
}

/// Configuration for SGDRegressor
#[derive(Debug, Clone)]
pub struct SGDRegressorConfig {
    /// The loss function to use
    pub loss: SGDLoss,
    /// The penalty (regularization term) to be used
    pub penalty: SGDPenalty,
    /// Constant that multiplies the regularization term
    pub alpha: Float,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Maximum number of passes over the training data
    pub max_iter: usize,
    /// The stopping criterion tolerance
    pub tol: Float,
    /// Whether to shuffle the training data before each epoch
    pub shuffle: bool,
    /// The learning rate schedule
    pub learning_rate: LearningRateSchedule,
    /// The initial learning rate
    pub eta0: Float,
    /// Random state for shuffling
    pub random_state: Option<u64>,
    /// Number of iterations with no improvement to wait before early stopping
    pub n_iter_no_change: usize,
    /// Whether to use averaging
    pub average: bool,
    /// Epsilon for epsilon-insensitive and Huber loss
    pub epsilon: Float,
}

impl Default for SGDRegressorConfig {
    fn default() -> Self {
        Self {
            loss: SGDLoss::SquaredError,
            penalty: SGDPenalty::L2,
            alpha: 0.0001,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            learning_rate: LearningRateSchedule::InvScaling(0.25),
            eta0: 0.01,
            random_state: None,
            n_iter_no_change: 5,
            average: false,
            epsilon: 0.1,
        }
    }
}

/// SGD Regressor
pub struct SGDRegressor<State = Untrained> {
    config: SGDRegressorConfig,
    state: PhantomData<State>,
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_iter_: Option<usize>,
    t_: Option<Float>,
}

impl SGDRegressor<Untrained> {
    /// Create a new SGDRegressor with default configuration
    pub fn new() -> Self {
        Self {
            config: SGDRegressorConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_iter_: None,
            t_: None,
        }
    }

    /// Set the loss function
    pub fn loss(mut self, loss: SGDLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the penalty type
    pub fn penalty(mut self, penalty: SGDPenalty) -> Self {
        self.config.penalty = penalty;
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.config.alpha = alpha;
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

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: Float) -> Self {
        self.config.epsilon = epsilon;
        self
    }
}

impl Default for SGDRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SGDRegressor<Untrained> {
    type Float = Float;
    type Config = SGDRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for SGDRegressor<Trained> {
    type Float = Float;
    type Config = SGDRegressorConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute regression loss and gradient
fn compute_regression_loss_gradient(
    loss: SGDLoss,
    y_true: Float,
    y_pred: Float,
    epsilon: Float,
) -> (Float, Float) {
    match loss {
        SGDLoss::SquaredError => {
            let error = y_pred - y_true;
            (0.5 * error * error, error)
        }
        SGDLoss::Huber => {
            let error = y_pred - y_true;
            let abs_error = error.abs();
            if abs_error <= epsilon {
                (0.5 * error * error, error)
            } else {
                (
                    epsilon * abs_error - 0.5 * epsilon * epsilon,
                    epsilon * error.signum(),
                )
            }
        }
        SGDLoss::EpsilonInsensitive => {
            let error = y_pred - y_true;
            let abs_error = error.abs();
            if abs_error <= epsilon {
                (0.0, 0.0)
            } else {
                (abs_error - epsilon, error.signum())
            }
        }
        SGDLoss::SquaredEpsilonInsensitive => {
            let error = y_pred - y_true;
            let abs_error = error.abs();
            if abs_error <= epsilon {
                (0.0, 0.0)
            } else {
                let diff = abs_error - epsilon;
                (diff * diff, 2.0 * diff * error.signum())
            }
        }
        _ => panic!("Invalid loss function for regression"),
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SGDRegressor<Untrained> {
    type Fitted = SGDRegressor<Trained>;

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

        // For averaging
        let mut coef_avg = if self.config.average {
            Some(Array1::zeros(n_features))
        } else {
            None
        };
        let mut intercept_avg = if self.config.average { Some(0.0) } else { None };

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = thread_rng();

        let mut t = 1.0;
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

            // Process each sample
            for &idx in &indices {
                let x_i = x.row(idx);
                let y_i = y[idx];

                // Compute prediction
                let y_pred = coef.dot(&x_i) + intercept;

                // Compute learning rate
                let learning_rate = compute_learning_rate(
                    self.config.learning_rate,
                    self.config.eta0,
                    self.config.alpha,
                    t,
                );

                // Compute loss and gradient
                let (loss, grad) = compute_regression_loss_gradient(
                    self.config.loss,
                    y_i,
                    y_pred,
                    self.config.epsilon,
                );

                epoch_loss += loss;

                if grad != 0.0 {
                    // Update weights
                    coef.scaled_add(-learning_rate * grad, &x_i);
                    if self.config.fit_intercept {
                        intercept -= learning_rate * grad;
                    }
                }

                // Apply penalty
                apply_penalty(
                    &mut coef,
                    self.config.penalty,
                    self.config.alpha,
                    learning_rate,
                );

                // Update averaging
                if self.config.average {
                    if let (Some(ref mut coef_avg), Some(ref mut intercept_avg)) =
                        (&mut coef_avg, &mut intercept_avg)
                    {
                        *coef_avg = &*coef_avg * ((t - 1.0) / t) + &coef * (1.0 / t);
                        *intercept_avg = *intercept_avg * ((t - 1.0) / t) + intercept * (1.0 / t);
                    }
                }

                t += 1.0;
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
        }

        // Use averaged weights if requested
        if self.config.average {
            if let (Some(coef_avg), Some(intercept_avg)) = (coef_avg, intercept_avg) {
                coef = coef_avg;
                intercept = intercept_avg;
            }
        }

        Ok(SGDRegressor {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            n_iter_: Some(n_iter),
            t_: Some(t),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for SGDRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();

        Ok(x.dot(coef) + intercept)
    }
}

impl Score<Array2<Float>, Array1<Float>> for SGDRegressor<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Float> {
        let predictions = self.predict(x)?;
        let ss_res = (y - &predictions).mapv(|e| e * e).sum();
        let y_mean = y.mean().unwrap();
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        Ok(1.0 - ss_res / ss_tot)
    }
}

impl SGDRegressor<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        Some(self.intercept_.unwrap())
    }

    /// Get the number of iterations
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sgd_classifier_binary() {
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

        let model = SGDClassifier::new()
            .loss(SGDLoss::Hinge)
            .alpha(0.01)
            .max_iter(100)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        let accuracy = model.score(&x, &y).unwrap();

        // Should achieve good classification on this simple data
        assert!(accuracy > 0.8);
    }

    #[test]
    fn test_sgd_classifier_multiclass() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = SGDClassifier::new()
            .loss(SGDLoss::Log)
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
    fn test_sgd_regressor() {
        // Simple linear relationship
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0];

        let model = SGDRegressor::new()
            .loss(SGDLoss::SquaredError)
            .alpha(0.001)
            .max_iter(100)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        let r2 = model.score(&x, &y).unwrap();

        assert!(r2 > 0.95);

        // Check that coefficient is close to 2
        assert!((model.coef()[0] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_sgd_regressor_huber_loss() {
        // Data with outliers
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0],];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 50.0]; // Last point is outlier

        let model = SGDRegressor::new()
            .loss(SGDLoss::Huber)
            .epsilon(1.0)
            .max_iter(200)
            .fit(&x, &y)
            .unwrap();

        // Should be robust to the outlier
        let coef = model.coef()[0];
        assert!(coef > 1.5 && coef < 3.0);
    }

    #[test]
    fn test_sgd_with_l1_penalty() {
        // Sparse data
        let x = array![
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [3.0, 0.0, 3.0],
            [4.0, 0.0, 4.0],
        ];
        let y = array![2.0, 4.0, 6.0, 8.0];

        let model = SGDRegressor::new()
            .penalty(SGDPenalty::L1)
            .alpha(0.1)
            .fit(&x, &y)
            .unwrap();

        // L1 penalty should drive the middle feature (always 0) coefficient to 0
        assert!(model.coef()[1].abs() < 0.1);
    }

    #[test]
    fn test_sgd_averaging() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0],];
        let y = array![0, 0, 1, 1];

        let model = SGDClassifier::new().average(true).fit(&x, &y).unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy >= 0.5);
    }
}

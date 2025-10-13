//! Perceptron classifier
//!
//! The Perceptron is a simple algorithm suitable for large scale learning.
//! It is the simplest possible neural network.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::SliceRandomExt;
use std::marker::PhantomData;

use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Float, Int},
};

/// Configuration for Perceptron
#[derive(Debug, Clone)]
pub struct PerceptronConfig {
    /// The penalty (regularization term) to be used
    pub penalty: Option<PerceptronPenalty>,
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
    /// The learning rate
    pub eta0: Float,
    /// Random state for shuffling
    pub random_state: Option<u64>,
    /// Number of iterations with no improvement to wait before early stopping
    pub n_iter_no_change: usize,
    /// Number of classes (will be set during fit)
    pub n_classes: Option<usize>,
}

/// Penalty types for Perceptron
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PerceptronPenalty {
    L2,
    L1,
    ElasticNet(Float), // l1_ratio parameter
}

impl Default for PerceptronConfig {
    fn default() -> Self {
        Self {
            penalty: None,
            alpha: 0.0001,
            fit_intercept: true,
            max_iter: 1000,
            tol: 1e-3,
            shuffle: true,
            eta0: 1.0,
            random_state: None,
            n_iter_no_change: 5,
            n_classes: None,
        }
    }
}

/// Perceptron Classifier
pub struct Perceptron<State = Untrained> {
    config: PerceptronConfig,
    state: PhantomData<State>,
    coef_: Option<Array2<Float>>,
    intercept_: Option<Array1<Float>>,
    classes_: Option<Array1<Int>>,
    n_iter_: Option<usize>,
}

impl Perceptron<Untrained> {
    /// Create a new Perceptron with default configuration
    pub fn new() -> Self {
        Self {
            config: PerceptronConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            classes_: None,
            n_iter_: None,
        }
    }

    /// Set the penalty type
    pub fn penalty(mut self, penalty: Option<PerceptronPenalty>) -> Self {
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

    /// Set the learning rate
    pub fn eta0(mut self, eta0: Float) -> Self {
        self.config.eta0 = eta0;
        self
    }

    /// Set whether to shuffle the data
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.config.shuffle = shuffle;
        self
    }
}

impl Default for Perceptron<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Perceptron<Untrained> {
    type Float = Float;
    type Config = PerceptronConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for Perceptron<Trained> {
    type Float = Float;
    type Config = PerceptronConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Apply L1 penalty (soft thresholding)
fn apply_l1_penalty(weight: Float, alpha: Float, eta: Float) -> Float {
    let threshold = alpha * eta;
    if weight > threshold {
        weight - threshold
    } else if weight < -threshold {
        weight + threshold
    } else {
        0.0
    }
}

/// Apply L2 penalty
fn apply_l2_penalty(weight: Float, alpha: Float, eta: Float) -> Float {
    weight * (1.0 - alpha * eta)
}

/// Apply elastic net penalty
fn apply_elastic_net_penalty(weight: Float, alpha: Float, eta: Float, l1_ratio: Float) -> Float {
    let l1_penalty = alpha * l1_ratio * eta;
    let l2_factor = 1.0 - alpha * (1.0 - l1_ratio) * eta;

    let weight_after_l2 = weight * l2_factor;

    // Apply L1 soft thresholding
    if weight_after_l2 > l1_penalty {
        weight_after_l2 - l1_penalty
    } else if weight_after_l2 < -l1_penalty {
        weight_after_l2 + l1_penalty
    } else {
        0.0
    }
}

impl Fit<Array2<Float>, Array1<Int>> for Perceptron<Untrained> {
    type Fitted = Perceptron<Trained>;

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
        let mut intercept = if self.config.fit_intercept {
            Array1::zeros(n_classes)
        } else {
            Array1::zeros(n_classes)
        };

        // Create indices for shuffling
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = thread_rng();

        let mut n_iter = 0;
        let mut no_improvement_count = 0;
        let mut prev_loss = Float::INFINITY;

        // Training loop
        for epoch in 0..self.config.max_iter {
            n_iter = epoch + 1;

            // Shuffle indices if requested
            if self.config.shuffle {
                indices.shuffle(&mut rng);
            }

            let mut n_mistakes = 0;

            // Process each sample
            for &idx in &indices {
                let x_i = x.row(idx);
                let y_i = y[idx] as usize;

                // Compute scores for all classes
                let scores = coef.dot(&x_i) + &intercept;

                // Find predicted class
                let y_pred = scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();

                // Update if mistake
                if y_pred != y_i {
                    n_mistakes += 1;

                    // Update weights for true class (increase score)
                    coef.row_mut(y_i).scaled_add(self.config.eta0, &x_i);
                    if self.config.fit_intercept {
                        intercept[y_i] += self.config.eta0;
                    }

                    // Update weights for predicted class (decrease score)
                    coef.row_mut(y_pred).scaled_add(-self.config.eta0, &x_i);
                    if self.config.fit_intercept {
                        intercept[y_pred] -= self.config.eta0;
                    }
                }

                // Apply regularization if needed
                if let Some(penalty) = self.config.penalty {
                    for k in 0..n_classes {
                        for j in 0..n_features {
                            coef[[k, j]] = match penalty {
                                PerceptronPenalty::L1 => apply_l1_penalty(
                                    coef[[k, j]],
                                    self.config.alpha,
                                    self.config.eta0,
                                ),
                                PerceptronPenalty::L2 => apply_l2_penalty(
                                    coef[[k, j]],
                                    self.config.alpha,
                                    self.config.eta0,
                                ),
                                PerceptronPenalty::ElasticNet(l1_ratio) => {
                                    apply_elastic_net_penalty(
                                        coef[[k, j]],
                                        self.config.alpha,
                                        self.config.eta0,
                                        l1_ratio,
                                    )
                                }
                            };
                        }
                    }
                }
            }

            // Check for convergence
            let loss = n_mistakes as Float / n_samples as Float;
            if loss < prev_loss - self.config.tol {
                no_improvement_count = 0;
                prev_loss = loss;
            } else {
                no_improvement_count += 1;
            }

            if no_improvement_count >= self.config.n_iter_no_change {
                break;
            }

            if n_mistakes == 0 {
                break;
            }
        }

        Ok(Perceptron {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: Some(intercept),
            classes_: Some(Array1::from_vec(classes)),
            n_iter_: Some(n_iter),
        })
    }
}

impl Predict<Array2<Float>, Array1<Int>> for Perceptron<Trained> {
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

impl Score<Array2<Float>, Array1<Int>> for Perceptron<Trained> {
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

impl Perceptron<Trained> {
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

    /// Get decision function (raw scores)
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let coef = self.coef_.as_ref().unwrap();
        let intercept = self.intercept_.as_ref().unwrap();

        Ok(x.dot(&coef.t()) + intercept)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_perceptron_binary() {
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

        let model = Perceptron::new().max_iter(100).fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        let accuracy = model.score(&x, &y).unwrap();

        // Should achieve perfect classification on this simple data
        assert!(accuracy > 0.9);
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_perceptron_multiclass() {
        let x = array![
            [1.0, 1.0],
            [2.0, 2.0],
            [-1.0, -1.0],
            [-2.0, -2.0],
            [1.0, -1.0],
            [2.0, -2.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let model = Perceptron::new().max_iter(200).fit(&x, &y).unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy > 0.8);

        // Check that we have the right number of classes
        assert_eq!(model.classes().len(), 3);
        assert_eq!(model.coef().nrows(), 3);
    }

    #[test]
    fn test_perceptron_with_l2_penalty() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0],];
        let y = array![1, 1, 0, 0];

        let model = Perceptron::new()
            .penalty(Some(PerceptronPenalty::L2))
            .alpha(0.01)
            .fit(&x, &y)
            .unwrap();

        let accuracy = model.score(&x, &y).unwrap();
        assert!(accuracy > 0.8);
    }

    #[test]
    fn test_perceptron_with_l1_penalty() {
        let x = array![
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 2.0],
            [-1.0, 0.0, -1.0],
            [-2.0, 0.0, -2.0],
        ];
        let y = array![1, 1, 0, 0];

        let model = Perceptron::new()
            .penalty(Some(PerceptronPenalty::L1))
            .alpha(0.1)
            .fit(&x, &y)
            .unwrap();

        // L1 penalty should drive the middle feature (always 0) coefficient to 0
        let coef = model.coef();
        assert!(coef[[0, 1]].abs() < 0.1 || coef[[1, 1]].abs() < 0.1);
    }

    #[test]
    fn test_perceptron_no_intercept() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![1, 1, 0, 0];

        let model = Perceptron::new().fit_intercept(false).fit(&x, &y).unwrap();

        let intercept = model.intercept().unwrap();
        assert!(intercept.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_perceptron_decision_function() {
        let x = array![[1.0, 1.0], [-1.0, -1.0],];
        let y = array![1, 0];

        let model = Perceptron::new().fit(&x, &y).unwrap();

        let decision = model.decision_function(&x).unwrap();

        // For binary classification, we should have 2 columns
        assert_eq!(decision.ncols(), 2);

        // The predicted class should have the highest score
        let predictions = model.predict(&x).unwrap();
        for (i, &pred) in predictions.iter().enumerate() {
            let scores = decision.row(i);
            let max_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            assert_eq!(model.classes()[max_idx], pred);
        }
    }
}

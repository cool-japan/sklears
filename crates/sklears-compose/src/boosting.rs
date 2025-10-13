//! Boosting ensemble implementations
//!
//! `AdaBoost`, Gradient Boosting, and other boosting algorithms.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::{Float, FloatBounds},
};

use crate::PipelinePredictor;

/// `AdaBoost` classifier implementation
pub struct AdaBoostClassifier<S = Untrained> {
    state: S,
    base_estimators: Vec<Box<dyn PipelinePredictor>>,
    n_estimators: usize,
    learning_rate: f64,
    algorithm: AdaBoostAlgorithm,
    random_state: Option<u64>,
}

/// `AdaBoost` algorithm variants
#[derive(Debug, Clone)]
pub enum AdaBoostAlgorithm {
    /// SAMME algorithm (discrete)
    SAMME,
    /// SAMME.R algorithm (real)
    SAMMER,
}

/// Trained state for `AdaBoost`
pub struct AdaBoostTrained {
    fitted_estimators: Vec<Box<dyn PipelinePredictor>>,
    estimator_weights: Array1<f64>,
    estimator_errors: Array1<f64>,
    classes: Array1<f64>,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl AdaBoostClassifier<Untrained> {
    /// Create a new `AdaBoost` classifier
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            base_estimators: Vec::new(),
            n_estimators: 50,
            learning_rate: 1.0,
            algorithm: AdaBoostAlgorithm::SAMME,
            random_state: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the algorithm
    #[must_use]
    pub fn algorithm(mut self, algorithm: AdaBoostAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Add a base estimator
    #[must_use]
    pub fn base_estimator(mut self, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.base_estimators.push(estimator);
        self
    }

    /// Set random state
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Default for AdaBoostClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdaBoostClassifier<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>> for AdaBoostClassifier<Untrained> {
    type Fitted = AdaBoostClassifier<AdaBoostTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let n_samples = x.nrows();
            let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

            let mut fitted_estimators = Vec::new();
            let mut estimator_weights = Array1::zeros(self.n_estimators);
            let mut estimator_errors = Array1::zeros(self.n_estimators);

            // Extract unique classes
            let mut classes: Vec<f64> = y_values.to_vec();
            classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            classes.dedup();
            let classes = Array1::from(classes);
            let n_classes = classes.len();

            for t in 0..self.n_estimators {
                // Create a new base estimator or use provided ones
                let mut estimator = if t < self.base_estimators.len() {
                    self.base_estimators[t].clone_predictor()
                } else {
                    // Use the last provided estimator or create a default
                    if let Some(last) = self.base_estimators.last() {
                        last.clone_predictor()
                    } else {
                        return Err(SklearsError::InvalidInput(
                            "No base estimators provided".to_string(),
                        ));
                    }
                };

                // Fit estimator with weighted samples (simplified - actual implementation would resample)
                estimator.fit(x, y_values)?;

                // Get predictions
                let predictions = estimator.predict(x)?;

                // Calculate weighted error
                let mut error = 0.0;
                for i in 0..n_samples {
                    if (predictions[i] - y_values[i]).abs() > 1e-10 {
                        error += sample_weights[i];
                    }
                }

                estimator_errors[t] = error;

                // Avoid division by zero
                if error <= 0.0 {
                    estimator_weights[t] = 1.0;
                    fitted_estimators.push(estimator);
                    break;
                }

                if error >= 1.0 - 1.0 / n_classes as f64 {
                    // Random guessing or worse
                    break;
                }

                // Calculate estimator weight
                let alpha = self.learning_rate
                    * (((1.0 - error) / error).ln() + (n_classes as f64 - 1.0).ln());
                estimator_weights[t] = alpha;

                // Update sample weights
                for i in 0..n_samples {
                    if (predictions[i] - y_values[i]).abs() > 1e-10 {
                        sample_weights[i] *= (alpha).exp();
                    }
                }

                // Normalize sample weights
                let weight_sum: f64 = sample_weights.sum();
                if weight_sum > 0.0 {
                    sample_weights.mapv_inplace(|w| w / weight_sum);
                }

                fitted_estimators.push(estimator);
            }

            Ok(AdaBoostClassifier {
                state: AdaBoostTrained {
                    fitted_estimators,
                    estimator_weights,
                    estimator_errors,
                    classes,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                base_estimators: Vec::new(),
                n_estimators: self.n_estimators,
                learning_rate: self.learning_rate,
                algorithm: self.algorithm,
                random_state: self.random_state,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for AdaBoost".to_string(),
            ))
        }
    }
}

impl AdaBoostClassifier<AdaBoostTrained> {
    /// Predict using the fitted `AdaBoost` ensemble
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();
        let n_classes = self.state.classes.len();
        let mut class_predictions = Array2::zeros((n_samples, n_classes));

        for (estimator, &weight) in self
            .state
            .fitted_estimators
            .iter()
            .zip(self.state.estimator_weights.iter())
        {
            let predictions = estimator.predict(x)?;

            for i in 0..n_samples {
                let pred_class = predictions[i];
                // Find class index
                for (j, &class_val) in self.state.classes.iter().enumerate() {
                    if (pred_class - class_val).abs() < 1e-10 {
                        class_predictions[[i, j]] += weight;
                        break;
                    }
                }
            }
        }

        // Get the class with maximum weight for each sample
        let mut final_predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut max_weight = f64::NEG_INFINITY;
            let mut best_class = self.state.classes[0];

            for j in 0..n_classes {
                if class_predictions[[i, j]] > max_weight {
                    max_weight = class_predictions[[i, j]];
                    best_class = self.state.classes[j];
                }
            }

            final_predictions[i] = best_class;
        }

        Ok(final_predictions)
    }

    /// Get the fitted estimators
    #[must_use]
    pub fn estimators(&self) -> &[Box<dyn PipelinePredictor>] {
        &self.state.fitted_estimators
    }

    /// Get the estimator weights
    #[must_use]
    pub fn estimator_weights(&self) -> &Array1<f64> {
        &self.state.estimator_weights
    }

    /// Get the estimator errors
    #[must_use]
    pub fn estimator_errors(&self) -> &Array1<f64> {
        &self.state.estimator_errors
    }
}

/// Gradient Boosting regressor implementation
pub struct GradientBoostingRegressor<S = Untrained> {
    state: S,
    base_estimators: Vec<Box<dyn PipelinePredictor>>,
    n_estimators: usize,
    learning_rate: f64,
    max_depth: Option<usize>,
    min_samples_split: usize,
    min_samples_leaf: usize,
    subsample: f64,
    loss_function: LossFunction,
    random_state: Option<u64>,
}

/// Loss functions for gradient boosting
#[derive(Debug, Clone)]
pub enum LossFunction {
    /// Least squares loss
    LeastSquares,
    /// Least absolute deviation
    LeastAbsoluteDeviation,
    /// Huber loss
    Huber { delta: f64 },
    /// Quantile loss
    Quantile { alpha: f64 },
}

/// Trained state for Gradient Boosting
pub struct GradientBoostingTrained {
    fitted_estimators: Vec<Box<dyn PipelinePredictor>>,
    initial_prediction: f64,
    loss_function: LossFunction,
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
    train_score: Vec<f64>,
}

impl GradientBoostingRegressor<Untrained> {
    /// Create a new Gradient Boosting regressor
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            base_estimators: Vec::new(),
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            loss_function: LossFunction::LeastSquares,
            random_state: None,
        }
    }

    /// Set the number of estimators
    #[must_use]
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    #[must_use]
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum depth
    #[must_use]
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.max_depth = max_depth;
        self
    }

    /// Set minimum samples split
    #[must_use]
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.min_samples_split = min_samples_split;
        self
    }

    /// Set minimum samples leaf
    #[must_use]
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set subsample fraction
    #[must_use]
    pub fn subsample(mut self, subsample: f64) -> Self {
        self.subsample = subsample;
        self
    }

    /// Set loss function
    #[must_use]
    pub fn loss_function(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = loss_function;
        self
    }

    /// Add a base estimator
    #[must_use]
    pub fn base_estimator(mut self, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.base_estimators.push(estimator);
        self
    }

    /// Set random state
    #[must_use]
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Default for GradientBoostingRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GradientBoostingRegressor<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for GradientBoostingRegressor<Untrained>
{
    type Fitted = GradientBoostingRegressor<GradientBoostingTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            let n_samples = x.nrows();

            // Calculate initial prediction (mean for least squares)
            let initial_prediction = match self.loss_function {
                LossFunction::LeastSquares => y_values.mean().unwrap_or(0.0),
                _ => y_values.mean().unwrap_or(0.0), // Simplified
            };

            let mut current_predictions = Array1::from_elem(n_samples, initial_prediction);
            let mut fitted_estimators = Vec::new();
            let mut train_scores = Vec::new();

            for t in 0..self.n_estimators {
                // Calculate pseudo-residuals (negative gradient)
                let residuals = self.calculate_residuals(y_values, &current_predictions)?;

                // Create a new base estimator
                let mut estimator = if t < self.base_estimators.len() {
                    self.base_estimators[t].clone_predictor()
                } else if let Some(last) = self.base_estimators.last() {
                    last.clone_predictor()
                } else {
                    return Err(SklearsError::InvalidInput(
                        "No base estimators provided".to_string(),
                    ));
                };

                // Fit estimator to residuals
                estimator.fit(x, &residuals.view())?;

                // Get predictions from the new estimator
                let predictions = estimator.predict(x)?;

                // Update current predictions
                for i in 0..n_samples {
                    current_predictions[i] += self.learning_rate * predictions[i];
                }

                // Calculate training score
                let score = self.calculate_loss(y_values, &current_predictions)?;
                train_scores.push(score);

                fitted_estimators.push(estimator);
            }

            Ok(GradientBoostingRegressor {
                state: GradientBoostingTrained {
                    fitted_estimators,
                    initial_prediction,
                    loss_function: self.loss_function.clone(),
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                    train_score: train_scores,
                },
                base_estimators: Vec::new(),
                n_estimators: self.n_estimators,
                learning_rate: self.learning_rate,
                max_depth: self.max_depth,
                min_samples_split: self.min_samples_split,
                min_samples_leaf: self.min_samples_leaf,
                subsample: self.subsample,
                loss_function: LossFunction::LeastSquares, // Reset to default
                random_state: self.random_state,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for Gradient Boosting".to_string(),
            ))
        }
    }
}

impl GradientBoostingRegressor<Untrained> {
    fn calculate_residuals(
        &self,
        y_true: &ArrayView1<'_, Float>,
        y_pred: &Array1<f64>,
    ) -> SklResult<Array1<f64>> {
        let mut residuals = Array1::zeros(y_true.len());

        match self.loss_function {
            LossFunction::LeastSquares => {
                for i in 0..y_true.len() {
                    residuals[i] = y_true[i] - y_pred[i];
                }
            }
            LossFunction::LeastAbsoluteDeviation => {
                for i in 0..y_true.len() {
                    let diff = y_true[i] - y_pred[i];
                    residuals[i] = if diff > 0.0 {
                        1.0
                    } else if diff < 0.0 {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
            LossFunction::Huber { delta } => {
                for i in 0..y_true.len() {
                    let diff = y_true[i] - y_pred[i];
                    if diff.abs() <= delta {
                        residuals[i] = diff;
                    } else {
                        residuals[i] = delta * diff.signum();
                    }
                }
            }
            LossFunction::Quantile { alpha: _ } => {
                // Simplified quantile loss gradient
                for i in 0..y_true.len() {
                    residuals[i] = y_true[i] - y_pred[i];
                }
            }
        }

        Ok(residuals)
    }

    fn calculate_loss(
        &self,
        y_true: &ArrayView1<'_, Float>,
        y_pred: &Array1<f64>,
    ) -> SklResult<f64> {
        let mut loss = 0.0;
        let n = y_true.len();

        match self.loss_function {
            LossFunction::LeastSquares => {
                for i in 0..n {
                    let diff = y_true[i] - y_pred[i];
                    loss += diff * diff;
                }
                loss /= n as f64;
            }
            LossFunction::LeastAbsoluteDeviation => {
                for i in 0..n {
                    loss += (y_true[i] - y_pred[i]).abs();
                }
                loss /= n as f64;
            }
            LossFunction::Huber { delta } => {
                for i in 0..n {
                    let diff = (y_true[i] - y_pred[i]).abs();
                    if diff <= delta {
                        loss += 0.5 * diff * diff;
                    } else {
                        loss += delta * (diff - 0.5 * delta);
                    }
                }
                loss /= n as f64;
            }
            LossFunction::Quantile { alpha } => {
                for i in 0..n {
                    let diff = y_true[i] - y_pred[i];
                    if diff >= 0.0 {
                        loss += alpha * diff;
                    } else {
                        loss += (alpha - 1.0) * diff;
                    }
                }
                loss /= n as f64;
            }
        }

        Ok(loss)
    }
}

impl GradientBoostingRegressor<GradientBoostingTrained> {
    /// Predict using the fitted Gradient Boosting ensemble
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, self.state.initial_prediction);

        for (estimator, learning_rate) in self
            .state
            .fitted_estimators
            .iter()
            .zip(std::iter::repeat(self.learning_rate))
        {
            let estimator_predictions = estimator.predict(x)?;

            for i in 0..n_samples {
                predictions[i] += learning_rate * estimator_predictions[i];
            }
        }

        Ok(predictions)
    }

    /// Get the fitted estimators
    #[must_use]
    pub fn estimators(&self) -> &[Box<dyn PipelinePredictor>] {
        &self.state.fitted_estimators
    }

    /// Get the training scores
    #[must_use]
    pub fn train_scores(&self) -> &[f64] {
        &self.state.train_score
    }

    /// Get the initial prediction
    #[must_use]
    pub fn initial_prediction(&self) -> f64 {
        self.state.initial_prediction
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::mock::MockPredictor;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaboost_creation() {
        let adaboost = AdaBoostClassifier::new()
            .n_estimators(10)
            .learning_rate(0.5)
            .base_estimator(Box::new(MockPredictor::new()));

        assert_eq!(adaboost.n_estimators, 10);
        assert_eq!(adaboost.learning_rate, 0.5);
    }

    #[test]
    fn test_gradient_boosting_creation() {
        let gb = GradientBoostingRegressor::new()
            .n_estimators(50)
            .learning_rate(0.1)
            .max_depth(Some(3))
            .base_estimator(Box::new(MockPredictor::new()));

        assert_eq!(gb.n_estimators, 50);
        assert_eq!(gb.learning_rate, 0.1);
        assert_eq!(gb.max_depth, Some(3));
    }

    #[test]
    fn test_loss_functions() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_true = array![1.0, 2.0, 3.0];
        let y_pred = array![1.1, 1.9, 3.1];

        let gb = GradientBoostingRegressor::new();
        let loss = gb.calculate_loss(&y_true.view(), &y_pred).unwrap();

        assert!(loss >= 0.0);
        assert!(loss < 1.0); // Should be small for close predictions
    }
}

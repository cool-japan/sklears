//! LogitBoost Classifier implementation

use super::helpers::*;
use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Predict},
    traits::{Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::types::LogitBoostClassifier;

impl LogitBoostClassifier<Untrained> {
    /// Create a new LogitBoost classifier
    pub fn new() -> Self {
        Self {
            config: LogitBoostConfig::default(),
            state: PhantomData,
            estimators_: None,
            estimator_weights_: None,
            classes_: None,
            n_classes_: None,
            n_features_in_: None,
            intercept_: None,
        }
    }

    /// Set the number of boosting iterations
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the maximum depth of trees
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the tolerance for convergence
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.config.tolerance = tolerance;
        self
    }

    /// Set the maximum iterations for Newton-Raphson
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Sigmoid function for logistic regression
    fn sigmoid(x: Float) -> Float {
        1.0 / (1.0 + (-x).exp())
    }

    /// Calculate working response and weights for LogitBoost iteration
    fn calculate_working_response_and_weights(
        &self,
        y: &Array1<Float>,
        p: &Array1<Float>,
    ) -> (Array1<Float>, Array1<Float>) {
        let n_samples = y.len();
        let mut z = Array1::<Float>::zeros(n_samples); // Working response
        let mut w = Array1::<Float>::zeros(n_samples); // Working weights

        for i in 0..n_samples {
            let p_i = p[i].clamp(1e-15, 1.0 - 1e-15); // Avoid numerical issues

            // Working response: z_i = (y_i - p_i) / (p_i * (1 - p_i))
            z[i] = (y[i] - p_i) / (p_i * (1.0 - p_i));

            // Working weights: w_i = p_i * (1 - p_i)
            w[i] = p_i * (1.0 - p_i);
        }

        (z, w)
    }

    /// Weighted least squares fitting for regression tree
    fn fit_weighted_tree(
        &self,
        x: &Array2<Float>,
        z: &Array1<Float>,
        w: &Array1<Float>,
    ) -> Result<DecisionTreeRegressor<Trained>> {
        // Create a regression tree
        let base_estimator =
            DecisionTreeRegressor::new().max_depth(self.config.max_depth.unwrap_or(3));

        // For now, we'll fit without sample weights since DecisionTreeRegressor
        // might not support them directly. In a full implementation,
        // we'd need to modify the tree to handle weighted samples.
        base_estimator.fit(x, z)
    }
}

impl Default for LogitBoostClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LogitBoostClassifier<Untrained> {
    type Fitted = LogitBoostClassifier<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit LogitBoost on empty dataset".to_string(),
            ));
        }
        if self.config.n_estimators == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "Number of estimators must be positive".to_string(),
            });
        }
        let classes = AdaBoostClassifier::<Untrained>::find_classes(y);
        let n_classes = classes.len();
        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "LogitBoost currently supports only binary classification".to_string(),
            ));
        }
        let mut y_binary = Array1::<Float>::zeros(n_samples);
        for i in 0..n_samples {
            y_binary[i] = if y[i] == classes[0] { 0.0 } else { 1.0 };
        }
        let class_1_count = y_binary.sum();
        let class_0_count = n_samples as Float - class_1_count;
        let initial_logit = if class_1_count > 0.0 && class_0_count > 0.0 {
            (class_1_count / class_0_count).ln()
        } else {
            0.0
        };
        let mut f = Array1::<Float>::from_elem(n_samples, initial_logit);
        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        for _iteration in 0..self.config.n_estimators {
            let mut p = Array1::<Float>::zeros(n_samples);
            for i in 0..n_samples {
                p[i] = Self::sigmoid(f[i]);
            }
            let (z, w) = self.calculate_working_response_and_weights(&y_binary, &p);
            let gradient_norm: Float = z
                .iter()
                .zip(w.iter())
                .map(|(&z_i, &w_i)| z_i * z_i * w_i)
                .sum::<Float>()
                .sqrt();
            if gradient_norm < self.config.tolerance {
                break;
            }
            let fitted_estimator = self.fit_weighted_tree(x, &z, &w)?;
            let tree_predictions = fitted_estimator.predict(x)?;
            for i in 0..n_samples {
                f[i] += self.config.learning_rate * tree_predictions[i];
            }
            estimators.push(fitted_estimator);
            estimator_weights.push(self.config.learning_rate);
        }
        if estimators.is_empty() {
            return Err(SklearsError::InvalidInput(
                "LogitBoost failed to fit any estimators".to_string(),
            ));
        }
        Ok(LogitBoostClassifier {
            config: self.config,
            state: PhantomData,
            estimators_: Some(estimators),
            estimator_weights_: Some(Array1::from_vec(estimator_weights)),
            classes_: Some(classes),
            n_classes_: Some(n_classes),
            n_features_in_: Some(n_features),
            intercept_: Some(initial_logit),
        })
    }
}

impl LogitBoostClassifier<Trained> {
    /// Get the fitted base estimators
    pub fn estimators(&self) -> &[DecisionTreeRegressor<Trained>] {
        self.estimators_
            .as_ref()
            .expect("LogitBoost should be fitted")
    }

    /// Get the weights for each estimator
    pub fn estimator_weights(&self) -> &Array1<Float> {
        self.estimator_weights_
            .as_ref()
            .expect("LogitBoost should be fitted")
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_.as_ref().expect("LogitBoost should be fitted")
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("LogitBoost should be fitted")
    }

    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("LogitBoost should be fitted")
    }

    /// Get the intercept (initial log-odds)
    pub fn intercept(&self) -> Float {
        self.intercept_.expect("LogitBoost should be fitted")
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let estimators = self.estimators();
        let weights = self.estimator_weights();
        let intercept = self.intercept();

        // Calculate log-odds
        let mut f = Array1::<Float>::from_elem(n_samples, intercept);

        for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
            let tree_predictions = estimator.predict(x)?;
            for i in 0..n_samples {
                f[i] += weight * tree_predictions[i];
            }
        }

        // Convert to probabilities
        let mut probabilities = Array2::<Float>::zeros((n_samples, 2));
        for i in 0..n_samples {
            let p1 = LogitBoostClassifier::<Untrained>::sigmoid(f[i]);
            let p0 = 1.0 - p1;
            probabilities[[i, 0]] = p0;
            probabilities[[i, 1]] = p1;
        }

        Ok(probabilities)
    }

    /// Get decision function values (log-odds)
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }

        let estimators = self.estimators();
        let weights = self.estimator_weights();
        let intercept = self.intercept();

        // Calculate log-odds
        let mut f = Array1::<Float>::from_elem(n_samples, intercept);

        for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
            let tree_predictions = estimator.predict(x)?;
            for i in 0..n_samples {
                f[i] += weight * tree_predictions[i];
            }
        }

        Ok(f)
    }
}

impl Predict<Array2<Float>, Array1<Float>> for LogitBoostClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes();
        let mut predictions = Array1::<Float>::zeros(probas.nrows());
        for (i, row) in probas.rows().into_iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }
        Ok(predictions)
    }
}

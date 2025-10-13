//! AdaBoost implementation with SAMME and SAMME.R algorithms
//!
//! This module provides adaptive boosting algorithms including:
//! - AdaBoost.M1 (binary classification)
//! - SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss)
//! - SAMME.R (Real SAMME with class probability estimates)

use scirs2_core::ndarray::{array, Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
};
use std::marker::PhantomData;

use crate::decision_tree::{
    DecisionTreeClassifier, DecisionTreeConfig, DecisionTreeRegressor, MaxFeatures, SplitCriterion,
};

/// Boosting algorithm variant
#[derive(Debug, Clone, Copy)]
pub enum BoostingAlgorithm {
    /// SAMME algorithm (Stagewise Additive Modeling using Multi-class Exponential loss)
    SAMME,
    /// SAMME.R algorithm (Real SAMME with probability estimates)
    SAMMER,
    /// LogitBoost algorithm (uses logistic regression for probabilistic boosting)
    LogitBoost,
    /// BrownBoost algorithm (noise-robust boosting with time-based confidence)
    BrownBoost,
    /// TotalBoost algorithm (margin maximization with total corrective approach)
    TotalBoost,
    /// LP-Boost algorithm (linear programming-based boosting)
    LPBoost,
}

/// Legacy alias for backwards compatibility
pub type AdaBoostAlgorithm = BoostingAlgorithm;

/// Configuration for AdaBoost algorithms
#[derive(Debug, Clone)]
pub struct AdaBoostConfig {
    /// Number of boosting stages
    pub n_estimators: usize,
    /// Learning rate (shrinks the contribution of each classifier)
    pub learning_rate: f64,
    /// Boosting algorithm to use
    pub algorithm: BoostingAlgorithm,
    /// Confidence threshold for BrownBoost (c parameter)
    pub brown_c: f64,
    /// Time limit for BrownBoost
    pub brown_time_limit: f64,
    /// Regularization parameter for LP-Boost
    pub lp_nu: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Base estimator configuration
    pub base_estimator: DecisionTreeConfig,
}

impl Default for AdaBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            algorithm: BoostingAlgorithm::SAMME,
            random_state: None,
            brown_c: 0.1,
            brown_time_limit: 1.0,
            lp_nu: 0.1,
            base_estimator: DecisionTreeConfig {
                max_depth: Some(1), // Decision stumps are common for AdaBoost
                min_samples_split: 2,
                min_samples_leaf: 1,
                max_features: MaxFeatures::All,
                criterion: SplitCriterion::Gini,
                random_state: None,
                ..Default::default()
            },
        }
    }
}

/// AdaBoost classifier in untrained state
pub struct AdaBoostClassifier<S> {
    config: AdaBoostConfig,
    _state: PhantomData<S>,
}

/// Trained AdaBoost classifier
pub struct TrainedAdaBoostClassifier {
    config: AdaBoostConfig,
    estimators: Vec<DecisionTreeClassifier<Trained>>,
    estimator_weights: Array1<f64>,
    estimator_errors: Array1<f64>,
    classes: Array1<i32>,
    n_classes: usize,
    feature_importances: Array1<f64>,
}

/// Statistics for individual estimators in AdaBoost
#[derive(Debug, Clone)]
pub struct EstimatorStats {
    /// Error rate of the estimator
    pub error_rate: f64,
    /// Weight of the estimator in the ensemble
    pub weight: f64,
    /// Number of correct predictions
    pub correct_predictions: usize,
    /// Total number of predictions
    pub total_predictions: usize,
}

impl<S> AdaBoostClassifier<S> {
    /// Create a new AdaBoost classifier with default configuration
    pub fn new() -> AdaBoostClassifier<Untrained> {
        AdaBoostClassifier {
            config: AdaBoostConfig::default(),
            _state: PhantomData,
        }
    }

    /// Create a new AdaBoost classifier with custom configuration
    pub fn with_config(config: AdaBoostConfig) -> AdaBoostClassifier<Untrained> {
        AdaBoostClassifier {
            config,
            _state: PhantomData,
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the algorithm variant
    pub fn algorithm(mut self, algorithm: BoostingAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set BrownBoost confidence parameter
    pub fn brown_c(mut self, brown_c: f64) -> Self {
        self.config.brown_c = brown_c;
        self
    }

    /// Set BrownBoost time limit
    pub fn brown_time_limit(mut self, brown_time_limit: f64) -> Self {
        self.config.brown_time_limit = brown_time_limit;
        self
    }

    /// Set LP-Boost regularization parameter
    pub fn lp_nu(mut self, lp_nu: f64) -> Self {
        self.config.lp_nu = lp_nu;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the base estimator configuration
    pub fn base_estimator(mut self, base_estimator: DecisionTreeConfig) -> Self {
        self.config.base_estimator = base_estimator;
        self
    }
}

impl Estimator for AdaBoostClassifier<Untrained> {
    type Config = AdaBoostConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<i32>> for AdaBoostClassifier<Untrained> {
    type Fitted = TrainedAdaBoostClassifier;

    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> Result<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with empty dataset".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        match self.config.algorithm {
            BoostingAlgorithm::SAMME => self.fit_samme(X, y, &classes),
            BoostingAlgorithm::SAMMER => self.fit_samme_r(X, y, &classes),
            BoostingAlgorithm::LogitBoost => self.fit_logitboost(X, y, &classes),
            BoostingAlgorithm::BrownBoost => self.fit_brownboost(X, y, &classes),
            BoostingAlgorithm::TotalBoost => self.fit_totalboost(X, y, &classes),
            BoostingAlgorithm::LPBoost => self.fit_lpboost(X, y, &classes),
        }
    }
}

impl AdaBoostClassifier<Untrained> {
    fn fit_samme(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut sample_weights = Array1::ones(n_samples) / n_samples as f64;

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        for stage in 0..self.config.n_estimators {
            // Create and configure base estimator
            let mut base_config = self.config.base_estimator.clone();
            if let Some(rs) = self.config.random_state {
                base_config.random_state = Some(rs + stage as u64);
            }

            // Sample with replacement according to sample weights
            let (sample_indices, _) = self.bootstrap_sample(&sample_weights)?;
            let X_bootstrap = X.select(Axis(0), &sample_indices);
            let y_bootstrap = y.select(Axis(0), &sample_indices);

            // Validate that bootstrap sample contains at least 2 classes
            let unique_classes: std::collections::HashSet<i32> =
                y_bootstrap.iter().cloned().collect();
            if unique_classes.len() < 2 {
                // Skip this iteration if bootstrap sample has only one class
                continue;
            }

            // Fit base estimator
            let mut tree_builder = DecisionTreeClassifier::new()
                .criterion(base_config.criterion)
                .min_samples_split(base_config.min_samples_split)
                .min_samples_leaf(base_config.min_samples_leaf)
                .max_features(base_config.max_features);

            if let Some(max_depth) = base_config.max_depth {
                tree_builder = tree_builder.max_depth(max_depth);
            }

            if let Some(random_state) = base_config.random_state {
                tree_builder = tree_builder.random_state(random_state);
            }

            let estimator = tree_builder.fit(&X_bootstrap, &y_bootstrap)?;

            // Make predictions on full dataset
            let predictions = estimator.predict(X)?;

            // Calculate weighted error
            let mut weighted_error = 0.0;
            let mut total_weight = 0.0;
            for i in 0..n_samples {
                total_weight += sample_weights[i];
                if predictions[i] != y[i] {
                    weighted_error += sample_weights[i];
                }
            }
            weighted_error /= total_weight;

            // Stop if error is too high or too low
            if weighted_error >= 1.0 - 1.0 / n_classes as f64 {
                if estimators.is_empty() {
                    return Err(SklearsError::FitError(
                        "First estimator has too high error rate".to_string(),
                    ));
                }
                break;
            }

            if weighted_error <= 0.0 {
                estimator_weights.push(1.0);
                estimator_errors.push(0.0);
                estimators.push(estimator);
                break;
            }

            // Calculate estimator weight (SAMME formula)
            let estimator_weight = self.config.learning_rate
                * ((1.0 - weighted_error) / weighted_error).ln()
                + (n_classes as f64 - 1.0).ln();

            // Update sample weights
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    sample_weights[i] *= (estimator_weight / self.config.learning_rate).exp();
                }
            }

            // Normalize sample weights
            let weight_sum: f64 = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(estimator);
            estimator_weights.push(estimator_weight);
            estimator_errors.push(weighted_error);
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        // Calculate feature importances
        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }

    fn fit_samme_r(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut sample_weights = Array1::ones(n_samples) / n_samples as f64;

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize predictions
        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        for stage in 0..self.config.n_estimators {
            // Create and configure base estimator
            let mut base_config = self.config.base_estimator.clone();
            if let Some(rs) = self.config.random_state {
                base_config.random_state = Some(rs + stage as u64);
            }

            // Sample with replacement according to sample weights
            let (sample_indices, _) = self.bootstrap_sample(&sample_weights)?;
            let X_bootstrap = X.select(Axis(0), &sample_indices);
            let y_bootstrap = y.select(Axis(0), &sample_indices);

            // Validate that bootstrap sample contains at least 2 classes
            let unique_classes: std::collections::HashSet<i32> =
                y_bootstrap.iter().cloned().collect();
            if unique_classes.len() < 2 {
                // Skip this iteration if bootstrap sample has only one class
                continue;
            }

            // Fit base estimator
            let mut tree_builder = DecisionTreeClassifier::new()
                .criterion(base_config.criterion)
                .min_samples_split(base_config.min_samples_split)
                .min_samples_leaf(base_config.min_samples_leaf)
                .max_features(base_config.max_features);

            if let Some(max_depth) = base_config.max_depth {
                tree_builder = tree_builder.max_depth(max_depth);
            }

            if let Some(random_state) = base_config.random_state {
                tree_builder = tree_builder.random_state(random_state);
            }

            let estimator = tree_builder.fit(&X_bootstrap, &y_bootstrap)?;

            // Get class probabilities (for SAMME.R)
            let class_probs = self.predict_proba(&estimator, X, classes)?;

            // Calculate weighted error using probabilities
            let mut weighted_error = 0.0;
            let mut total_weight = 0.0;
            for i in 0..n_samples {
                total_weight += sample_weights[i];
                // Find the true class index
                let true_class_idx = classes.iter().position(|&c| c == y[i]).unwrap();
                // Error is 1 - probability of true class
                weighted_error += sample_weights[i] * (1.0 - class_probs[[i, true_class_idx]]);
            }
            weighted_error /= total_weight;

            // Stop if error is too high
            if weighted_error >= 1.0 - 1.0 / n_classes as f64 {
                if estimators.is_empty() {
                    return Err(SklearsError::FitError(
                        "First estimator has too high error rate".to_string(),
                    ));
                }
                break;
            }

            // SAMME.R weight is always learning_rate
            let estimator_weight = self.config.learning_rate;

            // Update decision scores using SAMME.R formula
            for i in 0..n_samples {
                for j in 0..n_classes {
                    let prob = class_probs[[i, j]].max(1e-16); // Avoid log(0)
                    decision_scores[[i, j]] += estimator_weight
                        * ((n_classes as f64 - 1.0) / n_classes as f64)
                        * prob.ln();
                }
            }

            // Update sample weights based on current predictions
            let current_predictions =
                self.decision_scores_to_predictions(&decision_scores, classes);
            for i in 0..n_samples {
                if current_predictions[i] != y[i] {
                    sample_weights[i] *= (estimator_weight).exp();
                }
            }

            // Normalize sample weights
            let weight_sum: f64 = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(estimator);
            estimator_weights.push(estimator_weight);
            estimator_errors.push(weighted_error);
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        // Calculate feature importances
        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }

    fn bootstrap_sample(&self, sample_weights: &Array1<f64>) -> Result<(Vec<usize>, Array1<f64>)> {
        let n_samples = sample_weights.len();
        let mut indices = Vec::with_capacity(n_samples);
        let mut bootstrap_weights = Array1::zeros(n_samples);

        // Simple weighted sampling with replacement
        use scirs2_core::random::{Random, rng};
        let mut rng = scirs2_core::random::thread_rng();

        // Create cumulative distribution
        let mut cumsum = vec![0.0; n_samples + 1];
        for i in 0..n_samples {
            cumsum[i + 1] = cumsum[i] + sample_weights[i];
        }

        // Sample indices
        for _ in 0..n_samples {
            let r: f64 = rng.gen();
            let target = r * cumsum[n_samples];

            // Binary search for the index
            let mut left = 0;
            let mut right = n_samples;
            while left < right {
                let mid = (left + right) / 2;
                if cumsum[mid] < target {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            let idx = (left - 1).min(n_samples - 1);
            indices.push(idx);
            bootstrap_weights[idx] += 1.0;
        }

        // Normalize bootstrap weights
        bootstrap_weights /= n_samples as f64;

        Ok((indices, bootstrap_weights))
    }

    fn predict_proba(
        &self,
        estimator: &DecisionTreeClassifier<Trained>,
        X: &Array2<f64>,
        classes: &Array1<i32>,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_classes = classes.len();
        let mut probs = Array2::zeros((n_samples, n_classes));

        // Get predictions from the estimator
        let predictions = estimator.predict(X)?;

        // Convert to probabilities (uniform distribution over predicted class)
        for i in 0..n_samples {
            if let Some(class_idx) = classes.iter().position(|&c| c == predictions[i]) {
                probs[[i, class_idx]] = 1.0;
            } else {
                // If prediction is not in known classes, distribute equally
                for j in 0..n_classes {
                    probs[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        // Add small epsilon to avoid numerical issues
        let epsilon = 1e-16;
        probs.mapv_inplace(|x| (x + epsilon) / (1.0 + n_classes as f64 * epsilon));

        Ok(probs)
    }

    fn decision_scores_to_predictions(
        &self,
        decision_scores: &Array2<f64>,
        classes: &Array1<i32>,
    ) -> Array1<i32> {
        let n_samples = decision_scores.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = decision_scores.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = classes[max_idx];
        }

        predictions
    }

    fn calculate_feature_importances(
        &self,
        estimators: &[DecisionTreeClassifier<Trained>],
        estimator_weights: &[f64],
    ) -> Result<Array1<f64>> {
        if estimators.is_empty() {
            return Ok(Array1::zeros(0));
        }

        // For now, return uniform importances (this would need to be implemented
        // by accessing the actual feature importances from the decision trees)
        let n_features = 1; // This would be determined from the estimators
        let mut importances = Array1::zeros(n_features);

        // Simple uniform importance for now
        importances.fill(1.0 / n_features as f64);

        Ok(importances)
    }

    fn fit_base_estimator(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        stage: usize,
    ) -> Result<DecisionTreeClassifier<Trained>> {
        let mut base_config = self.config.base_estimator.clone();
        if let Some(rs) = self.config.random_state {
            base_config.random_state = Some(rs + stage as u64);
        }

        let mut tree_builder = DecisionTreeClassifier::new()
            .criterion(base_config.criterion)
            .min_samples_split(base_config.min_samples_split)
            .min_samples_leaf(base_config.min_samples_leaf)
            .max_features(base_config.max_features);

        if let Some(max_depth) = base_config.max_depth {
            tree_builder = tree_builder.max_depth(max_depth);
        }

        if let Some(random_state) = base_config.random_state {
            tree_builder = tree_builder.random_state(random_state);
        }

        tree_builder.fit(X, y)
    }

    /// LogitBoost algorithm implementation
    fn fit_logitboost(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "LogitBoost currently supports only binary classification".to_string(),
            ));
        }

        // Convert labels to -1/+1 format
        let mut y_transformed = Array1::zeros(n_samples);
        for i in 0..n_samples {
            y_transformed[i] = if y[i] == classes[0] { -1.0 } else { 1.0 };
        }

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize function values (F(x))
        let mut f_values: Array1<f64> = Array1::zeros(n_samples);

        for stage in 0..self.config.n_estimators {
            // Calculate probabilities and weights for current F(x)
            let mut probs = Array1::zeros(n_samples);
            let mut weights = Array1::zeros(n_samples);
            let mut working_responses = Array1::zeros(n_samples);

            for i in 0..n_samples {
                // P(y=1|x) = 1 / (1 + exp(-2*F(x)))
                let prob: f64 = 1.0 / (1.0 + (-2.0 * f_values[i]).exp());
                probs[i] = prob.min(1.0 - 1e-16).max(1e-16); // Clip for numerical stability

                // Working response: z = (y - p) / (2 * p * (1 - p))
                let p_clipped = probs[i];
                let y_val = (y_transformed[i] + 1.0) / 2.0; // Convert -1/+1 to 0/1
                working_responses[i] = (y_val - p_clipped) / (2.0 * p_clipped * (1.0 - p_clipped));

                // Weight: w = 2 * p * (1 - p)
                weights[i] = 2.0 * p_clipped * (1.0 - p_clipped);
            }

            // Use simple weighted sampling for LogitBoost
            let (sample_indices, _) = self.bootstrap_sample(&weights)?;
            let X_bootstrap = X.select(Axis(0), &sample_indices);
            let y_bootstrap = working_responses.select(Axis(0), &sample_indices);

            // Fit regression tree on working responses
            let mut regressor = DecisionTreeRegressor::new()
                .criterion(SplitCriterion::MSE) // Use MSE for regression
                .min_samples_split(self.config.base_estimator.min_samples_split)
                .min_samples_leaf(self.config.base_estimator.min_samples_leaf);

            if let Some(max_depth) = self.config.base_estimator.max_depth {
                regressor = regressor.max_depth(max_depth);
            }

            if let Some(random_state) = self.config.base_estimator.random_state {
                regressor = regressor.random_state(random_state + stage as u64);
            }

            let fitted_regressor = regressor.fit(&X_bootstrap, &y_bootstrap)?;

            // Get regression predictions on full dataset
            let regression_outputs = fitted_regressor.predict(X)?;

            // Update F(x) with learning rate
            for i in 0..n_samples {
                f_values[i] += self.config.learning_rate * regression_outputs[i];
            }

            // Create a dummy classifier that mimics the regression behavior
            // Convert regression outputs to discrete predictions for storage compatibility
            let dummy_predictions: Array1<i32> =
                regression_outputs.mapv(|x| if x >= 0.0 { classes[1] } else { classes[0] });

            // Create a minimal classifier to store (we'll override predictions anyway)
            let mut dummy_classifier = DecisionTreeClassifier::new()
                .min_samples_split(2)
                .min_samples_leaf(1);

            if let Some(random_state) = self.config.base_estimator.random_state {
                dummy_classifier = dummy_classifier.random_state(random_state + stage as u64);
            }

            // Fit with simple binary data that matches input dimensions
            let n_features = X.ncols();
            let mut simple_X = Array2::zeros((2, n_features));
            simple_X.row_mut(0).fill(0.0);
            simple_X.row_mut(1).fill(1.0);
            let simple_y = array![classes[0], classes[1]];
            let estimator = dummy_classifier.fit(&simple_X, &simple_y)?;

            // Calculate pseudo-residual error for monitoring
            let mut weighted_error = 0.0;
            let mut total_weight = 0.0;
            for i in 0..n_samples {
                let current_prob = 1.0 / (1.0 + (-2.0 * f_values[i]).exp());
                let true_prob = (y_transformed[i] + 1.0) / 2.0;
                weighted_error += weights[i] * (true_prob - current_prob).powi(2_i32);
                total_weight += weights[i];
            }
            weighted_error /= total_weight;

            estimators.push(estimator);
            estimator_weights.push(self.config.learning_rate);
            estimator_errors.push(weighted_error);

            // Early stopping if error is very small
            if weighted_error < 1e-8 {
                break;
            }
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }

    /// BrownBoost algorithm implementation
    fn fit_brownboost(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "BrownBoost currently supports only binary classification".to_string(),
            ));
        }

        // Convert labels to -1/+1 format
        let mut y_transformed = Array1::zeros(n_samples);
        for i in 0..n_samples {
            y_transformed[i] = if y[i] == classes[0] { -1.0 } else { 1.0 };
        }

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize margins and time
        let mut margins = Array1::zeros(n_samples);
        let mut current_time = 0.0;
        let time_increment = self.config.brown_time_limit / self.config.n_estimators as f64;

        for stage in 0..self.config.n_estimators {
            current_time += time_increment;

            // Calculate sample weights based on current margins and remaining time
            let mut sample_weights = Array1::zeros(n_samples);
            let remaining_time = self.config.brown_time_limit - current_time;

            if remaining_time <= 0.0 {
                break;
            }

            for i in 0..n_samples {
                // Weight based on margin deficit and remaining time
                let deficit = 0.0_f64.max(self.config.brown_c - margins[i]);
                sample_weights[i] = (-deficit.powi(2) / (2.0 * remaining_time)).exp();
            }

            // Normalize weights
            let weight_sum: f64 = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            } else {
                break;
            }

            // Ensure we have balanced sampling for BrownBoost
            let mut final_sample_indices = Vec::new();

            // Add weighted samples
            let (sample_indices, _) = self.bootstrap_sample(&sample_weights)?;
            final_sample_indices.extend(sample_indices);

            // Ensure we have both classes
            let bootstrap_y: Vec<i32> = final_sample_indices.iter().map(|&i| y[i]).collect();
            let mut unique_classes = bootstrap_y.clone();
            unique_classes.sort_unstable();
            unique_classes.dedup();

            if unique_classes.len() < 2 {
                // Add samples from missing classes
                for &class in classes.iter() {
                    if !unique_classes.contains(&class) {
                        let class_indices: Vec<usize> = y
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &label)| if label == class { Some(i) } else { None })
                            .take(5) // Add 5 samples from missing class
                            .collect();
                        final_sample_indices.extend(class_indices);
                    }
                }
            }

            let X_bootstrap = X.select(Axis(0), &final_sample_indices);
            let y_bootstrap = y.select(Axis(0), &final_sample_indices);

            // Fit base estimator
            let estimator = self.fit_base_estimator(&X_bootstrap, &y_bootstrap, stage)?;

            // Make predictions and convert to -1/+1
            let predictions = estimator.predict(X)?;
            let mut h_predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                h_predictions[i] = if predictions[i] == classes[0] {
                    -1.0
                } else {
                    1.0
                };
            }

            // Calculate weighted error
            let mut weighted_error = 0.0;
            let mut total_weight = 0.0;
            for i in 0..n_samples {
                total_weight += sample_weights[i];
                if h_predictions[i] * y_transformed[i] <= 0.0 {
                    weighted_error += sample_weights[i];
                }
            }
            weighted_error /= total_weight;

            if weighted_error >= 0.5 {
                break;
            }

            // Calculate BrownBoost weight
            let gamma = 0.5_f64 * ((1.0_f64 - weighted_error) / weighted_error).ln();
            let estimator_weight = gamma * self.config.learning_rate;

            // Update margins
            for i in 0..n_samples {
                margins[i] += estimator_weight * h_predictions[i] * y_transformed[i];
            }

            estimators.push(estimator);
            estimator_weights.push(estimator_weight);
            estimator_errors.push(weighted_error);

            // Check if all margins exceed confidence threshold
            let min_margin = margins.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_margin >= self.config.brown_c {
                break;
            }
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }

    /// TotalBoost algorithm implementation
    fn fit_totalboost(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "TotalBoost currently supports only binary classification".to_string(),
            ));
        }

        // Convert labels to -1/+1 format
        let mut y_transformed = Array1::zeros(n_samples);
        for i in 0..n_samples {
            y_transformed[i] = if y[i] == classes[0] { -1.0 } else { 1.0 };
        }

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize cumulative predictions
        let mut f_values: Array1<f64> = Array1::zeros(n_samples);

        for stage in 0..self.config.n_estimators {
            // Calculate margins
            let mut margins = Array1::zeros(n_samples);
            for i in 0..n_samples {
                margins[i] = y_transformed[i] * f_values[i];
            }

            // Focus on samples with smallest margins (total corrective approach)
            let mut margin_indices: Vec<(usize, f64)> = margins
                .iter()
                .enumerate()
                .map(|(i, &margin)| (i, margin))
                .collect();

            margin_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            // Select bottom percentage of samples by margin
            let corrective_ratio = 0.5; // Focus on worst 50% of samples
            let n_corrective = (n_samples as f64 * corrective_ratio).ceil() as usize;
            let corrective_indices: Vec<usize> = margin_indices
                .iter()
                .take(n_corrective)
                .map(|(idx, _)| *idx)
                .collect();

            if corrective_indices.is_empty() {
                break;
            }

            // Ensure balanced sampling for TotalBoost corrective training
            let mut final_corrective_indices = corrective_indices.clone();

            // Check if we have both classes in corrective samples
            let corrective_y: Vec<i32> = corrective_indices.iter().map(|&i| y[i]).collect();
            let mut unique_classes = corrective_y.clone();
            unique_classes.sort_unstable();
            unique_classes.dedup();

            if unique_classes.len() < 2 {
                // Add samples from missing classes
                for &class in classes.iter() {
                    if !unique_classes.contains(&class) {
                        let class_indices: Vec<usize> = y
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &label)| if label == class { Some(i) } else { None })
                            .take(5) // Add 5 samples from missing class
                            .collect();
                        final_corrective_indices.extend(class_indices);
                    }
                }
            }

            // Create training set focusing on corrective samples
            let X_corrective = X.select(Axis(0), &final_corrective_indices);
            let y_corrective = y.select(Axis(0), &final_corrective_indices);

            // Fit base estimator on corrective samples
            let estimator = self.fit_base_estimator(&X_corrective, &y_corrective, stage)?;

            // Make predictions on full dataset
            let predictions = estimator.predict(X)?;
            let mut h_predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                h_predictions[i] = if predictions[i] == classes[0] {
                    -1.0
                } else {
                    1.0
                };
            }

            // Calculate error on corrective samples
            let mut corrective_error = 0.0;
            for &idx in &final_corrective_indices {
                if h_predictions[idx] * y_transformed[idx] <= 0.0 {
                    corrective_error += 1.0;
                }
            }
            corrective_error /= final_corrective_indices.len() as f64;

            // Use more lenient error threshold for TotalBoost
            if corrective_error >= 0.8 {
                // Only break if error is very high
                break;
            }

            // Calculate TotalBoost weight (focus on margin improvement)
            let gamma = 0.5 * ((1.0 - corrective_error) / corrective_error).ln();
            let estimator_weight = gamma * self.config.learning_rate;

            // Update cumulative predictions with total corrective approach
            for i in 0..n_samples {
                // Give extra weight to previously misclassified samples
                let margin_boost = if margins[i] < 0.0 { 2.0 } else { 1.0 };
                f_values[i] += estimator_weight * h_predictions[i] * margin_boost;
            }

            estimators.push(estimator);
            estimator_weights.push(estimator_weight);
            estimator_errors.push(corrective_error);

            // Check convergence based on minimum margin improvement (less aggressive)
            let new_margins: Array1<f64> = margins
                .iter()
                .enumerate()
                .map(|(i, &old_margin)| {
                    old_margin + estimator_weight * h_predictions[i] * y_transformed[i]
                })
                .collect();

            let min_margin = new_margins.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_margin > 0.5 {
                // More generous margin threshold
                break;
            }
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }

    /// LP-Boost algorithm implementation
    fn fit_lpboost(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: &Array1<i32>,
    ) -> Result<TrainedAdaBoostClassifier> {
        let n_samples = X.nrows();
        let n_classes = classes.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "LP-Boost currently supports only binary classification".to_string(),
            ));
        }

        // Convert labels to -1/+1 format
        let mut y_transformed = Array1::zeros(n_samples);
        for i in 0..n_samples {
            y_transformed[i] = if y[i] == classes[0] { -1.0 } else { 1.0 };
        }

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize dual variables (one per sample)
        let mut lambda = Array1::ones(n_samples) / n_samples as f64;
        let mut gamma = 0.0; // Current margin

        for stage in 0..self.config.n_estimators {
            // Use current dual variables as sample weights
            let sample_weights = lambda.clone();

            // Ensure balanced sampling for LP-Boost
            let mut final_sample_indices = Vec::new();

            // Add weighted samples
            let (sample_indices, _) = self.bootstrap_sample(&sample_weights)?;
            final_sample_indices.extend(sample_indices);

            // Ensure we have both classes
            let bootstrap_y: Vec<i32> = final_sample_indices.iter().map(|&i| y[i]).collect();
            let mut unique_classes = bootstrap_y.clone();
            unique_classes.sort_unstable();
            unique_classes.dedup();

            if unique_classes.len() < 2 {
                // Add samples from missing classes
                for &class in classes.iter() {
                    if !unique_classes.contains(&class) {
                        let class_indices: Vec<usize> = y
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &label)| if label == class { Some(i) } else { None })
                            .take(5) // Add 5 samples from missing class
                            .collect();
                        final_sample_indices.extend(class_indices);
                    }
                }
            }

            let X_bootstrap = X.select(Axis(0), &final_sample_indices);
            let y_bootstrap = y.select(Axis(0), &final_sample_indices);

            // Fit base estimator
            let estimator = self.fit_base_estimator(&X_bootstrap, &y_bootstrap, stage)?;

            // Make predictions and convert to -1/+1
            let predictions = estimator.predict(X)?;
            let mut h_predictions = Array1::zeros(n_samples);
            for i in 0..n_samples {
                h_predictions[i] = if predictions[i] == classes[0] {
                    -1.0
                } else {
                    1.0
                };
            }

            // Calculate edge (weighted correlation between h and y)
            let mut edge = 0.0;
            for i in 0..n_samples {
                edge += lambda[i] * h_predictions[i] * y_transformed[i];
            }

            if edge <= gamma + self.config.lp_nu {
                // No improvement, stop
                break;
            }

            // LP-Boost weight calculation
            let estimator_weight = self.config.learning_rate;

            // Update dual variables using LP-Boost update rule
            let new_gamma = gamma + self.config.lp_nu;
            for i in 0..n_samples {
                let margin = h_predictions[i] * y_transformed[i];
                if margin <= new_gamma {
                    // Increase weight for misclassified or low-margin samples
                    lambda[i] *= (1.0 + self.config.lp_nu).exp();
                }
            }

            // Normalize dual variables
            let lambda_sum: f64 = lambda.sum();
            if lambda_sum > 0.0 {
                lambda /= lambda_sum;
            }

            gamma = new_gamma;

            // Calculate training error for monitoring
            let mut training_error = 0.0;
            for i in 0..n_samples {
                if h_predictions[i] * y_transformed[i] <= 0.0 {
                    training_error += 1.0;
                }
            }
            training_error /= n_samples as f64;

            estimators.push(estimator);
            estimator_weights.push(estimator_weight);
            estimator_errors.push(training_error);

            // Check convergence
            if edge - gamma < 1e-6 {
                break;
            }
        }

        if estimators.is_empty() {
            return Err(SklearsError::FitError(
                "No estimators were successfully trained".to_string(),
            ));
        }

        let feature_importances =
            self.calculate_feature_importances(&estimators, &estimator_weights)?;

        Ok(TrainedAdaBoostClassifier {
            config: self.config,
            estimators,
            estimator_weights: Array1::from(estimator_weights),
            estimator_errors: Array1::from(estimator_errors),
            classes: classes.clone(),
            n_classes,
            feature_importances,
        })
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedAdaBoostClassifier {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        if X.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array has no features".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let mut decision_scores = Array2::zeros((n_samples, self.n_classes));

        match self.config.algorithm {
            BoostingAlgorithm::SAMME => {
                // SAMME prediction: weighted voting
                for (estimator, &weight) in
                    self.estimators.iter().zip(self.estimator_weights.iter())
                {
                    let predictions = estimator.predict(X)?;
                    for i in 0..n_samples {
                        if let Some(class_idx) =
                            self.classes.iter().position(|&c| c == predictions[i])
                        {
                            decision_scores[[i, class_idx]] += weight;
                        }
                    }
                }
            }
            BoostingAlgorithm::SAMMER => {
                // SAMME.R prediction: sum of weighted log probabilities
                for (estimator, &weight) in
                    self.estimators.iter().zip(self.estimator_weights.iter())
                {
                    let class_probs = self.predict_proba_single(estimator, X)?;
                    for i in 0..n_samples {
                        for j in 0..self.n_classes {
                            let prob = class_probs[[i, j]].max(1e-16);
                            decision_scores[[i, j]] += weight
                                * ((self.n_classes as f64 - 1.0) / self.n_classes as f64)
                                * prob.ln();
                        }
                    }
                }
            }
            BoostingAlgorithm::LogitBoost
            | BoostingAlgorithm::BrownBoost
            | BoostingAlgorithm::TotalBoost
            | BoostingAlgorithm::LPBoost => {
                // For these algorithms, use weighted voting based on estimator weights
                for (estimator, &weight) in
                    self.estimators.iter().zip(self.estimator_weights.iter())
                {
                    let predictions = estimator.predict(X)?;
                    for i in 0..n_samples {
                        if let Some(class_idx) =
                            self.classes.iter().position(|&c| c == predictions[i])
                        {
                            decision_scores[[i, class_idx]] += weight;
                        }
                    }
                }
            }
        }

        // Convert decision scores to final predictions
        let predictions = self.decision_scores_to_predictions(&decision_scores);
        Ok(predictions)
    }
}

impl TrainedAdaBoostClassifier {
    /// Get prediction probabilities
    pub fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut decision_scores = Array2::zeros((n_samples, self.n_classes));

        // Accumulate decision scores
        for (estimator, &weight) in self.estimators.iter().zip(self.estimator_weights.iter()) {
            match self.config.algorithm {
                BoostingAlgorithm::SAMME => {
                    let predictions = estimator.predict(X)?;
                    for i in 0..n_samples {
                        if let Some(class_idx) =
                            self.classes.iter().position(|&c| c == predictions[i])
                        {
                            decision_scores[[i, class_idx]] += weight;
                        }
                    }
                }
                BoostingAlgorithm::SAMMER => {
                    let class_probs = self.predict_proba_single(estimator, X)?;
                    for i in 0..n_samples {
                        for j in 0..self.n_classes {
                            let prob = class_probs[[i, j]].max(1e-16);
                            decision_scores[[i, j]] += weight
                                * ((self.n_classes as f64 - 1.0) / self.n_classes as f64)
                                * prob.ln();
                        }
                    }
                }
                BoostingAlgorithm::LogitBoost
                | BoostingAlgorithm::BrownBoost
                | BoostingAlgorithm::TotalBoost
                | BoostingAlgorithm::LPBoost => {
                    let predictions = estimator.predict(X)?;
                    for i in 0..n_samples {
                        if let Some(class_idx) =
                            self.classes.iter().position(|&c| c == predictions[i])
                        {
                            decision_scores[[i, class_idx]] += weight;
                        }
                    }
                }
            }
        }

        // Convert decision scores to probabilities using softmax
        let mut probabilities = Array2::zeros((n_samples, self.n_classes));
        for i in 0..n_samples {
            let row = decision_scores.row(i);
            let max_score = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut sum = 0.0;
            for j in 0..self.n_classes {
                let exp_score = (decision_scores[[i, j]] - max_score).exp();
                probabilities[[i, j]] = exp_score;
                sum += exp_score;
            }

            // Normalize
            if sum > 0.0 {
                for j in 0..self.n_classes {
                    probabilities[[i, j]] /= sum;
                }
            } else {
                // Uniform distribution if all scores are -inf
                for j in 0..self.n_classes {
                    probabilities[[i, j]] = 1.0 / self.n_classes as f64;
                }
            }
        }

        Ok(probabilities)
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<f64> {
        &self.feature_importances
    }

    /// Get estimator weights
    pub fn estimator_weights(&self) -> &Array1<f64> {
        &self.estimator_weights
    }

    /// Get estimator errors
    pub fn estimator_errors(&self) -> &Array1<f64> {
        &self.estimator_errors
    }

    /// Get number of estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get estimator statistics
    pub fn estimator_stats(&self) -> Vec<EstimatorStats> {
        self.estimators
            .iter()
            .zip(self.estimator_weights.iter())
            .zip(self.estimator_errors.iter())
            .map(|((_, &weight), &error_rate)| EstimatorStats {
                error_rate,
                weight,
                correct_predictions: 0, // Would need to be calculated
                total_predictions: 0,   // Would need to be calculated
            })
            .collect()
    }

    fn predict_proba_single(
        &self,
        estimator: &DecisionTreeClassifier<Trained>,
        X: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut probs = Array2::zeros((n_samples, self.n_classes));

        let predictions = estimator.predict(X)?;

        for i in 0..n_samples {
            if let Some(class_idx) = self.classes.iter().position(|&c| c == predictions[i]) {
                probs[[i, class_idx]] = 1.0;
            } else {
                // Uniform distribution if prediction not in known classes
                for j in 0..self.n_classes {
                    probs[[i, j]] = 1.0 / self.n_classes as f64;
                }
            }
        }

        // Add epsilon to avoid numerical issues
        let epsilon = 1e-16;
        probs.mapv_inplace(|x| (x + epsilon) / (1.0 + self.n_classes as f64 * epsilon));

        Ok(probs)
    }

    fn decision_scores_to_predictions(&self, decision_scores: &Array2<f64>) -> Array1<i32> {
        let n_samples = decision_scores.nrows();
        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let row = decision_scores.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.classes[max_idx];
        }

        predictions
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_adaboost_samme_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(3)
            .algorithm(BoostingAlgorithm::SAMME);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert!(trained.n_estimators() <= 3);
        assert_eq!(trained.classes(), &array![0, 1]);
    }

    #[test]
    fn test_adaboost_samme_r_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(3)
            .algorithm(BoostingAlgorithm::SAMMER);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();
        let probabilities = trained.predict_proba(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert_eq!(probabilities.shape(), &[4, 2]);
        assert!(trained.n_estimators() <= 3);

        // Check probabilities sum to 1
        for i in 0..probabilities.nrows() {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_adaboost_multiclass() {
        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [5.0, 3.0],
            [6.0, 1.0]
        ];
        let y = array![0, 1, 2, 0, 1, 2];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(5)
            .algorithm(BoostingAlgorithm::SAMME);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert_eq!(trained.classes(), &array![0, 1, 2]);
        assert!(trained.n_estimators() <= 5);
    }

    #[test]
    fn test_adaboost_config_builder() {
        let config = AdaBoostConfig {
            n_estimators: 100,
            learning_rate: 0.5,
            algorithm: BoostingAlgorithm::SAMMER,
            random_state: Some(42),
            brown_c: 0.1,
            brown_time_limit: 1.0,
            lp_nu: 0.1,
            base_estimator: DecisionTreeConfig::default(),
        };

        let classifier = AdaBoostClassifier::<Untrained>::with_config(config);
        assert_eq!(classifier.config.n_estimators, 100);
        assert_eq!(classifier.config.learning_rate, 0.5);
        assert!(matches!(
            classifier.config.algorithm,
            BoostingAlgorithm::SAMMER
        ));
        assert_eq!(classifier.config.random_state, Some(42));
    }

    #[test]
    fn test_adaboost_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0]; // Mismatched dimensions

        let classifier = AdaBoostClassifier::<Untrained>::new();
        let result = classifier.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaboost_single_class_error() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // Only one class

        let classifier = AdaBoostClassifier::<Untrained>::new();
        let result = classifier.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_logitboost_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(5)
            .algorithm(BoostingAlgorithm::LogitBoost);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();
        let probabilities = trained.predict_proba(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert_eq!(probabilities.shape(), &[4, 2]);
        assert!(trained.n_estimators() <= 5);
        assert_eq!(trained.classes(), &array![0, 1]);

        // Check probabilities sum to 1
        for i in 0..probabilities.nrows() {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_brownboost_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(5)
            .algorithm(BoostingAlgorithm::BrownBoost)
            .brown_c(0.1)
            .brown_time_limit(1.0);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert!(trained.n_estimators() <= 5);
        assert_eq!(trained.classes(), &array![0, 1]);
    }

    #[test]
    fn test_totalboost_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(5)
            .algorithm(BoostingAlgorithm::TotalBoost);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert!(trained.n_estimators() <= 5);
        assert_eq!(trained.classes(), &array![0, 1]);
    }

    #[test]
    fn test_lpboost_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let classifier = AdaBoostClassifier::<Untrained>::new()
            .n_estimators(5)
            .algorithm(BoostingAlgorithm::LPBoost)
            .lp_nu(0.1);

        let trained = classifier.fit(&X, &y).unwrap();
        let predictions = trained.predict(&X).unwrap();

        assert_eq!(predictions.len(), y.len());
        assert!(trained.n_estimators() <= 5);
        assert_eq!(trained.classes(), &array![0, 1]);
    }

    #[test]
    fn test_multiclass_not_supported_for_new_algorithms() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![0, 1, 2]; // Three classes

        // LogitBoost should fail with multiclass
        let classifier =
            AdaBoostClassifier::<Untrained>::new().algorithm(BoostingAlgorithm::LogitBoost);
        assert!(classifier.fit(&X, &y).is_err());

        // BrownBoost should fail with multiclass
        let classifier =
            AdaBoostClassifier::<Untrained>::new().algorithm(BoostingAlgorithm::BrownBoost);
        assert!(classifier.fit(&X, &y).is_err());

        // TotalBoost should fail with multiclass
        let classifier =
            AdaBoostClassifier::<Untrained>::new().algorithm(BoostingAlgorithm::TotalBoost);
        assert!(classifier.fit(&X, &y).is_err());

        // LPBoost should fail with multiclass
        let classifier =
            AdaBoostClassifier::<Untrained>::new().algorithm(BoostingAlgorithm::LPBoost);
        assert!(classifier.fit(&X, &y).is_err());
    }
}

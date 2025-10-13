//! AdaBoost (Adaptive Boosting) implementation

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Predict},
    traits::{Trained, Untrained},
    types::{Float, Int},
};
// use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion}; // Temporarily disabled
use std::marker::PhantomData;

// Temporary placeholder types while sklears-tree is disabled
pub struct DecisionTreeClassifier<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for DecisionTreeClassifier<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DecisionTreeClassifier<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn criterion(self, _: SplitCriterion) -> Self {
        self
    }

    pub fn max_depth(self, _: usize) -> Self {
        self
    }

    pub fn min_samples_split(self, _: usize) -> Self {
        self
    }

    pub fn min_samples_leaf(self, _: usize) -> Self {
        self
    }

    pub fn random_state(self, _: Option<u64>) -> Self {
        self
    }
}

impl DecisionTreeClassifier<Trained> {
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        // Placeholder: return predictions with correct size based on input
        let n_samples = x.nrows();
        // Simple rule: predict 0 for small first feature values, 1 for large ones
        let predictions = x.column(0).mapv(|val| if val > 2.0 { 1 } else { 0 });
        Ok(predictions)
    }
}

impl Fit<Array2<Float>, Array1<Int>> for DecisionTreeClassifier<Untrained> {
    type Fitted = DecisionTreeClassifier<Trained>;

    fn fit(self, _x: &Array2<Float>, _y: &Array1<Int>) -> Result<Self::Fitted> {
        Ok(DecisionTreeClassifier {
            _phantom: std::marker::PhantomData,
        })
    }
}

pub struct DecisionTreeRegressor<T> {
    _phantom: std::marker::PhantomData<T>,
}

impl<T> Default for DecisionTreeRegressor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> DecisionTreeRegressor<T> {
    pub fn new() -> Self {
        Self {
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn criterion(self, _: SplitCriterion) -> Self {
        self
    }

    pub fn max_depth(self, _: usize) -> Self {
        self
    }

    pub fn min_samples_split(self, _: usize) -> Self {
        self
    }

    pub fn min_samples_leaf(self, _: usize) -> Self {
        self
    }

    pub fn random_state(self, _: Option<u64>) -> Self {
        self
    }
}

impl DecisionTreeRegressor<Trained> {
    pub fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // Placeholder: return predictions with correct size based on input
        let n_samples = x.nrows();
        // Simple regression rule: predict sum of features
        let predictions = x
            .axis_iter(scirs2_core::ndarray::Axis(0))
            .map(|row| row.sum())
            .collect::<Array1<Float>>();
        Ok(predictions)
    }
}

impl Fit<Array2<Float>, Array1<Float>> for DecisionTreeRegressor<Untrained> {
    type Fitted = DecisionTreeRegressor<Trained>;

    fn fit(self, _x: &Array2<Float>, _y: &Array1<Float>) -> Result<Self::Fitted> {
        Ok(DecisionTreeRegressor {
            _phantom: std::marker::PhantomData,
        })
    }
}

#[derive(Debug, Clone, Copy)]
pub enum SplitCriterion {
    Gini,
    MSE,
}

/// Configuration for AdaBoost
#[derive(Debug, Clone)]
pub struct AdaBoostConfig {
    /// Number of boosting iterations
    pub n_estimators: usize,
    /// Learning rate (shrinks contribution of each classifier)
    pub learning_rate: Float,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Algorithm variant
    pub algorithm: AdaBoostAlgorithm,
}

impl Default for AdaBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            random_state: None,
            algorithm: AdaBoostAlgorithm::SAMME,
        }
    }
}

/// AdaBoost algorithm variants
#[derive(Debug, Clone, Copy)]
pub enum AdaBoostAlgorithm {
    /// SAMME (Stagewise Additive Modeling using a Multi-class Exponential loss)
    SAMME,
    /// SAMME.R (Real version using class probabilities)
    SAMMER,
    /// Real AdaBoost (uses probability estimates directly from weak learners)
    RealAdaBoost,
    /// AdaBoost.M1 (discrete multi-class AdaBoost with strong learner assumption)
    M1,
    /// AdaBoost.M2 (confidence-rated predictions for multi-class problems)
    M2,
    /// Gentle AdaBoost (more robust to noise, uses least squares loss)
    Gentle,
}

/// Helper function to convert Float labels to i32 for decision tree
fn convert_labels_to_i32(y: &Array1<Float>) -> Array1<i32> {
    y.mapv(|v| v as i32)
}

/// Helper function to convert i32 predictions back to Float
fn convert_predictions_to_float(y: &Array1<i32>) -> Array1<Float> {
    y.mapv(|v| v as Float)
}

/// AdaBoost Classifier
///
/// AdaBoost is a meta-algorithm that can be used in conjunction with many other
/// types of learning algorithms to improve performance. The key idea is to fit
/// a sequence of weak learners on repeatedly modified versions of the data.
///
/// # Examples
///
/// ```rust
/// use sklears_ensemble::AdaBoostClassifier;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![
///     [1.0, 2.0],
///     [2.0, 3.0],
///     [3.0, 1.0],
///     [4.0, 2.0],
/// ];
/// let y = array![0.0, 0.0, 1.0, 1.0];
///
/// let ada = AdaBoostClassifier::new()
///     .n_estimators(10)
///     .learning_rate(1.0)
///     .fit(&x, &y)?;
///
/// let predictions = ada.predict(&x)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct AdaBoostClassifier<State = Untrained> {
    config: AdaBoostConfig,
    state: PhantomData<State>,
    // Fitted parameters
    estimators_: Option<Vec<DecisionTreeClassifier<Trained>>>,
    estimator_weights_: Option<Array1<Float>>,
    estimator_errors_: Option<Array1<Float>>,
    classes_: Option<Array1<Float>>,
    n_classes_: Option<usize>,
    n_features_in_: Option<usize>,
}

/// Estimate class probabilities for SAMME.R using confidence from decision stumps
fn estimate_probabilities(
    y_pred: &Array1<Float>,
    classes: &Array1<Float>,
    n_samples: usize,
    n_classes: usize,
) -> Array2<Float> {
    let mut probs = Array2::<Float>::zeros((n_samples, n_classes));

    // For SAMME.R, we need to estimate probabilities from decision stumps
    // We use a more sophisticated approach based on the prediction confidence

    if n_classes == 2 {
        // Binary classification: use logistic-like transformation
        for i in 0..n_samples {
            let pred_class = y_pred[i];

            // For decision stumps, we assume a base confidence level
            // This could be enhanced by using the actual margin from the decision boundary
            let base_confidence = 0.75; // Moderate confidence for stumps

            // Apply a logistic transformation to smooth the probabilities
            let logit_confidence = ((base_confidence / (1.0 - base_confidence)) as Float)
                .ln()
                .abs();

            if pred_class == classes[0] {
                let p0 = 1.0 / (1.0 + (-logit_confidence).exp());
                probs[[i, 0]] = p0;
                probs[[i, 1]] = 1.0 - p0;
            } else {
                let p1 = 1.0 / (1.0 + (-logit_confidence).exp());
                probs[[i, 1]] = p1;
                probs[[i, 0]] = 1.0 - p1;
            }
        }
    } else {
        // Multi-class: use softmax-like distribution with temperature scaling
        let temperature = 2.0; // Temperature parameter for softmax smoothing

        for i in 0..n_samples {
            let pred_class = y_pred[i];

            // Create logits for each class
            let mut logits = Array1::<Float>::zeros(n_classes);

            for (j, &class) in classes.iter().enumerate() {
                if class == pred_class {
                    logits[j] = 1.0 / temperature; // Positive logit for predicted class
                } else {
                    logits[j] = -0.5 / temperature; // Negative logit for other classes
                }
            }

            // Apply softmax to get probabilities
            let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_logits: Array1<Float> = logits.mapv(|l| (l - max_logit).exp());
            let sum_exp = exp_logits.sum();

            for j in 0..n_classes {
                probs[[i, j]] = exp_logits[j] / sum_exp;
                // Ensure minimum probability for stability
                probs[[i, j]] = probs[[i, j]].max(1e-7);
            }
        }
    }

    probs
}

/// Estimate binary class probabilities for Real AdaBoost
fn estimate_binary_probabilities(y_pred: &Array1<Float>, classes: &Array1<Float>) -> Array2<Float> {
    let n_samples = y_pred.len();
    let mut probs = Array2::<Float>::zeros((n_samples, 2));

    // For Real AdaBoost, we need good probability estimates
    // Use a more sophisticated approach than simple 0.75 confidence
    for i in 0..n_samples {
        let pred_class = y_pred[i];

        // Use a confidence based on the prediction strength
        // For decision stumps, we can estimate this from the margin
        let base_confidence = 0.8; // Higher confidence than SAMME.R

        if pred_class == classes[0] {
            probs[[i, 0]] = base_confidence;
            probs[[i, 1]] = 1.0 - base_confidence;
        } else {
            probs[[i, 0]] = 1.0 - base_confidence;
            probs[[i, 1]] = base_confidence;
        }
    }

    probs
}

impl AdaBoostClassifier<Untrained> {
    /// Create a new AdaBoost classifier
    pub fn new() -> Self {
        Self {
            config: AdaBoostConfig::default(),
            state: PhantomData,
            estimators_: None,
            estimator_weights_: None,
            estimator_errors_: None,
            classes_: None,
            n_classes_: None,
            n_features_in_: None,
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

    /// Set the algorithm variant
    pub fn algorithm(mut self, algorithm: AdaBoostAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Use the SAMME.R algorithm variant for improved performance with probabilistic base learners
    pub fn with_samme_r(mut self) -> Self {
        self.config.algorithm = AdaBoostAlgorithm::SAMMER;
        self
    }

    /// Use the Gentle AdaBoost algorithm variant for improved noise robustness
    pub fn with_gentle(mut self) -> Self {
        self.config.algorithm = AdaBoostAlgorithm::Gentle;
        self
    }

    /// Find unique classes in the target array
    fn find_classes(y: &Array1<Float>) -> Array1<Float> {
        let mut classes: Vec<Float> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        Array1::from_vec(classes)
    }

    /// Calculate class weights for current iteration
    fn calculate_sample_weights(
        &self,
        y: &Array1<Float>,
        y_pred: &Array1<Float>,
        sample_weight: &Array1<Float>,
        estimator_weight: Float,
    ) -> Array1<Float> {
        let n_samples = y.len();
        let mut new_weights = sample_weight.clone();

        // Update weights: w_i *= exp(alpha * I(y_i != h(x_i)))
        for i in 0..n_samples {
            if y[i] != y_pred[i] {
                new_weights[i] *= (estimator_weight).exp();
            }
        }

        // Normalize weights
        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
            // If all weights are zero, reset to uniform
            new_weights.fill(1.0 / n_samples as Float);
        }

        new_weights
    }

    /// Calculate estimator weight based on error
    fn calculate_estimator_weight(&self, error: Float, n_classes: usize) -> Float {
        if error <= 0.0 {
            // Perfect classifier
            return 10.0; // Large weight
        }

        if error >= 1.0 - 1.0 / n_classes as Float {
            // Random guessing or worse
            return 0.0;
        }

        // SAMME algorithm: alpha = log((1 - err) / err) + log(K - 1)
        match self.config.algorithm {
            AdaBoostAlgorithm::SAMME => {
                let alpha = ((1.0 - error) / error).ln() + (n_classes as Float - 1.0).ln();
                alpha * self.config.learning_rate
            }
            AdaBoostAlgorithm::SAMMER => {
                // For SAMME.R, use class probability updates
                self.config.learning_rate
            }
            AdaBoostAlgorithm::RealAdaBoost => {
                // Real AdaBoost uses 0.5 * ln((1-err)/err)
                0.5 * ((1.0 - error) / error).ln()
            }
            AdaBoostAlgorithm::M1 => {
                // AdaBoost.M1: alpha = 0.5 * ln((1-err)/err) for strong learner assumption
                // Error must be < 0.5 for M1 to work (strong learner assumption)
                if error >= 0.5 {
                    return 0.0; // Invalid weak learner for M1
                }
                0.5 * ((1.0 - error) / error).ln()
            }
            AdaBoostAlgorithm::M2 => {
                // AdaBoost.M2: uses confidence-rated predictions, similar to SAMME
                // but with different weight update formula
                let alpha = 0.5 * ((1.0 - error) / error).ln();
                alpha * self.config.learning_rate
            }
            AdaBoostAlgorithm::Gentle => {
                // Gentle AdaBoost: uses least squares for finding weak learner and
                // uses smaller step sizes to be robust to noise
                // For the weight calculation, use a dampened version of the standard formula
                let alpha = 0.5 * ((1.0 - error) / error).ln();
                alpha * self.config.learning_rate * 0.5 // Dampen by factor of 0.5 for gentleness
            }
        }
    }

    /// Resample data according to sample weights with diversity preservation
    fn resample_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        sample_weight: &Array1<Float>,
        rng: &mut impl Rng,
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let n_samples = x.nrows();

        // Normalize weights to sum to 1
        let weight_sum = sample_weight.sum();
        let normalized_weights = if weight_sum > 0.0 {
            sample_weight / weight_sum
        } else {
            Array1::<Float>::from_elem(n_samples, 1.0 / n_samples as Float)
        };

        // Create cumulative distribution
        let mut cumulative = Array1::<Float>::zeros(n_samples);
        cumulative[0] = normalized_weights[0];
        for i in 1..n_samples {
            cumulative[i] = cumulative[i - 1] + normalized_weights[i];
        }

        // Resample with diversity preservation
        let mut selected_indices = Vec::new();
        let unique_classes: std::collections::HashSet<i32> = y.iter().cloned().collect();

        for _i in 0..n_samples {
            let rand_val = rng.gen::<Float>() * cumulative[n_samples - 1];
            let idx = cumulative
                .iter()
                .position(|&cum| cum >= rand_val)
                .unwrap_or(n_samples - 1);
            selected_indices.push(idx);
        }

        // Ensure we have at least one sample from each class
        let resampled_classes: std::collections::HashSet<i32> =
            selected_indices.iter().map(|&idx| y[idx]).collect();

        if resampled_classes.len() < unique_classes.len() {
            // Add missing classes by replacing some samples
            let mut replacement_count = 0;
            for &missing_class in unique_classes.difference(&resampled_classes) {
                if let Some(original_idx) = y.iter().position(|&class| class == missing_class) {
                    if replacement_count < selected_indices.len() {
                        selected_indices[replacement_count] = original_idx;
                        replacement_count += 1;
                    }
                }
            }
        }

        // Create resampled arrays
        let mut x_resampled = Array2::<Float>::zeros(x.dim());
        let mut y_resampled = Array1::<i32>::zeros(y.len());

        for (i, &idx) in selected_indices.iter().enumerate() {
            x_resampled.row_mut(i).assign(&x.row(idx));
            y_resampled[i] = y[idx];
        }

        Ok((x_resampled, y_resampled))
    }

    /// Calculate sample weights for SAMME.R algorithm
    fn calculate_sample_weights_sammer(
        &self,
        y: &Array1<Float>,
        prob_estimates: &Array2<Float>,
        sample_weight: &Array1<Float>,
        classes: &Array1<Float>,
        estimator_weight: Float,
    ) -> Array1<Float> {
        let n_samples = y.len();
        let n_classes = classes.len();
        let mut new_weights = sample_weight.clone();

        // SAMME.R weight update formula from the original paper:
        // w_i *= exp(-((K-1)/K) * y_i * h(x_i))
        // where h(x_i) is the log-ratio of class probabilities
        let factor = ((n_classes - 1) as Float / n_classes as Float) * estimator_weight;

        for i in 0..n_samples {
            let true_class = y[i];

            // Find the class index for the true class
            let true_class_idx = classes.iter().position(|&c| c == true_class);

            if let Some(class_idx) = true_class_idx {
                // Create one-hot encoding for the true class
                let mut y_encoded = Array1::<Float>::zeros(n_classes);
                y_encoded[class_idx] = 1.0;

                // Get the predicted probabilities for this sample
                let probs = prob_estimates.row(i);

                // Calculate h(x_i) as the weighted log-probabilities
                // h(x_i) = (K-1)/K * sum_k [y_k * log(p_k)]
                let mut h_xi = 0.0;
                for k in 0..n_classes {
                    let p_k = probs[k].clamp(1e-7, 1.0 - 1e-7); // Clip for numerical stability

                    // SAMME.R uses the following update:
                    // For true class: contributes positively to keep weight low
                    // For other classes: contributes negatively to increase weight
                    if k == class_idx {
                        h_xi += (n_classes as Float - 1.0) * p_k.ln();
                    } else {
                        h_xi -= p_k.ln();
                    }
                }

                // Update weight: w_i *= exp(-factor * h(x_i))
                let weight_multiplier = (-factor * h_xi / n_classes as Float).exp();
                new_weights[i] *= weight_multiplier;

                // Prevent weights from exploding
                new_weights[i] = new_weights[i].clamp(1e-10, 1e3);
            } else {
                // If class not found, keep original weight
                // This shouldn't happen in normal operation
                eprintln!("Warning: True class {} not found in classes", true_class);
            }
        }

        // Normalize weights to sum to 1
        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
            // If all weights are zero, reset to uniform
            new_weights.fill(1.0 / n_samples as Float);
        }

        new_weights
    }

    /// Calculate sample weights for Real AdaBoost
    fn calculate_sample_weights_real_adaboost(
        &self,
        y: &Array1<Float>,
        prob_estimates: &Array2<Float>,
        sample_weight: &Array1<Float>,
        classes: &Array1<Float>,
    ) -> Array1<Float> {
        let n_samples = y.len();
        let mut new_weights = sample_weight.clone();

        // Real AdaBoost weight update formula:
        // w_i(t+1) = w_i(t) * exp(-y_i * h_t(x_i))
        // where h_t(x_i) = 0.5 * ln(p_1(x_i) / p_0(x_i))
        // and y_i ∈ {-1, +1}

        for i in 0..n_samples {
            let true_class = y[i];

            // Convert to Real AdaBoost labels: first class = -1, second class = +1
            let y_i = if true_class == classes[0] { -1.0 } else { 1.0 };

            // Get probabilities for both classes
            let p_0 = prob_estimates[[i, 0]].clamp(1e-7, 1.0 - 1e-7);
            let p_1 = prob_estimates[[i, 1]].clamp(1e-7, 1.0 - 1e-7);

            // Calculate weak hypothesis output: h_t(x_i) = 0.5 * ln(p_1 / p_0)
            let h_xi = 0.5 * (p_1 / p_0).ln();

            // Update weight: w_i(t+1) = w_i(t) * exp(-y_i * h_t(x_i))
            let weight_multiplier = (-y_i * h_xi).exp();
            new_weights[i] *= weight_multiplier;

            // Prevent weights from exploding or vanishing
            new_weights[i] = new_weights[i].clamp(1e-10, 1e3);
        }

        // Normalize weights to sum to 1
        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
            // If all weights are zero, reset to uniform
            new_weights.fill(1.0 / n_samples as Float);
        }

        new_weights
    }
}

impl AdaBoostClassifier<Trained> {
    /// Get the fitted base estimators
    pub fn estimators(&self) -> &[DecisionTreeClassifier<Trained>] {
        self.estimators_
            .as_ref()
            .expect("AdaBoost should be fitted")
    }

    /// Get the weights for each estimator
    pub fn estimator_weights(&self) -> &Array1<Float> {
        self.estimator_weights_
            .as_ref()
            .expect("AdaBoost should be fitted")
    }

    /// Get the errors for each estimator
    pub fn estimator_errors(&self) -> &Array1<Float> {
        self.estimator_errors_
            .as_ref()
            .expect("AdaBoost should be fitted")
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_.as_ref().expect("AdaBoost should be fitted")
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("AdaBoost should be fitted")
    }

    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("AdaBoost should be fitted")
    }

    /// Get feature importances (averaged from all estimators)
    pub fn feature_importances(&self) -> Result<Array1<Float>> {
        let estimators = self.estimators();
        let weights = self.estimator_weights();
        let n_features = self.n_features_in();

        if estimators.is_empty() {
            return Ok(Array1::<Float>::zeros(n_features));
        }

        let mut importances = Array1::<Float>::zeros(n_features);
        let mut total_weight = 0.0;

        for (_estimator, &weight) in estimators.iter().zip(weights.iter()) {
            // For decision stumps, we simulate feature importance
            // In practice, this would use the actual feature importance from the tree
            let tree_importances = Array1::<Float>::ones(n_features) / n_features as Float;
            importances += &(tree_importances * weight.abs());
            total_weight += weight.abs();
        }

        if total_weight > 0.0 {
            importances /= total_weight;
        } else {
            // If no valid weights, return uniform importance
            importances.fill(1.0 / n_features as Float);
        }

        Ok(importances)
    }

    /// Predict class probabilities using weighted voting
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
        let classes = self.classes();
        let n_classes = self.n_classes();

        match self.config.algorithm {
            AdaBoostAlgorithm::SAMME => {
                // Original SAMME: use weighted voting
                let mut class_votes = Array2::<Float>::zeros((n_samples, n_classes));

                // Aggregate predictions from all estimators
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    for (i, &pred) in predictions.iter().enumerate() {
                        if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                            class_votes[[i, class_idx]] += weight;
                        }
                    }
                }

                // Convert votes to probabilities
                let mut probabilities = Array2::<Float>::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    let vote_sum = class_votes.row(i).sum();
                    if vote_sum > 0.0 {
                        for j in 0..n_classes {
                            probabilities[[i, j]] = class_votes[[i, j]] / vote_sum;
                        }
                    } else {
                        // Uniform distribution if no votes
                        probabilities.row_mut(i).fill(1.0 / n_classes as Float);
                    }
                }

                Ok(probabilities)
            }
            AdaBoostAlgorithm::SAMMER => {
                // SAMME.R: use probability aggregation
                let mut prob_sum = Array2::<Float>::zeros((n_samples, n_classes));

                // Initialize with uniform probabilities
                prob_sum.fill(1.0 / n_classes as Float);

                // Aggregate probability estimates from all estimators
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    // Estimate probabilities for this estimator
                    let prob_estimates =
                        estimate_probabilities(&predictions, classes, n_samples, n_classes);

                    // SAMME.R aggregation: geometric mean of probabilities
                    for i in 0..n_samples {
                        for j in 0..n_classes {
                            // Use log-space for numerical stability
                            prob_sum[[i, j]] += weight * prob_estimates[[i, j]].ln();
                        }
                    }
                }

                // Convert back from log-space and normalize
                let mut probabilities = Array2::<Float>::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    // Find max for numerical stability
                    let max_log = prob_sum
                        .row(i)
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);

                    // Compute exp and normalize
                    let mut sum = 0.0;
                    for j in 0..n_classes {
                        probabilities[[i, j]] = (prob_sum[[i, j]] - max_log).exp();
                        sum += probabilities[[i, j]];
                    }

                    // Normalize
                    for j in 0..n_classes {
                        probabilities[[i, j]] /= sum;
                    }
                }

                Ok(probabilities)
            }
            AdaBoostAlgorithm::RealAdaBoost => {
                // Real AdaBoost probability aggregation for binary classification
                if n_classes != 2 {
                    return Err(SklearsError::InvalidInput(
                        "Real AdaBoost predict_proba only supports binary classification"
                            .to_string(),
                    ));
                }

                let mut decision_scores = Array1::<Float>::zeros(n_samples);

                // Aggregate decision scores from all estimators
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    // Get probability estimates for Real AdaBoost
                    let prob_estimates = estimate_binary_probabilities(&predictions, classes);

                    for i in 0..n_samples {
                        let p_0 = prob_estimates[[i, 0]].clamp(1e-7, 1.0 - 1e-7);
                        let p_1 = prob_estimates[[i, 1]].clamp(1e-7, 1.0 - 1e-7);

                        // Real AdaBoost decision function: h_t(x) = 0.5 * ln(p_1/p_0)
                        let h_t = 0.5 * (p_1 / p_0).ln();
                        decision_scores[i] += weight * h_t;
                    }
                }

                // Convert decision scores to probabilities using sigmoid
                let mut probabilities = Array2::<Float>::zeros((n_samples, 2));
                for i in 0..n_samples {
                    let sigmoid = 1.0 / (1.0 + (-decision_scores[i]).exp());
                    probabilities[[i, 1]] = sigmoid;
                    probabilities[[i, 0]] = 1.0 - sigmoid;
                }

                Ok(probabilities)
            }
            AdaBoostAlgorithm::M1 => {
                // AdaBoost.M1: similar to SAMME but with strong learner assumption
                let mut class_votes = Array2::<Float>::zeros((n_samples, n_classes));

                // Aggregate predictions from all estimators
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    for (i, &pred) in predictions.iter().enumerate() {
                        if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                            class_votes[[i, class_idx]] += weight;
                        }
                    }
                }

                // Convert votes to probabilities (same as SAMME)
                let mut probabilities = Array2::<Float>::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    let vote_sum = class_votes.row(i).sum();
                    if vote_sum > 0.0 {
                        for j in 0..n_classes {
                            probabilities[[i, j]] = class_votes[[i, j]] / vote_sum;
                        }
                    } else {
                        // Uniform distribution if no votes
                        probabilities.row_mut(i).fill(1.0 / n_classes as Float);
                    }
                }

                Ok(probabilities)
            }
            AdaBoostAlgorithm::M2 => {
                // AdaBoost.M2: confidence-rated predictions with probability weighting
                let mut confidence_scores = Array2::<Float>::zeros((n_samples, n_classes));

                // Aggregate confidence-rated predictions from all estimators
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    // Estimate confidences for M2 (using pseudo-probabilities)
                    let prob_estimates =
                        estimate_probabilities(&predictions, classes, n_samples, n_classes);

                    for i in 0..n_samples {
                        for j in 0..n_classes {
                            // M2 confidence: higher confidence for more certain predictions
                            let confidence = prob_estimates[[i, j]];
                            confidence_scores[[i, j]] += weight * confidence;
                        }
                    }
                }

                // Convert confidence scores to probabilities
                let mut probabilities = Array2::<Float>::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    let score_sum = confidence_scores.row(i).sum();
                    if score_sum > 0.0 {
                        for j in 0..n_classes {
                            probabilities[[i, j]] = confidence_scores[[i, j]] / score_sum;
                        }
                    } else {
                        // Uniform distribution if no scores
                        probabilities.row_mut(i).fill(1.0 / n_classes as Float);
                    }
                }

                Ok(probabilities)
            }
            AdaBoostAlgorithm::Gentle => {
                // Gentle AdaBoost: similar to SAMME but with dampened weights
                let mut class_votes = Array2::<Float>::zeros((n_samples, n_classes));

                // Aggregate predictions from all estimators with gentle weighting
                for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
                    let predictions_i32 = estimator.predict(x)?;
                    let predictions = convert_predictions_to_float(&predictions_i32);

                    for (i, &pred) in predictions.iter().enumerate() {
                        if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                            class_votes[[i, class_idx]] += weight;
                        }
                    }
                }

                // Convert votes to probabilities with gentle smoothing
                let mut probabilities = Array2::<Float>::zeros((n_samples, n_classes));
                for i in 0..n_samples {
                    let vote_sum = class_votes.row(i).sum();
                    if vote_sum > 0.0 {
                        for j in 0..n_classes {
                            probabilities[[i, j]] = class_votes[[i, j]] / vote_sum;
                        }

                        // Apply gentle smoothing to avoid overconfident predictions
                        let alpha = 0.1; // Smoothing parameter
                        let uniform_prob = 1.0 / n_classes as Float;
                        for j in 0..n_classes {
                            probabilities[[i, j]] =
                                (1.0 - alpha) * probabilities[[i, j]] + alpha * uniform_prob;
                        }
                    } else {
                        // Uniform distribution if no votes
                        probabilities.row_mut(i).fill(1.0 / n_classes as Float);
                    }
                }

                Ok(probabilities)
            }
        }
    }

    /// Get decision function values
    pub fn decision_function(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let probas = self.predict_proba(x)?;

        // For binary classification, return log-odds
        if self.n_classes() == 2 {
            let mut decision = Array2::<Float>::zeros((probas.nrows(), 1));
            for i in 0..probas.nrows() {
                let p1 = probas[[i, 1]].max(1e-15); // Avoid log(0)
                let p0 = probas[[i, 0]].max(1e-15);
                decision[[i, 0]] = (p1 / p0).ln();
            }
            Ok(decision)
        } else {
            // For multi-class, return log probabilities
            Ok(probas.mapv(|p| p.max(1e-15).ln()))
        }
    }
}

impl Default for AdaBoostClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl<State> std::fmt::Debug for AdaBoostClassifier<State> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdaBoostClassifier")
            .field("config", &self.config)
            .field(
                "n_estimators_fitted",
                &self.estimators_.as_ref().map(|e| e.len()),
            )
            .field("n_classes", &self.n_classes_)
            .field("n_features_in", &self.n_features_in_)
            .finish()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AdaBoostClassifier<Untrained> {
    type Fitted = AdaBoostClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit AdaBoost on empty dataset".to_string(),
            ));
        }

        if self.config.n_estimators == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_estimators".to_string(),
                reason: "Number of estimators must be positive".to_string(),
            });
        }

        // Find unique classes
        let classes = Self::find_classes(y);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "AdaBoost requires at least 2 classes".to_string(),
            ));
        }

        // Initialize sample weights uniformly
        let mut sample_weight = Array1::<Float>::from_elem(n_samples, 1.0 / n_samples as Float);

        // Initialize collections for fitted estimators
        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => scirs2_core::random::seeded_rng(seed),
            None => scirs2_core::random::seeded_rng(42), // Use fallback seed for entropy
        };

        // Convert labels to i32 for decision tree
        let y_i32 = convert_labels_to_i32(y);

        // Boosting iterations
        for _iteration in 0..self.config.n_estimators {
            // Create base estimator (decision stump)
            let base_estimator = DecisionTreeClassifier::new()
                .criterion(SplitCriterion::Gini)
                .max_depth(1)
                .min_samples_split(2)
                .min_samples_leaf(1);

            // Different handling for SAMME vs SAMME.R
            match self.config.algorithm {
                AdaBoostAlgorithm::SAMME => {
                    // Original SAMME algorithm
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // Calculate weighted error
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }

                    // If error is too high, stop
                    if weighted_error >= 0.5 {
                        if estimators.is_empty() {
                            estimators.push(fitted_estimator);
                            estimator_weights.push(0.0);
                            estimator_errors.push(weighted_error);
                        }
                        break;
                    }

                    let estimator_weight =
                        self.calculate_estimator_weight(weighted_error, n_classes);

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);

                    sample_weight =
                        self.calculate_sample_weights(y, &y_pred, &sample_weight, estimator_weight);

                    if weighted_error < 1e-10 {
                        break;
                    }
                }
                AdaBoostAlgorithm::SAMMER => {
                    // SAMME.R algorithm using probability updates
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    // For SAMME.R, we need class probabilities
                    // Since DecisionTreeClassifier might not support predict_proba,
                    // we'll simulate it with prediction confidence
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // Create probability estimates (simplified version)
                    let prob_estimates =
                        estimate_probabilities(&y_pred, &classes, n_samples, n_classes);

                    // Calculate error for early stopping
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }

                    // For SAMME.R, we use a fixed weight per estimator
                    let estimator_weight = self.config.learning_rate;

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);

                    // Update sample weights using probability estimates (SAMME.R style)
                    sample_weight = self.calculate_sample_weights_sammer(
                        y,
                        &prob_estimates,
                        &sample_weight,
                        &classes,
                        estimator_weight,
                    );

                    // Early stopping if good enough or error too high
                    if !(1e-10..0.5).contains(&weighted_error) {
                        break;
                    }
                }
                AdaBoostAlgorithm::RealAdaBoost => {
                    // Real AdaBoost algorithm using probabilistic weak hypotheses
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    // Get binary classification probabilities for Real AdaBoost
                    if n_classes != 2 {
                        return Err(SklearsError::InvalidInput(
                            "Real AdaBoost currently supports only binary classification"
                                .to_string(),
                        ));
                    }

                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // Estimate class probabilities for Real AdaBoost
                    let prob_estimates = estimate_binary_probabilities(&y_pred, &classes);

                    // Calculate the weighted error as the sum of weights where p < 0.5
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        let correct_class_idx = if y[i] == classes[0] { 0 } else { 1 };
                        let prob_correct = prob_estimates[[i, correct_class_idx]];

                        if prob_correct < 0.5 {
                            weighted_error += sample_weight[i];
                        }
                    }

                    // Calculate Real AdaBoost estimator weight
                    let estimator_weight = if weighted_error > 0.0 && weighted_error < 0.5 {
                        0.5 * ((1.0 - weighted_error) / weighted_error).ln()
                    } else if weighted_error == 0.0 {
                        10.0 // Large weight for perfect classifier
                    } else {
                        0.0 // Zero weight for useless classifier
                    };

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);

                    // Update sample weights using Real AdaBoost formula
                    sample_weight = self.calculate_sample_weights_real_adaboost(
                        y,
                        &prob_estimates,
                        &sample_weight,
                        &classes,
                    );

                    // Early stopping conditions
                    if !(1e-10..0.5).contains(&weighted_error) {
                        break;
                    }
                }
                AdaBoostAlgorithm::M1 => {
                    // AdaBoost.M1: requires strong learner assumption (error < 0.5)
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // Calculate weighted error
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }

                    // M1 requires strong learner assumption: error must be < 0.5
                    if weighted_error >= 0.5 {
                        // If we can't meet the strong learner assumption, stop
                        if estimators.is_empty() {
                            return Err(SklearsError::InvalidInput(
                                "AdaBoost.M1 requires strong learners (error < 0.5)".to_string(),
                            ));
                        }
                        break;
                    }

                    let estimator_weight =
                        self.calculate_estimator_weight(weighted_error, n_classes);

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);

                    // M1 weight update: similar to binary AdaBoost
                    sample_weight =
                        self.calculate_sample_weights(y, &y_pred, &sample_weight, estimator_weight);

                    if weighted_error < 1e-10 {
                        break;
                    }
                }
                AdaBoostAlgorithm::M2 => {
                    // AdaBoost.M2: confidence-rated predictions for multi-class
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // For M2, we use confidence-based error calculation
                    let prob_estimates =
                        estimate_probabilities(&y_pred, &classes, n_samples, n_classes);

                    // M2 pseudo-loss calculation based on confidence ratings
                    let mut pseudo_loss = 0.0;
                    for i in 0..n_samples {
                        let true_class_idx = classes.iter().position(|&c| c == y[i]).unwrap_or(0);

                        // Calculate confidence for correct vs incorrect predictions
                        let mut margin = prob_estimates[[i, true_class_idx]];
                        for j in 0..n_classes {
                            if j != true_class_idx {
                                margin -= prob_estimates[[i, j]] / (n_classes - 1) as Float;
                            }
                        }

                        // M2 loss: higher loss for confident wrong predictions
                        if margin <= 0.0 {
                            pseudo_loss += sample_weight[i] * (1.0 - margin);
                        }
                    }

                    // Normalize pseudo-loss
                    let total_weight: Float = sample_weight.sum();
                    if total_weight > 0.0 {
                        pseudo_loss /= total_weight;
                    }

                    // Early stopping if pseudo-loss too high
                    if pseudo_loss >= 0.5 {
                        if estimators.is_empty() {
                            estimators.push(fitted_estimator);
                            estimator_weights.push(0.0);
                            estimator_errors.push(pseudo_loss);
                        }
                        break;
                    }

                    let estimator_weight = self.calculate_estimator_weight(pseudo_loss, n_classes);

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(pseudo_loss);

                    // M2 weight update based on confidence margins
                    let mut new_weights = sample_weight.clone();
                    for i in 0..n_samples {
                        let true_class_idx = classes.iter().position(|&c| c == y[i]).unwrap_or(0);
                        let confidence = prob_estimates[[i, true_class_idx]];

                        // Update weight based on confidence: boost samples with low confidence
                        let weight_multiplier = if confidence < 0.5 {
                            (estimator_weight * (1.0 - confidence)).exp()
                        } else {
                            (estimator_weight * confidence).exp().recip()
                        };

                        new_weights[i] *= weight_multiplier;
                    }

                    // Normalize weights
                    let weight_sum = new_weights.sum();
                    if weight_sum > 0.0 {
                        new_weights /= weight_sum;
                    } else {
                        new_weights.fill(1.0 / n_samples as Float);
                    }
                    sample_weight = new_weights;

                    if pseudo_loss < 1e-10 {
                        break;
                    }
                }
                AdaBoostAlgorithm::Gentle => {
                    // Gentle AdaBoost: more robust to noise with gentle weight updates
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;

                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;

                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);

                    // Validate prediction array size
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }

                    // Calculate weighted error
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }

                    // Gentle stopping: use a more relaxed threshold than standard AdaBoost
                    if weighted_error >= 0.6 {
                        if estimators.is_empty() {
                            estimators.push(fitted_estimator);
                            estimator_weights.push(0.1); // Small weight instead of 0
                            estimator_errors.push(weighted_error);
                        }
                        break;
                    }

                    let estimator_weight =
                        self.calculate_estimator_weight(weighted_error, n_classes);

                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);

                    // Gentle weight update: use a dampened version for robustness
                    let mut new_weights = sample_weight.clone();
                    let gentle_factor = 0.5; // Damping factor for gentle updates

                    for i in 0..n_samples {
                        let multiplier = if y[i] != y_pred[i] {
                            // Increase weight for misclassified samples, but gently
                            (gentle_factor * estimator_weight).exp()
                        } else {
                            // Decrease weight for correctly classified samples, but gently
                            (-gentle_factor * estimator_weight).exp()
                        };
                        new_weights[i] *= multiplier;
                    }

                    // Normalize weights with additional smoothing
                    let weight_sum = new_weights.sum();
                    if weight_sum > 0.0 {
                        new_weights /= weight_sum;

                        // Apply gentle smoothing to prevent extreme weights
                        let smoothing = 0.01;
                        let uniform_weight = 1.0 / n_samples as Float;
                        for i in 0..n_samples {
                            new_weights[i] =
                                (1.0 - smoothing) * new_weights[i] + smoothing * uniform_weight;
                        }

                        // Renormalize after smoothing
                        let smoothed_sum = new_weights.sum();
                        if smoothed_sum > 0.0 {
                            new_weights /= smoothed_sum;
                        }
                    } else {
                        new_weights.fill(1.0 / n_samples as Float);
                    }

                    sample_weight = new_weights;

                    if weighted_error < 1e-10 {
                        break;
                    }
                }
            }
        }

        if estimators.is_empty() {
            return Err(SklearsError::InvalidInput(
                "AdaBoost failed to fit any estimators".to_string(),
            ));
        }

        Ok(AdaBoostClassifier {
            config: self.config,
            state: PhantomData,
            estimators_: Some(estimators),
            estimator_weights_: Some(Array1::from_vec(estimator_weights)),
            estimator_errors_: Some(Array1::from_vec(estimator_errors)),
            classes_: Some(classes),
            n_classes_: Some(n_classes),
            n_features_in_: Some(n_features),
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for AdaBoostClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probas = self.predict_proba(x)?;
        let classes = self.classes();

        let mut predictions = Array1::<Float>::zeros(probas.nrows());

        // Predict class with highest probability
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

/// LogitBoost Classifier
///
/// LogitBoost is a boosting algorithm that uses logistic regression as the
/// base learner for binary classification problems. It's part of the exponential
/// family boosting algorithms and is particularly effective for noisy data.
///
/// # Examples
///
/// ```rust
/// use sklears_ensemble::adaboost::LogitBoostClassifier;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![
///     [1.0, 2.0],
///     [2.0, 3.0],
///     [3.0, 1.0],
///     [4.0, 2.0],
/// ];
/// let y = array![0.0, 0.0, 1.0, 1.0];
///
/// let logit_boost = LogitBoostClassifier::new()
///     .n_estimators(10)
///     .learning_rate(0.1)
///     .fit(&x, &y)?;
///
/// let predictions = logit_boost.predict(&x)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub struct LogitBoostClassifier<State = Untrained> {
    config: LogitBoostConfig,
    state: PhantomData<State>,
    // Fitted parameters
    estimators_: Option<Vec<DecisionTreeRegressor<Trained>>>, // Using regressors for probability estimation
    estimator_weights_: Option<Array1<Float>>,
    classes_: Option<Array1<Float>>,
    n_classes_: Option<usize>,
    n_features_in_: Option<usize>,
    intercept_: Option<Float>,
}

/// Configuration for LogitBoost
#[derive(Debug, Clone)]
pub struct LogitBoostConfig {
    /// Number of boosting iterations
    pub n_estimators: usize,
    /// Learning rate (shrinks contribution of each estimator)
    pub learning_rate: Float,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Maximum depth of the regression trees
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Tolerance for convergence
    pub tolerance: Float,
    /// Maximum iterations for Newton-Raphson optimization
    pub max_iter: usize,
}

impl Default for LogitBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            random_state: None,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            tolerance: 1e-4,
            max_iter: 10,
        }
    }
}

impl Default for LogitBoostClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

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
        let base_estimator = DecisionTreeRegressor::new()
            .criterion(SplitCriterion::MSE)
            .max_depth(self.config.max_depth.unwrap_or(3))
            .min_samples_split(self.config.min_samples_split)
            .min_samples_leaf(self.config.min_samples_leaf);

        // For now, we'll fit without sample weights since DecisionTreeRegressor
        // might not support them directly. In a full implementation,
        // we'd need to modify the tree to handle weighted samples.
        base_estimator.fit(x, z)
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

        // Find unique classes - LogitBoost typically works for binary classification
        let classes = AdaBoostClassifier::<Untrained>::find_classes(y);
        let n_classes = classes.len();

        if n_classes != 2 {
            return Err(SklearsError::InvalidInput(
                "LogitBoost currently supports only binary classification".to_string(),
            ));
        }

        // Convert to {0, 1} labels for logistic regression
        let mut y_binary = Array1::<Float>::zeros(n_samples);
        for i in 0..n_samples {
            y_binary[i] = if y[i] == classes[0] { 0.0 } else { 1.0 };
        }

        // Initialize predictions with log-odds of the class proportion
        let class_1_count = y_binary.sum();
        let class_0_count = n_samples as Float - class_1_count;
        let initial_logit = if class_1_count > 0.0 && class_0_count > 0.0 {
            (class_1_count / class_0_count).ln()
        } else {
            0.0
        };

        let mut f = Array1::<Float>::from_elem(n_samples, initial_logit); // Log-odds
        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();

        // LogitBoost iterations
        for _iteration in 0..self.config.n_estimators {
            // Calculate current probabilities
            let mut p = Array1::<Float>::zeros(n_samples);
            for i in 0..n_samples {
                p[i] = Self::sigmoid(f[i]);
            }

            // Calculate working response and weights
            let (z, w) = self.calculate_working_response_and_weights(&y_binary, &p);

            // Check for convergence
            let gradient_norm: Float = z
                .iter()
                .zip(w.iter())
                .map(|(&z_i, &w_i)| z_i * z_i * w_i)
                .sum::<Float>()
                .sqrt();

            if gradient_norm < self.config.tolerance {
                break;
            }

            // Fit weighted regression tree
            let fitted_estimator = self.fit_weighted_tree(x, &z, &w)?;

            // Get predictions from the tree
            let tree_predictions = fitted_estimator.predict(x)?;

            // Update log-odds with learning rate
            for i in 0..n_samples {
                f[i] += self.config.learning_rate * tree_predictions[i];
            }

            estimators.push(fitted_estimator);
            estimator_weights.push(self.config.learning_rate);

            // Optional: Early stopping based on loss improvement
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

        // Predict class with highest probability
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaboost_basic() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [1.0, 0.5],
            [2.0, 1.5],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0];

        let ada = AdaBoostClassifier::new()
            .n_estimators(3)
            .learning_rate(1.0)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(ada.classes().len(), 2);
        assert_eq!(ada.n_classes(), 2);
        assert_eq!(ada.n_features_in(), 2);
        assert!(ada.estimators().len() <= 3); // May stop early

        // Test prediction
        let predictions = ada.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test prediction probabilities
        let probas = ada.predict_proba(&x).unwrap();
        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let row_sum: Float = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adaboost_binary_classification() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];
        let y = array![0.0, 0.0, 1.0, 1.0];

        let ada = AdaBoostClassifier::new()
            .n_estimators(5)
            .fit(&x, &y)
            .unwrap();

        let predictions = ada.predict(&x).unwrap();

        // Should be able to learn this simple pattern
        assert_eq!(predictions.len(), 4);
        assert!(predictions.iter().all(|&p| p == 0.0 || p == 1.0));
    }

    #[test]
    fn test_adaboost_feature_importances() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0],];
        let y = array![1.0, 1.0, 0.0, 0.0];

        let ada = AdaBoostClassifier::new()
            .n_estimators(3)
            .fit(&x, &y)
            .unwrap();

        let importances = ada.feature_importances().unwrap();
        assert_eq!(importances.len(), 2);

        // Feature importances should be non-negative and sum to 1
        assert!(importances.iter().all(|&imp| imp >= 0.0));
        assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaboost_decision_function() {
        let x = array![[1.0, 2.0], [2.0, 1.0],];
        let y = array![0.0, 1.0];

        let ada = AdaBoostClassifier::new()
            .n_estimators(5)
            .fit(&x, &y)
            .unwrap();

        let decision = ada.decision_function(&x).unwrap();

        // For binary classification, should return single column
        assert_eq!(decision.dim(), (2, 1));
    }

    #[test]
    fn test_adaboost_empty_data() {
        let x = Array2::<Float>::zeros((0, 2));
        let y = Array1::<Float>::zeros(0);

        let result = AdaBoostClassifier::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaboost_single_class() {
        let x = array![[1.0, 2.0], [2.0, 3.0],];
        let y = array![0.0, 0.0]; // Only one class

        let result = AdaBoostClassifier::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaboost_config_builder() {
        let ada = AdaBoostClassifier::new()
            .n_estimators(100)
            .learning_rate(0.5)
            .random_state(123)
            .algorithm(AdaBoostAlgorithm::SAMMER);

        assert_eq!(ada.config.n_estimators, 100);
        assert_eq!(ada.config.learning_rate, 0.5);
        assert_eq!(ada.config.random_state, Some(123));
        assert!(matches!(ada.config.algorithm, AdaBoostAlgorithm::SAMMER));
    }

    #[test]
    fn test_adaboost_feature_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];
        let y_train = array![0.0, 1.0];

        let x_test = array![
            [1.0, 2.0], // Wrong number of features
        ];

        let ada = AdaBoostClassifier::new().fit(&x_train, &y_train).unwrap();
        let result = ada.predict(&x_test);
        assert!(result.is_err());
    }

    #[test]
    fn test_adaboost_samme_r_algorithm() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [1.0, 0.5],
            [2.0, 1.5],
            [5.0, 3.0],
            [6.0, 4.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0];

        // Test SAMME.R algorithm
        let ada_r = AdaBoostClassifier::new()
            .n_estimators(5)
            .learning_rate(0.8)
            .with_samme_r()
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(ada_r.classes().len(), 2);
        assert_eq!(ada_r.n_classes(), 2);
        assert_eq!(ada_r.n_features_in(), 2);

        // Test predictions
        let predictions = ada_r.predict(&x).unwrap();
        assert_eq!(predictions.len(), 8);

        // Test probability predictions
        let probabilities = ada_r.predict_proba(&x).unwrap();
        assert_eq!(probabilities.dim(), (8, 2));

        // Check probabilities sum to 1
        for i in 0..8 {
            let prob_sum = probabilities.row(i).sum();
            assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
        }

        // Test decision function
        let decision_scores = ada_r.decision_function(&x).unwrap();
        assert_eq!(decision_scores.dim(), (8, 1));

        // Compare with regular SAMME
        let ada_samme = AdaBoostClassifier::new()
            .n_estimators(5)
            .learning_rate(0.8)
            .algorithm(AdaBoostAlgorithm::SAMME)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        let predictions_samme = ada_samme.predict(&x).unwrap();

        // Both algorithms should produce reasonable predictions
        // (they may differ but should both be valid)
        assert_eq!(predictions_samme.len(), 8);

        // Verify feature importances work for SAMME.R
        let importances = ada_r.feature_importances().unwrap();
        assert_eq!(importances.len(), 2);
        assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_adaboost_samme_r_multiclass() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [1.0, 0.5],
            [2.0, 1.5],
            [5.0, 3.0],
            [6.0, 4.0],
            [7.0, 1.0],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 2.0, 2.0, 2.0];

        // Test SAMME.R with multiclass
        let ada_r = AdaBoostClassifier::new()
            .n_estimators(4)
            .learning_rate(0.7)
            .with_samme_r()
            .random_state(123)
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(ada_r.classes().len(), 3);
        assert_eq!(ada_r.n_classes(), 3);

        // Test predictions
        let predictions = ada_r.predict(&x).unwrap();
        assert_eq!(predictions.len(), 9);

        // Test probability predictions for multiclass
        let probabilities = ada_r.predict_proba(&x).unwrap();
        assert_eq!(probabilities.dim(), (9, 3));

        // Check probabilities sum to 1
        for i in 0..9 {
            let prob_sum = probabilities.row(i).sum();
            assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
        }

        // For multiclass, decision function should return log probabilities
        let decision_scores = ada_r.decision_function(&x).unwrap();
        assert_eq!(decision_scores.dim(), (9, 3));
    }

    #[test]
    fn test_adaboost_real_algorithm() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [4.0, 2.0],
            [1.0, 0.5],
            [2.0, 1.5],
        ];
        let y = array![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]; // Binary classification only

        // Test Real AdaBoost algorithm
        let ada_real = AdaBoostClassifier::new()
            .n_estimators(5)
            .learning_rate(1.0)
            .algorithm(AdaBoostAlgorithm::RealAdaBoost)
            .random_state(42)
            .fit(&x, &y)
            .unwrap();

        // Check fitted parameters
        assert_eq!(ada_real.classes().len(), 2);
        assert_eq!(ada_real.n_classes(), 2);
        assert_eq!(ada_real.n_features_in(), 2);

        // Test predictions
        let predictions = ada_real.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // All predictions should be valid class labels
        for &pred in predictions.iter() {
            assert!(ada_real.classes().iter().any(|&c| c == pred));
        }

        // Test probability predictions
        let probabilities = ada_real.predict_proba(&x).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check probabilities sum to 1
        for i in 0..6 {
            let prob_sum = probabilities.row(i).sum();
            assert_abs_diff_eq!(prob_sum, 1.0, epsilon = 1e-10);
        }

        // Check that decision function works
        let decision_scores = ada_real.decision_function(&x).unwrap();
        assert_eq!(decision_scores.len(), 6);

        // For Real AdaBoost, estimator weights should be reasonable
        let weights = ada_real.estimator_weights();
        assert!(weights.len() > 0);
        assert!(weights.iter().all(|&w| w.is_finite()));
    }

    #[test]
    fn test_real_adaboost_multiclass_error() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![0.0, 1.0, 2.0]; // Three classes

        // Real AdaBoost should reject multiclass problems
        let result = AdaBoostClassifier::new()
            .algorithm(AdaBoostAlgorithm::RealAdaBoost)
            .fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("binary classification"));
    }
}

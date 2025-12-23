//! AdaBoost Classifier implementation

use super::helpers::*;
use super::types::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Predict},
    traits::{Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use super::types::AdaBoostClassifier;
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

    /// Use the SAMME.R algorithm variant
    pub fn with_samme_r(mut self) -> Self {
        self.config.algorithm = AdaBoostAlgorithm::SAMMER;
        self
    }

    /// Use the Gentle AdaBoost algorithm variant
    pub fn with_gentle(mut self) -> Self {
        self.config.algorithm = AdaBoostAlgorithm::Gentle;
        self
    }

    /// Find unique classes in the target array
    pub(crate) fn find_classes(y: &Array1<Float>) -> Array1<Float> {
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

        for i in 0..n_samples {
            if y[i] != y_pred[i] {
                new_weights[i] *= (estimator_weight).exp();
            }
        }

        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
            new_weights.fill(1.0 / n_samples as Float);
        }

        new_weights
    }

    /// Calculate estimator weight based on error
    fn calculate_estimator_weight(&self, error: Float, n_classes: usize) -> Float {
        if error <= 0.0 {
            return 10.0;
        }

        if error >= 1.0 - 1.0 / n_classes as Float {
            return 0.0;
        }

        match self.config.algorithm {
            AdaBoostAlgorithm::SAMME => {
                let alpha = ((1.0 - error) / error).ln() + (n_classes as Float - 1.0).ln();
                alpha * self.config.learning_rate
            }
            AdaBoostAlgorithm::SAMMER => self.config.learning_rate,
            AdaBoostAlgorithm::RealAdaBoost => 0.5 * ((1.0 - error) / error).ln(),
            AdaBoostAlgorithm::M1 => {
                if error >= 0.5 {
                    return 0.0;
                }
                0.5 * ((1.0 - error) / error).ln()
            }
            AdaBoostAlgorithm::M2 => {
                let alpha = 0.5 * ((1.0 - error) / error).ln();
                alpha * self.config.learning_rate
            }
            AdaBoostAlgorithm::Gentle => {
                let alpha = 0.5 * ((1.0 - error) / error).ln();
                alpha * self.config.learning_rate * 0.5
            }
            AdaBoostAlgorithm::Discrete => ((1.0 - error) / error).ln() * self.config.learning_rate,
        }
    }

    /// Resample data according to sample weights
    fn resample_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        sample_weight: &Array1<Float>,
        rng: &mut impl Rng,
    ) -> Result<(Array2<Float>, Array1<i32>)> {
        let n_samples = x.nrows();

        let weight_sum = sample_weight.sum();
        let normalized_weights = if weight_sum > 0.0 {
            sample_weight / weight_sum
        } else {
            Array1::<Float>::from_elem(n_samples, 1.0 / n_samples as Float)
        };

        let mut cumulative = Array1::<Float>::zeros(n_samples);
        cumulative[0] = normalized_weights[0];
        for i in 1..n_samples {
            cumulative[i] = cumulative[i - 1] + normalized_weights[i];
        }

        let mut selected_indices = Vec::new();
        let unique_classes: std::collections::HashSet<i32> = y.iter().cloned().collect();

        for _i in 0..n_samples {
            let rand_val = rng.random::<Float>() * cumulative[n_samples - 1];
            let idx = cumulative
                .iter()
                .position(|&cum| cum >= rand_val)
                .unwrap_or(n_samples - 1);
            selected_indices.push(idx);
        }

        let resampled_classes: std::collections::HashSet<i32> =
            selected_indices.iter().map(|&idx| y[idx]).collect();

        if resampled_classes.len() < unique_classes.len() {
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

        let factor = ((n_classes - 1) as Float / n_classes as Float) * estimator_weight;

        for i in 0..n_samples {
            let true_class = y[i];
            let true_class_idx = classes.iter().position(|&c| c == true_class);

            if let Some(class_idx) = true_class_idx {
                let probs = prob_estimates.row(i);

                let mut h_xi = 0.0;
                for k in 0..n_classes {
                    let p_k = probs[k].clamp(1e-7, 1.0 - 1e-7);

                    if k == class_idx {
                        h_xi += (n_classes as Float - 1.0) * p_k.ln();
                    } else {
                        h_xi -= p_k.ln();
                    }
                }

                let weight_multiplier = (-factor * h_xi / n_classes as Float).exp();
                new_weights[i] *= weight_multiplier;
                new_weights[i] = new_weights[i].clamp(1e-10, 1e3);
            }
        }

        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
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

        for i in 0..n_samples {
            let true_class = y[i];
            let y_i = if true_class == classes[0] { -1.0 } else { 1.0 };

            let p_0 = prob_estimates[[i, 0]].clamp(1e-7, 1.0 - 1e-7);
            let p_1 = prob_estimates[[i, 1]].clamp(1e-7, 1.0 - 1e-7);

            let h_xi = 0.5 * (p_1 / p_0).ln();

            let weight_multiplier = (-y_i * h_xi).exp();
            new_weights[i] *= weight_multiplier;
            new_weights[i] = new_weights[i].clamp(1e-10, 1e3);
        }

        let weight_sum = new_weights.sum();
        if weight_sum > 0.0 {
            new_weights /= weight_sum;
        } else {
            new_weights.fill(1.0 / n_samples as Float);
        }

        new_weights
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
        let classes = Self::find_classes(y);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "AdaBoost requires at least 2 classes".to_string(),
            ));
        }
        let mut sample_weight = Array1::<Float>::from_elem(n_samples, 1.0 / n_samples as Float);
        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();
        let mut rng = match self.config.random_state {
            Some(seed) => scirs2_core::random::seeded_rng(seed),
            None => scirs2_core::random::seeded_rng(42),
        };
        let y_i32 = convert_labels_to_i32(y);
        for _iteration in 0..self.config.n_estimators {
            let base_estimator = DecisionTreeClassifier::new()
                .criterion(SplitCriterion::Gini)
                .max_depth(1)
                .min_samples_split(2)
                .min_samples_leaf(1);
            match self.config.algorithm {
                AdaBoostAlgorithm::SAMME => {
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }
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
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let prob_estimates =
                        estimate_probabilities(&y_pred, &classes, n_samples, n_classes);
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }
                    let estimator_weight = self.config.learning_rate;
                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);
                    sample_weight = self.calculate_sample_weights_sammer(
                        y,
                        &prob_estimates,
                        &sample_weight,
                        &classes,
                        estimator_weight,
                    );
                    if !(1e-10..0.5).contains(&weighted_error) {
                        break;
                    }
                }
                AdaBoostAlgorithm::RealAdaBoost => {
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    if n_classes != 2 {
                        return Err(SklearsError::InvalidInput(
                            "Real AdaBoost currently supports only binary classification"
                                .to_string(),
                        ));
                    }
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let prob_estimates = estimate_binary_probabilities(&y_pred, &classes);
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        let correct_class_idx = if y[i] == classes[0] { 0 } else { 1 };
                        let prob_correct = prob_estimates[[i, correct_class_idx]];
                        if prob_correct < 0.5 {
                            weighted_error += sample_weight[i];
                        }
                    }
                    let estimator_weight = if weighted_error > 0.0 && weighted_error < 0.5 {
                        0.5 * ((1.0 - weighted_error) / weighted_error).ln()
                    } else if weighted_error == 0.0 {
                        10.0
                    } else {
                        0.0
                    };
                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);
                    sample_weight = self.calculate_sample_weights_real_adaboost(
                        y,
                        &prob_estimates,
                        &sample_weight,
                        &classes,
                    );
                    if !(1e-10..0.5).contains(&weighted_error) {
                        break;
                    }
                }
                AdaBoostAlgorithm::M1 => {
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }
                    if weighted_error >= 0.5 {
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
                    sample_weight =
                        self.calculate_sample_weights(y, &y_pred, &sample_weight, estimator_weight);
                    if weighted_error < 1e-10 {
                        break;
                    }
                }
                AdaBoostAlgorithm::M2 => {
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let prob_estimates =
                        estimate_probabilities(&y_pred, &classes, n_samples, n_classes);
                    let mut pseudo_loss = 0.0;
                    for i in 0..n_samples {
                        let true_class_idx = classes.iter().position(|&c| c == y[i]).unwrap_or(0);
                        let mut margin = prob_estimates[[i, true_class_idx]];
                        for j in 0..n_classes {
                            if j != true_class_idx {
                                margin -= prob_estimates[[i, j]] / (n_classes - 1) as Float;
                            }
                        }
                        if margin <= 0.0 {
                            pseudo_loss += sample_weight[i] * (1.0 - margin);
                        }
                    }
                    let total_weight: Float = sample_weight.sum();
                    if total_weight > 0.0 {
                        pseudo_loss /= total_weight;
                    }
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
                    let mut new_weights = sample_weight.clone();
                    for i in 0..n_samples {
                        let true_class_idx = classes.iter().position(|&c| c == y[i]).unwrap_or(0);
                        let confidence = prob_estimates[[i, true_class_idx]];
                        let weight_multiplier = if confidence < 0.5 {
                            (estimator_weight * (1.0 - confidence)).exp()
                        } else {
                            (estimator_weight * confidence).exp().recip()
                        };
                        new_weights[i] *= weight_multiplier;
                    }
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
                AdaBoostAlgorithm::Gentle | AdaBoostAlgorithm::Discrete => {
                    let (x_resampled, y_resampled) =
                        self.resample_data(x, &y_i32, &sample_weight, &mut rng)?;
                    let fitted_estimator = base_estimator.fit(&x_resampled, &y_resampled)?;
                    let y_pred_i32 = fitted_estimator.predict(x)?;
                    let y_pred = convert_predictions_to_float(&y_pred_i32);
                    if y_pred.len() != n_samples {
                        return Err(SklearsError::ShapeMismatch {
                            expected: format!("{} predictions", n_samples),
                            actual: format!("{} predictions", y_pred.len()),
                        });
                    }
                    let mut weighted_error = 0.0;
                    for i in 0..n_samples {
                        if y[i] != y_pred[i] {
                            weighted_error += sample_weight[i];
                        }
                    }
                    if weighted_error >= 0.6 {
                        if estimators.is_empty() {
                            estimators.push(fitted_estimator);
                            estimator_weights.push(0.1);
                            estimator_errors.push(weighted_error);
                        }
                        break;
                    }
                    let estimator_weight =
                        self.calculate_estimator_weight(weighted_error, n_classes);
                    estimators.push(fitted_estimator);
                    estimator_weights.push(estimator_weight);
                    estimator_errors.push(weighted_error);
                    let mut new_weights = sample_weight.clone();
                    let gentle_factor = 0.5;
                    for i in 0..n_samples {
                        let multiplier = if y[i] != y_pred[i] {
                            (gentle_factor * estimator_weight).exp()
                        } else {
                            (-gentle_factor * estimator_weight).exp()
                        };
                        new_weights[i] *= multiplier;
                    }
                    let weight_sum = new_weights.sum();
                    if weight_sum > 0.0 {
                        new_weights /= weight_sum;
                        let smoothing = 0.01;
                        let uniform_weight = 1.0 / n_samples as Float;
                        for i in 0..n_samples {
                            new_weights[i] =
                                (1.0 - smoothing) * new_weights[i] + smoothing * uniform_weight;
                        }
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
            AdaBoostAlgorithm::Gentle | AdaBoostAlgorithm::Discrete => {
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

impl Predict<Array2<Float>, Array1<Float>> for AdaBoostClassifier<Trained> {
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

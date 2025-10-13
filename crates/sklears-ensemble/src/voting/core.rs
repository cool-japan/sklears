//! Core VotingClassifier implementation with type-safe state management

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::voting::{
    config::{
        EnsembleSizeAnalysis, EnsembleSizeRecommendations, VotingClassifierConfig, VotingStrategy,
    },
    ensemble::{ensemble_utils, EnsembleMember},
    simd_ops::{
        simd_bayesian_averaging, simd_bootstrap_aggregate, simd_confidence_weighted_voting,
        simd_ensemble_disagreement, simd_entropy_weighted_voting, simd_hard_voting_weighted,
        simd_soft_voting_weighted, simd_variance_weighted_voting,
    },
    strategies::{
        adaptive_ensemble_voting, consensus_voting, dynamic_weight_adjustment, meta_voting,
        rank_based_voting, temperature_scaled_voting, uncertainty_aware_voting,
    },
};

/// Voting Classifier with type-safe state management
///
/// A voting classifier is an ensemble meta-algorithm that fits several base
/// classifiers, each on the whole dataset. It then aggregates the individual
/// predictions to form a final prediction.
///
/// # Examples
///
/// ```rust
/// use sklears_ensemble::voting::{VotingClassifier, VotingStrategy};
/// use sklears_ensemble::voting::config::VotingClassifierConfig;
///
/// // Create voting classifier with configuration
/// let config = VotingClassifierConfig {
///     voting: VotingStrategy::Soft,
///     weights: Some(vec![1.0, 2.0, 1.5]),
///     confidence_weighting: true,
///     confidence_threshold: 0.8,
///     ..Default::default()
/// };
/// let classifier = VotingClassifier::new(config);
/// assert_eq!(classifier.estimators().len(), 0);
/// ```
pub struct VotingClassifier<State = Untrained> {
    config: VotingClassifierConfig,
    estimators_: Vec<Box<dyn EnsembleMember + Send + Sync>>,
    classes_: Option<Array1<Float>>,
    n_features_in_: Option<usize>,
    state: PhantomData<State>,
}

/// Type alias for trained voting classifier
pub type TrainedVotingClassifier = VotingClassifier<Trained>;

impl VotingClassifier<Untrained> {
    /// Create a new untrained voting classifier
    pub fn new(config: VotingClassifierConfig) -> Self {
        Self {
            config,
            estimators_: Vec::new(),
            classes_: None,
            n_features_in_: None,
            state: PhantomData,
        }
    }

    /// Create a builder for configuring the voting classifier
    pub fn builder() -> VotingClassifierBuilder {
        VotingClassifierBuilder::new()
    }

    /// Add an estimator to the ensemble
    pub fn add_estimator(&mut self, estimator: Box<dyn EnsembleMember + Send + Sync>) {
        self.estimators_.push(estimator);
    }

    /// Get the current configuration
    pub fn config(&self) -> &VotingClassifierConfig {
        &self.config
    }

    /// Get the estimators in the ensemble
    pub fn estimators(&self) -> &[Box<dyn EnsembleMember + Send + Sync>] {
        &self.estimators_
    }

    /// Optimize ensemble size based on training data
    pub fn optimize_ensemble_size(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<EnsembleSizeRecommendations> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Simple heuristics for ensemble size recommendations
        let min_size = if n_features > 100 { 5 } else { 3 };
        let max_size = (n_samples / 10).min(50).max(10);
        let sweet_spot = (n_features / 5).max(5).min(20);
        let diminishing_returns_threshold = sweet_spot + (sweet_spot / 2);

        Ok(EnsembleSizeRecommendations {
            min_size,
            max_size,
            sweet_spot,
            diminishing_returns_threshold,
        })
    }

    /// Analyze ensemble size performance characteristics
    pub fn analyze_ensemble_size(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<EnsembleSizeAnalysis> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Generate synthetic performance and diversity curves
        let sizes: Vec<usize> = (1..=20).collect();
        let mut performance_curve = Array1::zeros(sizes.len());
        let mut diversity_curve = Array1::zeros(sizes.len());

        for (i, &size) in sizes.iter().enumerate() {
            // Simple model for performance curve: logarithmic growth with plateau
            let perf = 0.5 + 0.45 * (1.0 - (-0.3 * size as Float).exp());
            performance_curve[i] = perf;

            // Simple model for diversity curve: increase then plateau
            let div = 0.1 + 0.7 * (1.0 - (-0.2 * size as Float).exp());
            diversity_curve[i] = div;
        }

        let optimal_size = 8;
        let performance_plateau_size = 15;
        let diversity_saturation_size = 12;

        Ok(EnsembleSizeAnalysis {
            performance_curve,
            diversity_curve,
            optimal_size,
            performance_plateau_size,
            diversity_saturation_size,
        })
    }
}

impl VotingClassifier<Trained> {
    /// Get the classes discovered during training
    pub fn classes(&self) -> &Array1<Float> {
        self.classes_.as_ref().unwrap()
    }

    /// Get the number of features seen during training
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the estimators in the ensemble
    pub fn estimators(&self) -> &[Box<dyn EnsembleMember + Send + Sync>] {
        &self.estimators_
    }

    /// Make predictions with confidence scores
    pub fn predict_with_confidence(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let predictions = self.predict(x)?;

        // Calculate confidence based on ensemble agreement
        let n_samples = x.nrows();
        let mut confidence = Array1::ones(n_samples);

        if self.estimators_.len() > 1 {
            // Collect all predictions
            let mut all_predictions = Vec::new();
            for estimator in &self.estimators_ {
                if estimator.is_fitted() {
                    match estimator.predict(x) {
                        Ok(pred) => all_predictions.push(pred),
                        Err(_) => continue,
                    }
                }
            }

            // Calculate ensemble disagreement as inverse confidence
            if !all_predictions.is_empty() {
                let mut disagreement = Array1::zeros(n_samples);
                if simd_ensemble_disagreement(&all_predictions, &mut disagreement).is_ok() {
                    for i in 0..n_samples {
                        confidence[i] = 1.0 / (1.0 + disagreement[i]);
                    }
                }
            }
        }

        Ok((predictions, confidence))
    }

    /// Make predictions with confidence weighting
    pub fn predict_with_confidence_weighting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: x.ncols(),
            });
        }

        // Apply confidence weighting to all estimator predictions
        let mut all_predictions = Vec::new();
        let mut confidence_weights = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => {
                        all_predictions.push(pred);
                        confidence_weights.push(estimator.confidence());
                    }
                    Err(_) => continue,
                }
            }
        }

        if all_predictions.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No fitted estimators available".to_string(),
            ));
        }

        // Use weighted voting based on confidence scores
        let n_samples = x.nrows();
        let mut result = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            let mut weight_sum = 0.0;

            for (pred_idx, prediction) in all_predictions.iter().enumerate() {
                let weight = confidence_weights[pred_idx];
                weighted_sum += prediction[sample_idx] * weight;
                weight_sum += weight;
            }

            result[sample_idx] = if weight_sum > 1e-8 {
                weighted_sum / weight_sum
            } else {
                all_predictions
                    .iter()
                    .map(|pred| pred[sample_idx])
                    .sum::<Float>()
                    / all_predictions.len() as Float
            };
        }

        Ok(result)
    }

    /// Get confidence scores for individual estimators
    pub fn estimator_confidence_scores(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut confidence_scores = Array1::zeros(self.estimators_.len());

        for (i, estimator) in self.estimators_.iter().enumerate() {
            confidence_scores[i] = estimator.confidence();
        }

        Ok(confidence_scores)
    }

    /// Make probability predictions (if supported by estimators)
    pub fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.ncols() != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: x.ncols(),
            });
        }

        let mut all_probabilities = Vec::new();
        let mut weights = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() && estimator.supports_proba() {
                match estimator.predict_proba(x) {
                    Ok(proba) => {
                        all_probabilities.push(proba);
                        weights.push(estimator.weight());
                    }
                    Err(_) => continue,
                }
            }
        }

        if all_probabilities.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No estimators support probability predictions".to_string(),
            ));
        }

        // Use the configured voting strategy for probability aggregation
        match self.config.voting {
            VotingStrategy::Soft | VotingStrategy::Weighted => {
                Ok(simd_soft_voting_weighted(&all_probabilities, &weights))
            }
            VotingStrategy::EntropyWeighted => Ok(simd_entropy_weighted_voting(
                &all_probabilities,
                self.config.entropy_weight_factor as f32,
            )),
            VotingStrategy::VarianceWeighted => Ok(simd_variance_weighted_voting(
                &all_probabilities,
                self.config.variance_weight_factor as f32,
            )),
            VotingStrategy::ConfidenceWeighted => Ok(simd_confidence_weighted_voting(
                &all_probabilities,
                self.config.confidence_threshold as f32,
            )),
            VotingStrategy::BayesianAveraging => {
                // Use weights as model evidences
                Ok(simd_bayesian_averaging(&all_probabilities, &weights))
            }
            VotingStrategy::TemperatureScaled => {
                temperature_scaled_voting(&all_probabilities, self.config.temperature as f32)
            }
            _ => {
                // Fall back to simple soft voting
                Ok(simd_soft_voting_weighted(&all_probabilities, &weights))
            }
        }
    }

    /// Update ensemble weights dynamically based on performance
    pub fn update_weights_dynamically(&mut self, recent_performances: &[Float]) -> Result<()> {
        if recent_performances.len() != self.estimators_.len() {
            return Err(SklearsError::InvalidParameter {
                name: "recent_performances".to_string(),
                reason: "Performance array length must match number of estimators".to_string(),
            });
        }

        let current_weights: Vec<Float> = self.estimators_.iter().map(|e| e.weight()).collect();

        let new_weights = dynamic_weight_adjustment(
            &current_weights,
            recent_performances,
            self.config.weight_adjustment_rate as f32,
        )?;

        for (estimator, &new_weight) in self.estimators_.iter_mut().zip(new_weights.iter()) {
            estimator.set_weight(new_weight);
        }

        Ok(())
    }

    /// Get ensemble size recommendations
    pub fn get_ensemble_size_recommendations(&self) -> EnsembleSizeRecommendations {
        let current_size = self.estimators_.len();

        EnsembleSizeRecommendations {
            min_size: 3,
            max_size: current_size * 2,
            sweet_spot: (current_size + 5).min(15),
            diminishing_returns_threshold: current_size + 10,
        }
    }
}

// Default implementation for untrained classifier
impl Default for VotingClassifier<Untrained> {
    fn default() -> Self {
        Self::new(VotingClassifierConfig::default())
    }
}

// Debug implementations
impl std::fmt::Debug for VotingClassifier<Untrained> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VotingClassifier<Untrained>")
            .field("config", &self.config)
            .field("n_estimators", &self.estimators_.len())
            .finish()
    }
}

impl std::fmt::Debug for VotingClassifier<Trained> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VotingClassifier<Trained>")
            .field("config", &self.config)
            .field("n_estimators", &self.estimators_.len())
            .field("classes", &self.classes_)
            .field("n_features_in", &self.n_features_in_)
            .finish()
    }
}

// Implement Predict trait for trained classifier
impl Predict<Array2<Float>, Array1<Float>> for VotingClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: x.ncols(),
            });
        }

        if self.estimators_.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No estimators in ensemble".to_string(),
            ));
        }

        // Apply the configured voting strategy
        match self.config.voting {
            VotingStrategy::Hard => self.hard_voting(x),
            VotingStrategy::Soft => self.soft_voting(x),
            VotingStrategy::Weighted => self.weighted_voting(x),
            VotingStrategy::ConfidenceWeighted => self.confidence_weighted_voting(x),
            VotingStrategy::BayesianAveraging => self.bayesian_averaging(x),
            VotingStrategy::RankBased => self.rank_based_voting(x),
            VotingStrategy::MetaVoting => self.meta_voting(x),
            VotingStrategy::DynamicWeightAdjustment => self.dynamic_voting(x),
            VotingStrategy::UncertaintyAware => self.uncertainty_aware_voting(x),
            VotingStrategy::ConsensusBased => self.consensus_voting(x),
            VotingStrategy::EntropyWeighted => self.entropy_weighted_voting(x),
            VotingStrategy::VarianceWeighted => self.variance_weighted_voting(x),
            VotingStrategy::BootstrapAggregation => self.bootstrap_voting(x),
            VotingStrategy::TemperatureScaled => self.temperature_scaled_voting(x),
            VotingStrategy::AdaptiveEnsemble => self.adaptive_voting(x),
        }
    }
}

// Private implementation methods for different voting strategies
impl VotingClassifier<Trained> {
    fn hard_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_predictions = Vec::new();
        let mut weights = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => {
                        all_predictions.push(pred);
                        weights.push(estimator.weight());
                    }
                    Err(_) => continue,
                }
            }
        }

        if all_predictions.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No fitted estimators available".to_string(),
            ));
        }

        let classes = self.classes();
        Ok(simd_hard_voting_weighted(
            &all_predictions,
            &weights,
            classes.as_slice().unwrap(),
        ))
    }

    fn soft_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut result = Array1::zeros(n_samples);

        // Convert probabilities to class predictions
        for i in 0..n_samples {
            let mut max_prob = probabilities[[i, 0]];
            let mut best_class = 0;

            for j in 1..probabilities.ncols() {
                if probabilities[[i, j]] > max_prob {
                    max_prob = probabilities[[i, j]];
                    best_class = j;
                }
            }

            result[i] = best_class as Float;
        }

        Ok(result)
    }

    fn weighted_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // Use configured weights if available, otherwise use estimator weights
        let weights = if let Some(ref config_weights) = self.config.weights {
            config_weights.clone()
        } else {
            self.estimators_.iter().map(|e| e.weight()).collect()
        };

        let mut all_predictions = Vec::new();
        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }
            }
        }

        let classes = self.classes();
        Ok(simd_hard_voting_weighted(
            &all_predictions,
            &weights,
            classes.as_slice().unwrap(),
        ))
    }

    fn confidence_weighted_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        self.predict_with_confidence_weighting(x)
    }

    fn bayesian_averaging(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut max_prob = probabilities[[i, 0]];
            let mut best_class = 0;

            for j in 1..probabilities.ncols() {
                if probabilities[[i, j]] > max_prob {
                    max_prob = probabilities[[i, j]];
                    best_class = j;
                }
            }

            result[i] = best_class as Float;
        }

        Ok(result)
    }

    fn rank_based_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_probabilities = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() && estimator.supports_proba() {
                match estimator.predict_proba(x) {
                    Ok(proba) => all_probabilities.push(proba),
                    Err(_) => continue,
                }
            }
        }

        if all_probabilities.is_empty() {
            return self.hard_voting(x);
        }

        rank_based_voting(&all_probabilities)
    }

    fn meta_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // For simplicity, use uniform meta-weights
        let n_samples = x.nrows();
        let n_estimators = self.estimators_.len();
        let meta_weights = Array2::ones((n_samples, n_estimators)) / n_estimators as Float;

        let mut all_predictions = Vec::new();
        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }
            }
        }

        meta_voting(&all_predictions, &meta_weights)
    }

    fn dynamic_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // Use recent performance as dynamic weights
        let performances: Vec<Float> = self.estimators_.iter().map(|e| e.performance()).collect();

        let mut all_predictions = Vec::new();
        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }
            }
        }

        let classes = self.classes();
        Ok(simd_hard_voting_weighted(
            &all_predictions,
            &performances,
            classes.as_slice().unwrap(),
        ))
    }

    fn uncertainty_aware_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_predictions = Vec::new();
        let mut all_uncertainties = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match (estimator.predict(x), estimator.uncertainty(x)) {
                    (Ok(pred), Ok(unc)) => {
                        all_predictions.push(pred);
                        all_uncertainties.push(unc);
                    }
                    _ => continue,
                }
            }
        }

        if all_predictions.is_empty() {
            return self.hard_voting(x);
        }

        uncertainty_aware_voting(
            &all_predictions,
            &all_uncertainties,
            self.config.confidence_threshold as f32,
        )
    }

    fn consensus_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_predictions = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }
            }
        }

        consensus_voting(&all_predictions, self.config.consensus_threshold as f32)
    }

    fn entropy_weighted_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut max_prob = probabilities[[i, 0]];
            let mut best_class = 0;

            for j in 1..probabilities.ncols() {
                if probabilities[[i, j]] > max_prob {
                    max_prob = probabilities[[i, j]];
                    best_class = j;
                }
            }

            result[i] = best_class as Float;
        }

        Ok(result)
    }

    fn variance_weighted_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut max_prob = probabilities[[i, 0]];
            let mut best_class = 0;

            for j in 1..probabilities.ncols() {
                if probabilities[[i, j]] > max_prob {
                    max_prob = probabilities[[i, j]];
                    best_class = j;
                }
            }

            result[i] = best_class as Float;
        }

        Ok(result)
    }

    fn bootstrap_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_predictions = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }
            }
        }

        if all_predictions.is_empty() {
            return Err(SklearsError::InvalidOperation(
                "No fitted estimators available".to_string(),
            ));
        }

        let mut result = Array1::zeros(x.nrows());
        simd_bootstrap_aggregate(
            &all_predictions,
            self.config.n_bootstrap_samples,
            &mut result,
        )?;
        Ok(result)
    }

    fn temperature_scaled_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = probabilities.nrows();
        let mut result = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut max_prob = probabilities[[i, 0]];
            let mut best_class = 0;

            for j in 1..probabilities.ncols() {
                if probabilities[[i, j]] > max_prob {
                    max_prob = probabilities[[i, j]];
                    best_class = j;
                }
            }

            result[i] = best_class as Float;
        }

        Ok(result)
    }

    fn adaptive_voting(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let mut all_predictions = Vec::new();
        let mut all_probabilities = Vec::new();

        for estimator in &self.estimators_ {
            if estimator.is_fitted() {
                match estimator.predict(x) {
                    Ok(pred) => all_predictions.push(pred),
                    Err(_) => continue,
                }

                if estimator.supports_proba() {
                    if let Ok(proba) = estimator.predict_proba(x) {
                        all_probabilities.push(proba);
                    }
                }
            }
        }

        let diversity =
            ensemble_utils::calculate_ensemble_diversity(&self.estimators_, x).unwrap_or(0.5);
        let performance_history: Vec<f32> = self
            .estimators_
            .iter()
            .map(|e| e.performance() as f32)
            .collect();

        let probabilities = if all_probabilities.is_empty() {
            None
        } else {
            Some(&all_probabilities[..])
        };

        adaptive_ensemble_voting(
            &all_predictions,
            probabilities,
            diversity as f32,
            &performance_history,
        )
    }
}

// Implement Fit trait for VotingClassifier
impl Fit<Array2<Float>, Array1<Float>> for VotingClassifier<Untrained> {
    type Fitted = VotingClassifier<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Discover unique classes
        let mut classes: Vec<Float> = y.iter().cloned().collect();
        classes.sort_by(|a, b| a.partial_cmp(b).unwrap());
        classes.dedup();
        let classes_array = Array1::from_vec(classes);

        Ok(VotingClassifier {
            config: self.config,
            estimators_: self.estimators_,
            classes_: Some(classes_array),
            n_features_in_: Some(x.ncols()),
            state: PhantomData,
        })
    }
}

/// Builder for VotingClassifier
#[derive(Debug)]
pub struct VotingClassifierBuilder {
    config: VotingClassifierConfig,
}

impl VotingClassifierBuilder {
    pub fn new() -> Self {
        Self {
            config: VotingClassifierConfig::default(),
        }
    }

    pub fn voting(mut self, voting: VotingStrategy) -> Self {
        self.config.voting = voting;
        self
    }

    pub fn weights(mut self, weights: Vec<Float>) -> Self {
        self.config.weights = Some(weights);
        self
    }

    pub fn confidence_weighting(mut self, enable: bool) -> Self {
        self.config.confidence_weighting = enable;
        self
    }

    pub fn confidence_threshold(mut self, threshold: Float) -> Self {
        self.config.confidence_threshold = threshold;
        self
    }

    pub fn min_confidence_weight(mut self, weight: Float) -> Self {
        self.config.min_confidence_weight = weight;
        self
    }

    pub fn enable_uncertainty(mut self, enable: bool) -> Self {
        self.config.enable_uncertainty = enable;
        self
    }

    pub fn temperature(mut self, temp: Float) -> Self {
        self.config.temperature = temp;
        self
    }

    pub fn meta_regularization(mut self, reg: Float) -> Self {
        self.config.meta_regularization = reg;
        self
    }

    pub fn n_bootstrap_samples(mut self, n: usize) -> Self {
        self.config.n_bootstrap_samples = n;
        self
    }

    pub fn consensus_threshold(mut self, threshold: Float) -> Self {
        self.config.consensus_threshold = threshold;
        self
    }

    pub fn entropy_weight_factor(mut self, factor: Float) -> Self {
        self.config.entropy_weight_factor = factor;
        self
    }

    pub fn variance_weight_factor(mut self, factor: Float) -> Self {
        self.config.variance_weight_factor = factor;
        self
    }

    pub fn weight_adjustment_rate(mut self, rate: Float) -> Self {
        self.config.weight_adjustment_rate = rate;
        self
    }

    // Convenience methods for common configurations
    pub fn confidence_weighted() -> Self {
        Self::new()
            .voting(VotingStrategy::ConfidenceWeighted)
            .confidence_weighting(true)
    }

    pub fn bayesian_averaging() -> Self {
        Self::new().voting(VotingStrategy::BayesianAveraging)
    }

    pub fn meta_voting() -> Self {
        Self::new().voting(VotingStrategy::MetaVoting)
    }

    pub fn uncertainty_aware() -> Self {
        Self::new()
            .voting(VotingStrategy::UncertaintyAware)
            .enable_uncertainty(true)
    }

    pub fn entropy_weighted() -> Self {
        Self::new()
            .voting(VotingStrategy::EntropyWeighted)
            .entropy_weight_factor(1.0)
    }

    pub fn variance_weighted() -> Self {
        Self::new()
            .voting(VotingStrategy::VarianceWeighted)
            .variance_weight_factor(1.0)
    }

    pub fn consensus_based() -> Self {
        Self::new()
            .voting(VotingStrategy::ConsensusBased)
            .consensus_threshold(0.7)
    }

    pub fn build(self) -> VotingClassifier<Untrained> {
        VotingClassifier::new(self.config)
    }
}

impl Default for VotingClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

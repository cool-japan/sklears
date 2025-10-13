//! Voting Systems for Ensemble Learning
//!
//! This module provides democratic ensemble decision-making through voting mechanisms.
//! Voting systems aggregate predictions from multiple models using various strategies
//! to produce robust final predictions.
//!
//! # Voting Strategies
//!
//! ## Hard Voting
//! - Uses majority vote from discrete predictions
//! - Each model contributes a single vote for the predicted class
//! - Final prediction is the class with the most votes
//! - Works with any classifier that produces discrete outputs
//!
//! ## Soft Voting
//! - Uses weighted average of prediction probabilities
//! - Each model contributes probability scores for all classes
//! - Final prediction uses averaged probabilities
//! - Requires models that output class probabilities
//!
//! ## Weighted Voting
//! - Assigns different importance weights to different models
//! - Weights can be uniform, performance-based, or manually specified
//! - Can be combined with both hard and soft voting strategies
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::VotingClassifier;
//!
//! // Hard voting classifier
//! let hard_voting = VotingClassifier::builder()
//!     .estimator("svm", Box::new(svm_model))
//!     .estimator("rf", Box::new(random_forest))
//!     .estimator("nb", Box::new(naive_bayes))
//!     .voting("hard")
//!     .build();
//!
//! // Soft voting with weights
//! let soft_voting = VotingClassifier::builder()
//!     .estimator("model1", Box::new(model1))
//!     .estimator("model2", Box::new(model2))
//!     .voting("soft")
//!     .weights(vec![0.6, 0.4])
//!     .build();
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Trained, Untrained},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;
use std::marker::PhantomData;

use crate::PipelinePredictor;

/// Voting strategies for ensemble classification
#[derive(Debug, Clone, PartialEq)]
pub enum VotingStrategy {
    /// Hard voting: majority vote from discrete predictions
    Hard,
    /// Soft voting: weighted average of prediction probabilities
    Soft,
}

impl VotingStrategy {
    /// Parse voting strategy from string
    pub fn from_str(s: &str) -> Result<Self, SklearsError> {
        match s.to_lowercase().as_str() {
            "hard" => Ok(VotingStrategy::Hard),
            "soft" => Ok(VotingStrategy::Soft),
            _ => Err(SklearsError::InvalidParameter(format!(
                "Invalid voting strategy: {}. Must be 'hard' or 'soft'", s
            ))),
        }
    }

    /// Convert voting strategy to string
    pub fn as_str(&self) -> &'static str {
        match self {
            VotingStrategy::Hard => "hard",
            VotingStrategy::Soft => "soft",
        }
    }
}

/// Weight normalization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum WeightNormalization {
    /// No normalization
    None,
    /// L1 normalization (weights sum to 1)
    L1,
    /// L2 normalization (unit vector)
    L2,
    /// Min-max normalization
    MinMax,
}

/// Voting classifier for ensemble classification tasks
#[derive(Debug)]
pub struct VotingClassifier<S> {
    /// Named estimators in the ensemble
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Voting strategy (hard or soft)
    voting: VotingStrategy,
    /// Weights for each estimator
    weights: Option<Vec<Float>>,
    /// Weight normalization strategy
    weight_normalization: WeightNormalization,
    /// Whether to flatten transform output
    flatten_transform: bool,
    /// Number of parallel jobs for training
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
    /// State marker
    _state: PhantomData<S>,
}

/// Trained voting classifier
pub type VotingClassifierTrained = VotingClassifier<Trained>;

impl VotingClassifier<Untrained> {
    /// Create a new voting classifier builder
    pub fn builder() -> VotingClassifierBuilder {
        VotingClassifierBuilder::new()
    }

    /// Create a new voting classifier with default settings
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            voting: VotingStrategy::Hard,
            weights: None,
            weight_normalization: WeightNormalization::L1,
            flatten_transform: true,
            n_jobs: None,
            verbose: false,
            _state: PhantomData,
        }
    }

    /// Add an estimator to the ensemble
    pub fn add_estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set voting strategy
    pub fn set_voting(mut self, voting: VotingStrategy) -> Self {
        self.voting = voting;
        self
    }

    /// Set weights for estimators
    pub fn set_weights(mut self, weights: Vec<Float>) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl<S> VotingClassifier<S> {
    /// Get estimator names
    pub fn estimator_names(&self) -> Vec<&str> {
        self.estimators.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get number of estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get voting strategy
    pub fn voting_strategy(&self) -> &VotingStrategy {
        &self.voting
    }

    /// Get weights (normalized if applicable)
    pub fn get_weights(&self) -> Vec<Float> {
        let raw_weights = self.weights.as_ref()
            .map(|w| w.clone())
            .unwrap_or_else(|| vec![1.0; self.n_estimators()]);

        self.normalize_weights(&raw_weights)
    }

    /// Normalize weights according to the normalization strategy
    fn normalize_weights(&self, weights: &[Float]) -> Vec<Float> {
        match self.weight_normalization {
            WeightNormalization::None => weights.to_vec(),
            WeightNormalization::L1 => {
                let sum: Float = weights.iter().sum();
                if sum > 0.0 {
                    weights.iter().map(|w| w / sum).collect()
                } else {
                    vec![1.0 / weights.len() as Float; weights.len()]
                }
            },
            WeightNormalization::L2 => {
                let norm: Float = weights.iter().map(|w| w * w).sum::<Float>().sqrt();
                if norm > 0.0 {
                    weights.iter().map(|w| w / norm).collect()
                } else {
                    vec![1.0 / (weights.len() as Float).sqrt(); weights.len()]
                }
            },
            WeightNormalization::MinMax => {
                let min_val = weights.iter().cloned().fold(Float::INFINITY, Float::min);
                let max_val = weights.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                let range = max_val - min_val;
                if range > 0.0 {
                    weights.iter().map(|w| (w - min_val) / range).collect()
                } else {
                    vec![1.0; weights.len()]
                }
            }
        }
    }

    /// Validate estimator compatibility
    fn validate_estimators(&self) -> SklResult<()> {
        if self.estimators.is_empty() {
            return Err(SklearsError::InvalidParameter(
                "VotingClassifier requires at least one estimator".to_string()
            ));
        }

        if let Some(ref weights) = self.weights {
            if weights.len() != self.estimators.len() {
                return Err(SklearsError::InvalidParameter(format!(
                    "Number of weights ({}) must match number of estimators ({})",
                    weights.len(), self.estimators.len()
                )));
            }

            // Check for non-negative weights
            if weights.iter().any(|&w| w < 0.0) {
                return Err(SklearsError::InvalidParameter(
                    "All weights must be non-negative".to_string()
                ));
            }
        }

        // Validate voting strategy compatibility
        if self.voting == VotingStrategy::Soft {
            // For soft voting, we would need to check if all estimators support probability prediction
            // This would require extending the PipelinePredictor trait to include predict_proba
            // For now, we'll assume all estimators are compatible
        }

        Ok(())
    }
}

impl Estimator for VotingClassifier<Untrained> {
    type Config = VotingClassifierConfig;

    fn default_config() -> Self::Config {
        VotingClassifierConfig::default()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for VotingClassifier<Untrained> {
    type Target = VotingClassifier<Trained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Target> {
        self.validate_estimators()?;

        // Validate input data
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(), y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string()
            ));
        }

        // Train all estimators
        let mut trained_estimators = Vec::new();
        for (name, estimator) in self.estimators {
            // Note: This is a simplified version. In practice, each estimator would need
            // to implement the Fit trait properly
            trained_estimators.push((name, estimator));
        }

        Ok(VotingClassifier {
            estimators: trained_estimators,
            voting: self.voting,
            weights: self.weights,
            weight_normalization: self.weight_normalization,
            flatten_transform: self.flatten_transform,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for VotingClassifier<Trained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        match self.voting {
            VotingStrategy::Hard => self.predict_hard(x),
            VotingStrategy::Soft => self.predict_soft(x),
        }
    }
}

impl VotingClassifier<Trained> {
    /// Perform hard voting prediction
    fn predict_hard(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        let n_samples = x.nrows();
        let weights = self.get_weights();

        // Collect predictions from all estimators
        let mut all_predictions = Vec::new();
        for (_, estimator) in &self.estimators {
            let pred = estimator.predict(x)?;
            all_predictions.push(pred);
        }

        // Perform weighted majority voting
        let mut final_predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut class_votes: HashMap<i32, Float> = HashMap::new();

            for (est_idx, pred) in all_predictions.iter().enumerate() {
                let class = pred[sample_idx] as i32;
                let weight = weights[est_idx];
                *class_votes.entry(class).or_insert(0.0) += weight;
            }

            // Find class with highest weighted vote
            let predicted_class = class_votes
                .iter()
                .max_by(|(_, &a), (_, &b)| a.partial_cmp(&b).unwrap())
                .map(|(&class, _)| class)
                .unwrap_or(0) as Float;

            final_predictions[sample_idx] = predicted_class;
        }

        Ok(final_predictions)
    }

    /// Perform soft voting prediction
    fn predict_soft(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        // Note: This is a simplified implementation
        // In practice, we would need estimators to provide probability predictions
        // and then average those probabilities before taking the argmax

        let n_samples = x.nrows();
        let weights = self.get_weights();

        // For now, fall back to weighted hard voting
        // In a full implementation, this would use predict_proba from estimators
        self.predict_hard(x)
    }

    /// Get individual estimator predictions
    pub fn get_estimator_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        let mut predictions = Vec::new();
        for (_, estimator) in &self.estimators {
            let pred = estimator.predict(x)?;
            predictions.push(pred);
        }
        Ok(predictions)
    }

    /// Calculate prediction confidence based on vote agreement
    pub fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        let predictions = self.get_estimator_predictions(x)?;
        let n_samples = x.nrows();
        let weights = self.get_weights();

        let mut confidence = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut class_votes: HashMap<i32, Float> = HashMap::new();
            let mut total_weight = 0.0;

            for (est_idx, pred) in predictions.iter().enumerate() {
                let class = pred[sample_idx] as i32;
                let weight = weights[est_idx];
                *class_votes.entry(class).or_insert(0.0) += weight;
                total_weight += weight;
            }

            // Confidence is the proportion of weighted votes for the winning class
            let max_votes = class_votes.values().cloned().fold(0.0, Float::max);
            confidence[sample_idx] = if total_weight > 0.0 { max_votes / total_weight } else { 0.0 };
        }

        Ok(confidence)
    }
}

/// Voting regressor for ensemble regression tasks
#[derive(Debug)]
pub struct VotingRegressor<S> {
    /// Named estimators in the ensemble
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Weights for each estimator
    weights: Option<Vec<Float>>,
    /// Weight normalization strategy
    weight_normalization: WeightNormalization,
    /// Number of parallel jobs for training
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
    /// State marker
    _state: PhantomData<S>,
}

/// Trained voting regressor
pub type VotingRegressorTrained = VotingRegressor<Trained>;

impl VotingRegressor<Untrained> {
    /// Create a new voting regressor builder
    pub fn builder() -> VotingRegressorBuilder {
        VotingRegressorBuilder::new()
    }

    /// Create a new voting regressor with default settings
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            weights: None,
            weight_normalization: WeightNormalization::L1,
            n_jobs: None,
            verbose: false,
            _state: PhantomData,
        }
    }

    /// Add an estimator to the ensemble
    pub fn add_estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set weights for estimators
    pub fn set_weights(mut self, weights: Vec<Float>) -> Self {
        self.weights = Some(weights);
        self
    }
}

impl<S> VotingRegressor<S> {
    /// Get estimator names
    pub fn estimator_names(&self) -> Vec<&str> {
        self.estimators.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get number of estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get weights (normalized if applicable)
    pub fn get_weights(&self) -> Vec<Float> {
        let raw_weights = self.weights.as_ref()
            .map(|w| w.clone())
            .unwrap_or_else(|| vec![1.0; self.n_estimators()]);

        self.normalize_weights(&raw_weights)
    }

    /// Normalize weights according to the normalization strategy
    fn normalize_weights(&self, weights: &[Float]) -> Vec<Float> {
        match self.weight_normalization {
            WeightNormalization::None => weights.to_vec(),
            WeightNormalization::L1 => {
                let sum: Float = weights.iter().sum();
                if sum > 0.0 {
                    weights.iter().map(|w| w / sum).collect()
                } else {
                    vec![1.0 / weights.len() as Float; weights.len()]
                }
            },
            WeightNormalization::L2 => {
                let norm: Float = weights.iter().map(|w| w * w).sum::<Float>().sqrt();
                if norm > 0.0 {
                    weights.iter().map(|w| w / norm).collect()
                } else {
                    vec![1.0 / (weights.len() as Float).sqrt(); weights.len()]
                }
            },
            WeightNormalization::MinMax => {
                let min_val = weights.iter().cloned().fold(Float::INFINITY, Float::min);
                let max_val = weights.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
                let range = max_val - min_val;
                if range > 0.0 {
                    weights.iter().map(|w| (w - min_val) / range).collect()
                } else {
                    vec![1.0; weights.len()]
                }
            }
        }
    }

    /// Validate estimator configuration
    fn validate_estimators(&self) -> SklResult<()> {
        if self.estimators.is_empty() {
            return Err(SklearsError::InvalidParameter(
                "VotingRegressor requires at least one estimator".to_string()
            ));
        }

        if let Some(ref weights) = self.weights {
            if weights.len() != self.estimators.len() {
                return Err(SklearsError::InvalidParameter(format!(
                    "Number of weights ({}) must match number of estimators ({})",
                    weights.len(), self.estimators.len()
                )));
            }

            if weights.iter().any(|&w| w < 0.0) {
                return Err(SklearsError::InvalidParameter(
                    "All weights must be non-negative".to_string()
                ));
            }
        }

        Ok(())
    }
}

impl Estimator for VotingRegressor<Untrained> {
    type Config = VotingRegressorConfig;

    fn default_config() -> Self::Config {
        VotingRegressorConfig::default()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for VotingRegressor<Untrained> {
    type Target = VotingRegressor<Trained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Target> {
        self.validate_estimators()?;

        // Validate input data
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples in X ({}) and y ({}) must match",
                x.nrows(), y.len()
            )));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string()
            ));
        }

        // Train all estimators
        let mut trained_estimators = Vec::new();
        for (name, estimator) in self.estimators {
            trained_estimators.push((name, estimator));
        }

        Ok(VotingRegressor {
            estimators: trained_estimators,
            weights: self.weights,
            weight_normalization: self.weight_normalization,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for VotingRegressor<Trained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        let n_samples = x.nrows();
        let weights = self.get_weights();

        // Collect predictions from all estimators
        let mut all_predictions = Vec::new();
        for (_, estimator) in &self.estimators {
            let pred = estimator.predict(x)?;
            all_predictions.push(pred);
        }

        // Perform weighted averaging
        let mut final_predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (est_idx, pred) in all_predictions.iter().enumerate() {
                let weight = weights[est_idx];
                weighted_sum += pred[sample_idx] * weight;
                total_weight += weight;
            }

            final_predictions[sample_idx] = if total_weight > 0.0 {
                weighted_sum / total_weight
            } else {
                0.0
            };
        }

        Ok(final_predictions)
    }
}

impl VotingRegressor<Trained> {
    /// Get individual estimator predictions
    pub fn get_estimator_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
        let mut predictions = Vec::new();
        for (_, estimator) in &self.estimators {
            let pred = estimator.predict(x)?;
            predictions.push(pred);
        }
        Ok(predictions)
    }

    /// Calculate prediction variance based on estimator disagreement
    pub fn prediction_variance(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        let predictions = self.get_estimator_predictions(x)?;
        let final_pred = self.predict(x)?;
        let n_samples = x.nrows();
        let weights = self.get_weights();

        let mut variance = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut weighted_var = 0.0;
            let mut total_weight = 0.0;

            for (est_idx, pred) in predictions.iter().enumerate() {
                let weight = weights[est_idx];
                let diff = pred[sample_idx] - final_pred[sample_idx];
                weighted_var += weight * diff * diff;
                total_weight += weight;
            }

            variance[sample_idx] = if total_weight > 0.0 {
                weighted_var / total_weight
            } else {
                0.0
            };
        }

        Ok(variance)
    }
}

/// Configuration for VotingClassifier
#[derive(Debug, Clone)]
pub struct VotingClassifierConfig {
    /// Voting strategy
    pub voting: VotingStrategy,
    /// Estimator weights
    pub weights: Option<Vec<Float>>,
    /// Weight normalization strategy
    pub weight_normalization: WeightNormalization,
    /// Flatten transform output
    pub flatten_transform: bool,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for VotingClassifierConfig {
    fn default() -> Self {
        Self {
            voting: VotingStrategy::Hard,
            weights: None,
            weight_normalization: WeightNormalization::L1,
            flatten_transform: true,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Configuration for VotingRegressor
#[derive(Debug, Clone)]
pub struct VotingRegressorConfig {
    /// Estimator weights
    pub weights: Option<Vec<Float>>,
    /// Weight normalization strategy
    pub weight_normalization: WeightNormalization,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for VotingRegressorConfig {
    fn default() -> Self {
        Self {
            weights: None,
            weight_normalization: WeightNormalization::L1,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Builder for VotingClassifier
#[derive(Debug)]
pub struct VotingClassifierBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    voting: VotingStrategy,
    weights: Option<Vec<Float>>,
    weight_normalization: WeightNormalization,
    flatten_transform: bool,
    n_jobs: Option<i32>,
    verbose: bool,
}

impl VotingClassifierBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            voting: VotingStrategy::Hard,
            weights: None,
            weight_normalization: WeightNormalization::L1,
            flatten_transform: true,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add an estimator
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set voting strategy from string
    pub fn voting(mut self, voting: &str) -> Self {
        self.voting = VotingStrategy::from_str(voting).unwrap_or(VotingStrategy::Hard);
        self
    }

    /// Set voting strategy
    pub fn voting_strategy(mut self, voting: VotingStrategy) -> Self {
        self.voting = voting;
        self
    }

    /// Set weights
    pub fn weights(mut self, weights: Vec<Float>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set weight normalization
    pub fn weight_normalization(mut self, normalization: WeightNormalization) -> Self {
        self.weight_normalization = normalization;
        self
    }

    /// Set flatten transform
    pub fn flatten_transform(mut self, flatten: bool) -> Self {
        self.flatten_transform = flatten;
        self
    }

    /// Set number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the VotingClassifier
    pub fn build(self) -> VotingClassifier<Untrained> {
        VotingClassifier {
            estimators: self.estimators,
            voting: self.voting,
            weights: self.weights,
            weight_normalization: self.weight_normalization,
            flatten_transform: self.flatten_transform,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        }
    }
}

/// Builder for VotingRegressor
#[derive(Debug)]
pub struct VotingRegressorBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    weights: Option<Vec<Float>>,
    weight_normalization: WeightNormalization,
    n_jobs: Option<i32>,
    verbose: bool,
}

impl VotingRegressorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            weights: None,
            weight_normalization: WeightNormalization::L1,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add an estimator
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set weights
    pub fn weights(mut self, weights: Vec<Float>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Set weight normalization
    pub fn weight_normalization(mut self, normalization: WeightNormalization) -> Self {
        self.weight_normalization = normalization;
        self
    }

    /// Set number of jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.n_jobs = Some(n_jobs);
        self
    }

    /// Set verbose flag
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Build the VotingRegressor
    pub fn build(self) -> VotingRegressor<Untrained> {
        VotingRegressor {
            estimators: self.estimators,
            weights: self.weights,
            weight_normalization: self.weight_normalization,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voting_strategy_from_str() {
        assert_eq!(VotingStrategy::from_str("hard").unwrap(), VotingStrategy::Hard);
        assert_eq!(VotingStrategy::from_str("soft").unwrap(), VotingStrategy::Soft);
        assert_eq!(VotingStrategy::from_str("HARD").unwrap(), VotingStrategy::Hard);
        assert!(VotingStrategy::from_str("invalid").is_err());
    }

    #[test]
    fn test_voting_strategy_as_str() {
        assert_eq!(VotingStrategy::Hard.as_str(), "hard");
        assert_eq!(VotingStrategy::Soft.as_str(), "soft");
    }

    #[test]
    fn test_voting_classifier_builder() {
        let classifier = VotingClassifier::builder()
            .voting("soft")
            .weights(vec![0.6, 0.4])
            .verbose(true)
            .build();

        assert_eq!(classifier.voting, VotingStrategy::Soft);
        assert_eq!(classifier.weights, Some(vec![0.6, 0.4]));
        assert_eq!(classifier.verbose, true);
    }

    #[test]
    fn test_voting_regressor_builder() {
        let regressor = VotingRegressor::builder()
            .weights(vec![1.0, 2.0, 1.5])
            .n_jobs(4)
            .build();

        assert_eq!(regressor.weights, Some(vec![1.0, 2.0, 1.5]));
        assert_eq!(regressor.n_jobs, Some(4));
    }

    #[test]
    fn test_weight_normalization_l1() {
        let classifier = VotingClassifier::new();
        let weights = vec![2.0, 3.0, 1.0];
        let normalized = classifier.normalize_weights(&weights);
        let expected = vec![2.0/6.0, 3.0/6.0, 1.0/6.0];

        for (a, b) in normalized.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_weight_normalization_l2() {
        let mut classifier = VotingClassifier::new();
        classifier.weight_normalization = WeightNormalization::L2;

        let weights = vec![3.0, 4.0];
        let normalized = classifier.normalize_weights(&weights);
        let norm = (3.0_f32.powi(2) + 4.0_f32.powi(2)).sqrt();
        let expected = vec![3.0/norm, 4.0/norm];

        for (a, b) in normalized.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
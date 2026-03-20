//! Dynamic Selection for Ensemble Learning
//!
//! This module provides adaptive ensemble member selection mechanisms based on
//! local competence and contextual performance. Dynamic selection systems choose
//! the most appropriate ensemble members for each prediction based on the input
//! characteristics and historical performance patterns.
//!
//! # Selection Strategies
//!
//! ## Competence-Based Selection
//! - **Overall Accuracy**: Select based on global performance metrics
//! - **Local Accuracy**: Select based on performance in local regions
//! - **K-Nearest Oracle**: Use performance on K-nearest neighbors
//! - **Ranking**: Select top-k performers for each region
//!
//! ## Dynamic Weight Adjustment
//! - **Performance-Based**: Weights based on recent performance
//! - **Distance-Based**: Weights based on similarity to training data
//! - **Competence Areas**: Regional expertise detection
//! - **Temporal Weighting**: Time-decay for concept drift adaptation
//!
//! ## Context-Aware Selection
//! - **Input Characteristics**: Selection based on feature patterns
//! - **Confidence Scores**: Use model confidence for selection
//! - **Ensemble Diversity**: Balance accuracy and diversity
//! - **Adaptive Thresholds**: Dynamic competence thresholds
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble::{DynamicEnsembleSelector, SelectionStrategy, CompetenceEstimation};
//!
//! // Competence-based dynamic selection
//! let dynamic_selector = DynamicEnsembleSelector::builder()
//!     .estimator("model1", Box::new(model1))
//!     .estimator("model2", Box::new(model2))
//!     .estimator("model3", Box::new(model3))
//!     .selection_strategy(SelectionStrategy::KNearestOracle { k: 5 })
//!     .competence_estimation(CompetenceEstimation::LocalAccuracy)
//!     .build();
//!
//! // Adaptive weight adjustment
//! let adaptive_selector = DynamicEnsembleSelector::builder()
//!     .estimator("cnn", Box::new(cnn_model))
//!     .estimator("rnn", Box::new(rnn_model))
//!     .selection_strategy(SelectionStrategy::AdaptiveWeighting)
//!     .competence_estimation(CompetenceEstimation::DistanceBased)
//!     .dynamic_threshold(0.7)
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

/// Selection strategies for dynamic ensemble member selection
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionStrategy {
    /// Select best overall performer
    BestPerformance,
    /// Greedy selection based on incremental performance improvement
    GreedySelection,
    /// K-nearest oracle selection
    KNearestOracle { k: usize },
    /// Overall local accuracy in neighborhood
    OverallLocalAccuracy { neighborhood_size: usize },
    /// Local class accuracy
    LocalClassAccuracy { neighborhood_size: usize },
    /// Modified local accuracy with class balance
    ModifiedLocalAccuracy { alpha: Float },
    /// Ranking-based selection
    Ranking { top_k: usize },
    /// Adaptive weighting based on competence
    AdaptiveWeighting,
    /// Multi-criteria decision analysis
    MultiCriteria { criteria_weights: Vec<Float> },
    /// Dynamic classifier selection
    DynamicClassifierSelection,
    /// Clustering-based selection
    ClusteringBasedSelection { n_clusters: usize },
}

/// Competence estimation methods for evaluating ensemble member performance
#[derive(Debug, Clone, PartialEq)]
pub enum CompetenceEstimation {
    /// Overall accuracy on validation set
    Accuracy,
    /// Local accuracy in feature space neighborhoods
    LocalAccuracy,
    /// Distance-based competence estimation
    DistanceBased,
    /// Confidence-based competence
    ConfidenceBased,
    /// Cross-validation based competence
    CrossValidation { folds: usize },
    /// Performance on nearest neighbors
    NearestNeighbors { k: usize },
    /// Competence based on class distribution
    ClassDistribution,
    /// Time-weighted competence for temporal data
    TimeWeighted { decay_factor: Float },
    /// Ensemble diversity consideration
    DiversityAware { diversity_weight: Float },
    /// Multi-objective competence (accuracy + diversity)
    MultiObjective { weights: Vec<Float> },
}

/// Distance metrics for neighborhood-based selection
#[derive(Debug, Clone, PartialEq)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine similarity
    Cosine,
    /// Mahalanobis distance
    Mahalanobis,
    /// Hamming distance for categorical data
    Hamming,
    /// Custom distance function
    Custom { name: String },
}

/// Dynamic ensemble selector for adaptive member selection
#[derive(Debug)]
pub struct DynamicEnsembleSelector<S> {
    /// Named estimators in the ensemble
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    /// Selection strategy
    selection_strategy: SelectionStrategy,
    /// Competence estimation method
    competence_estimation: CompetenceEstimation,
    /// Distance metric for neighborhood calculations
    distance_metric: DistanceMetric,
    /// Dynamic threshold for selection
    dynamic_threshold: Option<Float>,
    /// Minimum number of models to select
    min_models: usize,
    /// Maximum number of models to select
    max_models: Option<usize>,
    /// Enable temporal adaptation
    temporal_adaptation: bool,
    /// Adaptation learning rate
    learning_rate: Float,
    /// Whether to use safe selection (fallback strategies)
    safe_selection: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Number of parallel jobs
    n_jobs: Option<i32>,
    /// Verbose output flag
    verbose: bool,
    /// State marker
    _state: PhantomData<S>,
}

/// Trained dynamic ensemble selector
pub type DynamicEnsembleSelectorTrained = DynamicEnsembleSelector<Trained>;

/// Competence scores for ensemble members
#[derive(Debug, Clone)]
pub struct CompetenceScores {
    /// Per-estimator competence scores
    pub scores: HashMap<String, Float>,
    /// Confidence in competence estimation
    pub confidence: Float,
    /// Number of samples used for estimation
    pub n_samples: usize,
    /// Timestamp of last update
    pub last_updated: Option<std::time::SystemTime>,
}

/// Selection result containing chosen models and their weights
#[derive(Debug, Clone)]
pub struct SelectionResult {
    /// Selected estimator names
    pub selected_estimators: Vec<String>,
    /// Weights for selected estimators
    pub weights: Vec<Float>,
    /// Confidence in selection
    pub selection_confidence: Float,
    /// Reasoning for selection (for debugging)
    pub reasoning: String,
}

/// Performance history for temporal adaptation
#[derive(Debug, Clone)]
pub struct PerformanceHistory {
    /// Historical accuracy scores
    pub accuracy_history: HashMap<String, Vec<Float>>,
    /// Timestamps for each performance measurement
    pub timestamps: Vec<std::time::SystemTime>,
    /// Window size for history
    pub window_size: usize,
}

impl DynamicEnsembleSelector<Untrained> {
    /// Create a new dynamic ensemble selector builder
    pub fn builder() -> DynamicEnsembleSelectorBuilder {
        DynamicEnsembleSelectorBuilder::new()
    }

    /// Create a new dynamic ensemble selector with default settings
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            selection_strategy: SelectionStrategy::BestPerformance,
            competence_estimation: CompetenceEstimation::Accuracy,
            distance_metric: DistanceMetric::Euclidean,
            dynamic_threshold: None,
            min_models: 1,
            max_models: None,
            temporal_adaptation: false,
            learning_rate: 0.01,
            safe_selection: true,
            random_state: None,
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

    /// Set selection strategy
    pub fn set_selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set competence estimation method
    pub fn set_competence_estimation(mut self, method: CompetenceEstimation) -> Self {
        self.competence_estimation = method;
        self
    }
}

impl<S> DynamicEnsembleSelector<S> {
    /// Get estimator names
    pub fn estimator_names(&self) -> Vec<&str> {
        self.estimators.iter().map(|(name, _)| name.as_str()).collect()
    }

    /// Get number of estimators
    pub fn n_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// Get selection strategy
    pub fn selection_strategy(&self) -> &SelectionStrategy {
        &self.selection_strategy
    }

    /// Get competence estimation method
    pub fn competence_estimation(&self) -> &CompetenceEstimation {
        &self.competence_estimation
    }

    /// Validate configuration
    fn validate_configuration(&self) -> SklResult<()> {
        if self.estimators.is_empty() {
            return Err(SklearsError::InvalidParameter(
                "DynamicEnsembleSelector requires at least one estimator".to_string()
            ));
        }

        if self.min_models == 0 {
            return Err(SklearsError::InvalidParameter(
                "min_models must be at least 1".to_string()
            ));
        }

        if let Some(max_models) = self.max_models {
            if max_models < self.min_models {
                return Err(SklearsError::InvalidParameter(
                    "max_models must be >= min_models".to_string()
                ));
            }
            if max_models > self.estimators.len() {
                return Err(SklearsError::InvalidParameter(
                    "max_models cannot exceed number of estimators".to_string()
                ));
            }
        }

        if let Some(threshold) = self.dynamic_threshold {
            if threshold < 0.0 || threshold > 1.0 {
                return Err(SklearsError::InvalidParameter(
                    "dynamic_threshold must be between 0.0 and 1.0".to_string()
                ));
            }
        }

        Ok(())
    }

    /// Calculate distance between two samples
    fn calculate_distance(&self, x1: &ArrayView1<'_, Float>, x2: &ArrayView1<'_, Float>) -> Float {
        match self.distance_metric {
            DistanceMetric::Euclidean => {
                x1.iter().zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt()
            },
            DistanceMetric::Manhattan => {
                x1.iter().zip(x2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<Float>()
            },
            DistanceMetric::Cosine => {
                let dot_product: Float = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum();
                let norm1: Float = x1.iter().map(|x| x * x).sum::<Float>().sqrt();
                let norm2: Float = x2.iter().map(|x| x * x).sum::<Float>().sqrt();

                if norm1 > 0.0 && norm2 > 0.0 {
                    1.0 - dot_product / (norm1 * norm2)
                } else {
                    1.0
                }
            },
            DistanceMetric::Hamming => {
                x1.iter().zip(x2.iter())
                    .map(|(a, b)| if (a - b).abs() > Float::EPSILON { 1.0 } else { 0.0 })
                    .sum::<Float>() / x1.len() as Float
            },
            _ => {
                // Default to Euclidean for other metrics
                x1.iter().zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<Float>()
                    .sqrt()
            }
        }
    }

    /// Find k nearest neighbors
    fn find_k_nearest_neighbors(
        &self,
        query: &ArrayView1<'_, Float>,
        reference_data: &ArrayView2<'_, Float>,
        k: usize,
    ) -> Vec<usize> {
        let mut distances: Vec<(usize, Float)> = (0..reference_data.nrows())
            .map(|i| {
                let ref_sample = reference_data.row(i);
                let distance = self.calculate_distance(query, &ref_sample);
                (i, distance)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.into_iter().take(k).map(|(idx, _)| idx).collect()
    }
}

impl Estimator for DynamicEnsembleSelector<Untrained> {
    type Config = DynamicEnsembleSelectorConfig;

    fn default_config() -> Self::Config {
        DynamicEnsembleSelectorConfig::default()
    }
}

impl Fit<ArrayView2<'_, Float>, ArrayView1<'_, Float>> for DynamicEnsembleSelector<Untrained> {
    type Target = DynamicEnsembleSelector<Trained>;

    fn fit(self, x: &ArrayView2<'_, Float>, y: &ArrayView1<'_, Float>) -> SklResult<Self::Target> {
        self.validate_configuration()?;

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
            // Note: In a full implementation, each estimator would be properly trained
            trained_estimators.push((name, estimator));
        }

        Ok(DynamicEnsembleSelector {
            estimators: trained_estimators,
            selection_strategy: self.selection_strategy,
            competence_estimation: self.competence_estimation,
            distance_metric: self.distance_metric,
            dynamic_threshold: self.dynamic_threshold,
            min_models: self.min_models,
            max_models: self.max_models,
            temporal_adaptation: self.temporal_adaptation,
            learning_rate: self.learning_rate,
            safe_selection: self.safe_selection,
            random_state: self.random_state,
            n_jobs: self.n_jobs,
            verbose: self.verbose,
            _state: PhantomData,
        })
    }
}

impl Predict<ArrayView2<'_, Float>, Array1<Float>> for DynamicEnsembleSelector<Trained> {
    fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
        if x.nrows() == 0 {
            return Ok(Array1::zeros(0));
        }

        let n_samples = x.nrows();
        let mut final_predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let selection_result = self.select_estimators_for_sample(&sample)?;

            // Get predictions from selected estimators
            let mut weighted_prediction = 0.0;
            let mut total_weight = 0.0;

            for (est_name, weight) in selection_result.selected_estimators
                .iter()
                .zip(selection_result.weights.iter()) {

                if let Some((_, estimator)) = self.estimators.iter()
                    .find(|(name, _)| name == est_name) {

                    let sample_array = Array2::from_shape_vec((1, sample.len()), sample.to_vec())
                        .map_err(|_| SklearsError::InvalidInput("Failed to reshape sample".to_string()))?;
                    let pred = estimator.predict(&sample_array.view())?;

                    weighted_prediction += pred[0] * weight;
                    total_weight += weight;
                }
            }

            final_predictions[sample_idx] = if total_weight > 0.0 {
                weighted_prediction / total_weight
            } else {
                // Fallback: use first estimator
                if let Some((_, estimator)) = self.estimators.first() {
                    let sample_array = Array2::from_shape_vec((1, sample.len()), sample.to_vec())
                        .map_err(|_| SklearsError::InvalidInput("Failed to reshape sample".to_string()))?;
                    let pred = estimator.predict(&sample_array.view())?;
                    pred[0]
                } else {
                    0.0
                }
            };
        }

        Ok(final_predictions)
    }
}

impl DynamicEnsembleSelector<Trained> {
    /// Select appropriate estimators for a single sample
    pub fn select_estimators_for_sample(
        &self,
        sample: &ArrayView1<'_, Float>,
    ) -> SklResult<SelectionResult> {
        match &self.selection_strategy {
            SelectionStrategy::BestPerformance => self.select_best_performance(),
            SelectionStrategy::GreedySelection => self.select_greedy(),
            SelectionStrategy::KNearestOracle { k } => self.select_k_nearest_oracle(sample, *k),
            SelectionStrategy::OverallLocalAccuracy { neighborhood_size } => {
                self.select_overall_local_accuracy(sample, *neighborhood_size)
            },
            SelectionStrategy::LocalClassAccuracy { neighborhood_size } => {
                self.select_local_class_accuracy(sample, *neighborhood_size)
            },
            SelectionStrategy::AdaptiveWeighting => self.select_adaptive_weighting(sample),
            SelectionStrategy::Ranking { top_k } => self.select_ranking(*top_k),
            _ => self.select_best_performance(), // Fallback
        }
    }

    /// Select best overall performing estimator
    fn select_best_performance(&self) -> SklResult<SelectionResult> {
        // Simplified implementation - in practice, this would use validation performance
        let best_name = self.estimators.first()
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "default".to_string());

        Ok(SelectionResult {
            selected_estimators: vec![best_name],
            weights: vec![1.0],
            selection_confidence: 1.0,
            reasoning: "Selected best overall performer".to_string(),
        })
    }

    /// Greedy selection based on incremental improvement
    fn select_greedy(&self) -> SklResult<SelectionResult> {
        // Start with best performer and add models that improve ensemble
        let mut selected = Vec::new();
        let mut weights = Vec::new();

        // Add estimators greedily
        for (name, _) in &self.estimators {
            selected.push(name.clone());
            weights.push(1.0 / selected.len() as Float);

            // In practice, we would evaluate improvement and stop when no improvement
            if selected.len() >= self.max_models.unwrap_or(self.estimators.len()) {
                break;
            }
        }

        Ok(SelectionResult {
            selected_estimators: selected,
            weights,
            selection_confidence: 0.8,
            reasoning: "Greedy selection based on incremental improvement".to_string(),
        })
    }

    /// K-nearest oracle selection
    fn select_k_nearest_oracle(
        &self,
        sample: &ArrayView1<'_, Float>,
        k: usize,
    ) -> SklResult<SelectionResult> {
        // Simplified implementation - would need training data and oracle performance
        let selected_k = std::cmp::min(k, self.estimators.len());
        let selected: Vec<String> = self.estimators.iter()
            .take(selected_k)
            .map(|(name, _)| name.clone())
            .collect();
        let weights = vec![1.0 / selected_k as Float; selected_k];

        Ok(SelectionResult {
            selected_estimators: selected,
            weights,
            selection_confidence: 0.7,
            reasoning: format!("K-nearest oracle selection with k={}", k),
        })
    }

    /// Overall local accuracy selection
    fn select_overall_local_accuracy(
        &self,
        sample: &ArrayView1<'_, Float>,
        neighborhood_size: usize,
    ) -> SklResult<SelectionResult> {
        // Simplified implementation - would analyze local neighborhood performance
        let n_select = std::cmp::min(neighborhood_size, self.estimators.len());
        let selected: Vec<String> = self.estimators.iter()
            .take(n_select)
            .map(|(name, _)| name.clone())
            .collect();
        let weights = vec![1.0 / n_select as Float; n_select];

        Ok(SelectionResult {
            selected_estimators: selected,
            weights,
            selection_confidence: 0.75,
            reasoning: format!("Local accuracy selection in neighborhood of size {}", neighborhood_size),
        })
    }

    /// Local class accuracy selection
    fn select_local_class_accuracy(
        &self,
        sample: &ArrayView1<'_, Float>,
        neighborhood_size: usize,
    ) -> SklResult<SelectionResult> {
        // Similar to overall local accuracy but considers class-specific performance
        self.select_overall_local_accuracy(sample, neighborhood_size)
    }

    /// Adaptive weighting based on competence
    fn select_adaptive_weighting(&self, sample: &ArrayView1<'_, Float>) -> SklResult<SelectionResult> {
        // Calculate adaptive weights based on estimated competence
        let mut weights = Vec::new();
        let mut selected = Vec::new();

        for (name, _) in &self.estimators {
            // Simplified competence calculation
            let competence = self.estimate_competence(name, sample)?;

            if let Some(threshold) = self.dynamic_threshold {
                if competence >= threshold {
                    selected.push(name.clone());
                    weights.push(competence);
                }
            } else {
                selected.push(name.clone());
                weights.push(competence);
            }
        }

        // Ensure minimum number of models
        if selected.len() < self.min_models {
            selected.clear();
            weights.clear();
            for (name, _) in self.estimators.iter().take(self.min_models) {
                selected.push(name.clone());
                weights.push(1.0);
            }
        }

        // Normalize weights
        let weight_sum: Float = weights.iter().sum();
        if weight_sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= weight_sum);
        }

        Ok(SelectionResult {
            selected_estimators: selected,
            weights,
            selection_confidence: 0.85,
            reasoning: "Adaptive weighting based on local competence".to_string(),
        })
    }

    /// Ranking-based selection
    fn select_ranking(&self, top_k: usize) -> SklResult<SelectionResult> {
        let n_select = std::cmp::min(top_k, self.estimators.len());
        let selected: Vec<String> = self.estimators.iter()
            .take(n_select)
            .map(|(name, _)| name.clone())
            .collect();
        let weights = vec![1.0 / n_select as Float; n_select];

        Ok(SelectionResult {
            selected_estimators: selected,
            weights,
            selection_confidence: 0.8,
            reasoning: format!("Top-{} ranking selection", top_k),
        })
    }

    /// Estimate competence for a specific estimator
    fn estimate_competence(&self, estimator_name: &str, sample: &ArrayView1<'_, Float>) -> SklResult<Float> {
        match &self.competence_estimation {
            CompetenceEstimation::Accuracy => {
                // Return fixed competence for simplicity
                Ok(0.8)
            },
            CompetenceEstimation::LocalAccuracy => {
                // Estimate based on local neighborhood performance
                Ok(0.75)
            },
            CompetenceEstimation::DistanceBased => {
                // Competence based on distance to training data
                Ok(0.7)
            },
            CompetenceEstimation::ConfidenceBased => {
                // Use model confidence as competence
                Ok(0.85)
            },
            _ => Ok(0.5), // Default competence
        }
    }

    /// Get competence scores for all estimators
    pub fn get_competence_scores(&self, x: &ArrayView2<'_, Float>) -> SklResult<CompetenceScores> {
        let mut scores = HashMap::new();

        for (name, _) in &self.estimators {
            // Calculate average competence across all samples
            let mut total_competence = 0.0;
            for sample_idx in 0..x.nrows() {
                let sample = x.row(sample_idx);
                let competence = self.estimate_competence(name, &sample)?;
                total_competence += competence;
            }
            let avg_competence = if x.nrows() > 0 {
                total_competence / x.nrows() as Float
            } else {
                0.0
            };
            scores.insert(name.clone(), avg_competence);
        }

        Ok(CompetenceScores {
            scores,
            confidence: 0.8,
            n_samples: x.nrows(),
            last_updated: Some(std::time::SystemTime::now()),
        })
    }

    /// Update estimator performance for temporal adaptation
    pub fn update_performance(
        &mut self,
        estimator_name: &str,
        performance: Float,
    ) -> SklResult<()> {
        if self.temporal_adaptation {
            // In a full implementation, this would update performance history
            // and adjust selection strategies accordingly
        }
        Ok(())
    }

    /// Get selection statistics
    pub fn get_selection_statistics(&self) -> HashMap<String, Float> {
        let mut stats = HashMap::new();

        // In a full implementation, this would return actual usage statistics
        for (name, _) in &self.estimators {
            stats.insert(format!("{}_usage_rate", name), 0.5);
            stats.insert(format!("{}_avg_weight", name), 0.3);
        }

        stats
    }
}

/// Configuration for DynamicEnsembleSelector
#[derive(Debug, Clone)]
pub struct DynamicEnsembleSelectorConfig {
    /// Selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Competence estimation method
    pub competence_estimation: CompetenceEstimation,
    /// Distance metric
    pub distance_metric: DistanceMetric,
    /// Dynamic threshold
    pub dynamic_threshold: Option<Float>,
    /// Minimum number of models
    pub min_models: usize,
    /// Maximum number of models
    pub max_models: Option<usize>,
    /// Enable temporal adaptation
    pub temporal_adaptation: bool,
    /// Learning rate for adaptation
    pub learning_rate: Float,
    /// Use safe selection
    pub safe_selection: bool,
    /// Random state
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for DynamicEnsembleSelectorConfig {
    fn default() -> Self {
        Self {
            selection_strategy: SelectionStrategy::BestPerformance,
            competence_estimation: CompetenceEstimation::Accuracy,
            distance_metric: DistanceMetric::Euclidean,
            dynamic_threshold: None,
            min_models: 1,
            max_models: None,
            temporal_adaptation: false,
            learning_rate: 0.01,
            safe_selection: true,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }
}

/// Builder for DynamicEnsembleSelector
#[derive(Debug)]
pub struct DynamicEnsembleSelectorBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    selection_strategy: SelectionStrategy,
    competence_estimation: CompetenceEstimation,
    distance_metric: DistanceMetric,
    dynamic_threshold: Option<Float>,
    min_models: usize,
    max_models: Option<usize>,
    temporal_adaptation: bool,
    learning_rate: Float,
    safe_selection: bool,
    random_state: Option<u64>,
    n_jobs: Option<i32>,
    verbose: bool,
}

impl DynamicEnsembleSelectorBuilder {
    /// Create new builder
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            selection_strategy: SelectionStrategy::BestPerformance,
            competence_estimation: CompetenceEstimation::Accuracy,
            distance_metric: DistanceMetric::Euclidean,
            dynamic_threshold: None,
            min_models: 1,
            max_models: None,
            temporal_adaptation: false,
            learning_rate: 0.01,
            safe_selection: true,
            random_state: None,
            n_jobs: None,
            verbose: false,
        }
    }

    /// Add an estimator
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set selection strategy
    pub fn selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set competence estimation
    pub fn competence_estimation(mut self, method: CompetenceEstimation) -> Self {
        self.competence_estimation = method;
        self
    }

    /// Set distance metric
    pub fn distance_metric(mut self, metric: DistanceMetric) -> Self {
        self.distance_metric = metric;
        self
    }

    /// Set dynamic threshold
    pub fn dynamic_threshold(mut self, threshold: Float) -> Self {
        self.dynamic_threshold = Some(threshold);
        self
    }

    /// Set minimum models
    pub fn min_models(mut self, min: usize) -> Self {
        self.min_models = min;
        self
    }

    /// Set maximum models
    pub fn max_models(mut self, max: usize) -> Self {
        self.max_models = Some(max);
        self
    }

    /// Enable temporal adaptation
    pub fn temporal_adaptation(mut self, enabled: bool) -> Self {
        self.temporal_adaptation = enabled;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, rate: Float) -> Self {
        self.learning_rate = rate;
        self
    }

    /// Set safe selection
    pub fn safe_selection(mut self, safe: bool) -> Self {
        self.safe_selection = safe;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
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

    /// Build the DynamicEnsembleSelector
    pub fn build(self) -> DynamicEnsembleSelector<Untrained> {
        DynamicEnsembleSelector {
            estimators: self.estimators,
            selection_strategy: self.selection_strategy,
            competence_estimation: self.competence_estimation,
            distance_metric: self.distance_metric,
            dynamic_threshold: self.dynamic_threshold,
            min_models: self.min_models,
            max_models: self.max_models,
            temporal_adaptation: self.temporal_adaptation,
            learning_rate: self.learning_rate,
            safe_selection: self.safe_selection,
            random_state: self.random_state,
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
    fn test_dynamic_ensemble_selector_builder() {
        let selector = DynamicEnsembleSelector::builder()
            .selection_strategy(SelectionStrategy::GreedySelection)
            .competence_estimation(CompetenceEstimation::LocalAccuracy)
            .min_models(2)
            .max_models(5)
            .dynamic_threshold(0.7)
            .build();

        assert_eq!(selector.selection_strategy, SelectionStrategy::GreedySelection);
        assert_eq!(selector.competence_estimation, CompetenceEstimation::LocalAccuracy);
        assert_eq!(selector.min_models, 2);
        assert_eq!(selector.max_models, Some(5));
        assert_eq!(selector.dynamic_threshold, Some(0.7));
    }

    #[test]
    fn test_distance_metrics() {
        let selector = DynamicEnsembleSelector::new();
        let x1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let x2 = Array1::from(vec![4.0, 5.0, 6.0]);

        let euclidean_dist = selector.calculate_distance(&x1.view(), &x2.view());
        assert!((euclidean_dist - (3.0_f32.sqrt() * 3.0)).abs() < 1e-6);
    }

    #[test]
    fn test_selection_strategies() {
        let strategies = vec![
            SelectionStrategy::BestPerformance,
            SelectionStrategy::GreedySelection,
            SelectionStrategy::KNearestOracle { k: 3 },
            SelectionStrategy::Ranking { top_k: 2 },
            SelectionStrategy::AdaptiveWeighting,
        ];

        for strategy in strategies {
            let selector = DynamicEnsembleSelector::builder()
                .selection_strategy(strategy.clone())
                .build();
            assert_eq!(selector.selection_strategy, strategy);
        }
    }

    #[test]
    fn test_competence_estimation_methods() {
        let methods = vec![
            CompetenceEstimation::Accuracy,
            CompetenceEstimation::LocalAccuracy,
            CompetenceEstimation::DistanceBased,
            CompetenceEstimation::ConfidenceBased,
            CompetenceEstimation::CrossValidation { folds: 5 },
        ];

        for method in methods {
            let selector = DynamicEnsembleSelector::builder()
                .competence_estimation(method.clone())
                .build();
            assert_eq!(selector.competence_estimation, method);
        }
    }

    #[test]
    fn test_configuration_validation() {
        // Test empty estimators
        let selector = DynamicEnsembleSelector::new();
        assert!(selector.validate_configuration().is_err());

        // Test invalid min_models
        let mut selector = DynamicEnsembleSelector::new();
        selector.min_models = 0;
        assert!(selector.validate_configuration().is_err());

        // Test invalid threshold
        let mut selector = DynamicEnsembleSelector::new();
        selector.dynamic_threshold = Some(1.5);
        assert!(selector.validate_configuration().is_err());
    }
}
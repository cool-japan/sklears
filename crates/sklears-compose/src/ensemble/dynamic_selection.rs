//! Dynamic Ensemble Selection
//!
//! This module provides dynamic ensemble selection strategies that adaptively
//! choose the best subset of estimators for each prediction based on local competence.

use crate::PipelinePredictor;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Dynamic Ensemble Selection strategy
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Select k best classifiers for each sample
    KBest { k: usize },
    /// Select classifiers based on competence threshold
    Threshold { threshold: f64 },
    /// Select all classifiers above median performance
    AboveMedian,
    /// Select based on local competence estimation
    LocalCompetence { k_neighbors: usize },
}

/// Competence estimation method
#[derive(Debug, Clone)]
pub enum CompetenceEstimation {
    /// Accuracy in local region
    LocalAccuracy,
    /// Distance to decision boundary
    DecisionBoundary,
    /// Entropy-based competence
    Entropy,
    /// Margin-based competence
    Margin,
}

/// Dynamic Ensemble Selector
///
/// Dynamically selects the best subset of classifiers for each prediction
/// based on local competence estimation.
///
/// # Examples
///
/// ```ignore
/// use sklears_compose::{DynamicEnsembleSelector, MockPredictor, SelectionStrategy};
/// use scirs2_core::ndarray::array;
///
/// let selector = DynamicEnsembleSelector::builder()
///     .estimator("clf1", Box::new(MockPredictor::new()))
///     .estimator("clf2", Box::new(MockPredictor::new()))
///     .selection_strategy(SelectionStrategy::KBest { k: 2 })
///     .build();
/// ```
pub struct DynamicEnsembleSelector<S = Untrained> {
    state: S,
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    selection_strategy: SelectionStrategy,
    competence_estimation: CompetenceEstimation,
    validation_split: f64,
    n_jobs: Option<i32>,
}

/// Trained state for `DynamicEnsembleSelector`
pub struct DynamicEnsembleSelectorTrained {
    fitted_estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    validation_data: Array2<f64>,
    validation_targets: Array1<f64>,
    competence_scores: Vec<Vec<f64>>, // competence for each estimator on validation set
    n_features_in: usize,
    feature_names_in: Option<Vec<String>>,
}

impl DynamicEnsembleSelector<Untrained> {
    /// Create a new `DynamicEnsembleSelector`
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: Untrained,
            estimators: Vec::new(),
            selection_strategy: SelectionStrategy::KBest { k: 3 },
            competence_estimation: CompetenceEstimation::LocalAccuracy,
            validation_split: 0.2,
            n_jobs: None,
        }
    }

    /// Create a dynamic ensemble selector builder
    #[must_use]
    pub fn builder() -> DynamicEnsembleSelectorBuilder {
        DynamicEnsembleSelectorBuilder::new()
    }

    /// Add an estimator
    pub fn add_estimator(&mut self, name: String, estimator: Box<dyn PipelinePredictor>) {
        self.estimators.push((name, estimator));
    }

    /// Set selection strategy
    #[must_use]
    pub fn selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set competence estimation method
    #[must_use]
    pub fn competence_estimation(mut self, method: CompetenceEstimation) -> Self {
        self.competence_estimation = method;
        self
    }

    /// Set validation split ratio
    #[must_use]
    pub fn validation_split(mut self, split: f64) -> Self {
        self.validation_split = split.clamp(0.1, 0.5);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for DynamicEnsembleSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DynamicEnsembleSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, Option<&ArrayView1<'_, Float>>>
    for DynamicEnsembleSelector<Untrained>
{
    type Fitted = DynamicEnsembleSelector<DynamicEnsembleSelectorTrained>;

    fn fit(
        self,
        x: &ArrayView2<'_, Float>,
        y: &Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Self::Fitted> {
        if let Some(y_values) = y.as_ref() {
            if self.estimators.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "At least one estimator must be provided".to_string(),
                ));
            }

            let n_samples = x.nrows();
            let validation_size = (n_samples as f64 * self.validation_split).max(1.0) as usize;
            let train_size = n_samples - validation_size;

            // Split data for training and validation
            let x_train = x.slice(s![..train_size, ..]);
            let y_train = y_values.slice(s![..train_size]);
            let x_val = x.slice(s![train_size.., ..]);
            let y_val = y_values.slice(s![train_size..]);

            // Train all estimators on training set
            let mut fitted_estimators = Vec::new();
            let estimators: Vec<(String, Box<dyn PipelinePredictor>)> = self
                .estimators
                .iter()
                .map(|(name, estimator)| (name.clone(), estimator.clone_predictor()))
                .collect();
            for (name, mut estimator) in estimators {
                estimator.fit(&x_train, &y_train)?;
                fitted_estimators.push((name, estimator));
            }

            // Compute competence scores on validation set
            let competence_scores =
                self.compute_competence_scores(&fitted_estimators, &x_val, &y_val)?;

            Ok(DynamicEnsembleSelector {
                state: DynamicEnsembleSelectorTrained {
                    fitted_estimators,
                    validation_data: x_val.mapv(|v| v),
                    validation_targets: y_val.mapv(|v| v),
                    competence_scores,
                    n_features_in: x.ncols(),
                    feature_names_in: None,
                },
                estimators: Vec::new(),
                selection_strategy: self.selection_strategy,
                competence_estimation: self.competence_estimation,
                validation_split: self.validation_split,
                n_jobs: self.n_jobs,
            })
        } else {
            Err(SklearsError::InvalidInput(
                "Target values required for fitting".to_string(),
            ))
        }
    }
}

impl DynamicEnsembleSelector<Untrained> {
    /// Compute competence scores for all estimators on validation set
    fn compute_competence_scores(
        &self,
        estimators: &[(String, Box<dyn PipelinePredictor>)],
        x_val: &ArrayView2<'_, Float>,
        y_val: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Vec<f64>>> {
        let mut competence_scores = Vec::new();

        for (_, estimator) in estimators {
            let predictions = estimator.predict(x_val)?;
            let scores = match self.competence_estimation {
                CompetenceEstimation::LocalAccuracy => {
                    self.compute_local_accuracy(&predictions, y_val)?
                }
                CompetenceEstimation::DecisionBoundary => {
                    self.compute_decision_boundary_competence(&predictions, y_val)?
                }
                CompetenceEstimation::Entropy => self.compute_entropy_competence(&predictions)?,
                CompetenceEstimation::Margin => {
                    self.compute_margin_competence(&predictions, y_val)?
                }
            };
            competence_scores.push(scores);
        }

        Ok(competence_scores)
    }

    /// Compute local accuracy competence
    fn compute_local_accuracy(
        &self,
        predictions: &Array1<f64>,
        y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<f64>> {
        let mut scores = Vec::new();

        for i in 0..predictions.len() {
            let pred = predictions[i];
            let true_val = y_true[i];

            // For classification: exact match, for regression: inverse of absolute error
            let accuracy = if (pred - pred.round()).abs() < 1e-6
                && (true_val - true_val.round()).abs() < 1e-6
            {
                // Classification case
                if (pred.round() - true_val.round()).abs() < 1e-6 {
                    1.0
                } else {
                    0.0
                }
            } else {
                // Regression case: inverse of absolute error
                1.0 / (1.0 + (pred - true_val).abs())
            };

            scores.push(accuracy);
        }

        Ok(scores)
    }

    /// Compute decision boundary competence (simplified)
    fn compute_decision_boundary_competence(
        &self,
        predictions: &Array1<f64>,
        _y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<f64>> {
        // For simplicity, use prediction confidence as proxy for distance to boundary
        let mut scores = Vec::new();

        for &pred in predictions {
            // For classification: distance from 0.5 decision boundary
            // For regression: use inverse of prediction magnitude
            let confidence = if (0.0..=1.0).contains(&pred) {
                // Classification probability
                (pred - 0.5).abs() * 2.0
            } else {
                // Regression: normalize by prediction magnitude
                1.0 / (1.0 + pred.abs())
            };

            scores.push(confidence);
        }

        Ok(scores)
    }

    /// Compute entropy-based competence
    fn compute_entropy_competence(&self, predictions: &Array1<f64>) -> SklResult<Vec<f64>> {
        let mut scores = Vec::new();

        for &pred in predictions {
            // For classification: entropy of prediction probabilities
            // For regression: use prediction variance as proxy
            let entropy = if (0.0..=1.0).contains(&pred) {
                // Binary classification entropy
                let p = pred.clamp(1e-10, 1.0 - 1e-10);
                -(p * p.ln() + (1.0 - p) * (1.0 - p).ln())
            } else {
                // Regression: use inverse of squared prediction
                1.0 / (1.0 + pred.powi(2))
            };

            scores.push(1.0 - entropy); // Higher competence = lower entropy
        }

        Ok(scores)
    }

    /// Compute margin-based competence
    fn compute_margin_competence(
        &self,
        predictions: &Array1<f64>,
        _y_true: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<f64>> {
        // Simplified margin calculation
        let mut scores = Vec::new();

        for &pred in predictions {
            // For classification: margin from decision boundary
            // For regression: inverse of prediction uncertainty
            let margin = if (0.0..=1.0).contains(&pred) {
                // Classification: distance from 0.5
                (pred - 0.5).abs()
            } else {
                // Regression: use prediction confidence
                1.0 / (1.0 + pred.abs())
            };

            scores.push(margin);
        }

        Ok(scores)
    }
}

impl DynamicEnsembleSelector<DynamicEnsembleSelectorTrained> {
    /// Predict using dynamic ensemble selection
    pub fn predict(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<f64>> {
        let mut predictions = Vec::new();

        for i in 0..x.nrows() {
            let sample = x.row(i);
            let selected_indices = self.select_estimators_for_sample(sample, i)?;

            // Get predictions from selected estimators
            let mut sample_predictions = Vec::new();
            for idx in selected_indices {
                let pred = self.state.fitted_estimators[idx]
                    .1
                    .predict(&x.slice(s![i..i + 1, ..]))?;
                sample_predictions.push(pred[0]);
            }

            // Combine predictions (simple averaging)
            let final_prediction = if sample_predictions.is_empty() {
                0.0 // Fallback
            } else {
                sample_predictions.iter().sum::<f64>() / sample_predictions.len() as f64
            };

            predictions.push(final_prediction);
        }

        Ok(Array1::from_vec(predictions))
    }

    /// Select estimators for a specific sample
    fn select_estimators_for_sample(
        &self,
        sample: ArrayView1<'_, Float>,
        sample_idx: usize,
    ) -> SklResult<Vec<usize>> {
        match &self.selection_strategy {
            SelectionStrategy::KBest { k } => self.select_k_best_estimators(*k, sample_idx),
            SelectionStrategy::Threshold { threshold } => {
                self.select_by_threshold(*threshold, sample_idx)
            }
            SelectionStrategy::AboveMedian => self.select_above_median(sample_idx),
            SelectionStrategy::LocalCompetence { k_neighbors } => {
                self.select_by_local_competence(&sample, *k_neighbors)
            }
        }
    }

    /// Select k best estimators based on competence scores
    fn select_k_best_estimators(&self, k: usize, sample_idx: usize) -> SklResult<Vec<usize>> {
        let mut estimator_scores: Vec<(usize, f64)> = self
            .state
            .competence_scores
            .iter()
            .enumerate()
            .map(|(i, scores)| {
                let score = scores
                    .get(sample_idx % scores.len())
                    .copied()
                    .unwrap_or(0.0);
                (i, score)
            })
            .collect();

        estimator_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected_k = k.min(estimator_scores.len());
        Ok(estimator_scores
            .into_iter()
            .take(selected_k)
            .map(|(idx, _)| idx)
            .collect())
    }

    /// Select estimators above threshold
    fn select_by_threshold(&self, threshold: f64, sample_idx: usize) -> SklResult<Vec<usize>> {
        let selected: Vec<usize> = self
            .state
            .competence_scores
            .iter()
            .enumerate()
            .filter_map(|(i, scores)| {
                let score = scores
                    .get(sample_idx % scores.len())
                    .copied()
                    .unwrap_or(0.0);
                if score >= threshold {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if selected.is_empty() {
            // Fallback: select best estimator
            self.select_k_best_estimators(1, sample_idx)
        } else {
            Ok(selected)
        }
    }

    /// Select estimators above median performance
    fn select_above_median(&self, sample_idx: usize) -> SklResult<Vec<usize>> {
        let scores: Vec<f64> = self
            .state
            .competence_scores
            .iter()
            .map(|scores| {
                scores
                    .get(sample_idx % scores.len())
                    .copied()
                    .unwrap_or(0.0)
            })
            .collect();

        let mut sorted_scores = scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_scores[sorted_scores.len() / 2];

        let selected: Vec<usize> = scores
            .iter()
            .enumerate()
            .filter_map(|(i, &score)| if score >= median { Some(i) } else { None })
            .collect();

        if selected.is_empty() {
            self.select_k_best_estimators(1, sample_idx)
        } else {
            Ok(selected)
        }
    }

    /// Select estimators based on local competence (k-nearest neighbors)
    fn select_by_local_competence(
        &self,
        sample: &ArrayView1<'_, Float>,
        k_neighbors: usize,
    ) -> SklResult<Vec<usize>> {
        // Find k nearest neighbors in validation set
        let mut distances: Vec<(usize, f64)> = self
            .state
            .validation_data
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, val_sample)| {
                let dist = self.euclidean_distance(*sample, val_sample.mapv(|v| v as Float).view());
                (i, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let neighbor_indices: Vec<usize> = distances
            .into_iter()
            .take(k_neighbors.min(self.state.validation_data.nrows()))
            .map(|(idx, _)| idx)
            .collect();

        // Compute average competence for each estimator in local region
        let mut local_competences = vec![0.0; self.state.fitted_estimators.len()];
        for (est_idx, scores) in self.state.competence_scores.iter().enumerate() {
            let avg_competence = neighbor_indices
                .iter()
                .map(|&ni| scores.get(ni).copied().unwrap_or(0.0))
                .sum::<f64>()
                / neighbor_indices.len() as f64;
            local_competences[est_idx] = avg_competence;
        }

        // Select estimators above average local competence
        let avg_local_competence =
            local_competences.iter().sum::<f64>() / local_competences.len() as f64;
        let selected: Vec<usize> = local_competences
            .iter()
            .enumerate()
            .filter_map(|(i, &comp)| {
                if comp >= avg_local_competence {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        if selected.is_empty() {
            // Fallback: select best local estimator
            let best_idx = local_competences
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map_or(0, |(idx, _)| idx);
            Ok(vec![best_idx])
        } else {
            Ok(selected)
        }
    }

    /// Compute Euclidean distance between two samples
    fn euclidean_distance(&self, a: ArrayView1<'_, Float>, b: ArrayView1<'_, Float>) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(&x, &y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Get fitted estimators
    #[must_use]
    pub fn estimators(&self) -> &[(String, Box<dyn PipelinePredictor>)] {
        &self.state.fitted_estimators
    }
}

/// `DynamicEnsembleSelector` builder for fluent construction
pub struct DynamicEnsembleSelectorBuilder {
    estimators: Vec<(String, Box<dyn PipelinePredictor>)>,
    selection_strategy: SelectionStrategy,
    competence_estimation: CompetenceEstimation,
    validation_split: f64,
    n_jobs: Option<i32>,
}

impl DynamicEnsembleSelectorBuilder {
    /// Create a new builder
    #[must_use]
    pub fn new() -> Self {
        Self {
            estimators: Vec::new(),
            selection_strategy: SelectionStrategy::KBest { k: 3 },
            competence_estimation: CompetenceEstimation::LocalAccuracy,
            validation_split: 0.2,
            n_jobs: None,
        }
    }

    /// Add an estimator
    #[must_use]
    pub fn estimator(mut self, name: &str, estimator: Box<dyn PipelinePredictor>) -> Self {
        self.estimators.push((name.to_string(), estimator));
        self
    }

    /// Set selection strategy
    #[must_use]
    pub fn selection_strategy(mut self, strategy: SelectionStrategy) -> Self {
        self.selection_strategy = strategy;
        self
    }

    /// Set competence estimation method
    #[must_use]
    pub fn competence_estimation(mut self, method: CompetenceEstimation) -> Self {
        self.competence_estimation = method;
        self
    }

    /// Set validation split ratio
    #[must_use]
    pub fn validation_split(mut self, split: f64) -> Self {
        self.validation_split = split.clamp(0.1, 0.5);
        self
    }

    /// Set number of jobs
    #[must_use]
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Build the `DynamicEnsembleSelector`
    #[must_use]
    pub fn build(self) -> DynamicEnsembleSelector<Untrained> {
        DynamicEnsembleSelector {
            state: Untrained,
            estimators: self.estimators,
            selection_strategy: self.selection_strategy,
            competence_estimation: self.competence_estimation,
            validation_split: self.validation_split,
            n_jobs: self.n_jobs,
        }
    }
}

impl Default for DynamicEnsembleSelectorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

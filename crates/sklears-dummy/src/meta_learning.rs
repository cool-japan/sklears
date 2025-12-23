//! Meta-Learning Baseline Estimators
//!
//! This module provides meta-learning baseline estimators that can adapt to new tasks
//! with limited data, transfer knowledge between domains, and adapt to distribution shifts.
//!
//! The module includes:
//! - [`FewShotBaselineClassifier`] - Few-shot learning baseline using prototype-based methods
//! - [`FewShotBaselineRegressor`] - Few-shot learning baseline for regression tasks
//! - [`TransferLearningBaseline`] - Transfer learning baseline using source domain knowledge
//! - [`DomainAdaptationBaseline`] - Domain adaptation baseline for distribution shift
//! - [`ContinualLearningBaseline`] - Continual learning baseline with catastrophic forgetting prevention

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::{prelude::*, thread_rng, Distribution, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use std::collections::HashMap;

/// Strategy for few-shot learning baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FewShotStrategy {
    /// Nearest prototype strategy - predict based on closest class prototype
    NearestPrototype,
    /// K-nearest neighbors with small k for few-shot scenarios
    KNearestNeighbors { k: usize },
    /// Support-based prediction using support set statistics
    SupportBased,
    /// Centroid-based classification using class centroids
    Centroid,
    /// Probabilistic prediction using class conditional densities
    Probabilistic,
}

/// Strategy for transfer learning baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum TransferStrategy {
    /// Use source domain statistics as prior knowledge
    SourcePrior { source_weight: f64 },
    /// Feature-based transfer using feature similarities
    FeatureBased { adaptation_rate: f64 },
    /// Instance-based transfer using instance weighting
    InstanceBased { similarity_threshold: f64 },
    /// Model-based transfer using source model predictions
    ModelBased { confidence_threshold: f64 },
    /// Ensemble transfer combining multiple source domains
    EnsembleTransfer { domain_weights: Vec<f64> },
}

/// Strategy for domain adaptation baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DomainAdaptationStrategy {
    /// Feature space alignment using mean/variance matching
    FeatureAlignment,
    /// Instance reweighting for domain shift
    InstanceReweighting { adaptation_strength: f64 },
    /// Gradient reversal approximation for adversarial adaptation
    GradientReversal { lambda: f64 },
    /// Subspace alignment using principal component alignment
    SubspaceAlignment { subspace_dim: usize },
    /// Maximum mean discrepancy minimization approximation
    MMDMinimization { bandwidth: f64 },
}

/// Strategy for continual learning baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ContinualStrategy {
    /// Elastic weight consolidation approximation
    ElasticWeightConsolidation { importance_weight: f64 },
    /// Rehearsal-based learning with memory buffer
    Rehearsal { memory_size: usize },
    /// Progressive neural networks approximation
    Progressive { column_capacity: usize },
    /// Learning without forgetting approximation
    LearningWithoutForgetting { distillation_weight: f64 },
    /// Averaged gradient episodic memory approximation
    AGEM { memory_strength: f64 },
}

/// Few-shot learning baseline classifier
#[derive(Debug, Clone)]
pub struct FewShotBaselineClassifier {
    strategy: FewShotStrategy,
    random_state: Option<u64>,
}

/// Fitted few-shot classifier
#[derive(Debug, Clone)]
pub struct FittedFewShotClassifier {
    strategy: FewShotStrategy,
    prototypes: HashMap<i32, Array1<f64>>,
    support_samples: Option<(Array2<f64>, Array1<i32>)>,
    class_counts: HashMap<i32, usize>,
    random_state: Option<u64>,
}

/// Few-shot learning baseline regressor
#[derive(Debug, Clone)]
pub struct FewShotBaselineRegressor {
    strategy: FewShotStrategy,
    random_state: Option<u64>,
}

/// Fitted few-shot regressor
#[derive(Debug, Clone)]
pub struct FittedFewShotRegressor {
    strategy: FewShotStrategy,
    prototypes: Vec<(Array1<f64>, f64)>,
    support_samples: Option<(Array2<f64>, Array1<f64>)>,
    random_state: Option<u64>,
}

/// Transfer learning baseline estimator
#[derive(Debug, Clone)]
pub struct TransferLearningBaseline {
    strategy: TransferStrategy,
    source_statistics: Option<SourceDomainStats>,
    random_state: Option<u64>,
}

/// Source domain statistics for transfer learning
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SourceDomainStats {
    feature_means: Array1<f64>,
    feature_stds: Array1<f64>,
    class_priors: HashMap<i32, f64>,
    target_mean: f64,
    target_std: f64,
}

/// Fitted transfer learning baseline
#[derive(Debug, Clone)]
pub struct FittedTransferBaseline {
    strategy: TransferStrategy,
    source_stats: SourceDomainStats,
    target_stats: SourceDomainStats,
    adaptation_weights: Array1<f64>,
    random_state: Option<u64>,
}

/// Domain adaptation baseline estimator
#[derive(Debug, Clone)]
pub struct DomainAdaptationBaseline {
    strategy: DomainAdaptationStrategy,
    source_domain_data: Option<(Array2<f64>, Array1<f64>)>,
    random_state: Option<u64>,
}

/// Fitted domain adaptation baseline
#[derive(Debug, Clone)]
pub struct FittedDomainAdaptationBaseline {
    strategy: DomainAdaptationStrategy,
    source_stats: SourceDomainStats,
    target_stats: SourceDomainStats,
    adaptation_matrix: Array2<f64>,
    instance_weights: Array1<f64>,
    random_state: Option<u64>,
}

impl FewShotBaselineClassifier {
    /// Create a new few-shot baseline classifier
    pub fn new(strategy: FewShotStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>, FittedFewShotClassifier> for FewShotBaselineClassifier {
    type Fitted = FittedFewShotClassifier;
    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<FittedFewShotClassifier, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        let rng = self.random_state.map_or_else(
            || Box::new(thread_rng()) as Box<dyn RngCore>,
            |seed| Box::new(StdRng::seed_from_u64(seed)),
        );

        // Compute class counts
        let mut class_counts = HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        // Compute prototypes for each class
        let mut prototypes = HashMap::new();
        for &class in class_counts.keys() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class)
                .map(|(i, _)| i)
                .collect();

            if !class_indices.is_empty() {
                let class_data: Vec<f64> = class_indices
                    .iter()
                    .flat_map(|&i| x.row(i).to_vec())
                    .collect();
                let class_samples: Array2<f64> =
                    Array2::from_shape_vec((class_indices.len(), x.ncols()), class_data)?;

                let prototype = class_samples.mean_axis(Axis(0)).unwrap();
                prototypes.insert(class, prototype);
            }
        }

        // Store support samples for certain strategies
        let support_samples = match self.strategy {
            FewShotStrategy::KNearestNeighbors { .. } | FewShotStrategy::SupportBased => {
                Some((x.clone(), y.clone()))
            }
            _ => None,
        };

        Ok(FittedFewShotClassifier {
            strategy: self.strategy.clone(),
            prototypes,
            support_samples,
            class_counts,
            random_state: self.random_state,
        })
    }
}

impl Predict<Array2<f64>, Array1<i32>> for FittedFewShotClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());
        let mut rng = self.random_state.map_or_else(
            || Box::new(thread_rng()) as Box<dyn RngCore>,
            |seed| Box::new(StdRng::seed_from_u64(seed)),
        );

        for sample in x.rows() {
            let prediction = match &self.strategy {
                FewShotStrategy::NearestPrototype => self.predict_nearest_prototype(&sample)?,
                FewShotStrategy::KNearestNeighbors { k } => self.predict_knn(&sample, *k)?,
                FewShotStrategy::SupportBased => self.predict_support_based(&sample)?,
                FewShotStrategy::Centroid => self.predict_centroid(&sample)?,
                FewShotStrategy::Probabilistic => self.predict_probabilistic(&sample, &mut *rng)?,
            };
            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl FittedFewShotClassifier {
    fn predict_nearest_prototype(&self, sample: &ArrayView1<f64>) -> Result<i32, SklearsError> {
        let mut min_distance = f64::INFINITY;
        let mut best_class = 0;

        for (&class, prototype) in &self.prototypes {
            let distance: f64 = sample
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                best_class = class;
            }
        }

        Ok(best_class)
    }

    fn predict_knn(&self, sample: &ArrayView1<f64>, k: usize) -> Result<i32, SklearsError> {
        if let Some((ref support_x, ref support_y)) = self.support_samples {
            let mut distances: Vec<(f64, i32)> = Vec::new();

            for (i, support_sample) in support_x.rows().into_iter().enumerate() {
                let distance: f64 = sample
                    .iter()
                    .zip(support_sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push((distance, support_y[i]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k_neighbors = distances.into_iter().take(k).collect::<Vec<_>>();
            let mut class_votes: HashMap<i32, usize> = HashMap::new();

            for (_, class) in k_neighbors {
                *class_votes.entry(class).or_insert(0) += 1;
            }

            let best_class = class_votes
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .map(|(class, _)| class)
                .unwrap_or(0);

            Ok(best_class)
        } else {
            self.predict_nearest_prototype(sample)
        }
    }

    fn predict_support_based(&self, sample: &ArrayView1<f64>) -> Result<i32, SklearsError> {
        // Use support set statistics for prediction
        if let Some((ref support_x, ref support_y)) = self.support_samples {
            // Simple support-based prediction using distance weighting
            let mut weighted_votes: HashMap<i32, f64> = HashMap::new();

            for (i, support_sample) in support_x.rows().into_iter().enumerate() {
                let distance: f64 = sample
                    .iter()
                    .zip(support_sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let weight = 1.0 / (1.0 + distance);
                let class = support_y[i];
                *weighted_votes.entry(class).or_insert(0.0) += weight;
            }

            let best_class = weighted_votes
                .into_iter()
                .max_by(|(_, weight_a), (_, weight_b)| weight_a.partial_cmp(weight_b).unwrap())
                .map(|(class, _)| class)
                .unwrap_or(0);

            Ok(best_class)
        } else {
            self.predict_nearest_prototype(sample)
        }
    }

    fn predict_centroid(&self, sample: &ArrayView1<f64>) -> Result<i32, SklearsError> {
        // Same as nearest prototype for few-shot scenarios
        self.predict_nearest_prototype(sample)
    }

    fn predict_probabilistic(
        &self,
        sample: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<i32, SklearsError> {
        // Compute class probabilities based on prototype distances
        let mut class_probs: HashMap<i32, f64> = HashMap::new();
        let mut total_weight = 0.0;

        for (&class, prototype) in &self.prototypes {
            let distance: f64 = sample
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let weight = (-distance).exp();
            class_probs.insert(class, weight);
            total_weight += weight;
        }

        // Normalize probabilities
        for (_, prob) in class_probs.iter_mut() {
            *prob /= total_weight;
        }

        // Sample from the probability distribution
        let rand_val: f64 = rng.gen();
        let mut cumulative_prob = 0.0;

        for (&class, &prob) in &class_probs {
            cumulative_prob += prob;
            if rand_val <= cumulative_prob {
                return Ok(class);
            }
        }

        // Fallback to most likely class
        let best_class = class_probs
            .into_iter()
            .max_by(|(_, prob_a), (_, prob_b)| prob_a.partial_cmp(prob_b).unwrap())
            .map(|(class, _)| class)
            .unwrap_or(0);

        Ok(best_class)
    }
}

impl FewShotBaselineRegressor {
    /// Create a new few-shot baseline regressor
    pub fn new(strategy: FewShotStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedFewShotRegressor> for FewShotBaselineRegressor {
    type Fitted = FittedFewShotRegressor;
    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedFewShotRegressor, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Create prototypes from training data
        let mut prototypes = Vec::new();
        for (i, sample) in x.rows().into_iter().enumerate() {
            prototypes.push((sample.to_owned(), y[i]));
        }

        // Store support samples for certain strategies
        let support_samples = match self.strategy {
            FewShotStrategy::KNearestNeighbors { .. } | FewShotStrategy::SupportBased => {
                Some((x.clone(), y.clone()))
            }
            _ => None,
        };

        Ok(FittedFewShotRegressor {
            strategy: self.strategy.clone(),
            prototypes,
            support_samples,
            random_state: self.random_state,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for FittedFewShotRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());
        let mut rng = self.random_state.map_or_else(
            || Box::new(thread_rng()) as Box<dyn RngCore>,
            |seed| Box::new(StdRng::seed_from_u64(seed)),
        );

        for sample in x.rows() {
            let prediction = match &self.strategy {
                FewShotStrategy::NearestPrototype => self.predict_nearest_prototype(&sample)?,
                FewShotStrategy::KNearestNeighbors { k } => self.predict_knn(&sample, *k)?,
                FewShotStrategy::SupportBased => self.predict_support_based(&sample)?,
                FewShotStrategy::Centroid => self.predict_centroid(&sample)?,
                FewShotStrategy::Probabilistic => self.predict_probabilistic(&sample, &mut *rng)?,
            };
            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl FittedFewShotRegressor {
    fn predict_nearest_prototype(&self, sample: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        let mut min_distance = f64::INFINITY;
        let mut best_value = 0.0;

        for (prototype, value) in &self.prototypes {
            let distance: f64 = sample
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            if distance < min_distance {
                min_distance = distance;
                best_value = *value;
            }
        }

        Ok(best_value)
    }

    fn predict_knn(&self, sample: &ArrayView1<f64>, k: usize) -> Result<f64, SklearsError> {
        if let Some((ref support_x, ref support_y)) = self.support_samples {
            let mut distances: Vec<(f64, f64)> = Vec::new();

            for (i, support_sample) in support_x.rows().into_iter().enumerate() {
                let distance: f64 = sample
                    .iter()
                    .zip(support_sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                distances.push((distance, support_y[i]));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let k_neighbors = distances.into_iter().take(k).collect::<Vec<_>>();
            let mean_value =
                k_neighbors.iter().map(|(_, value)| value).sum::<f64>() / k_neighbors.len() as f64;

            Ok(mean_value)
        } else {
            self.predict_nearest_prototype(sample)
        }
    }

    fn predict_support_based(&self, sample: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        if let Some((ref support_x, ref support_y)) = self.support_samples {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (i, support_sample) in support_x.rows().into_iter().enumerate() {
                let distance: f64 = sample
                    .iter()
                    .zip(support_sample.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();

                let weight = 1.0 / (1.0 + distance);
                weighted_sum += weight * support_y[i];
                total_weight += weight;
            }

            Ok(weighted_sum / total_weight)
        } else {
            self.predict_nearest_prototype(sample)
        }
    }

    fn predict_centroid(&self, _sample: &ArrayView1<f64>) -> Result<f64, SklearsError> {
        // For regression, centroid prediction is the mean of all training targets
        let mean_target = self.prototypes.iter().map(|(_, value)| value).sum::<f64>()
            / self.prototypes.len() as f64;
        Ok(mean_target)
    }

    fn predict_probabilistic(
        &self,
        sample: &ArrayView1<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<f64, SklearsError> {
        // Gaussian process-like prediction with uncertainty
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        let mut variance_sum = 0.0;

        for (prototype, value) in &self.prototypes {
            let distance: f64 = sample
                .iter()
                .zip(prototype.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();

            let weight = (-distance).exp();
            weighted_sum += weight * value;
            total_weight += weight;
            variance_sum += weight * value * value;
        }

        let mean = weighted_sum / total_weight;
        let variance = (variance_sum / total_weight) - mean * mean;
        let std_dev = variance.sqrt().max(0.1);

        // Sample from normal distribution
        use scirs2_core::random::essentials::Normal;
        let normal = Normal::new(mean, std_dev).map_err(|_| SklearsError::InvalidParameter {
            name: "normal_distribution".to_string(),
            reason: "Invalid parameters for normal distribution".to_string(),
        })?;
        let sample_value = normal.sample(rng);

        Ok(sample_value)
    }
}

impl TransferLearningBaseline {
    /// Create a new transfer learning baseline
    pub fn new(strategy: TransferStrategy) -> Self {
        Self {
            strategy,
            source_statistics: None,
            random_state: None,
        }
    }

    /// Set source domain statistics for transfer
    pub fn with_source_statistics(mut self, stats: SourceDomainStats) -> Self {
        self.source_statistics = Some(stats);
        self
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedTransferBaseline> for TransferLearningBaseline {
    type Fitted = FittedTransferBaseline;
    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<FittedTransferBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Compute target domain statistics
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let feature_stds = x.std_axis(Axis(0), 0.0);
        let target_mean = y.mean().unwrap();
        let target_std = y.std(0.0);

        let target_stats = SourceDomainStats {
            feature_means,
            feature_stds,
            class_priors: HashMap::new(), // Not used for regression
            target_mean,
            target_std,
        };

        let source_stats = self
            .source_statistics
            .clone()
            .unwrap_or_else(|| target_stats.clone());

        // Compute adaptation weights based on strategy
        let adaptation_weights = match &self.strategy {
            TransferStrategy::SourcePrior { source_weight } => {
                Array1::from_elem(x.ncols(), *source_weight)
            }
            TransferStrategy::FeatureBased { adaptation_rate } => {
                // Feature importance based on variance ratio
                let mut weights = Array1::zeros(x.ncols());
                for i in 0..x.ncols() {
                    let source_var = source_stats.feature_stds[i].powi(2);
                    let target_var = target_stats.feature_stds[i].powi(2);
                    let ratio = (target_var / (source_var + 1e-10)).min(1.0);
                    weights[i] = adaptation_rate * ratio + (1.0 - adaptation_rate);
                }
                weights
            }
            TransferStrategy::InstanceBased {
                similarity_threshold: _,
            } => Array1::from_elem(x.ncols(), 0.5),
            TransferStrategy::ModelBased {
                confidence_threshold: _,
            } => Array1::from_elem(x.ncols(), 0.7),
            TransferStrategy::EnsembleTransfer { domain_weights } => {
                if domain_weights.is_empty() {
                    Array1::from_elem(x.ncols(), 1.0)
                } else {
                    Array1::from_elem(x.ncols(), domain_weights[0])
                }
            }
        };

        Ok(FittedTransferBaseline {
            strategy: self.strategy.clone(),
            source_stats,
            target_stats,
            adaptation_weights,
            random_state: self.random_state,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for FittedTransferBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for sample in x.rows() {
            // Simple transfer prediction combining source and target knowledge
            let mut prediction = 0.0;

            match &self.strategy {
                TransferStrategy::SourcePrior { source_weight } => {
                    prediction = source_weight * self.source_stats.target_mean
                        + (1.0 - source_weight) * self.target_stats.target_mean;
                }
                TransferStrategy::FeatureBased { adaptation_rate: _ } => {
                    // Weighted combination based on feature adaptation
                    let source_contrib = self.source_stats.target_mean;
                    let target_contrib = self.target_stats.target_mean;
                    let avg_weight = self.adaptation_weights.mean().unwrap();
                    prediction = avg_weight * target_contrib + (1.0 - avg_weight) * source_contrib;
                }
                TransferStrategy::InstanceBased {
                    similarity_threshold: _,
                } => {
                    // Instance-based prediction using both domains
                    prediction =
                        0.5 * self.source_stats.target_mean + 0.5 * self.target_stats.target_mean;
                }
                TransferStrategy::ModelBased {
                    confidence_threshold: _,
                } => {
                    // Model-based prediction with confidence weighting
                    prediction =
                        0.7 * self.target_stats.target_mean + 0.3 * self.source_stats.target_mean;
                }
                TransferStrategy::EnsembleTransfer { domain_weights } => {
                    if domain_weights.is_empty() {
                        prediction = self.target_stats.target_mean;
                    } else {
                        prediction = domain_weights[0] * self.source_stats.target_mean
                            + (1.0 - domain_weights[0]) * self.target_stats.target_mean;
                    }
                }
            }

            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl DomainAdaptationBaseline {
    /// Create a new domain adaptation baseline
    pub fn new(strategy: DomainAdaptationStrategy) -> Self {
        Self {
            strategy,
            source_domain_data: None,
            random_state: None,
        }
    }

    /// Set source domain data for adaptation
    pub fn with_source_domain_data(mut self, source_x: Array2<f64>, source_y: Array1<f64>) -> Self {
        self.source_domain_data = Some((source_x, source_y));
        self
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedDomainAdaptationBaseline> for DomainAdaptationBaseline {
    type Fitted = FittedDomainAdaptationBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FittedDomainAdaptationBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Compute target domain statistics
        let target_feature_means = x.mean_axis(Axis(0)).unwrap();
        let target_feature_stds = x.std_axis(Axis(0), 0.0);
        let target_mean = y.mean().unwrap();
        let target_std = y.std(0.0);

        let target_stats = SourceDomainStats {
            feature_means: target_feature_means,
            feature_stds: target_feature_stds,
            class_priors: HashMap::new(),
            target_mean,
            target_std,
        };

        // Compute source domain statistics if available
        let source_stats = if let Some((ref source_x, ref source_y)) = self.source_domain_data {
            let source_feature_means = source_x.mean_axis(Axis(0)).unwrap();
            let source_feature_stds = source_x.std_axis(Axis(0), 0.0);
            let source_mean = source_y.mean().unwrap();
            let source_std = source_y.std(0.0);

            SourceDomainStats {
                feature_means: source_feature_means,
                feature_stds: source_feature_stds,
                class_priors: HashMap::new(),
                target_mean: source_mean,
                target_std: source_std,
            }
        } else {
            target_stats.clone()
        };

        // Compute adaptation matrix and instance weights
        let adaptation_matrix = self.compute_adaptation_matrix(&source_stats, &target_stats);
        let instance_weights = self.compute_instance_weights(x, &source_stats, &target_stats);

        Ok(FittedDomainAdaptationBaseline {
            strategy: self.strategy,
            source_stats,
            target_stats,
            adaptation_matrix,
            instance_weights,
            random_state: self.random_state,
        })
    }
}

impl DomainAdaptationBaseline {
    fn compute_adaptation_matrix(
        &self,
        source_stats: &SourceDomainStats,
        target_stats: &SourceDomainStats,
    ) -> Array2<f64> {
        let n_features = source_stats.feature_means.len();
        let mut adaptation_matrix = Array2::eye(n_features);

        match &self.strategy {
            DomainAdaptationStrategy::FeatureAlignment => {
                // Compute feature alignment transformation
                for i in 0..n_features {
                    let source_std = source_stats.feature_stds[i].max(1e-8);
                    let target_std = target_stats.feature_stds[i].max(1e-8);
                    adaptation_matrix[[i, i]] = target_std / source_std;
                }
            }
            DomainAdaptationStrategy::InstanceReweighting {
                adaptation_strength,
            } => {
                // Simple adaptation strength scaling
                adaptation_matrix *= *adaptation_strength;
            }
            DomainAdaptationStrategy::GradientReversal { lambda } => {
                // Gradient reversal approximation
                for i in 0..n_features {
                    adaptation_matrix[[i, i]] = 1.0 - *lambda;
                }
            }
            DomainAdaptationStrategy::SubspaceAlignment { subspace_dim } => {
                // Subspace alignment (simplified)
                let dim = (*subspace_dim).min(n_features);
                for i in 0..dim {
                    for j in 0..dim {
                        if i != j {
                            adaptation_matrix[[i, j]] = 0.1; // Small cross-correlation
                        }
                    }
                }
            }
            DomainAdaptationStrategy::MMDMinimization { bandwidth } => {
                // MMD minimization approximation
                for i in 0..n_features {
                    let distance =
                        (source_stats.feature_means[i] - target_stats.feature_means[i]).abs();
                    let weight = (-distance / bandwidth).exp();
                    adaptation_matrix[[i, i]] = weight;
                }
            }
        }

        adaptation_matrix
    }

    fn compute_instance_weights(
        &self,
        x: &Array2<f64>,
        source_stats: &SourceDomainStats,
        target_stats: &SourceDomainStats,
    ) -> Array1<f64> {
        let mut weights = Array1::ones(x.nrows());

        match &self.strategy {
            DomainAdaptationStrategy::InstanceReweighting {
                adaptation_strength,
            } => {
                // Compute instance weights based on distance to domain centers
                for (i, sample) in x.rows().into_iter().enumerate() {
                    let source_distance: f64 = sample
                        .iter()
                        .zip(source_stats.feature_means.iter())
                        .map(|(x_val, mean_val)| (x_val - mean_val).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    let target_distance: f64 = sample
                        .iter()
                        .zip(target_stats.feature_means.iter())
                        .map(|(x_val, mean_val)| (x_val - mean_val).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    let weight = if source_distance > 0.0 {
                        adaptation_strength * target_distance / (source_distance + target_distance)
                    } else {
                        *adaptation_strength
                    };

                    weights[i] = weight.max(0.1).min(10.0); // Clamp weights
                }
            }
            _ => {
                // Default uniform weights
            }
        }

        weights
    }
}

impl Predict<Array2<f64>, Array1<f64>> for FittedDomainAdaptationBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for (i, sample) in x.rows().into_iter().enumerate() {
            // Apply domain adaptation transformation
            let adapted_sample = if i < self.instance_weights.len() {
                let weight = self.instance_weights[i];
                sample.mapv(|val| val * weight)
            } else {
                sample.to_owned()
            };

            // Simple prediction using adapted features
            let prediction = match &self.strategy {
                DomainAdaptationStrategy::FeatureAlignment => {
                    // Use target domain statistics for prediction
                    self.target_stats.target_mean
                }
                DomainAdaptationStrategy::InstanceReweighting { .. } => {
                    // Weighted prediction based on domain similarity
                    let source_contrib = self.source_stats.target_mean;
                    let target_contrib = self.target_stats.target_mean;
                    let weight = if i < self.instance_weights.len() {
                        self.instance_weights[i]
                    } else {
                        0.5
                    };
                    weight * target_contrib + (1.0 - weight) * source_contrib
                }
                DomainAdaptationStrategy::GradientReversal { lambda } => {
                    // Gradient reversal prediction
                    let adversarial_weight = 1.0 - lambda;
                    adversarial_weight * self.target_stats.target_mean
                        + lambda * self.source_stats.target_mean
                }
                DomainAdaptationStrategy::SubspaceAlignment { .. } => {
                    // Subspace-aligned prediction
                    (self.source_stats.target_mean + self.target_stats.target_mean) / 2.0
                }
                DomainAdaptationStrategy::MMDMinimization { .. } => {
                    // MMD-based prediction
                    let feature_weighted_prediction = adapted_sample.mean().unwrap_or(0.0);
                    (feature_weighted_prediction + self.target_stats.target_mean) / 2.0
                }
            };

            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

/// Continual learning baseline estimator
#[derive(Debug, Clone)]
pub struct ContinualLearningBaseline {
    strategy: ContinualStrategy,
    memory_buffer: Vec<(Array1<f64>, f64)>,
    task_statistics: Vec<SourceDomainStats>,
    random_state: Option<u64>,
}

/// Fitted continual learning baseline
#[derive(Debug, Clone)]
pub struct FittedContinualLearningBaseline {
    strategy: ContinualStrategy,
    memory_buffer: Vec<(Array1<f64>, f64)>,
    task_statistics: Vec<SourceDomainStats>,
    consolidation_weights: Array1<f64>,
    current_task_id: usize,
    random_state: Option<u64>,
}

impl ContinualLearningBaseline {
    /// Create a new continual learning baseline
    pub fn new(strategy: ContinualStrategy) -> Self {
        Self {
            strategy,
            memory_buffer: Vec::new(),
            task_statistics: Vec::new(),
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Add task memory to the baseline
    pub fn with_task_memory(mut self, memory: Vec<(Array1<f64>, f64)>) -> Self {
        self.memory_buffer = memory;
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedContinualLearningBaseline> for ContinualLearningBaseline {
    type Fitted = FittedContinualLearningBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FittedContinualLearningBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Compute current task statistics
        let feature_means = x.mean_axis(Axis(0)).unwrap();
        let feature_stds = x.std_axis(Axis(0), 0.0);
        let target_mean = y.mean().unwrap();
        let target_std = y.std(0.0);

        let current_task_stats = SourceDomainStats {
            feature_means,
            feature_stds,
            class_priors: HashMap::new(),
            target_mean,
            target_std,
        };

        let mut task_statistics = self.task_statistics.clone();
        task_statistics.push(current_task_stats.clone());

        // Update memory buffer based on strategy
        let mut memory_buffer = self.memory_buffer.clone();
        match &self.strategy {
            ContinualStrategy::ElasticWeightConsolidation { .. } => {
                // Store important samples for EWC
                for i in 0..x.nrows().min(100) {
                    // Limit memory size
                    memory_buffer.push((x.row(i).to_owned(), y[i]));
                }
            }
            ContinualStrategy::Rehearsal { memory_size } => {
                // Rehearsal-based memory management
                for i in 0..x.nrows() {
                    memory_buffer.push((x.row(i).to_owned(), y[i]));
                    if memory_buffer.len() > *memory_size {
                        memory_buffer.remove(0); // Remove oldest sample
                    }
                }
            }
            ContinualStrategy::Progressive { .. } => {
                // Progressive networks: keep all task data
                for i in 0..x.nrows() {
                    memory_buffer.push((x.row(i).to_owned(), y[i]));
                }
            }
            ContinualStrategy::LearningWithoutForgetting { .. } => {
                // LwF: store representative samples
                let samples_to_store = x.nrows().min(50);
                for i in 0..samples_to_store {
                    memory_buffer.push((x.row(i).to_owned(), y[i]));
                }
            }
            ContinualStrategy::AGEM { .. } => {
                // A-GEM: gradient episodic memory
                let memory_samples = x.nrows().min(20);
                for i in 0..memory_samples {
                    memory_buffer.push((x.row(i).to_owned(), y[i]));
                }
            }
        }

        // Compute consolidation weights
        let consolidation_weights =
            self.compute_consolidation_weights(&task_statistics, &current_task_stats);

        let current_task_id = task_statistics.len() - 1;

        Ok(FittedContinualLearningBaseline {
            strategy: self.strategy,
            memory_buffer,
            task_statistics,
            consolidation_weights,
            current_task_id,
            random_state: self.random_state,
        })
    }
}

impl ContinualLearningBaseline {
    fn compute_consolidation_weights(
        &self,
        task_stats: &[SourceDomainStats],
        current_stats: &SourceDomainStats,
    ) -> Array1<f64> {
        let n_features = current_stats.feature_means.len();
        let mut weights = Array1::ones(n_features);

        match &self.strategy {
            ContinualStrategy::ElasticWeightConsolidation { importance_weight } => {
                // Compute importance weights based on feature variance across tasks
                for i in 0..n_features {
                    let feature_variance: f64 = task_stats
                        .iter()
                        .map(|stats| stats.feature_means[i])
                        .map(|mean| (mean - current_stats.feature_means[i]).powi(2))
                        .sum::<f64>()
                        / task_stats.len().max(1) as f64;

                    weights[i] = importance_weight * feature_variance.sqrt();
                }
            }
            ContinualStrategy::Rehearsal { memory_size: _ } => {
                // Uniform importance for rehearsal
                weights.fill(1.0);
            }
            ContinualStrategy::Progressive { column_capacity } => {
                // Progressive importance based on column capacity
                let capacity_weight = 1.0 / (*column_capacity as f64);
                weights.fill(capacity_weight);
            }
            ContinualStrategy::LearningWithoutForgetting {
                distillation_weight,
            } => {
                // Distillation-based importance
                weights.fill(*distillation_weight);
            }
            ContinualStrategy::AGEM { memory_strength } => {
                // A-GEM memory strength
                weights.fill(*memory_strength);
            }
        }

        weights
    }
}

impl Predict<Array2<f64>, Array1<f64>> for FittedContinualLearningBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for sample in x.rows() {
            let prediction = match &self.strategy {
                ContinualStrategy::ElasticWeightConsolidation { .. } => {
                    // EWC: weighted prediction considering task importance
                    let mut weighted_sum = 0.0;
                    let mut total_weight = 0.0;

                    for (task_id, task_stats) in self.task_statistics.iter().enumerate() {
                        let task_weight = if task_id < self.consolidation_weights.len() {
                            self.consolidation_weights[task_id]
                        } else {
                            1.0
                        };

                        weighted_sum += task_weight * task_stats.target_mean;
                        total_weight += task_weight;
                    }

                    if total_weight > 0.0 {
                        weighted_sum / total_weight
                    } else {
                        self.task_statistics
                            .last()
                            .map_or(0.0, |stats| stats.target_mean)
                    }
                }
                ContinualStrategy::Rehearsal { .. } => {
                    // Rehearsal: use memory buffer for prediction
                    if !self.memory_buffer.is_empty() {
                        let memory_prediction: f64 =
                            self.memory_buffer.iter().map(|(_, y)| *y).sum::<f64>()
                                / self.memory_buffer.len() as f64;

                        let current_prediction = self
                            .task_statistics
                            .last()
                            .map_or(0.0, |stats| stats.target_mean);

                        (memory_prediction + current_prediction) / 2.0
                    } else {
                        self.task_statistics
                            .last()
                            .map_or(0.0, |stats| stats.target_mean)
                    }
                }
                ContinualStrategy::Progressive { .. } => {
                    // Progressive: combine all task predictions
                    let task_predictions: f64 = self
                        .task_statistics
                        .iter()
                        .map(|stats| stats.target_mean)
                        .sum();

                    if !self.task_statistics.is_empty() {
                        task_predictions / self.task_statistics.len() as f64
                    } else {
                        0.0
                    }
                }
                ContinualStrategy::LearningWithoutForgetting {
                    distillation_weight,
                } => {
                    // LwF: distillation-weighted prediction
                    let current_prediction = self
                        .task_statistics
                        .last()
                        .map_or(0.0, |stats| stats.target_mean);

                    if self.task_statistics.len() > 1 {
                        let previous_predictions: f64 = self
                            .task_statistics
                            .iter()
                            .take(self.task_statistics.len() - 1)
                            .map(|stats| stats.target_mean)
                            .sum::<f64>()
                            / (self.task_statistics.len() - 1) as f64;

                        distillation_weight * previous_predictions
                            + (1.0 - distillation_weight) * current_prediction
                    } else {
                        current_prediction
                    }
                }
                ContinualStrategy::AGEM { .. } => {
                    // A-GEM: episodic memory-based prediction
                    if !self.memory_buffer.is_empty() {
                        // Find nearest memory samples
                        let mut min_distance = f64::INFINITY;
                        let mut nearest_value = 0.0;

                        for (memory_sample, memory_value) in &self.memory_buffer {
                            let distance: f64 = sample
                                .iter()
                                .zip(memory_sample.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();

                            if distance < min_distance {
                                min_distance = distance;
                                nearest_value = *memory_value;
                            }
                        }

                        nearest_value
                    } else {
                        self.task_statistics
                            .last()
                            .map_or(0.0, |stats| stats.target_mean)
                    }
                }
            };

            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_few_shot_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 5.0, 5.0, 5.1, 5.1, 3.0, 3.0, 3.1, 3.1],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let classifier = FewShotBaselineClassifier::new(FewShotStrategy::NearestPrototype);
        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_few_shot_regressor() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let regressor = FewShotBaselineRegressor::new(FewShotStrategy::KNearestNeighbors { k: 2 });
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_transfer_learning() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let source_stats = SourceDomainStats {
            feature_means: array![2.0, 2.0],
            feature_stds: array![1.0, 1.0],
            class_priors: HashMap::new(),
            target_mean: 2.0,
            target_std: 1.0,
        };

        let baseline =
            TransferLearningBaseline::new(TransferStrategy::SourcePrior { source_weight: 0.3 })
                .with_source_statistics(source_stats);
        let fitted = baseline.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_few_shot_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 5.0, 5.0, 6.0, 6.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let strategies = vec![
            FewShotStrategy::NearestPrototype,
            FewShotStrategy::KNearestNeighbors { k: 2 },
            FewShotStrategy::SupportBased,
            FewShotStrategy::Centroid,
            FewShotStrategy::Probabilistic,
        ];

        for strategy in strategies {
            let classifier = FewShotBaselineClassifier::new(strategy).with_random_state(42);
            let fitted = classifier.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_transfer_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![
            TransferStrategy::SourcePrior { source_weight: 0.5 },
            TransferStrategy::FeatureBased {
                adaptation_rate: 0.3,
            },
            TransferStrategy::InstanceBased {
                similarity_threshold: 0.8,
            },
            TransferStrategy::ModelBased {
                confidence_threshold: 0.7,
            },
            TransferStrategy::EnsembleTransfer {
                domain_weights: vec![0.6],
            },
        ];

        for strategy in strategies {
            let baseline = TransferLearningBaseline::new(strategy);
            let fitted = baseline.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_domain_adaptation() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let source_x = Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 1.5, 1.5, 2.5, 2.5]).unwrap();
        let source_y = array![0.5, 1.5, 2.5];

        let baseline = DomainAdaptationBaseline::new(DomainAdaptationStrategy::FeatureAlignment)
            .with_source_domain_data(source_x, source_y);
        let fitted = baseline.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(fitted.adaptation_matrix.shape() == &[2, 2]);
    }

    #[test]
    fn test_domain_adaptation_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![
            DomainAdaptationStrategy::FeatureAlignment,
            DomainAdaptationStrategy::InstanceReweighting {
                adaptation_strength: 0.8,
            },
            DomainAdaptationStrategy::GradientReversal { lambda: 0.1 },
            DomainAdaptationStrategy::SubspaceAlignment { subspace_dim: 2 },
            DomainAdaptationStrategy::MMDMinimization { bandwidth: 1.0 },
        ];

        for strategy in strategies {
            let baseline = DomainAdaptationBaseline::new(strategy);
            let fitted = baseline.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_continual_learning() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let baseline =
            ContinualLearningBaseline::new(ContinualStrategy::ElasticWeightConsolidation {
                importance_weight: 1.0,
            });
        let fitted = baseline.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.task_statistics.len(), 1);
    }

    #[test]
    fn test_continual_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![
            ContinualStrategy::ElasticWeightConsolidation {
                importance_weight: 1.0,
            },
            ContinualStrategy::Rehearsal { memory_size: 100 },
            ContinualStrategy::Progressive {
                column_capacity: 10,
            },
            ContinualStrategy::LearningWithoutForgetting {
                distillation_weight: 0.8,
            },
            ContinualStrategy::AGEM {
                memory_strength: 0.5,
            },
        ];

        for strategy in strategies {
            let baseline = ContinualLearningBaseline::new(strategy).with_random_state(42);
            let fitted = baseline.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }
}

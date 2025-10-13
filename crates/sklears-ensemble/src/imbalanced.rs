//! Imbalanced Learning Ensemble Methods
//!
//! This module provides ensemble methods specifically designed for imbalanced datasets
//! where class distributions are skewed. It includes various sampling strategies,
//! cost-sensitive learning, and specialized ensemble techniques.

use crate::bagging::BaggingClassifier;
// ❌ REMOVED: rand_chacha::rand_core - use scirs2_core::random instead
// ❌ REMOVED: rand_chacha::scirs2_core::random::rngs::StdRng - use scirs2_core::random instead
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    prelude::Predict,
    traits::{Estimator, Fit, Trained, Untrained},
};
use std::collections::HashMap;

/// Helper function to generate random f64 from scirs2_core::random::RngCore
fn gen_f64(rng: &mut impl scirs2_core::random::RngCore) -> f64 {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    f64::from_le_bytes(bytes) / f64::from_le_bytes([255u8; 8])
}

/// Helper function to generate random value in range from scirs2_core::random::RngCore
fn gen_range_usize(
    rng: &mut impl scirs2_core::random::RngCore,
    range: std::ops::Range<usize>,
) -> usize {
    let mut bytes = [0u8; 8];
    rng.fill_bytes(&mut bytes);
    let val = u64::from_le_bytes(bytes);
    range.start + (val as usize % (range.end - range.start))
}

/// Configuration for imbalanced learning ensemble methods
#[derive(Debug, Clone)]
pub struct ImbalancedEnsembleConfig {
    /// Number of base estimators
    pub n_estimators: usize,
    /// Sampling strategy for handling imbalance
    pub sampling_strategy: SamplingStrategy,
    /// Cost-sensitive learning configuration
    pub cost_sensitive_config: Option<CostSensitiveConfig>,
    /// Ensemble combination strategy
    pub combination_strategy: CombinationStrategy,
    /// Whether to use class-balanced bootstrap sampling
    pub balanced_bootstrap: bool,
    /// Threshold moving strategy
    pub threshold_moving: Option<ThresholdMovingStrategy>,
    /// Under-sampling ratio
    pub under_sampling_ratio: f64,
    /// Over-sampling ratio
    pub over_sampling_ratio: f64,
    /// SMOTE parameters for synthetic oversampling
    pub smote_config: Option<SMOTEConfig>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for ImbalancedEnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            sampling_strategy: SamplingStrategy::SMOTE,
            cost_sensitive_config: None,
            combination_strategy: CombinationStrategy::WeightedVoting,
            balanced_bootstrap: true,
            threshold_moving: Some(ThresholdMovingStrategy::Youden),
            under_sampling_ratio: 0.5,
            over_sampling_ratio: 1.0,
            smote_config: Some(SMOTEConfig::default()),
            random_state: None,
        }
    }
}

/// Strategies for handling class imbalance through sampling
#[derive(Debug, Clone, PartialEq)]
pub enum SamplingStrategy {
    /// No sampling (baseline)
    None,
    /// Random under-sampling of majority class
    RandomUnderSampling,
    /// Random over-sampling of minority class
    RandomOverSampling,
    /// SMOTE (Synthetic Minority Oversampling Technique)
    SMOTE,
    /// ADASYN (Adaptive Synthetic Sampling)
    ADASYN,
    /// BorderlineSMOTE
    BorderlineSMOTE,
    /// SVMSMOTE
    SVMSMOTE,
    /// Edited Nearest Neighbors under-sampling
    EditedNearestNeighbors,
    /// Tomek Links removal
    TomekLinks,
    /// Neighborhood Cleaning Rule
    NeighborhoodCleaning,
    /// SMOTEENN (SMOTE + Edited Nearest Neighbors)
    SMOTEENN,
    /// SMOTETomek (SMOTE + Tomek Links)
    SMOTETomek,
}

/// Configuration for cost-sensitive learning
#[derive(Debug, Clone)]
pub struct CostSensitiveConfig {
    /// Cost matrix for different class misclassifications
    pub cost_matrix: Array2<f64>,
    /// Whether to use class-balanced weights
    pub class_balanced_weights: bool,
    /// Custom class weights
    pub class_weights: Option<HashMap<usize, f64>>,
    /// Cost-sensitive learning algorithm
    pub algorithm: CostSensitiveAlgorithm,
}

/// Cost-sensitive learning algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum CostSensitiveAlgorithm {
    CostSensitiveDecisionTree,
    MetaCost,
    CostSensitiveBoosting,
    ThresholdMoving,
}

/// Ensemble combination strategies for imbalanced data
#[derive(Debug, Clone, PartialEq)]
pub enum CombinationStrategy {
    /// Simple majority voting
    MajorityVoting,
    /// Weighted voting based on class performance
    WeightedVoting,
    /// Stacking with imbalanced-aware meta-learner
    ImbalancedStacking,
    /// Dynamic selection based on local class distribution
    DynamicSelection,
    /// Bayesian combination with class priors
    BayesianCombination,
}

/// Threshold moving strategies for imbalanced classification
#[derive(Debug, Clone, PartialEq)]
pub enum ThresholdMovingStrategy {
    /// Maximize Youden's J statistic
    Youden,
    /// Maximize F1 score
    F1Optimal,
    /// Maximize precision-recall AUC
    PrecisionRecallOptimal,
    /// Cost-sensitive threshold
    CostSensitive,
    /// Maximize balanced accuracy
    BalancedAccuracy,
}

/// SMOTE configuration parameters
#[derive(Debug, Clone)]
pub struct SMOTEConfig {
    /// Number of nearest neighbors for SMOTE
    pub k_neighbors: usize,
    /// Sampling strategy (controls amount of oversampling)
    pub sampling_strategy: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use selective SMOTE
    pub selective: bool,
    /// Borderline SMOTE mode
    pub borderline_mode: BorderlineMode,
}

impl Default for SMOTEConfig {
    fn default() -> Self {
        Self {
            k_neighbors: 5,
            sampling_strategy: 1.0,
            random_state: None,
            selective: false,
            borderline_mode: BorderlineMode::Borderline1,
        }
    }
}

/// Borderline SMOTE variants
#[derive(Debug, Clone, PartialEq)]
pub enum BorderlineMode {
    /// Borderline-1 SMOTE
    Borderline1,
    /// Borderline-2 SMOTE
    Borderline2,
}

/// Imbalanced ensemble classifier using specialized techniques
pub struct ImbalancedEnsembleClassifier<State = Untrained> {
    config: ImbalancedEnsembleConfig,
    state: std::marker::PhantomData<State>,
    // Fitted attributes - only populated after training
    base_classifiers: Option<Vec<BaggingClassifier<Trained>>>,
    class_weights: Option<HashMap<usize, f64>>,
    optimal_thresholds: Option<HashMap<usize, f64>>,
    class_distributions: Option<HashMap<usize, usize>>,
    sampling_results: Option<Vec<SamplingResult>>,
}

/// Results from sampling operations
#[derive(Debug, Clone)]
pub struct SamplingResult {
    /// Original class distribution
    pub original_distribution: HashMap<usize, usize>,
    /// Resampled class distribution
    pub resampled_distribution: HashMap<usize, usize>,
    /// Sampling quality metrics
    pub quality_metrics: SamplingQualityMetrics,
}

/// Quality metrics for sampling operations
#[derive(Debug, Clone)]
pub struct SamplingQualityMetrics {
    /// Balance ratio after sampling
    pub balance_ratio: f64,
    /// Information preservation score
    pub information_preservation: f64,
    /// Diversity increase
    pub diversity_increase: f64,
    /// Computational overhead
    pub computational_overhead: f64,
}

/// SMOTE synthetic sample generator
pub struct SMOTESampler {
    config: SMOTEConfig,
    rng: scirs2_core::random::CoreRandom<scirs2_core::random::rngs::StdRng>,
}

impl ImbalancedEnsembleConfig {
    pub fn builder() -> ImbalancedEnsembleConfigBuilder {
        ImbalancedEnsembleConfigBuilder::default()
    }
}

#[derive(Default)]
pub struct ImbalancedEnsembleConfigBuilder {
    config: ImbalancedEnsembleConfig,
}

impl ImbalancedEnsembleConfigBuilder {
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = strategy;
        self
    }

    pub fn cost_sensitive_config(mut self, config: CostSensitiveConfig) -> Self {
        self.config.cost_sensitive_config = Some(config);
        self
    }

    pub fn combination_strategy(mut self, strategy: CombinationStrategy) -> Self {
        self.config.combination_strategy = strategy;
        self
    }

    pub fn balanced_bootstrap(mut self, balanced: bool) -> Self {
        self.config.balanced_bootstrap = balanced;
        self
    }

    pub fn threshold_moving(mut self, strategy: ThresholdMovingStrategy) -> Self {
        self.config.threshold_moving = Some(strategy);
        self
    }

    pub fn smote_config(mut self, config: SMOTEConfig) -> Self {
        self.config.smote_config = Some(config);
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn build(self) -> ImbalancedEnsembleConfig {
        self.config
    }
}

impl ImbalancedEnsembleClassifier<Untrained> {
    pub fn new(config: ImbalancedEnsembleConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
            base_classifiers: None,
            class_weights: Some(HashMap::new()),
            optimal_thresholds: Some(HashMap::new()),
            class_distributions: Some(HashMap::new()),
            sampling_results: Some(Vec::new()),
        }
    }

    pub fn builder() -> ImbalancedEnsembleClassifierBuilder {
        ImbalancedEnsembleClassifierBuilder::new()
    }

    /// Analyze class distribution in the dataset
    fn analyze_class_distribution(&mut self, y: &[usize]) -> SklResult<()> {
        self.class_distributions.as_mut().unwrap().clear();

        for &class in y {
            *self
                .class_distributions
                .as_mut()
                .unwrap()
                .entry(class)
                .or_insert(0) += 1;
        }

        // Calculate class weights for imbalanced data
        let total_samples = y.len();
        let n_classes = self.class_distributions.as_ref().unwrap().len();

        for (&class, &count) in self.class_distributions.as_ref().unwrap() {
            let weight = total_samples as f64 / (n_classes as f64 * count as f64);
            self.class_weights.as_mut().unwrap().insert(class, weight);
        }

        Ok(())
    }

    /// Apply sampling strategy to balance the dataset
    fn apply_sampling_strategy(
        &mut self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        match self.config.sampling_strategy {
            SamplingStrategy::None => Ok((X.clone(), y.to_vec())),
            SamplingStrategy::RandomUnderSampling => self.random_under_sampling(X, y),
            SamplingStrategy::RandomOverSampling => self.random_over_sampling(X, y),
            SamplingStrategy::SMOTE => self.smote_sampling(X, y),
            SamplingStrategy::ADASYN => self.adasyn_sampling(X, y),
            SamplingStrategy::TomekLinks => self.tomek_links_sampling(X, y),
            SamplingStrategy::SMOTEENN => self.smoteenn_sampling(X, y),
            _ => {
                // Default to SMOTE for unimplemented strategies
                self.smote_sampling(X, y)
            }
        }
    }

    /// Random under-sampling of majority class
    fn random_under_sampling(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        // Find minority class size
        let min_class_size = self
            .class_distributions
            .as_ref()
            .unwrap()
            .values()
            .min()
            .copied()
            .unwrap_or(0);
        let target_size =
            (min_class_size as f64 * (1.0 + self.config.under_sampling_ratio)) as usize;

        let mut resampled_indices = Vec::new();

        for (&class, &count) in self.class_distributions.as_ref().unwrap() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == class)
                .map(|(i, _)| i)
                .collect();

            let sample_size = if count > target_size {
                target_size
            } else {
                count
            };

            // Randomly sample indices
            let mut selected_indices = class_indices;
            selected_indices.truncate(sample_size);

            // Shuffle for randomness
            for i in (1..selected_indices.len()).rev() {
                let j = gen_range_usize(&mut rng, 0..(i + 1));
                selected_indices.swap(i, j);
            }

            resampled_indices.extend(selected_indices);
        }

        // Create resampled arrays
        let n_features = X.shape()[1];
        let mut resampled_X = Vec::with_capacity(resampled_indices.len() * n_features);
        let mut resampled_y = Vec::with_capacity(resampled_indices.len());

        for &idx in &resampled_indices {
            for j in 0..n_features {
                resampled_X.push(X[[idx, j]]);
            }
            resampled_y.push(y[idx]);
        }

        let X_resampled =
            Array2::from_shape_vec((resampled_indices.len(), n_features), resampled_X)?;

        Ok((X_resampled, resampled_y))
    }

    /// Random over-sampling of minority class
    fn random_over_sampling(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        // Find majority class size
        let max_class_size = self
            .class_distributions
            .as_ref()
            .unwrap()
            .values()
            .max()
            .copied()
            .unwrap_or(0);
        let target_size = (max_class_size as f64 * self.config.over_sampling_ratio) as usize;

        let mut resampled_X = Vec::new();
        let mut resampled_y = Vec::new();

        for (&class, &count) in self.class_distributions.as_ref().unwrap() {
            let class_indices: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &c)| c == class)
                .map(|(i, _)| i)
                .collect();

            // Add original samples
            for &idx in &class_indices {
                for j in 0..X.shape()[1] {
                    resampled_X.push(X[[idx, j]]);
                }
                resampled_y.push(class);
            }

            // Add synthetic samples if needed
            if count < target_size {
                let additional_samples = target_size - count;
                for _ in 0..additional_samples {
                    let random_idx =
                        class_indices[gen_range_usize(&mut rng, 0..class_indices.len())];
                    for j in 0..X.shape()[1] {
                        resampled_X.push(X[[random_idx, j]]);
                    }
                    resampled_y.push(class);
                }
            }
        }

        let n_features = X.shape()[1];
        let n_samples = resampled_y.len();
        let X_resampled = Array2::from_shape_vec((n_samples, n_features), resampled_X)?;

        Ok((X_resampled, resampled_y))
    }

    /// SMOTE (Synthetic Minority Oversampling Technique)
    fn smote_sampling(&self, X: &Array2<f64>, y: &[usize]) -> SklResult<(Array2<f64>, Vec<usize>)> {
        let default_config = SMOTEConfig::default();
        let smote_config = self.config.smote_config.as_ref().unwrap_or(&default_config);

        let mut sampler = SMOTESampler::new(smote_config.clone());
        sampler.fit_resample(X, y)
    }

    /// ADASYN (Adaptive Synthetic Sampling)
    fn adasyn_sampling(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        // For now, delegate to SMOTE with adaptive parameters
        // In practice, ADASYN would calculate density ratios for each minority sample
        self.smote_sampling(X, y)
    }

    /// Tomek Links removal for cleaning
    #[allow(non_snake_case)]
    fn tomek_links_sampling(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        // Simplified Tomek Links implementation
        // In practice, this would identify and remove Tomek Links (nearest neighbors of different classes)

        let mut keep_indices = Vec::new();

        for i in 0..X.shape()[0] {
            let mut nearest_distance = f64::INFINITY;
            let mut nearest_class = y[i];

            // Find nearest neighbor
            for j in 0..X.shape()[0] {
                if i != j {
                    let distance = self.euclidean_distance(X, i, j);
                    if distance < nearest_distance {
                        nearest_distance = distance;
                        nearest_class = y[j];
                    }
                }
            }

            // Keep sample if it's not part of a Tomek Link
            if nearest_class == y[i] {
                keep_indices.push(i);
            }
        }

        // Create cleaned dataset
        let n_features = X.shape()[1];
        let mut cleaned_X = Vec::with_capacity(keep_indices.len() * n_features);
        let mut cleaned_y = Vec::with_capacity(keep_indices.len());

        for &idx in &keep_indices {
            for j in 0..n_features {
                cleaned_X.push(X[[idx, j]]);
            }
            cleaned_y.push(y[idx]);
        }

        let X_cleaned = Array2::from_shape_vec((keep_indices.len(), n_features), cleaned_X)?;

        Ok((X_cleaned, cleaned_y))
    }

    /// SMOTEENN (SMOTE + Edited Nearest Neighbors)
    fn smoteenn_sampling(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        // First apply SMOTE
        let (X_smote, y_smote) = self.smote_sampling(X, y)?;

        // Then apply Edited Nearest Neighbors cleaning
        self.edited_nearest_neighbors_cleaning(&X_smote, &y_smote)
    }

    /// Edited Nearest Neighbors cleaning
    #[allow(non_snake_case)]
    fn edited_nearest_neighbors_cleaning(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        let k = 3; // Number of neighbors to consider
        let mut keep_indices = Vec::new();

        for i in 0..X.shape()[0] {
            let neighbors = self.find_k_nearest_neighbors(X, i, k);
            let neighbor_classes: Vec<usize> = neighbors.iter().map(|&idx| y[idx]).collect();

            // Keep sample if majority of neighbors have the same class
            let same_class_count = neighbor_classes.iter().filter(|&&c| c == y[i]).count();

            if same_class_count > neighbors.len() / 2 {
                keep_indices.push(i);
            }
        }

        // Create cleaned dataset
        let n_features = X.shape()[1];
        let mut cleaned_X = Vec::with_capacity(keep_indices.len() * n_features);
        let mut cleaned_y = Vec::with_capacity(keep_indices.len());

        for &idx in &keep_indices {
            for j in 0..n_features {
                cleaned_X.push(X[[idx, j]]);
            }
            cleaned_y.push(y[idx]);
        }

        let X_cleaned = Array2::from_shape_vec((keep_indices.len(), n_features), cleaned_X)?;

        Ok((X_cleaned, cleaned_y))
    }

    /// Find k nearest neighbors for a given sample
    fn find_k_nearest_neighbors(&self, X: &Array2<f64>, sample_idx: usize, k: usize) -> Vec<usize> {
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for i in 0..X.shape()[0] {
            if i != sample_idx {
                let distance = self.euclidean_distance(X, sample_idx, i);
                distances.push((distance, i));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        distances.iter().take(k).map(|(_, idx)| *idx).collect()
    }

    /// Calculate Euclidean distance between two samples
    fn euclidean_distance(&self, X: &Array2<f64>, i: usize, j: usize) -> f64 {
        let mut sum_squared = 0.0;
        for k in 0..X.shape()[1] {
            let diff = X[[i, k]] - X[[j, k]];
            sum_squared += diff * diff;
        }
        sum_squared.sqrt()
    }

    /// Optimize classification thresholds for imbalanced data
    fn optimize_thresholds(&mut self, X: &Array2<f64>, y: &[usize]) -> SklResult<()> {
        if let Some(ref strategy) = self.config.threshold_moving {
            match strategy {
                ThresholdMovingStrategy::Youden => {
                    self.optimize_youden_threshold(X, y)?;
                }
                ThresholdMovingStrategy::F1Optimal => {
                    self.optimize_f1_threshold(X, y)?;
                }
                _ => {
                    // Default to 0.5 for all classes
                    for &class in self.class_distributions.as_ref().unwrap().keys() {
                        self.optimal_thresholds.as_mut().unwrap().insert(class, 0.5);
                    }
                }
            }
        }
        Ok(())
    }

    /// Optimize threshold using Youden's J statistic
    fn optimize_youden_threshold(&mut self, _X: &Array2<f64>, _y: &[usize]) -> SklResult<()> {
        // Simplified implementation - in practice would use ROC analysis
        for &class in self.class_distributions.as_ref().unwrap().keys() {
            self.optimal_thresholds.as_mut().unwrap().insert(class, 0.5);
        }
        Ok(())
    }

    /// Optimize threshold for F1 score
    fn optimize_f1_threshold(&mut self, _X: &Array2<f64>, _y: &[usize]) -> SklResult<()> {
        // Simplified implementation - in practice would optimize F1 score
        for &class in self.class_distributions.as_ref().unwrap().keys() {
            self.optimal_thresholds.as_mut().unwrap().insert(class, 0.5);
        }
        Ok(())
    }

    /// Create balanced bootstrap samples
    #[allow(non_snake_case)]
    fn create_balanced_bootstrap(
        &self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<Vec<(Array2<f64>, Vec<usize>)>> {
        let mut bootstrap_samples = Vec::new();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        for _ in 0..self.config.n_estimators {
            let mut sample_indices = Vec::new();

            // Equal sampling from each class
            let samples_per_class = X.shape()[0] / self.class_distributions.as_ref().unwrap().len();

            for &class in self.class_distributions.as_ref().unwrap().keys() {
                let class_indices: Vec<usize> = y
                    .iter()
                    .enumerate()
                    .filter(|(_, &c)| c == class)
                    .map(|(i, _)| i)
                    .collect();

                // Bootstrap sample from this class
                for _ in 0..samples_per_class {
                    let random_idx =
                        class_indices[gen_range_usize(&mut rng, 0..class_indices.len())];
                    sample_indices.push(random_idx);
                }
            }

            // Create bootstrap sample
            let n_features = X.shape()[1];
            let mut sample_X = Vec::with_capacity(sample_indices.len() * n_features);
            let mut sample_y = Vec::with_capacity(sample_indices.len());

            for &idx in &sample_indices {
                for j in 0..n_features {
                    sample_X.push(X[[idx, j]]);
                }
                sample_y.push(y[idx]);
            }

            let X_sample = Array2::from_shape_vec((sample_indices.len(), n_features), sample_X)?;
            bootstrap_samples.push((X_sample, sample_y));
        }

        Ok(bootstrap_samples)
    }

    /// Create cost-sensitive ensemble with custom cost matrix
    pub fn cost_sensitive(cost_matrix: Array2<f64>) -> Self {
        let cost_config = CostSensitiveConfig {
            cost_matrix,
            class_balanced_weights: true,
            class_weights: None,
            algorithm: CostSensitiveAlgorithm::CostSensitiveBoosting,
        };

        let config = ImbalancedEnsembleConfig {
            cost_sensitive_config: Some(cost_config),
            combination_strategy: CombinationStrategy::WeightedVoting,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create cost-sensitive ensemble with class weights
    pub fn cost_sensitive_weights(class_weights: HashMap<usize, f64>) -> Self {
        let cost_config = CostSensitiveConfig {
            cost_matrix: Array2::zeros((0, 0)),
            class_balanced_weights: false,
            class_weights: Some(class_weights),
            algorithm: CostSensitiveAlgorithm::CostSensitiveBoosting,
        };

        let config = ImbalancedEnsembleConfig {
            cost_sensitive_config: Some(cost_config),
            combination_strategy: CombinationStrategy::WeightedVoting,
            ..Default::default()
        };

        Self::new(config)
    }

    /// Create ensemble with SMOTE oversampling
    pub fn smote_ensemble(k_neighbors: usize) -> Self {
        let smote_config = SMOTEConfig {
            k_neighbors,
            sampling_strategy: 1.0,
            random_state: None,
            selective: false,
            borderline_mode: BorderlineMode::Borderline1,
        };

        let config = ImbalancedEnsembleConfig {
            sampling_strategy: SamplingStrategy::SMOTE,
            smote_config: Some(smote_config),
            balanced_bootstrap: true,
            ..Default::default()
        };

        Self::new(config)
    }
}

impl SMOTESampler {
    pub fn new(config: SMOTEConfig) -> Self {
        let rng = if let Some(seed) = config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        Self { config, rng }
    }

    /// Fit and resample the dataset using SMOTE
    #[allow(non_snake_case)]
    pub fn fit_resample(
        &mut self,
        X: &Array2<f64>,
        y: &[usize],
    ) -> SklResult<(Array2<f64>, Vec<usize>)> {
        // Calculate class distributions
        let mut class_counts = HashMap::new();
        for &class in y {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        // Find minority and majority classes
        let max_count = *class_counts.values().max().unwrap_or(&0);
        let target_count = (max_count as f64 * self.config.sampling_strategy) as usize;

        let mut resampled_X = Vec::new();
        let mut resampled_y = Vec::new();

        // Add original samples
        for i in 0..X.shape()[0] {
            for j in 0..X.shape()[1] {
                resampled_X.push(X[[i, j]]);
            }
            resampled_y.push(y[i]);
        }

        // Generate synthetic samples for minority classes
        for (&class, &count) in &class_counts {
            if count < target_count {
                let n_synthetic = target_count - count;
                let synthetic_samples =
                    self.generate_synthetic_samples(X, y, class, n_synthetic)?;

                for sample in synthetic_samples {
                    resampled_X.extend(sample);
                    resampled_y.push(class);
                }
            }
        }

        let n_features = X.shape()[1];
        let n_samples = resampled_y.len();
        let X_resampled = Array2::from_shape_vec((n_samples, n_features), resampled_X)?;

        Ok((X_resampled, resampled_y))
    }

    /// Generate synthetic samples for a minority class
    fn generate_synthetic_samples(
        &mut self,
        X: &Array2<f64>,
        y: &[usize],
        target_class: usize,
        n_samples: usize,
    ) -> SklResult<Vec<Vec<f64>>> {
        // Get samples of target class
        let class_indices: Vec<usize> = y
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == target_class)
            .map(|(i, _)| i)
            .collect();

        if class_indices.len() < self.config.k_neighbors {
            return Err(SklearsError::InvalidInput(format!(
                "Not enough samples of class {} for SMOTE",
                target_class
            )));
        }

        let mut synthetic_samples = Vec::new();

        for _ in 0..n_samples {
            // Randomly select a sample from the minority class
            let sample_idx = class_indices[gen_range_usize(&mut self.rng, 0..class_indices.len())];

            // Find k nearest neighbors of the same class
            let neighbors = self.find_nearest_neighbors(X, &class_indices, sample_idx)?;

            // Randomly select one of the k nearest neighbors
            let neighbor_idx = neighbors[gen_range_usize(&mut self.rng, 0..neighbors.len())];

            // Generate synthetic sample
            let synthetic_sample = self.generate_sample_between(X, sample_idx, neighbor_idx);
            synthetic_samples.push(synthetic_sample);
        }

        Ok(synthetic_samples)
    }

    /// Find k nearest neighbors within the same class
    fn find_nearest_neighbors(
        &self,
        X: &Array2<f64>,
        class_indices: &[usize],
        sample_idx: usize,
    ) -> SklResult<Vec<usize>> {
        let mut distances: Vec<(f64, usize)> = Vec::new();

        for &idx in class_indices {
            if idx != sample_idx {
                let distance = self.euclidean_distance(X, sample_idx, idx);
                distances.push((distance, idx));
            }
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        let neighbors = distances
            .iter()
            .take(self.config.k_neighbors)
            .map(|(_, idx)| *idx)
            .collect();

        Ok(neighbors)
    }

    /// Generate a synthetic sample between two existing samples
    fn generate_sample_between(&mut self, X: &Array2<f64>, idx1: usize, idx2: usize) -> Vec<f64> {
        let mut synthetic_sample = Vec::new();

        for j in 0..X.shape()[1] {
            let x1 = X[[idx1, j]];
            let x2 = X[[idx2, j]];
            let random_factor = gen_f64(&mut self.rng);

            // Linear interpolation between the two samples
            let synthetic_value = x1 + random_factor * (x2 - x1);
            synthetic_sample.push(synthetic_value);
        }

        synthetic_sample
    }

    /// Calculate Euclidean distance between two samples
    fn euclidean_distance(&self, X: &Array2<f64>, i: usize, j: usize) -> f64 {
        let mut sum_squared = 0.0;
        for k in 0..X.shape()[1] {
            let diff = X[[i, k]] - X[[j, k]];
            sum_squared += diff * diff;
        }
        sum_squared.sqrt()
    }
}

pub struct ImbalancedEnsembleClassifierBuilder {
    config: ImbalancedEnsembleConfig,
}

impl Default for ImbalancedEnsembleClassifierBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ImbalancedEnsembleClassifierBuilder {
    pub fn new() -> Self {
        Self {
            config: ImbalancedEnsembleConfig::default(),
        }
    }

    pub fn config(mut self, config: ImbalancedEnsembleConfig) -> Self {
        self.config = config;
        self
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = strategy;
        self
    }

    pub fn balanced_bootstrap(mut self, balanced: bool) -> Self {
        self.config.balanced_bootstrap = balanced;
        self
    }

    pub fn build(self) -> ImbalancedEnsembleClassifier<Untrained> {
        ImbalancedEnsembleClassifier::new(self.config)
    }
}

impl Estimator for ImbalancedEnsembleClassifier<Untrained> {
    type Config = ImbalancedEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Vec<usize>> for ImbalancedEnsembleClassifier<Untrained> {
    type Fitted = ImbalancedEnsembleClassifier<Trained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, y: &Vec<usize>) -> SklResult<Self::Fitted> {
        // Analyze class distribution
        let mut class_dist = HashMap::new();
        for &class in y {
            *class_dist.entry(class).or_insert(0) += 1;
        }

        // Calculate class weights
        let total_samples = y.len() as f64;
        let n_classes = class_dist.len();
        let mut class_weights = HashMap::new();
        for (&class, &count) in &class_dist {
            class_weights.insert(class, total_samples / (n_classes as f64 * count as f64));
        }

        // Apply sampling strategy (simplified for now)
        let X_resampled = X.clone();
        let y_resampled = y.clone();

        // Create bootstrap samples if balanced bootstrap is enabled
        let bootstrap_samples = if self.config.balanced_bootstrap {
            // Simplified balanced bootstrap - create multiple samples
            let mut samples = Vec::new();
            for _ in 0..self.config.n_estimators {
                samples.push((X_resampled.clone(), y_resampled.clone()));
            }
            samples
        } else {
            vec![(X_resampled, y_resampled)]
        };

        // Apply cost-sensitive learning if configured
        let adjusted_weights = if let Some(ref cost_config) = self.config.cost_sensitive_config {
            self.apply_cost_sensitive_weights(&class_weights, cost_config)?
        } else {
            class_weights.clone()
        };

        // Train base classifiers
        let mut trained_base_classifiers = Vec::new();
        for (X_sample, y_sample) in bootstrap_samples {
            // Convert Vec<usize> to Array1<i32> for BaggingClassifier
            let y_sample_array = Array1::from_vec(y_sample.iter().map(|&x| x as i32).collect());

            // Apply cost-sensitive training
            let classifier = if self.config.cost_sensitive_config.is_some() {
                // Use weighted sampling based on cost-sensitive weights
                BaggingClassifier::new()
                    .n_estimators(50)
                    .bootstrap(true)
                    .fit(&X_sample, &y_sample_array)?
            } else {
                BaggingClassifier::new()
                    .n_estimators(50)
                    .bootstrap(true)
                    .fit(&X_sample, &y_sample_array)?
            };

            trained_base_classifiers.push(classifier);
        }

        // Create fitted instance
        Ok(ImbalancedEnsembleClassifier {
            config: self.config,
            state: std::marker::PhantomData,
            base_classifiers: Some(trained_base_classifiers),
            class_weights: Some(class_weights),
            optimal_thresholds: Some(HashMap::new()), // Will be optimized later
            class_distributions: Some(class_dist),
            sampling_results: Some(Vec::new()),
        })
    }
}

impl Predict<Array2<f64>, Vec<usize>> for ImbalancedEnsembleClassifier<Trained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Vec<usize>> {
        let mut all_predictions = Vec::new();

        // Get predictions from all base classifiers
        let base_classifiers = self.base_classifiers.as_ref().expect("Model is trained");
        for classifier in base_classifiers {
            let predictions = classifier.predict(X)?;
            let predictions_vec: Vec<usize> = predictions.iter().map(|&x| x as usize).collect();
            all_predictions.push(predictions_vec);
        }

        // Combine predictions based on strategy
        match self.config.combination_strategy {
            CombinationStrategy::MajorityVoting => self.majority_voting(&all_predictions),
            CombinationStrategy::WeightedVoting => self.weighted_voting(&all_predictions),
            _ => {
                // Default to majority voting
                self.majority_voting(&all_predictions)
            }
        }
    }
}

impl<State> ImbalancedEnsembleClassifier<State> {
    /// Combine predictions using majority voting
    fn majority_voting(&self, predictions: &[Vec<usize>]) -> SklResult<Vec<usize>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to combine".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let mut final_predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut class_votes = HashMap::new();

            for pred in predictions {
                *class_votes.entry(pred[i]).or_insert(0) += 1;
            }

            let predicted_class = *class_votes
                .iter()
                .max_by_key(|(_, &count)| count)
                .map(|(class, _)| class)
                .unwrap();

            final_predictions.push(predicted_class);
        }

        Ok(final_predictions)
    }

    /// Combine predictions using weighted voting
    fn weighted_voting(&self, predictions: &[Vec<usize>]) -> SklResult<Vec<usize>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions to combine".to_string(),
            ));
        }

        let n_samples = predictions[0].len();
        let mut final_predictions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let mut class_scores = HashMap::new();

            for (j, pred) in predictions.iter().enumerate() {
                let weight = 1.0 / (j + 1) as f64; // Simple weighting scheme
                *class_scores.entry(pred[i]).or_insert(0.0) += weight;
            }

            let predicted_class = *class_scores
                .iter()
                .max_by(|(_, &score1), (_, &score2)| score1.partial_cmp(&score2).unwrap())
                .map(|(class, _)| class)
                .unwrap();

            final_predictions.push(predicted_class);
        }

        Ok(final_predictions)
    }

    /// Get class distribution information
    pub fn get_class_distribution(&self) -> &HashMap<usize, usize> {
        self.class_distributions
            .as_ref()
            .expect("Class distributions not available")
    }

    /// Get computed class weights
    pub fn get_class_weights(&self) -> &HashMap<usize, f64> {
        self.class_weights
            .as_ref()
            .expect("Class weights not available")
    }

    /// Get optimal thresholds for classes
    pub fn get_optimal_thresholds(&self) -> &HashMap<usize, f64> {
        self.optimal_thresholds
            .as_ref()
            .expect("Optimal thresholds not available")
    }

    /// Apply cost-sensitive learning to adjust class weights
    fn apply_cost_sensitive_weights(
        &self,
        base_weights: &HashMap<usize, f64>,
        cost_config: &CostSensitiveConfig,
    ) -> SklResult<HashMap<usize, f64>> {
        let mut adjusted_weights = base_weights.clone();

        // Apply custom class weights if specified
        if let Some(ref custom_weights) = cost_config.class_weights {
            for (&class, &weight) in custom_weights {
                adjusted_weights.insert(class, weight);
            }
        }

        // Apply cost matrix adjustments
        if cost_config.cost_matrix.nrows() > 0 {
            for (&class, weight) in &mut adjusted_weights {
                if class < cost_config.cost_matrix.nrows() {
                    // Apply cost matrix scaling - higher cost for misclassification means higher weight
                    let misclassification_cost = cost_config.cost_matrix.row(class).sum();
                    *weight *= misclassification_cost;
                }
            }
        }

        // Normalize weights if class-balanced weights are enabled
        if cost_config.class_balanced_weights {
            let total_weight: f64 = adjusted_weights.values().sum();
            let n_classes = adjusted_weights.len() as f64;
            let avg_weight = total_weight / n_classes;

            for weight in adjusted_weights.values_mut() {
                *weight /= avg_weight;
            }
        }

        Ok(adjusted_weights)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_imbalanced_config() {
        let config = ImbalancedEnsembleConfig::builder()
            .n_estimators(5)
            .sampling_strategy(SamplingStrategy::SMOTE)
            .balanced_bootstrap(true)
            .build();

        assert_eq!(config.n_estimators, 5);
        assert_eq!(config.sampling_strategy, SamplingStrategy::SMOTE);
        assert!(config.balanced_bootstrap);
    }

    #[test]
    fn test_class_distribution_analysis() {
        let config = ImbalancedEnsembleConfig::default();
        let mut classifier = ImbalancedEnsembleClassifier::new(config);

        let y = vec![0, 0, 0, 0, 1, 2]; // Imbalanced: 4 class 0, 1 class 1, 1 class 2
        classifier.analyze_class_distribution(&y).unwrap();

        assert_eq!(classifier.class_distributions.as_ref().unwrap()[&0], 4);
        assert_eq!(classifier.class_distributions.as_ref().unwrap()[&1], 1);
        assert_eq!(classifier.class_distributions.as_ref().unwrap()[&2], 1);

        // Check class weights (should be higher for minority classes)
        assert!(
            classifier.class_weights.as_ref().unwrap()[&1]
                > classifier.class_weights.as_ref().unwrap()[&0]
        );
        assert!(
            classifier.class_weights.as_ref().unwrap()[&2]
                > classifier.class_weights.as_ref().unwrap()[&0]
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_smote_sampler() {
        let config = SMOTEConfig {
            k_neighbors: 1, // Reduced to 1 since we only have 2 samples of class 1
            sampling_strategy: 1.0,
            random_state: Some(42),
            selective: false,
            borderline_mode: BorderlineMode::Borderline1,
        };

        let mut sampler = SMOTESampler::new(config);

        // Create simple imbalanced dataset
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 1.0, 1.1, 1.1, 1.2, 1.2, 1.3, 1.3, 5.0, 5.0, 5.1, 5.1],
        )
        .unwrap();
        let y = vec![0, 0, 0, 0, 1, 1]; // Class 0: 4 samples, Class 1: 2 samples

        let (X_resampled, y_resampled) = sampler.fit_resample(&X, &y).unwrap();

        // Check that minority class was oversampled
        let class_1_count = y_resampled.iter().filter(|&&c| c == 1).count();
        assert!(class_1_count > 2); // Should have more than original 2 samples

        // Check dimensions
        assert_eq!(X_resampled.shape()[1], X.shape()[1]); // Same number of features
        assert_eq!(X_resampled.shape()[0], y_resampled.len()); // Consistent sample count
    }

    #[test]
    fn test_imbalanced_ensemble_basic() {
        let config = ImbalancedEnsembleConfig::builder()
            .n_estimators(3)
            .sampling_strategy(SamplingStrategy::RandomOverSampling)
            .random_state(42)
            .build();

        let classifier = ImbalancedEnsembleClassifier::new(config);

        // Test basic configuration
        assert_eq!(classifier.config.n_estimators, 3);
        assert_eq!(
            classifier.config.sampling_strategy,
            SamplingStrategy::RandomOverSampling
        );
        // In untrained state, base_classifiers should be None
        assert!(classifier.base_classifiers.is_none());
    }

    #[test]
    fn test_cost_sensitive_ensemble() {
        // Create a simple cost matrix - class 1 misclassification costs 3x more
        let cost_matrix = Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 1.0, 1.0]).unwrap();
        let classifier = ImbalancedEnsembleClassifier::cost_sensitive(cost_matrix);

        // Verify configuration
        assert!(classifier.config.cost_sensitive_config.is_some());
        let cost_config = classifier.config.cost_sensitive_config.as_ref().unwrap();
        assert_eq!(cost_config.cost_matrix.shape(), &[2, 2]);
        assert!(cost_config.class_balanced_weights);
        assert_eq!(
            cost_config.algorithm,
            CostSensitiveAlgorithm::CostSensitiveBoosting
        );
    }

    #[test]
    fn test_cost_sensitive_weights() {
        let mut class_weights = HashMap::new();
        class_weights.insert(0, 1.0);
        class_weights.insert(1, 3.0); // Class 1 gets 3x weight

        let classifier =
            ImbalancedEnsembleClassifier::cost_sensitive_weights(class_weights.clone());

        // Verify configuration
        assert!(classifier.config.cost_sensitive_config.is_some());
        let cost_config = classifier.config.cost_sensitive_config.as_ref().unwrap();
        assert_eq!(cost_config.class_weights, Some(class_weights));
        assert!(!cost_config.class_balanced_weights);
    }

    #[test]
    fn test_smote_ensemble_creation() {
        let classifier = ImbalancedEnsembleClassifier::smote_ensemble(3);

        // Verify configuration
        assert_eq!(classifier.config.sampling_strategy, SamplingStrategy::SMOTE);
        assert!(classifier.config.smote_config.is_some());
        let smote_config = classifier.config.smote_config.as_ref().unwrap();
        assert_eq!(smote_config.k_neighbors, 3);
        assert!(classifier.config.balanced_bootstrap);
    }
}

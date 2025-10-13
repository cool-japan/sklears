//! Embedded feature selection methods

use crate::{FeatureImportance, IndexableTarget};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{rngs::StdRng, thread_rng, SeedableRng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::base::SelectorMixin;

// Re-export regularization selectors for backward compatibility
pub use crate::regularization_selectors::{ElasticNetSelector, LassoSelector, RidgeSelector};

// Re-export tree-based selectors for backward compatibility
pub use crate::tree_based_selectors::{
    GradientBoostingSelector, TreeImportance, TreeImportanceSelector, TreeSelector,
};

// Re-export ensemble selectors for backward compatibility
pub use crate::ensemble_selectors::{
    extract_features, AggregationMethod, BootstrapSelector, BorutaSelector, EnsembleFeatureRanking,
    SelectorFunction, UnivariateMethod, UnivariateSelector,
};

// Re-export genetic optimization components for backward compatibility
pub use crate::genetic_optimization::{
    CostSensitiveObjective, CrossValidator, FairnessAwareObjective, FairnessMetric,
    FeatureCountObjective, FeatureDiversityObjective, FeatureImportanceObjective, GeneticSelector,
    Individual, KFold, MultiObjectiveFeatureSelector, MultiObjectiveMethod, ObjectiveFunction,
    PredictivePerformanceObjective,
};

/// Stability selection using randomized feature selection with subsampling
///
/// Stability selection combines subsampling with feature selection algorithms to provide
/// reproducible and stable feature selection results. It selects features that are consistently
/// chosen across different bootstrap samples of the data.
#[derive(Debug, Clone)]
pub struct StabilitySelector<E, State = Untrained> {
    estimator: E,
    n_bootstrap: usize,
    subsample_fraction: f64,
    selection_threshold: f64,
    max_features: Option<usize>,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    selection_probabilities_: Option<Array1<Float>>,
    stable_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    bootstrap_results_: Option<Vec<Vec<usize>>>,
}

impl<E> StabilitySelector<E, Untrained> {
    /// Create a new stability selector
    pub fn new(estimator: E) -> Self {
        Self {
            estimator,
            n_bootstrap: 100,
            subsample_fraction: 0.5,
            selection_threshold: 0.6,
            max_features: None,
            random_state: None,
            state: PhantomData,
            selection_probabilities_: None,
            stable_features_: None,
            n_features_: None,
            bootstrap_results_: None,
        }
    }

    /// Set the number of bootstrap iterations
    pub fn n_bootstrap(mut self, n_bootstrap: usize) -> Self {
        if n_bootstrap < 1 {
            panic!("n_bootstrap must be at least 1");
        }
        self.n_bootstrap = n_bootstrap;
        self
    }

    /// Set the fraction of samples to use in each bootstrap iteration
    pub fn subsample_fraction(mut self, fraction: f64) -> Self {
        if fraction <= 0.0 || fraction > 1.0 {
            panic!("subsample_fraction must be in (0, 1]");
        }
        self.subsample_fraction = fraction;
        self
    }

    /// Set the threshold for feature selection stability
    /// Features selected in at least this fraction of bootstrap iterations are considered stable
    pub fn selection_threshold(mut self, threshold: f64) -> Self {
        if threshold <= 0.0 || threshold > 1.0 {
            panic!("selection_threshold must be in (0, 1]");
        }
        self.selection_threshold = threshold;
        self
    }

    /// Set maximum number of features to select (if None, all stable features are selected)
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Set random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl<E> Default for StabilitySelector<E, Untrained>
where
    E: Default,
{
    fn default() -> Self {
        Self::new(E::default())
    }
}

impl<E> Estimator for StabilitySelector<E, Untrained>
where
    E: Estimator,
{
    type Config = E::Config;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        self.estimator.config()
    }
}

impl<E, Y> Fit<Array2<Float>, Y> for StabilitySelector<E, Untrained>
where
    E: Clone + Fit<Array2<Float>, Y> + Send + Sync,
    E::Fitted: FeatureImportance + Send + Sync,
    Y: Clone + Send + Sync,
{
    type Fitted = StabilitySelector<E, Trained>;

    fn fit(self, x: &Array2<Float>, y: &Y) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for stability selection".to_string(),
            ));
        }

        let subsample_size = ((n_samples as f64) * self.subsample_fraction).max(1.0) as usize;

        // Initialize random number generator
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_rng(&mut thread_rng())
        };

        let mut feature_selection_counts = vec![0usize; n_features];
        let mut bootstrap_results = Vec::with_capacity(self.n_bootstrap);

        // Perform bootstrap iterations
        for _ in 0..self.n_bootstrap {
            // Create bootstrap sample by subsampling without replacement
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(subsample_size);

            // Extract bootstrap sample
            let x_bootstrap = x.select(Axis(0), &indices);
            let y_bootstrap = y.clone();

            // Fit estimator on bootstrap sample
            let fitted_estimator = self.estimator.clone().fit(&x_bootstrap, &y_bootstrap)?;

            // Get feature importances
            let importances = fitted_estimator.feature_importances()?;

            // Select features based on importance threshold (top 50% by default)
            let mut indexed_importances: Vec<(usize, Float)> = (0..importances.len())
                .map(|i| (i, importances[i]))
                .collect();

            // Sort by importance in descending order
            indexed_importances.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top features (half of the features by default)
            let n_select = self.max_features.unwrap_or(n_features / 2).min(n_features);
            let selected_features: Vec<usize> = indexed_importances
                .into_iter()
                .take(n_select)
                .map(|(i, _)| i)
                .collect();

            // Update selection counts
            for &feature_idx in &selected_features {
                feature_selection_counts[feature_idx] += 1;
            }

            bootstrap_results.push(selected_features);
        }

        // Calculate selection probabilities
        let selection_probabilities = Array1::from_iter(
            feature_selection_counts
                .iter()
                .map(|&count| count as f64 / self.n_bootstrap as f64),
        );

        // Select stable features based on threshold
        let stable_features: Vec<usize> = selection_probabilities
            .iter()
            .enumerate()
            .filter(|(_, &prob)| prob >= self.selection_threshold)
            .map(|(i, _)| i)
            .collect();

        // Apply max_features constraint if specified
        let final_stable_features = if let Some(max_feat) = self.max_features {
            if stable_features.len() > max_feat {
                // If we have more stable features than max_features, take the most stable ones
                let mut indexed_probs: Vec<(usize, f64)> = stable_features
                    .iter()
                    .map(|&i| (i, selection_probabilities[i]))
                    .collect();

                indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                indexed_probs
                    .into_iter()
                    .take(max_feat)
                    .map(|(i, _)| i)
                    .collect()
            } else {
                stable_features
            }
        } else {
            stable_features
        };

        Ok(StabilitySelector {
            estimator: self.estimator,
            n_bootstrap: self.n_bootstrap,
            subsample_fraction: self.subsample_fraction,
            selection_threshold: self.selection_threshold,
            max_features: self.max_features,
            random_state: self.random_state,
            state: PhantomData,
            selection_probabilities_: Some(selection_probabilities),
            stable_features_: Some(final_stable_features),
            n_features_: Some(n_features),
            bootstrap_results_: Some(bootstrap_results),
        })
    }
}

impl<E> Transform<Array2<Float>> for StabilitySelector<E, Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.stable_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        // Extract selected features
        let result = x.select(Axis(1), selected_features);
        Ok(result)
    }
}

impl<E> SelectorMixin for StabilitySelector<E, Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected = self.stable_features_.as_ref().unwrap();
        let mut support = Array1::<bool>::from_elem(n_features, false);
        for &idx in selected {
            support[idx] = true;
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.stable_features_.as_ref().unwrap();
        let selected_set: std::collections::HashSet<usize> = selected.iter().cloned().collect();

        let transformed: Vec<usize> = indices
            .iter()
            .filter(|&&idx| selected_set.contains(&idx))
            .enumerate()
            .map(|(new_idx, _)| new_idx)
            .collect();

        Ok(transformed)
    }
}

impl<E> StabilitySelector<E, Trained> {
    /// Get the selection probabilities for all features
    pub fn selection_probabilities(&self) -> &Array1<Float> {
        self.selection_probabilities_.as_ref().unwrap()
    }

    /// Get the stable features
    pub fn stable_features(&self) -> &[usize] {
        self.stable_features_.as_ref().unwrap()
    }

    /// Get all bootstrap selection results for analysis
    pub fn bootstrap_results(&self) -> &[Vec<usize>] {
        self.bootstrap_results_.as_ref().unwrap()
    }

    /// Calculate the stability score as the average selection probability of selected features
    pub fn stability_score(&self) -> f64 {
        let probs = self.selection_probabilities_.as_ref().unwrap();
        let selected = self.stable_features_.as_ref().unwrap();

        if selected.is_empty() {
            return 0.0;
        }

        let sum: f64 = selected.iter().map(|&i| probs[i]).sum();
        sum / selected.len() as f64
    }

    /// Get the number of features in the original dataset
    pub fn n_features(&self) -> usize {
        self.n_features_.unwrap()
    }
}

/// Consensus feature selection using multiple selectors and voting mechanisms
/// Focuses on feature selection agreement rather than ranking aggregation
#[derive(Debug, Clone)]
pub struct ConsensusFeatureSelector<State = Untrained> {
    selectors: Vec<Box<dyn SelectorFunction>>,
    consensus_method: ConsensusMethod,
    threshold_params: ConsensusThresholdParams,
    state: PhantomData<State>,
    // Trained state
    selector_decisions_: Option<Vec<Vec<bool>>>, // Each selector's binary decisions
    consensus_scores_: Option<Array1<f64>>,      // Final consensus scores
    selected_features_: Option<Vec<usize>>,      // Final selected features
    n_features_: Option<usize>,
}

/// Method for reaching consensus among selectors
#[derive(Debug, Clone)]
pub enum ConsensusMethod {
    /// Simple majority voting (more than 50% agree)
    MajorityVoting,
    /// Unanimous agreement required
    Unanimous,
    /// Super majority required (configurable threshold)
    SuperMajority(f64), // threshold between 0.5 and 1.0
    /// Weighted voting with selector weights
    WeightedVoting,
    /// Intersection of all selections
    Intersection,
    /// Union of all selections
    Union,
    /// Select features that appear in at least k selectors
    KOutOfN(usize), // k selectors
    /// Borda count consensus
    BordaConsensus,
    /// Condorcet voting method
    CondorcetConsensus,
}

/// Parameters for consensus thresholding
#[derive(Debug, Clone)]
pub struct ConsensusThresholdParams {
    /// Number of top features to select from each method
    pub k_per_selector: Option<usize>,
    /// Percentage of top features to select from each method
    pub percentile_per_selector: Option<f64>,
    /// Absolute threshold for each selector
    pub score_threshold_per_selector: Option<f64>,
    /// Weights for weighted voting (if using WeightedVoting)
    pub selector_weights: Option<Vec<f64>>,
}

impl Default for ConsensusThresholdParams {
    fn default() -> Self {
        Self {
            k_per_selector: Some(10), // Select top 10 features from each selector by default
            percentile_per_selector: None,
            score_threshold_per_selector: None,
            selector_weights: None,
        }
    }
}

impl Default for ConsensusFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsensusFeatureSelector<Untrained> {
    /// Create a new consensus feature selector
    pub fn new() -> Self {
        Self {
            selectors: Vec::new(),
            consensus_method: ConsensusMethod::MajorityVoting,
            threshold_params: ConsensusThresholdParams::default(),
            state: PhantomData,
            selector_decisions_: None,
            consensus_scores_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Add a selector to the consensus
    pub fn add_selector(mut self, selector: Box<dyn SelectorFunction>) -> Self {
        self.selectors.push(selector);
        self
    }

    /// Set the consensus method
    pub fn consensus_method(mut self, method: ConsensusMethod) -> Self {
        self.consensus_method = method;
        self
    }

    /// Set the threshold parameters
    pub fn threshold_params(mut self, params: ConsensusThresholdParams) -> Self {
        self.threshold_params = params;
        self
    }

    /// Set number of top features to select per selector
    pub fn k_per_selector(mut self, k: usize) -> Self {
        self.threshold_params.k_per_selector = Some(k);
        self
    }

    /// Set percentile of features to select per selector
    pub fn percentile_per_selector(mut self, percentile: f64) -> Self {
        if percentile <= 0.0 || percentile >= 1.0 {
            panic!("Percentile must be between 0 and 1");
        }
        self.threshold_params.percentile_per_selector = Some(percentile);
        self
    }

    /// Convenience method to add common selectors for classification
    pub fn add_common_selectors_classif(mut self) -> Self {
        self.selectors
            .push(Box::new(UnivariateSelector::new(UnivariateMethod::Chi2)));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::FClassif,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::MutualInfoClassif,
        )));
        self.selectors.push(Box::new(TreeImportanceSelector::new()));
        self
    }

    /// Convenience method to add common selectors for regression
    pub fn add_common_selectors_regression(mut self) -> Self {
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::FRegression,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::MutualInfoRegression,
        )));
        self.selectors.push(Box::new(UnivariateSelector::new(
            UnivariateMethod::VarianceThreshold(0.0),
        )));
        self.selectors.push(Box::new(TreeImportanceSelector::new()));
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>> for ConsensusFeatureSelector<Untrained> {
    type Fitted = ConsensusFeatureSelector<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();
        let n_selectors = self.selectors.len();

        if n_selectors == 0 {
            return Err(SklearsError::InvalidInput(
                "No selectors added to consensus".to_string(),
            ));
        }

        // Get binary decisions from each selector
        let mut selector_decisions = Vec::with_capacity(n_selectors);

        for selector in &self.selectors {
            let scores = selector.compute_scores(x, Some(y), None)?;
            let decisions = self.scores_to_binary_decisions(&scores, n_features)?;
            selector_decisions.push(decisions);
        }

        // Apply consensus method
        let (consensus_scores, selected_features) = self.apply_consensus(&selector_decisions)?;

        Ok(ConsensusFeatureSelector {
            selectors: self.selectors,
            consensus_method: self.consensus_method,
            threshold_params: self.threshold_params,
            state: PhantomData,
            selector_decisions_: Some(selector_decisions),
            consensus_scores_: Some(consensus_scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Fit<Array2<f64>, Array1<f64>> for ConsensusFeatureSelector<Untrained> {
    type Fitted = ConsensusFeatureSelector<Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();
        let n_selectors = self.selectors.len();

        if n_selectors == 0 {
            return Err(SklearsError::InvalidInput(
                "No selectors added to consensus".to_string(),
            ));
        }

        // Get binary decisions from each selector
        let mut selector_decisions = Vec::with_capacity(n_selectors);

        for selector in &self.selectors {
            let scores = selector.compute_scores(x, None, Some(y))?;
            let decisions = self.scores_to_binary_decisions(&scores, n_features)?;
            selector_decisions.push(decisions);
        }

        // Apply consensus method
        let (consensus_scores, selected_features) = self.apply_consensus(&selector_decisions)?;

        Ok(ConsensusFeatureSelector {
            selectors: self.selectors,
            consensus_method: self.consensus_method,
            threshold_params: self.threshold_params,
            state: PhantomData,
            selector_decisions_: Some(selector_decisions),
            consensus_scores_: Some(consensus_scores),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl ConsensusFeatureSelector<Untrained> {
    /// Convert feature scores to binary selection decisions
    fn scores_to_binary_decisions(
        &self,
        scores: &Array1<f64>,
        n_features: usize,
    ) -> SklResult<Vec<bool>> {
        let mut decisions = vec![false; n_features];

        if let Some(k) = self.threshold_params.k_per_selector {
            // Select top k features
            let mut indexed_scores: Vec<(usize, f64)> = scores
                .indexed_iter()
                .map(|(idx, &score)| (idx, score))
                .collect();

            indexed_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, _) in indexed_scores.into_iter().take(k.min(n_features)) {
                decisions[idx] = true;
            }
        } else if let Some(percentile) = self.threshold_params.percentile_per_selector {
            // Select top percentile of features
            let k = ((n_features as f64) * percentile).ceil() as usize;
            let mut indexed_scores: Vec<(usize, f64)> = scores
                .indexed_iter()
                .map(|(idx, &score)| (idx, score))
                .collect();

            indexed_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, _) in indexed_scores.into_iter().take(k) {
                decisions[idx] = true;
            }
        } else if let Some(threshold) = self.threshold_params.score_threshold_per_selector {
            // Select features above threshold
            for (idx, &score) in scores.iter().enumerate() {
                if score >= threshold {
                    decisions[idx] = true;
                }
            }
        } else {
            // Default: select top 10 features
            let mut indexed_scores: Vec<(usize, f64)> = scores
                .indexed_iter()
                .map(|(idx, &score)| (idx, score))
                .collect();

            indexed_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            for (idx, _) in indexed_scores.into_iter().take(10.min(n_features)) {
                decisions[idx] = true;
            }
        }

        Ok(decisions)
    }

    /// Apply consensus method to selector decisions
    fn apply_consensus(
        &self,
        selector_decisions: &[Vec<bool>],
    ) -> SklResult<(Array1<f64>, Vec<usize>)> {
        let n_features = selector_decisions[0].len();
        let n_selectors = selector_decisions.len();

        let mut consensus_scores = Array1::<f64>::zeros(n_features);

        match &self.consensus_method {
            ConsensusMethod::MajorityVoting => {
                for i in 0..n_features {
                    let votes: usize = selector_decisions
                        .iter()
                        .map(|decisions| if decisions[i] { 1 } else { 0 })
                        .sum();
                    consensus_scores[i] = votes as f64 / n_selectors as f64;
                }
                // Select features with > 50% support
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score > 0.5 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::Unanimous => {
                for i in 0..n_features {
                    let all_agree = selector_decisions.iter().all(|decisions| decisions[i]);
                    consensus_scores[i] = if all_agree { 1.0 } else { 0.0 };
                }
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score == 1.0 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::SuperMajority(threshold) => {
                for i in 0..n_features {
                    let votes: usize = selector_decisions
                        .iter()
                        .map(|decisions| if decisions[i] { 1 } else { 0 })
                        .sum();
                    let support = votes as f64 / n_selectors as f64;
                    consensus_scores[i] = support;
                }
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score >= *threshold { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::WeightedVoting => {
                let weights = self
                    .threshold_params
                    .selector_weights
                    .as_ref()
                    .ok_or_else(|| {
                        SklearsError::InvalidInput(
                            "Weights required for weighted voting".to_string(),
                        )
                    })?;

                if weights.len() != n_selectors {
                    return Err(SklearsError::InvalidInput(
                        "Number of weights must match number of selectors".to_string(),
                    ));
                }

                let weight_sum: f64 = weights.iter().sum();

                for i in 0..n_features {
                    let weighted_votes: f64 = selector_decisions
                        .iter()
                        .zip(weights.iter())
                        .map(|(decisions, &weight)| if decisions[i] { weight } else { 0.0 })
                        .sum();
                    consensus_scores[i] = weighted_votes / weight_sum;
                }

                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score > 0.5 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::Intersection => {
                for i in 0..n_features {
                    let all_selected = selector_decisions.iter().all(|decisions| decisions[i]);
                    consensus_scores[i] = if all_selected { 1.0 } else { 0.0 };
                }
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score == 1.0 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::Union => {
                for i in 0..n_features {
                    let any_selected = selector_decisions.iter().any(|decisions| decisions[i]);
                    consensus_scores[i] = if any_selected { 1.0 } else { 0.0 };
                }
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score == 1.0 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::KOutOfN(k) => {
                for i in 0..n_features {
                    let votes: usize = selector_decisions
                        .iter()
                        .map(|decisions| if decisions[i] { 1 } else { 0 })
                        .sum();
                    consensus_scores[i] = votes as f64;
                }
                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score >= *k as f64 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::BordaConsensus => {
                // For Borda, we need the original scores, not just binary decisions
                // This is a simplified version using vote counts
                for i in 0..n_features {
                    let votes: usize = selector_decisions
                        .iter()
                        .map(|decisions| if decisions[i] { 1 } else { 0 })
                        .sum();
                    consensus_scores[i] = votes as f64;
                }

                // Select features in Borda order (top voted)
                let mut indexed_scores: Vec<(usize, f64)> = consensus_scores
                    .indexed_iter()
                    .map(|(idx, &score)| (idx, score))
                    .collect();

                indexed_scores
                    .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

                // Select top half by default
                let k = n_features / 2;
                let selected_features = indexed_scores
                    .into_iter()
                    .take(k)
                    .map(|(idx, _)| idx)
                    .collect();
                Ok((consensus_scores, selected_features))
            }

            ConsensusMethod::CondorcetConsensus => {
                // Simplified Condorcet: select features that beat majority in pairwise comparisons
                for i in 0..n_features {
                    let votes: usize = selector_decisions
                        .iter()
                        .map(|decisions| if decisions[i] { 1 } else { 0 })
                        .sum();
                    consensus_scores[i] = votes as f64 / n_selectors as f64;
                }

                let selected_features = consensus_scores
                    .indexed_iter()
                    .filter_map(|(idx, &score)| if score > 0.5 { Some(idx) } else { None })
                    .collect();
                Ok((consensus_scores, selected_features))
            }
        }
    }
}

impl ConsensusFeatureSelector<Trained> {
    /// Get the consensus scores for all features
    pub fn consensus_scores(&self) -> &Array1<f64> {
        self.consensus_scores_.as_ref().unwrap()
    }

    /// Get the final selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get individual selector decisions
    pub fn selector_decisions(&self) -> &[Vec<bool>] {
        self.selector_decisions_.as_ref().unwrap()
    }

    /// Get support mask for selected features
    pub fn get_support_mask(&self) -> Vec<bool> {
        let n_features = self.n_features_.unwrap();
        let mut support = vec![false; n_features];

        for &idx in self.selected_features() {
            support[idx] = true;
        }

        support
    }

    /// Get agreement matrix between selectors
    pub fn selector_agreement_matrix(&self) -> Array2<f64> {
        let decisions = self.selector_decisions_.as_ref().unwrap();
        let n_selectors = decisions.len();
        let n_features = decisions[0].len();

        let mut agreement_matrix = Array2::<f64>::zeros((n_selectors, n_selectors));

        for i in 0..n_selectors {
            for j in 0..n_selectors {
                if i == j {
                    agreement_matrix[[i, j]] = 1.0;
                } else {
                    let mut agreements = 0;
                    for k in 0..n_features {
                        if decisions[i][k] == decisions[j][k] {
                            agreements += 1;
                        }
                    }
                    agreement_matrix[[i, j]] = agreements as f64 / n_features as f64;
                }
            }
        }

        agreement_matrix
    }
}

impl Transform<Array2<f64>> for ConsensusFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let selected = self.selected_features();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut x_new = Array2::<f64>::zeros((n_samples, n_selected));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ConsensusFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        Ok(Array1::from(self.get_support_mask()))
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.selected_features();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Multi-task feature selection
/// Selects features that are relevant across multiple related tasks simultaneously
#[derive(Debug, Clone)]
pub struct MultiTaskFeatureSelector<State = Untrained> {
    n_features_to_select: usize,
    task_weights: Option<Vec<f64>>,
    regularization: f64,
    max_iter: usize,
    tolerance: f64,
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    selected_features_: Option<Vec<usize>>,
    feature_scores_: Option<Array2<f64>>, // scores per task
    task_specific_scores_: Option<Vec<Array1<f64>>>, // task-specific feature importance
    n_features_: Option<usize>,
    n_tasks_: Option<usize>,
}

impl MultiTaskFeatureSelector<Untrained> {
    /// Create a new multi-task feature selector
    pub fn new() -> Self {
        Self {
            n_features_to_select: 10,
            task_weights: None,
            regularization: 0.1,
            max_iter: 1000,
            tolerance: 1e-4,
            random_state: None,
            state: PhantomData,
            selected_features_: None,
            feature_scores_: None,
            task_specific_scores_: None,
            n_features_: None,
            n_tasks_: None,
        }
    }

    /// Set the number of features to select
    pub fn n_features_to_select(mut self, n_features: usize) -> Self {
        self.n_features_to_select = n_features;
        self
    }

    /// Set task weights (if None, all tasks are weighted equally)
    pub fn task_weights(mut self, weights: Option<Vec<f64>>) -> Self {
        self.task_weights = weights;
        self
    }

    /// Set regularization strength for task coupling
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg.max(0.0);
        self
    }

    /// Set maximum iterations for optimization
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Default for MultiTaskFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTaskFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<f64>, Vec<Array1<f64>>> for MultiTaskFeatureSelector<Untrained> {
    type Fitted = MultiTaskFeatureSelector<Trained>;

    fn fit(self, x: &Array2<f64>, y_tasks: &Vec<Array1<f64>>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let n_tasks = y_tasks.len();

        if n_tasks == 0 {
            return Err(SklearsError::InvalidInput(
                "At least one task is required".to_string(),
            ));
        }

        // Validate that all tasks have the same number of samples
        for (i, y_task) in y_tasks.iter().enumerate() {
            if y_task.len() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Task {} has {} samples, expected {}",
                    i,
                    y_task.len(),
                    n_samples
                )));
            }
        }

        if self.n_features_to_select > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Cannot select {} features from {} available",
                self.n_features_to_select, n_features
            )));
        }

        // Set task weights
        let task_weights = self
            .task_weights
            .clone()
            .unwrap_or_else(|| vec![1.0 / n_tasks as f64; n_tasks]);

        if task_weights.len() != n_tasks {
            return Err(SklearsError::InvalidInput(format!(
                "Task weights length {} doesn't match number of tasks {}",
                task_weights.len(),
                n_tasks
            )));
        }

        // Calculate feature scores for each task
        let mut task_scores = Vec::with_capacity(n_tasks);
        let mut feature_scores = Array2::<f64>::zeros((n_features, n_tasks));

        for (task_idx, y_task) in y_tasks.iter().enumerate() {
            let mut scores = Array1::<f64>::zeros(n_features);

            // Calculate correlation-based scores for each feature
            for feature_idx in 0..n_features {
                let feature_col = x.column(feature_idx).to_owned();
                let correlation = compute_pearson_correlation(&feature_col, y_task).abs();
                scores[feature_idx] = correlation;
            }

            // Store task-specific scores
            for (feature_idx, &score) in scores.iter().enumerate() {
                feature_scores[[feature_idx, task_idx]] = score;
            }

            task_scores.push(scores);
        }

        // Multi-task feature selection using joint optimization
        let selected_features = self.select_multi_task_features(
            &feature_scores,
            &task_weights,
            self.n_features_to_select,
        )?;

        Ok(MultiTaskFeatureSelector {
            n_features_to_select: self.n_features_to_select,
            task_weights: Some(task_weights),
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            state: PhantomData,
            selected_features_: Some(selected_features),
            feature_scores_: Some(feature_scores),
            task_specific_scores_: Some(task_scores),
            n_features_: Some(n_features),
            n_tasks_: Some(n_tasks),
        })
    }
}

impl MultiTaskFeatureSelector<Untrained> {
    /// Select features using multi-task optimization
    fn select_multi_task_features(
        &self,
        feature_scores: &Array2<f64>,
        task_weights: &[f64],
        n_select: usize,
    ) -> SklResult<Vec<usize>> {
        let (n_features, n_tasks) = feature_scores.dim();

        // Calculate weighted average scores across tasks
        let mut weighted_scores = Array1::<f64>::zeros(n_features);

        for feature_idx in 0..n_features {
            let mut weighted_score = 0.0;

            for task_idx in 0..n_tasks {
                weighted_score += task_weights[task_idx] * feature_scores[[feature_idx, task_idx]];
            }

            // Add regularization term to encourage feature sharing across tasks
            let mut task_consistency = 0.0;
            for task_idx in 0..n_tasks {
                for other_task_idx in 0..n_tasks {
                    if task_idx != other_task_idx {
                        let diff = feature_scores[[feature_idx, task_idx]]
                            - feature_scores[[feature_idx, other_task_idx]];
                        task_consistency -= self.regularization * diff * diff; // Penalty for inconsistency
                    }
                }
            }
            task_consistency /= (n_tasks * (n_tasks - 1)) as f64; // Normalize

            weighted_scores[feature_idx] = weighted_score + task_consistency;
        }

        // Select top features based on weighted scores
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            weighted_scores[b]
                .partial_cmp(&weighted_scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(feature_indices.into_iter().take(n_select).collect())
    }
}

impl MultiTaskFeatureSelector<Trained> {
    /// Get selected features
    pub fn get_selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }

    /// Get feature scores for all tasks
    pub fn get_feature_scores(&self) -> &Array2<f64> {
        self.feature_scores_.as_ref().unwrap()
    }

    /// Get task-specific feature scores
    pub fn get_task_scores(&self) -> &[Array1<f64>] {
        self.task_specific_scores_.as_ref().unwrap()
    }

    /// Get the most important features for a specific task
    pub fn get_task_specific_features(
        &self,
        task_idx: usize,
        n_features: usize,
    ) -> SklResult<Vec<usize>> {
        let task_scores = self.task_specific_scores_.as_ref().unwrap();

        if task_idx >= task_scores.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Task index {} out of range",
                task_idx
            )));
        }

        let scores = &task_scores[task_idx];
        let mut feature_indices: Vec<usize> = (0..scores.len()).collect();
        feature_indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(feature_indices.into_iter().take(n_features).collect())
    }

    /// Get features that are consistently important across all tasks
    pub fn get_consensus_features(&self, threshold: f64) -> SklResult<Vec<usize>> {
        let feature_scores = self.feature_scores_.as_ref().unwrap();
        let (n_features, n_tasks) = feature_scores.dim();
        let mut consensus_features = Vec::new();

        for feature_idx in 0..n_features {
            let mut above_threshold_count = 0;

            for task_idx in 0..n_tasks {
                if feature_scores[[feature_idx, task_idx]] >= threshold {
                    above_threshold_count += 1;
                }
            }

            // Feature is consensus if it's above threshold for all tasks
            if above_threshold_count == n_tasks {
                consensus_features.push(feature_idx);
            }
        }

        Ok(consensus_features)
    }

    /// Get task similarity matrix based on feature importance patterns
    pub fn get_task_similarity(&self) -> SklResult<Array2<f64>> {
        let feature_scores = self.feature_scores_.as_ref().unwrap();
        let (_, n_tasks) = feature_scores.dim();
        let mut similarity_matrix = Array2::<f64>::zeros((n_tasks, n_tasks));

        for i in 0..n_tasks {
            for j in 0..n_tasks {
                if i == j {
                    similarity_matrix[[i, j]] = 1.0;
                } else {
                    let task_i_scores = feature_scores.column(i).to_owned();
                    let task_j_scores = feature_scores.column(j).to_owned();
                    let correlation = compute_pearson_correlation(&task_i_scores, &task_j_scores);
                    similarity_matrix[[i, j]] = correlation.abs();
                }
            }
        }

        Ok(similarity_matrix)
    }
}

impl Transform<Array2<f64>> for MultiTaskFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_features = self.n_features_.unwrap();
        if x.ncols() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                n_features,
                x.ncols()
            )));
        }

        let selected = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected.len();

        let mut x_new = Array2::<f64>::zeros((n_samples, n_selected));
        for (new_idx, &old_idx) in selected.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for MultiTaskFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.n_features_.unwrap();
        let mut support = vec![false; n_features];

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(Array1::from(support))
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Helper function to compute Pearson correlation between two arrays
fn compute_pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = x.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

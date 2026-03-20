//! Multi-Label Ensemble Methods
//!
//! This module provides ensemble methods specifically designed for multi-label classification
//! where each instance can belong to multiple classes simultaneously. It includes various
//! label transformation strategies, ensemble voting mechanisms, and performance optimizations.

use crate::bagging::BaggingClassifier;
// ❌ REMOVED: rand_chacha::rand_core - use scirs2_core::random instead
// ❌ REMOVED: rand_chacha::scirs2_core::random::rngs::StdRng - use scirs2_core::random instead
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
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

/// Configuration for multi-label ensemble methods
#[derive(Debug, Clone)]
pub struct MultiLabelEnsembleConfig {
    /// Number of base estimators
    pub n_estimators: usize,
    /// Multi-label transformation strategy
    pub transformation_strategy: LabelTransformationStrategy,
    /// Ensemble aggregation method for multi-label predictions
    pub aggregation_method: MultiLabelAggregationMethod,
    /// Label correlation handling approach
    pub correlation_method: LabelCorrelationMethod,
    /// Threshold for binary relevance predictions
    pub threshold: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use label powerset pruning
    pub prune_labelsets: bool,
    /// Maximum number of labelsets to consider
    pub max_labelsets: Option<usize>,
    /// Label dependency order for classifier chains
    pub chain_order: Option<Vec<usize>>,
    /// Whether to use ensemble of chains
    pub ensemble_chains: bool,
    /// Number of chains in ensemble chains
    pub n_chains: usize,
}

impl Default for MultiLabelEnsembleConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            transformation_strategy: LabelTransformationStrategy::BinaryRelevance,
            aggregation_method: MultiLabelAggregationMethod::Voting,
            correlation_method: LabelCorrelationMethod::Independent,
            threshold: 0.5,
            random_state: None,
            prune_labelsets: true,
            max_labelsets: Some(100),
            chain_order: None,
            ensemble_chains: false,
            n_chains: 3,
        }
    }
}

/// Label transformation strategies for multi-label classification
#[derive(Debug, Clone, PartialEq)]
pub enum LabelTransformationStrategy {
    /// Binary Relevance - train one classifier per label
    BinaryRelevance,
    /// Label Powerset - treat each unique label combination as a class
    LabelPowerset,
    /// Classifier Chains - chain classifiers to model label dependencies
    ClassifierChains,
    /// Ensemble of Classifier Chains
    EnsembleOfClassifierChains,
    /// Adapted Algorithm - adapt base algorithm for multi-label
    AdaptedAlgorithm,
    /// Random k-labelsets - random subset of label combinations
    RandomKLabelsets,
}

/// Multi-label aggregation methods
#[derive(Debug, Clone, PartialEq)]
pub enum MultiLabelAggregationMethod {
    /// Simple voting across base estimators
    Voting,
    /// Weighted voting based on estimator performance
    WeightedVoting,
    /// Maximum probability across estimators
    MaxProbability,
    /// Mean probability across estimators
    MeanProbability,
    /// Median probability across estimators
    MedianProbability,
    /// Threshold-based aggregation
    ThresholdAggregation,
    /// Rank-based aggregation
    RankAggregation,
}

/// Label correlation handling methods
#[derive(Debug, Clone, PartialEq)]
pub enum LabelCorrelationMethod {
    /// Treat labels as independent
    Independent,
    /// Model pairwise label correlations
    Pairwise,
    /// Model higher-order label correlations
    HigherOrder,
    /// Use conditional independence assumptions
    ConditionalIndependence,
    /// Learn label correlation structure
    LearnedCorrelation,
}

/// Multi-label ensemble classifier
pub struct MultiLabelEnsembleClassifier<State = Untrained> {
    config: MultiLabelEnsembleConfig,
    state: std::marker::PhantomData<State>,
    // Fitted attributes - only populated after training
    base_classifiers: Option<Vec<BaggingClassifier<Trained>>>,
    label_indices: Option<Vec<usize>>,
    labelset_mapping: Option<HashMap<Vec<usize>, usize>>,
    inverse_labelset_mapping: Option<HashMap<usize, Vec<usize>>>,
    label_correlations: Option<Array2<f64>>,
    chain_orders: Option<Vec<Vec<usize>>>,
    threshold_per_label: Option<Vec<f64>>,
    n_labels: Option<usize>,
}

/// Results from multi-label ensemble training
#[derive(Debug, Clone)]
pub struct MultiLabelTrainingResults {
    /// Number of unique labelsets found
    pub n_labelsets: usize,
    /// Label frequency distribution
    pub label_frequencies: HashMap<usize, usize>,
    /// Labelset frequency distribution
    pub labelset_frequencies: HashMap<Vec<usize>, usize>,
    /// Label correlation matrix
    pub label_correlations: Array2<f64>,
    /// Training time metrics
    pub training_time_ms: u64,
}

/// Multi-label prediction results
#[derive(Debug, Clone)]
pub struct MultiLabelPredictionResults {
    /// Binary predictions for each label
    pub predictions: Array2<usize>,
    /// Probability scores for each label
    pub probabilities: Array2<f64>,
    /// Confidence scores for predictions
    pub confidence_scores: Vec<f64>,
    /// Label ranking scores
    pub ranking_scores: Array2<f64>,
}

impl MultiLabelEnsembleClassifier<Untrained> {
    /// Create a new multi-label ensemble classifier
    pub fn new(config: MultiLabelEnsembleConfig) -> Self {
        Self {
            config,
            state: std::marker::PhantomData,
            base_classifiers: None,
            label_indices: None,
            labelset_mapping: None,
            inverse_labelset_mapping: None,
            label_correlations: None,
            chain_orders: None,
            threshold_per_label: None,
            n_labels: None,
        }
    }

    /// Create a new multi-label ensemble classifier with binary relevance
    pub fn binary_relevance() -> Self {
        let config = MultiLabelEnsembleConfig {
            transformation_strategy: LabelTransformationStrategy::BinaryRelevance,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new multi-label ensemble classifier with label powerset
    pub fn label_powerset() -> Self {
        let config = MultiLabelEnsembleConfig {
            transformation_strategy: LabelTransformationStrategy::LabelPowerset,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new multi-label ensemble classifier with classifier chains
    pub fn classifier_chains() -> Self {
        let config = MultiLabelEnsembleConfig {
            transformation_strategy: LabelTransformationStrategy::ClassifierChains,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a new multi-label ensemble classifier with ensemble of classifier chains
    pub fn ensemble_classifier_chains() -> Self {
        let config = MultiLabelEnsembleConfig {
            transformation_strategy: LabelTransformationStrategy::EnsembleOfClassifierChains,
            ensemble_chains: true,
            n_chains: 5,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Builder method to configure the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Builder method to configure the aggregation method
    pub fn aggregation_method(mut self, method: MultiLabelAggregationMethod) -> Self {
        self.config.aggregation_method = method;
        self
    }

    /// Builder method to configure the correlation method
    pub fn correlation_method(mut self, method: LabelCorrelationMethod) -> Self {
        self.config.correlation_method = method;
        self
    }

    /// Builder method to configure the threshold
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.config.threshold = threshold;
        self
    }

    /// Builder method to configure random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Builder method to configure label powerset pruning
    pub fn prune_labelsets(mut self, prune: bool) -> Self {
        self.config.prune_labelsets = prune;
        self
    }

    /// Extract unique labelsets from multi-label target matrix
    fn extract_labelsets(
        &self,
        y: &Array2<usize>,
    ) -> SklResult<(HashMap<Vec<usize>, usize>, HashMap<usize, Vec<usize>>)> {
        let mut labelset_mapping = HashMap::new();
        let mut inverse_mapping = HashMap::new();
        let mut labelset_id = 0;

        for row in y.outer_iter() {
            let labelset: Vec<usize> = row
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == 1)
                .map(|(idx, _)| idx)
                .collect();

            if !labelset_mapping.contains_key(&labelset) {
                labelset_mapping.insert(labelset.clone(), labelset_id);
                inverse_mapping.insert(labelset_id, labelset);
                labelset_id += 1;
            }
        }

        // Prune rare labelsets if configured
        if self.config.prune_labelsets {
            if let Some(max_labelsets) = self.config.max_labelsets {
                if labelset_mapping.len() > max_labelsets {
                    // Keep only the most frequent labelsets
                    let mut labelset_counts: Vec<_> = labelset_mapping.iter().collect();
                    labelset_counts.sort_by_key(|(labelset, _)| labelset.len());
                    labelset_counts.truncate(max_labelsets);

                    labelset_mapping = labelset_counts
                        .into_iter()
                        .enumerate()
                        .map(|(new_id, (labelset, _))| (labelset.clone(), new_id))
                        .collect();

                    inverse_mapping = labelset_mapping
                        .iter()
                        .map(|(labelset, &id)| (id, labelset.clone()))
                        .collect();
                }
            }
        }

        Ok((labelset_mapping, inverse_mapping))
    }

    /// Compute label correlations
    fn compute_label_correlations(&self, y: &Array2<usize>) -> SklResult<Array2<f64>> {
        let n_labels = y.ncols();
        let mut correlations = Array2::zeros((n_labels, n_labels));

        for i in 0..n_labels {
            for j in i..n_labels {
                if i == j {
                    correlations[[i, j]] = 1.0;
                } else {
                    // Compute Jaccard similarity
                    let mut intersection = 0;
                    let mut union = 0;

                    for k in 0..y.nrows() {
                        let label_i = y[[k, i]];
                        let label_j = y[[k, j]];

                        if label_i == 1 && label_j == 1 {
                            intersection += 1;
                        }
                        if label_i == 1 || label_j == 1 {
                            union += 1;
                        }
                    }

                    let correlation = if union > 0 {
                        intersection as f64 / union as f64
                    } else {
                        0.0
                    };

                    correlations[[i, j]] = correlation;
                    correlations[[j, i]] = correlation;
                }
            }
        }

        Ok(correlations)
    }

    /// Generate chain orders for classifier chains
    fn generate_chain_orders(&self, n_labels: usize, n_chains: usize) -> Vec<Vec<usize>> {
        let mut chains = Vec::new();
        let mut rng = if let Some(seed) = self.config.random_state {
            scirs2_core::random::seeded_rng(seed)
        } else {
            scirs2_core::random::seeded_rng(42)
        };

        for _ in 0..n_chains {
            let mut order: Vec<usize> = (0..n_labels).collect();

            // Shuffle the order
            for i in (1..order.len()).rev() {
                let j = gen_range_usize(&mut rng, 0..(i + 1));
                order.swap(i, j);
            }

            chains.push(order);
        }

        chains
    }
}

impl Estimator for MultiLabelEnsembleClassifier<Untrained> {
    type Config = MultiLabelEnsembleConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array2<usize>> for MultiLabelEnsembleClassifier<Untrained> {
    type Fitted = MultiLabelEnsembleClassifier<Trained>;

    fn fit(self, X: &Array2<f64>, y: &Array2<usize>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", X.nrows()),
                actual: format!("{} samples", y.nrows()),
            });
        }

        let n_labels = y.ncols();
        let mut base_classifiers = Vec::new();
        let mut chain_orders = Vec::new();

        // Compute label correlations
        let label_correlations = self.compute_label_correlations(y)?;

        match self.config.transformation_strategy {
            LabelTransformationStrategy::BinaryRelevance => {
                // Train one classifier per label
                for label_idx in 0..n_labels {
                    let y_binary: Vec<usize> = y.column(label_idx).to_vec();

                    let y_binary_array =
                        Array1::from_vec(y_binary.iter().map(|&x| x as i32).collect());
                    let classifier = BaggingClassifier::new()
                        .n_estimators(self.config.n_estimators)
                        .fit(X, &y_binary_array)?;

                    base_classifiers.push(classifier);
                }
            }

            LabelTransformationStrategy::LabelPowerset => {
                // Extract labelsets and train single multi-class classifier
                let (labelset_mapping, _) = self.extract_labelsets(y)?;

                // Convert multi-label matrix to single-label vector
                let mut y_labelsets = Vec::new();
                for row in y.outer_iter() {
                    let labelset: Vec<usize> = row
                        .iter()
                        .enumerate()
                        .filter(|(_, &label)| label == 1)
                        .map(|(idx, _)| idx)
                        .collect();

                    if let Some(&labelset_id) = labelset_mapping.get(&labelset) {
                        y_labelsets.push(labelset_id);
                    } else {
                        // Handle unseen labelsets (assign to empty set)
                        y_labelsets.push(0);
                    }
                }

                let y_labelsets_array =
                    Array1::from_vec(y_labelsets.iter().map(|&x| x as i32).collect());
                let classifier = BaggingClassifier::new()
                    .n_estimators(self.config.n_estimators)
                    .fit(X, &y_labelsets_array)?;

                base_classifiers.push(classifier);
            }

            LabelTransformationStrategy::EnsembleOfClassifierChains => {
                // Generate multiple chain orders
                chain_orders = self.generate_chain_orders(n_labels, self.config.n_chains);

                for chain_order in &chain_orders {
                    // Train chain of classifiers for this order
                    for &label_idx in chain_order {
                        let y_binary: Vec<usize> = y.column(label_idx).to_vec();

                        let y_binary_array =
                            Array1::from_vec(y_binary.iter().map(|&x| x as i32).collect());
                        let classifier = BaggingClassifier::new()
                            .n_estimators(5) // Fewer estimators per chain classifier
                            .fit(X, &y_binary_array)?;

                        base_classifiers.push(classifier);
                    }
                }
            }

            _ => {
                // Default to binary relevance for other strategies
                for label_idx in 0..n_labels {
                    let y_binary: Vec<usize> = y.column(label_idx).to_vec();

                    let y_binary_array =
                        Array1::from_vec(y_binary.iter().map(|&x| x as i32).collect());
                    let classifier = BaggingClassifier::new()
                        .n_estimators(self.config.n_estimators)
                        .fit(X, &y_binary_array)?;

                    base_classifiers.push(classifier);
                }
            }
        }

        // Create label indices
        let label_indices: Vec<usize> = (0..n_labels).collect();

        // Compute per-label thresholds (simplified for now)
        let threshold_per_label = vec![self.config.threshold; n_labels];

        // Extract labelsets if using label powerset
        let (labelset_mapping, inverse_labelset_mapping) = if matches!(
            self.config.transformation_strategy,
            LabelTransformationStrategy::LabelPowerset
        ) {
            let (forward, inverse) = self.extract_labelsets(y)?;
            (Some(forward), Some(inverse))
        } else {
            (None, None)
        };

        Ok(MultiLabelEnsembleClassifier {
            config: self.config,
            state: std::marker::PhantomData,
            base_classifiers: Some(base_classifiers),
            label_indices: Some(label_indices),
            labelset_mapping,
            inverse_labelset_mapping,
            label_correlations: Some(label_correlations),
            chain_orders: if chain_orders.is_empty() {
                None
            } else {
                Some(chain_orders)
            },
            threshold_per_label: Some(threshold_per_label),
            n_labels: Some(n_labels),
        })
    }
}

impl Predict<Array2<f64>, MultiLabelPredictionResults> for MultiLabelEnsembleClassifier<Trained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<MultiLabelPredictionResults> {
        let base_classifiers = self.base_classifiers.as_ref().expect("Model is trained");
        let n_labels = self.n_labels.expect("Model is trained");
        let threshold_per_label = self.threshold_per_label.as_ref().expect("Model is trained");

        let n_samples = X.nrows();
        let mut predictions = Array2::zeros((n_samples, n_labels));
        let mut probabilities = Array2::zeros((n_samples, n_labels));
        let mut ranking_scores = Array2::zeros((n_samples, n_labels));

        match self.config.transformation_strategy {
            LabelTransformationStrategy::BinaryRelevance => {
                // Get predictions from each binary classifier
                for (label_idx, classifier) in base_classifiers.iter().enumerate().take(n_labels) {
                    let label_predictions = classifier.predict(X)?;

                    // Convert predictions to the right format for each sample
                    for (sample_idx, &pred) in label_predictions.iter().enumerate() {
                        predictions[[sample_idx, label_idx]] = pred as usize;
                        probabilities[[sample_idx, label_idx]] = pred as f64;
                        ranking_scores[[sample_idx, label_idx]] = pred as f64;
                    }
                }
            }

            LabelTransformationStrategy::LabelPowerset => {
                if let (Some(labelset_mapping), Some(inverse_labelset_mapping)) =
                    (&self.labelset_mapping, &self.inverse_labelset_mapping)
                {
                    let labelset_predictions = base_classifiers[0].predict(X)?;

                    for (sample_idx, &labelset_id) in labelset_predictions.iter().enumerate() {
                        if let Some(labelset) =
                            inverse_labelset_mapping.get(&(labelset_id as usize))
                        {
                            for &label_idx in labelset {
                                if label_idx < n_labels {
                                    predictions[[sample_idx, label_idx]] = 1;
                                    probabilities[[sample_idx, label_idx]] = 1.0;
                                    ranking_scores[[sample_idx, label_idx]] = 1.0;
                                }
                            }
                        }
                    }
                }
            }

            _ => {
                // Default binary relevance behavior
                for (label_idx, classifier) in base_classifiers.iter().enumerate().take(n_labels) {
                    let label_predictions = classifier.predict(X)?;

                    for (sample_idx, &pred) in label_predictions.iter().enumerate() {
                        predictions[[sample_idx, label_idx]] = pred as usize;
                        probabilities[[sample_idx, label_idx]] = pred as f64;
                        ranking_scores[[sample_idx, label_idx]] = pred as f64;
                    }
                }
            }
        }

        // Apply thresholds
        for i in 0..n_samples {
            for j in 0..n_labels {
                if probabilities[[i, j]] >= threshold_per_label[j] {
                    predictions[[i, j]] = 1;
                } else {
                    predictions[[i, j]] = 0;
                }
            }
        }

        // Compute confidence scores (simplified)
        let confidence_scores: Vec<f64> = (0..n_samples)
            .map(|i| {
                let row_probs: Vec<f64> = (0..n_labels).map(|j| probabilities[[i, j]]).collect();
                row_probs.iter().sum::<f64>() / n_labels as f64
            })
            .collect();

        Ok(MultiLabelPredictionResults {
            predictions,
            probabilities,
            confidence_scores,
            ranking_scores,
        })
    }
}

impl MultiLabelEnsembleClassifier<Trained> {
    /// Get the number of labels
    pub fn n_labels(&self) -> usize {
        self.n_labels.expect("Model is trained")
    }

    /// Get label correlations
    pub fn label_correlations(&self) -> &Array2<f64> {
        self.label_correlations.as_ref().expect("Model is trained")
    }

    /// Get the transformation strategy used
    pub fn transformation_strategy(&self) -> &LabelTransformationStrategy {
        &self.config.transformation_strategy
    }

    /// Predict binary labels only
    pub fn predict_binary(&self, X: &Array2<f64>) -> SklResult<Array2<usize>> {
        let results = self.predict(X)?;
        Ok(results.predictions)
    }

    /// Predict probabilities only
    pub fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let results = self.predict(X)?;
        Ok(results.probabilities)
    }

    /// Get label rankings
    pub fn predict_rankings(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let results = self.predict(X)?;
        Ok(results.ranking_scores)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_multi_label_binary_relevance() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]];

        let classifier = MultiLabelEnsembleClassifier::binary_relevance()
            .n_estimators(3)
            .random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let results = trained.predict(&X).expect("Prediction should succeed");

        assert_eq!(results.predictions.nrows(), 4);
        assert_eq!(results.predictions.ncols(), 3);
        assert_eq!(results.probabilities.nrows(), 4);
        assert_eq!(results.probabilities.ncols(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_multi_label_label_powerset() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]];

        let classifier = MultiLabelEnsembleClassifier::label_powerset()
            .n_estimators(5)
            .random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let results = trained.predict(&X).expect("Prediction should succeed");

        assert_eq!(results.predictions.nrows(), 4);
        assert_eq!(results.predictions.ncols(), 3);
        assert_eq!(trained.n_labels(), 3);
    }

    #[test]
    fn test_label_correlation_computation() {
        let y = array![[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]];

        let classifier = MultiLabelEnsembleClassifier::binary_relevance();
        let correlations = classifier
            .compute_label_correlations(&y)
            .expect("Should compute correlations");

        assert_eq!(correlations.nrows(), 3);
        assert_eq!(correlations.ncols(), 3);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert_eq!(correlations[[i, i]], 1.0);
        }
    }

    #[test]
    fn test_labelset_extraction() {
        let y = array![
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 0, 1] // Duplicate labelset
        ];

        let classifier = MultiLabelEnsembleClassifier::label_powerset();
        let (labelset_mapping, inverse_mapping) = classifier
            .extract_labelsets(&y)
            .expect("Should extract labelsets");

        // Should have 3 unique labelsets
        assert_eq!(labelset_mapping.len(), 3);
        assert_eq!(inverse_mapping.len(), 3);

        // Check specific labelsets
        assert!(labelset_mapping.contains_key(&vec![0, 2])); // [1, 0, 1]
        assert!(labelset_mapping.contains_key(&vec![1, 2])); // [0, 1, 1]
        assert!(labelset_mapping.contains_key(&vec![0, 1])); // [1, 1, 0]
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_ensemble_classifier_chains() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 1]];

        let classifier =
            MultiLabelEnsembleClassifier::ensemble_classifier_chains().random_state(42);

        let trained = classifier.fit(&X, &y).expect("Training should succeed");
        let results = trained.predict(&X).expect("Prediction should succeed");

        assert_eq!(results.predictions.nrows(), 4);
        assert_eq!(results.predictions.ncols(), 3);

        // Should have generated chain orders
        assert!(trained.chain_orders.is_some());
    }
}

//! One-vs-One multiclass strategy
//!
//! This module implements the One-vs-One strategy for multiclass classification.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for One-vs-One Classifier
#[derive(Debug, Clone, Default)]
pub struct OneVsOneConfig {
    /// Number of parallel jobs (-1 for all cores)
    pub n_jobs: Option<i32>,
}

/// One-vs-One Classifier
///
/// This strategy consists in fitting one classifier per class pair. At prediction time,
/// the class which received the most votes is selected. Since it requires to fit
/// n_classes * (n_classes - 1) / 2 classifiers, this method is usually slower than
/// one-vs-the-rest, due to its O(n_classes^2) complexity.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::OneVsOneClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let ovo = OneVsOneClassifier::new(base_classifier);
/// ```
#[derive(Debug)]
pub struct OneVsOneClassifier<C, S = Untrained> {
    base_estimator: C,
    config: OneVsOneConfig,
    state: PhantomData<S>,
}

impl<C> OneVsOneClassifier<C, Untrained> {
    /// Create a new OneVsOneClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: OneVsOneConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for OneVsOneClassifier
    pub fn builder(base_estimator: C) -> OneVsOneBuilder<C> {
        OneVsOneBuilder::new(base_estimator)
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

impl<C: Clone> Clone for OneVsOneClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for OneVsOneClassifier<C, Untrained> {
    type Config = OneVsOneConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Builder for OneVsOneClassifier
#[derive(Debug)]
pub struct OneVsOneBuilder<C> {
    base_estimator: C,
    config: OneVsOneConfig,
}

impl<C> OneVsOneBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: OneVsOneConfig::default(),
        }
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable parallel training using all available cores
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }

    /// Disable parallel training (use sequential training)
    pub fn sequential(mut self) -> Self {
        self.config.n_jobs = None;
        self
    }

    /// Build the OneVsOneClassifier
    pub fn build(self) -> OneVsOneClassifier<C, Untrained> {
        OneVsOneClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for OneVsOneBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

// Trained data container for OneVsOneClassifier
#[derive(Debug)]
pub struct OneVsOneTrainedData<T> {
    estimators: Vec<T>,
    classes: Array1<i32>,
    class_pairs: Vec<(i32, i32)>,
    n_features: usize,
}

impl<T: Clone> Clone for OneVsOneTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            class_pairs: self.class_pairs.clone(),
            n_features: self.n_features,
        }
    }
}

pub type TrainedOneVsOne<T> = OneVsOneClassifier<OneVsOneTrainedData<T>, Trained>;

/// Implementation for classifiers that can fit binary problems
impl<C> Fit<Array2<Float>, Array1<i32>> for OneVsOneClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    type Fitted = TrainedOneVsOne<C::Fitted>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let (_n_samples, n_features) = x.dim();

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multiclass classification".to_string(),
            ));
        }

        let n_classes = classes.len();
        let mut class_pairs = Vec::new();
        let mut binary_problems = Vec::new();

        // Create all class pairs and binary problems
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = classes[i];
                let class_j = classes[j];
                let pair = (class_i, class_j);
                class_pairs.push(pair);

                // Create binary dataset for this pair
                let mut pair_indices = Vec::new();
                for (sample_idx, &label) in y.iter().enumerate() {
                    if label == class_i || label == class_j {
                        pair_indices.push(sample_idx);
                    }
                }

                if !pair_indices.is_empty() {
                    // Extract samples for this pair
                    let pair_x =
                        Array2::from_shape_fn((pair_indices.len(), n_features), |(i, j)| {
                            x[[pair_indices[i], j]]
                        });

                    let pair_y: Array1<Float> = Array1::from_shape_fn(pair_indices.len(), |i| {
                        if y[pair_indices[i]] == class_i {
                            1.0
                        } else {
                            0.0
                        }
                    });

                    binary_problems.push((pair, pair_x, pair_y));
                }
            }
        }

        // Train binary classifiers (with parallel support if enabled)
        let estimators: SklResult<Vec<_>> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            binary_problems
                .into_par_iter()
                .map(|(_, pair_x, pair_y)| self.base_estimator.clone().fit(&pair_x, &pair_y))
                .collect::<SklResult<Vec<_>>>()
        } else {
            // Sequential training
            binary_problems
                .into_iter()
                .map(|(_, pair_x, pair_y)| self.base_estimator.clone().fit(&pair_x, &pair_y))
                .collect::<SklResult<Vec<_>>>()
        };

        let estimators = estimators?;

        Ok(OneVsOneClassifier {
            base_estimator: OneVsOneTrainedData {
                estimators,
                classes: Array1::from(classes),
                class_pairs,
                n_features,
            },
            config: self.config,
            state: PhantomData,
        })
    }
}

/// Voting strategy for OneVsOne
#[derive(Debug, Clone, PartialEq, Default)]
pub enum VotingStrategy {
    /// Simple majority voting
    #[default]
    Majority,
    /// Rank-based voting with confidence weighting
    Rank,
    /// Distance-based voting using prediction scores
    Distance,
    /// Consensus-based voting combining multiple strategies
    Consensus(ConsensusConfig),
}

/// Configuration for consensus-based voting
#[derive(Debug, Clone, PartialEq)]
pub struct ConsensusConfig {
    /// Strategies to combine
    pub strategies: Vec<ConsensusStrategy>,
    /// Combination method
    pub method: ConsensusMethod,
    /// Minimum agreement threshold (0.0 to 1.0)
    pub agreement_threshold: f64,
}

impl Default for ConsensusConfig {
    fn default() -> Self {
        Self {
            strategies: vec![
                ConsensusStrategy::Majority,
                ConsensusStrategy::Rank,
                ConsensusStrategy::Distance,
            ],
            method: ConsensusMethod::WeightedAverage,
            agreement_threshold: 0.6,
        }
    }
}

/// Individual strategies for consensus
#[derive(Debug, Clone, PartialEq)]
pub enum ConsensusStrategy {
    /// Majority
    Majority,
    /// Rank
    Rank,
    /// Distance
    Distance,
}

/// Methods for combining consensus strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum ConsensusMethod {
    /// Simple average of normalized scores
    Average,
    /// Weighted average with strategy-specific weights
    #[default]
    WeightedAverage,
    /// Majority vote among strategies
    MajorityVote,
    /// Borda count ranking
    BordaCount,
    /// Approval voting (strategies "approve" of top candidates)
    Approval,
}

/// Consensus voting result with metadata
#[derive(Debug, Clone)]
pub struct ConsensusResult {
    /// predictions
    pub predictions: Array1<i32>,
    /// agreement_scores
    pub agreement_scores: Array1<f64>,
    /// strategy_votes
    pub strategy_votes: Vec<Array1<i32>>,
    /// combined_scores
    pub combined_scores: Array2<f64>,
}

impl<T> TrainedOneVsOne<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the class pairs
    pub fn class_pairs(&self) -> &[(i32, i32)] {
        &self.base_estimator.class_pairs
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.classes.len()
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }

    /// Predict using specific voting strategy
    pub fn predict_with_strategy(
        &self,
        x: &Array2<Float>,
        strategy: VotingStrategy,
    ) -> SklResult<Array1<i32>> {
        match strategy {
            VotingStrategy::Majority => self.predict_majority_voting(x),
            VotingStrategy::Rank => self.predict_rank_voting(x),
            VotingStrategy::Distance => self.predict_distance_voting(x),
            VotingStrategy::Consensus(config) => self.predict_consensus_voting(x, &config),
        }
    }

    /// Predict using majority voting (original implementation)
    fn predict_majority_voting(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut votes = Array1::zeros(n_classes);
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get votes from each pairwise classifier
            for (pair_idx, &(class_i, class_j)) in
                self.base_estimator.class_pairs.iter().enumerate()
            {
                if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                    // Get prediction from binary classifier
                    let prediction = estimator.predict(&sample_matrix.to_owned())?;
                    let score = prediction[0];

                    let class_i_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_i)
                        .unwrap();
                    let class_j_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_j)
                        .unwrap();

                    // Vote based on prediction (>0.5 votes for class_i, <=0.5 for class_j)
                    if score > 0.5 {
                        votes[class_i_idx] += 1.0;
                    } else {
                        votes[class_j_idx] += 1.0;
                    }
                }
            }

            // Predict class with most votes
            let max_idx = votes
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b)| (**a).partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[sample_idx] = self.base_estimator.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Predict using rank-based voting
    ///
    /// Rank-based voting weights each vote by the confidence of the prediction,
    /// measured by the distance from the decision boundary (|score - 0.5|).
    fn predict_rank_voting(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut weighted_votes = Array1::zeros(n_classes);
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Get weighted votes from each pairwise classifier
            for (pair_idx, &(class_i, class_j)) in
                self.base_estimator.class_pairs.iter().enumerate()
            {
                if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                    let prediction = estimator.predict(&sample_matrix.to_owned())?;
                    let score = prediction[0];

                    let class_i_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_i)
                        .unwrap();
                    let class_j_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_j)
                        .unwrap();

                    // Weight = confidence = distance from decision boundary
                    let confidence = (score - 0.5).abs();

                    if score > 0.5 {
                        weighted_votes[class_i_idx] += confidence;
                    } else {
                        weighted_votes[class_j_idx] += confidence;
                    }
                }
            }

            // Predict class with highest weighted vote
            let max_idx = weighted_votes
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b)| (**a).partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[sample_idx] = self.base_estimator.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Predict using distance-based voting
    ///
    /// Distance-based voting accumulates prediction scores directly.
    /// Higher scores indicate stronger belief in a class.
    fn predict_distance_voting(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut distance_scores = Array1::zeros(n_classes);
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            // Accumulate distance scores from each pairwise classifier
            for (pair_idx, &(class_i, class_j)) in
                self.base_estimator.class_pairs.iter().enumerate()
            {
                if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                    let prediction = estimator.predict(&sample_matrix.to_owned())?;
                    let score = prediction[0];

                    let class_i_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_i)
                        .unwrap();
                    let class_j_idx = self
                        .base_estimator
                        .classes
                        .iter()
                        .position(|&c| c == class_j)
                        .unwrap();

                    // Accumulate scores: higher score = more evidence for class
                    distance_scores[class_i_idx] += score;
                    distance_scores[class_j_idx] += 1.0 - score;
                }
            }

            // Predict class with highest accumulated score
            let max_idx = distance_scores
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b)| (**a).partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            predictions[sample_idx] = self.base_estimator.classes[max_idx];
        }

        Ok(predictions)
    }

    /// Get voting decision scores for each class
    pub fn voting_scores(
        &self,
        x: &Array2<Float>,
        strategy: VotingStrategy,
    ) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut scores = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);
            let sample_matrix = sample.insert_axis(Axis(0));

            match strategy {
                VotingStrategy::Majority => {
                    let mut votes = Array1::zeros(n_classes);

                    for (pair_idx, &(class_i, class_j)) in
                        self.base_estimator.class_pairs.iter().enumerate()
                    {
                        if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                            let prediction = estimator.predict(&sample_matrix.to_owned())?;
                            let score = prediction[0];

                            let class_i_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_i)
                                .unwrap();
                            let class_j_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_j)
                                .unwrap();

                            if score > 0.5 {
                                votes[class_i_idx] += 1.0;
                            } else {
                                votes[class_j_idx] += 1.0;
                            }
                        }
                    }

                    for (class_idx, &vote) in votes.iter().enumerate() {
                        scores[[sample_idx, class_idx]] = vote;
                    }
                }

                VotingStrategy::Rank => {
                    let mut rank_scores = Array1::zeros(n_classes);
                    let mut pairwise_scores = Vec::new();

                    for (pair_idx, &(class_i, class_j)) in
                        self.base_estimator.class_pairs.iter().enumerate()
                    {
                        if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                            let prediction = estimator.predict(&sample_matrix.to_owned())?;
                            let score = prediction[0];

                            let class_i_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_i)
                                .unwrap();
                            let class_j_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_j)
                                .unwrap();

                            let confidence = (score - 0.5).abs() * 2.0;
                            if score > 0.5 {
                                pairwise_scores.push((class_i_idx, class_j_idx, confidence));
                            } else {
                                pairwise_scores.push((class_j_idx, class_i_idx, confidence));
                            }
                        }
                    }

                    pairwise_scores.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

                    for (rank, &(winner_idx, _loser_idx, confidence)) in
                        pairwise_scores.iter().enumerate()
                    {
                        let rank_value = (pairwise_scores.len() - rank) as f64 * confidence;
                        rank_scores[winner_idx] += rank_value;
                    }

                    for (class_idx, &score) in rank_scores.iter().enumerate() {
                        scores[[sample_idx, class_idx]] = score;
                    }
                }

                VotingStrategy::Distance => {
                    let mut distance_scores = Array1::zeros(n_classes);

                    for (pair_idx, &(class_i, class_j)) in
                        self.base_estimator.class_pairs.iter().enumerate()
                    {
                        if let Some(estimator) = self.base_estimator.estimators.get(pair_idx) {
                            let prediction = estimator.predict(&sample_matrix.to_owned())?;
                            let score = prediction[0];

                            let class_i_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_i)
                                .unwrap();
                            let class_j_idx = self
                                .base_estimator
                                .classes
                                .iter()
                                .position(|&c| c == class_j)
                                .unwrap();

                            let distance_i = (score - 0.5).max(0.0);
                            let distance_j = (0.5 - score).max(0.0);

                            distance_scores[class_i_idx] += distance_i;
                            distance_scores[class_j_idx] += distance_j;
                        }
                    }

                    for (class_idx, &score) in distance_scores.iter().enumerate() {
                        scores[[sample_idx, class_idx]] = score;
                    }
                }

                VotingStrategy::Consensus(ref config) => {
                    let single_sample_x = x
                        .slice(scirs2_core::ndarray::s![sample_idx..sample_idx + 1, ..])
                        .to_owned();
                    let consensus_result =
                        self.predict_consensus_with_details(&single_sample_x, config)?;
                    for (class_idx, &score) in
                        consensus_result.combined_scores.row(0).iter().enumerate()
                    {
                        scores[[sample_idx, class_idx]] = score;
                    }
                }
            }
        }

        Ok(scores)
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedOneVsOne<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        // Use rank-based voting as the default for better performance
        self.predict_with_strategy(x, VotingStrategy::Rank)
    }
}

/// Enhanced prediction methods for OneVsOne with consensus voting
impl<T> TrainedOneVsOne<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Predict using consensus-based voting
    fn predict_consensus_voting(
        &self,
        x: &Array2<Float>,
        config: &ConsensusConfig,
    ) -> SklResult<Array1<i32>> {
        let consensus_result = self.predict_consensus_with_details(x, config)?;
        Ok(consensus_result.predictions)
    }

    /// Predict using consensus voting with detailed results
    pub fn predict_consensus_with_details(
        &self,
        x: &Array2<Float>,
        config: &ConsensusConfig,
    ) -> SklResult<ConsensusResult> {
        let (n_samples, _) = x.dim();
        let _n_classes = self.base_estimator.classes.len();

        // Collect predictions and scores from all strategies
        let mut strategy_votes = Vec::new();
        let mut strategy_scores = Vec::new();

        for strategy in &config.strategies {
            let votes = match strategy {
                ConsensusStrategy::Majority => self.predict_majority_voting(x)?,
                ConsensusStrategy::Rank => self.predict_rank_voting(x)?,
                ConsensusStrategy::Distance => self.predict_distance_voting(x)?,
            };

            let scores = match strategy {
                ConsensusStrategy::Majority => self.voting_scores(x, VotingStrategy::Majority)?,
                ConsensusStrategy::Rank => self.voting_scores(x, VotingStrategy::Rank)?,
                ConsensusStrategy::Distance => self.voting_scores(x, VotingStrategy::Distance)?,
            };

            strategy_votes.push(votes);
            strategy_scores.push(scores);
        }

        // Combine strategies using specified method
        let (combined_scores, agreement_scores) = self.combine_consensus_strategies(
            &strategy_scores,
            &strategy_votes,
            &config.method,
            config.agreement_threshold,
        )?;

        // Make final predictions
        let mut predictions = Array1::zeros(n_samples);
        for sample_idx in 0..n_samples {
            let max_idx = combined_scores
                .row(sample_idx)
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[sample_idx] = self.base_estimator.classes[max_idx];
        }

        Ok(ConsensusResult {
            predictions,
            agreement_scores,
            strategy_votes,
            combined_scores,
        })
    }

    /// Combine multiple strategy results using consensus method
    fn combine_consensus_strategies(
        &self,
        strategy_scores: &[Array2<f64>],
        strategy_votes: &[Array1<i32>],
        _method: &ConsensusMethod,
        agreement_threshold: f64,
    ) -> SklResult<(Array2<f64>, Array1<f64>)> {
        let n_samples = strategy_scores[0].nrows();
        let n_classes = strategy_scores[0].ncols();
        let n_strategies = strategy_scores.len();

        let mut combined_scores = Array2::zeros((n_samples, n_classes));
        let mut agreement_scores = Array1::zeros(n_samples);

        // For simplicity, use weighted average for all methods in this basic implementation
        for sample_idx in 0..n_samples {
            for class_idx in 0..n_classes {
                let mut sum = 0.0;
                for scores in strategy_scores.iter().take(n_strategies) {
                    sum += scores[[sample_idx, class_idx]];
                }
                combined_scores[[sample_idx, class_idx]] = sum / (n_strategies as f64);
            }

            agreement_scores[sample_idx] =
                self.calculate_agreement_score(strategy_votes, sample_idx, agreement_threshold);
        }

        Ok((combined_scores, agreement_scores))
    }

    /// Calculate agreement score between strategies
    fn calculate_agreement_score(
        &self,
        strategy_votes: &[Array1<i32>],
        sample_idx: usize,
        _threshold: f64,
    ) -> f64 {
        let n_strategies = strategy_votes.len();
        if n_strategies <= 1 {
            return 1.0;
        }

        // Count how many strategies agree on the most common prediction
        let mut vote_counts = HashMap::new();
        for strategy_votes_array in strategy_votes {
            let vote = strategy_votes_array[sample_idx];
            *vote_counts.entry(vote).or_insert(0) += 1;
        }

        let max_agreement = vote_counts.values().max().copied().unwrap_or(0);
        max_agreement as f64 / n_strategies as f64
    }
}

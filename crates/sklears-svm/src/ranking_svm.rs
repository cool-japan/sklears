//! Ranking Support Vector Machines for Learning-to-Rank
//!
//! This module implements SVM-based ranking algorithms for learning-to-rank problems.
//! It supports pairwise ranking approaches where the goal is to learn a function
//! that can correctly order items based on their relevance or preference.

use crate::kernels::{Kernel, KernelType};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Ranking loss functions
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RankingLoss {
    /// Hinge loss for pairwise ranking
    #[default]
    Hinge,
    /// Squared hinge loss
    SquaredHinge,
    /// Logistic loss for ranking
    Logistic,
    /// Exponential loss
    Exponential,
}

/// Ranking approach
#[derive(Debug, Clone, PartialEq, Default)]
pub enum RankingApproach {
    /// Pairwise ranking - learn from pairs of items
    #[default]
    Pairwise,
    /// Pointwise ranking - treat as regression problem
    Pointwise,
}

/// Configuration for ranking SVM
#[derive(Debug, Clone)]
pub struct RankingSVMConfig {
    /// Regularization parameter
    pub c: Float,
    /// Ranking loss function
    pub loss: RankingLoss,
    /// Ranking approach
    pub approach: RankingApproach,
    /// Kernel function to use
    pub kernel: KernelType,
    /// Tolerance for stopping criterion
    pub tol: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Whether to fit an intercept
    pub fit_intercept: bool,
    /// Random seed
    pub random_state: Option<u64>,
    /// Learning rate for gradient-based optimization
    pub learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Number of negative samples per positive pair (for pairwise)
    pub negative_sampling_ratio: Float,
}

impl Default for RankingSVMConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            loss: RankingLoss::default(),
            approach: RankingApproach::default(),
            kernel: KernelType::Rbf { gamma: 1.0 },
            tol: 1e-3,
            max_iter: 1000,
            fit_intercept: true,
            random_state: None,
            learning_rate: 0.01,
            learning_rate_decay: 0.99,
            min_learning_rate: 1e-6,
            negative_sampling_ratio: 1.0,
        }
    }
}

/// Represents a ranking query with items and their relevance scores
#[derive(Debug, Clone)]
pub struct RankingQuery {
    /// Query ID
    pub query_id: String,
    /// Feature vectors for items in this query
    pub features: Array2<Float>,
    /// Relevance scores (higher is better)
    pub relevance_scores: Array1<Float>,
}

/// Training data for ranking SVM
#[derive(Debug, Clone)]
pub struct RankingData {
    /// List of ranking queries
    pub queries: Vec<RankingQuery>,
}

impl Default for RankingData {
    fn default() -> Self {
        Self::new()
    }
}

impl RankingData {
    /// Create new ranking data
    pub fn new() -> Self {
        Self {
            queries: Vec::new(),
        }
    }

    /// Add a ranking query
    pub fn add_query(&mut self, query: RankingQuery) {
        self.queries.push(query);
    }

    /// Create ranking data from arrays with query IDs
    pub fn from_arrays(
        features: Array2<Float>,
        relevance_scores: Array1<Float>,
        query_ids: Vec<String>,
    ) -> Result<Self> {
        if features.nrows() != relevance_scores.len() || features.nrows() != query_ids.len() {
            return Err(SklearsError::InvalidInput(
                "Features, relevance scores, and query IDs must have the same number of samples"
                    .to_string(),
            ));
        }

        let mut query_map: HashMap<String, (Vec<usize>, Vec<Float>)> = HashMap::new();

        for (i, query_id) in query_ids.iter().enumerate() {
            let entry = query_map
                .entry(query_id.clone())
                .or_insert((Vec::new(), Vec::new()));
            entry.0.push(i);
            entry.1.push(relevance_scores[i]);
        }

        let mut ranking_data = RankingData::new();

        for (query_id, (indices, scores)) in query_map {
            let n_items = indices.len();
            let n_features = features.ncols();
            let mut query_features = Array2::zeros((n_items, n_features));

            for (j, &idx) in indices.iter().enumerate() {
                query_features.row_mut(j).assign(&features.row(idx));
            }

            let query = RankingQuery {
                query_id,
                features: query_features,
                relevance_scores: Array1::from_vec(scores),
            };

            ranking_data.add_query(query);
        }

        Ok(ranking_data)
    }

    /// Get total number of items across all queries
    pub fn total_items(&self) -> usize {
        self.queries.iter().map(|q| q.features.nrows()).sum()
    }

    /// Get number of features (assumes all queries have same number of features)
    pub fn n_features(&self) -> usize {
        self.queries
            .first()
            .map(|q| q.features.ncols())
            .unwrap_or(0)
    }
}

/// Ranking Support Vector Machine
#[derive(Debug)]
pub struct RankingSVM<State = Untrained> {
    config: RankingSVMConfig,
    state: PhantomData<State>,
    // Fitted attributes
    support_vectors_: Option<Array2<Float>>,
    alpha_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_in_: Option<usize>,
    n_iter_: Option<usize>,
}

impl RankingSVM<Untrained> {
    /// Create a new ranking SVM
    pub fn new() -> Self {
        Self {
            config: RankingSVMConfig::default(),
            state: PhantomData,
            support_vectors_: None,
            alpha_: None,
            intercept_: None,
            n_features_in_: None,
            n_iter_: None,
        }
    }

    /// Set the regularization parameter C
    pub fn c(mut self, c: Float) -> Self {
        self.config.c = c;
        self
    }

    /// Set the ranking loss function
    pub fn loss(mut self, loss: RankingLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the ranking approach
    pub fn approach(mut self, approach: RankingApproach) -> Self {
        self.config.approach = approach;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.config.kernel = kernel;
        self
    }

    /// Set the tolerance for stopping criterion
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set whether to fit an intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the negative sampling ratio for pairwise ranking
    pub fn negative_sampling_ratio(mut self, ratio: Float) -> Self {
        self.config.negative_sampling_ratio = ratio;
        self
    }
}

impl Default for RankingSVM<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl RankingLoss {
    /// Compute the loss value for given margin
    pub fn loss(&self, margin: Float) -> Float {
        match self {
            RankingLoss::Hinge => (1.0 - margin).max(0.0),
            RankingLoss::SquaredHinge => {
                let hinge = (1.0 - margin).max(0.0);
                hinge * hinge
            }
            RankingLoss::Logistic => (1.0 + (-margin).exp()).ln(),
            RankingLoss::Exponential => (-margin).exp(),
        }
    }

    /// Compute the derivative of the loss function
    pub fn derivative(&self, margin: Float) -> Float {
        match self {
            RankingLoss::Hinge => {
                if margin < 1.0 {
                    -1.0
                } else {
                    0.0
                }
            }
            RankingLoss::SquaredHinge => {
                if margin < 1.0 {
                    -2.0 * (1.0 - margin)
                } else {
                    0.0
                }
            }
            RankingLoss::Logistic => -1.0 / (1.0 + margin.exp()),
            RankingLoss::Exponential => -(-margin).exp(),
        }
    }
}

/// Pairwise training sample
#[derive(Debug, Clone)]
struct PairwiseSample {
    /// Features of the preferred item
    preferred_features: Array1<Float>,
    /// Features of the non-preferred item  
    non_preferred_features: Array1<Float>,
    /// Difference vector (preferred - non_preferred)
    difference_vector: Array1<Float>,
}

impl RankingSVM<Untrained> {
    /// Convert ranking data to pairwise samples
    fn create_pairwise_samples(&self, ranking_data: &RankingData) -> Vec<PairwiseSample> {
        let mut samples = Vec::new();

        for query in &ranking_data.queries {
            let n_items = query.features.nrows();

            // Create pairs from items with different relevance scores
            for i in 0..n_items {
                for j in 0..n_items {
                    if i != j && query.relevance_scores[i] > query.relevance_scores[j] {
                        let preferred = query.features.row(i).to_owned();
                        let non_preferred = query.features.row(j).to_owned();
                        let difference = &preferred - &non_preferred;

                        samples.push(PairwiseSample {
                            preferred_features: preferred,
                            non_preferred_features: non_preferred,
                            difference_vector: difference,
                        });
                    }
                }
            }
        }

        samples
    }
}

impl RankingSVM<Untrained> {
    /// Fit the ranking SVM with ranking data
    pub fn fit_ranking(self, ranking_data: &RankingData) -> Result<RankingSVM<Trained>> {
        if ranking_data.queries.is_empty() {
            return Err(SklearsError::InvalidInput("Empty ranking data".to_string()));
        }

        let n_features = ranking_data.n_features();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features in ranking data".to_string(),
            ));
        }

        match self.config.approach {
            RankingApproach::Pairwise => self.fit_pairwise(ranking_data),
            RankingApproach::Pointwise => self.fit_pointwise(ranking_data),
        }
    }

    fn fit_pairwise(self, ranking_data: &RankingData) -> Result<RankingSVM<Trained>> {
        // Create pairwise samples
        let pairwise_samples = self.create_pairwise_samples(ranking_data);

        if pairwise_samples.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No valid pairwise samples could be created".to_string(),
            ));
        }

        let n_samples = pairwise_samples.len();
        let n_features = ranking_data.n_features();

        // Create training matrix from difference vectors
        let mut x_train = Array2::zeros((n_samples, n_features));
        for (i, sample) in pairwise_samples.iter().enumerate() {
            x_train.row_mut(i).assign(&sample.difference_vector);
        }

        // All pairwise samples have target +1 (preferred item should rank higher)
        let y_train = Array1::<Float>::ones(n_samples);

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        // Initialize model parameters
        let support_vectors = x_train.clone();
        let mut alpha = Array1::<Float>::zeros(n_samples);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // Training loop
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            for i in 0..n_samples {
                // Compute prediction (margin)
                let mut margin = if self.config.fit_intercept {
                    intercept
                } else {
                    0.0
                };

                for j in 0..n_samples {
                    if alpha[j].abs() > 1e-10 {
                        let k_val = kernel.compute(x_train.row(i), support_vectors.row(j));
                        margin += alpha[j] * k_val;
                    }
                }

                // Compute loss derivative
                let loss_derivative = self.config.loss.derivative(margin);

                // Update alpha using gradient descent
                let old_alpha = alpha[i];
                let gradient = loss_derivative + alpha[i] / self.config.c;
                alpha[i] -= current_lr * gradient;

                // Update intercept if needed
                if self.config.fit_intercept {
                    intercept -= current_lr * loss_derivative;
                }

                // Check convergence
                if (alpha[i] - old_alpha).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Filter out non-support vectors
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x_train.row(0));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x_train.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(RankingSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
        })
    }

    fn fit_pointwise(self, ranking_data: &RankingData) -> Result<RankingSVM<Trained>> {
        // Collect all features and relevance scores
        let total_items = ranking_data.total_items();
        let n_features = ranking_data.n_features();

        let mut x_train = Array2::zeros((total_items, n_features));
        let mut y_train = Array1::zeros(total_items);

        let mut item_idx = 0;
        for query in &ranking_data.queries {
            for i in 0..query.features.nrows() {
                x_train.row_mut(item_idx).assign(&query.features.row(i));
                y_train[item_idx] = query.relevance_scores[i];
                item_idx += 1;
            }
        }

        // Create kernel instance
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        // Initialize model parameters
        let support_vectors = x_train.clone();
        let mut alpha = Array1::<Float>::zeros(total_items);
        let mut intercept = 0.0;
        let mut current_lr = self.config.learning_rate;

        // Training loop for regression
        let mut n_iter = 0;
        for iteration in 0..self.config.max_iter {
            n_iter = iteration + 1;
            let mut converged = true;

            for i in 0..total_items {
                // Compute prediction
                let mut prediction = if self.config.fit_intercept {
                    intercept
                } else {
                    0.0
                };

                for j in 0..total_items {
                    if alpha[j].abs() > 1e-10 {
                        let k_val = kernel.compute(x_train.row(i), support_vectors.row(j));
                        prediction += alpha[j] * k_val;
                    }
                }

                // Compute error
                let error = prediction - y_train[i];

                // Update alpha using gradient descent (simple squared loss)
                let old_alpha = alpha[i];
                let gradient = error + alpha[i] / self.config.c;
                alpha[i] -= current_lr * gradient;

                // Update intercept if needed
                if self.config.fit_intercept {
                    intercept -= current_lr * error;
                }

                // Check convergence
                if (alpha[i] - old_alpha).abs() > self.config.tol {
                    converged = false;
                }
            }

            // Update learning rate
            current_lr =
                (current_lr * self.config.learning_rate_decay).max(self.config.min_learning_rate);

            if converged {
                break;
            }
        }

        // Filter out non-support vectors
        let support_indices: Vec<usize> = alpha
            .iter()
            .enumerate()
            .filter(|(_, &a)| a.abs() > 1e-8)
            .map(|(i, _)| i)
            .collect();

        let final_support_vectors = if support_indices.is_empty() {
            let mut sv = Array2::zeros((1, n_features));
            sv.row_mut(0).assign(&x_train.row(0));
            sv
        } else {
            let mut sv = Array2::zeros((support_indices.len(), n_features));
            for (i, &idx) in support_indices.iter().enumerate() {
                sv.row_mut(i).assign(&x_train.row(idx));
            }
            sv
        };

        let final_alpha = if support_indices.is_empty() {
            Array1::from_vec(vec![1e-8])
        } else {
            Array1::from_vec(support_indices.iter().map(|&i| alpha[i]).collect())
        };

        Ok(RankingSVM {
            config: self.config,
            state: PhantomData,
            support_vectors_: Some(final_support_vectors),
            alpha_: Some(final_alpha),
            intercept_: Some(intercept),
            n_features_in_: Some(n_features),
            n_iter_: Some(n_iter),
        })
    }
}

impl RankingSVM<Trained> {
    /// Score items for ranking
    pub fn score(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.n_features_in_.unwrap() {
            return Err(SklearsError::InvalidInput(
                "Feature mismatch: X has different number of features than training data"
                    .to_string(),
            ));
        }

        let support_vectors = self.support_vectors_.as_ref().unwrap();
        let alpha = self.alpha_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap();
        let kernel = match &self.config.kernel {
            KernelType::Linear => Box::new(crate::kernels::LinearKernel) as Box<dyn Kernel>,
            KernelType::Rbf { gamma } => {
                Box::new(crate::kernels::RbfKernel::new(*gamma)) as Box<dyn Kernel>
            }
            _ => Box::new(crate::kernels::RbfKernel::new(1.0)) as Box<dyn Kernel>,
        };

        let mut scores = Array1::zeros(x.nrows());

        for i in 0..x.nrows() {
            let mut score = if self.config.fit_intercept {
                intercept
            } else {
                0.0
            };

            for j in 0..support_vectors.nrows() {
                let k_val = kernel.compute(x.row(i), support_vectors.row(j));
                score += alpha[j] * k_val;
            }

            scores[i] = score;
        }

        Ok(scores)
    }

    /// Rank items by their scores (higher scores get lower ranks)
    pub fn rank(&self, x: &Array2<Float>) -> Result<Array1<usize>> {
        let scores = self.score(x)?;

        // Create vector of (score, index) pairs
        let mut score_index_pairs: Vec<(Float, usize)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (score, i))
            .collect();

        // Sort by score in descending order (higher scores first)
        score_index_pairs
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        // Create ranking array (rank 0 is best)
        let mut rankings = Array1::zeros(x.nrows());
        for (rank, (_, original_index)) in score_index_pairs.iter().enumerate() {
            rankings[*original_index] = rank;
        }

        Ok(rankings)
    }

    /// Get the indices of items sorted by their predicted scores
    pub fn argsort(&self, x: &Array2<Float>) -> Result<Array1<usize>> {
        let scores = self.score(x)?;

        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(Array1::from_vec(indices))
    }

    /// Get the support vectors
    pub fn support_vectors(&self) -> &Array2<Float> {
        self.support_vectors_.as_ref().unwrap()
    }

    /// Get the support vector coefficients (alpha values)
    pub fn alpha(&self) -> &Array1<Float> {
        self.alpha_.as_ref().unwrap()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap()
    }

    /// Get the number of features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.unwrap()
    }

    /// Get the number of iterations performed during training
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ranking_svm_creation() {
        let rsvm = RankingSVM::new()
            .c(2.0)
            .loss(RankingLoss::Hinge)
            .approach(RankingApproach::Pairwise)
            .kernel(KernelType::Linear)
            .negative_sampling_ratio(1.5);

        assert_eq!(rsvm.config.c, 2.0);
        assert_eq!(rsvm.config.loss, RankingLoss::Hinge);
        assert_eq!(rsvm.config.approach, RankingApproach::Pairwise);
        assert_eq!(rsvm.config.negative_sampling_ratio, 1.5);
    }

    #[test]
    fn test_ranking_data_creation() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let relevance_scores = array![3.0, 1.0, 4.0, 2.0];
        let query_ids = vec![
            "q1".to_string(),
            "q1".to_string(),
            "q2".to_string(),
            "q2".to_string(),
        ];

        let ranking_data = RankingData::from_arrays(features, relevance_scores, query_ids).unwrap();

        assert_eq!(ranking_data.queries.len(), 2);
        assert_eq!(ranking_data.total_items(), 4);
        assert_eq!(ranking_data.n_features(), 2);
    }

    #[test]
    fn test_ranking_loss_functions() {
        let hinge = RankingLoss::Hinge;
        let squared_hinge = RankingLoss::SquaredHinge;
        let logistic = RankingLoss::Logistic;

        // Test hinge loss
        assert!((hinge.loss(1.5) - 0.0).abs() < 1e-6); // margin > 1
        assert!((hinge.loss(0.5) - 0.5).abs() < 1e-6); // margin < 1
        assert!((hinge.derivative(0.5) - (-1.0)).abs() < 1e-6);
        assert!((hinge.derivative(1.5) - 0.0).abs() < 1e-6);

        // Test squared hinge loss
        assert!((squared_hinge.loss(0.5) - 0.25).abs() < 1e-6); // (1-0.5)^2
        assert!((squared_hinge.derivative(0.5) - (-1.0)).abs() < 1e-6); // -2*(1-0.5)

        // Test logistic loss (should be positive)
        assert!(logistic.loss(0.0) > 0.0);
        assert!(logistic.loss(1.0) > 0.0);
    }

    #[test]
    #[ignore = "Slow test: trains ranking SVM. Run with --ignored flag"]
    fn test_pairwise_ranking_svm_training() {
        // Create simple ranking data
        let features = array![
            [1.0, 2.0], // q1: item 1 (score 1)
            [2.0, 3.0], // q1: item 2 (score 3) - best for q1
            [3.0, 1.0], // q1: item 3 (score 2)
            [4.0, 5.0], // q2: item 1 (score 4) - best for q2
            [5.0, 2.0], // q2: item 2 (score 1)
        ];
        let relevance_scores = array![1.0, 3.0, 2.0, 4.0, 1.0];
        let query_ids = vec![
            "q1".to_string(),
            "q1".to_string(),
            "q1".to_string(),
            "q2".to_string(),
            "q2".to_string(),
        ];

        let ranking_data = RankingData::from_arrays(features, relevance_scores, query_ids).unwrap();

        let rsvm = RankingSVM::new()
            .c(1.0)
            .loss(RankingLoss::Hinge)
            .approach(RankingApproach::Pairwise)
            .kernel(KernelType::Linear)
            .max_iter(100)
            .learning_rate(0.01)
            .random_state(42);

        let fitted_model = rsvm.fit_ranking(&ranking_data).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert!(fitted_model.n_iter() > 0);

        // Test scoring
        let test_features = array![[1.5, 2.5], [4.5, 4.0],];

        let scores = fitted_model.score(&test_features).unwrap();
        assert_eq!(scores.len(), 2);

        // Test ranking
        let rankings = fitted_model.rank(&test_features).unwrap();
        assert_eq!(rankings.len(), 2);

        // Test argsort
        let sorted_indices = fitted_model.argsort(&test_features).unwrap();
        assert_eq!(sorted_indices.len(), 2);

        // Scores should be finite
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    #[ignore = "Slow test: trains pointwise ranking SVM. Run with --ignored flag"]
    fn test_pointwise_ranking_svm_training() {
        let features = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let relevance_scores = array![1.0, 2.0, 3.0, 4.0];
        let query_ids = vec![
            "q1".to_string(),
            "q1".to_string(),
            "q1".to_string(),
            "q1".to_string(),
        ];

        let ranking_data =
            RankingData::from_arrays(features.clone(), relevance_scores, query_ids).unwrap();

        let rsvm = RankingSVM::new()
            .c(1.0)
            .approach(RankingApproach::Pointwise)
            .kernel(KernelType::Linear)
            .max_iter(50)
            .learning_rate(0.01)
            .random_state(42);

        let fitted_model = rsvm.fit_ranking(&ranking_data).unwrap();

        assert_eq!(fitted_model.n_features_in(), 2);
        assert!(fitted_model.n_iter() > 0);

        let scores = fitted_model.score(&features).unwrap();
        assert_eq!(scores.len(), 4);

        // Scores should generally increase with relevance
        // (though not guaranteed due to simple training)
        for &score in scores.iter() {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_ranking_data_shape_mismatch() {
        let features = array![[1.0, 2.0], [3.0, 4.0]];
        let relevance_scores = array![1.0]; // Wrong length
        let query_ids = vec!["q1".to_string(), "q1".to_string()];

        let result = RankingData::from_arrays(features, relevance_scores, query_ids);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("same number of samples"));
    }

    #[test]
    fn test_empty_ranking_data() {
        let ranking_data = RankingData::new();
        let rsvm = RankingSVM::new();

        let result = rsvm.fit_ranking(&ranking_data);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Empty ranking data"));
    }
}

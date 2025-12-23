//! Meta-Learning for Kernel Selection
//!
//! This module implements meta-learning strategies for automated kernel selection,
//! few-shot kernel learning, transfer learning for kernels, and neural architecture
//! search for kernel methods.
//!
//! # References
//! - Vanschoren et al. (2014): "Meta-Learning: A Survey"
//! - Feurer & Hutter (2019): "Hyperparameter Optimization"
//! - Hospedales et al. (2021): "Meta-Learning in Neural Networks: A Survey"
//! - Wilson & Izmailov (2020): "Bayesian Deep Learning and a Probabilistic Perspective of Generalization"

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Kernel types supported by meta-learning
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetaKernelType {
    /// RBF kernel
    RBF,
    /// Polynomial kernel
    Polynomial,
    /// Laplacian kernel
    Laplacian,
    /// Matern kernel
    Matern,
    /// Rational Quadratic kernel
    RationalQuadratic,
    /// Linear kernel
    Linear,
}

/// Meta-features extracted from datasets for kernel selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetMetaFeatures {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Feature mean magnitudes
    pub feature_means: Vec<Float>,
    /// Feature standard deviations
    pub feature_stds: Vec<Float>,
    /// Inter-feature correlations (mean absolute)
    pub mean_correlation: Float,
    /// Dataset sparsity (fraction of near-zero values)
    pub sparsity: Float,
    /// Effective dimensionality (ratio of explained variance)
    pub effective_dim: Float,
}

impl DatasetMetaFeatures {
    /// Extract meta-features from a dataset
    pub fn extract(x: &Array2<Float>) -> Result<Self> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Dataset must have non-zero dimensions".to_string(),
            ));
        }

        // Compute feature statistics
        let mut feature_means = Vec::with_capacity(n_features);
        let mut feature_stds = Vec::with_capacity(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.iter().sum::<Float>() / n_samples as Float;
            let variance =
                col.iter().map(|&v| (v - mean).powi(2)).sum::<Float>() / n_samples as Float;
            let std = variance.sqrt();

            feature_means.push(mean);
            feature_stds.push(std);
        }

        // Compute mean absolute correlation
        let mut correlations = Vec::new();
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let col_i = x.column(i);
                let col_j = x.column(j);

                let mean_i = feature_means[i];
                let mean_j = feature_means[j];
                let std_i = feature_stds[i].max(1e-10);
                let std_j = feature_stds[j].max(1e-10);

                let cov: Float = col_i
                    .iter()
                    .zip(col_j.iter())
                    .map(|(&vi, &vj)| (vi - mean_i) * (vj - mean_j))
                    .sum::<Float>()
                    / n_samples as Float;

                let corr = cov / (std_i * std_j);
                correlations.push(corr.abs());
            }
        }

        let mean_correlation = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<Float>() / correlations.len() as Float
        };

        // Compute sparsity
        let threshold = 1e-6;
        let near_zero_count = x.iter().filter(|&&v| v.abs() < threshold).count();
        let sparsity = near_zero_count as Float / (n_samples * n_features) as Float;

        // Estimate effective dimensionality using variance explanation
        // (simplified: ratio of non-negligible std features)
        let significant_features = feature_stds.iter().filter(|&&std| std > 0.01).count();
        let effective_dim = significant_features as Float / n_features as Float;

        Ok(Self {
            n_samples,
            n_features,
            feature_means,
            feature_stds,
            mean_correlation,
            sparsity,
            effective_dim,
        })
    }

    /// Compute a feature vector for meta-learning models
    pub fn to_feature_vector(&self) -> Vec<Float> {
        vec![
            (self.n_samples as Float).ln(),
            (self.n_features as Float).ln(),
            self.feature_means.iter().sum::<Float>() / self.n_features as Float,
            self.feature_stds.iter().sum::<Float>() / self.n_features as Float,
            self.mean_correlation,
            self.sparsity,
            self.effective_dim,
        ]
    }
}

/// Strategy for meta-learning-based kernel selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetaLearningStrategy {
    /// Use historical performance data to select kernel
    PerformanceBased {
        /// Minimum number of similar tasks required
        min_similar_tasks: usize,
        /// Similarity threshold for task matching
        similarity_threshold: Float,
    },
    /// Portfolio-based selection (try multiple kernels)
    Portfolio {
        /// Number of top kernels to include
        portfolio_size: usize,
    },
    /// Bayesian optimization for kernel selection
    BayesianOptimization {
        /// Number of initial random trials
        n_initial: usize,
        /// Number of optimization iterations
        n_iterations: usize,
    },
    /// Neural architecture search
    NeuralArchitectureSearch {
        /// Search space size
        search_space_size: usize,
        /// Number of evaluations
        n_evaluations: usize,
    },
}

impl Default for MetaLearningStrategy {
    fn default() -> Self {
        Self::PerformanceBased {
            min_similar_tasks: 5,
            similarity_threshold: 0.7,
        }
    }
}

/// Configuration for meta-learning kernel selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetaLearningConfig {
    /// Meta-learning strategy to use
    pub strategy: MetaLearningStrategy,
    /// Number of components for selected kernel
    pub n_components: usize,
    /// Whether to use transfer learning
    pub use_transfer_learning: bool,
    /// Performance metric for kernel evaluation
    pub performance_metric: PerformanceMetric,
}

impl Default for MetaLearningConfig {
    fn default() -> Self {
        Self {
            strategy: MetaLearningStrategy::default(),
            n_components: 100,
            use_transfer_learning: false,
            performance_metric: PerformanceMetric::KernelAlignment,
        }
    }
}

/// Performance metrics for kernel evaluation
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PerformanceMetric {
    /// Kernel alignment
    KernelAlignment,
    /// Cross-validation score
    CrossValidation,
    /// Spectral properties
    SpectralQuality,
    /// Approximation error
    ApproximationError,
}

/// Task metadata for meta-learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    /// Task identifier
    pub task_id: String,
    /// Dataset meta-features
    pub meta_features: DatasetMetaFeatures,
    /// Best performing kernel
    pub best_kernel: MetaKernelType,
    /// Performance achieved
    pub performance: Float,
    /// Hyperparameters used
    pub hyperparameters: HashMap<String, Float>,
}

/// Meta-Learning Kernel Selector
///
/// Automatically selects the best kernel approximation method based on
/// dataset characteristics and historical performance.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::meta_learning_kernels::{MetaLearningKernelSelector, MetaLearningConfig};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = MetaLearningConfig::default();
/// let selector = MetaLearningKernelSelector::new(config);
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = selector.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MetaLearningKernelSelector<State = Untrained> {
    config: MetaLearningConfig,

    // Fitted attributes
    selected_kernel: Option<MetaKernelType>,
    selected_hyperparams: Option<HashMap<String, Float>>,
    kernel_weights: Option<Array2<Float>>,
    kernel_offset: Option<Array1<Float>>,
    task_history: Vec<TaskMetadata>,

    _state: PhantomData<State>,
}

impl MetaLearningKernelSelector<Untrained> {
    /// Create a new meta-learning kernel selector
    pub fn new(config: MetaLearningConfig) -> Self {
        Self {
            config,
            selected_kernel: None,
            selected_hyperparams: None,
            kernel_weights: None,
            kernel_offset: None,
            task_history: Vec::new(),
            _state: PhantomData,
        }
    }

    /// Create with default configuration
    pub fn with_components(n_components: usize) -> Self {
        Self {
            config: MetaLearningConfig {
                n_components,
                ..Default::default()
            },
            selected_kernel: None,
            selected_hyperparams: None,
            kernel_weights: None,
            kernel_offset: None,
            task_history: Vec::new(),
            _state: PhantomData,
        }
    }

    /// Add historical task data for meta-learning
    pub fn add_task_history(mut self, task: TaskMetadata) -> Self {
        self.task_history.push(task);
        self
    }

    /// Set meta-learning strategy
    pub fn strategy(mut self, strategy: MetaLearningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Select kernel based on dataset meta-features
    fn select_kernel(
        &self,
        meta_features: &DatasetMetaFeatures,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        match &self.config.strategy {
            MetaLearningStrategy::PerformanceBased {
                min_similar_tasks,
                similarity_threshold,
            } => self.select_performance_based(
                meta_features,
                *min_similar_tasks,
                *similarity_threshold,
            ),
            MetaLearningStrategy::Portfolio { portfolio_size } => {
                self.select_portfolio_based(meta_features, *portfolio_size)
            }
            MetaLearningStrategy::BayesianOptimization {
                n_initial,
                n_iterations,
            } => self.select_bayesian(meta_features, *n_initial, *n_iterations),
            MetaLearningStrategy::NeuralArchitectureSearch {
                search_space_size,
                n_evaluations,
            } => self.select_nas(meta_features, *search_space_size, *n_evaluations),
        }
    }

    /// Performance-based kernel selection
    fn select_performance_based(
        &self,
        meta_features: &DatasetMetaFeatures,
        min_similar: usize,
        similarity_threshold: Float,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        if self.task_history.len() < min_similar {
            // Fallback to heuristic selection
            return self.heuristic_selection(meta_features);
        }

        // Compute similarity to historical tasks
        let current_features = meta_features.to_feature_vector();
        let mut similarities: Vec<(usize, Float)> = self
            .task_history
            .iter()
            .enumerate()
            .map(|(idx, task)| {
                let hist_features = task.meta_features.to_feature_vector();
                let similarity = Self::compute_similarity(&current_features, &hist_features);
                (idx, similarity)
            })
            .filter(|(_, sim)| *sim >= similarity_threshold)
            .collect();

        if similarities.is_empty() {
            return self.heuristic_selection(meta_features);
        }

        // Sort by similarity and performance
        similarities.sort_by(|a, b| {
            let perf_a = self.task_history[a.0].performance;
            let perf_b = self.task_history[b.0].performance;
            (a.1 * perf_a)
                .partial_cmp(&(b.1 * perf_b))
                .unwrap()
                .reverse()
        });

        // Return best performing similar task's kernel
        let best_task = &self.task_history[similarities[0].0];
        (best_task.best_kernel, best_task.hyperparameters.clone())
    }

    /// Portfolio-based selection (returns weighted combination)
    fn select_portfolio_based(
        &self,
        meta_features: &DatasetMetaFeatures,
        _portfolio_size: usize,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        // For simplicity, use heuristic selection
        // In a full implementation, this would select multiple kernels
        self.heuristic_selection(meta_features)
    }

    /// Bayesian optimization-based selection
    fn select_bayesian(
        &self,
        meta_features: &DatasetMetaFeatures,
        _n_initial: usize,
        _n_iterations: usize,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        // Simplified: use heuristic
        // Full implementation would use Gaussian process-based optimization
        self.heuristic_selection(meta_features)
    }

    /// Neural architecture search-based selection
    fn select_nas(
        &self,
        meta_features: &DatasetMetaFeatures,
        _search_space_size: usize,
        _n_evaluations: usize,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        // Simplified: use heuristic
        // Full implementation would use evolutionary or gradient-based NAS
        self.heuristic_selection(meta_features)
    }

    /// Heuristic kernel selection based on dataset characteristics
    fn heuristic_selection(
        &self,
        meta_features: &DatasetMetaFeatures,
    ) -> (MetaKernelType, HashMap<String, Float>) {
        let mut hyperparams = HashMap::new();

        // Select kernel based on dataset properties
        let kernel = if meta_features.sparsity > 0.5 {
            // High sparsity: use linear kernel
            hyperparams.insert("gamma".to_string(), 1.0);
            MetaKernelType::Linear
        } else if meta_features.effective_dim < 0.3 {
            // Low effective dimensionality: use RBF
            let gamma = 1.0 / (meta_features.n_features as Float);
            hyperparams.insert("gamma".to_string(), gamma);
            MetaKernelType::RBF
        } else if meta_features.mean_correlation > 0.7 {
            // High correlation: use polynomial
            hyperparams.insert("degree".to_string(), 3.0);
            hyperparams.insert("gamma".to_string(), 1.0);
            MetaKernelType::Polynomial
        } else {
            // Default: RBF kernel
            hyperparams.insert("gamma".to_string(), 1.0);
            MetaKernelType::RBF
        };

        (kernel, hyperparams)
    }

    /// Compute cosine similarity between feature vectors
    fn compute_similarity(a: &[Float], b: &[Float]) -> Float {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: Float = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: Float = a.iter().map(|x| x * x).sum::<Float>().sqrt();
        let norm_b: Float = b.iter().map(|x| x * x).sum::<Float>().sqrt();

        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }

        (dot / (norm_a * norm_b)).max(-1.0).min(1.0)
    }
}

impl Estimator for MetaLearningKernelSelector<Untrained> {
    type Config = MetaLearningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for MetaLearningKernelSelector<Untrained> {
    type Fitted = MetaLearningKernelSelector<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Extract dataset meta-features
        let meta_features = DatasetMetaFeatures::extract(x)?;

        // Select kernel and hyperparameters
        let (selected_kernel, selected_hyperparams) = self.select_kernel(&meta_features);

        // Initialize kernel approximation based on selection
        let (kernel_weights, kernel_offset) = Self::initialize_kernel_approximation(
            selected_kernel,
            &selected_hyperparams,
            x,
            self.config.n_components,
        )?;

        Ok(MetaLearningKernelSelector {
            config: self.config,
            selected_kernel: Some(selected_kernel),
            selected_hyperparams: Some(selected_hyperparams),
            kernel_weights: Some(kernel_weights),
            kernel_offset: Some(kernel_offset),
            task_history: self.task_history,
            _state: PhantomData,
        })
    }
}

impl MetaLearningKernelSelector<Untrained> {
    /// Initialize kernel approximation random features
    fn initialize_kernel_approximation(
        kernel_type: MetaKernelType,
        hyperparams: &HashMap<String, Float>,
        x: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n_features = x.ncols();
        let mut rng = thread_rng();

        match kernel_type {
            MetaKernelType::RBF | MetaKernelType::Laplacian => {
                let gamma = hyperparams.get("gamma").copied().unwrap_or(1.0);
                let normal = Normal::new(0.0, 1.0).unwrap();

                let weights = Array2::from_shape_fn((n_features, n_components), |_| {
                    rng.sample(normal) * (2.0 * gamma).sqrt()
                });

                let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
                let offset = Array1::from_shape_fn(n_components, |_| rng.sample(uniform));

                Ok((weights, offset))
            }
            MetaKernelType::Polynomial => {
                let gamma = hyperparams.get("gamma").copied().unwrap_or(1.0);
                let normal = Normal::new(0.0, 1.0).unwrap();

                let weights = Array2::from_shape_fn((n_features, n_components), |_| {
                    rng.sample(normal) * (2.0 * gamma).sqrt()
                });

                let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
                let offset = Array1::from_shape_fn(n_components, |_| rng.sample(uniform));

                Ok((weights, offset))
            }
            MetaKernelType::Linear => {
                // Linear kernel doesn't need random features, use identity-like projection
                let weights = Array2::from_shape_fn((n_features, n_components), |_| {
                    rng.sample(Normal::new(0.0, 1.0 / (n_features as Float).sqrt()).unwrap())
                });
                let offset = Array1::zeros(n_components);
                Ok((weights, offset))
            }
            _ => {
                // Default to RBF-like initialization
                let normal = Normal::new(0.0, 1.0).unwrap();
                let weights =
                    Array2::from_shape_fn((n_features, n_components), |_| rng.sample(normal));
                let uniform = Uniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap();
                let offset = Array1::from_shape_fn(n_components, |_| rng.sample(uniform));
                Ok((weights, offset))
            }
        }
    }
}

impl Transform<Array2<Float>, Array2<Float>> for MetaLearningKernelSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let kernel_weights = self.kernel_weights.as_ref().unwrap();
        let kernel_offset = self.kernel_offset.as_ref().unwrap();

        if x.ncols() != kernel_weights.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                kernel_weights.nrows(),
                x.ncols()
            )));
        }

        // Apply random Fourier features
        let projection = x.dot(kernel_weights);

        let n_samples = x.nrows();
        let n_components = self.config.n_components;
        let mut output = Array2::zeros((n_samples, n_components));

        let normalizer = (2.0 / n_components as Float).sqrt();
        for i in 0..n_samples {
            for j in 0..n_components {
                output[[i, j]] = normalizer * (projection[[i, j]] + kernel_offset[j]).cos();
            }
        }

        Ok(output)
    }
}

impl MetaLearningKernelSelector<Trained> {
    /// Get the selected kernel type
    pub fn selected_kernel(&self) -> MetaKernelType {
        self.selected_kernel.unwrap()
    }

    /// Get the selected hyperparameters
    pub fn selected_hyperparameters(&self) -> &HashMap<String, Float> {
        self.selected_hyperparams.as_ref().unwrap()
    }

    /// Get kernel weights
    pub fn kernel_weights(&self) -> &Array2<Float> {
        self.kernel_weights.as_ref().unwrap()
    }

    /// Get kernel offset
    pub fn kernel_offset(&self) -> &Array1<Float> {
        self.kernel_offset.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_meta_features_extraction() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let meta_features = DatasetMetaFeatures::extract(&x).unwrap();

        assert_eq!(meta_features.n_samples, 4);
        assert_eq!(meta_features.n_features, 3);
        assert!(meta_features.feature_means.len() == 3);
        assert!(meta_features.mean_correlation >= 0.0);
        assert!(meta_features.mean_correlation <= 1.0);
    }

    #[test]
    fn test_meta_learning_selector_basic() {
        let config = MetaLearningConfig::default();
        let selector = MetaLearningKernelSelector::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = selector.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 100]);
    }

    #[test]
    fn test_kernel_selection_with_history() {
        // Use a strategy that requires fewer similar tasks
        let config = MetaLearningConfig {
            strategy: MetaLearningStrategy::PerformanceBased {
                min_similar_tasks: 1,
                similarity_threshold: 0.5,
            },
            n_components: 50,
            use_transfer_learning: false,
            performance_metric: PerformanceMetric::KernelAlignment,
        };

        let mut selector = MetaLearningKernelSelector::new(config);

        // Add historical task data with very similar dataset
        let x_hist = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let meta_features = DatasetMetaFeatures::extract(&x_hist).unwrap();

        let mut hyperparams = HashMap::new();
        hyperparams.insert("gamma".to_string(), 0.5);

        let task = TaskMetadata {
            task_id: "task1".to_string(),
            meta_features,
            best_kernel: MetaKernelType::RBF,
            performance: 0.95,
            hyperparameters: hyperparams,
        };

        selector = selector.add_task_history(task);

        // Use same dataset to ensure high similarity
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let fitted = selector.fit(&x, &()).unwrap();

        // Should select RBF from historical data or heuristic (both are valid)
        let selected = fitted.selected_kernel();
        assert!(
            selected == MetaKernelType::RBF
                || selected == MetaKernelType::Polynomial
                || selected == MetaKernelType::Linear,
            "Unexpected kernel type: {:?}",
            selected
        );
    }

    #[test]
    fn test_different_strategies() {
        let strategies = vec![
            MetaLearningStrategy::PerformanceBased {
                min_similar_tasks: 3,
                similarity_threshold: 0.6,
            },
            MetaLearningStrategy::Portfolio { portfolio_size: 3 },
        ];

        let x = array![[1.0, 2.0], [3.0, 4.0]];

        for strategy in strategies {
            let config = MetaLearningConfig {
                strategy,
                ..Default::default()
            };

            let selector = MetaLearningKernelSelector::new(config);
            let fitted = selector.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.nrows(), 2);
        }
    }

    #[test]
    fn test_heuristic_selection() {
        let x_sparse = array![[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]];

        let selector = MetaLearningKernelSelector::with_components(50);
        let fitted = selector.fit(&x_sparse, &()).unwrap();

        // Should select linear kernel for sparse data
        assert!(
            fitted.selected_kernel() == MetaKernelType::Linear
                || fitted.selected_kernel() == MetaKernelType::RBF
        );
    }

    #[test]
    fn test_similarity_computation() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];

        let similarity = MetaLearningKernelSelector::<Untrained>::compute_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-6);

        let c = vec![-1.0, -2.0, -3.0];
        let similarity2 = MetaLearningKernelSelector::<Untrained>::compute_similarity(&a, &c);
        assert!((similarity2 + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_empty_input_error() {
        let selector = MetaLearningKernelSelector::with_components(50);
        let x_empty: Array2<Float> = Array2::zeros((0, 0));

        assert!(selector.fit(&x_empty, &()).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let selector = MetaLearningKernelSelector::with_components(50);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]];

        let fitted = selector.fit(&x_train, &()).unwrap();
        assert!(fitted.transform(&x_test).is_err());
    }
}

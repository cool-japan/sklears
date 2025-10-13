//! Transfer Learning for Multi-Task Learning
//!
//! This module provides transfer learning algorithms for multi-task scenarios,
//! including domain adaptation, progressive transfer, and continual learning methods.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::RandNormal;
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Untrained},
    types::Float,
};

/// Cross-Task Transfer Learning
///
/// Implements cross-task transfer learning for multi-task scenarios where
/// knowledge from source tasks is transferred to target tasks.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::transfer_learning::CrossTaskTransferLearning;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let source_data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let source_labels = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
/// let target_data = array![[1.1, 2.1], [2.1, 3.1]];
/// let target_labels = array![[1.0, 0.0], [0.0, 1.0]];
///
/// let transfer = CrossTaskTransferLearning::new()
///     .transfer_strength(0.5)
///     .learning_rate(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct CrossTaskTransferLearning<S = Untrained> {
    state: S,
    transfer_strength: Float,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct CrossTaskTransferLearningTrained {
    source_weights: Array2<Float>,
    target_weights: Array2<Float>,
    transfer_matrix: Array2<Float>,
    n_features: usize,
    n_source_tasks: usize,
    n_target_tasks: usize,
}

impl CrossTaskTransferLearning<Untrained> {
    /// Create a new CrossTaskTransferLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            transfer_strength: 0.5,
            learning_rate: 0.01,
            max_iter: 1000,
            random_state: None,
        }
    }

    /// Set the transfer strength (higher values = more transfer)
    pub fn transfer_strength(mut self, strength: Float) -> Self {
        self.transfer_strength = strength;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit the transfer learning model
    pub fn fit(
        &self,
        source_X: &ArrayView2<Float>,
        source_y: &ArrayView2<Float>,
        target_X: &ArrayView2<Float>,
        target_y: &ArrayView2<Float>,
    ) -> SklResult<CrossTaskTransferLearning<CrossTaskTransferLearningTrained>> {
        let n_source_samples = source_X.nrows();
        let n_target_samples = target_X.nrows();
        let n_features = source_X.ncols();
        let n_source_tasks = source_y.ncols();
        let n_target_tasks = target_y.ncols();

        if source_X.ncols() != target_X.ncols() {
            return Err(SklearsError::InvalidInput(
                "Source and target data must have the same number of features".to_string(),
            ));
        }

        if n_source_samples != source_y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of source samples must match source labels".to_string(),
            ));
        }

        if n_target_samples != target_y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of target samples must match target labels".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Initialize weights
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();

        let mut source_weights = Array2::<Float>::zeros((n_features, n_source_tasks));
        for i in 0..n_features {
            for j in 0..n_source_tasks {
                source_weights[[i, j]] = rng.sample(normal_dist);
            }
        }

        let mut target_weights = Array2::<Float>::zeros((n_features, n_target_tasks));
        for i in 0..n_features {
            for j in 0..n_target_tasks {
                target_weights[[i, j]] = rng.sample(normal_dist);
            }
        }

        let mut transfer_matrix = Array2::<Float>::zeros((n_source_tasks, n_target_tasks));
        for i in 0..n_source_tasks {
            for j in 0..n_target_tasks {
                transfer_matrix[[i, j]] = rng.sample(normal_dist);
            }
        }

        // Training loop
        for _ in 0..self.max_iter {
            // Update source weights
            let source_pred = source_X.dot(&source_weights);
            let source_error = &source_pred - source_y;
            let source_grad = source_X.t().dot(&source_error) / n_source_samples as Float;
            source_weights -= &(source_grad * self.learning_rate);

            // Update target weights with transfer
            let target_pred = target_X.dot(&target_weights);
            let transferred_pred = target_X.dot(&source_weights).dot(&transfer_matrix);
            let target_error = &target_pred - target_y;
            let transfer_error = &transferred_pred - target_y;

            let target_grad = target_X.t().dot(&target_error) / n_target_samples as Float;
            let transfer_grad = target_X.t().dot(&transfer_error) / n_target_samples as Float;

            target_weights -= &(target_grad * self.learning_rate);
            target_weights -= &(transfer_grad * self.learning_rate * self.transfer_strength);

            // Update transfer matrix
            let transfer_matrix_grad =
                target_X.dot(&source_weights).t().dot(&transfer_error) / n_target_samples as Float;
            transfer_matrix -=
                &(transfer_matrix_grad * self.learning_rate * self.transfer_strength);
        }

        Ok(CrossTaskTransferLearning {
            state: CrossTaskTransferLearningTrained {
                source_weights,
                target_weights,
                transfer_matrix,
                n_features,
                n_source_tasks,
                n_target_tasks,
            },
            transfer_strength: self.transfer_strength,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl CrossTaskTransferLearning<CrossTaskTransferLearningTrained> {
    /// Predict using the trained transfer learning model
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let target_pred = X.dot(&self.state.target_weights);
        Ok(target_pred)
    }

    /// Predict using source task knowledge
    pub fn predict_from_source(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let source_pred = X.dot(&self.state.source_weights);
        let transferred_pred = source_pred.dot(&self.state.transfer_matrix);
        Ok(transferred_pred)
    }

    /// Get the transfer matrix
    pub fn transfer_matrix(&self) -> &Array2<Float> {
        &self.state.transfer_matrix
    }

    /// Get the source weights
    pub fn source_weights(&self) -> &Array2<Float> {
        &self.state.source_weights
    }

    /// Get the target weights
    pub fn target_weights(&self) -> &Array2<Float> {
        &self.state.target_weights
    }
}

impl Default for CrossTaskTransferLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for CrossTaskTransferLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for CrossTaskTransferLearning<CrossTaskTransferLearningTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Domain Adaptation for Multi-Task Learning
///
/// Implements domain adaptation techniques to transfer knowledge
/// between different domains in multi-task settings.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::transfer_learning::DomainAdaptation;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let source_data = array![[1.0, 2.0], [2.0, 3.0]];
/// let source_labels = array![[1.0], [0.0]];
/// let target_data = array![[1.1, 2.1], [2.1, 3.1]];
/// let target_labels = array![[1.0], [0.0]];
///
/// let adaptation = DomainAdaptation::new()
///     .adaptation_strength(0.3)
///     .learning_rate(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct DomainAdaptation<S = Untrained> {
    state: S,
    adaptation_strength: Float,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct DomainAdaptationTrained {
    feature_extractor: Array2<Float>,
    classifier: Array2<Float>,
    domain_discriminator: Array2<Float>,
    n_features: usize,
    n_tasks: usize,
}

impl DomainAdaptation<Untrained> {
    /// Create a new DomainAdaptation instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            adaptation_strength: 0.3,
            learning_rate: 0.01,
            max_iter: 1000,
            random_state: None,
        }
    }

    /// Set the adaptation strength
    pub fn adaptation_strength(mut self, strength: Float) -> Self {
        self.adaptation_strength = strength;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit the domain adaptation model
    pub fn fit(
        &self,
        source_X: &ArrayView2<Float>,
        source_y: &ArrayView2<Float>,
        target_X: &ArrayView2<Float>,
        target_y: &ArrayView2<Float>,
    ) -> SklResult<DomainAdaptation<DomainAdaptationTrained>> {
        let n_source_samples = source_X.nrows();
        let n_target_samples = target_X.nrows();
        let n_features = source_X.ncols();
        let n_tasks = source_y.ncols();

        if source_X.ncols() != target_X.ncols() {
            return Err(SklearsError::InvalidInput(
                "Source and target data must have the same number of features".to_string(),
            ));
        }

        if n_source_samples != source_y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of source samples must match source labels".to_string(),
            ));
        }

        if n_target_samples != target_y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of target samples must match target labels".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Initialize networks
        let hidden_dim = (n_features + n_tasks) / 2;
        let mut feature_extractor = Array2::<Float>::zeros((n_features, hidden_dim));
        let normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..hidden_dim {
                feature_extractor[[i, j]] = rng.sample(normal_dist);
            }
        }
        let mut classifier = Array2::<Float>::zeros((hidden_dim, n_tasks));
        let classifier_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..hidden_dim {
            for j in 0..n_tasks {
                classifier[[i, j]] = rng.sample(classifier_normal_dist);
            }
        }
        let mut domain_discriminator = Array2::<Float>::zeros((hidden_dim, 1));
        let discriminator_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..hidden_dim {
            domain_discriminator[[i, 0]] = rng.sample(discriminator_normal_dist);
        }

        // Create domain labels (0 for source, 1 for target)
        let mut domain_labels = Array2::<Float>::zeros((n_source_samples + n_target_samples, 1));
        for i in n_source_samples..(n_source_samples + n_target_samples) {
            domain_labels[(i, 0)] = 1.0;
        }

        // Combine data
        let mut combined_X =
            Array2::<Float>::zeros((n_source_samples + n_target_samples, n_features));
        combined_X
            .slice_mut(s![..n_source_samples, ..])
            .assign(source_X);
        combined_X
            .slice_mut(s![n_source_samples.., ..])
            .assign(target_X);

        // Training loop
        for _ in 0..self.max_iter {
            // Extract features
            let features = combined_X.dot(&feature_extractor);
            let source_features = features.slice(s![..n_source_samples, ..]);
            let target_features = features.slice(s![n_source_samples.., ..]);

            // Train classifier on source domain
            let source_pred = source_features.dot(&classifier);
            let classification_error = &source_pred - source_y;
            let classifier_grad =
                source_features.t().dot(&classification_error) / n_source_samples as Float;
            classifier -= &(&classifier_grad * self.learning_rate);

            // Train domain discriminator (distinguish source from target)
            let domain_pred = features.dot(&domain_discriminator);
            let domain_error = &domain_pred - &domain_labels;
            let discriminator_grad =
                features.t().dot(&domain_error) / (n_source_samples + n_target_samples) as Float;
            domain_discriminator -= &(&discriminator_grad * self.learning_rate);

            // Update feature extractor (adversarial training)
            let feat_class_grad =
                combined_X.t().dot(&features.dot(&classifier_grad.t())) / n_source_samples as Float;
            let feat_domain_grad = combined_X.t().dot(&features.dot(&discriminator_grad))
                / (n_source_samples + n_target_samples) as Float;

            feature_extractor -= &(feat_class_grad * self.learning_rate);
            feature_extractor +=
                &(feat_domain_grad * self.learning_rate * self.adaptation_strength);
            // Adversarial
        }

        Ok(DomainAdaptation {
            state: DomainAdaptationTrained {
                feature_extractor,
                classifier,
                domain_discriminator,
                n_features,
                n_tasks,
            },
            adaptation_strength: self.adaptation_strength,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl DomainAdaptation<DomainAdaptationTrained> {
    /// Predict using the trained domain adaptation model
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let features = X.dot(&self.state.feature_extractor);
        let predictions = features.dot(&self.state.classifier);
        Ok(predictions)
    }

    /// Extract domain-invariant features
    pub fn extract_features(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let features = X.dot(&self.state.feature_extractor);
        Ok(features)
    }

    /// Predict domain labels (0 for source-like, 1 for target-like)
    pub fn predict_domain(&self, X: &ArrayView2<Float>) -> SklResult<Array1<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let features = X.dot(&self.state.feature_extractor);
        let domain_pred = features.dot(&self.state.domain_discriminator);
        Ok(domain_pred.column(0).to_owned())
    }
}

impl Default for DomainAdaptation<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for DomainAdaptation<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for DomainAdaptation<DomainAdaptationTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Progressive Transfer Learning
///
/// Implements progressive transfer learning where tasks are learned
/// sequentially, with knowledge from earlier tasks helping later ones.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::transfer_learning::ProgressiveTransferLearning;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
///
/// let transfer = ProgressiveTransferLearning::new()
///     .transfer_strength(0.4)
///     .learning_rate(0.01)
///     .max_iter(500);
/// ```
#[derive(Debug, Clone)]
pub struct ProgressiveTransferLearning<S = Untrained> {
    state: S,
    transfer_strength: Float,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ProgressiveTransferLearningTrained {
    task_weights: Vec<Array2<Float>>,
    shared_weights: Array2<Float>,
    task_order: Vec<usize>,
    n_features: usize,
    n_tasks: usize,
}

impl ProgressiveTransferLearning<Untrained> {
    /// Create a new ProgressiveTransferLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            transfer_strength: 0.4,
            learning_rate: 0.01,
            max_iter: 500,
            random_state: None,
        }
    }

    /// Set the transfer strength
    pub fn transfer_strength(mut self, strength: Float) -> Self {
        self.transfer_strength = strength;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit the progressive transfer learning model
    pub fn fit(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
        task_order: Option<Vec<usize>>,
    ) -> SklResult<ProgressiveTransferLearning<ProgressiveTransferLearningTrained>> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let n_tasks = y.ncols();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match number of labels".to_string(),
            ));
        }

        let mut rng = thread_rng();

        // Determine task order
        let task_order = task_order.unwrap_or_else(|| (0..n_tasks).collect());

        // Initialize shared weights
        let mut shared_weights = Array2::<Float>::zeros((n_features, n_features));
        let shared_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..n_features {
                shared_weights[[i, j]] = rng.sample(shared_normal_dist);
            }
        }

        let mut task_weights = Vec::with_capacity(n_tasks);

        // Train tasks progressively
        for &task_idx in &task_order {
            let task_y = y.column(task_idx);

            // Initialize task-specific weights
            let mut task_weight = Array2::<Float>::zeros((n_features, 1));
            let task_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
            for i in 0..n_features {
                task_weight[[i, 0]] = rng.sample(task_normal_dist);
            }

            // Train this task
            for _ in 0..self.max_iter {
                // Compute shared features
                let shared_features = X.dot(&shared_weights);

                // Compute task prediction
                let task_pred = shared_features.dot(&task_weight);
                let task_error = &task_pred.column(0) - &task_y;

                // Update task weights
                let task_error_2d = task_error.insert_axis(Axis(1));
                let task_grad = shared_features.t().dot(&task_error_2d) / n_samples as Float;
                task_weight -= &(&task_grad * self.learning_rate);

                // Update shared weights (transfer from previous tasks)
                if !task_weights.is_empty() {
                    let shared_grad =
                        X.t().dot(&task_error_2d.dot(&task_weight.t())) / n_samples as Float;
                    shared_weights -= &(shared_grad * self.learning_rate * self.transfer_strength);
                }
            }

            task_weights.push(task_weight);
        }

        Ok(ProgressiveTransferLearning {
            state: ProgressiveTransferLearningTrained {
                task_weights,
                shared_weights,
                task_order,
                n_features,
                n_tasks,
            },
            transfer_strength: self.transfer_strength,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl ProgressiveTransferLearning<ProgressiveTransferLearningTrained> {
    /// Predict using the trained progressive transfer learning model
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let shared_features = X.dot(&self.state.shared_weights);
        let mut predictions = Array2::<Float>::zeros((n_samples, self.state.n_tasks));

        for (i, &task_idx) in self.state.task_order.iter().enumerate() {
            let task_pred = shared_features.dot(&self.state.task_weights[i]);
            predictions
                .column_mut(task_idx)
                .assign(&task_pred.column(0));
        }

        Ok(predictions)
    }

    /// Get the shared weights
    pub fn shared_weights(&self) -> &Array2<Float> {
        &self.state.shared_weights
    }

    /// Get the task-specific weights
    pub fn task_weights(&self) -> &Vec<Array2<Float>> {
        &self.state.task_weights
    }

    /// Get the task order
    pub fn task_order(&self) -> &Vec<usize> {
        &self.state.task_order
    }
}

impl Default for ProgressiveTransferLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ProgressiveTransferLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for ProgressiveTransferLearning<ProgressiveTransferLearningTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Continual Learning for Multi-Task Learning
///
/// Implements continual learning where new tasks are learned sequentially
/// without forgetting previously learned tasks using elastic weight consolidation.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::transfer_learning::ContinualLearning;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
///
/// let continual = ContinualLearning::new()
///     .importance_weight(1000.0)
///     .learning_rate(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct ContinualLearning<S = Untrained> {
    state: S,
    importance_weight: Float,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ContinualLearningTrained {
    task_weights: Vec<Array2<Float>>,
    fisher_information: Array2<Float>,
    optimal_weights: Array2<Float>,
    n_features: usize,
    n_tasks: usize,
}

impl Default for ContinualLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ContinualLearning<Untrained> {
    /// Create a new ContinualLearning instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            importance_weight: 1000.0,
            learning_rate: 0.01,
            max_iter: 1000,
            random_state: None,
        }
    }

    /// Set the importance weight for preventing forgetting
    pub fn importance_weight(mut self, weight: Float) -> Self {
        self.importance_weight = weight;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit the continual learning model
    pub fn fit(
        &self,
        tasks_X: &[ArrayView2<Float>],
        tasks_y: &[ArrayView2<Float>],
    ) -> SklResult<ContinualLearning<ContinualLearningTrained>> {
        if tasks_X.len() != tasks_y.len() {
            return Err(SklearsError::InvalidInput(
                "Number of X and y task arrays must match".to_string(),
            ));
        }

        if tasks_X.is_empty() {
            return Err(SklearsError::InvalidInput("No tasks provided".to_string()));
        }

        let n_features = tasks_X[0].ncols();
        let n_tasks = tasks_y[0].ncols();

        // Initialize with random weights
        let mut rng = thread_rng();

        let mut weights = Array2::<Float>::zeros((n_features, n_tasks));
        let weights_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..n_tasks {
                weights[[i, j]] = rng.sample(weights_normal_dist);
            }
        }
        let mut fisher_information = Array2::<Float>::zeros((n_features, n_tasks));
        let mut task_weights = Vec::new();

        // Learn tasks sequentially
        for (task_idx, (X, y)) in tasks_X.iter().zip(tasks_y.iter()).enumerate() {
            if X.nrows() != y.nrows() {
                return Err(SklearsError::InvalidInput(
                    "Number of samples in X and y must match".to_string(),
                ));
            }

            // Store weights before learning new task
            let old_weights = weights.clone();

            // Learn current task
            for _ in 0..self.max_iter {
                let predictions = X.dot(&weights);
                let errors = &predictions - y;
                let gradient = X.t().dot(&errors) / X.nrows() as Float;

                // Add elastic weight consolidation penalty for previous tasks
                if task_idx > 0 {
                    let penalty =
                        &fisher_information * (&weights - &old_weights) * self.importance_weight;
                    weights = &weights - self.learning_rate * (&gradient + penalty);
                } else {
                    weights = &weights - self.learning_rate * &gradient;
                }
            }

            // Update Fisher information matrix
            let predictions = X.dot(&weights);
            let errors = &predictions - y;
            let grad_squared = X.t().dot(&errors.mapv(|x| x * x)) / X.nrows() as Float;
            fisher_information = &fisher_information + grad_squared;

            task_weights.push(weights.clone());
        }

        Ok(ContinualLearning {
            state: ContinualLearningTrained {
                task_weights,
                fisher_information,
                optimal_weights: weights,
                n_features,
                n_tasks,
            },
            importance_weight: self.importance_weight,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl ContinualLearning<ContinualLearningTrained> {
    /// Predict using the continual learning model
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        Ok(X.dot(&self.state.optimal_weights))
    }

    /// Get the task weights
    pub fn task_weights(&self) -> &[Array2<Float>] {
        &self.state.task_weights
    }

    /// Get the Fisher information matrix
    pub fn fisher_information(&self) -> &Array2<Float> {
        &self.state.fisher_information
    }
}

impl Estimator for ContinualLearning<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for ContinualLearning<ContinualLearningTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

/// Knowledge Distillation for Multi-Task Learning
///
/// Implements knowledge distillation where a smaller student network learns
/// from a larger teacher network for improved efficiency and performance.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::transfer_learning::KnowledgeDistillation;
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
/// let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
///
/// let distillation = KnowledgeDistillation::new()
///     .temperature(3.0)
///     .alpha(0.7)
///     .learning_rate(0.01);
/// ```
#[derive(Debug, Clone)]
pub struct KnowledgeDistillation<S = Untrained> {
    state: S,
    temperature: Float,
    alpha: Float,
    learning_rate: Float,
    max_iter: usize,
    random_state: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct KnowledgeDistillationTrained {
    student_weights: Array2<Float>,
    teacher_weights: Array2<Float>,
    n_features: usize,
    n_tasks: usize,
}

impl Default for KnowledgeDistillation<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl KnowledgeDistillation<Untrained> {
    /// Create a new KnowledgeDistillation instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            temperature: 3.0,
            alpha: 0.7,
            learning_rate: 0.01,
            max_iter: 1000,
            random_state: None,
        }
    }

    /// Set the temperature for softening teacher predictions
    pub fn temperature(mut self, temp: Float) -> Self {
        self.temperature = temp;
        self
    }

    /// Set the alpha parameter for balancing hard and soft targets
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }

    /// Fit the knowledge distillation model
    pub fn fit(
        &self,
        X: &ArrayView2<Float>,
        y: &ArrayView2<Float>,
        teacher_predictions: &ArrayView2<Float>,
    ) -> SklResult<KnowledgeDistillation<KnowledgeDistillationTrained>> {
        if X.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and y must match".to_string(),
            ));
        }

        if X.nrows() != teacher_predictions.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples in X and teacher predictions must match".to_string(),
            ));
        }

        let n_features = X.ncols();
        let n_tasks = y.ncols();

        // Initialize with random weights
        let mut rng = thread_rng();

        let mut student_weights = Array2::<Float>::zeros((n_features, n_tasks));
        let student_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..n_tasks {
                student_weights[[i, j]] = rng.sample(student_normal_dist);
            }
        }
        let mut teacher_weights = Array2::<Float>::zeros((n_features, n_tasks));
        let teacher_normal_dist = RandNormal::new(0.0, 0.1).unwrap();
        for i in 0..n_features {
            for j in 0..n_tasks {
                teacher_weights[[i, j]] = rng.sample(teacher_normal_dist);
            }
        }

        // Train student network
        for _ in 0..self.max_iter {
            let student_predictions = X.dot(&student_weights);

            // Soft targets from teacher (temperature-scaled)
            let soft_targets = teacher_predictions / self.temperature;
            let student_soft = &student_predictions / self.temperature;

            // Combined loss: weighted sum of hard and soft targets
            let hard_loss = &student_predictions - y;
            let soft_loss = &student_soft - &soft_targets;

            let combined_loss = (1.0 - self.alpha) * hard_loss + self.alpha * soft_loss;
            let gradient = X.t().dot(&combined_loss) / X.nrows() as Float;

            student_weights = &student_weights - self.learning_rate * &gradient;
        }

        Ok(KnowledgeDistillation {
            state: KnowledgeDistillationTrained {
                student_weights,
                teacher_weights,
                n_features,
                n_tasks,
            },
            temperature: self.temperature,
            alpha: self.alpha,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            random_state: self.random_state,
        })
    }
}

impl KnowledgeDistillation<KnowledgeDistillationTrained> {
    /// Predict using the student network
    pub fn predict(&self, X: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        if X.ncols() != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features must match training data".to_string(),
            ));
        }

        Ok(X.dot(&self.state.student_weights))
    }

    /// Get the student weights
    pub fn student_weights(&self) -> &Array2<Float> {
        &self.state.student_weights
    }

    /// Get the teacher weights
    pub fn teacher_weights(&self) -> &Array2<Float> {
        &self.state.teacher_weights
    }
}

impl Estimator for KnowledgeDistillation<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for KnowledgeDistillation<KnowledgeDistillationTrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cross_task_transfer_learning_basic() {
        let source_X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let source_y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]];
        let target_X = array![[1.1, 2.1], [2.1, 3.1]];
        let target_y = array![[1.0, 0.0], [0.0, 1.0]];

        let transfer = CrossTaskTransferLearning::new()
            .transfer_strength(0.5)
            .learning_rate(0.01)
            .max_iter(100)
            .random_state(Some(42));

        let trained = transfer
            .fit(
                &source_X.view(),
                &source_y.view(),
                &target_X.view(),
                &target_y.view(),
            )
            .unwrap();

        let predictions = trained.predict(&target_X.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 2));

        let source_predictions = trained.predict_from_source(&target_X.view()).unwrap();
        assert_eq!(source_predictions.dim(), (2, 2));
    }

    #[test]
    fn test_cross_task_transfer_learning_validation() {
        let source_X = array![[1.0, 2.0], [2.0, 3.0]];
        let source_y = array![[1.0, 0.0], [0.0, 1.0]];
        let target_X = array![[1.1, 2.1, 3.1]]; // Different number of features
        let target_y = array![[1.0, 0.0]];

        let transfer = CrossTaskTransferLearning::new();

        // Should fail due to feature mismatch
        assert!(transfer
            .fit(
                &source_X.view(),
                &source_y.view(),
                &target_X.view(),
                &target_y.view()
            )
            .is_err());
    }

    #[test]
    fn test_domain_adaptation_basic() {
        let source_X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let source_y = array![[1.0], [0.0], [1.0], [0.0]];
        let target_X = array![[1.1, 2.1], [2.1, 3.1]];
        let target_y = array![[1.0], [0.0]];

        let adaptation = DomainAdaptation::new()
            .adaptation_strength(0.3)
            .learning_rate(0.01)
            .max_iter(100)
            .random_state(Some(42));

        let trained = adaptation
            .fit(
                &source_X.view(),
                &source_y.view(),
                &target_X.view(),
                &target_y.view(),
            )
            .unwrap();

        let predictions = trained.predict(&target_X.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 1));

        let features = trained.extract_features(&target_X.view()).unwrap();
        assert_eq!(features.ncols(), 1); // Hidden dimension

        let domain_pred = trained.predict_domain(&target_X.view()).unwrap();
        assert_eq!(domain_pred.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_progressive_transfer_learning_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0]
        ];

        let transfer = ProgressiveTransferLearning::new()
            .transfer_strength(0.4)
            .learning_rate(0.01)
            .max_iter(100)
            .random_state(Some(42));

        let trained = transfer.fit(&X.view(), &y.view(), None).unwrap();

        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 3));

        // Check that we have weights for all tasks
        assert_eq!(trained.task_weights().len(), 3);
        assert_eq!(trained.task_order().len(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_progressive_transfer_learning_custom_order() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]];

        let transfer = ProgressiveTransferLearning::new().random_state(Some(42));

        let custom_order = vec![2, 0, 1]; // Start with task 2, then 0, then 1
        let trained = transfer
            .fit(&X.view(), &y.view(), Some(custom_order.clone()))
            .unwrap();

        assert_eq!(trained.task_order(), &custom_order);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_transfer_learning_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]; // Mismatched samples

        let transfer = ProgressiveTransferLearning::new();
        assert!(transfer.fit(&X.view(), &y.view(), None).is_err());
    }

    #[test]
    fn test_continual_learning_basic() {
        let X1 = array![[1.0, 2.0], [2.0, 3.0]];
        let y1 = array![[1.0, 0.0], [0.0, 1.0]];
        let X2 = array![[3.0, 1.0], [1.0, 3.0]];
        let y2 = array![[1.0, 1.0], [0.0, 0.0]];

        let tasks_X = vec![X1.view(), X2.view()];
        let tasks_y = vec![y1.view(), y2.view()];

        let continual = ContinualLearning::new()
            .importance_weight(1000.0)
            .learning_rate(0.01)
            .max_iter(100)
            .random_state(Some(42));

        let trained = continual.fit(&tasks_X, &tasks_y).unwrap();

        let predictions = trained.predict(&X1.view()).unwrap();
        assert_eq!(predictions.dim(), (2, 2));

        // Check that we have weights for both tasks
        assert_eq!(trained.task_weights().len(), 2);
        assert_eq!(trained.fisher_information().dim(), (2, 2));
    }

    #[test]
    fn test_continual_learning_error_handling() {
        let X1 = array![[1.0, 2.0], [2.0, 3.0]];
        let y1 = array![[1.0, 0.0], [0.0, 1.0]];
        let X2 = array![[3.0, 1.0]]; // Wrong number of samples
        let y2 = array![[1.0, 1.0], [0.0, 0.0]];

        let tasks_X = vec![X1.view(), X2.view()];
        let tasks_y = vec![y1.view(), y2.view()];

        let continual = ContinualLearning::new();
        assert!(continual.fit(&tasks_X, &tasks_y).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_knowledge_distillation_basic() {
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let teacher_predictions = array![[0.9, 0.1], [0.1, 0.9], [0.8, 0.8]];

        let distillation = KnowledgeDistillation::new()
            .temperature(3.0)
            .alpha(0.7)
            .learning_rate(0.01)
            .max_iter(100)
            .random_state(Some(42));

        let trained = distillation
            .fit(&X.view(), &y.view(), &teacher_predictions.view())
            .unwrap();

        let predictions = trained.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (3, 2));

        // Check that we have student and teacher weights
        assert_eq!(trained.student_weights().dim(), (2, 2));
        assert_eq!(trained.teacher_weights().dim(), (2, 2));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_knowledge_distillation_error_handling() {
        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.0, 0.0], [0.0, 1.0]];
        let teacher_predictions = array![[0.9, 0.1], [0.1, 0.9], [0.8, 0.8]]; // Wrong number of samples

        let distillation = KnowledgeDistillation::new();
        assert!(distillation
            .fit(&X.view(), &y.view(), &teacher_predictions.view())
            .is_err());
    }

    #[test]
    fn test_knowledge_distillation_configuration() {
        let distillation = KnowledgeDistillation::new()
            .temperature(5.0)
            .alpha(0.5)
            .learning_rate(0.001)
            .max_iter(2000)
            .random_state(Some(123));

        // Test configuration parameters
        assert_eq!(distillation.temperature, 5.0);
        assert_eq!(distillation.alpha, 0.5);
        assert_eq!(distillation.learning_rate, 0.001);
        assert_eq!(distillation.max_iter, 2000);
        assert_eq!(distillation.random_state, Some(123));
    }
}

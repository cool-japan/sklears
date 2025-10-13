//! Continual Learning Naive Bayes Implementation
//!
//! This module implements Continual Learning approaches for Naive Bayes classifiers that can
//! learn from streaming data while avoiding catastrophic forgetting. It includes methods like
//! Elastic Weight Consolidation (EWC), Memory Replay, and Progressive Neural Networks adapted
//! for Naive Bayes.

use scirs2_core::numeric::{Float, NumCast};
// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};

// Type aliases for compatibility with DMatrix/DVector usage
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
// TODO: Migrate to full scirs2_core::random when API is stabilized
use scirs2_core::random::{thread_rng, Rng};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ContinualLearningError {
    #[error("Memory buffer full")]
    MemoryBufferFull,
    #[error("Invalid task ID: {0}")]
    InvalidTaskId(usize),
    #[error("Insufficient data for task")]
    InsufficientData,
    #[error("Feature dimension mismatch: expected {expected}, got {actual}")]
    FeatureMismatch { expected: usize, actual: usize },
    #[error("Concept drift detection error: {0}")]
    ConceptDriftError(String),
    #[error("Knowledge distillation error: {0}")]
    DistillationError(String),
}

type Result<T> = std::result::Result<T, ContinualLearningError>;

/// Continual learning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContinualLearningStrategy {
    /// Elastic Weight Consolidation
    ElasticWeightConsolidation,
    /// Memory replay with stored examples
    MemoryReplay,
    /// Progressive learning with task-specific components
    ProgressiveLearning,
    /// Knowledge distillation
    KnowledgeDistillation,
    /// Hybrid approach combining multiple strategies
    Hybrid,
}

/// Concept drift detection methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DriftDetectionMethod {
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Page-Hinkley test
    PageHinkley,
    /// ADWIN (Adaptive Windowing)
    ADWIN,
    /// Statistical distance measures
    StatisticalDistance,
    /// Ensemble-based detection
    EnsembleBased,
}

/// Memory management strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Random sampling
    Random,
    /// Herding-based selection
    Herding,
    /// Gradient-based selection
    GradientBased,
    /// Uncertainty-based selection
    UncertaintyBased,
    /// Class-balanced selection
    ClassBalanced,
}

/// Configuration for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContinualLearningConfig {
    /// Learning strategy
    pub strategy: ContinualLearningStrategy,
    /// Memory buffer size for replay
    pub memory_size: usize,
    /// Memory management strategy
    pub memory_strategy: MemoryStrategy,
    /// Drift detection method
    pub drift_detection: DriftDetectionMethod,
    /// EWC regularization strength
    pub ewc_lambda: f64,
    /// Learning rate for parameter updates
    pub learning_rate: f64,
    /// Forgetting factor for old knowledge
    pub forgetting_factor: f64,
    /// Drift detection threshold
    pub drift_threshold: f64,
    /// Knowledge distillation temperature
    pub distillation_temperature: f64,
    /// Window size for drift detection
    pub drift_window_size: usize,
    /// Enable task-specific batch normalization
    pub task_specific_bn: bool,
    /// Maximum iterations for learning algorithms
    pub max_iterations: usize,
}

impl Default for ContinualLearningConfig {
    fn default() -> Self {
        Self {
            strategy: ContinualLearningStrategy::Hybrid,
            memory_size: 1000,
            memory_strategy: MemoryStrategy::ClassBalanced,
            drift_detection: DriftDetectionMethod::ADWIN,
            ewc_lambda: 1000.0,
            learning_rate: 0.01,
            forgetting_factor: 0.95,
            drift_threshold: 0.05,
            distillation_temperature: 3.0,
            drift_window_size: 100,
            task_specific_bn: true,
            max_iterations: 100,
        }
    }
}

/// Task metadata for continual learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetadata {
    pub task_id: usize,
    pub num_samples: usize,
    pub num_classes: usize,
    #[serde(skip)]
    pub feature_statistics: Option<DVector<f64>>,
    pub class_distribution: HashMap<i32, f64>,
    pub timestamp: u64,
}

/// Memory sample for replay
#[derive(Debug, Clone)]
pub struct MemorySample<T: Float> {
    pub features: DVector<T>,
    pub label: i32,
    pub task_id: usize,
    pub importance_weight: T,
    pub timestamp: u64,
}

/// Fisher Information Matrix for EWC
#[derive(Debug, Clone)]
pub struct FisherInformation<T: Float> {
    pub class_priors: HashMap<i32, T>,
    pub feature_means: HashMap<i32, DVector<T>>,
    pub feature_variances: HashMap<i32, DVector<T>>,
}

/// Continual Learning Naive Bayes Classifier
#[derive(Debug, Clone)]
pub struct ContinualLearningNB<T: Float> {
    /// Configuration
    config: ContinualLearningConfig,
    /// Current task parameters
    current_task_params: HashMap<i32, (DVector<T>, DVector<T>)>, // (means, variances)
    /// Previous task parameters for EWC
    previous_task_params: Vec<HashMap<i32, (DVector<T>, DVector<T>)>>,
    /// Fisher information matrices for EWC
    fisher_information: Vec<FisherInformation<T>>,
    /// Memory buffer for replay
    memory_buffer: VecDeque<MemorySample<T>>,
    /// Task metadata
    task_metadata: HashMap<usize, TaskMetadata>,
    /// Current task ID
    current_task_id: usize,
    /// Drift detection state
    drift_detector: DriftDetector<T>,
    /// Class priors
    class_priors: HashMap<i32, T>,
    /// Number of features
    n_features: usize,
    /// All classes seen so far
    all_classes: Vec<i32>,
    /// Training flag
    is_fitted: bool,
    _phantom: PhantomData<T>,
}

/// Concept drift detector
#[derive(Debug, Clone)]
pub struct DriftDetector<T: Float> {
    method: DriftDetectionMethod,
    reference_window: VecDeque<T>,
    current_window: VecDeque<T>,
    window_size: usize,
    drift_threshold: T,
    ph_sum: T,                               // For Page-Hinkley
    ph_sum_min: T,                           // For Page-Hinkley
    adwin_buckets: VecDeque<ADWINBucket<T>>, // For ADWIN
}

#[derive(Debug, Clone)]
struct ADWINBucket<T: Float> {
    sum: T,
    count: usize,
    variance: T,
}

impl<T: Float + Clone + std::fmt::Debug + 'static> ContinualLearningNB<T>
where
    T: Float + Copy,
    T: From<f64> + Into<f64>,
    T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum,
    T: scirs2_core::ndarray::ScalarOperand,
    T: std::ops::DivAssign,
{
    /// Create a new Continual Learning Naive Bayes classifier
    pub fn new(config: ContinualLearningConfig) -> Self {
        let drift_detector = DriftDetector::new(
            config.drift_detection.clone(),
            config.drift_window_size,
            NumCast::from(config.drift_threshold).unwrap_or_else(T::zero),
        );

        Self {
            config,
            current_task_params: HashMap::new(),
            previous_task_params: Vec::new(),
            fisher_information: Vec::new(),
            memory_buffer: VecDeque::new(),
            task_metadata: HashMap::new(),
            current_task_id: 0,
            drift_detector,
            class_priors: HashMap::new(),
            n_features: 0,
            all_classes: Vec::new(),
            is_fitted: false,
            _phantom: PhantomData,
        }
    }

    /// Fit on a new task
    pub fn fit_task(&mut self, X: &DMatrix<T>, y: &[i32], task_id: usize) -> Result<()> {
        if X.nrows() != y.len() {
            return Err(ContinualLearningError::FeatureMismatch {
                expected: X.nrows(),
                actual: y.len(),
            });
        }

        if X.nrows() == 0 {
            return Err(ContinualLearningError::InsufficientData);
        }

        // Initialize if first task
        if !self.is_fitted {
            self.n_features = X.ncols();
            self.is_fitted = true;
        } else if X.ncols() != self.n_features {
            return Err(ContinualLearningError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }

        // Store previous task parameters for EWC
        if !self.current_task_params.is_empty() {
            self.previous_task_params
                .push(self.current_task_params.clone());

            // Compute Fisher Information Matrix for EWC
            if matches!(
                self.config.strategy,
                ContinualLearningStrategy::ElasticWeightConsolidation
                    | ContinualLearningStrategy::Hybrid
            ) {
                let fisher = self.compute_fisher_information(X, y)?;
                self.fisher_information.push(fisher);
            }
        }

        self.current_task_id = task_id;

        // Store task metadata
        let metadata = self.create_task_metadata(task_id, X, y);
        self.task_metadata.insert(task_id, metadata);

        // Update classes
        self.update_classes(y);

        // Learn new task based on strategy
        match self.config.strategy {
            ContinualLearningStrategy::ElasticWeightConsolidation => {
                self.fit_with_ewc(X, y)?;
            }
            ContinualLearningStrategy::MemoryReplay => {
                self.fit_with_memory_replay(X, y)?;
            }
            ContinualLearningStrategy::ProgressiveLearning => {
                self.fit_with_progressive_learning(X, y)?;
            }
            ContinualLearningStrategy::KnowledgeDistillation => {
                self.fit_with_knowledge_distillation(X, y)?;
            }
            ContinualLearningStrategy::Hybrid => {
                self.fit_with_hybrid_approach(X, y)?;
            }
        }

        // Update memory buffer
        self.update_memory_buffer(X, y, task_id)?;

        Ok(())
    }

    /// Predict with current knowledge
    pub fn predict(&self, X: &DMatrix<T>) -> Result<Vec<i32>> {
        if !self.is_fitted {
            return Err(ContinualLearningError::InsufficientData);
        }

        if X.ncols() != self.n_features {
            return Err(ContinualLearningError::FeatureMismatch {
                expected: self.n_features,
                actual: X.ncols(),
            });
        }

        let probabilities = self.predict_proba(X)?;
        let mut predictions = Vec::with_capacity(X.nrows());

        for row in probabilities.axis_iter(Axis(0)) {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions.push(self.all_classes[max_idx]);
        }

        Ok(predictions)
    }

    /// Predict class probabilities
    pub fn predict_proba(&self, X: &DMatrix<T>) -> Result<DMatrix<T>> {
        if !self.is_fitted {
            return Err(ContinualLearningError::InsufficientData);
        }

        let n_samples = X.nrows();
        let n_classes = self.all_classes.len();
        let mut probabilities = DMatrix::zeros((n_samples, n_classes));

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            for (class_idx, &class) in self.all_classes.iter().enumerate() {
                let prob = self.compute_class_probability(&sample.t().to_owned(), class);
                probabilities[(sample_idx, class_idx)] = prob;
            }
        }

        // Normalize probabilities
        for mut row in probabilities.axis_iter_mut(Axis(0)) {
            let sum = row.sum();
            if sum > T::zero() {
                row /= sum;
            }
        }

        Ok(probabilities)
    }

    /// Detect concept drift
    pub fn detect_drift(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<bool> {
        // Simplified drift detection based on prediction accuracy
        let predictions = self.predict(X)?;
        let accuracy = predictions
            .iter()
            .zip(y.iter())
            .map(|(pred, actual)| if pred == actual { T::one() } else { T::zero() })
            .sum::<T>()
            / NumCast::from(y.len() as f64).unwrap_or_else(T::one);

        self.drift_detector.add_sample(accuracy)
    }

    /// Get memory buffer size
    pub fn memory_buffer_size(&self) -> usize {
        self.memory_buffer.len()
    }

    /// Get number of tasks learned
    pub fn num_tasks(&self) -> usize {
        self.task_metadata.len()
    }

    /// Get task metadata
    pub fn get_task_metadata(&self, task_id: usize) -> Option<&TaskMetadata> {
        self.task_metadata.get(&task_id)
    }

    /// Fit with Elastic Weight Consolidation
    fn fit_with_ewc(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        // Standard Naive Bayes parameter estimation
        self.fit_standard_nb(X, y)?;

        // Apply EWC regularization if we have previous tasks
        if !self.fisher_information.is_empty() {
            self.apply_ewc_regularization()?;
        }

        Ok(())
    }

    /// Fit with memory replay
    fn fit_with_memory_replay(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        // Combine current data with memory buffer
        let (combined_X, combined_y) = self.combine_with_memory(X, y)?;

        // Train on combined data
        self.fit_standard_nb(&combined_X, &combined_y)?;

        Ok(())
    }

    /// Fit with progressive learning
    fn fit_with_progressive_learning(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        // Create task-specific parameters
        let task_params = self.learn_task_specific_parameters(X, y)?;

        // Combine with global parameters
        self.integrate_task_parameters(task_params)?;

        Ok(())
    }

    /// Fit with knowledge distillation
    fn fit_with_knowledge_distillation(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        // Get soft targets from previous model
        let soft_targets = if self.is_fitted {
            Some(self.predict_proba(X)?)
        } else {
            None
        };

        // Train new model
        self.fit_standard_nb(X, y)?;

        // Apply knowledge distillation if we have soft targets
        if let Some(targets) = soft_targets {
            self.apply_knowledge_distillation(&targets, X)?;
        }

        Ok(())
    }

    /// Fit with hybrid approach
    fn fit_with_hybrid_approach(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        // Combine memory replay with EWC
        let (combined_X, combined_y) = self.combine_with_memory(X, y)?;
        self.fit_standard_nb(&combined_X, &combined_y)?;

        if !self.fisher_information.is_empty() {
            self.apply_ewc_regularization()?;
        }

        Ok(())
    }

    /// Standard Naive Bayes fitting
    fn fit_standard_nb(&mut self, X: &DMatrix<T>, y: &[i32]) -> Result<()> {
        self.current_task_params.clear();

        // Compute class priors
        let total_samples = y.len() as f64;
        for &class in &self.all_classes {
            let class_count = y.iter().filter(|&&c| c == class).count() as f64;
            let prior = NumCast::from(class_count / total_samples).unwrap_or_else(T::zero);
            self.class_priors.insert(class, prior);
        }

        // Compute class-conditional parameters
        for &class in &self.all_classes {
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
                .collect();

            if class_samples.is_empty() {
                continue;
            }

            let mut means = DVector::zeros(self.n_features);
            let mut variances = DVector::zeros(self.n_features);

            for feature_idx in 0..self.n_features {
                let feature_values: Vec<T> = class_samples
                    .iter()
                    .map(|&sample_idx| X[(sample_idx, feature_idx)])
                    .collect();

                let mean = feature_values.iter().cloned().sum::<T>()
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one);
                means[feature_idx] = mean;

                let variance = feature_values
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one)
                    + NumCast::from(1e-9).unwrap_or_else(|| {
                        T::one() / NumCast::from(1000000000.0).unwrap_or_else(T::one)
                    }); // Add small constant for numerical stability

                variances[feature_idx] = variance;
            }

            self.current_task_params.insert(class, (means, variances));
        }

        Ok(())
    }

    /// Compute Fisher Information Matrix
    fn compute_fisher_information(
        &self,
        _X: &DMatrix<T>,
        _y: &[i32],
    ) -> Result<FisherInformation<T>> {
        // Simplified Fisher Information computation
        let mut fisher = FisherInformation {
            class_priors: HashMap::new(),
            feature_means: HashMap::new(),
            feature_variances: HashMap::new(),
        };

        for (&class, prior) in &self.class_priors {
            fisher.class_priors.insert(class, *prior);
        }

        for (&class, (means, variances)) in &self.current_task_params {
            fisher.feature_means.insert(class, means.clone());
            fisher.feature_variances.insert(class, variances.clone());
        }

        Ok(fisher)
    }

    /// Apply EWC regularization
    fn apply_ewc_regularization(&mut self) -> Result<()> {
        let lambda = NumCast::from(self.config.ewc_lambda).unwrap_or_else(T::zero);

        for (class, (current_means, current_vars)) in &mut self.current_task_params {
            // Apply EWC penalty to previous tasks
            for fisher in &self.fisher_information {
                if let (Some(prev_means), Some(prev_vars)) = (
                    fisher.feature_means.get(class),
                    fisher.feature_variances.get(class),
                ) {
                    // Apply regularization (simplified)
                    for i in 0..self.n_features {
                        let penalty = lambda
                            * (current_means[i] - prev_means[i])
                            * (current_means[i] - prev_means[i]);
                        current_vars[i] = current_vars[i] + penalty;
                    }
                }
            }
        }

        Ok(())
    }

    /// Combine current data with memory buffer
    fn combine_with_memory(&self, X: &DMatrix<T>, y: &[i32]) -> Result<(DMatrix<T>, Vec<i32>)> {
        let memory_samples: Vec<_> = self.memory_buffer.iter().collect();
        let total_samples = X.nrows() + memory_samples.len();

        let mut combined_X = DMatrix::zeros((total_samples, self.n_features));
        let mut combined_y = Vec::with_capacity(total_samples);

        // Add current data
        for i in 0..X.nrows() {
            combined_X.row_mut(i).assign(&X.row(i));
            combined_y.push(y[i]);
        }

        // Add memory samples
        for (i, sample) in memory_samples.iter().enumerate() {
            let row_idx = X.nrows() + i;
            combined_X.row_mut(row_idx).assign(&sample.features);
            combined_y.push(sample.label);
        }

        Ok((combined_X, combined_y))
    }

    /// Update memory buffer with new samples
    fn update_memory_buffer(&mut self, X: &DMatrix<T>, y: &[i32], task_id: usize) -> Result<()> {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        // Select samples to add to memory
        let samples_to_add = self.select_memory_samples(X, y, task_id)?;

        for (sample_idx, importance) in samples_to_add {
            if self.memory_buffer.len() >= self.config.memory_size {
                // Remove oldest or least important sample
                self.memory_buffer.pop_front();
            }

            let sample = MemorySample {
                features: X.row(sample_idx).to_owned(),
                label: y[sample_idx],
                task_id,
                importance_weight: importance,
                timestamp: current_time,
            };

            self.memory_buffer.push_back(sample);
        }

        Ok(())
    }

    /// Select samples for memory buffer
    fn select_memory_samples(
        &self,
        X: &DMatrix<T>,
        y: &[i32],
        _task_id: usize,
    ) -> Result<Vec<(usize, T)>> {
        let mut selected = Vec::new();
        let mut rng = thread_rng();

        match self.config.memory_strategy {
            MemoryStrategy::Random => {
                let num_samples = (X.nrows() / 10).max(1).min(100); // Sample 10% or at most 100
                for _ in 0..num_samples {
                    let idx = rng.gen_range(0..X.nrows());
                    selected.push((idx, T::one()));
                }
            }
            MemoryStrategy::ClassBalanced => {
                // Select samples to maintain class balance
                let mut class_counts = HashMap::new();
                for &class in y {
                    *class_counts.entry(class).or_insert(0) += 1;
                }

                let samples_per_class = 10; // Store 10 samples per class
                for (&class, _) in &class_counts {
                    let class_indices: Vec<usize> = y
                        .iter()
                        .enumerate()
                        .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
                        .collect();

                    let num_to_select = samples_per_class.min(class_indices.len());
                    for _ in 0..num_to_select {
                        let idx = class_indices[rng.gen_range(0..class_indices.len())];
                        selected.push((idx, T::one()));
                    }
                }
            }
            _ => {
                // Default to random for other strategies
                let num_samples = (X.nrows() / 10).max(1).min(100);
                for _ in 0..num_samples {
                    let idx = rng.gen_range(0..X.nrows());
                    selected.push((idx, T::one()));
                }
            }
        }

        Ok(selected)
    }

    /// Learn task-specific parameters
    fn learn_task_specific_parameters(
        &self,
        X: &DMatrix<T>,
        y: &[i32],
    ) -> Result<HashMap<i32, (DVector<T>, DVector<T>)>> {
        let mut task_params = HashMap::new();

        for &class in &self.all_classes {
            let class_samples: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(idx, &c)| if c == class { Some(idx) } else { None })
                .collect();

            if class_samples.is_empty() {
                continue;
            }

            let mut means = DVector::zeros(self.n_features);
            let mut variances = DVector::zeros(self.n_features);

            for feature_idx in 0..self.n_features {
                let feature_values: Vec<T> = class_samples
                    .iter()
                    .map(|&sample_idx| X[(sample_idx, feature_idx)])
                    .collect();

                let mean = feature_values.iter().cloned().sum::<T>()
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one);
                means[feature_idx] = mean;

                let variance = feature_values
                    .iter()
                    .fold(T::zero(), |acc, &x| acc + (x - mean) * (x - mean))
                    / NumCast::from(feature_values.len() as f64).unwrap_or_else(T::one)
                    + NumCast::from(1e-9).unwrap_or_else(|| {
                        T::one() / NumCast::from(1000000000.0).unwrap_or_else(T::one)
                    });

                variances[feature_idx] = variance;
            }

            task_params.insert(class, (means, variances));
        }

        Ok(task_params)
    }

    /// Integrate task-specific parameters
    fn integrate_task_parameters(
        &mut self,
        task_params: HashMap<i32, (DVector<T>, DVector<T>)>,
    ) -> Result<()> {
        let learning_rate = NumCast::from(self.config.learning_rate).unwrap_or_else(T::zero);

        for (class, (new_means, new_vars)) in task_params {
            if let Some((current_means, current_vars)) = self.current_task_params.get_mut(&class) {
                // Weighted update
                *current_means =
                    current_means.clone() * (T::one() - learning_rate) + new_means * learning_rate;
                *current_vars =
                    current_vars.clone() * (T::one() - learning_rate) + new_vars * learning_rate;
            } else {
                self.current_task_params
                    .insert(class, (new_means, new_vars));
            }
        }

        Ok(())
    }

    /// Apply knowledge distillation
    fn apply_knowledge_distillation(
        &mut self,
        _soft_targets: &DMatrix<T>,
        _X: &DMatrix<T>,
    ) -> Result<()> {
        // Simplified knowledge distillation
        // In practice, this would involve minimizing the KL divergence between
        // current predictions and soft targets
        Ok(())
    }

    /// Compute class probability
    fn compute_class_probability(&self, sample: &DVector<T>, class: i32) -> T {
        if let (Some(prior), Some((means, variances))) = (
            self.class_priors.get(&class),
            self.current_task_params.get(&class),
        ) {
            let mut log_prob = Float::ln(*prior);

            for i in 0..self.n_features {
                let x = sample[i];
                let mean = means[i];
                let var = variances[i];

                // Gaussian log-likelihood
                let diff = x - mean;
                let gaussian_ll = -NumCast::from(0.5)
                    .unwrap_or_else(|| T::one() / NumCast::from(2.0).unwrap_or_else(T::one))
                    * (diff * diff / var
                        + Float::ln(var)
                        + Float::ln(
                            NumCast::from(2.0 * std::f64::consts::PI).unwrap_or_else(|| {
                                NumCast::from(std::f64::consts::TAU).unwrap_or_else(T::one)
                            }),
                        ));
                log_prob = log_prob + gaussian_ll;
            }

            Float::exp(log_prob)
        } else {
            NumCast::from(1e-10)
                .unwrap_or_else(|| T::one() / NumCast::from(10000000000.0).unwrap_or_else(T::one))
            // Small probability for unseen classes
        }
    }

    /// Create task metadata
    fn create_task_metadata(&self, task_id: usize, X: &DMatrix<T>, y: &[i32]) -> TaskMetadata {
        let mut class_distribution = HashMap::new();
        let total_samples = y.len() as f64;

        for &class in y {
            *class_distribution.entry(class).or_insert(0.0) += 1.0 / total_samples;
        }

        // Compute feature statistics
        let mut feature_stats = DVector::zeros(self.n_features);
        for i in 0..self.n_features {
            let column_sum: T = X.column(i).iter().cloned().sum();
            feature_stats[i] = column_sum.into();
        }

        TaskMetadata {
            task_id,
            num_samples: X.nrows(),
            num_classes: class_distribution.len(),
            feature_statistics: Some(feature_stats),
            class_distribution,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        }
    }

    /// Update list of all classes
    fn update_classes(&mut self, y: &[i32]) {
        for &class in y {
            if !self.all_classes.contains(&class) {
                self.all_classes.push(class);
            }
        }
        self.all_classes.sort_unstable();
    }
}

impl<T: Float + std::iter::Sum> DriftDetector<T> {
    fn new(method: DriftDetectionMethod, window_size: usize, threshold: T) -> Self {
        Self {
            method,
            reference_window: VecDeque::new(),
            current_window: VecDeque::new(),
            window_size,
            drift_threshold: threshold,
            ph_sum: T::zero(),
            ph_sum_min: T::zero(),
            adwin_buckets: VecDeque::new(),
        }
    }

    fn add_sample(&mut self, sample: T) -> Result<bool> {
        match self.method {
            DriftDetectionMethod::PageHinkley => self.page_hinkley_test(sample),
            DriftDetectionMethod::ADWIN => self.adwin_test(sample),
            _ => self.simple_window_test(sample),
        }
    }

    fn page_hinkley_test(&mut self, sample: T) -> Result<bool> {
        self.ph_sum = self.ph_sum + sample
            - NumCast::from(0.5)
                .unwrap_or_else(|| T::one() / NumCast::from(2.0).unwrap_or_else(T::one)); // Assuming target mean is 0.5
        if self.ph_sum < self.ph_sum_min {
            self.ph_sum_min = self.ph_sum;
        }

        let drift_detected = (self.ph_sum - self.ph_sum_min) > self.drift_threshold;

        if drift_detected {
            // Reset detector
            self.ph_sum = T::zero();
            self.ph_sum_min = T::zero();
        }

        Ok(drift_detected)
    }

    fn adwin_test(&mut self, sample: T) -> Result<bool> {
        // Simplified ADWIN implementation
        let bucket = ADWINBucket {
            sum: sample,
            count: 1,
            variance: T::zero(),
        };

        self.adwin_buckets.push_back(bucket);

        // Keep only recent buckets (simplified)
        if self.adwin_buckets.len() > self.window_size {
            self.adwin_buckets.pop_front();
        }

        // Simplified drift detection based on variance change
        if self.adwin_buckets.len() >= 2 {
            let recent_var = self.compute_recent_variance();
            let overall_var = self.compute_overall_variance();
            Ok((recent_var - overall_var).abs() > self.drift_threshold)
        } else {
            Ok(false)
        }
    }

    fn simple_window_test(&mut self, sample: T) -> Result<bool> {
        self.current_window.push_back(sample);

        if self.current_window.len() > self.window_size {
            self.current_window.pop_front();
        }

        if self.current_window.len() == self.window_size && self.reference_window.is_empty() {
            self.reference_window = self.current_window.clone();
            return Ok(false);
        }

        if self.current_window.len() == self.window_size && !self.reference_window.is_empty() {
            // Simple statistical test
            let ref_mean = self.reference_window.iter().cloned().sum::<T>()
                / NumCast::from(self.window_size as f64).unwrap_or_else(T::one);
            let cur_mean = self.current_window.iter().cloned().sum::<T>()
                / NumCast::from(self.window_size as f64).unwrap_or_else(T::one);

            Ok((ref_mean - cur_mean).abs() > self.drift_threshold)
        } else {
            Ok(false)
        }
    }

    fn compute_recent_variance(&self) -> T {
        if self.adwin_buckets.len() < 2 {
            return T::zero();
        }

        let recent_buckets: Vec<_> = self
            .adwin_buckets
            .iter()
            .rev()
            .take(self.adwin_buckets.len() / 2)
            .collect();
        let sum: T = recent_buckets.iter().map(|b| b.sum).sum();
        let count: usize = recent_buckets.iter().map(|b| b.count).sum();

        if count == 0 {
            return T::zero();
        }

        let mean = sum / NumCast::from(count as f64).unwrap_or_else(T::one);
        recent_buckets
            .iter()
            .map(|b| {
                let val = b.sum / NumCast::from(b.count as f64).unwrap_or_else(T::one) - mean;
                val * val
            })
            .sum::<T>()
            / NumCast::from(recent_buckets.len() as f64).unwrap_or_else(T::one)
    }

    fn compute_overall_variance(&self) -> T {
        if self.adwin_buckets.is_empty() {
            return T::zero();
        }

        let sum: T = self.adwin_buckets.iter().map(|b| b.sum).sum();
        let count: usize = self.adwin_buckets.iter().map(|b| b.count).sum();

        if count == 0 {
            return T::zero();
        }

        let mean = sum / NumCast::from(count as f64).unwrap_or_else(T::one);
        self.adwin_buckets
            .iter()
            .map(|b| {
                let val = b.sum / NumCast::from(b.count as f64).unwrap_or_else(T::one) - mean;
                val * val
            })
            .sum::<T>()
            / NumCast::from(self.adwin_buckets.len() as f64).unwrap_or_else(T::one)
    }
}

/// Builder for Continual Learning Naive Bayes
pub struct ContinualLearningNBBuilder<T: Float> {
    config: ContinualLearningConfig,
    _phantom: PhantomData<T>,
}

impl<
        T: Float
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::DivAssign
            + Clone
            + std::fmt::Debug
            + 'static,
    > ContinualLearningNBBuilder<T>
where
    T: From<f64> + Into<f64>,
    T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum,
{
    pub fn new() -> Self {
        Self {
            config: ContinualLearningConfig::default(),
            _phantom: PhantomData,
        }
    }

    pub fn strategy(mut self, strategy: ContinualLearningStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn memory_size(mut self, size: usize) -> Self {
        self.config.memory_size = size;
        self
    }

    pub fn memory_strategy(mut self, strategy: MemoryStrategy) -> Self {
        self.config.memory_strategy = strategy;
        self
    }

    pub fn drift_detection(mut self, method: DriftDetectionMethod) -> Self {
        self.config.drift_detection = method;
        self
    }

    pub fn ewc_lambda(mut self, lambda: f64) -> Self {
        self.config.ewc_lambda = lambda;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.config.learning_rate = lr;
        self
    }

    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_iterations = max_iter;
        self
    }

    pub fn forgetting_factor(mut self, factor: f64) -> Self {
        self.config.forgetting_factor = factor;
        self
    }

    pub fn drift_threshold(mut self, threshold: f64) -> Self {
        self.config.drift_threshold = threshold;
        self
    }

    pub fn build(self) -> ContinualLearningNB<T>
    where
        T: Float + Copy + From<f64> + Into<f64> + std::fmt::Debug + 'static,
        T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum,
    {
        ContinualLearningNB::new(self.config)
    }
}

impl<
        T: Float
            + scirs2_core::ndarray::ScalarOperand
            + std::ops::DivAssign
            + Clone
            + std::fmt::Debug
            + 'static,
    > Default for ContinualLearningNBBuilder<T>
where
    T: From<f64> + Into<f64>,
    T: for<'a> std::iter::Sum<&'a T> + std::iter::Sum,
{
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_continual_learning_nb_creation() {
        let config = ContinualLearningConfig::default();
        let model = ContinualLearningNB::<f64>::new(config);
        assert!(!model.is_fitted);
        assert_eq!(model.memory_buffer_size(), 0);
        assert_eq!(model.num_tasks(), 0);
    }

    #[test]
    fn test_continual_learning_nb_builder() {
        let model = ContinualLearningNBBuilder::<f64>::new()
            .strategy(ContinualLearningStrategy::MemoryReplay)
            .memory_size(500)
            .ewc_lambda(100.0)
            .learning_rate(0.001)
            .build();

        assert_eq!(model.config.memory_size, 500);
        assert_eq!(model.config.ewc_lambda, 100.0);
        assert_eq!(model.config.learning_rate, 0.001);
    }

    #[test]
    fn test_continual_learning_fit_multiple_tasks() {
        let mut model = ContinualLearningNBBuilder::<f64>::new()
            .strategy(ContinualLearningStrategy::MemoryReplay)
            .memory_size(100)
            .build();

        // Task 1: Simple binary classification
        let X1 = Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
            .unwrap();
        let y1 = vec![0, 0, 1, 1];
        assert!(model.fit_task(&X1, &y1, 1).is_ok());
        assert_eq!(model.num_tasks(), 1);

        // Task 2: Another binary classification
        let X2 = Array2::from_shape_vec((4, 2), vec![5.0, 6.0, 6.0, 7.0, 15.0, 16.0, 16.0, 17.0])
            .unwrap();
        let y2 = vec![0, 0, 1, 1];
        assert!(model.fit_task(&X2, &y2, 2).is_ok());
        assert_eq!(model.num_tasks(), 2);

        // Test prediction
        let predictions = model.predict(&X1).unwrap();
        assert_eq!(predictions.len(), 4);

        let probabilities = model.predict_proba(&X1).unwrap();
        assert_eq!(probabilities.dim(), (4, 2));
    }

    #[test]
    fn test_memory_buffer_management() {
        let mut model = ContinualLearningNBBuilder::<f64>::new()
            .memory_size(5) // Small buffer
            .memory_strategy(MemoryStrategy::ClassBalanced)
            .build();

        let X = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0, 6.0, 7.0, 7.0, 8.0, 8.0, 9.0,
                9.0, 10.0, 10.0, 11.0,
            ],
        )
        .unwrap();
        let y = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        model.fit_task(&X, &y, 1).unwrap();

        // Memory buffer should not exceed the limit
        assert!(model.memory_buffer_size() <= 5);
    }

    #[test]
    fn test_drift_detection() {
        let mut model = ContinualLearningNBBuilder::<f64>::new()
            .drift_detection(DriftDetectionMethod::PageHinkley)
            .drift_threshold(0.1)
            .build();

        // Initial task
        let X1 =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y1 = vec![0, 0, 1, 1];
        model.fit_task(&X1, &y1, 1).unwrap();

        // Test drift detection (may or may not detect drift)
        let X2 =
            Array2::from_shape_vec((4, 2), vec![10.0, 11.0, 11.0, 12.0, 12.0, 13.0, 13.0, 14.0])
                .unwrap();
        let y2 = vec![0, 0, 1, 1];
        let _drift_detected = model.detect_drift(&X2, &y2).unwrap();
    }

    #[test]
    fn test_different_continual_strategies() {
        for strategy in [
            ContinualLearningStrategy::ElasticWeightConsolidation,
            ContinualLearningStrategy::MemoryReplay,
            ContinualLearningStrategy::ProgressiveLearning,
            ContinualLearningStrategy::KnowledgeDistillation,
            ContinualLearningStrategy::Hybrid,
        ] {
            let mut model = ContinualLearningNBBuilder::<f64>::new()
                .strategy(strategy)
                .max_iterations(5)
                .build();

            let X =
                Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 10.0, 11.0, 11.0, 12.0])
                    .unwrap();
            let y = vec![0, 0, 1, 1];

            assert!(model.fit_task(&X, &y, 1).is_ok());
            assert!(model.predict(&X).is_ok());
        }
    }

    #[test]
    fn test_task_metadata() {
        let mut model = ContinualLearningNBBuilder::<f64>::new().build();

        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 10.0, 11.0, 11.0, 12.0, 12.0, 13.0,
            ],
        )
        .unwrap();
        let y = vec![0, 0, 0, 1, 1, 1];

        model.fit_task(&X, &y, 42).unwrap();

        let metadata = model.get_task_metadata(42).unwrap();
        assert_eq!(metadata.task_id, 42);
        assert_eq!(metadata.num_samples, 6);
        assert_eq!(metadata.num_classes, 2);
        assert!(metadata.feature_statistics.is_some());
    }

    #[test]
    fn test_error_handling() {
        let mut model = ContinualLearningNBBuilder::<f64>::new().build();

        // Test prediction before fitting
        let X = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!(model.predict(&X).is_err());

        // Test dimension mismatch
        let X_fit = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = vec![0, 1];
        model.fit_task(&X_fit, &y, 1).unwrap();

        let X_wrong = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        assert!(model.predict(&X_wrong).is_err());

        // Test insufficient data
        let X_empty = Array2::from_shape_vec((0, 2), vec![]).unwrap();
        let y_empty = vec![];
        assert!(model.fit_task(&X_empty, &y_empty, 2).is_err());
    }
}

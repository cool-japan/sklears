//! Adaptive Discriminant Learning
//!
//! This module implements adaptive discriminant learning algorithms that adjust their
//! parameters dynamically based on data characteristics, performance feedback, or
//! changing data distributions. The adaptive approach allows the discriminant models
//! to maintain performance in non-stationary environments.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::VecDeque;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub enum AdaptationStrategy {
    /// Exponential moving average with decay rate
    ExponentialMovingAverage { decay_rate: Float },
    /// Sliding window with fixed size
    SlidingWindow { window_size: usize },
    /// Performance-based adaptation
    PerformanceBased {
        performance_threshold: Float,

        adaptation_rate: Float,
    },
    /// Concept drift detection based adaptation
    ConceptDriftDetection {
        drift_threshold: Float,
        detection_window: usize,
    },
    /// Bayesian adaptation with prior strength
    BayesianAdaptation {
        prior_strength: Float,
        update_rate: Float,
    },
}

#[derive(Debug, Clone)]
pub enum BaseDiscriminant {
    /// Linear Discriminant Analysis
    LDA,
    /// Quadratic Discriminant Analysis
    QDA,
    /// Regularized LDA
    RegularizedLDA { reg_param: Float },
}

#[derive(Debug, Clone)]
pub struct AdaptiveDiscriminantLearningConfig {
    /// Base discriminant model type
    pub base_discriminant: BaseDiscriminant,
    /// Adaptation strategy
    pub adaptation_strategy: AdaptationStrategy,
    /// Initial learning rate
    pub initial_learning_rate: Float,
    /// Minimum learning rate
    pub min_learning_rate: Float,
    /// Learning rate decay
    pub learning_rate_decay: Float,
    /// Performance monitoring window
    pub performance_window: usize,
    /// Adaptation frequency (number of samples between adaptations)
    pub adaptation_frequency: usize,
    /// Convergence tolerance for adaptation
    pub adaptation_tol: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for AdaptiveDiscriminantLearningConfig {
    fn default() -> Self {
        Self {
            base_discriminant: BaseDiscriminant::LDA,
            adaptation_strategy: AdaptationStrategy::ExponentialMovingAverage { decay_rate: 0.95 },
            initial_learning_rate: 0.01,
            min_learning_rate: 1e-6,
            learning_rate_decay: 0.99,
            performance_window: 100,
            adaptation_frequency: 10,
            adaptation_tol: 1e-4,
            random_state: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveDiscriminantLearning<State = Untrained> {
    config: AdaptiveDiscriminantLearningConfig,
    data: Option<TrainedData>,
    _state: PhantomData<State>,
}

#[derive(Debug, Clone)]
struct TrainedData {
    /// Current class means
    class_means: Array2<Float>,
    /// Current covariance matrix
    covariance: Array2<Float>,
    /// Current precision matrix
    precision: Array2<Float>,
    /// Class priors
    class_priors: Array1<Float>,
    /// Classes
    classes: Array1<i32>,
    /// Current learning rate
    current_learning_rate: Float,
    /// Sample count for each class
    class_counts: Array1<usize>,
    /// Total samples seen
    total_samples: usize,
    /// Performance history
    performance_history: VecDeque<Float>,
    /// Adaptation history
    adaptation_history: Vec<AdaptationEvent>,
    /// Data window for sliding window strategy
    data_window: VecDeque<(Array1<Float>, i32)>,
    /// Number of features
    n_features: usize,
}

#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// sample_count
    pub sample_count: usize,
    /// performance
    pub performance: Float,
    /// learning_rate
    pub learning_rate: Float,
    /// adaptation_type
    pub adaptation_type: String,
}

impl AdaptiveDiscriminantLearning<Untrained> {
    pub fn new() -> Self {
        Self {
            config: AdaptiveDiscriminantLearningConfig::default(),
            data: None,
            _state: PhantomData,
        }
    }

    pub fn with_config(config: AdaptiveDiscriminantLearningConfig) -> Self {
        Self {
            config,
            data: None,
            _state: PhantomData,
        }
    }

    pub fn base_discriminant(mut self, base_discriminant: BaseDiscriminant) -> Self {
        self.config.base_discriminant = base_discriminant;
        self
    }

    pub fn adaptation_strategy(mut self, adaptation_strategy: AdaptationStrategy) -> Self {
        self.config.adaptation_strategy = adaptation_strategy;
        self
    }

    pub fn initial_learning_rate(mut self, learning_rate: Float) -> Self {
        self.config.initial_learning_rate = learning_rate;
        self
    }

    pub fn min_learning_rate(mut self, min_learning_rate: Float) -> Self {
        self.config.min_learning_rate = min_learning_rate;
        self
    }

    pub fn learning_rate_decay(mut self, decay: Float) -> Self {
        self.config.learning_rate_decay = decay;
        self
    }

    pub fn performance_window(mut self, window: usize) -> Self {
        self.config.performance_window = window;
        self
    }

    pub fn adaptation_frequency(mut self, frequency: usize) -> Self {
        self.config.adaptation_frequency = frequency;
        self
    }

    pub fn adaptation_tol(mut self, tol: Float) -> Self {
        self.config.adaptation_tol = tol;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Default for AdaptiveDiscriminantLearning<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

pub type TrainedAdaptiveDiscriminantLearning = AdaptiveDiscriminantLearning<Trained>;

impl AdaptiveDiscriminantLearning<Trained> {
    pub fn class_means(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().class_means
    }

    pub fn covariance(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().covariance
    }

    pub fn precision(&self) -> &Array2<Float> {
        &self.data.as_ref().unwrap().precision
    }

    pub fn class_priors(&self) -> &Array1<Float> {
        &self.data.as_ref().unwrap().class_priors
    }

    pub fn classes(&self) -> &Array1<i32> {
        &self.data.as_ref().unwrap().classes
    }

    pub fn current_learning_rate(&self) -> Float {
        self.data.as_ref().unwrap().current_learning_rate
    }

    pub fn total_samples(&self) -> usize {
        self.data.as_ref().unwrap().total_samples
    }

    pub fn performance_history(&self) -> &VecDeque<Float> {
        &self.data.as_ref().unwrap().performance_history
    }

    pub fn adaptation_history(&self) -> &Vec<AdaptationEvent> {
        &self.data.as_ref().unwrap().adaptation_history
    }

    pub fn n_features(&self) -> usize {
        self.data.as_ref().unwrap().n_features
    }

    /// Perform online adaptation with new data
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        let (_n_samples, n_features) = x.dim();

        // Check dimensions first
        {
            let data = self.data.as_ref().unwrap();
            if n_features != data.n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected {} features, got {}",
                    data.n_features, n_features
                )));
            }
        }

        for (sample, &label) in x.axis_iter(Axis(0)).zip(y.iter()) {
            self.update_with_sample(&sample.to_owned(), label)?;
        }

        // Check if adaptation is needed
        let adaptation_frequency = self.config.adaptation_frequency;
        let should_adapt = {
            let data = self.data.as_ref().unwrap();
            data.total_samples % adaptation_frequency == 0
        };

        if should_adapt {
            self.adapt_parameters()?;
        }

        Ok(())
    }

    /// Update model with a single sample
    fn update_with_sample(&mut self, sample: &Array1<Float>, label: i32) -> Result<()> {
        // Find class index first
        let class_idx = {
            let data = self.data.as_ref().unwrap();
            data.classes
                .iter()
                .position(|&c| c == label)
                .ok_or_else(|| {
                    SklearsError::InvalidInput(format!("Unknown class label: {}", label))
                })?
        };

        // Update based on adaptation strategy
        match &self.config.adaptation_strategy {
            AdaptationStrategy::ExponentialMovingAverage { decay_rate } => {
                self.update_ema(sample, class_idx, *decay_rate)?;
            }
            AdaptationStrategy::SlidingWindow { window_size } => {
                self.update_sliding_window(sample, label, *window_size)?;
            }
            AdaptationStrategy::PerformanceBased { .. } => {
                self.update_performance_based(sample, class_idx)?;
            }
            AdaptationStrategy::ConceptDriftDetection { .. } => {
                self.update_drift_detection(sample, label)?;
            }
            AdaptationStrategy::BayesianAdaptation { .. } => {
                self.update_bayesian(sample, class_idx)?;
            }
        }

        // Update counts after strategy updates
        {
            let data = self.data.as_mut().unwrap();
            data.total_samples += 1;
            data.class_counts[class_idx] += 1;
        }

        Ok(())
    }

    /// Update using exponential moving average
    fn update_ema(
        &mut self,
        sample: &Array1<Float>,
        class_idx: usize,
        decay_rate: Float,
    ) -> Result<()> {
        let data = self.data.as_mut().unwrap();
        let alpha = 1.0 - decay_rate;

        // Update class mean
        let old_mean = data.class_means.row(class_idx).to_owned();
        let new_mean = &old_mean * decay_rate + sample * alpha;
        data.class_means.row_mut(class_idx).assign(&new_mean);

        // Update covariance matrix
        let diff_old = sample - &old_mean;
        let diff_new = sample - &new_mean;
        let outer_product = diff_old
            .insert_axis(Axis(1))
            .dot(&diff_new.insert_axis(Axis(0)));

        data.covariance = &data.covariance * decay_rate + &outer_product * alpha;

        // Recompute precision matrix
        data.precision = Self::invert_matrix(&data.covariance)?;

        Ok(())
    }

    /// Update using sliding window
    fn update_sliding_window(
        &mut self,
        sample: &Array1<Float>,
        label: i32,
        window_size: usize,
    ) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        // Add new sample to window
        data.data_window.push_back((sample.clone(), label));

        // Remove old samples if window is full
        if data.data_window.len() > window_size {
            data.data_window.pop_front();
        }

        // Recompute statistics from window
        self.recompute_from_window()?;

        Ok(())
    }

    /// Update using performance-based adaptation
    fn update_performance_based(&mut self, sample: &Array1<Float>, class_idx: usize) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        // Make prediction and compute performance
        let prediction = Self::predict_single_static(
            &data.class_means,
            &data.precision,
            &data.classes,
            &data.class_priors,
            &self.config.base_discriminant,
            sample,
        )?;
        let true_class = data.classes[class_idx];
        let performance = if prediction == true_class { 1.0 } else { 0.0 };

        // Update performance history
        data.performance_history.push_back(performance);
        if data.performance_history.len() > self.config.performance_window {
            data.performance_history.pop_front();
        }

        // Update model parameters based on performance
        let learning_rate = data.current_learning_rate;
        self.update_parameters_with_feedback(sample, class_idx, performance, learning_rate)?;

        Ok(())
    }

    /// Update using concept drift detection
    fn update_drift_detection(&mut self, sample: &Array1<Float>, label: i32) -> Result<()> {
        // For simplicity, use performance-based detection
        let class_idx = self
            .data
            .as_ref()
            .unwrap()
            .classes
            .iter()
            .position(|&c| c == label)
            .unwrap();

        self.update_performance_based(sample, class_idx)?;

        // Check for concept drift
        let data = self.data.as_ref().unwrap();
        if data.performance_history.len() >= self.config.performance_window {
            let recent_performance: Float = data
                .performance_history
                .iter()
                .rev()
                .take(20)
                .sum::<Float>()
                / 20.0;
            let older_performance: Float =
                data.performance_history.iter().take(20).sum::<Float>() / 20.0;

            if let AdaptationStrategy::ConceptDriftDetection {
                drift_threshold, ..
            } = &self.config.adaptation_strategy
            {
                if (recent_performance - older_performance).abs() > *drift_threshold {
                    // Concept drift detected - increase learning rate temporarily
                    let data_mut = self.data.as_mut().unwrap();
                    data_mut.current_learning_rate = (data_mut.current_learning_rate * 2.0)
                        .min(self.config.initial_learning_rate);

                    data_mut.adaptation_history.push(AdaptationEvent {
                        sample_count: data_mut.total_samples,
                        performance: recent_performance,
                        learning_rate: data_mut.current_learning_rate,
                        adaptation_type: "ConceptDrift".to_string(),
                    });
                }
            }
        }

        Ok(())
    }

    /// Update using Bayesian adaptation
    fn update_bayesian(&mut self, sample: &Array1<Float>, class_idx: usize) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        if let AdaptationStrategy::BayesianAdaptation {
            prior_strength,
            update_rate,
        } = &self.config.adaptation_strategy
        {
            let n = data.class_counts[class_idx] as Float;
            let weight = update_rate / (prior_strength + n);

            // Bayesian update of class mean
            let old_mean = data.class_means.row(class_idx).to_owned();
            let new_mean = &old_mean * (1.0 - weight) + sample * weight;
            data.class_means.row_mut(class_idx).assign(&new_mean);

            // Update covariance with Bayesian approach
            let diff = sample - &new_mean;
            let outer_product = diff
                .clone()
                .insert_axis(Axis(1))
                .dot(&diff.insert_axis(Axis(0)));
            data.covariance = &data.covariance * (1.0 - weight) + &outer_product * weight;

            // Recompute precision
            data.precision = Self::invert_matrix(&data.covariance)?;
        }

        Ok(())
    }

    /// Recompute statistics from sliding window
    fn recompute_from_window(&mut self) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        if data.data_window.is_empty() {
            return Ok(());
        }

        let n_classes = data.classes.len();
        let n_features = data.n_features;

        // Reset statistics
        data.class_means.fill(0.0);
        data.class_counts.fill(0);
        let mut class_samples: Vec<Vec<Array1<Float>>> = vec![Vec::new(); n_classes];

        // Collect samples by class
        for (sample, label) in &data.data_window {
            if let Some(class_idx) = data.classes.iter().position(|&c| c == *label) {
                class_samples[class_idx].push(sample.clone());
                data.class_counts[class_idx] += 1;
            }
        }

        // Compute class means
        for (class_idx, samples) in class_samples.iter().enumerate() {
            if !samples.is_empty() {
                let sum: Array1<Float> = samples
                    .iter()
                    .fold(Array1::zeros(n_features), |acc, x| acc + x);
                data.class_means
                    .row_mut(class_idx)
                    .assign(&(sum / samples.len() as Float));
            }
        }

        // Compute global covariance
        data.covariance.fill(0.0);
        let total_samples = data.data_window.len() as Float;

        for (sample, label) in &data.data_window {
            if let Some(class_idx) = data.classes.iter().position(|&c| c == *label) {
                let diff = sample - &data.class_means.row(class_idx);
                let outer_product = diff
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&diff.insert_axis(Axis(0)));
                data.covariance = &data.covariance + &outer_product;
            }
        }

        if total_samples > 1.0 {
            data.covariance = &data.covariance / (total_samples - 1.0);
        }

        // Add regularization
        for i in 0..n_features {
            data.covariance[[i, i]] += 1e-6;
        }

        // Recompute precision
        data.precision = Self::invert_matrix(&data.covariance)?;

        Ok(())
    }

    /// Update parameters with feedback
    fn update_parameters_with_feedback(
        &mut self,
        sample: &Array1<Float>,
        class_idx: usize,
        performance: Float,
        learning_rate: Float,
    ) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        // Adaptive learning rate based on performance
        let adaptive_lr = if performance > 0.5 {
            learning_rate * 0.8 // Reduce learning rate if performing well
        } else {
            learning_rate * 1.2 // Increase learning rate if performing poorly
        };

        // Update class mean
        let old_mean = data.class_means.row(class_idx).to_owned();
        let error = sample - &old_mean;
        let new_mean = &old_mean + &error * adaptive_lr;
        data.class_means.row_mut(class_idx).assign(&new_mean);

        // Update covariance
        let outer_product = error
            .clone()
            .insert_axis(Axis(1))
            .dot(&error.insert_axis(Axis(0)));
        data.covariance = &data.covariance * (1.0 - adaptive_lr) + &outer_product * adaptive_lr;

        // Recompute precision
        data.precision = Self::invert_matrix(&data.covariance)?;

        Ok(())
    }

    /// Adapt global parameters
    fn adapt_parameters(&mut self) -> Result<()> {
        let data = self.data.as_mut().unwrap();

        // Update learning rate with decay
        data.current_learning_rate = (data.current_learning_rate * self.config.learning_rate_decay)
            .max(self.config.min_learning_rate);

        // Update class priors
        let total_count: usize = data.class_counts.sum();
        if total_count > 0 {
            for (i, &count) in data.class_counts.iter().enumerate() {
                data.class_priors[i] = count as Float / total_count as Float;
            }
        }

        // Record adaptation event
        let performance = if !data.performance_history.is_empty() {
            data.performance_history.iter().sum::<Float>() / data.performance_history.len() as Float
        } else {
            0.0
        };

        data.adaptation_history.push(AdaptationEvent {
            sample_count: data.total_samples,
            performance,
            learning_rate: data.current_learning_rate,
            adaptation_type: "ParameterUpdate".to_string(),
        });

        Ok(())
    }

    /// Predict for a single sample
    fn predict_single(&self, sample: &Array1<Float>) -> Result<i32> {
        let data = self.data.as_ref().unwrap();
        let n_classes = data.classes.len();
        let mut scores = Array1::zeros(n_classes);

        for (class_idx, &_class_label) in data.classes.iter().enumerate() {
            let class_mean = data.class_means.row(class_idx);
            let diff = sample - &class_mean;

            // Compute discriminant score based on base discriminant type
            let score = match &self.config.base_discriminant {
                BaseDiscriminant::LDA => {
                    // Linear discriminant: log p(x|class) + log p(class)
                    let mahalanobis = diff.dot(&data.precision.dot(&diff));
                    -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                }
                BaseDiscriminant::QDA => {
                    // Quadratic discriminant (simplified - using shared covariance)
                    let mahalanobis = diff.dot(&data.precision.dot(&diff));
                    -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                }
                BaseDiscriminant::RegularizedLDA { reg_param } => {
                    // Regularized linear discriminant
                    let regularized_precision =
                        &data.precision + &(Array2::<Float>::eye(data.n_features) * *reg_param);
                    let mahalanobis = diff.dot(&regularized_precision.dot(&diff));
                    -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                }
            };

            scores[class_idx] = score;
        }

        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;

        Ok(data.classes[max_idx])
    }

    /// Static version of predict_single for use when we have mutable borrows
    fn predict_single_static(
        class_means: &Array2<Float>,
        precision: &Array2<Float>,
        classes: &Array1<i32>,
        class_priors: &Array1<Float>,
        base_discriminant: &BaseDiscriminant,
        sample: &Array1<Float>,
    ) -> Result<i32> {
        let n_classes = classes.len();
        let mut scores = Array1::zeros(n_classes);

        for (class_idx, &_class_label) in classes.iter().enumerate() {
            let class_mean = class_means.row(class_idx);
            let diff = sample - &class_mean;

            // Compute discriminant score based on base discriminant type
            let score = match base_discriminant {
                BaseDiscriminant::LDA => {
                    // Linear discriminant: log p(x|class) + log p(class)
                    let mahalanobis = diff.dot(&precision.dot(&diff));
                    -0.5 * mahalanobis + class_priors[class_idx].ln()
                }
                BaseDiscriminant::QDA => {
                    // Quadratic discriminant (simplified - using shared covariance)
                    let mahalanobis = diff.dot(&precision.dot(&diff));
                    -0.5 * mahalanobis + class_priors[class_idx].ln()
                }
                BaseDiscriminant::RegularizedLDA { reg_param } => {
                    // Regularized linear discriminant
                    let n_features = precision.nrows();
                    let regularized_precision =
                        precision + &(Array2::<Float>::eye(n_features) * *reg_param);
                    let mahalanobis = diff.dot(&regularized_precision.dot(&diff));
                    -0.5 * mahalanobis + class_priors[class_idx].ln()
                }
            };

            scores[class_idx] = score;
        }

        let max_idx = scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
            .0;

        Ok(classes[max_idx])
    }

    /// Invert a matrix using LU decomposition
    fn invert_matrix(matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let mut lu = matrix.clone();
        let mut inv = Array2::eye(n);

        // LU decomposition with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            for k in (i + 1)..n {
                if lu[[k, i]].abs() > lu[[max_idx, i]].abs() {
                    max_idx = k;
                }
            }

            if lu[[max_idx, i]].abs() < 1e-12 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }

            if max_idx != i {
                // Swap rows
                for j in 0..n {
                    let temp = lu[[i, j]];
                    lu[[i, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;

                    let temp = inv[[i, j]];
                    inv[[i, j]] = inv[[max_idx, j]];
                    inv[[max_idx, j]] = temp;
                }
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = lu[[k, i]] / lu[[i, i]];
                for j in (i + 1)..n {
                    lu[[k, j]] -= factor * lu[[i, j]];
                }
                for j in 0..n {
                    inv[[k, j]] -= factor * inv[[i, j]];
                }
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in 0..n {
                for k in (i + 1)..n {
                    inv[[i, j]] -= lu[[i, k]] * inv[[k, j]];
                }
                inv[[i, j]] /= lu[[i, i]];
            }
        }

        Ok(inv)
    }
}

impl Estimator<Untrained> for AdaptiveDiscriminantLearning<Untrained> {
    type Config = AdaptiveDiscriminantLearningConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for AdaptiveDiscriminantLearning<Untrained> {
    type Fitted = AdaptiveDiscriminantLearning<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Input data and targets have different lengths".to_string(),
            ));
        }

        // Extract unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }

        // Initialize class means
        let mut class_means = Array2::zeros((n_classes, n_features));
        let mut class_counts = Array1::zeros(n_classes);

        for (class_idx, &class_label) in classes.iter().enumerate() {
            let class_mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter(|(_, &label)| label == class_label)
                .map(|(idx, _)| idx)
                .collect();

            if !class_mask.is_empty() {
                let class_data = x.select(Axis(0), &class_mask);
                let mean = class_data.mean_axis(Axis(0)).unwrap();
                class_means.row_mut(class_idx).assign(&mean);
                class_counts[class_idx] = class_mask.len();
            }
        }

        // Compute initial covariance matrix
        let mut covariance = Array2::zeros((n_features, n_features));
        for (sample, &label) in x.axis_iter(Axis(0)).zip(y.iter()) {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();
            let diff = &sample - &class_means.row(class_idx);
            let outer_product = diff
                .clone()
                .insert_axis(Axis(1))
                .dot(&diff.insert_axis(Axis(0)));
            covariance = covariance + outer_product;
        }
        covariance /= (n_samples - 1) as Float;

        // Add regularization for numerical stability
        for i in 0..n_features {
            covariance[[i, i]] += 1e-6;
        }

        // Compute precision matrix
        let precision = self.invert_matrix(&covariance)?;

        // Compute class priors
        let mut class_priors = Array1::zeros(n_classes);
        for (i, &count) in class_counts.iter().enumerate() {
            class_priors[i] = count as Float / n_samples as Float;
        }

        // Initialize data window for sliding window strategy
        let data_window =
            if let AdaptationStrategy::SlidingWindow { .. } = self.config.adaptation_strategy {
                let mut window = VecDeque::new();
                for (sample, &label) in x.axis_iter(Axis(0)).zip(y.iter()) {
                    window.push_back((sample.to_owned(), label));
                }
                window
            } else {
                VecDeque::new()
            };

        let trained_data = TrainedData {
            class_means,
            covariance,
            precision,
            class_priors,
            classes,
            current_learning_rate: self.config.initial_learning_rate,
            class_counts,
            total_samples: n_samples,
            performance_history: VecDeque::new(),
            adaptation_history: Vec::new(),
            data_window,
            n_features,
        };

        Ok(AdaptiveDiscriminantLearning {
            config: self.config,
            data: Some(trained_data),
            _state: PhantomData,
        })
    }
}

impl AdaptiveDiscriminantLearning<Untrained> {
    /// Invert a matrix using LU decomposition (helper for fit method)
    fn invert_matrix(&self, matrix: &Array2<Float>) -> Result<Array2<Float>> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        let mut lu = matrix.clone();
        let mut inv = Array2::eye(n);

        // LU decomposition with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_idx = i;
            for k in (i + 1)..n {
                if lu[[k, i]].abs() > lu[[max_idx, i]].abs() {
                    max_idx = k;
                }
            }

            if lu[[max_idx, i]].abs() < 1e-12 {
                return Err(SklearsError::InvalidInput("Matrix is singular".to_string()));
            }

            if max_idx != i {
                // Swap rows
                for j in 0..n {
                    let temp = lu[[i, j]];
                    lu[[i, j]] = lu[[max_idx, j]];
                    lu[[max_idx, j]] = temp;

                    let temp = inv[[i, j]];
                    inv[[i, j]] = inv[[max_idx, j]];
                    inv[[max_idx, j]] = temp;
                }
            }

            // Eliminate column
            for k in (i + 1)..n {
                let factor = lu[[k, i]] / lu[[i, i]];
                for j in (i + 1)..n {
                    lu[[k, j]] -= factor * lu[[i, j]];
                }
                for j in 0..n {
                    inv[[k, j]] -= factor * inv[[i, j]];
                }
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in 0..n {
                for k in (i + 1)..n {
                    inv[[i, j]] -= lu[[i, k]] * inv[[k, j]];
                }
                inv[[i, j]] /= lu[[i, i]];
            }
        }

        Ok(inv)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for AdaptiveDiscriminantLearning<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_features
            )));
        }

        let mut predictions = Array1::zeros(n_samples);

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            predictions[i] = self.predict_single(&sample.to_owned())?;
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for AdaptiveDiscriminantLearning<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.n_features() {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features(),
                n_features
            )));
        }

        let data = self.data.as_ref().unwrap();
        let n_classes = data.classes.len();
        let mut probas = Array2::zeros((n_samples, n_classes));

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            let mut log_probas = Array1::zeros(n_classes);

            for (class_idx, _) in data.classes.iter().enumerate() {
                let class_mean = data.class_means.row(class_idx);
                let diff = &sample - &class_mean;

                // Compute log probability
                let log_prob = match &self.config.base_discriminant {
                    BaseDiscriminant::LDA => {
                        let mahalanobis = diff.dot(&data.precision.dot(&diff));
                        -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                    }
                    BaseDiscriminant::QDA => {
                        let mahalanobis = diff.dot(&data.precision.dot(&diff));
                        -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                    }
                    BaseDiscriminant::RegularizedLDA { reg_param } => {
                        let regularized_precision =
                            &data.precision + &(Array2::<Float>::eye(data.n_features) * *reg_param);
                        let mahalanobis = diff.dot(&regularized_precision.dot(&diff));
                        -0.5 * mahalanobis + data.class_priors[class_idx].ln()
                    }
                };

                log_probas[class_idx] = log_prob;
            }

            // Convert to probabilities using softmax
            let max_log_proba = log_probas.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_log_probas: Array1<Float> = log_probas.mapv(|x| (x - max_log_proba).exp());
            let sum_exp: Float = exp_log_probas.sum();

            if sum_exp > 1e-10 {
                probas.row_mut(i).assign(&(exp_log_probas / sum_exp));
            } else {
                probas.row_mut(i).fill(1.0 / n_classes as Float);
            }
        }

        Ok(probas)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for AdaptiveDiscriminantLearning<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Transform data using the current precision matrix
        let data = self.data.as_ref().unwrap();
        let mut transformed = Array2::zeros(x.dim());

        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            // Project onto discriminant space
            let centered_sample = &sample - &data.class_means.mean_axis(Axis(0)).unwrap();
            let transformed_sample = data.precision.dot(&centered_sample);
            transformed.row_mut(i).assign(&transformed_sample);
        }

        Ok(transformed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_adaptive_discriminant_basic() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new();
        let fitted = ada.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.classes().len(), 2);
        assert_eq!(fitted.n_features(), 2);
    }

    #[test]
    fn test_adaptive_predict_proba() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new();
        let fitted = ada.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (6, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_adaptive_partial_fit() {
        let x_initial = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y_initial = array![0, 0, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new();
        let mut fitted = ada.fit(&x_initial, &y_initial).unwrap();

        // Add more data
        let x_new = array![[1.2, 1.2], [3.2, 3.2]];
        let y_new = array![0, 1];

        fitted.partial_fit(&x_new, &y_new).unwrap();

        assert_eq!(fitted.total_samples(), 6);

        let predictions = fitted.predict(&x_new).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_different_adaptation_strategies() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let strategies = vec![
            AdaptationStrategy::ExponentialMovingAverage { decay_rate: 0.9 },
            AdaptationStrategy::SlidingWindow { window_size: 4 },
            AdaptationStrategy::PerformanceBased {
                performance_threshold: 0.8,
                adaptation_rate: 0.1,
            },
            AdaptationStrategy::BayesianAdaptation {
                prior_strength: 1.0,
                update_rate: 0.1,
            },
        ];

        for strategy in strategies {
            let ada = AdaptiveDiscriminantLearning::new().adaptation_strategy(strategy);
            let fitted = ada.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 6);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_different_base_discriminants() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let base_discriminants = vec![
            BaseDiscriminant::LDA,
            BaseDiscriminant::QDA,
            BaseDiscriminant::RegularizedLDA { reg_param: 0.1 },
        ];

        for base_discriminant in base_discriminants {
            let ada = AdaptiveDiscriminantLearning::new().base_discriminant(base_discriminant);
            let fitted = ada.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.len(), 6);
            assert_eq!(fitted.classes().len(), 2);
        }
    }

    #[test]
    fn test_adaptive_transform() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2],
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new();
        let fitted = ada.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (6, 2));
    }

    #[test]
    fn test_learning_rate_adaptation() {
        let x = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y = array![0, 0, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new()
            .initial_learning_rate(0.1)
            .learning_rate_decay(0.9)
            .adaptation_frequency(1);

        let mut fitted = ada.fit(&x, &y).unwrap();
        let initial_lr = fitted.current_learning_rate();

        // Perform some partial fits to trigger adaptation
        let x_new = array![[1.2, 1.2]];
        let y_new = array![0];

        for _ in 0..5 {
            fitted.partial_fit(&x_new, &y_new).unwrap();
        }

        let final_lr = fitted.current_learning_rate();
        assert!(final_lr < initial_lr); // Learning rate should decay
    }

    #[test]
    fn test_adaptation_history() {
        let x = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y = array![0, 0, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new().adaptation_frequency(2);
        let mut fitted = ada.fit(&x, &y).unwrap();

        // Perform partial fits to trigger adaptations
        let x_new = array![[1.2, 1.2], [3.2, 3.2]];
        let y_new = array![0, 1];

        fitted.partial_fit(&x_new, &y_new).unwrap();

        let history = fitted.adaptation_history();
        assert!(!history.is_empty());

        for event in history {
            assert!(event.learning_rate > 0.0);
            assert!(event.sample_count > 0);
        }
    }

    #[test]
    fn test_accessor_methods() {
        let x = array![[1.0, 1.0], [1.1, 1.1], [3.0, 3.0], [3.1, 3.1]];
        let y = array![0, 0, 1, 1];

        let ada = AdaptiveDiscriminantLearning::new();
        let fitted = ada.fit(&x, &y).unwrap();

        // Test accessor methods
        assert_eq!(fitted.class_means().dim(), (2, 2));
        assert_eq!(fitted.covariance().dim(), (2, 2));
        assert_eq!(fitted.precision().dim(), (2, 2));
        assert_eq!(fitted.class_priors().len(), 2);
        assert_eq!(fitted.classes().len(), 2);
        assert!(fitted.current_learning_rate() > 0.0);
        assert_eq!(fitted.total_samples(), 4);
        assert_eq!(fitted.n_features(), 2);
    }
}

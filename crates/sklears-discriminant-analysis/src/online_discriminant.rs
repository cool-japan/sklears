//! Online Discriminant Analysis
//!
//! This module implements online/streaming versions of discriminant analysis
//! that can incrementally update their parameters as new data arrives.

// ✅ Using SciRS2 dependencies following SciRS2 policy
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Estimator, Fit, Predict, PredictProba, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Update strategy for online learning
#[derive(Debug, Clone, PartialEq)]
pub enum UpdateStrategy {
    /// Exponential moving average with decay rate
    ExponentialMovingAverage { decay_rate: Float },
    /// Sliding window with fixed size
    SlidingWindow { window_size: usize },
    /// Adaptive window that adjusts based on concept drift
    AdaptiveWindow { drift_threshold: Float },
    /// Simple cumulative update
    Cumulative,
}

impl Default for UpdateStrategy {
    fn default() -> Self {
        UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.95 }
    }
}

/// Configuration for Online Discriminant Analysis
#[derive(Debug, Clone)]
pub struct OnlineDiscriminantAnalysisConfig {
    /// update_strategy
    pub update_strategy: UpdateStrategy,
    /// n_components
    pub n_components: Option<usize>,
    /// reg_param
    pub reg_param: Float,
    /// drift_detection
    pub drift_detection: bool,
    /// drift_threshold
    pub drift_threshold: Float,
    /// warm_up_samples
    pub warm_up_samples: usize,
    /// batch_size
    pub batch_size: usize,
    /// forgetting_factor
    pub forgetting_factor: Float,
}

impl Default for OnlineDiscriminantAnalysisConfig {
    fn default() -> Self {
        Self {
            update_strategy: UpdateStrategy::default(),
            n_components: None,
            reg_param: 1e-4,
            drift_detection: true,
            drift_threshold: 0.05,
            warm_up_samples: 100,
            batch_size: 32,
            forgetting_factor: 0.99,
        }
    }
}

/// Online Discriminant Analysis
///
/// Implements online/streaming discriminant analysis that can adapt to new data
/// in real-time without retraining from scratch. Supports various update strategies
/// and concept drift detection.
///
/// # Mathematical Background
///
/// Online LDA maintains running estimates of:
/// - Class priors π_k
/// - Class means μ_k
/// - Pooled covariance matrix Σ
/// - Discriminant directions W
///
/// Updates use exponential moving averages or sliding windows to adapt to
/// changing data distributions while maintaining computational efficiency.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::array;
/// use sklears_discriminant_analysis::*;
/// use sklears_core::traits::{Predict, Fit};
///
/// let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
/// let y_initial = array![0, 1];
///
/// let mut oda = OnlineDiscriminantAnalysis::new()
///     .update_strategy(UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.9 });
/// let mut fitted = oda.fit(&x_initial, &y_initial).unwrap();
///
/// // Update with new data
/// let x_new = array![[1.5, 2.5], [2.5, 3.5]];
/// let y_new = array![0, 1];
/// fitted.partial_fit(&x_new, &y_new).unwrap();
///
/// let predictions = fitted.predict(&x_new).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OnlineDiscriminantAnalysis {
    config: OnlineDiscriminantAnalysisConfig,
}

impl Default for OnlineDiscriminantAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

impl OnlineDiscriminantAnalysis {
    /// Create a new online discriminant analysis instance
    pub fn new() -> Self {
        Self {
            config: OnlineDiscriminantAnalysisConfig::default(),
        }
    }

    /// Set the update strategy
    pub fn update_strategy(mut self, update_strategy: UpdateStrategy) -> Self {
        self.config.update_strategy = update_strategy;
        self
    }

    /// Set the number of components for dimensionality reduction
    pub fn n_components(mut self, n_components: Option<usize>) -> Self {
        self.config.n_components = n_components;
        self
    }

    /// Set the regularization parameter
    pub fn reg_param(mut self, reg_param: Float) -> Self {
        self.config.reg_param = reg_param;
        self
    }

    /// Enable/disable concept drift detection
    pub fn drift_detection(mut self, drift_detection: bool) -> Self {
        self.config.drift_detection = drift_detection;
        self
    }

    /// Set the concept drift threshold
    pub fn drift_threshold(mut self, drift_threshold: Float) -> Self {
        self.config.drift_threshold = drift_threshold;
        self
    }

    /// Set the number of warm-up samples before online updates
    pub fn warm_up_samples(mut self, warm_up_samples: usize) -> Self {
        self.config.warm_up_samples = warm_up_samples;
        self
    }

    /// Set the batch size for mini-batch updates
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.config.batch_size = batch_size;
        self
    }

    /// Set the forgetting factor for old data
    pub fn forgetting_factor(mut self, forgetting_factor: Float) -> Self {
        self.config.forgetting_factor = forgetting_factor;
        self
    }
}

/// Trained Online Discriminant Analysis model
#[derive(Debug, Clone)]
pub struct TrainedOnlineDiscriminantAnalysis {
    config: OnlineDiscriminantAnalysisConfig,
    classes: Array1<i32>,
    class_priors: HashMap<i32, Float>,
    class_means: HashMap<i32, Array1<Float>>,
    class_covariances: HashMap<i32, Array2<Float>>,
    pooled_covariance: Array2<Float>,
    components: Array2<Float>,
    eigenvalues: Array1<Float>,
    n_samples_seen: usize,
    class_counts: HashMap<i32, Float>,
    data_buffer: Option<Array2<Float>>,
    label_buffer: Option<Array1<i32>>,
    n_components: usize,
    drift_scores: Vec<Float>,
    last_update_time: usize,
}

impl TrainedOnlineDiscriminantAnalysis {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.classes
    }

    /// Get the number of components
    pub fn n_components(&self) -> usize {
        self.n_components
    }

    /// Get the class priors
    pub fn class_priors(&self) -> &HashMap<i32, Float> {
        &self.class_priors
    }

    /// Get the class means
    pub fn class_means(&self) -> &HashMap<i32, Array1<Float>> {
        &self.class_means
    }

    /// Get the pooled covariance matrix
    pub fn pooled_covariance(&self) -> &Array2<Float> {
        &self.pooled_covariance
    }

    /// Get the discriminant components
    pub fn components(&self) -> &Array2<Float> {
        &self.components
    }

    /// Get the eigenvalues
    pub fn eigenvalues(&self) -> &Array1<Float> {
        &self.eigenvalues
    }

    /// Get the number of samples seen so far
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Get the drift scores over time
    pub fn drift_scores(&self) -> &Vec<Float> {
        &self.drift_scores
    }

    /// Incrementally update the model with new data
    pub fn partial_fit(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatch between X and y dimensions".to_string(),
            ));
        }

        let n_new_samples = x.nrows();
        let n_features = x.ncols();

        // Check for new classes
        for &label in y.iter() {
            if !self.class_counts.contains_key(&label) {
                self.add_new_class(label, n_features)?;
            }
        }

        // Handle different update strategies
        match &self.config.update_strategy {
            UpdateStrategy::ExponentialMovingAverage { decay_rate } => {
                self.update_exponential_moving_average(x, y, *decay_rate)?;
            }
            UpdateStrategy::SlidingWindow { window_size } => {
                self.update_sliding_window(x, y, *window_size)?;
            }
            UpdateStrategy::AdaptiveWindow { drift_threshold } => {
                self.update_adaptive_window(x, y, *drift_threshold)?;
            }
            UpdateStrategy::Cumulative => {
                self.update_cumulative(x, y)?;
            }
        }

        // Detect concept drift if enabled
        if self.config.drift_detection {
            let drift_score = self.detect_concept_drift(x, y)?;
            self.drift_scores.push(drift_score);

            if drift_score > self.config.drift_threshold {
                self.handle_concept_drift()?;
            }
        }

        // Update discriminant components
        self.update_discriminant_components()?;

        self.n_samples_seen += n_new_samples;
        self.last_update_time = self.n_samples_seen;

        Ok(())
    }

    /// Add a new class to the model
    fn add_new_class(&mut self, label: i32, n_features: usize) -> Result<()> {
        self.class_counts.insert(label, 0.0);
        self.class_priors.insert(label, 0.0);
        self.class_means.insert(label, Array1::zeros(n_features));
        self.class_covariances
            .insert(label, Array2::eye(n_features));

        // Update classes array
        let mut classes = self.classes.to_vec();
        if !classes.contains(&label) {
            classes.push(label);
            classes.sort_unstable();
            self.classes = Array1::from_vec(classes);
        }

        Ok(())
    }

    /// Update using exponential moving average
    fn update_exponential_moving_average(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        decay_rate: Float,
    ) -> Result<()> {
        let n_new_samples = x.nrows();
        let learning_rate = 1.0 - decay_rate;

        // Update class statistics
        for (i, &label) in y.iter().enumerate() {
            let sample = x.row(i);

            // Update class count
            let current_count = self.class_counts.get(&label).unwrap_or(&0.0);
            self.class_counts.insert(label, current_count + 1.0);

            // Update class mean
            if let Some(class_mean) = self.class_means.get_mut(&label) {
                for j in 0..sample.len() {
                    class_mean[j] = decay_rate * class_mean[j] + learning_rate * sample[j];
                }
            }
        }

        // Update class priors
        let total_count: Float = self.class_counts.values().sum();
        for (&label, &count) in &self.class_counts {
            let prior = count / total_count;
            self.class_priors.insert(label, prior);
        }

        // Update pooled covariance
        self.update_pooled_covariance_online(x, y, learning_rate)?;

        Ok(())
    }

    /// Update using sliding window
    fn update_sliding_window(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        window_size: usize,
    ) -> Result<()> {
        // Add new data to buffer
        if let (Some(ref mut data_buffer), Some(ref mut label_buffer)) =
            (&mut self.data_buffer, &mut self.label_buffer)
        {
            // Append new data
            let new_data = data_buffer.clone();
            let new_labels = label_buffer.clone();

            for (i, &label) in y.iter().enumerate() {
                // Add new sample (simplified - in practice you'd use proper concatenation)
                // For now, we'll just update statistics
                if let Some(class_mean) = self.class_means.get_mut(&label) {
                    let sample = x.row(i);
                    for j in 0..sample.len() {
                        class_mean[j] = (class_mean[j] + sample[j]) / 2.0; // Simplified update
                    }
                }
            }

            // Trim buffer if exceeds window size
            if new_data.nrows() + x.nrows() > window_size {
                let excess = new_data.nrows() + x.nrows() - window_size;
                // Remove oldest samples (simplified)
                for _ in 0..excess {
                    // In practice, you'd implement proper buffer management
                }
            }
        } else {
            // Initialize buffers
            self.data_buffer = Some(x.clone());
            self.label_buffer = Some(y.clone());
        }

        Ok(())
    }

    /// Update using adaptive window
    fn update_adaptive_window(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        drift_threshold: Float,
    ) -> Result<()> {
        // Detect if concept drift occurred
        let drift_score = self.detect_concept_drift(x, y)?;

        if drift_score > drift_threshold {
            // Reset model if significant drift detected
            self.reset_statistics()?;
        }

        // Update with new data
        self.update_cumulative(x, y)?;

        Ok(())
    }

    /// Update using cumulative approach
    fn update_cumulative(&mut self, x: &Array2<Float>, y: &Array1<i32>) -> Result<()> {
        let n_new_samples = x.nrows();

        // Update class statistics
        for (i, &label) in y.iter().enumerate() {
            let sample = x.row(i);

            // Update class count
            let current_count = *self.class_counts.get(&label).unwrap_or(&0.0);
            let new_count = current_count + 1.0;
            self.class_counts.insert(label, new_count);

            // Update class mean incrementally
            if let Some(class_mean) = self.class_means.get_mut(&label) {
                for j in 0..sample.len() {
                    class_mean[j] = (class_mean[j] * current_count + sample[j]) / new_count;
                }
            }
        }

        // Update class priors
        let total_count: Float = self.class_counts.values().sum();
        for (&label, &count) in &self.class_counts {
            let prior = count / total_count;
            self.class_priors.insert(label, prior);
        }

        Ok(())
    }

    /// Update pooled covariance matrix online
    fn update_pooled_covariance_online(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        learning_rate: Float,
    ) -> Result<()> {
        let n_features = x.ncols();
        let mut new_pooled_cov = Array2::zeros((n_features, n_features));
        let mut total_weight = 0.0;

        // Compute weighted covariance update
        for (i, &label) in y.iter().enumerate() {
            let sample = x.row(i);

            if let Some(class_mean) = self.class_means.get(&label) {
                let diff = &sample - class_mean;
                let class_weight = self.class_counts.get(&label).unwrap_or(&1.0);

                // Outer product update
                for j in 0..n_features {
                    for k in 0..n_features {
                        new_pooled_cov[[j, k]] += class_weight * diff[j] * diff[k];
                    }
                }
                total_weight += class_weight;
            }
        }

        if total_weight > 0.0 {
            new_pooled_cov /= total_weight;
        }

        // Exponential moving average update
        for i in 0..n_features {
            for j in 0..n_features {
                self.pooled_covariance[[i, j]] = (1.0 - learning_rate)
                    * self.pooled_covariance[[i, j]]
                    + learning_rate * new_pooled_cov[[i, j]];
            }
        }

        // Add regularization
        for i in 0..n_features {
            self.pooled_covariance[[i, i]] += self.config.reg_param;
        }

        Ok(())
    }

    /// Detect concept drift using statistical tests
    fn detect_concept_drift(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Float> {
        if self.n_samples_seen < self.config.warm_up_samples {
            return Ok(0.0); // Not enough data to detect drift
        }

        let mut drift_score = 0.0;
        let n_samples = x.nrows();

        // Compute prediction accuracy on new batch
        let predictions = self.predict(x)?;
        let mut correct_predictions = 0;

        for (i, &true_label) in y.iter().enumerate() {
            if predictions[i] == true_label {
                correct_predictions += 1;
            }
        }

        let current_accuracy = correct_predictions as Float / n_samples as Float;

        // Compare with expected accuracy (simplified)
        let expected_accuracy = 0.8; // This should be based on historical performance
        drift_score = (expected_accuracy - current_accuracy).abs();

        Ok(drift_score)
    }

    /// Handle concept drift by partially resetting model
    fn handle_concept_drift(&mut self) -> Result<()> {
        // Reduce influence of old data by scaling down statistics
        let forgetting_factor = self.config.forgetting_factor;

        // Scale down class counts
        for (_, count) in self.class_counts.iter_mut() {
            *count *= forgetting_factor;
        }

        // Update priors
        let total_count: Float = self.class_counts.values().sum();
        for (&label, &count) in &self.class_counts {
            let prior = count / total_count;
            self.class_priors.insert(label, prior);
        }

        // Add small deterministic noise to break ties and encourage exploration
        for (class_idx, (_, mean)) in self.class_means.iter_mut().enumerate() {
            for (value_idx, value) in mean.iter_mut().enumerate() {
                let noise =
                    0.001 * ((class_idx as Float * 7.0 + value_idx as Float * 3.0).sin() * 0.5);
                *value += noise;
            }
        }

        Ok(())
    }

    /// Reset all statistics (for adaptive window)
    fn reset_statistics(&mut self) -> Result<()> {
        let n_features = self.pooled_covariance.nrows();

        // Reset all statistics
        self.class_counts.clear();
        self.class_priors.clear();

        for mean in self.class_means.values_mut() {
            mean.fill(0.0);
        }

        self.pooled_covariance = Array2::eye(n_features);
        self.n_samples_seen = 0;

        Ok(())
    }

    /// Update discriminant components
    fn update_discriminant_components(&mut self) -> Result<()> {
        // For online learning, we use simplified component update
        // In practice, you'd use incremental eigenvalue decomposition

        let n_features = self.pooled_covariance.nrows();
        let n_components = self.n_components.min(n_features);

        // Simplified: use identity transformation for now
        // In a full implementation, you'd compute incremental LDA components
        self.components = Array2::eye(n_features)
            .slice(s![.., ..n_components])
            .to_owned();
        self.eigenvalues = Array1::ones(n_components);

        Ok(())
    }
}

impl Estimator for OnlineDiscriminantAnalysis {
    type Config = OnlineDiscriminantAnalysisConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>, TrainedOnlineDiscriminantAnalysis>
    for OnlineDiscriminantAnalysis
{
    type Fitted = TrainedOnlineDiscriminantAnalysis;
    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<TrainedOnlineDiscriminantAnalysis> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "Mismatch between X and y dimensions".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        // Determine number of components
        let max_components = (n_classes - 1).min(n_features);
        let n_components = self.config.n_components.unwrap_or(max_components);

        if n_components > max_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) cannot be larger than min(n_classes-1, n_features) ({})",
                n_components, max_components
            )));
        }

        // Initialize model
        let mut trained = TrainedOnlineDiscriminantAnalysis {
            config: self.config.clone(),
            classes: classes.clone(),
            class_priors: HashMap::new(),
            class_means: HashMap::new(),
            class_covariances: HashMap::new(),
            pooled_covariance: Array2::eye(n_features),
            components: Array2::eye(n_features)
                .slice(s![.., ..n_components])
                .to_owned(),
            eigenvalues: Array1::ones(n_components),
            n_samples_seen: 0,
            class_counts: HashMap::new(),
            data_buffer: None,
            label_buffer: None,
            n_components,
            drift_scores: Vec::new(),
            last_update_time: 0,
        };

        // Initialize class statistics
        for &class in &classes {
            trained.class_counts.insert(class, 0.0);
            trained.class_priors.insert(class, 0.0);
            trained.class_means.insert(class, Array1::zeros(n_features));
            trained
                .class_covariances
                .insert(class, Array2::eye(n_features));
        }

        // Initial fit using all data
        trained.partial_fit(x, y)?;

        Ok(trained)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for TrainedOnlineDiscriminantAnalysis {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let mut predictions = Array1::zeros(x.nrows());

        for (i, row) in probabilities.axis_iter(Axis(0)).enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap()
                .0;
            predictions[i] = self.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl PredictProba<Array2<Float>, Array2<Float>> for TrainedOnlineDiscriminantAnalysis {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let n_samples = x.nrows();
        let n_classes = self.classes.len();
        let mut probabilities = Array2::zeros((n_samples, n_classes));

        // Project samples to discriminant subspace
        let projected = x.dot(&self.components);

        // Compute class-conditional probabilities
        for (sample_idx, projected_sample) in projected.axis_iter(Axis(0)).enumerate() {
            let mut class_scores = Array1::zeros(n_classes);

            for (class_idx, &class) in self.classes.iter().enumerate() {
                if let (Some(class_mean), Some(class_prior)) =
                    (self.class_means.get(&class), self.class_priors.get(&class))
                {
                    // Project class mean
                    let projected_mean = class_mean.dot(&self.components);

                    // Compute log-likelihood (simplified)
                    let diff = &projected_sample - &projected_mean;
                    let distance = diff.iter().map(|&x| x * x).sum::<Float>();

                    // Log-likelihood: log(prior) - 0.5 * distance
                    class_scores[class_idx] = class_prior.ln() - 0.5 * distance;
                }
            }

            // Convert to probabilities using softmax
            let max_score = class_scores
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_scores = class_scores.mapv(|x| (x - max_score).exp());
            let sum_exp = exp_scores.sum();

            if sum_exp > 0.0 {
                exp_scores /= sum_exp;
            } else {
                exp_scores.fill(1.0 / n_classes as Float);
            }

            probabilities.row_mut(sample_idx).assign(&exp_scores);
        }

        Ok(probabilities)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for TrainedOnlineDiscriminantAnalysis {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        let transformed = x.dot(&self.components);
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
    fn test_online_discriminant_analysis_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new();
        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_online_partial_fit() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let y_initial = array![0, 1];

        let oda = OnlineDiscriminantAnalysis::new();
        let mut fitted = oda.fit(&x_initial, &y_initial).unwrap();

        // Add more data
        let x_new = array![[1.5, 2.5], [2.5, 3.5]];
        let y_new = array![0, 1];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        assert_eq!(fitted.n_samples_seen(), 4);

        let predictions = fitted.predict(&x_new).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_exponential_moving_average() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new()
            .update_strategy(UpdateStrategy::ExponentialMovingAverage { decay_rate: 0.9 });
        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_sliding_window() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new()
            .update_strategy(UpdateStrategy::SlidingWindow { window_size: 10 });
        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_adaptive_window() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda =
            OnlineDiscriminantAnalysis::new().update_strategy(UpdateStrategy::AdaptiveWindow {
                drift_threshold: 0.1,
            });
        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_online_predict_proba() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new();
        let fitted = oda.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.dim(), (4, 2));

        // Check that probabilities sum to 1
        for row in probas.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_online_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0]
        ];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new().n_components(Some(1));
        let fitted = oda.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.dim(), (4, 1));
    }

    #[test]
    fn test_concept_drift_detection() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new()
            .drift_detection(true)
            .warm_up_samples(2);
        let mut fitted = oda.fit(&x, &y).unwrap();

        // Add data that might cause drift
        let x_drift = array![[10.0, 20.0], [20.0, 30.0]];
        let y_drift = array![0, 1];
        fitted.partial_fit(&x_drift, &y_drift).unwrap();

        assert!(fitted.drift_scores().len() > 0);
    }

    #[test]
    fn test_new_class_addition() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let y_initial = array![0, 1];

        let oda = OnlineDiscriminantAnalysis::new();
        let mut fitted = oda.fit(&x_initial, &y_initial).unwrap();

        // Add data with new class
        let x_new = array![[5.0, 6.0], [6.0, 7.0]];
        let y_new = array![2, 2]; // New class
        fitted.partial_fit(&x_new, &y_new).unwrap();

        assert_eq!(fitted.classes().len(), 3);

        let predictions = fitted.predict(&x_new).unwrap();
        assert_eq!(predictions.len(), 2);
    }

    #[test]
    fn test_cumulative_update() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 0, 1, 1];

        let oda = OnlineDiscriminantAnalysis::new().update_strategy(UpdateStrategy::Cumulative);
        let fitted = oda.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(fitted.classes().len(), 2);
    }

    #[test]
    fn test_class_priors_update() {
        let x_initial = array![[1.0, 2.0], [2.0, 3.0]];
        let y_initial = array![0, 1];

        let oda = OnlineDiscriminantAnalysis::new();
        let mut fitted = oda.fit(&x_initial, &y_initial).unwrap();

        // Check initial priors
        let initial_priors = fitted.class_priors();
        assert!((initial_priors[&0] - 0.5).abs() < 1e-6);
        assert!((initial_priors[&1] - 0.5).abs() < 1e-6);

        // Add more samples of class 0
        let x_new = array![[1.1, 2.1], [1.2, 2.2]];
        let y_new = array![0, 0];
        fitted.partial_fit(&x_new, &y_new).unwrap();

        // Class 0 should now have higher prior
        let updated_priors = fitted.class_priors();
        assert!(updated_priors[&0] > updated_priors[&1]);
    }
}

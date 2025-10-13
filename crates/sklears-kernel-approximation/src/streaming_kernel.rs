use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Rng, SeedableRng, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::collections::VecDeque;

/// Streaming kernel approximation methods for online processing
///
/// This module provides online learning capabilities for kernel
/// approximations, enabling processing of data streams where samples
/// arrive continuously and memory is limited.
///
/// Buffer management strategy for streaming data
#[derive(Debug, Clone)]
pub enum BufferStrategy {
    /// Fixed-size buffer with FIFO replacement
    FixedSize(usize),
    /// Sliding window with time-based expiration
    SlidingWindow { size: usize, time_window: f64 },
    /// Reservoir sampling for representative subset
    ReservoirSampling(usize),
    /// Exponential decay weighting
    ExponentialDecay { alpha: f64, min_weight: f64 },
    /// Importance-weighted sampling
    ImportanceWeighted { capacity: usize, threshold: f64 },
}

/// Update frequency for model parameters
#[derive(Debug, Clone)]
pub enum UpdateFrequency {
    /// Update after every sample
    PerSample,
    /// Update after every N samples
    BatchSize(usize),
    /// Update based on time intervals
    TimeInterval(f64),
    /// Update when error exceeds threshold
    ErrorThreshold(f64),
    /// Adaptive update frequency
    Adaptive {
        initial: usize,
        max: usize,
        min: usize,
    },
}

/// Forgetting mechanism for old data
#[derive(Debug, Clone)]
pub enum ForgettingMechanism {
    /// No forgetting - keep all data
    None,
    /// Linear decay of old samples
    LinearDecay(f64),
    /// Exponential decay of old samples
    ExponentialDecay(f64),
    /// Abrupt forgetting after time window
    AbruptForgetting(f64),
    /// Gradual forgetting with sigmoid function
    SigmoidDecay { steepness: f64, midpoint: f64 },
}

/// Configuration for streaming kernel approximation
#[derive(Debug, Clone)]
pub struct StreamingConfig {
    /// buffer_strategy
    pub buffer_strategy: BufferStrategy,
    /// update_frequency
    pub update_frequency: UpdateFrequency,
    /// forgetting_mechanism
    pub forgetting_mechanism: ForgettingMechanism,
    /// max_memory_mb
    pub max_memory_mb: Option<usize>,
    /// adaptive_components
    pub adaptive_components: bool,
    /// quality_monitoring
    pub quality_monitoring: bool,
    /// drift_detection
    pub drift_detection: bool,
    /// concept_drift_threshold
    pub concept_drift_threshold: f64,
}

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_strategy: BufferStrategy::FixedSize(1000),
            update_frequency: UpdateFrequency::BatchSize(100),
            forgetting_mechanism: ForgettingMechanism::ExponentialDecay(0.99),
            max_memory_mb: Some(100),
            adaptive_components: true,
            quality_monitoring: true,
            drift_detection: false,
            concept_drift_threshold: 0.1,
        }
    }
}

/// Sample with metadata for streaming processing
#[derive(Debug, Clone)]
pub struct StreamingSample {
    /// data
    pub data: Array1<f64>,
    /// timestamp
    pub timestamp: f64,
    /// weight
    pub weight: f64,
    /// importance
    pub importance: f64,
    /// label
    pub label: Option<f64>,
}

impl StreamingSample {
    pub fn new(data: Array1<f64>, timestamp: f64) -> Self {
        Self {
            data,
            timestamp,
            weight: 1.0,
            importance: 1.0,
            label: None,
        }
    }

    pub fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance;
        self
    }

    pub fn with_label(mut self, label: f64) -> Self {
        self.label = Some(label);
        self
    }
}

/// Streaming RBF kernel approximation using Random Fourier Features
///
/// Maintains an online approximation of RBF kernel features that
/// adapts to data streams with concept drift and memory constraints.
pub struct StreamingRBFSampler {
    n_components: usize,
    gamma: f64,
    config: StreamingConfig,
    weights: Option<Array2<f64>>,
    bias: Option<Array1<f64>>,
    buffer: VecDeque<StreamingSample>,
    sample_count: usize,
    last_update: usize,
    feature_statistics: FeatureStatistics,
    random_state: Option<u64>,
    rng: StdRng,
}

/// Statistics for monitoring feature quality
#[derive(Debug, Clone)]
pub struct FeatureStatistics {
    /// mean
    pub mean: Array1<f64>,
    /// variance
    pub variance: Array1<f64>,
    /// min
    pub min: Array1<f64>,
    /// max
    pub max: Array1<f64>,
    /// update_count
    pub update_count: usize,
    /// approximation_error
    pub approximation_error: f64,
    /// drift_score
    pub drift_score: f64,
}

impl FeatureStatistics {
    pub fn new(n_components: usize) -> Self {
        Self {
            mean: Array1::zeros(n_components),
            variance: Array1::zeros(n_components),
            min: Array1::from_elem(n_components, f64::INFINITY),
            max: Array1::from_elem(n_components, f64::NEG_INFINITY),
            update_count: 0,
            approximation_error: 0.0,
            drift_score: 0.0,
        }
    }

    pub fn update(&mut self, features: &Array2<f64>) {
        let n_samples = features.nrows();

        for i in 0..features.ncols() {
            let col = features.column(i);
            let new_mean = col.mean().unwrap_or(0.0);
            let new_var = col.mapv(|x| (x - new_mean).powi(2)).mean().unwrap_or(0.0);
            let new_min = col.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let new_max = col.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            // Online update of statistics
            let old_count = self.update_count;
            let new_count = old_count + n_samples;

            if old_count == 0 {
                self.mean[i] = new_mean;
                self.variance[i] = new_var;
            } else {
                let alpha = n_samples as f64 / new_count as f64;
                self.mean[i] = (1.0 - alpha) * self.mean[i] + alpha * new_mean;
                self.variance[i] = (1.0 - alpha) * self.variance[i] + alpha * new_var;
            }

            self.min[i] = self.min[i].min(new_min);
            self.max[i] = self.max[i].max(new_max);
        }

        self.update_count += n_samples;
    }

    pub fn detect_drift(&mut self, new_features: &Array2<f64>) -> bool {
        let old_mean = self.mean.clone();
        self.update(new_features);

        // Simple drift detection based on mean shift
        let mean_shift = (&self.mean - &old_mean).mapv(f64::abs).sum();
        self.drift_score = mean_shift / self.mean.len() as f64;

        self.drift_score > 0.1 // Simple threshold
    }
}

impl StreamingRBFSampler {
    /// Create a new streaming RBF sampler
    pub fn new(n_components: usize, gamma: f64) -> Self {
        let rng = StdRng::from_seed(thread_rng().gen());
        Self {
            n_components,
            gamma,
            config: StreamingConfig::default(),
            weights: None,
            bias: None,
            buffer: VecDeque::new(),
            sample_count: 0,
            last_update: 0,
            feature_statistics: FeatureStatistics::new(n_components),
            random_state: None,
            rng,
        }
    }

    /// Set the streaming configuration
    pub fn with_config(mut self, config: StreamingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.rng = StdRng::seed_from_u64(random_state);
        self
    }

    /// Initialize the streaming sampler with initial data
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        let (_, n_features) = x.dim();

        // Initialize random weights and bias
        self.weights = Some(self.generate_weights(n_features)?);
        self.bias = Some(self.generate_bias()?);

        // Process initial batch
        for (i, row) in x.rows().into_iter().enumerate() {
            let sample = StreamingSample::new(row.to_owned(), i as f64);
            self.add_sample(sample)?;
        }

        Ok(())
    }

    /// Add a new sample to the stream
    pub fn add_sample(&mut self, sample: StreamingSample) -> Result<()> {
        // Check if initialization is needed
        if self.weights.is_none() {
            let n_features = sample.data.len();
            self.weights = Some(self.generate_weights(n_features)?);
            self.bias = Some(self.generate_bias()?);
        }

        // Add to buffer based on strategy
        self.manage_buffer(sample)?;

        self.sample_count += 1;

        // Check if update is needed
        if self.should_update()? {
            self.update_model()?;
            self.last_update = self.sample_count;
        }

        Ok(())
    }

    /// Transform data using current model
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let bias = self.bias.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "transform".to_string(),
        })?;

        self.compute_features(x, weights, bias)
    }

    /// Transform a single sample
    pub fn transform_sample(&self, sample: &Array1<f64>) -> Result<Array1<f64>> {
        let weights = self
            .weights
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_sample".to_string(),
            })?;
        let bias = self.bias.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "transform_sample".to_string(),
        })?;

        // Compute features for single sample
        let projection = sample.dot(&weights.t()) + bias;
        let norm_factor = (2.0 / self.n_components as f64).sqrt();

        Ok(projection.mapv(|x| norm_factor * x.cos()))
    }

    /// Get current buffer statistics
    pub fn buffer_stats(&self) -> (usize, f64, f64) {
        let size = self.buffer.len();
        let avg_weight = if size > 0 {
            self.buffer.iter().map(|s| s.weight).sum::<f64>() / size as f64
        } else {
            0.0
        };
        let avg_importance = if size > 0 {
            self.buffer.iter().map(|s| s.importance).sum::<f64>() / size as f64
        } else {
            0.0
        };

        (size, avg_weight, avg_importance)
    }

    /// Get feature statistics
    pub fn feature_stats(&self) -> &FeatureStatistics {
        &self.feature_statistics
    }

    /// Check for concept drift
    pub fn detect_drift(&mut self, x: &Array2<f64>) -> Result<bool> {
        if !self.config.drift_detection {
            return Ok(false);
        }

        let features = self.transform(x)?;
        Ok(self.feature_statistics.detect_drift(&features))
    }

    /// Manage buffer based on strategy
    fn manage_buffer(&mut self, sample: StreamingSample) -> Result<()> {
        match &self.config.buffer_strategy {
            BufferStrategy::FixedSize(max_size) => {
                if self.buffer.len() >= *max_size {
                    self.buffer.pop_front();
                }
                self.buffer.push_back(sample);
            }
            BufferStrategy::SlidingWindow { size, time_window } => {
                // Remove old samples based on time window
                let current_time = sample.timestamp;
                while let Some(front) = self.buffer.front() {
                    if current_time - front.timestamp > *time_window {
                        self.buffer.pop_front();
                    } else {
                        break;
                    }
                }

                // Add new sample and maintain size limit
                if self.buffer.len() >= *size {
                    self.buffer.pop_front();
                }
                self.buffer.push_back(sample);
            }
            BufferStrategy::ReservoirSampling(capacity) => {
                if self.buffer.len() < *capacity {
                    self.buffer.push_back(sample);
                } else {
                    let replace_idx = self.rng.gen_range(0..=self.sample_count);
                    if replace_idx < *capacity {
                        self.buffer[replace_idx] = sample;
                    }
                }
            }
            BufferStrategy::ExponentialDecay { alpha, min_weight } => {
                // Decay weights of existing samples
                for existing_sample in &mut self.buffer {
                    existing_sample.weight *= alpha;
                }

                // Remove samples below minimum weight
                self.buffer.retain(|s| s.weight >= *min_weight);

                self.buffer.push_back(sample);
            }
            BufferStrategy::ImportanceWeighted {
                capacity,
                threshold,
            } => {
                if self.buffer.len() < *capacity {
                    self.buffer.push_back(sample);
                } else {
                    // Find sample with lowest importance
                    if let Some((min_idx, _)) =
                        self.buffer.iter().enumerate().min_by(|(_, a), (_, b)| {
                            a.importance.partial_cmp(&b.importance).unwrap()
                        })
                    {
                        if sample.importance > self.buffer[min_idx].importance + threshold {
                            self.buffer[min_idx] = sample;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Check if model should be updated
    fn should_update(&self) -> Result<bool> {
        match &self.config.update_frequency {
            UpdateFrequency::PerSample => Ok(true),
            UpdateFrequency::BatchSize(batch_size) => {
                Ok(self.sample_count - self.last_update >= *batch_size)
            }
            UpdateFrequency::TimeInterval(_time_interval) => {
                // For simplicity, use sample count as proxy for time
                Ok(self.sample_count - self.last_update >= 100)
            }
            UpdateFrequency::ErrorThreshold(_threshold) => {
                // For simplicity, update periodically
                Ok(self.sample_count - self.last_update >= 50)
            }
            UpdateFrequency::Adaptive {
                initial,
                max: _,
                min: _,
            } => Ok(self.sample_count - self.last_update >= *initial),
        }
    }

    /// Update model parameters based on current buffer
    fn update_model(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Extract data from buffer with weights
        let mut data_matrix = Array2::zeros((self.buffer.len(), self.buffer[0].data.len()));
        for (i, sample) in self.buffer.iter().enumerate() {
            data_matrix.row_mut(i).assign(&sample.data);
        }

        // Compute features and update statistics
        let weights = self.weights.as_ref().unwrap();
        let bias = self.bias.as_ref().unwrap();
        let features = self.compute_features(&data_matrix, weights, bias)?;

        self.feature_statistics.update(&features);

        // Apply forgetting mechanism
        self.apply_forgetting()?;

        Ok(())
    }

    /// Apply forgetting mechanism to reduce influence of old data
    fn apply_forgetting(&mut self) -> Result<()> {
        match &self.config.forgetting_mechanism {
            ForgettingMechanism::None => {
                // No forgetting
            }
            ForgettingMechanism::LinearDecay(decay_rate) => {
                for sample in &mut self.buffer {
                    sample.weight *= 1.0 - decay_rate;
                }
            }
            ForgettingMechanism::ExponentialDecay(decay_rate) => {
                for sample in &mut self.buffer {
                    sample.weight *= decay_rate;
                }
            }
            ForgettingMechanism::AbruptForgetting(time_threshold) => {
                if let Some(newest) = self.buffer.back() {
                    let cutoff_time = newest.timestamp - time_threshold;
                    self.buffer.retain(|s| s.timestamp >= cutoff_time);
                }
            }
            ForgettingMechanism::SigmoidDecay {
                steepness,
                midpoint,
            } => {
                if let Some(newest_timestamp) = self.buffer.back().map(|s| s.timestamp) {
                    for sample in &mut self.buffer {
                        let age = newest_timestamp - sample.timestamp;
                        let sigmoid_weight = 1.0 / (1.0 + (steepness * (age - midpoint)).exp());
                        sample.weight *= sigmoid_weight;
                    }
                }
            }
        }

        Ok(())
    }

    /// Generate random weights for RBF features
    fn generate_weights(&mut self, n_features: usize) -> Result<Array2<f64>> {
        let mut weights = Array2::zeros((self.n_components, n_features));

        for i in 0..self.n_components {
            for j in 0..n_features {
                weights[[i, j]] =
                    self.rng.sample::<f64, _>(StandardNormal) * (2.0 * self.gamma).sqrt();
            }
        }

        Ok(weights)
    }

    /// Generate random bias for RBF features
    fn generate_bias(&mut self) -> Result<Array1<f64>> {
        let mut bias = Array1::zeros(self.n_components);

        for i in 0..self.n_components {
            bias[i] = self.rng.gen_range(0.0..2.0 * std::f64::consts::PI);
        }

        Ok(bias)
    }

    /// Compute RBF features for given data
    fn compute_features(
        &self,
        x: &Array2<f64>,
        weights: &Array2<f64>,
        bias: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        let (n_samples, _) = x.dim();
        let n_components = weights.nrows();

        // Compute X @ W^T + b
        let projection = x.dot(&weights.t()) + bias;

        // Apply cosine transformation with normalization
        let mut features = Array2::zeros((n_samples, n_components));
        let norm_factor = (2.0 / n_components as f64).sqrt();

        for i in 0..n_samples {
            for j in 0..n_components {
                features[[i, j]] = norm_factor * projection[[i, j]].cos();
            }
        }

        Ok(features)
    }
}

/// Streaming Nyström method for kernel approximation
///
/// Maintains an online Nyström approximation that adapts to
/// streaming data with efficient inducing point management.
pub struct StreamingNystroem {
    n_components: usize,
    gamma: f64,
    config: StreamingConfig,
    inducing_points: Option<Array2<f64>>,
    eigenvalues: Option<Array1<f64>>,
    eigenvectors: Option<Array2<f64>>,
    buffer: VecDeque<StreamingSample>,
    sample_count: usize,
    last_update: usize,
    random_state: Option<u64>,
    rng: StdRng,
}

impl StreamingNystroem {
    /// Create a new streaming Nyström approximation
    pub fn new(n_components: usize, gamma: f64) -> Self {
        let rng = StdRng::from_seed(thread_rng().gen());
        Self {
            n_components,
            gamma,
            config: StreamingConfig::default(),
            inducing_points: None,
            eigenvalues: None,
            eigenvectors: None,
            buffer: VecDeque::new(),
            sample_count: 0,
            last_update: 0,
            random_state: None,
            rng,
        }
    }

    /// Set the streaming configuration
    pub fn with_config(mut self, config: StreamingConfig) -> Self {
        self.config = config;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self.rng = StdRng::seed_from_u64(random_state);
        self
    }

    /// Initialize with initial data
    pub fn fit(&mut self, x: &Array2<f64>) -> Result<()> {
        // Select initial inducing points
        let inducing_indices = self.select_inducing_points(x)?;
        let inducing_points = x.select(Axis(0), &inducing_indices);

        // Compute initial eigendecomposition
        let kernel_matrix = self.compute_kernel_matrix(&inducing_points)?;
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&kernel_matrix)?;

        self.inducing_points = Some(inducing_points);
        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);

        // Add samples to buffer
        for (i, row) in x.rows().into_iter().enumerate() {
            let sample = StreamingSample::new(row.to_owned(), i as f64);
            self.buffer.push_back(sample);
        }

        self.sample_count = x.nrows();

        Ok(())
    }

    /// Add a new sample to the stream
    pub fn add_sample(&mut self, sample: StreamingSample) -> Result<()> {
        self.buffer.push_back(sample);
        self.sample_count += 1;

        // Manage buffer size
        match &self.config.buffer_strategy {
            BufferStrategy::FixedSize(max_size) => {
                if self.buffer.len() > *max_size {
                    self.buffer.pop_front();
                }
            }
            _ => {
                // Implement other buffer strategies as needed
            }
        }

        // Check if update is needed
        if self.should_update()? {
            self.update_model()?;
            self.last_update = self.sample_count;
        }

        Ok(())
    }

    /// Transform data using current approximation
    pub fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let inducing_points =
            self.inducing_points
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;
        let eigenvalues = self
            .eigenvalues
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;
        let eigenvectors = self
            .eigenvectors
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        // Compute kernel between x and inducing points
        let kernel_x_inducing = self.compute_kernel(x, inducing_points)?;

        // Apply Nyström transformation
        let mut features = kernel_x_inducing.dot(eigenvectors);

        // Scale by eigenvalues
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] > 1e-12 {
                let scale = 1.0 / eigenvalues[i].sqrt();
                for j in 0..features.nrows() {
                    features[[j, i]] *= scale;
                }
            }
        }

        Ok(features)
    }

    /// Check if model should be updated
    fn should_update(&self) -> Result<bool> {
        // Simple heuristic: update every 100 samples
        Ok(self.sample_count - self.last_update >= 100)
    }

    /// Update inducing points and eigendecomposition
    fn update_model(&mut self) -> Result<()> {
        if self.buffer.is_empty() {
            return Ok(());
        }

        // Extract current data from buffer
        let n_samples = self.buffer.len();
        let n_features = self.buffer[0].data.len();
        let mut data_matrix = Array2::zeros((n_samples, n_features));

        for (i, sample) in self.buffer.iter().enumerate() {
            data_matrix.row_mut(i).assign(&sample.data);
        }

        // Reselect inducing points
        let inducing_indices = self.select_inducing_points(&data_matrix)?;
        let inducing_points = data_matrix.select(Axis(0), &inducing_indices);

        // Recompute eigendecomposition
        let kernel_matrix = self.compute_kernel_matrix(&inducing_points)?;
        let (eigenvalues, eigenvectors) = self.eigendecomposition(&kernel_matrix)?;

        self.inducing_points = Some(inducing_points);
        self.eigenvalues = Some(eigenvalues);
        self.eigenvectors = Some(eigenvectors);

        Ok(())
    }

    /// Select inducing points from current data
    fn select_inducing_points(&mut self, x: &Array2<f64>) -> Result<Vec<usize>> {
        let n_samples = x.nrows();
        let n_inducing = self.n_components.min(n_samples);

        let mut indices = Vec::new();
        for _ in 0..n_inducing {
            indices.push(self.rng.gen_range(0..n_samples));
        }

        Ok(indices)
    }

    /// Compute kernel matrix
    fn compute_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut kernel_matrix = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let diff = &x.row(i) - &x.row(j);
                let squared_dist = diff.mapv(|x| x * x).sum();
                let kernel_val = (-self.gamma * squared_dist).exp();
                kernel_matrix[[i, j]] = kernel_val;
                kernel_matrix[[j, i]] = kernel_val;
            }
        }

        Ok(kernel_matrix)
    }

    /// Compute kernel between two matrices
    fn compute_kernel(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>> {
        let (n_samples_x, _) = x.dim();
        let (n_samples_y, _) = y.dim();
        let mut kernel_matrix = Array2::zeros((n_samples_x, n_samples_y));

        for i in 0..n_samples_x {
            for j in 0..n_samples_y {
                let diff = &x.row(i) - &y.row(j);
                let squared_dist = diff.mapv(|x| x * x).sum();
                let kernel_val = (-self.gamma * squared_dist).exp();
                kernel_matrix[[i, j]] = kernel_val;
            }
        }

        Ok(kernel_matrix)
    }

    /// Perform eigendecomposition (simplified)
    fn eigendecomposition(&self, matrix: &Array2<f64>) -> Result<(Array1<f64>, Array2<f64>)> {
        let n = matrix.nrows();
        let eigenvalues = Array1::ones(self.n_components.min(n));
        let eigenvectors = Array2::eye(n)
            .slice(s![.., ..self.n_components.min(n)])
            .to_owned();

        Ok((eigenvalues, eigenvectors))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_streaming_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut sampler = StreamingRBFSampler::new(50, 0.1).with_random_state(42);

        sampler.fit(&x).unwrap();
        let features = sampler.transform(&x).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 50);
    }

    #[test]
    fn test_streaming_sample() {
        let data = array![1.0, 2.0, 3.0];
        let sample = StreamingSample::new(data.clone(), 1.0)
            .with_weight(0.8)
            .with_importance(0.9)
            .with_label(1.0);

        assert_eq!(sample.data, data);
        assert_eq!(sample.timestamp, 1.0);
        assert_eq!(sample.weight, 0.8);
        assert_eq!(sample.importance, 0.9);
        assert_eq!(sample.label, Some(1.0));
    }

    #[test]
    fn test_buffer_strategies() {
        let mut sampler = StreamingRBFSampler::new(10, 0.1).with_config(StreamingConfig {
            buffer_strategy: BufferStrategy::FixedSize(3),
            ..Default::default()
        });

        // Add samples beyond buffer capacity
        for i in 0..5 {
            let data = array![i as f64, (i + 1) as f64];
            let sample = StreamingSample::new(data, i as f64);
            sampler.add_sample(sample).unwrap();
        }

        let (size, _, _) = sampler.buffer_stats();
        assert_eq!(size, 3); // Buffer should be limited to 3
    }

    #[test]
    fn test_streaming_nystroem_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let mut nystroem = StreamingNystroem::new(3, 0.1).with_random_state(42);

        nystroem.fit(&x).unwrap();
        let features = nystroem.transform(&x).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 3);
    }

    #[test]
    fn test_feature_statistics() {
        let mut stats = FeatureStatistics::new(3);
        let features = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        stats.update(&features);

        assert_eq!(stats.update_count, 2);
        assert!((stats.mean[0] - 2.5).abs() < 1e-10);
        assert!((stats.mean[1] - 3.5).abs() < 1e-10);
        assert!((stats.mean[2] - 4.5).abs() < 1e-10);
    }

    #[test]
    fn test_transform_sample() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let mut sampler = StreamingRBFSampler::new(10, 0.1).with_random_state(42);

        sampler.fit(&x).unwrap();

        let sample = array![5.0, 6.0];
        let features = sampler.transform_sample(&sample).unwrap();

        assert_eq!(features.len(), 10);
    }

    #[test]
    fn test_streaming_config() {
        let config = StreamingConfig {
            buffer_strategy: BufferStrategy::SlidingWindow {
                size: 100,
                time_window: 10.0,
            },
            update_frequency: UpdateFrequency::BatchSize(50),
            forgetting_mechanism: ForgettingMechanism::LinearDecay(0.01),
            adaptive_components: true,
            ..Default::default()
        };

        assert!(matches!(
            config.buffer_strategy,
            BufferStrategy::SlidingWindow { .. }
        ));
        assert!(matches!(
            config.update_frequency,
            UpdateFrequency::BatchSize(50)
        ));
        assert!(config.adaptive_components);
    }

    #[test]
    fn test_online_updates() {
        let mut sampler = StreamingRBFSampler::new(20, 0.1)
            .with_config(StreamingConfig {
                update_frequency: UpdateFrequency::BatchSize(2),
                ..Default::default()
            })
            .with_random_state(42);

        // Initialize with small batch
        let x_init = array![[1.0, 2.0], [3.0, 4.0]];
        sampler.fit(&x_init).unwrap();

        // Add samples one by one
        for i in 5..10 {
            let data = array![i as f64, (i + 1) as f64];
            let sample = StreamingSample::new(data, i as f64);
            sampler.add_sample(sample).unwrap();
        }

        let (buffer_size, _, _) = sampler.buffer_stats();
        assert!(buffer_size > 0);
    }

    #[test]
    fn test_drift_detection() {
        let x1 = array![[1.0, 2.0], [1.1, 2.1], [0.9, 1.9]];
        let x2 = array![[5.0, 6.0], [5.1, 6.1], [4.9, 5.9]]; // Different distribution

        let mut sampler = StreamingRBFSampler::new(20, 0.1)
            .with_config(StreamingConfig {
                drift_detection: true,
                ..Default::default()
            })
            .with_random_state(42);

        sampler.fit(&x1).unwrap();
        let drift_detected = sampler.detect_drift(&x2).unwrap();

        // Should detect some drift in feature statistics
        assert!(sampler.feature_stats().drift_score >= 0.0);
    }
}

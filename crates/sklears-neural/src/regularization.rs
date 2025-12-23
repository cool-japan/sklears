//! Regularization techniques for neural networks.
//!
//! This module provides various regularization methods including L1, L2, elastic net,
//! dropout, batch normalization, and other techniques to prevent overfitting.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::{NumCast, One};
use scirs2_core::random::essentials::{Normal, Uniform};
use scirs2_core::random::{thread_rng, Distribution, Rng};
use sklears_core::types::FloatBounds;

/// Types of regularization
#[derive(Debug, Clone, PartialEq)]
pub enum RegularizationType {
    /// L1 regularization (Lasso): λ * ||w||₁
    L1,
    /// L2 regularization (Ridge): λ * ||w||₂²
    L2,
    /// Elastic Net: λ₁ * ||w||₁ + λ₂ * ||w||₂²
    ElasticNet,
    /// No regularization
    None,
}

/// Configuration for regularization
#[derive(Debug, Clone)]
pub struct RegularizationConfig<T: FloatBounds> {
    /// Type of regularization
    pub regularization_type: RegularizationType,
    /// L1 regularization strength
    pub l1_lambda: T,
    /// L2 regularization strength
    pub l2_lambda: T,
    /// Whether to include bias terms in regularization
    pub regularize_bias: bool,
}

impl<T: FloatBounds> Default for RegularizationConfig<T> {
    fn default() -> Self {
        Self {
            regularization_type: RegularizationType::None,
            l1_lambda: T::zero(),
            l2_lambda: T::zero(),
            regularize_bias: false,
        }
    }
}

impl<T: FloatBounds> RegularizationConfig<T> {
    /// Create L1 regularization configuration
    pub fn l1(lambda: T) -> Self {
        Self {
            regularization_type: RegularizationType::L1,
            l1_lambda: lambda,
            l2_lambda: T::zero(),
            regularize_bias: false,
        }
    }

    /// Create L2 regularization configuration
    pub fn l2(lambda: T) -> Self {
        Self {
            regularization_type: RegularizationType::L2,
            l1_lambda: T::zero(),
            l2_lambda: lambda,
            regularize_bias: false,
        }
    }

    /// Create Elastic Net regularization configuration
    pub fn elastic_net(l1_lambda: T, l2_lambda: T) -> Self {
        Self {
            regularization_type: RegularizationType::ElasticNet,
            l1_lambda,
            l2_lambda,
            regularize_bias: false,
        }
    }

    /// Set whether to regularize bias terms
    pub fn regularize_bias(mut self, regularize: bool) -> Self {
        self.regularize_bias = regularize;
        self
    }
}

/// Regularization implementation
pub struct Regularizer<T: FloatBounds> {
    config: RegularizationConfig<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Regularizer<T> {
    /// Create a new regularizer with the given configuration
    pub fn new(config: RegularizationConfig<T>) -> Self {
        Self { config }
    }

    /// Compute regularization loss for a weight matrix
    pub fn compute_loss(&self, weights: &Array2<T>, bias: Option<&Array1<T>>) -> T {
        let mut loss = T::zero();

        // Regularize weights
        match self.config.regularization_type {
            RegularizationType::L1 => {
                loss = loss + self.config.l1_lambda * self.l1_norm_2d(weights);
            }
            RegularizationType::L2 => {
                loss = loss + self.config.l2_lambda * self.l2_norm_squared_2d(weights);
            }
            RegularizationType::ElasticNet => {
                loss = loss + self.config.l1_lambda * self.l1_norm_2d(weights);
                loss = loss + self.config.l2_lambda * self.l2_norm_squared_2d(weights);
            }
            RegularizationType::None => {}
        }

        // Regularize bias if configured
        if self.config.regularize_bias {
            if let Some(bias_vec) = bias {
                match self.config.regularization_type {
                    RegularizationType::L1 => {
                        loss = loss + self.config.l1_lambda * self.l1_norm_1d(bias_vec);
                    }
                    RegularizationType::L2 => {
                        loss = loss + self.config.l2_lambda * self.l2_norm_squared_1d(bias_vec);
                    }
                    RegularizationType::ElasticNet => {
                        loss = loss + self.config.l1_lambda * self.l1_norm_1d(bias_vec);
                        loss = loss + self.config.l2_lambda * self.l2_norm_squared_1d(bias_vec);
                    }
                    RegularizationType::None => {}
                }
            }
        }

        loss
    }

    /// Compute regularization gradients for weights
    pub fn compute_weight_gradients(&self, weights: &Array2<T>) -> Array2<T> {
        let mut gradients = Array2::zeros(weights.dim());

        match self.config.regularization_type {
            RegularizationType::L1 => {
                gradients = gradients + &self.l1_gradient_2d(weights) * self.config.l1_lambda;
            }
            RegularizationType::L2 => {
                gradients = gradients + &self.l2_gradient_2d(weights) * self.config.l2_lambda;
            }
            RegularizationType::ElasticNet => {
                gradients = gradients + &self.l1_gradient_2d(weights) * self.config.l1_lambda;
                gradients = gradients + &self.l2_gradient_2d(weights) * self.config.l2_lambda;
            }
            RegularizationType::None => {}
        }

        gradients
    }

    /// Compute regularization gradients for bias
    pub fn compute_bias_gradients(&self, bias: &Array1<T>) -> Array1<T> {
        if !self.config.regularize_bias {
            return Array1::zeros(bias.len());
        }

        let mut gradients = Array1::zeros(bias.len());

        match self.config.regularization_type {
            RegularizationType::L1 => {
                gradients = gradients + self.l1_gradient_1d(bias) * self.config.l1_lambda;
            }
            RegularizationType::L2 => {
                gradients = gradients + self.l2_gradient_1d(bias) * self.config.l2_lambda;
            }
            RegularizationType::ElasticNet => {
                gradients = gradients + self.l1_gradient_1d(bias) * self.config.l1_lambda;
                gradients = gradients + self.l2_gradient_1d(bias) * self.config.l2_lambda;
            }
            RegularizationType::None => {}
        }

        gradients
    }

    /// Compute L1 norm of a 2D array
    fn l1_norm_2d(&self, array: &Array2<T>) -> T {
        array.iter().fold(T::zero(), |acc, &x| acc + x.abs())
    }

    /// Compute L1 norm of a 1D array
    fn l1_norm_1d(&self, array: &Array1<T>) -> T {
        array.iter().fold(T::zero(), |acc, &x| acc + x.abs())
    }

    /// Compute squared L2 norm of a 2D array
    fn l2_norm_squared_2d(&self, array: &Array2<T>) -> T {
        let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
        half * array.iter().fold(T::zero(), |acc, &x| acc + x * x)
    }

    /// Compute squared L2 norm of a 1D array
    fn l2_norm_squared_1d(&self, array: &Array1<T>) -> T {
        let half = T::from(0.5).unwrap_or_else(|| T::one() / (T::one() + T::one()));
        half * array.iter().fold(T::zero(), |acc, &x| acc + x * x)
    }

    /// Compute L1 regularization gradient (subgradient) for 2D array
    fn l1_gradient_2d(&self, array: &Array2<T>) -> Array2<T> {
        array.mapv(|x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero() // Subgradient at 0 can be any value in [-1, 1], we choose 0
            }
        })
    }

    /// Compute L1 regularization gradient (subgradient) for 1D array
    fn l1_gradient_1d(&self, array: &Array1<T>) -> Array1<T> {
        array.mapv(|x| {
            if x > T::zero() {
                T::one()
            } else if x < T::zero() {
                -T::one()
            } else {
                T::zero()
            }
        })
    }

    /// Compute L2 regularization gradient for 2D array
    fn l2_gradient_2d(&self, array: &Array2<T>) -> Array2<T> {
        array.clone()
    }

    /// Compute L2 regularization gradient for 1D array
    fn l2_gradient_1d(&self, array: &Array1<T>) -> Array1<T> {
        array.clone()
    }
}

/// Proximal operator for L1 regularization (soft thresholding)
pub fn soft_threshold<T: FloatBounds>(x: T, lambda: T) -> T {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        T::zero()
    }
}

/// Apply proximal operator for L1 regularization to an array
pub fn apply_soft_threshold_2d<T: FloatBounds>(array: &Array2<T>, lambda: T) -> Array2<T> {
    array.mapv(|x| soft_threshold(x, lambda))
}

/// Apply proximal operator for L1 regularization to a 1D array
pub fn apply_soft_threshold_1d<T: FloatBounds>(array: &Array1<T>, lambda: T) -> Array1<T> {
    array.mapv(|x| soft_threshold(x, lambda))
}

/// Early stopping implementation
#[derive(Debug, Clone)]
pub struct EarlyStopping<T: FloatBounds> {
    /// Patience: number of epochs with no improvement after which training stops
    patience: usize,
    /// Minimum change in monitored quantity to qualify as an improvement
    min_delta: T,
    /// Number of epochs with no improvement
    wait: usize,
    /// Best value seen so far
    best_value: Option<T>,
    /// Whether lower values are better (for loss) or higher values are better (for accuracy)
    minimize: bool,
    /// Whether early stopping has been triggered
    stopped: bool,
}

impl<T: FloatBounds> EarlyStopping<T> {
    /// Create a new early stopping monitor
    ///
    /// # Arguments
    /// * `patience` - Number of epochs with no improvement after which training stops
    /// * `min_delta` - Minimum change to qualify as an improvement
    /// * `minimize` - Whether lower values are better (true for loss, false for accuracy)
    pub fn new(patience: usize, min_delta: T, minimize: bool) -> Self {
        Self {
            patience,
            min_delta,
            wait: 0,
            best_value: None,
            minimize,
            stopped: false,
        }
    }

    /// Update the early stopping monitor with a new value
    ///
    /// Returns true if training should stop
    pub fn update(&mut self, value: T) -> bool {
        if self.stopped {
            return true;
        }

        let is_improvement = match self.best_value {
            None => true,
            Some(best) => {
                if self.minimize {
                    value < best - self.min_delta
                } else {
                    value > best + self.min_delta
                }
            }
        };

        if is_improvement {
            self.best_value = Some(value);
            self.wait = 0;
        } else {
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
                return true;
            }
        }

        false
    }

    /// Get the best value seen so far
    pub fn best_value(&self) -> Option<T> {
        self.best_value
    }

    /// Reset the early stopping monitor
    pub fn reset(&mut self) {
        self.wait = 0;
        self.best_value = None;
        self.stopped = false;
    }

    /// Check if early stopping has been triggered
    pub fn is_stopped(&self) -> bool {
        self.stopped
    }
}

/// Noise injection types for regularization
#[derive(Debug, Clone, PartialEq)]
pub enum NoiseType {
    /// Gaussian noise with zero mean and specified standard deviation
    Gaussian { std_dev: f64 },
    /// Uniform noise in the range [-magnitude, magnitude]
    Uniform { magnitude: f64 },
    /// Salt-and-pepper noise (random values set to min/max)
    SaltPepper {
        probability: f64,
        min_value: f64,
        max_value: f64,
    },
    /// Dropout noise (randomly set values to zero)
    Dropout { probability: f64 },
}

/// Noise injection configuration
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// Type of noise to inject
    pub noise_type: NoiseType,
    /// Whether to apply noise during training only or both training and inference
    pub training_only: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for NoiseConfig {
    fn default() -> Self {
        Self {
            noise_type: NoiseType::Gaussian { std_dev: 0.01 },
            training_only: true,
            seed: None,
        }
    }
}

impl NoiseConfig {
    /// Create Gaussian noise configuration
    pub fn gaussian(std_dev: f64) -> Self {
        Self {
            noise_type: NoiseType::Gaussian { std_dev },
            training_only: true,
            seed: None,
        }
    }

    /// Create uniform noise configuration
    pub fn uniform(magnitude: f64) -> Self {
        Self {
            noise_type: NoiseType::Uniform { magnitude },
            training_only: true,
            seed: None,
        }
    }

    /// Create salt-and-pepper noise configuration
    pub fn salt_pepper(probability: f64, min_value: f64, max_value: f64) -> Self {
        Self {
            noise_type: NoiseType::SaltPepper {
                probability,
                min_value,
                max_value,
            },
            training_only: true,
            seed: None,
        }
    }

    /// Create dropout noise configuration
    pub fn dropout(probability: f64) -> Self {
        Self {
            noise_type: NoiseType::Dropout { probability },
            training_only: true,
            seed: None,
        }
    }

    /// Set whether to apply noise only during training
    pub fn training_only(mut self, training_only: bool) -> Self {
        self.training_only = training_only;
        self
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Noise injector for robustness training
pub struct NoiseInjector {
    config: NoiseConfig,
}

impl NoiseInjector {
    /// Create a new noise injector
    pub fn new(config: NoiseConfig) -> Self {
        Self { config }
    }

    /// Apply noise to input data
    pub fn apply_noise<T>(&self, input: &Array2<T>, is_training: bool) -> Array2<T>
    where
        T: FloatBounds + From<f64>,
    {
        if self.config.training_only && !is_training {
            return input.clone();
        }

        match &self.config.noise_type {
            NoiseType::Gaussian { std_dev } => self.apply_gaussian_noise(input, *std_dev),
            NoiseType::Uniform { magnitude } => self.apply_uniform_noise(input, *magnitude),
            NoiseType::SaltPepper {
                probability,
                min_value,
                max_value,
            } => self.apply_salt_pepper_noise(input, *probability, *min_value, *max_value),
            NoiseType::Dropout { probability } => self.apply_dropout_noise(input, *probability),
        }
    }

    /// Apply noise to 1D data (e.g., biases)
    pub fn apply_noise_1d<T>(&self, input: &Array1<T>, is_training: bool) -> Array1<T>
    where
        T: FloatBounds + From<f64>,
    {
        if self.config.training_only && !is_training {
            return input.clone();
        }

        match &self.config.noise_type {
            NoiseType::Gaussian { std_dev } => self.apply_gaussian_noise_1d(input, *std_dev),
            NoiseType::Uniform { magnitude } => self.apply_uniform_noise_1d(input, *magnitude),
            NoiseType::SaltPepper {
                probability,
                min_value,
                max_value,
            } => self.apply_salt_pepper_noise_1d(input, *probability, *min_value, *max_value),
            NoiseType::Dropout { probability } => self.apply_dropout_noise_1d(input, *probability),
        }
    }

    /// Apply Gaussian noise to 2D array
    fn apply_gaussian_noise<T>(&self, input: &Array2<T>, std_dev: f64) -> Array2<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std_dev).unwrap();

        input.mapv(|x| {
            let noise = NumCast::from(normal.sample(&mut rng)).unwrap_or(T::zero());
            x + noise
        })
    }

    /// Apply Gaussian noise to 1D array
    fn apply_gaussian_noise_1d<T>(&self, input: &Array1<T>, std_dev: f64) -> Array1<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, std_dev).unwrap();

        input.mapv(|x| {
            let noise = NumCast::from(normal.sample(&mut rng)).unwrap_or(T::zero());
            x + noise
        })
    }

    /// Apply uniform noise to 2D array
    fn apply_uniform_noise<T>(&self, input: &Array2<T>, magnitude: f64) -> Array2<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();
        let uniform = Uniform::new(-magnitude, magnitude).unwrap();

        input.mapv(|x| {
            let noise = NumCast::from(uniform.sample(&mut rng)).unwrap_or(T::zero());
            x + noise
        })
    }

    /// Apply uniform noise to 1D array
    fn apply_uniform_noise_1d<T>(&self, input: &Array1<T>, magnitude: f64) -> Array1<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();
        let uniform = Uniform::new(-magnitude, magnitude).unwrap();

        input.mapv(|x| {
            let noise = NumCast::from(uniform.sample(&mut rng)).unwrap_or(T::zero());
            x + noise
        })
    }

    /// Apply salt-and-pepper noise to 2D array
    fn apply_salt_pepper_noise<T>(
        &self,
        input: &Array2<T>,
        probability: f64,
        min_value: f64,
        max_value: f64,
    ) -> Array2<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();

        input.mapv(|x| {
            if rng.gen::<f64>() < probability {
                if rng.gen::<bool>() {
                    NumCast::from(min_value).unwrap_or(T::zero())
                } else {
                    NumCast::from(max_value).unwrap_or(T::zero())
                }
            } else {
                x
            }
        })
    }

    /// Apply salt-and-pepper noise to 1D array
    fn apply_salt_pepper_noise_1d<T>(
        &self,
        input: &Array1<T>,
        probability: f64,
        min_value: f64,
        max_value: f64,
    ) -> Array1<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();

        input.mapv(|x| {
            if rng.gen::<f64>() < probability {
                if rng.gen::<bool>() {
                    NumCast::from(min_value).unwrap_or(T::zero())
                } else {
                    NumCast::from(max_value).unwrap_or(T::zero())
                }
            } else {
                x
            }
        })
    }

    /// Apply dropout noise to 2D array
    fn apply_dropout_noise<T>(&self, input: &Array2<T>, probability: f64) -> Array2<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();

        input.mapv(|x| {
            if rng.gen::<f64>() < probability {
                T::zero()
            } else {
                // Scale up remaining values to maintain expected value
                x / NumCast::from(1.0 - probability).unwrap_or_else(T::one)
            }
        })
    }

    /// Apply dropout noise to 1D array
    fn apply_dropout_noise_1d<T>(&self, input: &Array1<T>, probability: f64) -> Array1<T>
    where
        T: FloatBounds + From<f64>,
    {
        let mut rng = thread_rng();

        input.mapv(|x| {
            if rng.gen::<f64>() < probability {
                T::zero()
            } else {
                // Scale up remaining values to maintain expected value
                x / NumCast::from(1.0 - probability).unwrap_or_else(T::one)
            }
        })
    }
}

/// Spectral normalization for constraining the spectral norm of weight matrices
#[derive(Debug, Clone)]
pub struct SpectralNormalization<T: FloatBounds> {
    /// Number of power iteration steps
    power_iterations: usize,
    /// Tolerance for convergence
    eps: T,
    /// Cached dominant left singular vector
    u: Option<Array1<T>>,
    /// Cached dominant right singular vector  
    v: Option<Array1<T>>,
    /// Whether to initialize vectors
    initialized: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> SpectralNormalization<T> {
    /// Create a new spectral normalization instance
    pub fn new(power_iterations: usize, eps: T) -> Self {
        Self {
            power_iterations,
            eps,
            u: None,
            v: None,
            initialized: false,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self::new(1, T::from(1e-12).unwrap_or_else(|| T::epsilon()))
    }

    /// Apply spectral normalization to a weight matrix
    pub fn normalize_weights(&mut self, weights: &Array2<T>) -> Array2<T>
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone + std::fmt::Debug,
    {
        let (m, n) = weights.dim();

        // Initialize u and v vectors if needed
        if !self.initialized || self.u.is_none() || self.v.is_none() {
            self.initialize_vectors(m, n);
        }

        // Ensure vectors have correct dimensions
        if let (Some(ref u), Some(ref v)) = (&self.u, &self.v) {
            if u.len() != m || v.len() != n {
                self.initialize_vectors(m, n);
            }
        }

        // Perform power iteration to find dominant singular value
        let sigma = self.power_iteration(weights);

        // Normalize weights by dividing by spectral norm
        if sigma > self.eps {
            weights / sigma
        } else {
            weights.clone()
        }
    }

    /// Initialize u and v vectors with random values
    fn initialize_vectors(&mut self, m: usize, n: usize) {
        let mut rng = thread_rng();

        // Initialize u vector (left singular vector)
        let u_data: Vec<T> = (0..m)
            .map(|_| T::from(rng.gen::<f64>() * 2.0 - 1.0).unwrap_or(T::zero()))
            .collect();
        let mut u = Array1::from_vec(u_data);
        self.normalize_vector(&mut u);
        self.u = Some(u);

        // Initialize v vector (right singular vector)
        let v_data: Vec<T> = (0..n)
            .map(|_| T::from(rng.gen::<f64>() * 2.0 - 1.0).unwrap_or(T::zero()))
            .collect();
        let mut v = Array1::from_vec(v_data);
        self.normalize_vector(&mut v);
        self.v = Some(v);

        self.initialized = true;
    }

    /// Perform power iteration to estimate dominant singular value
    fn power_iteration(&mut self, weights: &Array2<T>) -> T
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone,
    {
        for _ in 0..self.power_iterations {
            // v = W^T @ u / ||W^T @ u||
            let wt_u = {
                let u = self.u.as_ref().unwrap();
                weights.t().dot(u)
            };
            *self.v.as_mut().unwrap() = wt_u;
            Self::normalize_vector_static(&mut self.v.as_mut().unwrap(), self.eps);

            // u = W @ v / ||W @ v||
            let w_v = {
                let v = self.v.as_ref().unwrap();
                weights.dot(v)
            };
            *self.u.as_mut().unwrap() = w_v;
            Self::normalize_vector_static(&mut self.u.as_mut().unwrap(), self.eps);
        }

        // Compute spectral norm: σ = u^T @ W @ v
        let u = self.u.as_ref().unwrap();
        let v = self.v.as_ref().unwrap();
        let w_v = weights.dot(v);
        u.dot(&w_v)
    }

    /// Normalize a vector to unit length
    fn normalize_vector(&self, vector: &mut Array1<T>)
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone,
    {
        Self::normalize_vector_static(vector, self.eps);
    }

    /// Static version of normalize_vector to avoid borrow checker issues
    fn normalize_vector_static(vector: &mut Array1<T>, eps: T)
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone,
    {
        let norm_squared = vector.iter().fold(T::zero(), |acc, &x| acc + x * x);
        let norm = norm_squared.sqrt();

        if norm > eps {
            vector.mapv_inplace(|x| x / norm);
        }
    }

    /// Get the current estimate of the spectral norm
    pub fn get_spectral_norm(&mut self, weights: &Array2<T>) -> T
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone,
    {
        if !self.initialized {
            let (m, n) = weights.dim();
            self.initialize_vectors(m, n);
        }

        self.power_iteration(weights)
    }

    /// Reset the cached vectors (useful when weight dimensions change)
    pub fn reset(&mut self) {
        self.u = None;
        self.v = None;
        self.initialized = false;
    }
}

/// Spectral normalization layer that can be applied to any linear layer
#[derive(Debug, Clone)]
pub struct SpectralNormLayer<T: FloatBounds> {
    spectral_norm: SpectralNormalization<T>,
    /// Whether spectral normalization is enabled
    enabled: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> SpectralNormLayer<T> {
    /// Create a new spectral normalization layer
    pub fn new(power_iterations: usize, eps: T) -> Self {
        Self {
            spectral_norm: SpectralNormalization::new(power_iterations, eps),
            enabled: true,
        }
    }

    /// Create with default parameters
    pub fn default() -> Self {
        Self {
            spectral_norm: SpectralNormalization::default(),
            enabled: true,
        }
    }

    /// Enable or disable spectral normalization
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }

    /// Apply spectral normalization to weights
    pub fn normalize(&mut self, weights: &Array2<T>) -> Array2<T>
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone + std::fmt::Debug,
    {
        if self.enabled {
            self.spectral_norm.normalize_weights(weights)
        } else {
            weights.clone()
        }
    }

    /// Get spectral norm of weights
    pub fn spectral_norm(&mut self, weights: &Array2<T>) -> T
    where
        T: scirs2_core::ndarray::ScalarOperand + Clone,
    {
        self.spectral_norm.get_spectral_norm(weights)
    }

    /// Reset the spectral normalization state
    pub fn reset(&mut self) {
        self.spectral_norm.reset();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_l1_regularization() {
        let config = RegularizationConfig::l1(0.1);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, -2.0], [3.0, -4.0]];
        let loss = regularizer.compute_loss(&weights, None);

        // L1 norm = |1| + |-2| + |3| + |-4| = 10
        // L1 loss = 0.1 * 10 = 1.0
        assert_abs_diff_eq!(loss, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_regularization() {
        let config = RegularizationConfig::l2(0.1);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let loss = regularizer.compute_loss(&weights, None);

        // L2 squared norm = 1² + 2² + 3² + 4² = 30
        // L2 loss = 0.1 * 0.5 * 30 = 1.5
        assert_abs_diff_eq!(loss, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_elastic_net_regularization() {
        let config = RegularizationConfig::elastic_net(0.1, 0.05);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, -2.0], [3.0, -4.0]];
        let loss = regularizer.compute_loss(&weights, None);

        // L1 norm = 10, L2 squared norm = 30
        // Elastic net loss = 0.1 * 10 + 0.05 * 0.5 * 30 = 1.0 + 0.75 = 1.75
        assert_abs_diff_eq!(loss, 1.75, epsilon = 1e-10);
    }

    #[test]
    fn test_l1_gradients() {
        let config = RegularizationConfig::l1(0.1);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, -2.0, 0.0], [3.0, -4.0, 0.0]];
        let gradients = regularizer.compute_weight_gradients(&weights);

        let expected = array![[0.1, -0.1, 0.0], [0.1, -0.1, 0.0]];
        // Compare element by element since approx doesn't implement AbsDiffEq for Array2
        for (g, e) in gradients.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_l2_gradients() {
        let config = RegularizationConfig::l2(0.1);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, 2.0], [3.0, 4.0]];
        let gradients = regularizer.compute_weight_gradients(&weights);

        let expected = &weights * 0.1;
        // Compare element by element since approx doesn't implement AbsDiffEq for Array2
        for (g, e) in gradients.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*g, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_bias_regularization() {
        let config = RegularizationConfig::l2(0.1).regularize_bias(true);
        let regularizer = Regularizer::new(config);

        let weights = array![[1.0, 2.0]];
        let bias = array![3.0, 4.0];
        let loss = regularizer.compute_loss(&weights, Some(&bias));

        // L2 loss for weights = 0.1 * 0.5 * (1² + 2²) = 0.25
        // L2 loss for bias = 0.1 * 0.5 * (3² + 4²) = 1.25
        // Total = 1.5
        assert_abs_diff_eq!(loss, 1.5, epsilon = 1e-10);
    }

    #[test]
    fn test_soft_threshold() {
        assert_abs_diff_eq!(soft_threshold(3.0, 1.0), 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(soft_threshold(-3.0, 1.0), -2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(soft_threshold(0.5, 1.0), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(soft_threshold(-0.5, 1.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_soft_threshold_array() {
        let input = array![[3.0, -2.0, 0.5], [-0.3, 4.0, -1.5]];
        let result = apply_soft_threshold_2d(&input, 1.0);

        let expected = array![[2.0, -1.0, 0.0], [0.0, 3.0, -0.5]];
        // Compare element by element since approx doesn't implement AbsDiffEq for Array2
        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(*r, *e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_early_stopping_minimize() {
        let mut early_stopping = EarlyStopping::new(3, 0.01, true);

        // Improving values
        assert!(!early_stopping.update(1.0));
        assert!(!early_stopping.update(0.5));
        assert!(!early_stopping.update(0.3));

        // No improvement for 3 epochs
        assert!(!early_stopping.update(0.31)); // wait = 1
        assert!(!early_stopping.update(0.32)); // wait = 2
        assert!(early_stopping.update(0.33)); // wait = 3, should stop

        assert_abs_diff_eq!(early_stopping.best_value().unwrap(), 0.3, epsilon = 1e-10);
    }

    #[test]
    fn test_early_stopping_maximize() {
        let mut early_stopping = EarlyStopping::new(2, 0.01, false);

        // Improving values (higher is better)
        assert!(!early_stopping.update(0.7));
        assert!(!early_stopping.update(0.8));
        assert!(!early_stopping.update(0.9));

        // No improvement for 2 epochs
        assert!(!early_stopping.update(0.89)); // wait = 1
        assert!(early_stopping.update(0.88)); // wait = 2, should stop

        assert_abs_diff_eq!(early_stopping.best_value().unwrap(), 0.9, epsilon = 1e-10);
    }

    #[test]
    fn test_early_stopping_reset() {
        let mut early_stopping = EarlyStopping::new(2, 0.01, true);

        early_stopping.update(1.0);
        early_stopping.update(1.1); // wait = 1
        early_stopping.update(1.2); // wait = 2, should stop

        assert!(early_stopping.is_stopped());

        early_stopping.reset();
        assert!(!early_stopping.is_stopped());
        assert!(early_stopping.best_value().is_none());
    }

    #[test]
    fn test_gaussian_noise() {
        let config = NoiseConfig::gaussian(0.1);
        let injector = NoiseInjector::new(config);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let noisy_output = injector.apply_noise(&input, true);

        // Check that output has same shape
        assert_eq!(noisy_output.shape(), input.shape());

        // Check that noise was actually applied (values should be different)
        let mut has_differences = false;
        for (original, noisy) in input.iter().zip(noisy_output.iter()) {
            if ((*original - *noisy) as f64).abs() > 1e-6_f64 {
                has_differences = true;
                break;
            }
        }
        assert!(has_differences);
    }

    #[test]
    fn test_uniform_noise() {
        let config = NoiseConfig::uniform(0.5);
        let injector = NoiseInjector::new(config);

        let input = array![[1.0, 2.0], [3.0, 4.0]];
        let noisy_output = injector.apply_noise(&input, true);

        // Check that output has same shape
        assert_eq!(noisy_output.shape(), input.shape());

        // Check that noise is within expected bounds (roughly)
        for (original, noisy) in input.iter().zip(noisy_output.iter()) {
            let diff = ((*original - *noisy) as f64).abs();
            assert!(diff <= 0.6_f64); // Allow some tolerance for floating point
        }
    }

    #[test]
    fn test_dropout_noise() {
        let config = NoiseConfig::dropout(0.5);
        let injector = NoiseInjector::new(config);

        let input = Array2::from_elem((100, 4), 1.0); // Create larger array to test probability
        let noisy_output = injector.apply_noise(&input, true);

        // Check that approximately half the values are zero (with some tolerance)
        let zero_count = noisy_output.iter().filter(|&&x| x == 0.0).count();
        let total_count = noisy_output.len();
        let zero_ratio = zero_count as f64 / total_count as f64;

        // Should be roughly 50% with some tolerance for randomness
        assert!(zero_ratio > 0.3 && zero_ratio < 0.7);
    }

    #[test]
    fn test_salt_pepper_noise() {
        let config = NoiseConfig::salt_pepper(0.3, -1.0, 1.0);
        let injector = NoiseInjector::new(config);

        let input = Array2::from_elem((10, 10), 0.5); // Create uniform array
        let noisy_output = injector.apply_noise(&input, true);

        // Check that some values are now -1.0 or 1.0
        let extreme_count = noisy_output
            .iter()
            .filter(|&&x| (x - (-1.0_f64)).abs() < 1e-6_f64 || (x - 1.0_f64).abs() < 1e-6_f64)
            .count();

        assert!(extreme_count > 0);
    }

    #[test]
    fn test_noise_training_only() {
        let config = NoiseConfig::gaussian(0.1).training_only(true);
        let injector = NoiseInjector::new(config);

        let input = array![[1.0, 2.0], [3.0, 4.0]];

        // During training - should apply noise
        let training_output = injector.apply_noise(&input, true);
        assert_ne!(training_output, input);

        // During inference - should not apply noise
        let inference_output = injector.apply_noise(&input, false);
        assert_eq!(inference_output, input);
    }

    #[test]
    fn test_noise_1d_arrays() {
        let config = NoiseConfig::gaussian(0.1);
        let injector = NoiseInjector::new(config);

        let input = array![1.0, 2.0, 3.0, 4.0];
        let noisy_output = injector.apply_noise_1d(&input, true);

        // Check that output has same shape
        assert_eq!(noisy_output.len(), input.len());

        // Check that noise was applied
        assert_ne!(noisy_output, input);
    }

    #[test]
    fn test_spectral_normalization_basic() {
        let mut spec_norm = SpectralNormalization::new(5, 1e-6);

        // Create a matrix with known spectral norm
        let weights = array![[3.0, 0.0], [0.0, 2.0]]; // Spectral norm should be 3.0

        let normalized = spec_norm.normalize_weights(&weights);

        // Check that spectral norm is approximately 1.0
        let spectral_norm = spec_norm.get_spectral_norm(&normalized);
        assert!(spectral_norm >= 0.95 && spectral_norm <= 1.05);
    }

    #[test]
    fn test_spectral_normalization_preserves_shape() {
        let mut spec_norm = SpectralNormalization::new(3, 1e-6);

        let weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let normalized = spec_norm.normalize_weights(&weights);

        // Shape should be preserved
        assert_eq!(normalized.shape(), weights.shape());
    }

    #[test]
    fn test_spectral_normalization_identity() {
        let mut spec_norm = SpectralNormalization::new(1, 1e-6);

        // Identity matrix should have spectral norm 1, so normalization should not change much
        let weights = array![[1.0, 0.0], [0.0, 1.0]];
        let normalized = spec_norm.normalize_weights(&weights);

        // Should be close to original since spectral norm is already ~1
        for (orig, norm) in weights.iter().zip(normalized.iter()) {
            assert_abs_diff_eq!(orig, norm, epsilon = 1e-1);
        }
    }

    #[test]
    fn test_spectral_normalization_large_values() {
        let mut spec_norm = SpectralNormalization::new(10, 1e-8);

        // Matrix with large values
        let weights = array![[100.0, 50.0], [75.0, 200.0]];
        let normalized = spec_norm.normalize_weights(&weights);

        // Spectral norm should be approximately 1.0
        let spectral_norm = spec_norm.get_spectral_norm(&normalized);
        assert!(spectral_norm >= 0.98 && spectral_norm <= 1.02);
    }

    #[test]
    fn test_spectral_norm_layer() {
        let mut layer = SpectralNormLayer::default();

        let weights = array![[5.0, 0.0], [0.0, 3.0]];

        // Should normalize when enabled
        let normalized = layer.normalize(&weights);
        let spectral_norm = layer.spectral_norm(&normalized);
        assert!(
            spectral_norm >= 0.9 && spectral_norm <= 1.3,
            "Expected spectral norm between 0.9 and 1.3, got {}",
            spectral_norm
        );

        // Should not normalize when disabled
        layer.set_enabled(false);
        let not_normalized = layer.normalize(&weights);
        assert_eq!(not_normalized, weights);
    }

    #[test]
    fn test_spectral_normalization_reset() {
        let mut spec_norm = SpectralNormalization::new(3, 1e-6);

        // Initialize with one matrix
        let weights1 = array![[1.0, 2.0], [3.0, 4.0]];
        let _ = spec_norm.normalize_weights(&weights1);
        assert!(spec_norm.initialized);

        // Reset
        spec_norm.reset();
        assert!(!spec_norm.initialized);
        assert!(spec_norm.u.is_none());
        assert!(spec_norm.v.is_none());

        // Should work with different sized matrix after reset
        let weights2 = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let _ = spec_norm.normalize_weights(&weights2);
        assert!(spec_norm.initialized);
    }

    #[test]
    fn test_spectral_normalization_convergence() {
        let mut spec_norm = SpectralNormalization::new(1, 1e-6);

        let weights = array![[2.0, 1.0], [1.0, 2.0]];

        // Test with different numbers of power iterations
        let norm_1_iter = spec_norm.get_spectral_norm(&weights);

        spec_norm.power_iterations = 10;
        spec_norm.reset();
        let norm_10_iter = spec_norm.get_spectral_norm(&weights);

        // More iterations should give more accurate result
        // For this matrix, true spectral norm is 3.0
        assert!((norm_10_iter - 3.0_f64).abs() <= (norm_1_iter - 3.0_f64).abs());
    }
}

//! Deep Learning Integration for Kernel Approximation
//!
//! This module implements advanced kernel methods inspired by deep learning,
//! including Neural Tangent Kernels (NTK), Deep Kernel Learning, and
//! infinite-width network approximations.
//!
//! # References
//! - Jacot et al. (2018): "Neural Tangent Kernel: Convergence and Generalization in Neural Networks"
//! - Wilson et al. (2016): "Deep Kernel Learning"
//! - Lee et al. (2018): "Deep Neural Networks as Gaussian Processes"
//! - Arora et al. (2019): "Exact solutions to the nonlinear dynamics of learning in deep linear neural networks"

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
use std::marker::PhantomData;

const PI: Float = std::f64::consts::PI;

/// Activation functions for neural network kernels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum Activation {
    /// ReLU activation: max(0, x)
    ReLU,
    /// Tanh activation
    Tanh,
    /// Sigmoid activation: 1 / (1 + exp(-x))
    Sigmoid,
    /// Error function (erf)
    Erf,
    /// Linear activation (identity)
    Linear,
    /// GELU activation
    GELU,
    /// Swish activation: x * sigmoid(x)
    Swish,
}

impl Activation {
    /// Apply activation function element-wise
    pub fn apply(&self, x: Float) -> Float {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Tanh => x.tanh(),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Erf => {
                // Abramowitz and Stegun approximation for erf(x)
                let sign = if x >= 0.0 { 1.0 } else { -1.0 };
                let x_abs = x.abs();
                let t = 1.0 / (1.0 + 0.3275911 * x_abs);
                let approx = 1.0
                    - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736)
                        * t
                        + 0.254829592)
                        * t
                        * (-x_abs * x_abs).exp();
                sign * approx
            }
            Activation::Linear => x,
            Activation::GELU => {
                // GELU(x) = x * Φ(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
                let sqrt_2_over_pi = (2.0 / PI).sqrt();
                0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh())
            }
            Activation::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
        }
    }

    /// Compute the kernel for this activation at correlation rho
    /// This is used in NTK computation
    pub fn kernel_value(&self, rho: Float) -> Float {
        match self {
            Activation::ReLU => {
                // K(x, x') = ||x|| ||x'|| / (2π) * (sin(θ) + (π - θ) cos(θ))
                // where cos(θ) = rho / (||x|| ||x'||)
                let theta = rho.max(-1.0).min(1.0).acos();
                (theta.sin() + (PI - theta) * theta.cos()) / (2.0 * PI)
            }
            Activation::Tanh => {
                // For tanh, use approximation from literature
                2.0 / PI * (rho * (1.0 + rho.powi(2)).sqrt()).asin()
            }
            Activation::Erf => {
                // For erf activation
                2.0 / PI * (rho / (1.0 + (1.0 - rho.powi(2)).sqrt())).asin()
            }
            Activation::Linear => rho,
            Activation::Sigmoid => {
                // Approximation for sigmoid kernel
                2.0 / PI * (rho / (1.0 + (1.0 - rho.powi(2).abs()).sqrt())).asin()
            }
            Activation::GELU => {
                // GELU kernel approximation
                let theta = rho.max(-1.0).min(1.0).acos();
                (theta.sin() + (PI - theta) * theta.cos()) / (2.0 * PI) * 1.702
            }
            Activation::Swish => {
                // Swish kernel approximation (similar to GELU)
                let theta = rho.max(-1.0).min(1.0).acos();
                (theta.sin() + (PI - theta) * theta.cos()) / (2.0 * PI) * 1.5
            }
        }
    }
}

/// Configuration for Neural Tangent Kernel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NTKConfig {
    /// Number of layers in the neural network
    pub n_layers: usize,
    /// Width of hidden layers (for finite-width approximation)
    pub hidden_width: Option<usize>,
    /// Activation function
    pub activation: Activation,
    /// Whether to use the infinite-width limit
    pub infinite_width: bool,
    /// Variance of weight initialization
    pub weight_variance: Float,
    /// Variance of bias initialization
    pub bias_variance: Float,
}

impl Default for NTKConfig {
    fn default() -> Self {
        Self {
            n_layers: 3,
            hidden_width: Some(1024),
            activation: Activation::ReLU,
            infinite_width: true,
            weight_variance: 1.0,
            bias_variance: 1.0,
        }
    }
}

/// Neural Tangent Kernel (NTK) Approximation
///
/// The Neural Tangent Kernel describes the evolution of an infinitely-wide neural network
/// during gradient descent. This implementation provides both exact infinite-width
/// computation and finite-width approximations.
///
/// # Mathematical Background
///
/// For a fully-connected neural network with L layers and activation σ:
/// - The NTK is defined as: Θ(x, x') = E[∂f/∂θ(x) · ∂f/∂θ(x')]
/// - In the infinite-width limit, the NTK remains constant during training
/// - The network evolution follows: df/dt = Θ(X, X) ∇f L(f)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::deep_learning_kernels::{NeuralTangentKernel, NTKConfig, Activation};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = NTKConfig {
///     n_layers: 3,
///     activation: Activation::ReLU,
///     infinite_width: true,
///     ..Default::default()
///  };
///
/// let ntk = NeuralTangentKernel::new(config);
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
/// let fitted = ntk.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// assert_eq!(features.shape()[0], 2);
/// ```
#[derive(Debug, Clone)]
pub struct NeuralTangentKernel<State = Untrained> {
    config: NTKConfig,
    n_components: usize,

    // Fitted attributes
    x_train: Option<Array2<Float>>,
    eigenvectors: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl NeuralTangentKernel<Untrained> {
    /// Create a new Neural Tangent Kernel with the given configuration
    pub fn new(config: NTKConfig) -> Self {
        Self {
            config,
            n_components: 100,
            x_train: None,
            eigenvectors: None,
            _state: PhantomData,
        }
    }

    /// Create a new NTK with default configuration
    pub fn with_layers(n_layers: usize) -> Self {
        Self {
            config: NTKConfig {
                n_layers,
                ..Default::default()
            },
            n_components: 100,
            x_train: None,
            eigenvectors: None,
            _state: PhantomData,
        }
    }

    /// Set activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.config.activation = activation;
        self
    }

    /// Set whether to use infinite-width limit
    pub fn infinite_width(mut self, infinite: bool) -> Self {
        self.config.infinite_width = infinite;
        self
    }

    /// Set number of components
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Compute the NTK kernel matrix between two sets of points
    fn compute_ntk_kernel(&self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Array2<Float>> {
        let n_samples_x = x.nrows();
        let n_samples_y = y.nrows();

        // Initialize kernel matrix with dot product kernel (layer 0)
        let mut kernel = x.dot(&y.t());

        // Normalize by input dimension for NTK parameterization
        let d = x.ncols() as Float;
        kernel.mapv_inplace(|k| k / d);

        // Recursively compute kernels through layers
        for _layer in 0..self.config.n_layers {
            let mut new_kernel = Array2::zeros((n_samples_x, n_samples_y));

            for i in 0..n_samples_x {
                for j in 0..n_samples_y {
                    let k_ij = kernel[[i, j]];
                    let k_ii = if i < kernel.nrows() && i < kernel.ncols() {
                        kernel[[i, i]]
                    } else {
                        1.0
                    };
                    let k_jj = if j < kernel.nrows() && j < kernel.ncols() {
                        kernel[[j, j]]
                    } else {
                        1.0
                    };

                    // Compute correlation
                    let norm = (k_ii * k_jj).sqrt().max(1e-10);
                    let rho = (k_ij / norm).max(-1.0).min(1.0);

                    // Apply activation kernel
                    let activated = self.config.activation.kernel_value(rho);

                    // Scale by variances
                    new_kernel[[i, j]] =
                        self.config.weight_variance * norm * activated + self.config.bias_variance;
                }
            }

            kernel = new_kernel;
        }

        Ok(kernel)
    }

    /// Compute top k eigenvectors using power iteration
    fn compute_top_eigenvectors(&self, kernel: &Array2<Float>, k: usize) -> Result<Array2<Float>> {
        let n = kernel.nrows();
        let mut eigenvectors = Array2::zeros((n, k));
        let mut kernel_deflated = kernel.clone();

        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        for i in 0..k {
            // Random initialization
            let mut v = Array1::from_shape_fn(n, |_| rng.sample(normal));

            // Power iteration
            for _iter in 0..50 {
                v = kernel_deflated.dot(&v);
                let norm = v.dot(&v).sqrt();
                if norm > 1e-10 {
                    v /= norm;
                } else {
                    break;
                }
            }

            // Store eigenvector
            for j in 0..n {
                eigenvectors[[j, i]] = v[j];
            }

            // Deflate kernel
            let lambda = v.dot(&kernel_deflated.dot(&v));
            for row in 0..n {
                for col in 0..n {
                    kernel_deflated[[row, col]] -= lambda * v[row] * v[col];
                }
            }
        }

        Ok(eigenvectors)
    }
}

impl Estimator for NeuralTangentKernel<Untrained> {
    type Config = NTKConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for NeuralTangentKernel<Untrained> {
    type Fitted = NeuralTangentKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        // Store training data
        let x_train = x.clone();

        // For dimensionality reduction, compute eigendecomposition of kernel matrix
        let kernel = self.compute_ntk_kernel(x, x)?;

        // Use eigendecomposition for feature extraction
        let n_components = x.nrows().min(self.n_components);

        // Simple eigendecomposition using power iteration for top eigenvectors
        let eigenvectors = if n_components < x.nrows() {
            Some(self.compute_top_eigenvectors(&kernel, n_components)?)
        } else {
            None
        };

        Ok(NeuralTangentKernel {
            config: self.config,
            n_components: self.n_components,
            x_train: Some(x_train),
            eigenvectors,
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for NeuralTangentKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_train = self.x_train.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                x_train.ncols(),
                x.ncols()
            )));
        }

        // Compute kernel between x and training data
        let ntk = NeuralTangentKernel::<Untrained> {
            config: self.config.clone(),
            n_components: self.n_components,
            x_train: None,
            eigenvectors: None,
            _state: PhantomData,
        };
        let kernel = ntk.compute_ntk_kernel(x, x_train)?;

        // Project onto eigenvectors if available
        if let Some(ref eigvecs) = self.eigenvectors {
            Ok(kernel.dot(eigvecs))
        } else {
            Ok(kernel)
        }
    }
}

impl NeuralTangentKernel<Trained> {
    /// Get the training data
    pub fn x_train(&self) -> &Array2<Float> {
        self.x_train.as_ref().unwrap()
    }

    /// Get the eigenvectors
    pub fn eigenvectors(&self) -> Option<&Array2<Float>> {
        self.eigenvectors.as_ref()
    }
}

/// Deep Kernel Learning combines deep neural networks with kernel methods
///
/// This approach uses a deep neural network to learn a feature representation,
/// then applies a kernel method in the learned feature space.
///
/// # Mathematical Background
///
/// Deep Kernel Learning defines a composite kernel:
/// k(x, x') = k_base(φ(x; θ), φ(x'; θ))
/// where:
/// - φ(·; θ) is a deep neural network with parameters θ
/// - k_base is a base kernel (e.g., RBF)
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::deep_learning_kernels::{DeepKernelLearning, DKLConfig};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = DKLConfig {
///     feature_layers: vec![10, 20, 10],
///     n_components: 50,
///     ..Default::default()
/// };
///
/// let dkl = DeepKernelLearning::new(config);
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let fitted = dkl.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// assert_eq!(features.shape(), &[3, 50]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DKLConfig {
    /// Sizes of feature extraction layers
    pub feature_layers: Vec<usize>,
    /// Number of random Fourier features for final kernel
    pub n_components: usize,
    /// Activation function for feature layers
    pub activation: Activation,
    /// Bandwidth for final RBF kernel
    pub gamma: Float,
    /// Learning rate for feature learning (currently not trainable, uses random features)
    pub learning_rate: Float,
}

impl Default for DKLConfig {
    fn default() -> Self {
        Self {
            feature_layers: vec![64, 32],
            n_components: 100,
            activation: Activation::ReLU,
            gamma: 1.0,
            learning_rate: 0.01,
        }
    }
}

#[derive(Debug, Clone)]
pub struct DeepKernelLearning<State = Untrained> {
    config: DKLConfig,

    // Fitted attributes
    layer_weights: Option<Vec<Array2<Float>>>,
    layer_biases: Option<Vec<Array1<Float>>>,
    random_weights: Option<Array2<Float>>,
    random_offset: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl DeepKernelLearning<Untrained> {
    /// Create new Deep Kernel Learning with configuration
    pub fn new(config: DKLConfig) -> Self {
        Self {
            config,
            layer_weights: None,
            layer_biases: None,
            random_weights: None,
            random_offset: None,
            _state: PhantomData,
        }
    }

    /// Create with default configuration
    pub fn with_components(n_components: usize) -> Self {
        Self {
            config: DKLConfig {
                n_components,
                ..Default::default()
            },
            layer_weights: None,
            layer_biases: None,
            random_weights: None,
            random_offset: None,
            _state: PhantomData,
        }
    }

    /// Set activation function
    pub fn activation(mut self, activation: Activation) -> Self {
        self.config.activation = activation;
        self
    }

    /// Set gamma (bandwidth) for final RBF kernel
    pub fn gamma(mut self, gamma: Float) -> Self {
        self.config.gamma = gamma;
        self
    }
}

impl Estimator for DeepKernelLearning<Untrained> {
    type Config = DKLConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for DeepKernelLearning<Untrained> {
    type Fitted = DeepKernelLearning<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        // Initialize feature extraction layers with random weights
        let mut layer_weights = Vec::new();
        let mut layer_biases = Vec::new();

        let mut in_features = x.ncols();
        for &out_features in &self.config.feature_layers {
            // Xavier/Glorot initialization
            let scale = (2.0 / (in_features + out_features) as Float).sqrt();

            let weights = Array2::from_shape_fn((in_features, out_features), |_| {
                rng.sample(normal_dist) * scale
            });

            let biases = Array1::from_shape_fn(out_features, |_| rng.sample(normal_dist) * 0.01);

            layer_weights.push(weights);
            layer_biases.push(biases);
            in_features = out_features;
        }

        // Initialize random Fourier features for final kernel
        let final_features = if self.config.feature_layers.is_empty() {
            x.ncols()
        } else {
            *self.config.feature_layers.last().unwrap()
        };

        let random_weights =
            Array2::from_shape_fn((final_features, self.config.n_components), |_| {
                rng.sample(normal_dist) * (2.0 * self.config.gamma).sqrt()
            });

        let uniform_dist = Uniform::new(0.0, 2.0 * PI).unwrap();
        let random_offset =
            Array1::from_shape_fn(self.config.n_components, |_| rng.sample(uniform_dist));

        Ok(DeepKernelLearning {
            config: self.config,
            layer_weights: Some(layer_weights),
            layer_biases: Some(layer_biases),
            random_weights: Some(random_weights),
            random_offset: Some(random_offset),
            _state: PhantomData,
        })
    }
}

impl DeepKernelLearning<Trained> {
    /// Apply feature extraction layers
    fn extract_features(&self, x: &Array2<Float>) -> Array2<Float> {
        let mut features = x.clone();
        let layer_weights = self.layer_weights.as_ref().unwrap();
        let layer_biases = self.layer_biases.as_ref().unwrap();

        for (weights, biases) in layer_weights.iter().zip(layer_biases.iter()) {
            // Linear transformation
            features = features.dot(weights);

            // Add bias
            for i in 0..features.nrows() {
                for j in 0..features.ncols() {
                    features[[i, j]] += biases[j];
                }
            }

            // Apply activation
            features.mapv_inplace(|v| self.config.activation.apply(v));
        }

        features
    }
}

impl Transform<Array2<Float>, Array2<Float>> for DeepKernelLearning<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        // Extract deep features
        let deep_features = self.extract_features(x);

        // Apply random Fourier features to deep features
        let random_weights = self.random_weights.as_ref().unwrap();
        let random_offset = self.random_offset.as_ref().unwrap();

        let projection = deep_features.dot(random_weights);

        let n_samples = x.nrows();
        let mut output = Array2::zeros((n_samples, self.config.n_components));

        let normalizer = (2.0 / self.config.n_components as Float).sqrt();
        for i in 0..n_samples {
            for j in 0..self.config.n_components {
                output[[i, j]] = normalizer * (projection[[i, j]] + random_offset[j]).cos();
            }
        }

        Ok(output)
    }
}

impl DeepKernelLearning<Trained> {
    /// Get the layer weights
    pub fn layer_weights(&self) -> &Vec<Array2<Float>> {
        self.layer_weights.as_ref().unwrap()
    }

    /// Get the layer biases
    pub fn layer_biases(&self) -> &Vec<Array1<Float>> {
        self.layer_biases.as_ref().unwrap()
    }

    /// Get the random weights
    pub fn random_weights(&self) -> &Array2<Float> {
        self.random_weights.as_ref().unwrap()
    }

    /// Get the random offset
    pub fn random_offset(&self) -> &Array1<Float> {
        self.random_offset.as_ref().unwrap()
    }
}

/// Infinite-Width Network Kernel
///
/// This implements the kernel corresponding to an infinitely-wide neural network,
/// also known as the Neural Network Gaussian Process (NNGP).
///
/// # Mathematical Background
///
/// For a neural network f(x; θ) with random weights θ ~ N(0, σ²/n_h):
/// As the hidden layer width n_h → ∞, f(x; θ) → GP(0, K)
/// where K is the NNGP kernel determined by the architecture and activation.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::deep_learning_kernels::{InfiniteWidthKernel, Activation};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let kernel = InfiniteWidthKernel::new(3, Activation::ReLU);
/// let X = array![[1.0, 2.0], [3.0, 4.0]];
/// let fitted = kernel.fit(&X, &()).unwrap();
/// let features = fitted.transform(&X).unwrap();
/// assert_eq!(features.shape()[0], 2);
/// ```
#[derive(Debug, Clone)]
pub struct InfiniteWidthKernel<State = Untrained> {
    n_layers: usize,
    activation: Activation,
    n_components: usize,

    // Fitted attributes
    x_train: Option<Array2<Float>>,
    eigenvectors: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl InfiniteWidthKernel<Untrained> {
    /// Create new infinite-width kernel
    pub fn new(n_layers: usize, activation: Activation) -> Self {
        Self {
            n_layers,
            activation,
            n_components: 100,
            x_train: None,
            eigenvectors: None,
            _state: PhantomData,
        }
    }

    /// Set number of components for dimensionality reduction
    pub fn n_components(mut self, n: usize) -> Self {
        self.n_components = n;
        self
    }

    /// Compute the NNGP kernel matrix
    fn compute_nngp_kernel(&self, x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let n_x = x.nrows();
        let n_y = y.nrows();
        let d = x.ncols() as Float;

        // Initialize with normalized dot product
        let mut kernel = x.dot(&y.t());
        kernel.mapv_inplace(|k| k / d);

        // Recursively apply activation kernels
        for _ in 0..self.n_layers {
            let mut new_kernel = Array2::zeros((n_x, n_y));

            for i in 0..n_x {
                for j in 0..n_y {
                    let k_ij = kernel[[i, j]];
                    let k_ii = if i < n_y { kernel[[i, i]] } else { 1.0 };
                    let k_jj = if j < n_x { kernel[[j, j]] } else { 1.0 };

                    let norm = (k_ii * k_jj).sqrt().max(1e-10);
                    let rho = (k_ij / norm).max(-1.0).min(1.0);

                    new_kernel[[i, j]] = norm * self.activation.kernel_value(rho);
                }
            }

            kernel = new_kernel;
        }

        kernel
    }

    fn compute_top_eigenvectors(&self, kernel: &Array2<Float>, k: usize) -> Result<Array2<Float>> {
        let n = kernel.nrows();
        let mut eigenvectors = Array2::zeros((n, k));
        let mut kernel_deflated = kernel.clone();

        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0, 1.0).unwrap();

        for i in 0..k {
            let mut v = Array1::from_shape_fn(n, |_| rng.sample(normal_dist));

            // Power iteration
            for _iter in 0..50 {
                v = kernel_deflated.dot(&v);
                let norm = v.dot(&v).sqrt();
                if norm > 1e-10 {
                    v /= norm;
                } else {
                    break;
                }
            }

            for j in 0..n {
                eigenvectors[[j, i]] = v[j];
            }

            let lambda = v.dot(&kernel_deflated.dot(&v));
            for row in 0..n {
                for col in 0..n {
                    kernel_deflated[[row, col]] -= lambda * v[row] * v[col];
                }
            }
        }

        Ok(eigenvectors)
    }
}

impl Estimator for InfiniteWidthKernel<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, ()> for InfiniteWidthKernel<Untrained> {
    type Fitted = InfiniteWidthKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array cannot be empty".to_string(),
            ));
        }

        let x_train = x.clone();
        let kernel = self.compute_nngp_kernel(x, x);

        // Compute eigenvectors using power iteration
        let n_components = self.n_components.min(x.nrows());
        let eigenvectors = self.compute_top_eigenvectors(&kernel, n_components)?;

        Ok(InfiniteWidthKernel {
            n_layers: self.n_layers,
            activation: self.activation,
            n_components: self.n_components,
            x_train: Some(x_train),
            eigenvectors: Some(eigenvectors),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for InfiniteWidthKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_train = self.x_train.as_ref().unwrap();
        let eigenvectors = self.eigenvectors.as_ref().unwrap();

        if x.ncols() != x_train.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Feature dimension mismatch: expected {}, got {}",
                x_train.ncols(),
                x.ncols()
            )));
        }

        let kernel_obj = InfiniteWidthKernel::<Untrained> {
            n_layers: self.n_layers,
            activation: self.activation,
            n_components: self.n_components,
            x_train: None,
            eigenvectors: None,
            _state: PhantomData,
        };

        let kernel = kernel_obj.compute_nngp_kernel(x, x_train);
        Ok(kernel.dot(eigenvectors))
    }
}

impl InfiniteWidthKernel<Trained> {
    /// Get the training data
    pub fn x_train(&self) -> &Array2<Float> {
        self.x_train.as_ref().unwrap()
    }

    /// Get the eigenvectors
    pub fn eigenvectors(&self) -> &Array2<Float> {
        self.eigenvectors.as_ref().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_activation_functions() {
        let activations = vec![
            Activation::ReLU,
            Activation::Tanh,
            Activation::Sigmoid,
            Activation::Linear,
            Activation::GELU,
            Activation::Swish,
            Activation::Erf,
        ];

        for act in activations {
            let val = act.apply(0.5);
            assert!(val.is_finite());

            let kernel_val = act.kernel_value(0.5);
            assert!(kernel_val.is_finite());
        }
    }

    #[test]
    fn test_neural_tangent_kernel_basic() {
        let config = NTKConfig {
            n_layers: 2,
            hidden_width: Some(512),
            activation: Activation::ReLU,
            infinite_width: true,
            weight_variance: 1.0,
            bias_variance: 0.1,
        };

        let ntk = NeuralTangentKernel::new(config);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = ntk.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.nrows(), 3);
        assert!(features.ncols() > 0);
    }

    #[test]
    fn test_deep_kernel_learning() {
        let config = DKLConfig {
            feature_layers: vec![10, 20],
            n_components: 50,
            activation: Activation::ReLU,
            gamma: 1.0,
            learning_rate: 0.01,
        };

        let dkl = DeepKernelLearning::new(config);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let fitted = dkl.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[3, 50]);
    }

    #[test]
    fn test_infinite_width_kernel() {
        let kernel = InfiniteWidthKernel::new(3, Activation::ReLU);
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let fitted = kernel.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.nrows(), 4);
        assert!(features.ncols() > 0);
    }

    #[test]
    fn test_ntk_different_activations() {
        let activations = vec![Activation::ReLU, Activation::Tanh, Activation::GELU];
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        for act in activations {
            let ntk = NeuralTangentKernel::with_layers(2).activation(act);
            let fitted = ntk.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.nrows(), 2);
        }
    }

    #[test]
    fn test_dkl_feature_extraction() {
        let config = DKLConfig {
            feature_layers: vec![8, 4],
            n_components: 20,
            activation: Activation::Tanh,
            gamma: 0.5,
            learning_rate: 0.01,
        };

        let dkl = DeepKernelLearning::new(config);
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let fitted = dkl.fit(&x, &()).unwrap();

        // Test that features have correct shape
        let features = fitted.transform(&x).unwrap();
        assert_eq!(features.shape(), &[2, 20]);

        // Test that all features are finite
        for val in features.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_empty_input_error() {
        let ntk = NeuralTangentKernel::with_layers(2);
        let x_empty: Array2<Float> = Array2::zeros((0, 0));

        assert!(ntk.fit(&x_empty, &()).is_err());
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let ntk = NeuralTangentKernel::with_layers(2);
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]];

        let fitted = ntk.fit(&x_train, &()).unwrap();
        assert!(fitted.transform(&x_test).is_err());
    }
}

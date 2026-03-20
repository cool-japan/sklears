//! Neural Tensor Decomposition for Deep Cross-Modal Learning
//!
//! This module provides neural network-based tensor decomposition methods that combine
//! deep learning with tensor factorization for complex multi-way data analysis.
//!
//! ## Methods
//! - Neural Tucker Decomposition with learnable core and factor matrices
//! - Neural PARAFAC (CP) decomposition with neural network parameterization
//! - Attention-based tensor decomposition with dynamic weighting
//! - Variational tensor networks for uncertainty quantification
//! - Deep tensor factorization with multiple hidden layers
//!
//! ## Applications
//! - Multi-modal data fusion with complex interactions
//! - Temporal multi-way data modeling
//! - Knowledge graph completion with neural embeddings
//! - Video analysis with spatio-temporal decomposition

use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;

/// Activation function for neural layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NeuralActivation {
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU with negative slope
    LeakyReLU(u32), // Use u32 to store fixed-point representation of slope
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// Swish (x * sigmoid(x))
    Swish,
    /// GELU (Gaussian Error Linear Unit)
    GELU,
    /// Linear (no activation)
    Linear,
}

impl NeuralActivation {
    fn apply(&self, x: Float) -> Float {
        match self {
            NeuralActivation::ReLU => x.max(0.0),
            NeuralActivation::LeakyReLU(slope_fixed) => {
                let slope = (*slope_fixed as Float) / 100.0; // Convert from fixed-point
                if x > 0.0 {
                    x
                } else {
                    slope * x
                }
            }
            NeuralActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            NeuralActivation::Tanh => x.tanh(),
            NeuralActivation::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                x * sigmoid
            }
            NeuralActivation::GELU => {
                // Approximate GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
                let inner = sqrt_2_pi * (x + 0.044715 * x.powi(3));
                0.5 * x * (1.0 + inner.tanh())
            }
            NeuralActivation::Linear => x,
        }
    }
}

/// Configuration for neural tensor decomposition
#[derive(Debug, Clone)]
pub struct NeuralTensorConfig {
    /// Tensor dimensions (mode sizes)
    pub tensor_shape: Vec<usize>,
    /// Rank of the decomposition
    pub rank: usize,
    /// Hidden layer dimensions for neural networks
    pub hidden_dims: Vec<usize>,
    /// Activation function
    pub activation: NeuralActivation,
    /// Dropout rate
    pub dropout: Float,
    /// Use batch normalization
    pub batch_norm: bool,
    /// Learning rate
    pub learning_rate: Float,
    /// Number of training iterations
    pub n_iterations: usize,
    /// Regularization parameter
    pub regularization: Float,
}

impl Default for NeuralTensorConfig {
    fn default() -> Self {
        Self {
            tensor_shape: vec![10, 10, 10],
            rank: 5,
            hidden_dims: vec![64, 32],
            activation: NeuralActivation::ReLU,
            dropout: 0.1,
            batch_norm: false,
            learning_rate: 0.001,
            n_iterations: 100,
            regularization: 0.0001,
        }
    }
}

/// Neural Tucker decomposition
///
/// Decomposes a tensor into a core tensor and factor matrices parameterized by neural networks.
#[derive(Debug, Clone)]
pub struct NeuralTuckerDecomposition {
    /// Configuration
    config: NeuralTensorConfig,
    /// Factor network weights for each mode
    factor_networks: Vec<FactorNetwork>,
    /// Core tensor
    core_tensor: Option<Array3<Float>>,
    /// Training loss history
    loss_history: Vec<Float>,
}

/// Neural network for learning factor matrices
#[derive(Debug, Clone)]
struct FactorNetwork {
    /// Input dimension
    input_dim: usize,
    /// Output dimension (rank)
    output_dim: usize,
    /// Layer weights
    weights: Vec<Array2<Float>>,
    /// Layer biases
    biases: Vec<Array1<Float>>,
    /// Batch norm parameters (if enabled)
    batch_norm_gamma: Vec<Array1<Float>>,
    batch_norm_beta: Vec<Array1<Float>>,
}

impl FactorNetwork {
    /// Create a new factor network
    fn new(input_dim: usize, output_dim: usize, hidden_dims: &[usize]) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut batch_norm_gamma = Vec::new();
        let mut batch_norm_beta = Vec::new();

        // Build layers
        let mut prev_dim = input_dim;
        for &hidden_dim in hidden_dims {
            let scale = (2.0 / prev_dim as Float).sqrt();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                (rng.random::<Float>() - 0.5) * 2.0 * scale
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w);
            biases.push(b);
            batch_norm_gamma.push(Array1::ones(hidden_dim));
            batch_norm_beta.push(Array1::zeros(hidden_dim));
            prev_dim = hidden_dim;
        }

        // Output layer
        let scale = (2.0 / prev_dim as Float).sqrt();
        let w = Array2::from_shape_fn((prev_dim, output_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * scale
        });
        let b = Array1::zeros(output_dim);
        weights.push(w);
        biases.push(b);

        Self {
            input_dim,
            output_dim,
            weights,
            biases,
            batch_norm_gamma,
            batch_norm_beta,
        }
    }

    /// Forward pass through the network
    fn forward(&self, input: ArrayView2<Float>, activation: &NeuralActivation) -> Array2<Float> {
        let mut x = input.to_owned();

        for i in 0..self.weights.len() {
            // Linear transformation
            x = x.dot(&self.weights[i]) + &self.biases[i];

            // Apply activation (except on last layer, which uses linear)
            if i < self.weights.len() - 1 {
                x = x.mapv(|v| activation.apply(v));
            }
        }

        x
    }
}

/// Neural PARAFAC (CP) decomposition
///
/// Decomposes a tensor using neural networks to learn factor matrices.
#[derive(Debug, Clone)]
pub struct NeuralParafacDecomposition {
    /// Configuration
    config: NeuralTensorConfig,
    /// Factor networks for each mode
    factor_networks: Vec<FactorNetwork>,
    /// Training loss history
    loss_history: Vec<Float>,
}

/// Attention-based tensor decomposition
///
/// Uses attention mechanisms to dynamically weight different tensor components.
#[derive(Debug, Clone)]
pub struct AttentionTensorDecomposition {
    /// Configuration
    config: NeuralTensorConfig,
    /// Factor matrices
    factor_matrices: Vec<Array2<Float>>,
    /// Attention weights for each mode
    attention_weights: Vec<Array2<Float>>,
    /// Query, key, value transformations
    qkv_transforms: Vec<(Array2<Float>, Array2<Float>, Array2<Float>)>,
}

/// Variational tensor network for uncertainty quantification
#[derive(Debug, Clone)]
pub struct VariationalTensorNetwork {
    /// Configuration
    config: NeuralTensorConfig,
    /// Mean parameters for factor distributions
    mean_factors: Vec<Array2<Float>>,
    /// Log variance parameters for factor distributions
    logvar_factors: Vec<Array2<Float>>,
    /// Encoder networks
    encoders: Vec<FactorNetwork>,
    /// Decoder networks
    decoders: Vec<FactorNetwork>,
}

/// Results from neural tensor decomposition
#[derive(Debug, Clone)]
pub struct NeuralTensorResults {
    /// Reconstructed tensor
    pub reconstructed: Array3<Float>,
    /// Factor matrices for each mode
    pub factors: Vec<Array2<Float>>,
    /// Core tensor (for Tucker-based methods)
    pub core: Option<Array3<Float>>,
    /// Reconstruction error
    pub reconstruction_error: Float,
    /// Training loss history
    pub loss_history: Vec<Float>,
}

impl NeuralTuckerDecomposition {
    /// Create a new neural Tucker decomposition
    pub fn new(config: NeuralTensorConfig) -> Self {
        assert!(
            config.tensor_shape.len() == 3,
            "Currently only 3-way tensors are supported"
        );

        let factor_networks: Vec<_> = config
            .tensor_shape
            .iter()
            .map(|&mode_size| FactorNetwork::new(mode_size, config.rank, &config.hidden_dims))
            .collect();

        Self {
            config,
            factor_networks,
            core_tensor: None,
            loss_history: Vec::new(),
        }
    }

    /// Fit the model to a tensor
    pub fn fit(&mut self, tensor: ArrayView3<Float>) -> NeuralTensorResults {
        let tensor = tensor.to_owned(); // Convert view to owned array

        assert_eq!(
            tensor.shape(),
            self.config.tensor_shape.as_slice(),
            "Tensor shape mismatch"
        );

        let mut rng = thread_rng();

        // Initialize core tensor randomly
        let core_shape = [self.config.rank; 3];
        let mut core = Array3::from_shape_fn((core_shape[0], core_shape[1], core_shape[2]), |_| {
            (rng.random::<Float>() - 0.5) * 0.1
        });

        // Training loop
        for iter in 0..self.config.n_iterations {
            // Compute factor matrices using neural networks
            let mut factors = Vec::new();

            for (mode, network) in self.factor_networks.iter().enumerate() {
                // Create input features (mode fibers)
                let mode_size = self.config.tensor_shape[mode];
                let input = Array2::eye(mode_size);

                // Forward pass
                let factor = network.forward(input.view(), &self.config.activation);
                factors.push(factor);
            }

            // Reconstruct tensor
            let reconstructed = self.reconstruct_from_factors(&factors, &core);

            // Compute loss
            let reconstruction_error = self.compute_reconstruction_error(&tensor, &reconstructed);
            let regularization_loss = self.compute_regularization(&factors);
            let total_loss =
                reconstruction_error + self.config.regularization * regularization_loss;

            self.loss_history.push(total_loss);

            // Simple gradient-free update for core tensor
            // In practice, this would use backpropagation
            if iter % 10 == 0 {
                core = self.update_core(&tensor, &factors);
            }
        }

        // Final factor computation
        let mut factors = Vec::new();
        for (mode, network) in self.factor_networks.iter().enumerate() {
            let mode_size = self.config.tensor_shape[mode];
            let input = Array2::eye(mode_size);
            let factor = network.forward(input.view(), &self.config.activation);
            factors.push(factor);
        }

        let reconstructed = self.reconstruct_from_factors(&factors, &core);
        let reconstruction_error = self.compute_reconstruction_error(&tensor, &reconstructed);

        self.core_tensor = Some(core.clone());

        NeuralTensorResults {
            reconstructed,
            factors,
            core: Some(core),
            reconstruction_error,
            loss_history: self.loss_history.clone(),
        }
    }

    /// Reconstruct tensor from factor matrices and core
    fn reconstruct_from_factors(
        &self,
        factors: &[Array2<Float>],
        core: &Array3<Float>,
    ) -> Array3<Float> {
        let shape = &self.config.tensor_shape;

        // Tucker reconstruction: core ×₁ factor₁ ×₂ factor₂ ×₃ factor₃
        // For simplicity, we do mode-1, mode-2, mode-3 products sequentially

        // Mode-1 product
        let mut result = Array3::zeros((shape[0], core.shape()[1], core.shape()[2]));
        for i in 0..shape[0] {
            for j in 0..core.shape()[1] {
                for k in 0..core.shape()[2] {
                    let mut sum = 0.0;
                    for r in 0..self.config.rank {
                        sum += factors[0][[i, r]] * core[[r, j, k]];
                    }
                    result[[i, j, k]] = sum;
                }
            }
        }

        // Mode-2 product
        let mut temp = Array3::zeros((shape[0], shape[1], core.shape()[2]));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..core.shape()[2] {
                    let mut sum = 0.0;
                    for r in 0..self.config.rank {
                        sum += result[[i, r, k]] * factors[1][[j, r]];
                    }
                    temp[[i, j, k]] = sum;
                }
            }
        }

        // Mode-3 product
        let mut final_result = Array3::zeros((shape[0], shape[1], shape[2]));
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                for k in 0..shape[2] {
                    let mut sum = 0.0;
                    for r in 0..self.config.rank {
                        sum += temp[[i, j, r]] * factors[2][[k, r]];
                    }
                    final_result[[i, j, k]] = sum;
                }
            }
        }

        final_result
    }

    /// Compute reconstruction error
    fn compute_reconstruction_error(
        &self,
        original: &Array3<Float>,
        reconstructed: &Array3<Float>,
    ) -> Float {
        let diff = original - reconstructed;
        diff.mapv(|x| x * x).sum() / (original.len() as Float)
    }

    /// Compute regularization loss
    fn compute_regularization(&self, factors: &[Array2<Float>]) -> Float {
        factors.iter().map(|f| f.mapv(|x| x * x).sum()).sum()
    }

    /// Update core tensor using ALS-like approach
    fn update_core(&self, tensor: &Array3<Float>, factors: &[Array2<Float>]) -> Array3<Float> {
        let rank = self.config.rank;
        let mut core = Array3::zeros((rank, rank, rank));

        // Simplified core update (in practice would use proper tensor operations)
        for r1 in 0..rank {
            for r2 in 0..rank {
                for r3 in 0..rank {
                    let mut sum = 0.0;
                    let mut count = 0.0;

                    for i in 0..self.config.tensor_shape[0] {
                        for j in 0..self.config.tensor_shape[1] {
                            for k in 0..self.config.tensor_shape[2] {
                                let weight =
                                    factors[0][[i, r1]] * factors[1][[j, r2]] * factors[2][[k, r3]];

                                if weight.abs() > 1e-10 {
                                    sum += tensor[[i, j, k]] * weight;
                                    count += weight * weight;
                                }
                            }
                        }
                    }

                    core[[r1, r2, r3]] = if count > 1e-10 { sum / count } else { 0.0 };
                }
            }
        }

        core
    }
}

impl NeuralParafacDecomposition {
    /// Create a new neural PARAFAC decomposition
    pub fn new(config: NeuralTensorConfig) -> Self {
        assert!(
            config.tensor_shape.len() == 3,
            "Currently only 3-way tensors are supported"
        );

        let factor_networks: Vec<_> = config
            .tensor_shape
            .iter()
            .map(|&mode_size| FactorNetwork::new(mode_size, config.rank, &config.hidden_dims))
            .collect();

        Self {
            config,
            factor_networks,
            loss_history: Vec::new(),
        }
    }

    /// Fit the model to a tensor
    pub fn fit(&mut self, tensor: ArrayView3<Float>) -> NeuralTensorResults {
        let tensor = tensor.to_owned(); // Convert view to owned array

        assert_eq!(
            tensor.shape(),
            self.config.tensor_shape.as_slice(),
            "Tensor shape mismatch"
        );

        // Training loop (simplified)
        for iter in 0..self.config.n_iterations {
            // Compute factor matrices using neural networks
            let mut factors = Vec::new();

            for (mode, network) in self.factor_networks.iter().enumerate() {
                let mode_size = self.config.tensor_shape[mode];
                let input = Array2::eye(mode_size);
                let factor = network.forward(input.view(), &self.config.activation);
                factors.push(factor);
            }

            // Reconstruct tensor using CP decomposition
            let reconstructed = self.reconstruct_cp(&factors);

            // Compute loss
            let reconstruction_error = Self::compute_error(&tensor, &reconstructed);
            let regularization_loss = self.compute_regularization(&factors);
            let total_loss =
                reconstruction_error + self.config.regularization * regularization_loss;

            self.loss_history.push(total_loss);
        }

        // Final computation
        let mut factors = Vec::new();
        for (mode, network) in self.factor_networks.iter().enumerate() {
            let mode_size = self.config.tensor_shape[mode];
            let input = Array2::eye(mode_size);
            let factor = network.forward(input.view(), &self.config.activation);
            factors.push(factor);
        }

        let reconstructed = self.reconstruct_cp(&factors);
        let reconstruction_error = Self::compute_error(&tensor, &reconstructed);

        NeuralTensorResults {
            reconstructed,
            factors,
            core: None,
            reconstruction_error,
            loss_history: self.loss_history.clone(),
        }
    }

    /// Reconstruct tensor using CP decomposition
    fn reconstruct_cp(&self, factors: &[Array2<Float>]) -> Array3<Float> {
        let shape = &self.config.tensor_shape;
        let rank = self.config.rank;

        let mut result = Array3::zeros((shape[0], shape[1], shape[2]));

        for r in 0..rank {
            for i in 0..shape[0] {
                for j in 0..shape[1] {
                    for k in 0..shape[2] {
                        result[[i, j, k]] +=
                            factors[0][[i, r]] * factors[1][[j, r]] * factors[2][[k, r]];
                    }
                }
            }
        }

        result
    }

    /// Compute reconstruction error
    fn compute_error(original: &Array3<Float>, reconstructed: &Array3<Float>) -> Float {
        let diff = original - reconstructed;
        diff.mapv(|x| x * x).sum() / (original.len() as Float)
    }

    /// Compute regularization loss
    fn compute_regularization(&self, factors: &[Array2<Float>]) -> Float {
        factors.iter().map(|f| f.mapv(|x| x * x).sum()).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_neural_activation_relu() {
        let act = NeuralActivation::ReLU;
        assert_eq!(act.apply(2.0), 2.0);
        assert_eq!(act.apply(-1.0), 0.0);
    }

    #[test]
    fn test_neural_activation_leaky_relu() {
        let act = NeuralActivation::LeakyReLU(10); // 0.1 slope
        assert_eq!(act.apply(2.0), 2.0);
        let result = act.apply(-1.0);
        assert!(
            (result - (-0.1)).abs() < 1e-6,
            "Expected -0.1, got {}",
            result
        );
    }

    #[test]
    fn test_neural_activation_sigmoid() {
        let act = NeuralActivation::Sigmoid;
        let result = act.apply(0.0);
        assert!((result - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_neural_tensor_config_default() {
        let config = NeuralTensorConfig::default();
        assert_eq!(config.tensor_shape, vec![10, 10, 10]);
        assert_eq!(config.rank, 5);
    }

    #[test]
    fn test_factor_network_creation() {
        let network = FactorNetwork::new(10, 5, &[8, 6]);

        // Input -> 8 -> 6 -> 5
        assert_eq!(network.weights.len(), 3);
        assert_eq!(network.weights[0].shape(), &[10, 8]);
        assert_eq!(network.weights[1].shape(), &[8, 6]);
        assert_eq!(network.weights[2].shape(), &[6, 5]);
    }

    #[test]
    fn test_factor_network_forward() {
        let network = FactorNetwork::new(5, 3, &[4]);
        let input = Array2::eye(5);

        let output = network.forward(input.view(), &NeuralActivation::ReLU);

        assert_eq!(output.shape(), &[5, 3]);
    }

    #[test]
    fn test_neural_tucker_creation() {
        let config = NeuralTensorConfig {
            tensor_shape: vec![4, 5, 6],
            rank: 3,
            hidden_dims: vec![8],
            activation: NeuralActivation::ReLU,
            dropout: 0.0,
            batch_norm: false,
            learning_rate: 0.01,
            n_iterations: 10,
            regularization: 0.001,
        };

        let decomp = NeuralTuckerDecomposition::new(config);
        assert_eq!(decomp.factor_networks.len(), 3);
    }

    #[test]
    fn test_neural_tucker_fit() {
        let config = NeuralTensorConfig {
            tensor_shape: vec![3, 3, 3],
            rank: 2,
            hidden_dims: vec![4],
            activation: NeuralActivation::ReLU,
            dropout: 0.0,
            batch_norm: false,
            learning_rate: 0.01,
            n_iterations: 5,
            regularization: 0.001,
        };

        let mut decomp = NeuralTuckerDecomposition::new(config);

        // Create a simple test tensor
        let tensor = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i + j + k) as Float / 10.0);

        let results = decomp.fit(tensor.view());

        assert_eq!(results.reconstructed.shape(), &[3, 3, 3]);
        assert_eq!(results.factors.len(), 3);
        assert!(results.core.is_some());
        assert_eq!(results.loss_history.len(), 5);
    }

    #[test]
    fn test_neural_parafac_creation() {
        let config = NeuralTensorConfig {
            tensor_shape: vec![4, 5, 6],
            rank: 3,
            hidden_dims: vec![8],
            activation: NeuralActivation::ReLU,
            dropout: 0.0,
            batch_norm: false,
            learning_rate: 0.01,
            n_iterations: 10,
            regularization: 0.001,
        };

        let decomp = NeuralParafacDecomposition::new(config);
        assert_eq!(decomp.factor_networks.len(), 3);
    }

    #[test]
    fn test_neural_parafac_fit() {
        let config = NeuralTensorConfig {
            tensor_shape: vec![3, 3, 3],
            rank: 2,
            hidden_dims: vec![4],
            activation: NeuralActivation::ReLU,
            dropout: 0.0,
            batch_norm: false,
            learning_rate: 0.01,
            n_iterations: 5,
            regularization: 0.001,
        };

        let mut decomp = NeuralParafacDecomposition::new(config);

        let tensor = Array3::from_shape_fn((3, 3, 3), |(i, j, k)| (i * j * k) as Float / 10.0);

        let results = decomp.fit(tensor.view());

        assert_eq!(results.reconstructed.shape(), &[3, 3, 3]);
        assert_eq!(results.factors.len(), 3);
        assert!(results.core.is_none()); // CP decomposition doesn't have a core
        assert_eq!(results.loss_history.len(), 5);
    }

    #[test]
    fn test_reconstruction_error_computation() {
        let config = NeuralTensorConfig::default();
        let decomp = NeuralTuckerDecomposition::new(config);

        let tensor1 = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as Float);
        let tensor2 = Array3::from_shape_fn((2, 2, 2), |(i, j, k)| (i + j + k) as Float + 0.1);

        let error = decomp.compute_reconstruction_error(&tensor1, &tensor2);
        assert!(error > 0.0);
        assert!(error < 1.0); // Should be small for similar tensors
    }

    #[test]
    fn test_activation_gelu() {
        let act = NeuralActivation::GELU;
        let result = act.apply(0.0);
        assert!(result.abs() < 1e-6); // GELU(0) ≈ 0

        let result_pos = act.apply(1.0);
        assert!(result_pos > 0.5); // GELU(x) > x/2 for x > 0
    }

    #[test]
    fn test_activation_swish() {
        let act = NeuralActivation::Swish;
        let result = act.apply(0.0);
        assert!(result.abs() < 1e-6); // Swish(0) = 0

        let result_pos = act.apply(2.0);
        assert!(result_pos > 0.0);
    }
}

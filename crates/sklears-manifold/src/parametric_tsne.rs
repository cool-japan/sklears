//! Parametric t-SNE implementation
//! This module provides Parametric t-SNE for non-linear dimensionality reduction with neural network mapping.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Parametric t-SNE (t-Distributed Stochastic Neighbor Embedding)
///
/// Parametric t-SNE extends regular t-SNE by learning a parametric mapping
/// (neural network) from high-dimensional space to low-dimensional space,
/// allowing for out-of-sample projection and continuous mapping.
///
/// # Parameters
///
/// * `n_components` - Dimension of the embedded space
/// * `perplexity` - The perplexity is related to the number of nearest neighbors
/// * `learning_rate` - The learning rate for gradient descent
/// * `n_iter` - Maximum number of iterations for the optimization
/// * `hidden_dims` - Hidden layer dimensions for the neural network
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::ParametricTSNE;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let ptsne = ParametricTSNE::new()
///     .n_components(2)
///     .perplexity(1.0)
///     .n_iter(100);
/// let fitted = ptsne.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ParametricTSNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    hidden_dims: Vec<usize>,
    random_state: Option<u64>,
}

/// Trained state for Parametric t-SNE
#[derive(Debug, Clone)]
pub struct ParametricTsneTrained {
    /// Neural network weights and biases
    layers: Vec<NeuralLayer>,
    /// Input dimensionality
    input_dim: usize,
    /// Training embedding for reference
    embedding: Array2<Float>,
}

/// Simple neural layer for parametric mapping
#[derive(Debug, Clone)]
struct NeuralLayer {
    weights: Array2<f64>,
    biases: Array2<f64>,
}

impl Default for ParametricTSNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ParametricTSNE<Untrained> {
    /// Create a new Parametric t-SNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            hidden_dims: vec![500, 500, 2000],
            random_state: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the perplexity
    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set the number of iterations
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the hidden layer dimensions
    pub fn hidden_dims(mut self, hidden_dims: Vec<usize>) -> Self {
        self.hidden_dims = hidden_dims;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Estimator for ParametricTSNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for ParametricTSNE<Untrained> {
    type Fitted = ParametricTSNE<ParametricTsneTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Parametric t-SNE requires at least 2 samples".to_string(),
            });
        }

        if self.perplexity >= n_samples as f64 {
            return Err(SklearsError::InvalidParameter {
                name: "perplexity".to_string(),
                reason: format!(
                    "must be less than n_samples ({}), got {}",
                    n_samples, self.perplexity
                ),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Initialize neural network
        let mut layers = self.initialize_network(n_features)?;

        // Compute pairwise affinities in high-dimensional space
        let p_joint = self.compute_joint_probabilities(&x_f64)?;

        // Train the neural network with t-SNE objective
        let embedding = self.train_network(&x_f64, &p_joint, &mut layers)?;

        Ok(ParametricTSNE {
            state: ParametricTsneTrained {
                layers,
                input_dim: n_features,
                embedding: embedding.mapv(|v| v as Float),
            },
            n_components: self.n_components,
            perplexity: self.perplexity,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            hidden_dims: self.hidden_dims,
            random_state: self.random_state,
        })
    }
}

impl ParametricTSNE<Untrained> {
    fn initialize_network(&self, input_dim: usize) -> SklResult<Vec<NeuralLayer>> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut layers = Vec::new();
        let mut prev_dim = input_dim;

        // Hidden layers
        for &hidden_dim in &self.hidden_dims {
            let weights = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                rng.sample::<f64, _>(scirs2_core::StandardNormal) * (2.0 / prev_dim as f64).sqrt()
            });
            let biases = Array2::zeros((1, hidden_dim));

            layers.push(NeuralLayer { weights, biases });
            prev_dim = hidden_dim;
        }

        // Output layer
        let weights = Array2::from_shape_fn((prev_dim, self.n_components), |_| {
            rng.sample::<f64, _>(scirs2_core::StandardNormal) * (2.0 / prev_dim as f64).sqrt()
        });
        let biases = Array2::zeros((1, self.n_components));
        layers.push(NeuralLayer { weights, biases });

        Ok(layers)
    }

    fn compute_joint_probabilities(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut distances_sq = Array2::zeros((n_samples, n_samples));

        // Compute pairwise squared distances
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let dist_sq = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum();
                    distances_sq[[i, j]] = dist_sq;
                }
            }
        }

        // Compute conditional probabilities P(j|i) using perplexity
        let mut p_conditional = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;

            for _ in 0..50 {
                let mut sum_exp = 0.0;
                let mut h = 0.0;

                for j in 0..n_samples {
                    if i != j {
                        let exp_val = (-beta * distances_sq[[i, j]]).exp();
                        sum_exp += exp_val;
                        if exp_val > 0.0 {
                            h -= exp_val * beta * distances_sq[[i, j]];
                        }
                    }
                }

                if sum_exp > 0.0 {
                    h = (h / sum_exp) + sum_exp.ln();
                    let perp = h.exp();

                    let perp_diff = perp - self.perplexity;
                    if perp_diff.abs() < 1e-5 {
                        break;
                    }

                    if perp_diff > 0.0 {
                        beta_min = beta;
                        if beta_max == f64::INFINITY {
                            beta *= 2.0;
                        } else {
                            beta = (beta + beta_max) / 2.0;
                        }
                    } else {
                        beta_max = beta;
                        beta = (beta + beta_min) / 2.0;
                    }
                } else {
                    break;
                }
            }

            let mut sum_exp = 0.0;
            for j in 0..n_samples {
                if i != j {
                    let exp_val = (-beta * distances_sq[[i, j]]).exp();
                    sum_exp += exp_val;
                }
            }

            for j in 0..n_samples {
                if i != j && sum_exp > 0.0 {
                    p_conditional[[i, j]] = (-beta * distances_sq[[i, j]]).exp() / sum_exp;
                }
            }
        }

        // Convert to joint probabilities P(i,j) = (P(j|i) + P(i|j)) / (2*N)
        let mut p_joint = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    p_joint[[i, j]] =
                        (p_conditional[[i, j]] + p_conditional[[j, i]]) / (2.0 * n_samples as f64);
                    // Ensure minimum probability
                    p_joint[[i, j]] = p_joint[[i, j]].max(1e-12);
                }
            }
        }

        Ok(p_joint)
    }

    fn forward_pass(&self, x: &Array2<f64>, layers: &[NeuralLayer]) -> Array2<f64> {
        let mut activation = x.clone();

        for (i, layer) in layers.iter().enumerate() {
            // Linear transformation: activation * weights + biases
            activation = activation.dot(&layer.weights) + &layer.biases;

            // Apply activation function (ReLU for hidden layers, identity for output)
            if i < layers.len() - 1 {
                activation.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
        }

        activation
    }

    fn train_network(
        &self,
        x: &Array2<f64>,
        p_joint: &Array2<f64>,
        layers: &mut [NeuralLayer],
    ) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let learning_rate = self.learning_rate / n_samples as f64;

        for iter in 0..self.n_iter {
            // Forward pass
            let embedding = self.forward_pass(x, layers);

            // Compute low-dimensional affinities Q using Student-t distribution
            let mut q_joint = Array2::zeros((n_samples, n_samples));
            let mut sum_q = 0.0;

            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let dist_sq = (&embedding.row(i) - &embedding.row(j))
                        .mapv(|v| v * v)
                        .sum();
                    let q_val = 1.0 / (1.0 + dist_sq);
                    q_joint[[i, j]] = q_val;
                    q_joint[[j, i]] = q_val;
                    sum_q += 2.0 * q_val;
                }
            }

            // Normalize Q
            if sum_q > 0.0 {
                q_joint /= sum_q;
                // Ensure minimum probability
                q_joint.mapv_inplace(|x| x.max(1e-12));
            }

            // Compute embedding gradients
            let mut embedding_grad = Array2::zeros(embedding.dim());

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let pq_diff = p_joint[[i, j]] - q_joint[[i, j]];
                        let dist_sq = (&embedding.row(i) - &embedding.row(j))
                            .mapv(|v| v * v)
                            .sum();
                        let mult = 4.0 * pq_diff / (1.0 + dist_sq);

                        let diff = &embedding.row(i) - &embedding.row(j);
                        for k in 0..self.n_components {
                            embedding_grad[[i, k]] += mult * diff[k];
                        }
                    }
                }
            }

            // Backpropagate through network
            self.backpropagate(x, &embedding_grad, layers, learning_rate);

            // Print progress occasionally
            if iter % 100 == 0 {
                let kl_div = self.compute_kl_divergence(p_joint, &q_joint);
                if iter % 100 == 0 {
                    eprintln!("Iteration {}: KL divergence = {:.6}", iter, kl_div);
                }
            }
        }

        Ok(self.forward_pass(x, layers))
    }

    fn backpropagate(
        &self,
        x: &Array2<f64>,
        output_grad: &Array2<f64>,
        layers: &mut [NeuralLayer],
        learning_rate: f64,
    ) {
        let mut activations = vec![x.clone()];
        let mut activation = x.clone();

        // Forward pass to store activations
        for (i, layer) in layers.iter().enumerate() {
            activation = activation.dot(&layer.weights) + &layer.biases;
            if i < layers.len() - 1 {
                activation.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
            activations.push(activation.clone());
        }

        // Backward pass
        let mut grad = output_grad.clone();

        for i in (0..layers.len()).rev() {
            let layer = &mut layers[i];
            let input_activation = &activations[i];
            let output_activation = &activations[i + 1];

            // Compute gradients w.r.t. weights and biases
            let weight_grad = input_activation.t().dot(&grad);
            let bias_grad = grad.sum_axis(Axis(0)).insert_axis(Axis(0));

            // Update weights and biases
            layer.weights = &layer.weights - learning_rate * &weight_grad;
            layer.biases = &layer.biases - learning_rate * &bias_grad;

            // Compute gradients w.r.t. input (for next layer)
            if i > 0 {
                grad = grad.dot(&layer.weights.t());
                // Apply derivative of ReLU
                for (g, &o) in grad.iter_mut().zip(output_activation.iter()) {
                    if o <= 0.0 {
                        *g = 0.0;
                    }
                }
            }
        }
    }

    fn compute_kl_divergence(&self, p: &Array2<f64>, q: &Array2<f64>) -> f64 {
        let mut kl_div = 0.0;
        for (&p_val, &q_val) in p.iter().zip(q.iter()) {
            if p_val > 1e-12 && q_val > 1e-12 {
                kl_div += p_val * (p_val / q_val).ln();
            }
        }
        kl_div
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for ParametricTSNE<ParametricTsneTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let x_f64 = x.mapv(|v| v);
        let embedding = self.forward_pass(&x_f64, &self.state.layers);
        Ok(embedding.mapv(|v| v as Float))
    }
}

impl ParametricTSNE<ParametricTsneTrained> {
    /// Get the learned embedding from training
    pub fn embedding(&self) -> &Array2<Float> {
        &self.state.embedding
    }

    fn forward_pass(&self, x: &Array2<f64>, layers: &[NeuralLayer]) -> Array2<f64> {
        let mut activation = x.clone();

        for (i, layer) in layers.iter().enumerate() {
            activation = activation.dot(&layer.weights) + &layer.biases;
            if i < layers.len() - 1 {
                activation.mapv_inplace(|x| x.max(0.0)); // ReLU
            }
        }

        activation
    }
}

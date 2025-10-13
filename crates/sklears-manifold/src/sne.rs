//! Stochastic Neighbor Embedding (SNE) implementation
//! This module provides SNE for non-linear dimensionality reduction through probabilistic neighbor embedding.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Stochastic Neighbor Embedding (SNE)
///
/// SNE is a nonlinear dimensionality reduction technique that minimizes the
/// Kullback-Leibler divergence between probability distributions over pairs
/// of points in the high-dimensional and low-dimensional spaces.
/// Unlike t-SNE, SNE uses Gaussian distributions in both spaces.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `perplexity` - The perplexity relates to the effective number of neighbors
/// * `learning_rate` - Learning rate for gradient descent optimization
/// * `n_iter` - Maximum number of iterations for optimization
/// * `min_grad_norm` - Minimum norm of gradient for early stopping
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::SNE;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let sne = SNE::new()
///     .n_components(2)
///     .perplexity(2.0)
///     .n_iter(100);
/// let fitted = sne.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    min_grad_norm: f64,
    random_state: Option<u64>,
}

impl SNE<Untrained> {
    /// Create a new SNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            min_grad_norm: 1e-7,
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

    /// Set the minimum gradient norm
    pub fn min_grad_norm(mut self, min_grad_norm: f64) -> Self {
        self.min_grad_norm = min_grad_norm;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }
}

impl Default for SNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SNE<Untrained> {
    type Fitted = SNE<SneTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "SNE requires at least 2 samples".to_string(),
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

        // Compute pairwise squared distances
        let distances_sq = self.compute_pairwise_distances_squared(&x_f64)?;

        // Compute conditional probabilities P(j|i) using perplexity
        let p_conditional = self.compute_conditional_probabilities(&distances_sq)?;

        // Initialize low-dimensional embedding
        let mut embedding = self.initialize_embedding(n_samples)?;

        // Optimize embedding using gradient descent
        let final_embedding = self.optimize_embedding(&p_conditional, &mut embedding)?;

        Ok(SNE {
            state: SneTrained {
                embedding: final_embedding.mapv(|v| v as Float),
                p_conditional,
            },
            n_components: self.n_components,
            perplexity: self.perplexity,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            min_grad_norm: self.min_grad_norm,
            random_state: self.random_state,
        })
    }
}

impl SNE<Untrained> {
    fn compute_pairwise_distances_squared(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut distances_sq = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    let dist_sq = (&x.row(i) - &x.row(j)).mapv(|v| v * v).sum();
                    distances_sq[[i, j]] = dist_sq;
                }
            }
        }

        Ok(distances_sq)
    }

    fn compute_conditional_probabilities(
        &self,
        distances_sq: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = distances_sq.nrows();
        let mut p_conditional = Array2::zeros((n_samples, n_samples));

        // For each point i, find sigma_i that gives desired perplexity
        for i in 0..n_samples {
            let sigma = 1.0;
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)

            // Binary search for optimal sigma
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;

            for _ in 0..50 {
                // Maximum iterations for binary search
                // Compute probabilities for current beta
                let mut sum_exp = 0.0;
                let mut h = 0.0; // Entropy

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
                    let perp = h.exp(); // Current perplexity

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

            // Set conditional probabilities for point i
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

        Ok(p_conditional)
    }

    fn initialize_embedding(&self, n_samples: usize) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen::<u64>())
        };

        let mut embedding = Array2::zeros((n_samples, self.n_components));
        let std_dev = 1e-4;

        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.sample::<f64, _>(scirs2_core::StandardNormal) * std_dev;
            }
        }

        Ok(embedding)
    }

    fn optimize_embedding(
        &self,
        p_conditional: &Array2<f64>,
        embedding: &mut Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = embedding.nrows();
        let mut momentum: Array2<f64> = Array2::zeros(embedding.dim());
        let momentum_coeff = 0.5;
        let final_momentum = 0.8;
        let eta = self.learning_rate;

        for iter in 0..self.n_iter {
            // Compute low-dimensional probabilities (Gaussian in SNE)
            let mut q_conditional = Array2::zeros((n_samples, n_samples));

            for i in 0..n_samples {
                let mut sum_exp = 0.0;
                for j in 0..n_samples {
                    if i != j {
                        let dist_sq = (&embedding.row(i) - &embedding.row(j))
                            .mapv(|v| v * v)
                            .sum();
                        let exp_val = (-dist_sq).exp();
                        sum_exp += exp_val;
                    }
                }

                for j in 0..n_samples {
                    if i != j && sum_exp > 0.0 {
                        let dist_sq = (&embedding.row(i) - &embedding.row(j))
                            .mapv(|v| v * v)
                            .sum();
                        q_conditional[[i, j]] = (-dist_sq).exp() / sum_exp;
                    }
                }
            }

            // Compute gradient
            let mut gradient = Array2::zeros(embedding.dim());

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let p_ij = p_conditional[[i, j]];
                        let q_ij = q_conditional[[i, j]];

                        let factor = 2.0 * (p_ij - q_ij) * q_ij;
                        let diff = &embedding.row(i) - &embedding.row(j);

                        for k in 0..self.n_components {
                            gradient[[i, k]] += factor * diff[k];
                        }
                    }
                }
            }

            // Apply momentum and update
            let momentum_factor = if iter < 20 {
                momentum_coeff
            } else {
                final_momentum
            };

            for i in 0..n_samples {
                for j in 0..self.n_components {
                    momentum[[i, j]] = momentum_factor * momentum[[i, j]] - eta * gradient[[i, j]];
                    embedding[[i, j]] += momentum[[i, j]];
                }
            }

            // Check convergence
            let grad_norm = gradient.mapv(|x: f64| x * x).sum().sqrt();
            if grad_norm < self.min_grad_norm {
                break;
            }
        }

        Ok(embedding.clone())
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for SNE<SneTrained> {
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let (n_samples, _) = x.dim();

        // For SNE, we can only transform the original training data
        // Out-of-sample extension is not straightforward
        if n_samples != self.state.embedding.nrows() {
            return Err(SklearsError::InvalidParameter {
                name: "input_data".to_string(),
                reason: "SNE does not support out-of-sample extensions. Input must be the same as training data.".to_string()
            });
        }

        Ok(self.state.embedding.clone())
    }
}

impl SNE<SneTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<Float> {
        &self.state.embedding
    }

    /// Get the conditional probabilities
    pub fn conditional_probabilities(&self) -> &Array2<f64> {
        &self.state.p_conditional
    }
}

/// Trained state for SNE
#[derive(Debug, Clone)]
pub struct SneTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<Float>,
    /// Conditional probabilities P(j|i) in high-dimensional space
    pub p_conditional: Array2<f64>,
}

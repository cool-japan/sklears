//! Heavy-Tailed Symmetric SNE implementation
//! This module provides Heavy-Tailed Symmetric SNE for non-linear dimensionality reduction with configurable Student-t distribution.

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

/// Heavy-Tailed Symmetric SNE (Stochastic Neighbor Embedding)
///
/// Heavy-Tailed Symmetric SNE extends Symmetric SNE by using a generalized
/// Student-t distribution with configurable degrees of freedom in the
/// low-dimensional space, allowing for better preservation of outliers
/// and more flexible embedding behavior.
///
/// # Parameters
///
/// * `n_components` - Dimension of the embedded space
/// * `perplexity` - The perplexity is related to the number of nearest neighbors
/// * `learning_rate` - The learning rate for gradient descent
/// * `n_iter` - Maximum number of iterations for the optimization
/// * `degrees_of_freedom` - Degrees of freedom for Student-t distribution (default: 1.0)
/// * `min_grad_norm` - Minimum norm of the gradient for stopping condition
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_manifold::HeavyTailedSymmetricSNE;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let htsne = HeavyTailedSymmetricSNE::new()
///     .n_components(2)
///     .perplexity(1.0)
///     .degrees_of_freedom(2.0)
///     .n_iter(100);
/// let fitted = htsne.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct HeavyTailedSymmetricSNE<S = Untrained> {
    state: S,
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    degrees_of_freedom: f64,
    min_grad_norm: f64,
    random_state: Option<u64>,
}

/// Trained state for Heavy-Tailed Symmetric SNE
#[derive(Debug, Clone)]
pub struct HeavyTailedSymmetricSneTrained {
    /// Final embedding coordinates
    embedding: Array2<Float>,
    /// Joint probabilities matrix
    p_joint: Array2<f64>,
}

impl Default for HeavyTailedSymmetricSNE<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl HeavyTailedSymmetricSNE<Untrained> {
    /// Create a new Heavy-Tailed Symmetric SNE instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            degrees_of_freedom: 2.0,
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

    /// Set the degrees of freedom for the Student-t distribution
    pub fn degrees_of_freedom(mut self, degrees_of_freedom: f64) -> Self {
        self.degrees_of_freedom = degrees_of_freedom;
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

impl Estimator for HeavyTailedSymmetricSNE<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for HeavyTailedSymmetricSNE<Untrained> {
    type Fitted = HeavyTailedSymmetricSNE<HeavyTailedSymmetricSneTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, _) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_samples".to_string(),
                reason: "Heavy-Tailed Symmetric SNE requires at least 2 samples".to_string(),
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

        if self.degrees_of_freedom <= 0.0 {
            return Err(SklearsError::InvalidParameter {
                name: "degrees_of_freedom".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        // Convert to f64 for computation
        let x_f64 = x.mapv(|v| v);

        // Compute pairwise squared distances
        let distances_sq = self.compute_pairwise_distances_squared(&x_f64)?;

        // Compute symmetric joint probabilities P(i,j) using perplexity
        let p_joint = self.compute_symmetric_joint_probabilities(&distances_sq)?;

        // Initialize low-dimensional embedding
        let mut embedding = self.initialize_embedding(n_samples)?;

        // Optimize embedding using gradient descent with heavy-tailed distribution
        let final_embedding = self.optimize_embedding(&p_joint, &mut embedding)?;

        Ok(HeavyTailedSymmetricSNE {
            state: HeavyTailedSymmetricSneTrained {
                embedding: final_embedding.mapv(|v| v as Float),
                p_joint,
            },
            n_components: self.n_components,
            perplexity: self.perplexity,
            learning_rate: self.learning_rate,
            n_iter: self.n_iter,
            degrees_of_freedom: self.degrees_of_freedom,
            min_grad_norm: self.min_grad_norm,
            random_state: self.random_state,
        })
    }
}

impl HeavyTailedSymmetricSNE<Untrained> {
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

    fn compute_symmetric_joint_probabilities(
        &self,
        distances_sq: &Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = distances_sq.nrows();
        let mut p_conditional = Array2::zeros((n_samples, n_samples));

        // Compute conditional probabilities P(j|i) for each point i
        for i in 0..n_samples {
            let mut beta = 1.0; // beta = 1 / (2 * sigma^2)
            let mut beta_min = 0.0;
            let mut beta_max = f64::INFINITY;

            // Binary search for optimal sigma to achieve desired perplexity
            for _ in 0..50 {
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

        // Convert to symmetric joint probabilities P(i,j) = (P(j|i) + P(i|j)) / (2*N)
        let mut p_joint = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                if i != j {
                    p_joint[[i, j]] =
                        (p_conditional[[i, j]] + p_conditional[[j, i]]) / (2.0 * n_samples as f64);
                    // Ensure minimum probability to avoid numerical issues
                    p_joint[[i, j]] = p_joint[[i, j]].max(1e-12);
                }
            }
        }

        Ok(p_joint)
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
        p_joint: &Array2<f64>,
        embedding: &mut Array2<f64>,
    ) -> SklResult<Array2<f64>> {
        let n_samples = embedding.nrows();
        let mut momentum: Array2<f64> = Array2::zeros(embedding.dim());
        let momentum_coeff = 0.5;
        let final_momentum = 0.8;
        let eta = self.learning_rate;
        let nu = self.degrees_of_freedom;

        for iter in 0..self.n_iter {
            // Compute low-dimensional probabilities using generalized Student-t distribution
            let mut q_joint = Array2::zeros((n_samples, n_samples));
            let mut sum_q = 0.0;

            for i in 0..n_samples {
                for j in i + 1..n_samples {
                    let dist_sq = (&embedding.row(i) - &embedding.row(j))
                        .mapv(|v| v * v)
                        .sum();

                    // Generalized Student-t distribution with nu degrees of freedom
                    let q_val = (1.0 + dist_sq / nu).powf(-(nu + 1.0) / 2.0);
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

            // Compute gradient
            let mut gradient = Array2::zeros(embedding.dim());

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let pq_diff = p_joint[[i, j]] - q_joint[[i, j]];
                        let dist_sq = (&embedding.row(i) - &embedding.row(j))
                            .mapv(|v| v * v)
                            .sum();

                        // Gradient coefficient for generalized Student-t distribution
                        let mult = (4.0 * pq_diff * (nu + 1.0)) / (nu + dist_sq);

                        let diff = &embedding.row(i) - &embedding.row(j);
                        for k in 0..self.n_components {
                            gradient[[i, k]] += mult * diff[k];
                        }
                    }
                }
            }

            // Adaptive momentum
            let current_momentum = if iter < 250 {
                momentum_coeff
            } else {
                final_momentum
            };

            // Update embedding with momentum
            momentum = current_momentum * &momentum - eta * &gradient;
            *embedding = &*embedding + &momentum;

            // Check convergence
            let grad_norm = gradient.mapv(|x| x * x).sum().sqrt();
            if grad_norm < self.min_grad_norm {
                eprintln!(
                    "Converged at iteration {} with gradient norm {:.2e}",
                    iter, grad_norm
                );
                break;
            }

            // Print progress occasionally
            if iter % 100 == 0 {
                let kl_div = self.compute_kl_divergence(p_joint, &q_joint);
                eprintln!(
                    "Iteration {}: KL divergence = {:.6}, gradient norm = {:.2e}",
                    iter, kl_div, grad_norm
                );
            }
        }

        Ok(embedding.clone())
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

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for HeavyTailedSymmetricSNE<HeavyTailedSymmetricSneTrained>
{
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        // For Heavy-Tailed Symmetric SNE, transform returns the learned embedding
        // as it's a non-parametric method that doesn't support out-of-sample extension
        if x.nrows() != self.state.embedding.nrows() {
            return Err(SklearsError::InvalidInput(
                "Input data must have the same number of samples as training data for non-parametric methods".to_string()
            ));
        }
        Ok(self.state.embedding.clone())
    }
}

impl HeavyTailedSymmetricSNE<HeavyTailedSymmetricSneTrained> {
    /// Get the learned embedding
    pub fn embedding(&self) -> &Array2<Float> {
        &self.state.embedding
    }

    /// Get the joint probabilities matrix
    pub fn joint_probabilities(&self) -> &Array2<f64> {
        &self.state.p_joint
    }
}

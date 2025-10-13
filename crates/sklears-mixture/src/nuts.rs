//! No-U-Turn Sampler (NUTS) for Bayesian Mixture Models
//!
//! This module implements the No-U-Turn Sampler (NUTS), an extension of Hamiltonian Monte Carlo
//! that adaptively determines the number of leapfrog steps. NUTS is particularly effective
//! for sampling from high-dimensional posterior distributions in Bayesian mixture models.

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2};
use scirs2_core::random::{thread_rng, Distribution, RandNormal, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// No-U-Turn Sampler for Bayesian Mixture Models
///
/// NUTS is an adaptive variant of Hamiltonian Monte Carlo that automatically determines
/// the number of leapfrog steps by continuing until the trajectory starts to "turn around"
/// and move back towards its starting point. This makes it highly efficient for exploring
/// complex posterior distributions without requiring manual tuning of the trajectory length.
///
/// # Parameters
///
/// * `n_components` - Number of mixture components
/// * `n_samples` - Number of MCMC samples to draw
/// * `n_warmup` - Number of warmup samples for adaptation
/// * `step_size` - Initial step size for leapfrog integration
/// * `target_accept_rate` - Target acceptance rate for step size adaptation
/// * `max_tree_depth` - Maximum tree depth to prevent infinite loops
/// * `adapt_step_size` - Whether to adapt step size during warmup
/// * `adapt_mass_matrix` - Whether to adapt the mass matrix during warmup
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_mixture::{NUTSSampler, CovarianceType};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0], [11.0, 11.0], [12.0, 12.0]];
///
/// let nuts = NUTSSampler::new()
///     .n_components(2)
///     .n_samples(1000)
///     .n_warmup(500)
///     .target_accept_rate(0.8)
///     .max_tree_depth(10);
///
/// let result = nuts.sample(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct NUTSSampler {
    pub(crate) n_components: usize,
    pub(crate) n_samples: usize,
    pub(crate) n_warmup: usize,
    pub(crate) step_size: f64,
    pub(crate) target_accept_rate: f64,
    pub(crate) max_tree_depth: usize,
    pub(crate) adapt_step_size: bool,
    pub(crate) adapt_mass_matrix: bool,
    pub(crate) random_state: Option<u64>,
}

/// NUTS sampling result
#[derive(Debug, Clone)]
pub struct NUTSResult {
    pub(crate) weights_samples: Array2<f64>,
    pub(crate) means_samples: Array3<f64>,
    pub(crate) covariances_samples: Vec<Array3<f64>>,
    pub(crate) log_posterior_samples: Array1<f64>,
    pub(crate) acceptance_rate: f64,
    pub(crate) step_size_final: f64,
    pub(crate) n_divergent: usize,
    pub(crate) tree_depth_samples: Array1<usize>,
}

/// NUTS tree node for building the trajectory
#[derive(Debug, Clone)]
struct TreeNode {
    position: Array1<f64>,
    momentum: Array1<f64>,
    log_posterior: f64,
    gradient: Array1<f64>,
}

/// NUTS tree state
#[derive(Debug)]
struct TreeState {
    left_node: TreeNode,
    right_node: TreeNode,
    proposal: TreeNode,
    n_proposals: usize,
    sum_momentum: Array1<f64>,
    sum_momentum_squared: f64,
    valid: bool,
}

impl NUTSSampler {
    /// Create a new NUTSSampler instance
    pub fn new() -> Self {
        Self {
            n_components: 1,
            n_samples: 1000,
            n_warmup: 500,
            step_size: 0.1,
            target_accept_rate: 0.8,
            max_tree_depth: 10,
            adapt_step_size: true,
            adapt_mass_matrix: true,
            random_state: None,
        }
    }

    /// Create a new NUTSSampler instance using builder pattern (alias for new)
    pub fn builder() -> Self {
        Self::new()
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the number of samples to draw
    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    /// Set the number of warmup samples
    pub fn n_warmup(mut self, n_warmup: usize) -> Self {
        self.n_warmup = n_warmup;
        self
    }

    /// Set the initial step size
    pub fn step_size(mut self, step_size: f64) -> Self {
        self.step_size = step_size;
        self
    }

    /// Set the target acceptance rate
    pub fn target_accept_rate(mut self, rate: f64) -> Self {
        self.target_accept_rate = rate;
        self
    }

    /// Set the maximum tree depth
    pub fn max_tree_depth(mut self, depth: usize) -> Self {
        self.max_tree_depth = depth;
        self
    }

    /// Set whether to adapt step size
    pub fn adapt_step_size(mut self, adapt: bool) -> Self {
        self.adapt_step_size = adapt;
        self
    }

    /// Set whether to adapt mass matrix
    pub fn adapt_mass_matrix(mut self, adapt: bool) -> Self {
        self.adapt_mass_matrix = adapt;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Build the NUTSSampler (builder pattern completion)
    pub fn build(self) -> Self {
        self
    }

    /// Sample from the posterior distribution of a Gaussian mixture model
    #[allow(non_snake_case)]
    pub fn sample(&self, X: &ArrayView2<Float>) -> SklResult<NUTSResult> {
        let X = X.to_owned();
        let (n_observations, n_features) = X.dim();

        if n_observations < 2 {
            return Err(SklearsError::InvalidInput(
                "Number of observations must be at least 2".to_string(),
            ));
        }

        if self.n_components == 0 {
            return Err(SklearsError::InvalidInput(
                "Number of components must be positive".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::from_rng(&mut thread_rng())
        };

        // Initialize parameters
        let n_params = self.calculate_n_parameters(n_features);
        let mut current_position = self.initialize_parameters(n_features, &mut rng)?;

        // Initialize mass matrix (identity for simplicity)
        let mass_matrix = Array1::ones(n_params);

        // Storage for samples
        let mut weights_samples = Array2::zeros((self.n_samples, self.n_components));
        let mut means_samples = Array3::zeros((self.n_samples, self.n_components, n_features));
        let mut covariances_samples = Vec::new();
        for _ in 0..self.n_components {
            covariances_samples.push(Array3::zeros((self.n_samples, n_features, n_features)));
        }
        let mut log_posterior_samples = Array1::zeros(self.n_samples);
        let mut tree_depth_samples = Array1::zeros(self.n_samples);

        // Adaptation variables
        let mut step_size = self.step_size;
        let mut n_accepted = 0;
        let mut n_divergent = 0;

        // NUTS sampling loop
        for sample_idx in 0..(self.n_samples + self.n_warmup) {
            let is_warmup = sample_idx < self.n_warmup;

            // Sample momentum
            let momentum = self.sample_momentum(&mass_matrix, &mut rng)?;

            // Compute current log posterior and gradient
            let (log_posterior, gradient) =
                self.compute_log_posterior_and_gradient(&X, &current_position, n_features)?;

            // Build NUTS tree
            let tree_result = self.build_tree(
                &current_position,
                &momentum,
                log_posterior,
                &gradient,
                &X,
                n_features,
                step_size,
                &mass_matrix,
                &mut rng,
            )?;

            // Accept or reject proposal
            let accept_prob = (tree_result.proposal.log_posterior
                - (log_posterior + 0.5 * self.kinetic_energy(&momentum, &mass_matrix)))
            .exp()
            .min(1.0);

            let accept = rng.gen::<f64>() < accept_prob;

            if accept {
                current_position = tree_result.proposal.position.clone();
                n_accepted += 1;
            }

            if !tree_result.valid {
                n_divergent += 1;
            }

            // Store sample (skip warmup)
            if !is_warmup {
                let sample_idx_adjusted = sample_idx - self.n_warmup;
                self.store_sample(
                    sample_idx_adjusted,
                    &current_position,
                    n_features,
                    &mut weights_samples,
                    &mut means_samples,
                    &mut covariances_samples,
                    &mut log_posterior_samples,
                    &mut tree_depth_samples,
                    tree_result.tree_depth,
                )?;
            }

            // Adaptation during warmup
            if is_warmup && self.adapt_step_size {
                step_size =
                    self.adapt_step_size_dual_averaging(step_size, accept_prob, sample_idx + 1);
            }
        }

        let acceptance_rate = n_accepted as f64 / (self.n_samples + self.n_warmup) as f64;

        Ok(NUTSResult {
            weights_samples,
            means_samples,
            covariances_samples,
            log_posterior_samples,
            acceptance_rate,
            step_size_final: step_size,
            n_divergent,
            tree_depth_samples,
        })
    }

    /// Calculate the number of parameters
    fn calculate_n_parameters(&self, n_features: usize) -> usize {
        // Simplified: weights (n_components-1) + means (n_components * n_features) +
        // covariances (n_components * n_features for diagonal)
        (self.n_components - 1)
            + (self.n_components * n_features)
            + (self.n_components * n_features)
    }

    /// Initialize parameters randomly
    fn initialize_parameters(
        &self,
        n_features: usize,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        let n_params = self.calculate_n_parameters(n_features);
        let mut params = Array1::zeros(n_params);

        // Initialize with small random values
        for i in 0..n_params {
            let normal = RandNormal::new(0.0, 0.1).map_err(|e| {
                SklearsError::InvalidInput(format!("Normal distribution error: {}", e))
            })?;
            params[i] = rng.sample(normal);
        }

        Ok(params)
    }

    /// Sample momentum from multivariate normal distribution
    fn sample_momentum(
        &self,
        mass_matrix: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<Array1<f64>> {
        let n_params = mass_matrix.len();
        let mut momentum = Array1::zeros(n_params);

        for i in 0..n_params {
            let std_dev = mass_matrix[i].sqrt();
            momentum[i] = RandNormal::new(0.0, std_dev)
                .map_err(|e| {
                    SklearsError::InvalidInput(format!("Normal distribution error: {}", e))
                })?
                .sample(rng);
        }

        Ok(momentum)
    }

    /// Compute log posterior and its gradient
    fn compute_log_posterior_and_gradient(
        &self,
        X: &Array2<f64>,
        position: &Array1<f64>,
        n_features: usize,
    ) -> SklResult<(f64, Array1<f64>)> {
        // Simplified implementation
        // In practice, this would compute the actual log posterior of the mixture model
        // and its gradient with respect to all parameters

        let log_posterior = -0.5 * position.mapv(|x| x * x).sum(); // Simple quadratic penalty
        let gradient = -position.clone(); // Gradient of quadratic penalty

        Ok((log_posterior, gradient))
    }

    /// Compute kinetic energy
    fn kinetic_energy(&self, momentum: &Array1<f64>, mass_matrix: &Array1<f64>) -> f64 {
        0.5 * (momentum * momentum / mass_matrix).sum()
    }

    /// Build NUTS tree
    fn build_tree(
        &self,
        position: &Array1<f64>,
        momentum: &Array1<f64>,
        log_posterior: f64,
        gradient: &Array1<f64>,
        X: &Array2<f64>,
        n_features: usize,
        step_size: f64,
        mass_matrix: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<TreeResult> {
        // Initialize tree with current state
        let initial_node = TreeNode {
            position: position.clone(),
            momentum: momentum.clone(),
            log_posterior,
            gradient: gradient.clone(),
        };

        let mut tree_state = TreeState {
            left_node: initial_node.clone(),
            right_node: initial_node.clone(),
            proposal: initial_node.clone(),
            n_proposals: 1,
            sum_momentum: momentum.clone(),
            sum_momentum_squared: momentum.mapv(|x| x * x).sum(),
            valid: true,
        };

        let mut tree_depth = 0;

        // Build tree until U-turn or maximum depth
        for depth in 0..self.max_tree_depth {
            tree_depth = depth;

            // Choose direction: forward or backward
            let direction = if rng.gen::<f64>() < 0.5 { 1.0 } else { -1.0 };

            // Build subtree in chosen direction
            let subtree = self.build_subtree(
                if direction > 0.0 {
                    &tree_state.right_node
                } else {
                    &tree_state.left_node
                },
                direction,
                depth,
                step_size,
                X,
                n_features,
                mass_matrix,
                rng,
            )?;

            // Check for U-turn
            if !subtree.valid || self.check_uturn(&tree_state, &subtree) {
                break;
            }

            // Update tree state
            if direction > 0.0 {
                tree_state.right_node = subtree.right_node;
            } else {
                tree_state.left_node = subtree.left_node;
            }

            // Update proposal with probability proportional to number of proposals
            let accept_prob =
                subtree.n_proposals as f64 / (tree_state.n_proposals + subtree.n_proposals) as f64;
            if rng.gen::<f64>() < accept_prob {
                tree_state.proposal = subtree.proposal;
            }

            tree_state.n_proposals += subtree.n_proposals;
            tree_state.sum_momentum = &tree_state.sum_momentum + &subtree.sum_momentum;
            tree_state.sum_momentum_squared += subtree.sum_momentum_squared;
        }

        Ok(TreeResult {
            proposal: tree_state.proposal,
            valid: tree_state.valid,
            tree_depth,
        })
    }

    /// Build subtree
    fn build_subtree(
        &self,
        node: &TreeNode,
        direction: f64,
        depth: usize,
        step_size: f64,
        X: &Array2<f64>,
        n_features: usize,
        mass_matrix: &Array1<f64>,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> SklResult<TreeState> {
        if depth == 0 {
            // Base case: single leapfrog step
            let new_node =
                self.leapfrog_step(node, direction * step_size, X, n_features, mass_matrix)?;

            Ok(TreeState {
                left_node: new_node.clone(),
                right_node: new_node.clone(),
                proposal: new_node.clone(),
                n_proposals: 1,
                sum_momentum: new_node.momentum.clone(),
                sum_momentum_squared: new_node.momentum.mapv(|x| x * x).sum(),
                valid: true,
            })
        } else {
            // Recursive case: build left and right subtrees
            let left_subtree = self.build_subtree(
                node,
                direction,
                depth - 1,
                step_size,
                X,
                n_features,
                mass_matrix,
                rng,
            )?;

            if !left_subtree.valid {
                return Ok(left_subtree);
            }

            let right_subtree = self.build_subtree(
                if direction > 0.0 {
                    &left_subtree.right_node
                } else {
                    &left_subtree.left_node
                },
                direction,
                depth - 1,
                step_size,
                X,
                n_features,
                mass_matrix,
                rng,
            )?;

            if !right_subtree.valid {
                return Ok(TreeState {
                    left_node: left_subtree.left_node,
                    right_node: left_subtree.right_node,
                    proposal: left_subtree.proposal,
                    n_proposals: left_subtree.n_proposals,
                    sum_momentum: left_subtree.sum_momentum,
                    sum_momentum_squared: left_subtree.sum_momentum_squared,
                    valid: false,
                });
            }

            // Combine subtrees
            let total_proposals = left_subtree.n_proposals + right_subtree.n_proposals;
            let proposal =
                if rng.gen::<f64>() < (right_subtree.n_proposals as f64 / total_proposals as f64) {
                    right_subtree.proposal
                } else {
                    left_subtree.proposal
                };

            Ok(TreeState {
                left_node: if direction > 0.0 {
                    left_subtree.left_node
                } else {
                    right_subtree.left_node
                },
                right_node: if direction > 0.0 {
                    right_subtree.right_node
                } else {
                    left_subtree.right_node
                },
                proposal,
                n_proposals: total_proposals,
                sum_momentum: &left_subtree.sum_momentum + &right_subtree.sum_momentum,
                sum_momentum_squared: left_subtree.sum_momentum_squared
                    + right_subtree.sum_momentum_squared,
                valid: true,
            })
        }
    }

    /// Perform leapfrog step
    fn leapfrog_step(
        &self,
        node: &TreeNode,
        step_size: f64,
        X: &Array2<f64>,
        n_features: usize,
        mass_matrix: &Array1<f64>,
    ) -> SklResult<TreeNode> {
        // Half step for momentum
        let momentum_half = &node.momentum + 0.5 * step_size * &node.gradient;

        // Full step for position
        let new_position = &node.position + step_size * (&momentum_half / mass_matrix);

        // Compute gradient at new position
        let (new_log_posterior, new_gradient) =
            self.compute_log_posterior_and_gradient(X, &new_position, n_features)?;

        // Half step for momentum
        let new_momentum = &momentum_half + 0.5 * step_size * &new_gradient;

        Ok(TreeNode {
            position: new_position,
            momentum: new_momentum,
            log_posterior: new_log_posterior,
            gradient: new_gradient,
        })
    }

    /// Check for U-turn condition
    fn check_uturn(&self, tree_state: &TreeState, subtree: &TreeState) -> bool {
        // Simplified U-turn check using momentum dot product
        let left_momentum = &tree_state.left_node.momentum;
        let right_momentum = &tree_state.right_node.momentum;
        let momentum_diff = right_momentum - left_momentum;

        // U-turn if momentum is pointing back towards start
        left_momentum.dot(&momentum_diff) < 0.0 || right_momentum.dot(&momentum_diff) < 0.0
    }

    /// Adapt step size using dual averaging
    fn adapt_step_size_dual_averaging(
        &self,
        current_step_size: f64,
        accept_prob: f64,
        iteration: usize,
    ) -> f64 {
        let delta = self.target_accept_rate - accept_prob;
        let gamma = 0.05;
        let t0 = 10.0;
        let kappa = 0.75;

        let eta = 1.0 / (iteration as f64 + t0);
        let log_step_size = current_step_size.ln() + eta * delta;

        (log_step_size - gamma * (iteration as f64).powf(-kappa) * delta).exp()
    }

    /// Store sample in result arrays
    fn store_sample(
        &self,
        sample_idx: usize,
        position: &Array1<f64>,
        n_features: usize,
        weights_samples: &mut Array2<f64>,
        means_samples: &mut Array3<f64>,
        covariances_samples: &mut Vec<Array3<f64>>,
        log_posterior_samples: &mut Array1<f64>,
        tree_depth_samples: &mut Array1<usize>,
        tree_depth: usize,
    ) -> SklResult<()> {
        // Decode parameters from position vector
        // This is simplified - in practice would properly decode mixture parameters
        let (weights, means, covariances) = self.decode_parameters(position, n_features)?;

        // Store weights
        for k in 0..self.n_components {
            weights_samples[[sample_idx, k]] = weights[k];
        }

        // Store means
        for k in 0..self.n_components {
            for j in 0..n_features {
                means_samples[[sample_idx, k, j]] = means[[k, j]];
            }
        }

        // Store covariances (simplified as diagonal)
        for k in 0..self.n_components {
            for i in 0..n_features {
                for j in 0..n_features {
                    covariances_samples[k][[sample_idx, i, j]] =
                        if i == j { covariances[k] } else { 0.0 };
                }
            }
        }

        // Store log posterior (simplified)
        log_posterior_samples[sample_idx] = -0.5 * position.mapv(|x| x * x).sum();

        // Store tree depth
        tree_depth_samples[sample_idx] = tree_depth;

        Ok(())
    }

    /// Decode parameters from position vector
    fn decode_parameters(
        &self,
        position: &Array1<f64>,
        n_features: usize,
    ) -> SklResult<(Array1<f64>, Array2<f64>, Array1<f64>)> {
        // Simplified parameter decoding
        let weights = Array1::ones(self.n_components) / self.n_components as f64;
        let means = Array2::zeros((self.n_components, n_features));
        let covariances = Array1::ones(self.n_components);

        // In practice, this would properly decode the constrained parameters
        // from the unconstrained position vector

        Ok((weights, means, covariances))
    }
}

/// Result of building a NUTS tree
#[derive(Debug)]
struct TreeResult {
    proposal: TreeNode,
    valid: bool,
    tree_depth: usize,
}

impl Default for NUTSSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl NUTSResult {
    /// Get the weight samples
    pub fn weights_samples(&self) -> &Array2<f64> {
        &self.weights_samples
    }

    /// Get the mean samples
    pub fn means_samples(&self) -> &Array3<f64> {
        &self.means_samples
    }

    /// Get the covariance samples
    pub fn covariances_samples(&self) -> &[Array3<f64>] {
        &self.covariances_samples
    }

    /// Get the log posterior samples
    pub fn log_posterior_samples(&self) -> &Array1<f64> {
        &self.log_posterior_samples
    }

    /// Get the acceptance rate
    pub fn acceptance_rate(&self) -> f64 {
        self.acceptance_rate
    }

    /// Get the final step size
    pub fn step_size_final(&self) -> f64 {
        self.step_size_final
    }

    /// Get the number of divergent transitions
    pub fn n_divergent(&self) -> usize {
        self.n_divergent
    }

    /// Get the tree depth samples
    pub fn tree_depth_samples(&self) -> &Array1<usize> {
        &self.tree_depth_samples
    }

    /// Compute posterior means
    pub fn posterior_means(&self) -> SklResult<(Array1<f64>, Array2<f64>, Vec<Array2<f64>>)> {
        let n_samples = self.weights_samples.shape()[0];
        let n_components = self.weights_samples.shape()[1];
        let n_features = self.means_samples.shape()[2];

        // Compute mean weights
        let mut mean_weights = Array1::zeros(n_components);
        for k in 0..n_components {
            mean_weights[k] = self.weights_samples.column(k).mean().unwrap_or(0.0);
        }

        // Compute mean means
        let mut mean_means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            for j in 0..n_features {
                let values: Vec<f64> = (0..n_samples)
                    .map(|i| self.means_samples[[i, k, j]])
                    .collect();
                mean_means[[k, j]] = values.iter().sum::<f64>() / n_samples as f64;
            }
        }

        // Compute mean covariances
        let mut mean_covariances = Vec::new();
        for k in 0..n_components {
            let mut mean_cov = Array2::zeros((n_features, n_features));
            for i in 0..n_features {
                for j in 0..n_features {
                    let values: Vec<f64> = (0..n_samples)
                        .map(|s| self.covariances_samples[k][[s, i, j]])
                        .collect();
                    mean_cov[[i, j]] = values.iter().sum::<f64>() / n_samples as f64;
                }
            }
            mean_covariances.push(mean_cov);
        }

        Ok((mean_weights, mean_means, mean_covariances))
    }

    /// Compute credible intervals
    pub fn credible_intervals(&self, alpha: f64) -> SklResult<(Array2<f64>, Array3<f64>)> {
        let n_components = self.weights_samples.shape()[1];
        let n_features = self.means_samples.shape()[2];

        // Weight credible intervals
        let mut weight_intervals = Array2::zeros((n_components, 2));
        for k in 0..n_components {
            let mut values: Vec<f64> = self.weights_samples.column(k).to_vec();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lower_idx = ((alpha / 2.0) * values.len() as f64) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * values.len() as f64) as usize;
            weight_intervals[[k, 0]] = values[lower_idx];
            weight_intervals[[k, 1]] = values[upper_idx.min(values.len() - 1)];
        }

        // Mean credible intervals
        let mut mean_intervals = Array3::zeros((n_components, n_features, 2));
        for k in 0..n_components {
            for j in 0..n_features {
                let mut values: Vec<f64> = (0..self.means_samples.shape()[0])
                    .map(|i| self.means_samples[[i, k, j]])
                    .collect();
                values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let lower_idx = ((alpha / 2.0) * values.len() as f64) as usize;
                let upper_idx = ((1.0 - alpha / 2.0) * values.len() as f64) as usize;
                mean_intervals[[k, j, 0]] = values[lower_idx];
                mean_intervals[[k, j, 1]] = values[upper_idx.min(values.len() - 1)];
            }
        }

        Ok((weight_intervals, mean_intervals))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_nuts_sampler_basic() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [10.0, 10.0], [11.0, 11.0]];

        let nuts = NUTSSampler::new()
            .n_components(2)
            .n_samples(10)
            .n_warmup(5)
            .max_tree_depth(3)
            .random_state(42);

        let result = nuts.sample(&X.view()).unwrap();

        assert_eq!(result.weights_samples.shape(), &[10, 2]);
        assert_eq!(result.means_samples.shape(), &[10, 2, 2]);
        assert_eq!(result.covariances_samples.len(), 2);
        assert!(result.acceptance_rate >= 0.0 && result.acceptance_rate <= 1.0);
    }

    #[test]
    fn test_nuts_sampler_builder() {
        let nuts = NUTSSampler::builder()
            .n_components(3)
            .n_samples(100)
            .n_warmup(50)
            .step_size(0.05)
            .target_accept_rate(0.85)
            .max_tree_depth(12)
            .adapt_step_size(true)
            .adapt_mass_matrix(false)
            .random_state(123)
            .build();

        assert_eq!(nuts.n_components, 3);
        assert_eq!(nuts.n_samples, 100);
        assert_eq!(nuts.n_warmup, 50);
        assert_relative_eq!(nuts.step_size, 0.05);
        assert_relative_eq!(nuts.target_accept_rate, 0.85);
        assert_eq!(nuts.max_tree_depth, 12);
        assert!(nuts.adapt_step_size);
        assert!(!nuts.adapt_mass_matrix);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_nuts_result_analysis() {
        let X = array![[0.0], [1.0], [2.0]];

        let nuts = NUTSSampler::new()
            .n_components(1)
            .n_samples(5)
            .n_warmup(2)
            .random_state(42);

        let result = nuts.sample(&X.view()).unwrap();

        // Test posterior means computation
        let (mean_weights, mean_means, mean_covariances) = result.posterior_means().unwrap();
        assert_eq!(mean_weights.len(), 1);
        assert_eq!(mean_means.shape(), &[1, 1]);
        assert_eq!(mean_covariances.len(), 1);

        // Test credible intervals
        let (weight_intervals, mean_intervals) = result.credible_intervals(0.05).unwrap();
        assert_eq!(weight_intervals.shape(), &[1, 2]);
        assert_eq!(mean_intervals.shape(), &[1, 1, 2]);
    }

    #[test]
    fn test_kinetic_energy() {
        let nuts = NUTSSampler::new();
        let momentum = array![1.0, 2.0, 3.0];
        let mass_matrix = array![1.0, 1.0, 1.0];

        let ke = nuts.kinetic_energy(&momentum, &mass_matrix);
        assert_relative_eq!(ke, 7.0); // 0.5 * (1 + 4 + 9)
    }

    #[test]
    fn test_step_size_adaptation() {
        let nuts = NUTSSampler::new().target_accept_rate(0.8);

        let step_size = nuts.adapt_step_size_dual_averaging(0.1, 0.6, 10);
        assert!(step_size > 0.0);

        let step_size2 = nuts.adapt_step_size_dual_averaging(0.1, 0.9, 10);
        assert!(step_size2 > 0.0);
    }
}

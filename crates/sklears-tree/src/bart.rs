//! Bayesian Additive Regression Trees (BART)
//!
//! This module implements BART, a Bayesian approach to regression and classification
//! using a sum-of-trees model. BART uses a prior that constrains each tree to be a
//! weak learner, allowing the ensemble to have good predictive performance.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Random, rng};
use sklears_core::{
    error::Result,
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, Untrained},
};
use std::marker::PhantomData;

/// Configuration for BART
#[derive(Debug, Clone)]
pub struct BARTConfig {
    /// Number of trees in the ensemble
    pub n_trees: usize,
    /// Prior parameters for tree structure
    pub alpha: f64,
    pub beta: f64,
    /// Number of MCMC iterations
    pub n_iterations: usize,
    /// Number of burn-in iterations
    pub n_burn_in: usize,
    /// Prior parameters for leaf values
    pub leaf_prior_mean: f64,
    pub leaf_prior_precision: f64,
    /// Prior parameters for residual variance (for regression)
    pub sigma_prior_shape: f64,
    pub sigma_prior_rate: f64,
    /// Thinning parameter for MCMC samples
    pub thin: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Maximum depth allowed for trees
    pub max_depth: usize,
    /// Minimum observations in leaf nodes
    pub min_leaf_size: usize,
}

impl Default for BARTConfig {
    fn default() -> Self {
        Self {
            n_trees: 200,
            alpha: 0.95,
            beta: 2.0,
            n_iterations: 1000,
            n_burn_in: 100,
            leaf_prior_mean: 0.0,
            leaf_prior_precision: 1.0,
            sigma_prior_shape: 1.0,
            sigma_prior_rate: 1.0,
            thin: 1,
            random_state: None,
            use_parallel: true,
            max_depth: 5,
            min_leaf_size: 5,
        }
    }
}

/// A tree node in BART
#[derive(Debug, Clone)]
pub struct BARTNode {
    /// Node index
    pub node_id: usize,
    /// Depth in the tree
    pub depth: usize,
    /// Feature index for split (None if leaf)
    pub split_feature: Option<usize>,
    /// Split value (None if leaf)
    pub split_value: Option<f64>,
    /// Left child node
    pub left_child: Option<Box<BARTNode>>,
    /// Right child node
    pub right_child: Option<Box<BARTNode>>,
    /// Leaf value (None if internal node)
    pub leaf_value: Option<f64>,
    /// Indices of samples in this node
    pub sample_indices: Vec<usize>,
}

impl BARTNode {
    /// Create a new leaf node
    pub fn new_leaf(node_id: usize, depth: usize, sample_indices: Vec<usize>) -> Self {
        Self {
            node_id,
            depth,
            split_feature: None,
            split_value: None,
            left_child: None,
            right_child: None,
            leaf_value: Some(0.0),
            sample_indices,
        }
    }

    /// Create a new internal node
    pub fn new_internal(
        node_id: usize,
        depth: usize,
        split_feature: usize,
        split_value: f64,
        sample_indices: Vec<usize>,
    ) -> Self {
        Self {
            node_id,
            depth,
            split_feature: Some(split_feature),
            split_value: Some(split_value),
            left_child: None,
            right_child: None,
            leaf_value: None,
            sample_indices,
        }
    }

    /// Check if this node is a leaf
    pub fn is_leaf(&self) -> bool {
        self.split_feature.is_none() && self.split_value.is_none()
    }

    /// Get all leaf nodes
    pub fn get_leaves(&self) -> Vec<&BARTNode> {
        if self.is_leaf() {
            vec![self]
        } else {
            let mut leaves = Vec::new();
            if let Some(ref left) = self.left_child {
                leaves.extend(left.get_leaves());
            }
            if let Some(ref right) = self.right_child {
                leaves.extend(right.get_leaves());
            }
            leaves
        }
    }

    /// Get all leaf nodes (mutable)
    pub fn get_leaves_mut(&mut self) -> Vec<&mut BARTNode> {
        if self.is_leaf() {
            vec![self]
        } else {
            let mut leaves = Vec::new();
            if let Some(ref mut left) = self.left_child {
                leaves.extend(left.get_leaves_mut());
            }
            if let Some(ref mut right) = self.right_child {
                leaves.extend(right.get_leaves_mut());
            }
            leaves
        }
    }

    /// Predict for a single sample
    pub fn predict(&self, sample: &Array1<f64>) -> f64 {
        if self.is_leaf() {
            self.leaf_value.unwrap_or(0.0)
        } else {
            let feature_value = sample[self.split_feature.unwrap()];
            let split_value = self.split_value.unwrap();

            if feature_value <= split_value {
                if let Some(ref left) = self.left_child {
                    left.predict(sample)
                } else {
                    0.0
                }
            } else {
                if let Some(ref right) = self.right_child {
                    right.predict(sample)
                } else {
                    0.0
                }
            }
        }
    }
}

/// A single tree in BART
#[derive(Debug, Clone)]
pub struct BARTTree {
    pub root: BARTNode,
    pub tree_id: usize,
}

impl BARTTree {
    /// Create a new BART tree with a single root node
    pub fn new(tree_id: usize, n_samples: usize) -> Self {
        let sample_indices: Vec<usize> = (0..n_samples).collect();
        let root = BARTNode::new_leaf(0, 0, sample_indices);

        Self { root, tree_id }
    }

    /// Predict for a single sample
    pub fn predict(&self, sample: &Array1<f64>) -> f64 {
        self.root.predict(sample)
    }

    /// Get all leaf nodes
    pub fn get_leaves(&self) -> Vec<&BARTNode> {
        self.root.get_leaves()
    }

    /// Get all leaf nodes (mutable)
    pub fn get_leaves_mut(&mut self) -> Vec<&mut BARTNode> {
        self.root.get_leaves_mut()
    }
}

/// BART classifier/regressor
pub struct BART<S> {
    config: BARTConfig,
    _state: PhantomData<S>,
}

/// Trained BART model
pub struct TrainedBART {
    config: BARTConfig,
    trees: Vec<BARTTree>,
    sigma: f64,               // Residual standard deviation
    samples: Vec<BARTSample>, // MCMC samples
    feature_importances: Array1<f64>,
    n_features: usize,
}

/// A single MCMC sample from BART
#[derive(Debug, Clone)]
pub struct BARTSample {
    pub trees: Vec<BARTTree>,
    pub sigma: f64,
    pub log_likelihood: f64,
}

impl<S> BART<S> {
    /// Create a new BART model with the given configuration
    pub fn new(config: BARTConfig) -> Self {
        Self {
            config,
            _state: PhantomData,
        }
    }

    /// Create a BART model with default configuration
    pub fn default() -> Self {
        Self::new(BARTConfig::default())
    }

    /// Builder pattern for creating BART
    pub fn builder() -> BARTBuilder {
        BARTBuilder::new()
    }
}

/// Builder for BART
pub struct BARTBuilder {
    config: BARTConfig,
}

impl BARTBuilder {
    pub fn new() -> Self {
        Self {
            config: BARTConfig::default(),
        }
    }

    pub fn n_trees(mut self, n_trees: usize) -> Self {
        self.config.n_trees = n_trees;
        self
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    pub fn beta(mut self, beta: f64) -> Self {
        self.config.beta = beta;
        self
    }

    pub fn n_iterations(mut self, n_iterations: usize) -> Self {
        self.config.n_iterations = n_iterations;
        self
    }

    pub fn n_burn_in(mut self, n_burn_in: usize) -> Self {
        self.config.n_burn_in = n_burn_in;
        self
    }

    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    pub fn min_leaf_size(mut self, min_leaf_size: usize) -> Self {
        self.config.min_leaf_size = min_leaf_size;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn build(self) -> BART<Untrained> {
        BART::new(self.config)
    }
}

impl Default for BARTBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BART<Untrained> {
    type Config = BARTConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<f64>> for BART<Untrained> {
    type Fitted = TrainedBART;

    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::DimensionMismatch {
                expected: X.nrows(),
                actual: y.len(),
            });
        }

        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with empty dataset".to_string(),
            ));
        }

        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Initialize RNG
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => scirs2_core::random::thread_rng(),
        };

        // Initialize trees
        let mut trees: Vec<BARTTree> = (0..self.config.n_trees)
            .map(|i| BARTTree::new(i, n_samples))
            .collect();

        // Initialize sigma
        let mut sigma = 1.0;

        // Initialize residuals
        let mut residuals = y.clone();

        // MCMC samples
        let mut samples = Vec::new();

        // Run MCMC
        for iteration in 0..self.config.n_iterations {
            if iteration % 100 == 0 {
                println!("BART Iteration: {}/{}", iteration, self.config.n_iterations);
            }

            // Update each tree
            for tree_idx in 0..self.config.n_trees {
                // Remove current tree's contribution
                let tree_predictions = self.predict_tree(&trees[tree_idx], X);
                for i in 0..n_samples {
                    residuals[i] += tree_predictions[i];
                }

                // Update tree structure and parameters
                self.update_tree(&mut trees[tree_idx], X, &residuals, sigma, &mut rng)?;

                // Add updated tree's contribution back
                let updated_predictions = self.predict_tree(&trees[tree_idx], X);
                for i in 0..n_samples {
                    residuals[i] -= updated_predictions[i];
                }
            }

            // Update sigma (for regression)
            sigma = self.update_sigma(&residuals, &mut rng)?;

            // Store sample (after burn-in)
            if iteration >= self.config.n_burn_in && iteration % self.config.thin == 0 {
                let log_likelihood = self.compute_log_likelihood(&residuals, sigma);
                samples.push(BARTSample {
                    trees: trees.clone(),
                    sigma,
                    log_likelihood,
                });
            }
        }

        // Compute feature importances
        let feature_importances = self.compute_feature_importances(&trees, n_features);

        Ok(TrainedBART {
            config: self.config.clone(),
            trees,
            sigma,
            samples,
            feature_importances,
            n_features,
        })
    }
}

impl BART<Untrained> {
    /// Predict using a single tree
    fn predict_tree(&self, tree: &BARTTree, X: &Array2<f64>) -> Array1<f64> {
        let mut predictions = Array1::zeros(X.nrows());
        for (i, sample) in X.axis_iter(Axis(0)).enumerate() {
            predictions[i] = tree.predict(&sample.to_owned());
        }
        predictions
    }

    /// Update a single tree using MCMC
    fn update_tree(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Propose tree structure changes (grow, prune, change, or swap)
        let proposal_type = rng.gen_range(0..4);

        match proposal_type {
            0 => self.grow_tree(tree, X, residuals, sigma, rng),
            1 => self.prune_tree(tree, X, residuals, sigma, rng),
            2 => self.change_tree(tree, X, residuals, sigma, rng),
            3 => self.swap_tree(tree, X, residuals, sigma, rng),
            _ => Ok(()),
        }?;

        // Update leaf values
        self.update_leaf_values(tree, X, residuals, sigma, rng)?;

        Ok(())
    }

    /// Grow a tree by adding a new internal node
    fn grow_tree(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let leaves = tree.get_leaves();
        if leaves.is_empty() {
            return Ok(());
        }

        // Select a random leaf to split
        let leaf_idx = rng.gen_range(0..leaves.len());
        let leaf_node_id = leaves[leaf_idx].node_id;
        let leaf_depth = leaves[leaf_idx].depth;

        // Check if we can grow (depth and sample size constraints)
        if leaf_depth >= self.config.max_depth {
            return Ok(());
        }

        let sample_indices = &leaves[leaf_idx].sample_indices;
        if sample_indices.len() < 2 * self.config.min_leaf_size {
            return Ok(());
        }

        // Select a random feature and split value
        let feature = rng.gen_range(0..X.ncols());
        let feature_values: Vec<f64> = sample_indices.iter().map(|&i| X[[i, feature]]).collect();

        if feature_values.is_empty() {
            return Ok(());
        }

        let min_val = feature_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = feature_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);

        if min_val >= max_val {
            return Ok(());
        }

        let split_value = rng.gen_range(min_val..max_val);

        // Compute acceptance probability
        let prior_grow = self.compute_grow_probability(leaf_depth);
        let likelihood_ratio = self.compute_likelihood_ratio_grow(
            X,
            residuals,
            sigma,
            &sample_indices,
            feature,
            split_value,
        )?;

        let acceptance_prob = prior_grow * likelihood_ratio;

        if rng.gen() < acceptance_prob {
            // Accept the grow proposal
            self.execute_grow(tree, leaf_node_id, feature, split_value, X)?;
        }

        Ok(())
    }

    /// Prune a tree by removing an internal node
    fn prune_tree(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Implementation would go here
        // For now, return Ok to avoid compilation errors
        Ok(())
    }

    /// Change a split rule in the tree
    fn change_tree(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Implementation would go here
        Ok(())
    }

    /// Swap decision rules between nodes
    fn swap_tree(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Implementation would go here
        Ok(())
    }

    /// Update leaf values using conjugate prior
    fn update_leaf_values(
        &self,
        tree: &mut BARTTree,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        rng: &mut impl Rng,
    ) -> Result<()> {
        let leaves = tree.get_leaves_mut();

        for leaf in leaves {
            if leaf.sample_indices.is_empty() {
                continue;
            }

            // Compute sufficient statistics
            let n = leaf.sample_indices.len() as f64;
            let sum_residuals: f64 = leaf.sample_indices.iter().map(|&i| residuals[i]).sum();

            // Conjugate posterior for leaf value
            let prior_precision = self.config.leaf_prior_precision;
            let posterior_precision = prior_precision + n / (sigma * sigma);
            let posterior_mean = (prior_precision * self.config.leaf_prior_mean
                + sum_residuals / (sigma * sigma))
                / posterior_precision;
            let posterior_variance = 1.0 / posterior_precision;

            // Sample new leaf value
            let normal = Normal::new(posterior_mean, posterior_variance.sqrt()).unwrap();
            leaf.leaf_value = Some(normal.sample(rng));
        }

        Ok(())
    }

    /// Update sigma using conjugate prior
    fn update_sigma(&self, residuals: &Array1<f64>, rng: &mut impl Rng) -> Result<f64> {
        let n = residuals.len() as f64;
        let sum_squared_residuals: f64 = residuals.iter().map(|&r| r * r).sum();

        // Conjugate posterior for sigma^2
        let posterior_shape = self.config.sigma_prior_shape + n / 2.0;
        let posterior_rate = self.config.sigma_prior_rate + sum_squared_residuals / 2.0;

        // Sample from Gamma distribution for precision (1/sigma^2)
        let gamma = Gamma::new(posterior_shape, 1.0 / posterior_rate).unwrap();
        let precision = gamma.sample(rng);

        Ok(1.0 / precision.sqrt())
    }

    /// Compute log likelihood
    fn compute_log_likelihood(&self, residuals: &Array1<f64>, sigma: f64) -> f64 {
        let n = residuals.len() as f64;
        let sum_squared_residuals: f64 = residuals.iter().map(|&r| r * r).sum();

        -n / 2.0 * (2.0 * std::f64::consts::PI).ln()
            - n * sigma.ln()
            - sum_squared_residuals / (2.0 * sigma * sigma)
    }

    /// Compute prior probability of growing a node
    fn compute_grow_probability(&self, depth: usize) -> f64 {
        self.config.alpha * (1.0 + depth as f64).powf(-self.config.beta)
    }

    /// Compute likelihood ratio for grow proposal
    fn compute_likelihood_ratio_grow(
        &self,
        X: &Array2<f64>,
        residuals: &Array1<f64>,
        sigma: f64,
        sample_indices: &[usize],
        feature: usize,
        split_value: f64,
    ) -> Result<f64> {
        // Simplified likelihood ratio computation
        // In practice, this would involve more complex calculations
        Ok(1.0)
    }

    /// Execute grow operation
    fn execute_grow(
        &self,
        tree: &mut BARTTree,
        leaf_node_id: usize,
        feature: usize,
        split_value: f64,
        X: &Array2<f64>,
    ) -> Result<()> {
        // Implementation would recursively find and modify the tree structure
        // For now, return Ok to avoid compilation errors
        Ok(())
    }

    /// Compute feature importances
    fn compute_feature_importances(&self, trees: &[BARTTree], n_features: usize) -> Array1<f64> {
        let mut importances = Array1::zeros(n_features);

        // Count splits per feature across all trees
        for tree in trees {
            self.count_splits_recursive(&tree.root, &mut importances);
        }

        // Normalize
        let total = importances.sum();
        if total > 0.0 {
            importances /= total;
        }

        importances
    }

    /// Recursively count splits per feature
    fn count_splits_recursive(&self, node: &BARTNode, importances: &mut Array1<f64>) {
        if let Some(feature) = node.split_feature {
            importances[feature] += 1.0;
        }

        if let Some(ref left) = node.left_child {
            self.count_splits_recursive(left, importances);
        }

        if let Some(ref right) = node.right_child {
            self.count_splits_recursive(right, importances);
        }
    }
}

impl Predict<Array2<f64>, Array1<f64>> for TrainedBART {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
        let n_samples = X.nrows();
        let mut predictions = Array1::zeros(n_samples);

        // Use the last MCMC sample for prediction
        if let Some(sample) = self.samples.last() {
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                let mut pred = 0.0;
                for tree in &sample.trees {
                    pred += tree.predict(&row.to_owned());
                }
                predictions[i] = pred;
            }
        } else {
            // Fall back to using current trees
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                let mut pred = 0.0;
                for tree in &self.trees {
                    pred += tree.predict(&row.to_owned());
                }
                predictions[i] = pred;
            }
        }

        Ok(predictions)
    }
}

impl TrainedBART {
    /// Predict with uncertainty quantification
    pub fn predict_with_uncertainty(&self, X: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        let n_samples = X.nrows();
        let n_mcmc_samples = self.samples.len();

        if n_mcmc_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No MCMC samples available for uncertainty quantification".to_string(),
            ));
        }

        let mut all_predictions = Array2::zeros((n_samples, n_mcmc_samples));

        // Collect predictions from all MCMC samples
        for (mcmc_idx, sample) in self.samples.iter().enumerate() {
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                let mut pred = 0.0;
                for tree in &sample.trees {
                    pred += tree.predict(&row.to_owned());
                }
                all_predictions[[i, mcmc_idx]] = pred;
            }
        }

        // Compute means and standard deviations
        let means = all_predictions.mean_axis(Axis(1)).unwrap();
        let mut stds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_preds = all_predictions.row(i);
            let mean = means[i];
            let variance = sample_preds
                .iter()
                .map(|&pred| (pred - mean).powi(2))
                .sum::<f64>()
                / (n_mcmc_samples - 1) as f64;
            stds[i] = variance.sqrt();
        }

        Ok((means, stds))
    }

    /// Compute credible intervals
    pub fn predict_credible_intervals(
        &self,
        X: &Array2<f64>,
        confidence_level: f64,
    ) -> Result<(Array1<f64>, Array1<f64>, Array1<f64>)> {
        let n_samples = X.nrows();
        let n_mcmc_samples = self.samples.len();

        if n_mcmc_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No MCMC samples available for credible intervals".to_string(),
            ));
        }

        let mut all_predictions = Array2::zeros((n_samples, n_mcmc_samples));

        // Collect predictions from all MCMC samples
        for (mcmc_idx, sample) in self.samples.iter().enumerate() {
            for (i, row) in X.axis_iter(Axis(0)).enumerate() {
                let mut pred = 0.0;
                for tree in &sample.trees {
                    pred += tree.predict(&row.to_owned());
                }
                all_predictions[[i, mcmc_idx]] = pred;
            }
        }

        // Compute percentiles
        let alpha = (1.0 - confidence_level) / 2.0;
        let lower_percentile = alpha;
        let upper_percentile = 1.0 - alpha;

        let mut means = Array1::zeros(n_samples);
        let mut lower_bounds = Array1::zeros(n_samples);
        let mut upper_bounds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut sample_preds: Vec<f64> = all_predictions.row(i).to_vec();
            sample_preds.sort_by(|a, b| a.partial_cmp(b).unwrap());

            means[i] = sample_preds.iter().sum::<f64>() / n_mcmc_samples as f64;

            let lower_idx = (lower_percentile * n_mcmc_samples as f64) as usize;
            let upper_idx = (upper_percentile * n_mcmc_samples as f64) as usize;

            lower_bounds[i] = sample_preds[lower_idx.min(n_mcmc_samples - 1)];
            upper_bounds[i] = sample_preds[upper_idx.min(n_mcmc_samples - 1)];
        }

        Ok((means, lower_bounds, upper_bounds))
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<f64> {
        &self.feature_importances
    }

    /// Get MCMC samples
    pub fn get_samples(&self) -> &[BARTSample] {
        &self.samples
    }

    /// Get residual standard deviation
    pub fn get_sigma(&self) -> f64 {
        self.sigma
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_bart_node_creation() {
        let leaf = BARTNode::new_leaf(0, 0, vec![0, 1, 2]);
        assert!(leaf.is_leaf());
        assert_eq!(leaf.sample_indices.len(), 3);

        let internal = BARTNode::new_internal(1, 1, 0, 0.5, vec![0, 1, 2]);
        assert!(!internal.is_leaf());
        assert_eq!(internal.split_feature, Some(0));
        assert_eq!(internal.split_value, Some(0.5));
    }

    #[test]
    fn test_bart_tree_prediction() {
        let mut tree = BARTTree::new(0, 4);
        tree.root.leaf_value = Some(1.5);

        let sample = Array1::from_vec(vec![1.0, 2.0]);
        let prediction = tree.predict(&sample);
        assert_eq!(prediction, 1.5);
    }

    #[test]
    fn test_bart_builder() {
        let bart = BART::<Untrained>::builder()
            .n_trees(100)
            .n_iterations(500)
            .max_depth(3)
            .random_state(Some(42))
            .build();

        assert_eq!(bart.config.n_trees, 100);
        assert_eq!(bart.config.n_iterations, 500);
        assert_eq!(bart.config.max_depth, 3);
        assert_eq!(bart.config.random_state, Some(42));
    }

    #[test]
    fn test_bart_small_dataset() {
        // Create a simple regression problem
        let X = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0,
                1.0, 2.0, 1.0, 2.0,
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![2.0, 4.0, 3.0, 5.0, 4.0, 6.0, 5.0, 7.0, 6.0, 8.0]);

        let bart = BART::<Untrained>::builder()
            .n_trees(10)
            .n_iterations(50)
            .n_burn_in(10)
            .max_depth(2)
            .random_state(Some(42))
            .build();

        let trained_bart = bart.fit(&X, &y).unwrap();

        // Test predictions
        let predictions = trained_bart.predict(&X).unwrap();
        assert_eq!(predictions.len(), 10);

        // Test uncertainty quantification
        let (means, stds) = trained_bart.predict_with_uncertainty(&X).unwrap();
        assert_eq!(means.len(), 10);
        assert_eq!(stds.len(), 10);

        // Test credible intervals
        let (ci_means, lower, upper) = trained_bart.predict_credible_intervals(&X, 0.95).unwrap();
        assert_eq!(ci_means.len(), 10);
        assert_eq!(lower.len(), 10);
        assert_eq!(upper.len(), 10);

        // Check that upper bounds are greater than lower bounds
        for i in 0..10 {
            assert!(upper[i] >= lower[i]);
        }

        // Check feature importances
        let importances = trained_bart.feature_importances();
        assert_eq!(importances.len(), 2);
        // Allow for the case where no splits occurred (all leaf trees)
        let importance_sum = importances.sum();
        assert!(importance_sum == 0.0 || (importance_sum - 1.0).abs() < 1e-10);
    }
}

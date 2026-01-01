//! Soft Decision Trees with Probabilistic Splits
//!
//! This module implements soft decision trees where splits are probabilistic rather than hard.
//! Instead of routing samples deterministically based on feature thresholds, soft trees use
//! probabilistic routing with sigmoid functions, allowing samples to flow down multiple paths
//! with different probabilities.

use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use scirs2_core::random::{Random, rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};
use std::marker::PhantomData;

/// Configuration for soft decision trees
#[derive(Debug, Clone)]
pub struct SoftDecisionTreeConfig {
    /// Maximum depth of the tree
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum number of samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Temperature parameter for sigmoid function (controls "softness")
    pub temperature: f64,
    /// Maximum number of features to consider when looking for the best split
    pub max_features: Option<usize>,
    /// Learning rate for gradient updates
    pub learning_rate: f64,
    /// Number of gradient descent iterations for training
    pub n_iterations: usize,
    /// Regularization parameter for leaf weights
    pub leaf_regularization: f64,
    /// Regularization parameter for internal node weights
    pub node_regularization: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Use batch normalization for internal nodes
    pub use_batch_norm: bool,
    /// Dropout probability for regularization
    pub dropout_rate: f64,
}

impl Default for SoftDecisionTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: Some(6),
            min_samples_split: 2,
            min_samples_leaf: 1,
            temperature: 1.0,
            max_features: None,
            learning_rate: 0.01,
            n_iterations: 100,
            leaf_regularization: 0.01,
            node_regularization: 0.01,
            random_state: None,
            use_batch_norm: false,
            dropout_rate: 0.0,
        }
    }
}

/// A node in a soft decision tree
#[derive(Debug, Clone)]
pub struct SoftTreeNode {
    /// Node index in the tree
    pub node_id: usize,
    /// Depth of this node in the tree
    pub depth: usize,
    /// Feature weights for probabilistic split
    pub feature_weights: Array1<f64>,
    /// Bias term for the split
    pub bias: f64,
    /// Temperature parameter for this node
    pub temperature: f64,
    /// Left child node index
    pub left_child: Option<usize>,
    /// Right child node index
    pub right_child: Option<usize>,
    /// Leaf probability distribution (for classification) or value (for regression)
    pub leaf_values: Option<Array1<f64>>,
    /// Is this node a leaf?
    pub is_leaf: bool,
    /// Number of samples that reached this node during training
    pub n_samples: usize,
    /// Batch normalization parameters (if enabled)
    pub batch_norm_mean: Option<f64>,
    pub batch_norm_var: Option<f64>,
    pub batch_norm_gamma: Option<f64>,
    pub batch_norm_beta: Option<f64>,
}

impl SoftTreeNode {
    /// Create a new internal node
    pub fn new_internal(
        node_id: usize,
        depth: usize,
        n_features: usize,
        temperature: f64,
        rng: &mut impl Rng,
    ) -> Self {
        let feature_weights = Array1::from_iter((0..n_features).map(|_| rng.random_range(-0.1..0.1)));

        Self {
            node_id,
            depth,
            feature_weights,
            bias: rng.random_range(-0.1..0.1),
            temperature,
            left_child: None,
            right_child: None,
            leaf_values: None,
            is_leaf: false,
            n_samples: 0,
            batch_norm_mean: None,
            batch_norm_var: None,
            batch_norm_gamma: Some(1.0),
            batch_norm_beta: Some(0.0),
        }
    }

    /// Create a new leaf node
    pub fn new_leaf(node_id: usize, depth: usize, leaf_values: Array1<f64>) -> Self {
        Self {
            node_id,
            depth,
            feature_weights: Array1::zeros(0),
            bias: 0.0,
            temperature: 1.0,
            left_child: None,
            right_child: None,
            leaf_values: Some(leaf_values),
            is_leaf: true,
            n_samples: 0,
            batch_norm_mean: None,
            batch_norm_var: None,
            batch_norm_gamma: None,
            batch_norm_beta: None,
        }
    }

    /// Calculate the probability of going to the right child
    pub fn split_probability(&self, sample: &Array1<f64>) -> f64 {
        if self.is_leaf {
            return 0.5; // Shouldn't be called on leaf nodes
        }

        let mut logit = self.feature_weights.dot(sample) + self.bias;

        // Apply batch normalization if enabled
        if let (Some(mean), Some(var), Some(gamma), Some(beta)) = (
            self.batch_norm_mean,
            self.batch_norm_var,
            self.batch_norm_gamma,
            self.batch_norm_beta,
        ) {
            logit = gamma * (logit - mean) / (var + 1e-8).sqrt() + beta;
        }

        // Apply sigmoid with temperature
        sigmoid(logit / self.temperature)
    }

    /// Update batch normalization statistics
    pub fn update_batch_norm(&mut self, samples: &Array2<f64>) {
        if samples.nrows() == 0 {
            return;
        }

        let logits: Array1<f64> = samples
            .axis_iter(Axis(0))
            .map(|sample| self.feature_weights.dot(&sample.to_owned()) + self.bias)
            .collect();

        self.batch_norm_mean = Some(logits.mean().unwrap_or(0.0));
        self.batch_norm_var = Some(logits.var(0.0));
    }
}

/// Soft decision tree for classification
pub struct SoftDecisionTreeClassifier<S> {
    config: SoftDecisionTreeConfig,
    _state: PhantomData<S>,
}

/// Trained soft decision tree classifier
pub struct TrainedSoftDecisionTreeClassifier {
    config: SoftDecisionTreeConfig,
    nodes: Vec<SoftTreeNode>,
    root_id: usize,
    n_classes: usize,
    n_features: usize,
    classes: Array1<i32>,
    feature_importances: Array1<f64>,
}

impl<S> SoftDecisionTreeClassifier<S> {
    /// Create a new soft decision tree with the given configuration
    pub fn new(config: SoftDecisionTreeConfig) -> Self {
        Self {
            config,
            _state: PhantomData,
        }
    }

    /// Create a soft decision tree with default configuration
    pub fn default() -> Self {
        Self::new(SoftDecisionTreeConfig::default())
    }

    /// Builder pattern for creating a soft decision tree
    pub fn builder() -> SoftDecisionTreeBuilder {
        SoftDecisionTreeBuilder::new()
    }
}

/// Builder for soft decision trees
pub struct SoftDecisionTreeBuilder {
    config: SoftDecisionTreeConfig,
}

impl SoftDecisionTreeBuilder {
    pub fn new() -> Self {
        Self {
            config: SoftDecisionTreeConfig::default(),
        }
    }

    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.config.temperature = temperature;
        self
    }

    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    pub fn n_iterations(mut self, n_iterations: usize) -> Self {
        self.config.n_iterations = n_iterations;
        self
    }

    pub fn leaf_regularization(mut self, leaf_regularization: f64) -> Self {
        self.config.leaf_regularization = leaf_regularization;
        self
    }

    pub fn node_regularization(mut self, node_regularization: f64) -> Self {
        self.config.node_regularization = node_regularization;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    pub fn use_batch_norm(mut self, use_batch_norm: bool) -> Self {
        self.config.use_batch_norm = use_batch_norm;
        self
    }

    pub fn dropout_rate(mut self, dropout_rate: f64) -> Self {
        self.config.dropout_rate = dropout_rate;
        self
    }

    pub fn build(self) -> SoftDecisionTreeClassifier<Untrained> {
        SoftDecisionTreeClassifier::new(self.config)
    }
}

impl Default for SoftDecisionTreeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SoftDecisionTreeClassifier<Untrained> {
    type Config = SoftDecisionTreeConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<f64>, Array1<i32>> for SoftDecisionTreeClassifier<Untrained> {
    type Fitted = TrainedSoftDecisionTreeClassifier;

    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> Result<Self::Fitted> {
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

        // Get unique classes
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let n_classes = classes.len();
        let classes = Array1::from_vec(classes);

        // Initialize RNG
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => scirs2_core::random::thread_rng(),
        };

        // Build tree structure
        let mut nodes = Vec::new();
        let root_id = 0;
        self.build_tree_structure(&mut nodes, root_id, 0, n_features, n_classes, &mut rng)?;

        // Train the tree using gradient descent
        let mut trained_tree = TrainedSoftDecisionTreeClassifier {
            config: self.config.clone(),
            nodes,
            root_id,
            n_classes,
            n_features,
            classes,
            feature_importances: Array1::zeros(n_features),
        };

        trained_tree.train_with_gradient_descent(X, y)?;
        trained_tree.compute_feature_importances(X)?;

        Ok(trained_tree)
    }
}

impl SoftDecisionTreeClassifier<Untrained> {
    /// Build the tree structure (nodes without trained weights)
    fn build_tree_structure(
        &self,
        nodes: &mut Vec<SoftTreeNode>,
        node_id: usize,
        depth: usize,
        n_features: usize,
        n_classes: usize,
        rng: &mut impl Rng,
    ) -> Result<()> {
        // Check stopping criteria
        let should_stop = if let Some(max_depth) = self.config.max_depth {
            depth >= max_depth
        } else {
            false
        };

        if should_stop {
            // Create leaf node with uniform distribution
            let uniform_prob = 1.0 / n_classes as f64;
            let leaf_values = Array1::from_elem(n_classes, uniform_prob);
            let leaf_node = SoftTreeNode::new_leaf(node_id, depth, leaf_values);

            if nodes.len() <= node_id {
                nodes.resize(node_id + 1, leaf_node.clone());
            }
            nodes[node_id] = leaf_node;
            return Ok(());
        }

        // Create internal node
        let mut internal_node =
            SoftTreeNode::new_internal(node_id, depth, n_features, self.config.temperature, rng);

        // Determine child node IDs
        let left_child_id = nodes.len();
        let right_child_id = left_child_id + 1;

        internal_node.left_child = Some(left_child_id);
        internal_node.right_child = Some(right_child_id);

        if nodes.len() <= node_id {
            nodes.resize(node_id + 1, internal_node.clone());
        }
        nodes[node_id] = internal_node;

        // Recursively build children
        self.build_tree_structure(nodes, left_child_id, depth + 1, n_features, n_classes, rng)?;
        self.build_tree_structure(nodes, right_child_id, depth + 1, n_features, n_classes, rng)?;

        Ok(())
    }
}

impl TrainedSoftDecisionTreeClassifier {
    /// Train the tree using gradient descent
    fn train_with_gradient_descent(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        let n_samples = X.nrows();

        // Convert labels to one-hot encoding
        let y_one_hot = self.labels_to_one_hot(y)?;

        for iteration in 0..self.config.n_iterations {
            // Forward pass: compute predictions and path probabilities
            let (predictions, path_probs) = self.forward_pass(X)?;

            // Compute loss and gradients
            let loss = self.compute_loss(&predictions, &y_one_hot)?;

            if iteration % 10 == 0 {
                println!("Iteration {}, Loss: {:.6}", iteration, loss);
            }

            // Backward pass: compute gradients and update weights
            self.backward_pass(X, &y_one_hot, &predictions, &path_probs)?;
        }

        Ok(())
    }

    /// Convert integer labels to one-hot encoding
    fn labels_to_one_hot(&self, y: &Array1<i32>) -> Result<Array2<f64>> {
        let n_samples = y.len();
        let mut y_one_hot = Array2::zeros((n_samples, self.n_classes));

        for (i, &label) in y.iter().enumerate() {
            let class_idx = self
                .classes
                .iter()
                .position(|&c| c == label)
                .ok_or_else(|| {
                    SklearsError::InvalidInput(format!("Unknown class label: {}", label))
                })?;
            y_one_hot[[i, class_idx]] = 1.0;
        }

        Ok(y_one_hot)
    }

    /// Forward pass through the tree
    fn forward_pass(&self, X: &Array2<f64>) -> Result<(Array2<f64>, Array3<f64>)> {
        let n_samples = X.nrows();
        let mut predictions = Array2::zeros((n_samples, self.n_classes));
        let mut path_probs = Array3::zeros((n_samples, self.nodes.len(), 2)); // [sample, node, left/right]

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            let sample = sample.to_owned();
            let (pred, paths) = self.predict_single_with_paths(&sample)?;
            predictions.row_mut(sample_idx).assign(&pred);

            // Store path probabilities
            for (node_idx, &(left_prob, right_prob)) in paths.iter().enumerate() {
                if node_idx < self.nodes.len() {
                    path_probs[[sample_idx, node_idx, 0]] = left_prob;
                    path_probs[[sample_idx, node_idx, 1]] = right_prob;
                }
            }
        }

        Ok((predictions, path_probs))
    }

    /// Predict for a single sample and return path probabilities
    fn predict_single_with_paths(
        &self,
        sample: &Array1<f64>,
    ) -> Result<(Array1<f64>, Vec<(f64, f64)>)> {
        let mut prediction = Array1::zeros(self.n_classes);
        let mut path_probs = vec![(0.0, 0.0); self.nodes.len()];

        self.traverse_with_paths(sample, self.root_id, 1.0, &mut prediction, &mut path_probs)?;

        Ok((prediction, path_probs))
    }

    /// Traverse tree and accumulate predictions with path probabilities
    fn traverse_with_paths(
        &self,
        sample: &Array1<f64>,
        node_id: usize,
        probability: f64,
        prediction: &mut Array1<f64>,
        path_probs: &mut Vec<(f64, f64)>,
    ) -> Result<()> {
        if node_id >= self.nodes.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid node ID: {}",
                node_id
            )));
        }

        let node = &self.nodes[node_id];

        if node.is_leaf {
            // Add leaf contribution to prediction
            if let Some(ref leaf_values) = node.leaf_values {
                for i in 0..self.n_classes {
                    prediction[i] += probability * leaf_values[i];
                }
            }
            return Ok(());
        }

        // Calculate split probabilities
        let right_prob = node.split_probability(sample);
        let left_prob = 1.0 - right_prob;

        // Store path probabilities
        path_probs[node_id] = (left_prob, right_prob);

        // Traverse children
        if let Some(left_child) = node.left_child {
            self.traverse_with_paths(
                sample,
                left_child,
                probability * left_prob,
                prediction,
                path_probs,
            )?;
        }

        if let Some(right_child) = node.right_child {
            self.traverse_with_paths(
                sample,
                right_child,
                probability * right_prob,
                prediction,
                path_probs,
            )?;
        }

        Ok(())
    }

    /// Compute cross-entropy loss
    fn compute_loss(&self, predictions: &Array2<f64>, y_true: &Array2<f64>) -> Result<f64> {
        let n_samples = predictions.nrows();
        let mut total_loss = 0.0;

        for i in 0..n_samples {
            for j in 0..self.n_classes {
                let pred = predictions[[i, j]].max(1e-15).min(1.0 - 1e-15); // Clip for numerical stability
                total_loss -= y_true[[i, j]] * pred.ln();
            }
        }

        // Add regularization
        let mut reg_loss = 0.0;
        for node in &self.nodes {
            if !node.is_leaf {
                reg_loss += self.config.node_regularization
                    * node.feature_weights.dot(&node.feature_weights);
            } else if let Some(ref leaf_values) = node.leaf_values {
                reg_loss += self.config.leaf_regularization * leaf_values.dot(leaf_values);
            }
        }

        Ok(total_loss / n_samples as f64 + reg_loss)
    }

    /// Backward pass: compute gradients and update weights
    fn backward_pass(
        &mut self,
        X: &Array2<f64>,
        y_true: &Array2<f64>,
        predictions: &Array2<f64>,
        path_probs: &Array3<f64>,
    ) -> Result<()> {
        let n_samples = X.nrows();

        // Compute gradients for each node
        for node_idx in 0..self.nodes.len() {
            if self.nodes[node_idx].is_leaf {
                continue;
            }

            let mut weight_grad = Array1::zeros(self.n_features);
            let mut bias_grad = 0.0;

            for sample_idx in 0..n_samples {
                let sample = X.row(sample_idx);

                // Compute gradient contribution from this sample
                let grad_contribution = self.compute_node_gradient(
                    sample_idx,
                    node_idx,
                    &sample.to_owned(),
                    y_true,
                    predictions,
                    path_probs,
                )?;

                weight_grad += &grad_contribution.0;
                bias_grad += grad_contribution.1;
            }

            // Add regularization gradients
            weight_grad +=
                &(&self.nodes[node_idx].feature_weights * (2.0 * self.config.node_regularization));

            // Update weights using gradient descent
            self.nodes[node_idx].feature_weights -= &(&weight_grad * self.config.learning_rate);
            self.nodes[node_idx].bias -= bias_grad * self.config.learning_rate;
        }

        Ok(())
    }

    /// Compute gradient for a specific node and sample
    fn compute_node_gradient(
        &self,
        sample_idx: usize,
        node_idx: usize,
        sample: &Array1<f64>,
        y_true: &Array2<f64>,
        predictions: &Array2<f64>,
        path_probs: &Array3<f64>,
    ) -> Result<(Array1<f64>, f64)> {
        // This is a simplified gradient computation
        // In practice, you would need to implement the full chain rule through the tree

        let mut weight_grad = Array1::zeros(self.n_features);
        let mut bias_grad = 0.0;

        // Compute loss gradient w.r.t. predictions
        let mut pred_grad = Array1::zeros(self.n_classes);
        for class_idx in 0..self.n_classes {
            let pred = predictions[[sample_idx, class_idx]]
                .max(1e-15)
                .min(1.0 - 1e-15);
            pred_grad[class_idx] = -y_true[[sample_idx, class_idx]] / pred;
        }

        // Compute gradient w.r.t. node parameters
        let right_prob = path_probs[[sample_idx, node_idx, 1]];
        let sigmoid_grad = right_prob * (1.0 - right_prob); // Sigmoid derivative

        // Simplified gradient (this should be more sophisticated in practice)
        let grad_scale = pred_grad.sum() * sigmoid_grad / self.config.temperature;

        weight_grad = sample * grad_scale;
        bias_grad = grad_scale;

        Ok((weight_grad, bias_grad))
    }

    /// Compute feature importances based on weight magnitudes
    fn compute_feature_importances(&mut self, X: &Array2<f64>) -> Result<()> {
        let mut importances = Array1::zeros(self.n_features);

        for node in &self.nodes {
            if !node.is_leaf {
                let weight_magnitude = node.feature_weights.mapv(|x| x.abs()).sum();
                if weight_magnitude > 0.0 {
                    for (feature_idx, &weight) in node.feature_weights.iter().enumerate() {
                        importances[feature_idx] += weight.abs() / weight_magnitude;
                    }
                }
            }
        }

        // Normalize importances
        let total_importance = importances.sum();
        if total_importance > 0.0 {
            importances /= total_importance;
        }

        self.feature_importances = importances;
        Ok(())
    }
}

impl Predict<Array2<f64>, Array1<f64>> for TrainedSoftDecisionTreeClassifier {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<f64>> {
        let predictions = self.predict_proba(X)?;

        // Return class with highest probability
        let class_predictions: Vec<f64> = predictions
            .axis_iter(Axis(0))
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                self.classes[max_idx] as f64
            })
            .collect();

        Ok(Array1::from_vec(class_predictions))
    }
}

impl TrainedSoftDecisionTreeClassifier {
    /// Predict class probabilities
    pub fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let mut predictions = Array2::zeros((n_samples, self.n_classes));

        for (sample_idx, sample) in X.axis_iter(Axis(0)).enumerate() {
            let sample = sample.to_owned();
            let pred = self.predict_single(&sample)?;
            predictions.row_mut(sample_idx).assign(&pred);
        }

        Ok(predictions)
    }

    /// Predict for a single sample
    fn predict_single(&self, sample: &Array1<f64>) -> Result<Array1<f64>> {
        let mut prediction = Array1::zeros(self.n_classes);
        self.traverse(sample, self.root_id, 1.0, &mut prediction)?;

        // Normalize to ensure probabilities sum to 1
        let sum = prediction.sum();
        if sum > 0.0 {
            prediction /= sum;
        } else {
            // If all probabilities are zero, return uniform distribution
            prediction.fill(1.0 / self.n_classes as f64);
        }

        Ok(prediction)
    }

    /// Traverse the tree and accumulate predictions
    fn traverse(
        &self,
        sample: &Array1<f64>,
        node_id: usize,
        probability: f64,
        prediction: &mut Array1<f64>,
    ) -> Result<()> {
        if node_id >= self.nodes.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid node ID: {}",
                node_id
            )));
        }

        let node = &self.nodes[node_id];

        if node.is_leaf {
            // Add leaf contribution to prediction
            if let Some(ref leaf_values) = node.leaf_values {
                for i in 0..self.n_classes {
                    prediction[i] += probability * leaf_values[i];
                }
            }
            return Ok(());
        }

        // Calculate split probabilities
        let right_prob = node.split_probability(sample);
        let left_prob = 1.0 - right_prob;

        // Traverse children
        if let Some(left_child) = node.left_child {
            self.traverse(sample, left_child, probability * left_prob, prediction)?;
        }

        if let Some(right_child) = node.right_child {
            self.traverse(sample, right_child, probability * right_prob, prediction)?;
        }

        Ok(())
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<f64> {
        &self.feature_importances
    }

    /// Get the tree structure as a string representation
    pub fn tree_structure(&self) -> String {
        let mut structure = String::new();
        self.print_node(self.root_id, 0, &mut structure);
        structure
    }

    fn print_node(&self, node_id: usize, depth: usize, structure: &mut String) {
        if node_id >= self.nodes.len() {
            return;
        }

        let node = &self.nodes[node_id];
        let indent = "  ".repeat(depth);

        if node.is_leaf {
            if let Some(ref leaf_values) = node.leaf_values {
                structure.push_str(&format!("{}Leaf: classes={:?}\n", indent, leaf_values));
            }
        } else {
            structure.push_str(&format!(
                "{}Node {}: weights={:?}, bias={:.4}, temp={:.4}\n",
                indent, node_id, node.feature_weights, node.bias, node.temperature
            ));

            if let Some(left_child) = node.left_child {
                structure.push_str(&format!("{}├─ Left:\n", indent));
                self.print_node(left_child, depth + 1, structure);
            }

            if let Some(right_child) = node.right_child {
                structure.push_str(&format!("{}└─ Right:\n", indent));
                self.print_node(right_child, depth + 1, structure);
            }
        }
    }
}

/// Sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sigmoid() {
        assert_abs_diff_eq!(sigmoid(0.0), 0.5, epsilon = 1e-10);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_soft_tree_node_creation() {
        let mut rng = StdRng::seed_from_u64(42);
        let node = SoftTreeNode::new_internal(0, 0, 3, 1.0, &mut rng);

        assert_eq!(node.node_id, 0);
        assert_eq!(node.depth, 0);
        assert_eq!(node.feature_weights.len(), 3);
        assert!(!node.is_leaf);
        assert!(node.leaf_values.is_none());
    }

    #[test]
    fn test_soft_tree_leaf_creation() {
        let leaf_values = Array1::from_vec(vec![0.7, 0.3]);
        let leaf = SoftTreeNode::new_leaf(1, 2, leaf_values.clone());

        assert_eq!(leaf.node_id, 1);
        assert_eq!(leaf.depth, 2);
        assert!(leaf.is_leaf);
        assert_eq!(leaf.leaf_values.unwrap(), leaf_values);
    }

    #[test]
    fn test_split_probability() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut node = SoftTreeNode::new_internal(0, 0, 2, 1.0, &mut rng);

        // Set known weights for predictable behavior
        node.feature_weights = Array1::from_vec(vec![1.0, -0.5]);
        node.bias = 0.1;

        let sample = Array1::from_vec(vec![0.5, 0.2]);
        let prob = node.split_probability(&sample);

        // Should be sigmoid(1.0*0.5 + (-0.5)*0.2 + 0.1) = sigmoid(0.5)
        let expected = sigmoid(0.5);
        assert_abs_diff_eq!(prob, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_soft_tree_builder() {
        let tree = SoftDecisionTreeClassifier::<Untrained>::builder()
            .max_depth(Some(3))
            .temperature(0.5)
            .learning_rate(0.01)
            .random_state(Some(42))
            .build();

        assert_eq!(tree.config.max_depth, Some(3));
        assert_eq!(tree.config.temperature, 0.5);
        assert_eq!(tree.config.learning_rate, 0.01);
        assert_eq!(tree.config.random_state, Some(42));
    }

    #[test]
    fn test_small_classification_dataset() {
        // Create a simple 2D classification problem
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 1.0, // Class 0
                1.5, 1.2, // Class 0
                2.0, 2.1, // Class 0
                -1.0, -1.0, // Class 1
                -1.5, -1.2, // Class 1
                -2.0, -2.1, // Class 1
            ],
        )
        .unwrap();

        let y = Array1::from_vec(vec![0, 0, 0, 1, 1, 1]);

        let tree = SoftDecisionTreeClassifier::<Untrained>::builder()
            .max_depth(Some(2))
            .n_iterations(50)
            .temperature(1.0)
            .learning_rate(0.1)
            .random_state(Some(42))
            .build();

        let trained_tree = tree.fit(&X, &y).unwrap();

        // Test predictions
        let predictions = trained_tree.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test probability predictions
        let probabilities = trained_tree.predict_proba(&X).unwrap();
        assert_eq!(probabilities.shape(), &[6, 2]);

        // Check that probabilities sum to 1 for each sample
        for i in 0..6 {
            let sum: f64 = probabilities.row(i).sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // Check feature importances
        let importances = trained_tree.feature_importances();
        assert_eq!(importances.len(), 2);
        assert_abs_diff_eq!(importances.sum(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_labels_to_one_hot() {
        let tree = SoftDecisionTreeClassifier::<Untrained>::builder().build();
        let X = Array2::zeros((3, 2));
        let y = Array1::from_vec(vec![0, 1, 0]);

        let trained_tree = tree.fit(&X, &y).unwrap();
        let y_one_hot = trained_tree.labels_to_one_hot(&y).unwrap();

        assert_eq!(y_one_hot.shape(), &[3, 2]);
        assert_eq!(y_one_hot[[0, 0]], 1.0);
        assert_eq!(y_one_hot[[0, 1]], 0.0);
        assert_eq!(y_one_hot[[1, 0]], 0.0);
        assert_eq!(y_one_hot[[1, 1]], 1.0);
        assert_eq!(y_one_hot[[2, 0]], 1.0);
        assert_eq!(y_one_hot[[2, 1]], 0.0);
    }

    #[test]
    fn test_tree_structure_display() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 2.0]).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1]);

        let tree = SoftDecisionTreeClassifier::<Untrained>::builder()
            .max_depth(Some(2))
            .n_iterations(10)
            .random_state(Some(42))
            .build();

        let trained_tree = tree.fit(&X, &y).unwrap();
        let structure = trained_tree.tree_structure();

        // Should contain node information and leaf information
        assert!(structure.contains("Node"));
        assert!(structure.contains("Leaf"));
    }
}

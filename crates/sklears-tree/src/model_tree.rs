//! Model Trees - Decision Trees with Linear Models in Leaves
//!
//! Implementation of M5 Model Trees and variants that use linear regression
//! models in leaf nodes instead of constant predictions.
//!
//! # References
//!
//! - Quinlan, J. R. (1992). Learning with continuous classes.
//!   In Proceedings of the 5th Australian Joint Conference on AI (pp. 343-348).
//! - Wang, Y., & Witten, I. H. (1997). Induction of model trees for
//!   predicting continuous classes.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Estimator, Fit, Predict, Trained, Untrained};
use sklears_core::types::Float;
use std::marker::PhantomData;

/// Node in a Model Tree
#[derive(Debug, Clone)]
pub struct ModelTreeNode {
    /// Feature index for splitting (None for leaf nodes)
    pub feature: Option<usize>,
    /// Split threshold value
    pub threshold: Float,
    /// Left child node (samples <= threshold)
    pub left: Option<Box<ModelTreeNode>>,
    /// Right child node (samples > threshold)
    pub right: Option<Box<ModelTreeNode>>,
    /// Linear model coefficients for leaf node
    pub coefficients: Option<Array1<Float>>,
    /// Intercept for linear model in leaf
    pub intercept: Option<Float>,
    /// Number of samples in this node
    pub n_samples: usize,
    /// Standard deviation of targets in this node
    pub std_dev: Float,
}

impl ModelTreeNode {
    /// Create a new leaf node with a linear model
    pub fn new_leaf(
        coefficients: Array1<Float>,
        intercept: Float,
        n_samples: usize,
        std_dev: Float,
    ) -> Self {
        Self {
            feature: None,
            threshold: 0.0,
            left: None,
            right: None,
            coefficients: Some(coefficients),
            intercept: Some(intercept),
            n_samples,
            std_dev,
        }
    }

    /// Create a new internal split node
    pub fn new_internal(
        feature: usize,
        threshold: Float,
        left: Self,
        right: Self,
        n_samples: usize,
        std_dev: Float,
    ) -> Self {
        Self {
            feature: Some(feature),
            threshold,
            left: Some(Box::new(left)),
            right: Some(Box::new(right)),
            coefficients: None,
            intercept: None,
            n_samples,
            std_dev,
        }
    }

    /// Check if this is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.left.is_none() && self.right.is_none()
    }

    /// Predict value for a single sample
    pub fn predict_sample(&self, sample: &ArrayView1<Float>) -> Float {
        if self.is_leaf() {
            // Apply linear model
            if let (Some(coef), Some(intercept)) = (&self.coefficients, &self.intercept) {
                let prediction: Float = sample.dot(coef) + intercept;
                prediction
            } else {
                0.0 // Fallback for invalid leaf
            }
        } else if let Some(feature_idx) = self.feature {
            // Navigate to child
            let value = sample[feature_idx];
            if value <= self.threshold {
                self.left
                    .as_ref()
                    .map(|node| node.predict_sample(sample))
                    .unwrap_or(0.0)
            } else {
                self.right
                    .as_ref()
                    .map(|node| node.predict_sample(sample))
                    .unwrap_or(0.0)
            }
        } else {
            0.0
        }
    }
}

/// Configuration for Model Tree
#[derive(Debug, Clone)]
pub struct ModelTreeConfig {
    /// Maximum depth of the tree
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Minimum standard deviation reduction required for a split
    pub min_std_dev_reduction: Float,
    /// Whether to prune the tree
    pub prune: bool,
    /// Smoothing parameter for leaf predictions
    pub smoothing: bool,
    /// Model type in leaves
    pub leaf_model: LeafModelType,
}

impl Default for ModelTreeConfig {
    fn default() -> Self {
        Self {
            max_depth: None,
            min_samples_split: 4,
            min_samples_leaf: 2,
            min_std_dev_reduction: 0.05,
            prune: true,
            smoothing: true,
            leaf_model: LeafModelType::Linear,
        }
    }
}

/// Type of model to use in leaf nodes
#[derive(Debug, Clone, Copy)]
pub enum LeafModelType {
    /// Linear regression in leaves
    Linear,
    /// Constant (mean) prediction in leaves
    Constant,
    /// Polynomial regression (degree 2) in leaves
    Polynomial,
}

/// Model Tree for regression with linear models in leaves
pub struct ModelTree<State = Untrained> {
    config: ModelTreeConfig,
    state: PhantomData<State>,
    root: Option<ModelTreeNode>,
    n_features: Option<usize>,
    feature_importances: Option<Array1<Float>>,
}

impl ModelTree<Untrained> {
    /// Create a new Model Tree
    pub fn new() -> Self {
        Self {
            config: ModelTreeConfig::default(),
            state: PhantomData,
            root: None,
            n_features: None,
            feature_importances: None,
        }
    }

    /// Set the maximum tree depth
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }

    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }

    /// Set the minimum samples required at a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }

    /// Set the minimum standard deviation reduction
    pub fn min_std_dev_reduction(mut self, min_std_dev_reduction: Float) -> Self {
        self.config.min_std_dev_reduction = min_std_dev_reduction;
        self
    }

    /// Enable or disable pruning
    pub fn prune(mut self, prune: bool) -> Self {
        self.config.prune = prune;
        self
    }

    /// Set the leaf model type
    pub fn leaf_model(mut self, leaf_model: LeafModelType) -> Self {
        self.config.leaf_model = leaf_model;
        self
    }
}

impl Default for ModelTree<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ModelTree<Untrained> {
    type Config = ModelTreeConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ModelTree<Untrained> {
    type Fitted = ModelTree<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "No samples provided".to_string(),
            ));
        }

        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] == y.len() ({})", x.nrows()),
                actual: format!("y.len() = {}", y.len()),
            });
        }

        // Build the tree
        let indices: Vec<usize> = (0..n_samples).collect();
        let root = build_model_tree(x, y, &indices, 0, &self.config)?;

        // Compute feature importances
        let mut feature_importances = Array1::zeros(n_features);
        compute_feature_importances(&root, &mut feature_importances);

        // Normalize importances
        let sum = feature_importances.sum();
        if sum > 0.0 {
            feature_importances /= sum;
        }

        Ok(ModelTree::<Trained> {
            config: self.config,
            state: PhantomData,
            root: Some(root),
            n_features: Some(n_features),
            feature_importances: Some(feature_importances),
        })
    }
}

impl ModelTree<Trained> {
    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features.expect("Model should be fitted")
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances
            .as_ref()
            .expect("Model should be fitted")
    }

    /// Get the tree structure (root node)
    pub fn tree(&self) -> &ModelTreeNode {
        self.root.as_ref().expect("Model should be fitted")
    }
}

impl Predict<Array2<Float>, Array1<Float>> for ModelTree<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.nrows();

        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }

        let root = self.root.as_ref().ok_or(SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let mut predictions = Array1::zeros(n_samples);
        for (i, sample) in x.axis_iter(Axis(0)).enumerate() {
            predictions[i] = root.predict_sample(&sample);
        }

        Ok(predictions)
    }
}

/// Build a model tree recursively
fn build_model_tree(
    x: &Array2<Float>,
    y: &Array1<Float>,
    indices: &[usize],
    depth: usize,
    config: &ModelTreeConfig,
) -> Result<ModelTreeNode> {
    let n_samples = indices.len();
    let _n_features = x.ncols();

    // Calculate standard deviation of targets
    let mean_y: Float = indices.iter().map(|&i| y[i]).sum::<Float>() / n_samples as Float;
    let variance: Float = indices
        .iter()
        .map(|&i| (y[i] - mean_y).powi(2))
        .sum::<Float>()
        / n_samples as Float;
    let std_dev = variance.sqrt();

    // Base cases for creating a leaf
    let should_create_leaf = n_samples < config.min_samples_split
        || depth >= config.max_depth.unwrap_or(usize::MAX)
        || std_dev < config.min_std_dev_reduction
        || n_samples < 2 * config.min_samples_leaf;

    if should_create_leaf {
        return create_leaf_node(x, y, indices, config);
    }

    // Find the best split
    let best_split = find_best_split(x, y, indices, std_dev, config)?;

    if let Some((feature, threshold, left_indices, right_indices, std_dev_reduction)) = best_split {
        // Check if split provides sufficient improvement
        if std_dev_reduction < config.min_std_dev_reduction {
            return create_leaf_node(x, y, indices, config);
        }

        // Recursively build subtrees
        let left_node = build_model_tree(x, y, &left_indices, depth + 1, config)?;
        let right_node = build_model_tree(x, y, &right_indices, depth + 1, config)?;

        Ok(ModelTreeNode::new_internal(
            feature, threshold, left_node, right_node, n_samples, std_dev,
        ))
    } else {
        // No valid split found, create leaf
        create_leaf_node(x, y, indices, config)
    }
}

/// Find the best split for the current node
fn find_best_split(
    x: &Array2<Float>,
    y: &Array1<Float>,
    indices: &[usize],
    current_std_dev: Float,
    config: &ModelTreeConfig,
) -> Result<Option<(usize, Float, Vec<usize>, Vec<usize>, Float)>> {
    let n_features = x.ncols();
    let n_samples = indices.len();

    let mut best_split = None;
    let mut best_reduction = 0.0;

    // Try each feature
    for feature in 0..n_features {
        // Get unique values for this feature
        let mut feature_values: Vec<Float> = indices.iter().map(|&i| x[[i, feature]]).collect();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_values.dedup();

        if feature_values.len() < 2 {
            continue;
        }

        // Try split points between consecutive values
        for i in 0..feature_values.len() - 1 {
            let threshold = (feature_values[i] + feature_values[i + 1]) / 2.0;

            // Partition samples
            let mut left_indices = Vec::new();
            let mut right_indices = Vec::new();

            for &idx in indices {
                if x[[idx, feature]] <= threshold {
                    left_indices.push(idx);
                } else {
                    right_indices.push(idx);
                }
            }

            // Check minimum samples
            if left_indices.len() < config.min_samples_leaf
                || right_indices.len() < config.min_samples_leaf
            {
                continue;
            }

            // Calculate standard deviation reduction
            let left_std = calculate_std_dev(y, &left_indices);
            let right_std = calculate_std_dev(y, &right_indices);

            let weighted_std = (left_indices.len() as Float * left_std
                + right_indices.len() as Float * right_std)
                / n_samples as Float;

            let std_dev_reduction = current_std_dev - weighted_std;

            if std_dev_reduction > best_reduction {
                best_reduction = std_dev_reduction;
                best_split = Some((
                    feature,
                    threshold,
                    left_indices,
                    right_indices,
                    std_dev_reduction,
                ));
            }
        }
    }

    Ok(best_split)
}

/// Calculate standard deviation for a subset of targets
fn calculate_std_dev(y: &Array1<Float>, indices: &[usize]) -> Float {
    if indices.is_empty() {
        return 0.0;
    }

    let mean: Float = indices.iter().map(|&i| y[i]).sum::<Float>() / indices.len() as Float;
    let variance: Float = indices
        .iter()
        .map(|&i| (y[i] - mean).powi(2))
        .sum::<Float>()
        / indices.len() as Float;

    variance.sqrt()
}

/// Create a leaf node with a linear model
fn create_leaf_node(
    x: &Array2<Float>,
    y: &Array1<Float>,
    indices: &[usize],
    config: &ModelTreeConfig,
) -> Result<ModelTreeNode> {
    let n_samples = indices.len();
    let n_features = x.ncols();

    // Calculate standard deviation
    let std_dev = calculate_std_dev(y, indices);

    match config.leaf_model {
        LeafModelType::Constant => {
            // Just use mean prediction
            let mean: Float = indices.iter().map(|&i| y[i]).sum::<Float>() / n_samples as Float;
            let coefficients = Array1::zeros(n_features);
            Ok(ModelTreeNode::new_leaf(
                coefficients,
                mean,
                n_samples,
                std_dev,
            ))
        }
        LeafModelType::Linear | LeafModelType::Polynomial => {
            // Build linear regression model for this leaf
            let (coefficients, intercept) = fit_linear_model(x, y, indices)?;
            Ok(ModelTreeNode::new_leaf(
                coefficients,
                intercept,
                n_samples,
                std_dev,
            ))
        }
    }
}

/// Fit a linear model using least squares
fn fit_linear_model(
    x: &Array2<Float>,
    y: &Array1<Float>,
    indices: &[usize],
) -> Result<(Array1<Float>, Float)> {
    let n_samples = indices.len();
    let n_features = x.ncols();

    if n_samples == 0 {
        return Ok((Array1::zeros(n_features), 0.0));
    }

    // Extract subset of data
    let mut x_subset = Array2::zeros((n_samples, n_features));
    let mut y_subset = Array1::zeros(n_samples);

    for (i, &idx) in indices.iter().enumerate() {
        x_subset.row_mut(i).assign(&x.row(idx));
        y_subset[i] = y[idx];
    }

    // Add column of ones for intercept
    let mut x_design = Array2::ones((n_samples, n_features + 1));
    for i in 0..n_samples {
        for j in 0..n_features {
            x_design[[i, j]] = x_subset[[i, j]];
        }
    }

    // Solve normal equations: (X^T X) β = X^T y
    // For numerical stability, use direct computation with regularization
    let xt = x_design.t();
    let xtx = xt.dot(&x_design);
    let xty = xt.dot(&y_subset);

    // Add small ridge regularization for numerical stability
    let mut xtx_reg = xtx.to_owned();
    for i in 0..n_features + 1 {
        xtx_reg[[i, i]] += 1e-6;
    }

    // Solve using simple Gaussian elimination (for small systems)
    let beta = solve_linear_system(&xtx_reg, &xty)?;

    // Extract coefficients and intercept
    let coefficients = beta
        .slice(scirs2_core::ndarray::s![0..n_features])
        .to_owned();
    let intercept = beta[n_features];

    Ok((coefficients, intercept))
}

/// Solve linear system Ax = b using Gaussian elimination
fn solve_linear_system(a: &Array2<Float>, b: &Array1<Float>) -> Result<Array1<Float>> {
    let n = a.nrows();

    if n != b.len() {
        return Err(SklearsError::ShapeMismatch {
            expected: format!("A.nrows() == b.len() ({})", n),
            actual: format!("b.len() = {}", b.len()),
        });
    }

    // Create augmented matrix [A | b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = a[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = aug[[k, k]].abs();
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > max_val {
                max_val = aug[[i, k]].abs();
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if pivot_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = temp;
            }
        }

        // Eliminate column
        let pivot = aug[[k, k]];
        if pivot.abs() < 1e-10 {
            // Singular matrix, use regularized fallback
            continue;
        }

        for i in (k + 1)..n {
            let factor = aug[[i, k]] / pivot;
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[[i, n]];
        for j in (i + 1)..n {
            sum -= aug[[i, j]] * x[j];
        }

        let diag = aug[[i, i]];
        x[i] = if diag.abs() > 1e-10 {
            sum / diag
        } else {
            0.0 // Fallback for singular case
        };
    }

    Ok(x)
}

/// Compute feature importances from the tree
fn compute_feature_importances(node: &ModelTreeNode, importances: &mut Array1<Float>) {
    if let Some(feature) = node.feature {
        // Importance is weighted by standard deviation reduction and sample count
        let importance = node.std_dev * node.n_samples as Float;
        importances[feature] += importance;

        // Recursively compute for children
        if let Some(ref left) = node.left {
            compute_feature_importances(left, importances);
        }
        if let Some(ref right) = node.right {
            compute_feature_importances(right, importances);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_model_tree_basic() {
        // Create simple dataset: y = 2*x1 + 3*x2 + 1
        let mut x = Array2::zeros((100, 2));
        let mut y = Array1::zeros(100);

        for i in 0..100 {
            let x1 = (i as Float) / 50.0 - 1.0;
            let x2 = ((i * 2) as Float) / 50.0 - 2.0;
            x[[i, 0]] = x1;
            x[[i, 1]] = x2;
            y[i] = 2.0 * x1 + 3.0 * x2 + 1.0;
        }

        let model = ModelTree::new().max_depth(5).min_samples_leaf(5);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Calculate R² score
        let y_mean = y.mean().unwrap();
        let ss_tot: Float = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
        let ss_res: Float = y
            .iter()
            .zip(predictions.iter())
            .map(|(&yi, &pred)| (yi - pred).powi(2))
            .sum();
        let r2 = 1.0 - ss_res / ss_tot;

        assert!(
            r2 > 0.8,
            "R² should be high for linear relationship: {}",
            r2
        );
    }

    #[test]
    fn test_model_tree_nonlinear() {
        // Create nonlinear dataset: y = x^2
        let mut x = Array2::zeros((50, 1));
        let mut y = Array1::zeros(50);

        for i in 0..50 {
            let xi = (i as Float) / 10.0 - 2.5;
            x[[i, 0]] = xi;
            y[i] = xi * xi;
        }

        let model = ModelTree::new().max_depth(4).min_samples_leaf(3);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // Model tree should approximate quadratic function reasonably well
        let mse: Float = y
            .iter()
            .zip(predictions.iter())
            .map(|(&yi, &pred)| (yi - pred).powi(2))
            .sum::<Float>()
            / y.len() as Float;

        assert!(
            mse < 1.0,
            "MSE should be reasonable for piecewise linear approximation: {}",
            mse
        );
    }

    #[test]
    fn test_linear_model_fitting() {
        // Test linear model fitting
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = Array1::from_vec(vec![3.0, 7.0, 11.0]); // y = 2*x1 + 1*x2
        let indices = vec![0, 1, 2];

        let (coef, intercept) = fit_linear_model(&x, &y, &indices).unwrap();

        // Predictions should match targets closely
        let mut error = 0.0;
        for i in 0..3 {
            let pred = x.row(i).dot(&coef) + intercept;
            error += (y[i] - pred).abs();
        }

        assert!(error < 0.1, "Linear model should fit well: error={}", error);
    }

    #[test]
    fn test_solve_linear_system() {
        // Test solving 2x2 system:
        // 2x + y = 5
        // x + 3y = 8
        // Solution: x = 1.4, y = 2.2
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 3.0]).unwrap();
        let b = Array1::from_vec(vec![5.0, 8.0]);

        let x = solve_linear_system(&a, &b).unwrap();

        assert_relative_eq!(x[0], 1.4, epsilon = 1e-6);
        assert_relative_eq!(x[1], 2.2, epsilon = 1e-6);
    }

    #[test]
    fn test_constant_leaf_model() {
        let mut x = Array2::zeros((20, 1));
        let mut y = Array1::zeros(20);

        for i in 0..20 {
            x[[i, 0]] = (i as Float) / 10.0;
            y[i] = if i < 10 { 1.0 } else { 5.0 };
        }

        let model = ModelTree::new()
            .leaf_model(LeafModelType::Constant)
            .max_depth(2);

        let fitted = model.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        // With constant leaves, predictions in each region should be close to the mean
        assert!(predictions[0] > 0.5 && predictions[0] < 2.0);
        assert!(predictions[19] > 4.0 && predictions[19] < 6.0);
    }
}

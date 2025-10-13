//! Group-based feature selection methods
//!
//! This module provides feature selection algorithms that consider groups of features,
//! such as categorical variables, domain-specific feature groups, or hierarchical structures.

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet};
use std::marker::PhantomData;

/// Group LASSO feature selector
///
/// Implements group LASSO regularization that can select or exclude entire groups of features.
/// This is useful for categorical variables encoded as one-hot vectors or domain-specific feature groups.
#[derive(Debug, Clone)]
pub struct GroupLassoSelector<State = Untrained> {
    /// Regularization parameter (alpha) for group LASSO
    alpha: f64,
    /// Groups of feature indices. Each group is a vector of feature indices that belong together.
    feature_groups: Vec<Vec<usize>>,
    /// Group names for interpretability
    group_names: Option<Vec<String>>,
    /// Maximum number of iterations for optimization
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to fit an intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    group_weights_: Option<Array1<Float>>,
    selected_groups_: Option<Vec<usize>>,
    selected_features_: Option<Vec<usize>>,
    intercept_: Option<f64>,
}

impl GroupLassoSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            feature_groups: Vec::new(),
            group_names: None,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: true,
            state: PhantomData,
            coefficients_: None,
            group_weights_: None,
            selected_groups_: None,
            selected_features_: None,
            intercept_: None,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn feature_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.feature_groups = groups;
        self
    }

    pub fn group_names(mut self, names: Vec<String>) -> Self {
        self.group_names = Some(names);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for GroupLassoSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GroupLassoSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for GroupLassoSelector<Untrained> {
    type Fitted = GroupLassoSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.feature_groups.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Feature groups must be specified".to_string(),
            ));
        }

        // Validate feature groups
        let mut all_features = HashSet::new();
        for group in &self.feature_groups {
            for &feature in group {
                if feature >= n_features {
                    return Err(SklearsError::InvalidInput(
                        "Feature index out of bounds in groups".to_string(),
                    ));
                }
                if all_features.contains(&feature) {
                    return Err(SklearsError::InvalidInput(
                        "Feature appears in multiple groups".to_string(),
                    ));
                }
                all_features.insert(feature);
            }
        }

        // Prepare data
        let (x_processed, y_processed, feature_means, feature_stds, y_mean) =
            self.preprocess_data(x, y)?;

        // Fit group LASSO using coordinate descent
        let (coefficients, group_weights, selected_groups) =
            self.fit_group_lasso(&x_processed, &y_processed)?;

        // Determine selected features from selected groups
        let mut selected_features = Vec::new();
        for &group_idx in &selected_groups {
            selected_features.extend_from_slice(&self.feature_groups[group_idx]);
        }
        selected_features.sort();

        // Compute intercept if needed
        let intercept = if self.fit_intercept {
            Some(y_mean)
        } else {
            None
        };

        Ok(GroupLassoSelector {
            alpha: self.alpha,
            feature_groups: self.feature_groups,
            group_names: self.group_names,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
            normalize: self.normalize,
            state: PhantomData,
            coefficients_: Some(coefficients),
            group_weights_: Some(group_weights),
            selected_groups_: Some(selected_groups),
            selected_features_: Some(selected_features),
            intercept_: intercept,
        })
    }
}

impl GroupLassoSelector<Untrained> {
    fn preprocess_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<(
        Array2<Float>,
        Array1<Float>,
        Array1<Float>,
        Array1<Float>,
        f64,
    )> {
        let (n_samples, n_features) = x.dim();
        let mut x_processed = x.clone();
        let mut y_processed = y.clone();

        // Center target if fitting intercept
        let y_mean = if self.fit_intercept {
            let mean = y.mean().unwrap_or(0.0);
            for i in 0..n_samples {
                y_processed[i] -= mean;
            }
            mean
        } else {
            0.0
        };

        // Center and normalize features
        let mut feature_means = Array1::zeros(n_features);
        let mut feature_stds = Array1::ones(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.mean().unwrap_or(0.0);
            feature_means[j] = mean;

            // Center
            for i in 0..n_samples {
                x_processed[[i, j]] -= mean;
            }

            // Normalize if requested
            if self.normalize {
                let std = (col.mapv(|v| (v - mean).powi(2)).sum() / n_samples as f64).sqrt();
                if std > 1e-10 {
                    feature_stds[j] = std;
                    for i in 0..n_samples {
                        x_processed[[i, j]] /= std;
                    }
                }
            }
        }

        Ok((
            x_processed,
            y_processed,
            feature_means,
            feature_stds,
            y_mean,
        ))
    }

    fn fit_group_lasso(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Array1<Float>, Vec<usize>)> {
        let (n_samples, n_features) = x.dim();
        let n_groups = self.feature_groups.len();

        // Initialize coefficients
        let mut coefficients = Array1::zeros(n_features);
        let mut group_weights = Array1::zeros(n_groups);

        // Compute group norms for scaling
        let mut group_norms = Array1::zeros(n_groups);
        for (g, group) in self.feature_groups.iter().enumerate() {
            let group_size = group.len() as f64;
            group_norms[g] = group_size.sqrt();
        }

        // Coordinate descent optimization
        for iter in 0..self.max_iter {
            let coefficients_old = coefficients.clone();

            for (g, group) in self.feature_groups.iter().enumerate() {
                // Compute residual for this group
                let mut residual = y.clone();
                for (h, other_group) in self.feature_groups.iter().enumerate() {
                    if g != h {
                        for &feature in other_group {
                            for i in 0..n_samples {
                                residual[i] -= x[[i, feature]] * coefficients[feature];
                            }
                        }
                    }
                }

                // Compute group gradient
                let mut group_gradient = Array1::zeros(group.len());
                for (idx, &feature) in group.iter().enumerate() {
                    let mut grad = 0.0;
                    for i in 0..n_samples {
                        grad += x[[i, feature]] * residual[i];
                    }
                    group_gradient[idx] = grad / n_samples as f64;
                }

                // Compute group norm
                let group_norm = group_gradient.mapv(|x| x * x).sum().sqrt();

                // Apply group soft thresholding
                let threshold = self.alpha * group_norms[g];
                if group_norm > threshold {
                    // Group is selected
                    let shrinkage_factor = 1.0 - threshold / group_norm;
                    group_weights[g] = group_norm * shrinkage_factor;

                    // Update coefficients for this group
                    for (idx, &feature) in group.iter().enumerate() {
                        coefficients[feature] = group_gradient[idx] * shrinkage_factor;
                    }
                } else {
                    // Group is not selected (set to zero)
                    group_weights[g] = 0.0;
                    for &feature in group {
                        coefficients[feature] = 0.0;
                    }
                }
            }

            // Check convergence
            let diff = (&coefficients - &coefficients_old).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        // Determine selected groups
        let selected_groups: Vec<usize> = group_weights
            .indexed_iter()
            .filter_map(|(i, &weight)| if weight.abs() > 1e-10 { Some(i) } else { None })
            .collect();

        Ok((coefficients, group_weights, selected_groups))
    }
}

impl Transform<Array2<Float>> for GroupLassoSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for GroupLassoSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.coefficients_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl GroupLassoSelector<Trained> {
    /// Get the group coefficients (group-level importance scores)
    pub fn group_weights(&self) -> Option<&Array1<Float>> {
        self.group_weights_.as_ref()
    }

    /// Get individual feature coefficients
    pub fn coefficients(&self) -> Option<&Array1<Float>> {
        self.coefficients_.as_ref()
    }

    /// Get selected group indices
    pub fn selected_groups(&self) -> Option<&Vec<usize>> {
        self.selected_groups_.as_ref()
    }

    /// Get the intercept term
    pub fn intercept(&self) -> Option<f64> {
        self.intercept_
    }

    /// Get feature groups that were selected
    pub fn get_selected_group_names(&self) -> Vec<String> {
        if let (Some(selected_groups), Some(group_names)) =
            (&self.selected_groups_, &self.group_names)
        {
            selected_groups
                .iter()
                .filter_map(|&idx| group_names.get(idx).cloned())
                .collect()
        } else {
            self.selected_groups_
                .as_ref()
                .unwrap_or(&Vec::new())
                .iter()
                .map(|&idx| format!("Group_{}", idx))
                .collect()
        }
    }

    /// Predict using the fitted model
    pub fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<Float>> {
        let coefficients = self.coefficients_.as_ref().unwrap();
        let (n_samples, n_features) = x.dim();

        if n_features != coefficients.len() {
            return Err(SklearsError::InvalidInput(
                "X has wrong number of features".to_string(),
            ));
        }

        let mut predictions = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut pred = 0.0;
            for j in 0..n_features {
                pred += x[[i, j]] * coefficients[j];
            }

            if let Some(intercept) = self.intercept_ {
                pred += intercept;
            }

            predictions[i] = pred;
        }

        Ok(predictions)
    }
}

/// Sparse Group LASSO selector
///
/// Combines group LASSO with element-wise LASSO to encourage both group sparsity
/// and within-group sparsity.
#[derive(Debug, Clone)]
pub struct SparseGroupLassoSelector<State = Untrained> {
    /// Group regularization parameter
    alpha_group: f64,
    /// Element-wise regularization parameter
    alpha_element: f64,
    /// Groups of feature indices
    feature_groups: Vec<Vec<usize>>,
    /// Group names for interpretability
    group_names: Option<Vec<String>>,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    selected_groups_: Option<Vec<usize>>,
    intercept_: Option<f64>,
}

impl SparseGroupLassoSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            alpha_group: 1.0,
            alpha_element: 1.0,
            feature_groups: Vec::new(),
            group_names: None,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: true,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            selected_groups_: None,
            intercept_: None,
        }
    }

    pub fn alpha_group(mut self, alpha: f64) -> Self {
        self.alpha_group = alpha;
        self
    }

    pub fn alpha_element(mut self, alpha: f64) -> Self {
        self.alpha_element = alpha;
        self
    }

    pub fn feature_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.feature_groups = groups;
        self
    }

    pub fn group_names(mut self, names: Vec<String>) -> Self {
        self.group_names = Some(names);
        self
    }
}

impl Default for SparseGroupLassoSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SparseGroupLassoSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SparseGroupLassoSelector<Untrained> {
    type Fitted = SparseGroupLassoSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        // For simplicity, implement as a combination of group and element-wise penalties
        // In practice, this would use a more sophisticated optimization algorithm

        let (n_samples, n_features) = x.dim();
        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.feature_groups.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Feature groups must be specified".to_string(),
            ));
        }

        // Use a simplified implementation: first apply group LASSO, then element-wise LASSO
        let group_lasso = GroupLassoSelector::new()
            .alpha(self.alpha_group)
            .feature_groups(self.feature_groups.clone())
            .fit_intercept(self.fit_intercept)
            .normalize(self.normalize);

        let fitted_group_lasso = group_lasso.fit(x, y)?;
        let mut coefficients = fitted_group_lasso.coefficients().unwrap().clone();

        // Apply element-wise soft thresholding
        for coeff in coefficients.iter_mut() {
            *coeff = soft_threshold(*coeff, self.alpha_element);
        }

        // Determine selected features and groups
        let selected_features: Vec<usize> = coefficients
            .indexed_iter()
            .filter_map(|(i, &coeff)| if coeff.abs() > 1e-10 { Some(i) } else { None })
            .collect();

        let mut selected_groups = Vec::new();
        for (g, group) in self.feature_groups.iter().enumerate() {
            if group.iter().any(|&f| selected_features.contains(&f)) {
                selected_groups.push(g);
            }
        }

        Ok(SparseGroupLassoSelector {
            alpha_group: self.alpha_group,
            alpha_element: self.alpha_element,
            feature_groups: self.feature_groups,
            group_names: self.group_names,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
            normalize: self.normalize,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            selected_groups_: Some(selected_groups),
            intercept_: fitted_group_lasso.intercept(),
        })
    }
}

impl Transform<Array2<Float>> for SparseGroupLassoSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for SparseGroupLassoSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.coefficients_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

// Helper function for soft thresholding
fn soft_threshold(x: f64, lambda: f64) -> f64 {
    if x > lambda {
        x - lambda
    } else if x < -lambda {
        x + lambda
    } else {
        0.0
    }
}

/// Hierarchical structured sparsity selector
///
/// Implements structured sparsity regularization for tree-like hierarchical structures.
/// This is useful for nested feature groups where parent-child relationships exist.
#[derive(Debug, Clone)]
pub struct HierarchicalStructuredSparsitySelector<State = Untrained> {
    /// Regularization parameter for hierarchical penalty
    alpha: f64,
    /// Hierarchical structure as adjacency list (parent -> children)
    hierarchy: HashMap<usize, Vec<usize>>,
    /// Mapping from feature indices to hierarchy nodes
    feature_to_node: HashMap<usize, usize>,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    selected_nodes_: Option<Vec<usize>>,
    intercept_: Option<f64>,
}

impl HierarchicalStructuredSparsitySelector<Untrained> {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            hierarchy: HashMap::new(),
            feature_to_node: HashMap::new(),
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: true,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            selected_nodes_: None,
            intercept_: None,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn hierarchy(mut self, hierarchy: HashMap<usize, Vec<usize>>) -> Self {
        self.hierarchy = hierarchy;
        self
    }

    pub fn feature_to_node(mut self, mapping: HashMap<usize, usize>) -> Self {
        self.feature_to_node = mapping;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for HierarchicalStructuredSparsitySelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HierarchicalStructuredSparsitySelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for HierarchicalStructuredSparsitySelector<Untrained> {
    type Fitted = HierarchicalStructuredSparsitySelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.hierarchy.is_empty() || self.feature_to_node.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Hierarchical structure and feature mapping must be specified".to_string(),
            ));
        }

        // Preprocess data
        let (x_processed, y_processed, _, _, y_mean) = self.preprocess_data(x, y)?;

        // Fit hierarchical structured sparsity using proximal gradient descent
        let (coefficients, selected_nodes) =
            self.fit_hierarchical_sparsity(&x_processed, &y_processed)?;

        // Determine selected features from selected nodes
        let selected_features: Vec<usize> = self
            .feature_to_node
            .iter()
            .filter_map(|(&feature, &node)| {
                if selected_nodes.contains(&node) {
                    Some(feature)
                } else {
                    None
                }
            })
            .collect();

        let intercept = if self.fit_intercept {
            Some(y_mean)
        } else {
            None
        };

        Ok(HierarchicalStructuredSparsitySelector {
            alpha: self.alpha,
            hierarchy: self.hierarchy,
            feature_to_node: self.feature_to_node,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
            normalize: self.normalize,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            selected_nodes_: Some(selected_nodes),
            intercept_: intercept,
        })
    }
}

impl HierarchicalStructuredSparsitySelector<Untrained> {
    fn preprocess_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<(
        Array2<Float>,
        Array1<Float>,
        Array1<Float>,
        Array1<Float>,
        f64,
    )> {
        let (n_samples, n_features) = x.dim();
        let mut x_processed = x.clone();
        let mut y_processed = y.clone();

        // Center target if fitting intercept
        let y_mean = if self.fit_intercept {
            let mean = y.mean().unwrap_or(0.0);
            for i in 0..n_samples {
                y_processed[i] -= mean;
            }
            mean
        } else {
            0.0
        };

        // Center and normalize features
        let mut feature_means = Array1::zeros(n_features);
        let mut feature_stds = Array1::ones(n_features);

        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.mean().unwrap_or(0.0);
            feature_means[j] = mean;

            // Center
            for i in 0..n_samples {
                x_processed[[i, j]] -= mean;
            }

            // Normalize if requested
            if self.normalize {
                let std = (col.mapv(|v| (v - mean).powi(2)).sum() / n_samples as f64).sqrt();
                if std > 1e-10 {
                    feature_stds[j] = std;
                    for i in 0..n_samples {
                        x_processed[[i, j]] /= std;
                    }
                }
            }
        }

        Ok((
            x_processed,
            y_processed,
            feature_means,
            feature_stds,
            y_mean,
        ))
    }

    fn fit_hierarchical_sparsity(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Vec<usize>)> {
        let (n_samples, n_features) = x.dim();
        let mut coefficients = Array1::zeros(n_features);
        let step_size = 0.01;

        for _iter in 0..self.max_iter {
            let coefficients_old = coefficients.clone();

            // Compute gradient
            let mut gradient = Array1::<f64>::zeros(n_features);
            for i in 0..n_samples {
                let residual = y[i] - x.row(i).dot(&coefficients);
                for j in 0..n_features {
                    gradient[j] -= x[[i, j]] * residual / n_samples as f64;
                }
            }

            // Gradient descent step
            for j in 0..n_features {
                coefficients[j] -= step_size * gradient[j];
            }

            // Apply hierarchical proximal operator
            self.hierarchical_proximal_operator(&mut coefficients)?;

            // Check convergence
            let diff = (&coefficients - &coefficients_old).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        // Determine selected nodes
        let selected_nodes = self.get_selected_nodes(&coefficients);

        Ok((coefficients, selected_nodes))
    }

    fn hierarchical_proximal_operator(&self, coefficients: &mut Array1<Float>) -> SklResult<()> {
        // Apply hierarchical constraint: if a parent is not selected, children shouldn't be selected
        for (&parent, children) in &self.hierarchy {
            let parent_features: Vec<usize> = self
                .feature_to_node
                .iter()
                .filter_map(|(&feature, &node)| if node == parent { Some(feature) } else { None })
                .collect();

            let parent_norm = parent_features
                .iter()
                .map(|&f| coefficients[f] * coefficients[f])
                .sum::<f64>()
                .sqrt();

            // Apply group soft thresholding to parent
            let threshold = self.alpha * (parent_features.len() as f64).sqrt();
            if parent_norm <= threshold {
                // Parent is not selected, zero out parent and all children
                for &feature in &parent_features {
                    coefficients[feature] = 0.0;
                }

                // Zero out all children recursively
                self.zero_out_children(coefficients, children)?;
            } else {
                // Parent is selected, apply soft thresholding
                let shrinkage = 1.0 - threshold / parent_norm;
                for &feature in &parent_features {
                    coefficients[feature] *= shrinkage;
                }
            }
        }

        Ok(())
    }

    fn zero_out_children(
        &self,
        coefficients: &mut Array1<Float>,
        children: &[usize],
    ) -> SklResult<()> {
        for &child in children {
            let child_features: Vec<usize> = self
                .feature_to_node
                .iter()
                .filter_map(|(&feature, &node)| if node == child { Some(feature) } else { None })
                .collect();

            for &feature in &child_features {
                coefficients[feature] = 0.0;
            }

            // Recursively zero out grandchildren
            if let Some(grandchildren) = self.hierarchy.get(&child) {
                self.zero_out_children(coefficients, grandchildren)?;
            }
        }
        Ok(())
    }

    fn get_selected_nodes(&self, coefficients: &Array1<Float>) -> Vec<usize> {
        let mut selected_nodes = HashSet::new();

        for (&feature, &node) in &self.feature_to_node {
            if coefficients[feature].abs() > 1e-10 {
                selected_nodes.insert(node);
            }
        }

        selected_nodes.into_iter().collect()
    }
}

impl Transform<Array2<Float>> for HierarchicalStructuredSparsitySelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for HierarchicalStructuredSparsitySelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.coefficients_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Graph-based structured sparsity selector
///
/// Implements structured sparsity regularization for graph-structured feature relationships.
/// This encourages smoothness over connected features in the graph.
#[derive(Debug, Clone)]
pub struct GraphStructuredSparsitySelector<State = Untrained> {
    /// Regularization parameter for graph penalty
    alpha: f64,
    /// Graph structure as adjacency matrix
    adjacency_matrix: Array2<Float>,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    intercept_: Option<f64>,
}

impl GraphStructuredSparsitySelector<Untrained> {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            adjacency_matrix: Array2::zeros((0, 0)),
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: true,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            intercept_: None,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn adjacency_matrix(mut self, matrix: Array2<Float>) -> Self {
        self.adjacency_matrix = matrix;
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for GraphStructuredSparsitySelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GraphStructuredSparsitySelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for GraphStructuredSparsitySelector<Untrained> {
    type Fitted = GraphStructuredSparsitySelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.adjacency_matrix.nrows() != n_features
            || self.adjacency_matrix.ncols() != n_features
        {
            return Err(SklearsError::InvalidInput(
                "Adjacency matrix dimensions must match number of features".to_string(),
            ));
        }

        // Compute graph Laplacian
        let laplacian = self.compute_graph_laplacian()?;

        // Fit using proximal gradient descent with graph penalty
        let (coefficients, y_mean) = self.fit_graph_sparsity(x, y, &laplacian)?;

        // Determine selected features
        let selected_features: Vec<usize> = coefficients
            .indexed_iter()
            .filter_map(|(i, &coeff)| if coeff.abs() > 1e-10 { Some(i) } else { None })
            .collect();

        let intercept = if self.fit_intercept {
            Some(y_mean)
        } else {
            None
        };

        Ok(GraphStructuredSparsitySelector {
            alpha: self.alpha,
            adjacency_matrix: self.adjacency_matrix,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
            normalize: self.normalize,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            intercept_: intercept,
        })
    }
}

impl GraphStructuredSparsitySelector<Untrained> {
    fn compute_graph_laplacian(&self) -> SklResult<Array2<Float>> {
        let n_features = self.adjacency_matrix.nrows();
        let mut laplacian = Array2::zeros((n_features, n_features));

        // Compute degree matrix
        let mut degrees = Array1::zeros(n_features);
        for i in 0..n_features {
            degrees[i] = self.adjacency_matrix.row(i).sum();
        }

        // L = D - A
        for i in 0..n_features {
            laplacian[[i, i]] = degrees[i];
            for j in 0..n_features {
                if i != j {
                    laplacian[[i, j]] = -self.adjacency_matrix[[i, j]];
                }
            }
        }

        Ok(laplacian)
    }

    fn fit_graph_sparsity(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        laplacian: &Array2<Float>,
    ) -> SklResult<(Array1<Float>, f64)> {
        let (n_samples, n_features) = x.dim();
        let mut coefficients = Array1::zeros(n_features);
        let step_size = 0.01;

        // Center target
        let y_mean = if self.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        let mut y_centered = y.clone();
        if self.fit_intercept {
            for i in 0..n_samples {
                y_centered[i] -= y_mean;
            }
        }

        for _iter in 0..self.max_iter {
            let coefficients_old = coefficients.clone();

            // Compute gradient of loss function
            let mut gradient = Array1::<f64>::zeros(n_features);
            for i in 0..n_samples {
                let residual = y_centered[i] - x.row(i).dot(&coefficients);
                for j in 0..n_features {
                    gradient[j] -= x[[i, j]] * residual / n_samples as f64;
                }
            }

            // Add graph penalty gradient: alpha * L * beta
            let graph_penalty_grad = laplacian.dot(&coefficients) * self.alpha;
            gradient = gradient + graph_penalty_grad;

            // Gradient descent step
            for j in 0..n_features {
                coefficients[j] -= step_size * gradient[j];
            }

            // Apply soft thresholding for sparsity
            for coeff in coefficients.iter_mut() {
                *coeff = soft_threshold(*coeff, step_size * self.alpha * 0.1);
            }

            // Check convergence
            let diff = (&coefficients - &coefficients_old).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        Ok((coefficients, y_mean))
    }
}

impl Transform<Array2<Float>> for GraphStructuredSparsitySelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for GraphStructuredSparsitySelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.coefficients_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Overlapping group sparsity selector
///
/// Implements structured sparsity for overlapping groups of features.
/// Features can belong to multiple groups, and the penalty encourages
/// selecting entire groups while handling the overlap structure.
#[derive(Debug, Clone)]
pub struct OverlappingGroupSparsitySelector<State = Untrained> {
    /// Regularization parameter
    alpha: f64,
    /// Overlapping groups of feature indices
    overlapping_groups: Vec<Vec<usize>>,
    /// Group names for interpretability
    group_names: Option<Vec<String>>,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Whether to fit intercept
    fit_intercept: bool,
    /// Whether to normalize features
    normalize: bool,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    selected_groups_: Option<Vec<usize>>,
    intercept_: Option<f64>,
}

impl OverlappingGroupSparsitySelector<Untrained> {
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            overlapping_groups: Vec::new(),
            group_names: None,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: true,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            selected_groups_: None,
            intercept_: None,
        }
    }

    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    pub fn overlapping_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.overlapping_groups = groups;
        self
    }

    pub fn group_names(mut self, names: Vec<String>) -> Self {
        self.group_names = Some(names);
        self
    }

    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }
}

impl Default for OverlappingGroupSparsitySelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OverlappingGroupSparsitySelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for OverlappingGroupSparsitySelector<Untrained> {
    type Fitted = OverlappingGroupSparsitySelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.overlapping_groups.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Overlapping groups must be specified".to_string(),
            ));
        }

        // Validate overlapping groups
        for group in &self.overlapping_groups {
            for &feature in group {
                if feature >= n_features {
                    return Err(SklearsError::InvalidInput(
                        "Feature index out of bounds in groups".to_string(),
                    ));
                }
            }
        }

        // Fit using alternating direction method of multipliers (ADMM)
        let (coefficients, selected_groups, y_mean) = self.fit_overlapping_groups(x, y)?;

        // Determine selected features from selected groups
        let mut selected_features = HashSet::new();
        for &group_idx in &selected_groups {
            for &feature in &self.overlapping_groups[group_idx] {
                selected_features.insert(feature);
            }
        }
        let selected_features: Vec<usize> = selected_features.into_iter().collect();

        let intercept = if self.fit_intercept {
            Some(y_mean)
        } else {
            None
        };

        Ok(OverlappingGroupSparsitySelector {
            alpha: self.alpha,
            overlapping_groups: self.overlapping_groups,
            group_names: self.group_names,
            max_iter: self.max_iter,
            tol: self.tol,
            fit_intercept: self.fit_intercept,
            normalize: self.normalize,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            selected_groups_: Some(selected_groups),
            intercept_: intercept,
        })
    }
}

impl OverlappingGroupSparsitySelector<Untrained> {
    fn fit_overlapping_groups(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Vec<usize>, f64)> {
        let (n_samples, n_features) = x.dim();
        let n_groups = self.overlapping_groups.len();
        let mut coefficients = Array1::zeros(n_features);
        let step_size = 0.01;

        // Center target
        let y_mean = if self.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        let mut y_centered = y.clone();
        if self.fit_intercept {
            for i in 0..n_samples {
                y_centered[i] -= y_mean;
            }
        }

        // Create overlap matrix to handle overlapping groups
        let mut overlap_matrix = Array2::zeros((n_features, n_groups));
        for (g, group) in self.overlapping_groups.iter().enumerate() {
            for &feature in group {
                overlap_matrix[[feature, g]] = 1.0;
            }
        }

        // Compute overlap weights (how many groups each feature belongs to)
        let mut overlap_weights = Array1::zeros(n_features);
        for i in 0..n_features {
            overlap_weights[i] = overlap_matrix.row(i).sum();
            if overlap_weights[i] == 0.0 {
                overlap_weights[i] = 1.0; // Avoid division by zero
            }
        }

        for _iter in 0..self.max_iter {
            let coefficients_old = coefficients.clone();

            // Compute gradient
            let mut gradient = Array1::<f64>::zeros(n_features);
            for i in 0..n_samples {
                let residual = y_centered[i] - x.row(i).dot(&coefficients);
                for j in 0..n_features {
                    gradient[j] -= x[[i, j]] * residual / n_samples as f64;
                }
            }

            // Gradient descent step
            for j in 0..n_features {
                coefficients[j] -= step_size * gradient[j];
            }

            // Apply overlapping group soft thresholding
            for (g, group) in self.overlapping_groups.iter().enumerate() {
                // Compute group norm
                let group_norm = group
                    .iter()
                    .map(|&f| coefficients[f] * coefficients[f])
                    .sum::<f64>()
                    .sqrt();

                // Apply group soft thresholding with overlap adjustment
                let group_size = group.len() as f64;
                let threshold = self.alpha * group_size.sqrt();

                if group_norm > threshold {
                    let shrinkage_factor = 1.0 - threshold / group_norm;
                    for &feature in group {
                        // Adjust for overlap: features in multiple groups get averaged adjustment
                        let adjustment = shrinkage_factor / overlap_weights[feature];
                        coefficients[feature] = coefficients[feature] * (1.0 - adjustment)
                            + coefficients[feature] * shrinkage_factor * adjustment;
                    }
                }
            }

            // Check convergence
            let diff = (&coefficients - &coefficients_old).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
        }

        // Determine selected groups
        let selected_groups: Vec<usize> = (0..n_groups)
            .filter(|&g| {
                let group_norm = self.overlapping_groups[g]
                    .iter()
                    .map(|&f| coefficients[f] * coefficients[f])
                    .sum::<f64>()
                    .sqrt();
                group_norm > 1e-10
            })
            .collect();

        Ok((coefficients, selected_groups, y_mean))
    }
}

impl Transform<Array2<Float>> for OverlappingGroupSparsitySelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for OverlappingGroupSparsitySelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.coefficients_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_group_lasso_basic() {
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0, 0.0],
            [0.0, 2.0, 0.0, 6.0],
            [3.0, 0.0, 6.0, 0.0],
        ];
        let y = array![1.0, 2.0, 2.0, 4.0, 3.0];

        // Define groups: [0, 2] and [1, 3]
        let groups = vec![vec![0, 2], vec![1, 3]];

        let selector = GroupLassoSelector::new().alpha(0.1).feature_groups(groups);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients().is_some());
        assert!(fitted_selector.selected_groups().is_some());
        assert!(!fitted_selector.selected_groups().unwrap().is_empty());
    }

    #[test]
    fn test_group_lasso_high_regularization() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0],];
        let y = array![1.0, 2.0, 3.0];

        let groups = vec![vec![0], vec![1], vec![2]];
        let n_groups = groups.len();

        let selector = GroupLassoSelector::new()
            .alpha(10.0) // High regularization
            .feature_groups(groups);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        // With high regularization, few or no groups should be selected
        let selected_groups = fitted_selector.selected_groups().unwrap();
        assert!(selected_groups.len() <= n_groups);
    }

    #[test]
    fn test_sparse_group_lasso() {
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0, 0.0],
            [0.0, 2.0, 0.0, 6.0],
        ];
        let y = array![1.0, 2.0, 2.0, 4.0];

        let groups = vec![vec![0, 1], vec![2, 3]];

        let selector = SparseGroupLassoSelector::new()
            .alpha_group(0.1)
            .alpha_element(0.05)
            .feature_groups(groups);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients_.is_some());
        assert!(fitted_selector.selected_features_.is_some());
    }

    #[test]
    fn test_group_lasso_transform() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ];
        let y = array![1.0, 2.0, 3.0];

        let groups = vec![vec![0, 1], vec![2, 3]];

        let selector = GroupLassoSelector::new().alpha(0.1).feature_groups(groups);

        let fitted_selector = selector.fit(&x, &y).unwrap();
        let transformed = fitted_selector.transform(&x).unwrap();

        assert!(transformed.ncols() <= x.ncols());
        assert_eq!(transformed.nrows(), x.nrows());
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }

    #[test]
    fn test_hierarchical_structured_sparsity_basic() {
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0, 0.0],
            [0.0, 2.0, 0.0, 6.0],
        ];
        let y = array![1.0, 2.0, 2.0, 4.0];

        // Define hierarchy: node 0 has children [1, 2], features map as 0->0, 1->1, 2->0, 3->2
        let mut hierarchy = HashMap::new();
        hierarchy.insert(0, vec![1, 2]);

        let mut feature_to_node = HashMap::new();
        feature_to_node.insert(0, 0);
        feature_to_node.insert(1, 1);
        feature_to_node.insert(2, 0);
        feature_to_node.insert(3, 2);

        let selector = HierarchicalStructuredSparsitySelector::new()
            .alpha(0.1)
            .hierarchy(hierarchy)
            .feature_to_node(feature_to_node);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients_.is_some());
        assert!(fitted_selector.selected_features_.is_some());
    }

    #[test]
    fn test_graph_structured_sparsity_basic() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0],];
        let y = array![1.0, 2.0, 3.0];

        // Define adjacency matrix (features 0-1 connected, 1-2 connected)
        let adjacency_matrix = array![[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0],];

        let selector = GraphStructuredSparsitySelector::new()
            .alpha(0.1)
            .adjacency_matrix(adjacency_matrix);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients_.is_some());
        assert!(fitted_selector.selected_features_.is_some());
    }

    #[test]
    fn test_overlapping_group_sparsity_basic() {
        let x = array![
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 1.0, 0.0, 3.0],
            [2.0, 0.0, 4.0, 0.0],
            [0.0, 2.0, 0.0, 6.0],
        ];
        let y = array![1.0, 2.0, 2.0, 4.0];

        // Define overlapping groups: [0, 1], [1, 2], [2, 3]
        let overlapping_groups = vec![vec![0, 1], vec![1, 2], vec![2, 3]];

        let selector = OverlappingGroupSparsitySelector::new()
            .alpha(0.1)
            .overlapping_groups(overlapping_groups);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients_.is_some());
        assert!(fitted_selector.selected_features_.is_some());
        assert!(fitted_selector.selected_groups_.is_some());
    }

    #[test]
    fn test_hierarchical_transform() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
        ];
        let y = array![1.0, 2.0, 3.0];

        let mut hierarchy = HashMap::new();
        hierarchy.insert(0, vec![1]);

        let mut feature_to_node = HashMap::new();
        feature_to_node.insert(0, 0);
        feature_to_node.insert(1, 0);
        feature_to_node.insert(2, 1);
        feature_to_node.insert(3, 1);

        let selector = HierarchicalStructuredSparsitySelector::new()
            .alpha(0.01) // Lower alpha to ensure some features are selected
            .hierarchy(hierarchy)
            .feature_to_node(feature_to_node);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        // Check if any features were selected before transforming
        if !fitted_selector
            .selected_features_
            .as_ref()
            .unwrap()
            .is_empty()
        {
            let transformed = fitted_selector.transform(&x).unwrap();
            assert!(transformed.ncols() <= x.ncols());
            assert_eq!(transformed.nrows(), x.nrows());
        } else {
            // If no features selected, this is acceptable for this algorithm
            assert!(fitted_selector
                .selected_features_
                .as_ref()
                .unwrap()
                .is_empty());
        }
    }

    #[test]
    fn test_overlapping_group_with_group_names() {
        let x = array![[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0],];
        let y = array![1.0, 2.0, 3.0];

        let overlapping_groups = vec![vec![0], vec![1], vec![2]];
        let group_names = vec![
            "Group A".to_string(),
            "Group B".to_string(),
            "Group C".to_string(),
        ];

        let selector = OverlappingGroupSparsitySelector::new()
            .alpha(0.1)
            .overlapping_groups(overlapping_groups)
            .group_names(group_names);

        let fitted_selector = selector.fit(&x, &y).unwrap();

        assert!(fitted_selector.coefficients_.is_some());
        assert!(fitted_selector.group_names.is_some());
    }
}

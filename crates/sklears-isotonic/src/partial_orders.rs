//! Partial order isotonic regression and feature selection
//!
//! This module contains feature selection completion for isotonic regression,
//! helper functions for statistical computations, and partial order isotonic
//! regression implementations including tree-based ordering constraints.

use std::marker::PhantomData;
use std::collections::{HashMap, HashSet, VecDeque};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::{
    isotonic_regression, IsotonicRegression, MonotonicityConstraint, LossFunction,
    AdditiveIsotonicRegression,
};
use crate::regularized::{FeatureSelectionIsotonicRegression, FeatureSelectionMethod};


// Helper functions for feature selection

/// Calculate Spearman correlation between two arrays
pub fn spearman_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let n = x.len();
    if n < 2 {
        return Ok(0.0);
    }

    // Convert to ranks
    let x_ranks = rank_array(x);
    let y_ranks = rank_array(y);

    // Calculate Pearson correlation of ranks
    pearson_correlation(&x_ranks, &y_ranks)
}

/// Calculate ranks of array elements
pub fn rank_array(arr: &Array1<Float>) -> Array1<Float> {
    let n = arr.len();
    let mut indices: Vec<usize> = (0..n).collect();

    // Sort indices by array values
    indices.sort_by(|&a, &b| arr[a].partial_cmp(&arr[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = Array1::zeros(n);
    for (rank, &idx) in indices.iter().enumerate() {
        ranks[idx] = (rank + 1) as Float;
    }

    ranks
}

/// Calculate Pearson correlation between two arrays
pub fn pearson_correlation(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() || x.is_empty() {
        return Ok(0.0);
    }

    let n = x.len() as Float;
    let x_mean = x.mean().unwrap_or(0.0);
    let y_mean = y.mean().unwrap_or(0.0);

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..x.len() {
        let x_diff = x[i] - x_mean;
        let y_diff = y[i] - y_mean;
        numerator += x_diff * y_diff;
        x_var += x_diff * x_diff;
        y_var += y_diff * y_diff;
    }

    if x_var == 0.0 || y_var == 0.0 {
        return Ok(0.0);
    }

    Ok(numerator / (x_var * y_var).sqrt())
}

/// Calculate mutual information between two continuous variables
pub fn mutual_information(x: &Array1<Float>, y: &Array1<Float>) -> Result<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let n = x.len();
    if n < 2 {
        return Ok(0.0);
    }

    // Simple histogram-based mutual information estimation
    let n_bins = (n as Float).sqrt().ceil() as usize;
    let n_bins = n_bins.max(5).min(20); // Reasonable range

    // Get ranges
    let x_min = x.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let x_max = x.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_min = y.iter().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
    let y_max = y.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();

    let x_range = x_max - x_min;
    let y_range = y_max - y_min;

    if x_range == 0.0 || y_range == 0.0 {
        return Ok(0.0);
    }

    // Create histograms
    let mut joint_hist = vec![vec![0usize; n_bins]; n_bins];
    let mut x_hist = vec![0usize; n_bins];
    let mut y_hist = vec![0usize; n_bins];

    for i in 0..n {
        let x_bin = ((x[i] - x_min) / x_range * n_bins as Float) as usize;
        let y_bin = ((y[i] - y_min) / y_range * n_bins as Float) as usize;

        let x_bin = x_bin.min(n_bins - 1);
        let y_bin = y_bin.min(n_bins - 1);

        joint_hist[x_bin][y_bin] += 1;
        x_hist[x_bin] += 1;
        y_hist[y_bin] += 1;
    }

    // Calculate mutual information
    let mut mi = 0.0;
    let n_f = n as Float;

    for i in 0..n_bins {
        for j in 0..n_bins {
            if joint_hist[i][j] > 0 && x_hist[i] > 0 && y_hist[j] > 0 {
                let p_xy = joint_hist[i][j] as Float / n_f;
                let p_x = x_hist[i] as Float / n_f;
                let p_y = y_hist[j] as Float / n_f;

                mi += p_xy * (p_xy / (p_x * p_y)).ln();
            }
        }
    }

    Ok(mi)
}

/// Extract subset of data for given samples and features
pub fn extract_subset_data(
    x: &Array2<Float>,
    sample_indices: &[usize],
    feature_indices: &[usize],
) -> Array2<Float> {
    let n_samples = sample_indices.len();
    let n_features = feature_indices.len();
    let mut subset = Array2::zeros((n_samples, n_features));

    for (i, &sample_idx) in sample_indices.iter().enumerate() {
        for (j, &feature_idx) in feature_indices.iter().enumerate() {
            subset[[i, j]] = x[[sample_idx, feature_idx]];
        }
    }

    subset
}

/// Extract subset of labels for given samples
pub fn extract_subset_labels(y: &Array1<Float>, sample_indices: &[usize]) -> Array1<Float> {
    let mut subset = Array1::zeros(sample_indices.len());
    for (i, &idx) in sample_indices.iter().enumerate() {
        subset[i] = y[idx];
    }
    subset
}

/// Calculate mean squared error
pub fn mean_squared_error(y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Float {
    if y_true.len() != y_pred.len() {
        return Float::INFINITY;
    }

    let n = y_true.len() as Float;
    let mut mse = 0.0;

    for i in 0..y_true.len() {
        let diff = y_true[i] - y_pred[i];
        mse += diff * diff;
    }

    mse / n
}

/// Find the index of the maximum value in an array
pub fn argmax(arr: &Array1<Float>) -> usize {
    let mut max_idx = 0;
    let mut max_val = arr[0];

    for i in 1..arr.len() {
        if arr[i] > max_val {
            max_val = arr[i];
            max_idx = i;
        }
    }

    max_idx
}

/// Convenience function for feature selection isotonic regression
pub fn feature_selection_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    selection_method: FeatureSelectionMethod,
    n_features_to_select: Option<usize>,
) -> Result<Array1<Float>> {
    let regressor = FeatureSelectionIsotonicRegression::new()
        .selection_method(selection_method)
        .n_features_to_select(n_features_to_select.unwrap_or(x.ncols().min(5)));

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

/// Partial order isotonic regression with general ordering constraints
///
/// This struct allows for arbitrary partial ordering relationships between observations,
/// going beyond simple monotonic constraints based on coordinate ordering.
#[derive(Debug, Clone)]
/// PartialOrderIsotonicRegression
pub struct PartialOrderIsotonicRegression<State = Untrained> {
    /// Partial order relationships as pairs (i, j) where i ≤ j must hold
    pub partial_order: Vec<(usize, usize)>,
    /// Loss function for optimization
    pub loss: LossFunction,
    /// Tolerance for constraint violations
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Whether to use exact or approximate algorithm
    pub exact: bool,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    order_graph_: Option<HashMap<usize, Vec<usize>>>,

    _state: PhantomData<State>,
}

impl PartialOrderIsotonicRegression<Untrained> {
    pub fn new() -> Self {
        Self {
            partial_order: Vec::new(),
            loss: LossFunction::SquaredLoss,
            tolerance: 1e-6,
            max_iterations: 1000,
            exact: true,
            fitted_values_: None,
            order_graph_: None,
            _state: PhantomData,
        }
    }

    /// Set the partial order constraints
    pub fn partial_order(mut self, order: Vec<(usize, usize)>) -> Self {
        self.partial_order = order;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set tolerance for constraint violations
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set whether to use exact algorithm
    pub fn exact(mut self, exact: bool) -> Self {
        self.exact = exact;
        self
    }
}

impl Default for PartialOrderIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PartialOrderIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for PartialOrderIsotonicRegression<Untrained> {
    type Fitted = PartialOrderIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "X and y cannot be empty".to_string(),
            ));
        }

        // Validate partial order constraints
        for &(i, j) in &self.partial_order {
            if i >= n || j >= n {
                return Err(SklearsError::InvalidInput(format!(
                    "Partial order constraint ({}, {}) references invalid indices for array of length {}",
                    i, j, n
                )));
            }
        }

        // Build order graph
        let order_graph = self.build_order_graph(n);

        // Solve partial order isotonic regression
        let fitted_values = if self.exact {
            self.solve_exact_partial_order(y, &order_graph)?
        } else {
            self.solve_approximate_partial_order(y, &order_graph)?
        };

        Ok(PartialOrderIsotonicRegression {
            partial_order: self.partial_order,
            loss: self.loss,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            exact: self.exact,
            fitted_values_: Some(fitted_values),
            order_graph_: Some(order_graph),
            _state: PhantomData,
        })
    }
}

impl PartialOrderIsotonicRegression<Untrained> {
    /// Build directed graph representation of partial order
    fn build_order_graph(&self, n: usize) -> HashMap<usize, Vec<usize>> {
        let mut graph = HashMap::new();

        // Initialize empty adjacency lists
        for i in 0..n {
            graph.insert(i, Vec::new());
        }

        // Add edges for partial order constraints
        for &(i, j) in &self.partial_order {
            graph.get_mut(&i).unwrap().push(j);
        }

        graph
    }

    /// Solve exact partial order isotonic regression using iterative projection
    fn solve_exact_partial_order(
        &self,
        y: &Array1<Float>,
        order_graph: &HashMap<usize, Vec<usize>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut fitted = y.clone();

        for iteration in 0..self.max_iterations {
            let prev_fitted = fitted.clone();

            // Project onto partial order constraints
            fitted = self.project_onto_partial_order(&fitted, order_graph)?;

            // Check convergence
            let change = fitted
                .iter()
                .zip(prev_fitted.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |max: f64, x| max.max(x));

            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted)
    }

    /// Solve approximate partial order isotonic regression using topological sorting
    fn solve_approximate_partial_order(
        &self,
        y: &Array1<Float>,
        order_graph: &HashMap<usize, Vec<usize>>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut fitted = y.clone();

        // Compute topological order
        let topo_order = self.topological_sort(order_graph)?;

        // Apply constraints in topological order
        for &node in &topo_order {
            if let Some(successors) = order_graph.get(&node) {
                for &successor in successors {
                    if fitted[node] > fitted[successor] {
                        // Violation: adjust to satisfy constraint
                        let avg = (fitted[node] + fitted[successor]) / 2.0;
                        fitted[node] = avg;
                        fitted[successor] = avg;
                    }
                }
            }
        }

        Ok(fitted)
    }

    /// Project values onto partial order constraints
    fn project_onto_partial_order(
        &self,
        values: &Array1<Float>,
        order_graph: &HashMap<usize, Vec<usize>>,
    ) -> Result<Array1<Float>> {
        let mut projected = values.clone();

        // Iteratively enforce constraints
        let mut changed = true;
        while changed {
            changed = false;

            for (&node, successors) in order_graph {
                for &successor in successors {
                    if projected[node] > projected[successor] {
                        // Pool adjacent violators
                        let avg = (projected[node] + projected[successor]) / 2.0;
                        projected[node] = avg;
                        projected[successor] = avg;
                        changed = true;
                    }
                }
            }
        }

        Ok(projected)
    }

    /// Compute topological sorting of the partial order graph
    fn topological_sort(&self, graph: &HashMap<usize, Vec<usize>>) -> Result<Vec<usize>> {
        let n = graph.len();
        let mut in_degree = vec![0; n];
        let mut queue = VecDeque::new();
        let mut result = Vec::new();

        // Compute in-degrees
        for (_, successors) in graph {
            for &successor in successors {
                in_degree[successor] += 1;
            }
        }

        // Initialize queue with nodes having zero in-degree
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        // Process nodes in topological order
        while let Some(node) = queue.pop_front() {
            result.push(node);

            if let Some(successors) = graph.get(&node) {
                for &successor in successors {
                    in_degree[successor] -= 1;
                    if in_degree[successor] == 0 {
                        queue.push_back(successor);
                    }
                }
            }
        }

        if result.len() != n {
            return Err(SklearsError::InvalidInput(
                "Partial order contains cycles".to_string(),
            ));
        }

        Ok(result)
    }
}

impl Predict<Array1<Float>, Array1<Float>> for PartialOrderIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_values = self.fitted_values_.as_ref().ok_or_else(|| {
            SklearsError::NotFitted {
                operation: "predict".to_string(),
            }
        })?;

        // For partial order regression, prediction is just returning fitted values
        // In a more sophisticated implementation, we might interpolate based on x values
        if x.len() != fitted_values.len() {
            return Err(SklearsError::InvalidInput(
                "Prediction input length must match training data length".to_string(),
            ));
        }

        Ok(fitted_values.clone())
    }
}

impl PartialOrderIsotonicRegression<Trained> {
    /// Get the fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }

    /// Get the partial order graph
    pub fn order_graph(&self) -> &HashMap<usize, Vec<usize>> {
        self.order_graph_.as_ref().unwrap()
    }

    /// Check if the fitted values satisfy all partial order constraints
    pub fn validate_constraints(&self) -> bool {
        let fitted = self.fitted_values_.as_ref().unwrap();
        let graph = self.order_graph_.as_ref().unwrap();

        for (&i, successors) in graph {
            for &j in successors {
                if fitted[i] > fitted[j] + self.tolerance {
                    return false;
                }
            }
        }

        true
    }
}

/// Convenience function for partial order isotonic regression
pub fn partial_order_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    partial_order: Vec<(usize, usize)>,
) -> Result<Array1<Float>> {
    let regressor = PartialOrderIsotonicRegression::new().partial_order(partial_order);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

/// Tree-order isotonic regression for hierarchical constraints
///
/// Implements isotonic regression with tree-structured ordering constraints where
/// data points are organized in a tree hierarchy and monotonicity is enforced
/// along tree edges (parent-child relationships).
#[derive(Debug, Clone)]
/// TreeOrderIsotonicRegression
pub struct TreeOrderIsotonicRegression<State = Untrained> {
    /// Tree structure as parent-child relationships
    pub tree_edges: Vec<(usize, usize)>,
    /// Root nodes of the forest (nodes with no parents)
    pub roots: Vec<usize>,
    /// Whether children should be greater than parents (true) or less (false)
    pub increasing: bool,
    /// Loss function
    pub loss: LossFunction,
    /// Tolerance for constraint violations
    pub tolerance: Float,
    /// Maximum number of iterations
    pub max_iterations: usize,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    tree_structure_: Option<HashMap<usize, Vec<usize>>>,

    _state: PhantomData<State>,
}

impl TreeOrderIsotonicRegression<Untrained> {
    /// Create a new tree-order isotonic regression model
    pub fn new() -> Self {
        Self {
            tree_edges: Vec::new(),
            roots: Vec::new(),
            increasing: true,
            loss: LossFunction::SquaredLoss,
            tolerance: 1e-6,
            max_iterations: 1000,
            fitted_values_: None,
            tree_structure_: None,
            _state: PhantomData,
        }
    }

    /// Set tree edges (parent, child) pairs
    pub fn tree_edges(mut self, edges: Vec<(usize, usize)>) -> Self {
        self.tree_edges = edges;
        self
    }

    /// Set root nodes
    pub fn roots(mut self, roots: Vec<usize>) -> Self {
        self.roots = roots;
        self
    }

    /// Set whether children should be greater than parents
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Build tree from edges automatically detecting roots
    pub fn from_edges(edges: Vec<(usize, usize)>) -> Self {
        let mut children: HashSet<usize> = HashSet::new();
        let mut all_nodes: HashSet<usize> = HashSet::new();

        for &(parent, child) in &edges {
            children.insert(child);
            all_nodes.insert(parent);
            all_nodes.insert(child);
        }

        let roots: Vec<usize> = all_nodes.difference(&children).cloned().collect();

        Self::new().tree_edges(edges).roots(roots)
    }
}

impl Default for TreeOrderIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for TreeOrderIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for TreeOrderIsotonicRegression<Untrained> {
    type Fitted = TreeOrderIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let n = x.len();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "X and y cannot be empty".to_string(),
            ));
        }

        // Validate tree structure
        for &(parent, child) in &self.tree_edges {
            if parent >= n || child >= n {
                return Err(SklearsError::InvalidInput(format!(
                    "Tree edge ({}, {}) references invalid indices for array of length {}",
                    parent, child, n
                )));
            }
        }

        // Build tree structure
        let tree_structure = self.build_tree_structure(n);

        // Validate that the structure is actually a tree (no cycles)
        self.validate_tree_structure(&tree_structure)?;

        // Solve tree-order isotonic regression
        let fitted_values = self.solve_tree_isotonic(y, &tree_structure)?;

        Ok(TreeOrderIsotonicRegression {
            tree_edges: self.tree_edges,
            roots: self.roots,
            increasing: self.increasing,
            loss: self.loss,
            tolerance: self.tolerance,
            max_iterations: self.max_iterations,
            fitted_values_: Some(fitted_values),
            tree_structure_: Some(tree_structure),
            _state: PhantomData,
        })
    }
}

impl TreeOrderIsotonicRegression<Untrained> {
    /// Build tree structure as adjacency list
    fn build_tree_structure(&self, n: usize) -> HashMap<usize, Vec<usize>> {
        let mut tree = HashMap::new();

        // Initialize empty adjacency lists
        for i in 0..n {
            tree.insert(i, Vec::new());
        }

        // Add children for each parent
        for &(parent, child) in &self.tree_edges {
            tree.get_mut(&parent).unwrap().push(child);
        }

        tree
    }

    /// Validate that the structure is a forest (no cycles)
    fn validate_tree_structure(&self, tree: &HashMap<usize, Vec<usize>>) -> Result<()> {
        let n = tree.len();
        let mut visited = vec![false; n];
        let mut rec_stack = vec![false; n];

        for &root in &self.roots {
            if self.has_cycle_util(root, &mut visited, &mut rec_stack, tree) {
                return Err(SklearsError::InvalidInput(
                    "Tree structure contains cycles".to_string(),
                ));
            }
        }

        Ok(())
    }

    /// DFS utility for cycle detection
    fn has_cycle_util(
        &self,
        node: usize,
        visited: &mut [bool],
        rec_stack: &mut [bool],
        tree: &HashMap<usize, Vec<usize>>,
    ) -> bool {
        if rec_stack[node] {
            return true;
        }

        if visited[node] {
            return false;
        }

        visited[node] = true;
        rec_stack[node] = true;

        if let Some(children) = tree.get(&node) {
            for &child in children {
                if self.has_cycle_util(child, visited, rec_stack, tree) {
                    return true;
                }
            }
        }

        rec_stack[node] = false;
        false
    }

    /// Solve tree-order isotonic regression using bottom-up approach
    fn solve_tree_isotonic(
        &self,
        y: &Array1<Float>,
        tree: &HashMap<usize, Vec<usize>>,
    ) -> Result<Array1<Float>> {
        let mut fitted = y.clone();

        for iteration in 0..self.max_iterations {
            let prev_fitted = fitted.clone();

            // Apply constraints for each root
            for &root in &self.roots {
                self.apply_tree_constraints_recursive(root, &mut fitted, tree);
            }

            // Check convergence
            let change = fitted
                .iter()
                .zip(prev_fitted.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, |max: f64, x| max.max(x));

            if change < self.tolerance {
                break;
            }
        }

        Ok(fitted)
    }

    /// Recursively apply tree constraints (post-order traversal)
    fn apply_tree_constraints_recursive(
        &self,
        node: usize,
        fitted: &mut Array1<Float>,
        tree: &HashMap<usize, Vec<usize>>,
    ) {
        if let Some(children) = tree.get(&node) {
            // First, recursively process children
            for &child in children {
                self.apply_tree_constraints_recursive(child, fitted, tree);
            }

            // Then apply constraints between this node and its children
            for &child in children {
                if self.increasing {
                    // Parent should be <= child
                    if fitted[node] > fitted[child] {
                        let avg = (fitted[node] + fitted[child]) / 2.0;
                        fitted[node] = avg;
                        fitted[child] = avg;
                    }
                } else {
                    // Parent should be >= child
                    if fitted[node] < fitted[child] {
                        let avg = (fitted[node] + fitted[child]) / 2.0;
                        fitted[node] = avg;
                        fitted[child] = avg;
                    }
                }
            }
        }
    }
}

impl TreeOrderIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }

    /// Get tree structure
    pub fn tree_structure(&self) -> &HashMap<usize, Vec<usize>> {
        self.tree_structure_.as_ref().unwrap()
    }

    /// Validate that fitted values satisfy tree constraints
    pub fn validate_tree_constraints(&self) -> bool {
        let fitted = self.fitted_values();
        let tree = self.tree_structure();

        for (&parent, children) in tree {
            for &child in children {
                if self.increasing && fitted[parent] > fitted[child] + self.tolerance {
                    return false;
                }
                if !self.increasing && fitted[parent] < fitted[child] - self.tolerance {
                    return false;
                }
            }
        }

        true
    }
}

impl Predict<Array1<Float>, Array1<Float>> for TreeOrderIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_values = self.fitted_values_.as_ref().ok_or_else(|| {
            SklearsError::NotFitted {
                operation: "predict".to_string(),
            }
        })?;

        if x.len() != fitted_values.len() {
            return Err(SklearsError::InvalidInput(
                "Prediction input length must match training data length".to_string(),
            ));
        }

        Ok(fitted_values.clone())
    }
}

/// Convenience function for tree-order isotonic regression
pub fn tree_order_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    tree_edges: Vec<(usize, usize)>,
    increasing: bool,
) -> Result<Array1<Float>> {
    let regressor = TreeOrderIsotonicRegression::from_edges(tree_edges).increasing(increasing);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}
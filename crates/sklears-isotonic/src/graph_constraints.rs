//! Graph-based order constraints for isotonic regression

use crate::core::LossFunction;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::marker::PhantomData;

/// Graph-based order constraints for isotonic regression
#[derive(Debug, Clone)]
/// GraphOrderIsotonicRegression
pub struct GraphOrderIsotonicRegression<State = Untrained> {
    /// Graph edges (i, j) meaning node i should be <= node j
    pub graph_edges: Vec<(usize, usize)>,
    /// Loss function
    pub loss: LossFunction,
    /// Bounds
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,
    /// Optimization parameters
    pub max_iterations: usize,
    pub tolerance: Float,
    /// Learning rate for gradient descent
    pub learning_rate: Float,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    adjacency_list_: Option<HashMap<usize, Vec<usize>>>,
    topological_order_: Option<Vec<usize>>,

    _state: PhantomData<State>,
}

impl GraphOrderIsotonicRegression<Untrained> {
    /// Create new graph-order isotonic regression
    pub fn new() -> Self {
        Self {
            graph_edges: Vec::new(),
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            fitted_values_: None,
            adjacency_list_: None,
            topological_order_: None,
            _state: PhantomData,
        }
    }

    /// Set graph edges (i, j) meaning y[i] <= y[j]
    pub fn graph_edges(mut self, edges: Vec<(usize, usize)>) -> Self {
        self.graph_edges = edges;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set bounds
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(
        mut self,
        max_iterations: usize,
        tolerance: Float,
        learning_rate: Float,
    ) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self.learning_rate = learning_rate;
        self
    }
}

impl Default for GraphOrderIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GraphOrderIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for GraphOrderIsotonicRegression<Untrained> {
    type Fitted = GraphOrderIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let n = y.len();

        // Validate graph edges
        for &(i, j) in &self.graph_edges {
            if i >= n || j >= n {
                return Err(SklearsError::InvalidInput(format!(
                    "Graph edge ({}, {}) contains indices outside data range [0, {})",
                    i,
                    j,
                    n - 1
                )));
            }
        }

        // Build adjacency list
        let adjacency_list = build_adjacency_list(&self.graph_edges, n);

        // Check for cycles
        if has_cycle(&adjacency_list, n)? {
            return Err(SklearsError::InvalidInput(
                "Graph contains cycles - not a valid partial order".to_string(),
            ));
        }

        // Get topological ordering
        let topological_order = topological_sort(&adjacency_list, n)?;

        // Fit isotonic regression with graph constraints
        let fitted_values = fit_graph_isotonic(
            y,
            &self.graph_edges,
            &adjacency_list,
            &self.loss,
            self.max_iterations,
            self.tolerance,
            self.learning_rate,
        )?;

        // Apply bounds if specified
        let mut bounded_values = fitted_values;
        if let Some(min_val) = self.y_min {
            bounded_values.mapv_inplace(|v| v.max(min_val));
        }
        if let Some(max_val) = self.y_max {
            bounded_values.mapv_inplace(|v| v.min(max_val));
        }

        Ok(GraphOrderIsotonicRegression {
            graph_edges: self.graph_edges,
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            fitted_values_: Some(bounded_values),
            adjacency_list_: Some(adjacency_list),
            topological_order_: Some(topological_order),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for GraphOrderIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        // For graph-based constraints, we use the fitted values directly
        // since the constraint structure depends on the training data indices
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        if x.len() == fitted_values.len() {
            // If same length, assume same data points
            Ok(fitted_values.clone())
        } else {
            // For new data, we can't apply the same graph structure
            // Return interpolated values based on the original x values used in training
            Err(SklearsError::InvalidInput(
                "Graph-based isotonic regression requires same data points for prediction"
                    .to_string(),
            ))
        }
    }
}

impl GraphOrderIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }

    /// Get the adjacency list representation of the graph
    pub fn adjacency_list(&self) -> &HashMap<usize, Vec<usize>> {
        self.adjacency_list_.as_ref().unwrap()
    }

    /// Get the topological ordering of the graph
    pub fn topological_order(&self) -> &[usize] {
        self.topological_order_.as_ref().unwrap()
    }
}

/// Hierarchical constraints for isotonic regression
#[derive(Debug, Clone)]
/// HierarchicalIsotonicRegression
pub struct HierarchicalIsotonicRegression<State = Untrained> {
    /// Hierarchical levels - nodes at level i must be <= nodes at level i+1
    pub hierarchy_levels: Vec<Vec<usize>>,
    /// Loss function
    pub loss: LossFunction,
    /// Bounds
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,
    /// Optimization parameters
    pub max_iterations: usize,
    pub tolerance: Float,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl HierarchicalIsotonicRegression<Untrained> {
    /// Create new hierarchical isotonic regression
    pub fn new() -> Self {
        Self {
            hierarchy_levels: Vec::new(),
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            max_iterations: 1000,
            tolerance: 1e-6,
            fitted_values_: None,
            _state: PhantomData,
        }
    }

    /// Set hierarchy levels (each level contains node indices)
    pub fn hierarchy_levels(mut self, levels: Vec<Vec<usize>>) -> Self {
        self.hierarchy_levels = levels;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set bounds
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(mut self, max_iterations: usize, tolerance: Float) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }
}

impl Default for HierarchicalIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for HierarchicalIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for HierarchicalIsotonicRegression<Untrained> {
    type Fitted = HierarchicalIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        let n = y.len();

        // Validate hierarchy levels
        let mut all_indices = HashSet::new();
        for level in &self.hierarchy_levels {
            for &idx in level {
                if idx >= n {
                    return Err(SklearsError::InvalidInput(format!(
                        "Hierarchy level contains index {} outside data range [0, {})",
                        idx,
                        n - 1
                    )));
                }
                if !all_indices.insert(idx) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Index {} appears in multiple hierarchy levels",
                        idx
                    )));
                }
            }
        }

        if all_indices.len() != n {
            return Err(SklearsError::InvalidInput(
                "Hierarchy levels must cover all data points exactly once".to_string(),
            ));
        }

        // Convert hierarchy to graph edges
        let graph_edges = hierarchy_to_graph_edges(&self.hierarchy_levels);

        // Fit isotonic regression with hierarchical constraints
        let adjacency_list = build_adjacency_list(&graph_edges, n);
        let fitted_values = fit_graph_isotonic(
            y,
            &graph_edges,
            &adjacency_list,
            &self.loss,
            self.max_iterations,
            self.tolerance,
            0.01,
        )?;

        // Apply bounds if specified
        let mut bounded_values = fitted_values;
        if let Some(min_val) = self.y_min {
            bounded_values.mapv_inplace(|v| v.max(min_val));
        }
        if let Some(max_val) = self.y_max {
            bounded_values.mapv_inplace(|v| v.min(max_val));
        }

        Ok(HierarchicalIsotonicRegression {
            hierarchy_levels: self.hierarchy_levels,
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            fitted_values_: Some(bounded_values),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for HierarchicalIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        if x.len() == fitted_values.len() {
            Ok(fitted_values.clone())
        } else {
            Err(SklearsError::InvalidInput(
                "Hierarchical isotonic regression requires same data points for prediction"
                    .to_string(),
            ))
        }
    }
}

impl HierarchicalIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }

    /// Get the hierarchy levels
    pub fn hierarchy_levels(&self) -> &[Vec<usize>] {
        &self.hierarchy_levels
    }
}

// Helper functions

/// Build adjacency list from graph edges
fn build_adjacency_list(edges: &[(usize, usize)], n: usize) -> HashMap<usize, Vec<usize>> {
    let mut adj_list = HashMap::new();

    // Initialize all nodes
    for i in 0..n {
        adj_list.insert(i, Vec::new());
    }

    // Add edges
    for &(from, to) in edges {
        adj_list.get_mut(&from).unwrap().push(to);
    }

    adj_list
}

/// Check if the graph has cycles using DFS
fn has_cycle(adj_list: &HashMap<usize, Vec<usize>>, n: usize) -> Result<bool> {
    let mut color = vec![0; n]; // 0: white, 1: gray, 2: black

    for i in 0..n {
        if color[i] == 0 {
            if dfs_cycle_check(i, adj_list, &mut color) {
                return Ok(true);
            }
        }
    }

    Ok(false)
}

/// DFS helper for cycle detection
fn dfs_cycle_check(node: usize, adj_list: &HashMap<usize, Vec<usize>>, color: &mut [i32]) -> bool {
    color[node] = 1; // Gray

    if let Some(neighbors) = adj_list.get(&node) {
        for &neighbor in neighbors {
            if color[neighbor] == 1 {
                return true; // Back edge found (cycle)
            }
            if color[neighbor] == 0 && dfs_cycle_check(neighbor, adj_list, color) {
                return true;
            }
        }
    }

    color[node] = 2; // Black
    false
}

/// Topological sort using Kahn's algorithm
fn topological_sort(adj_list: &HashMap<usize, Vec<usize>>, n: usize) -> Result<Vec<usize>> {
    let mut in_degree = vec![0; n];

    // Calculate in-degrees
    for neighbors in adj_list.values() {
        for &neighbor in neighbors {
            in_degree[neighbor] += 1;
        }
    }

    // Initialize queue with nodes having in-degree 0
    let mut queue = VecDeque::new();
    for i in 0..n {
        if in_degree[i] == 0 {
            queue.push_back(i);
        }
    }

    let mut topo_order = Vec::new();

    while let Some(node) = queue.pop_front() {
        topo_order.push(node);

        if let Some(neighbors) = adj_list.get(&node) {
            for &neighbor in neighbors {
                in_degree[neighbor] -= 1;
                if in_degree[neighbor] == 0 {
                    queue.push_back(neighbor);
                }
            }
        }
    }

    if topo_order.len() != n {
        return Err(SklearsError::InvalidInput(
            "Graph contains cycles - cannot perform topological sort".to_string(),
        ));
    }

    Ok(topo_order)
}

/// Convert hierarchy levels to graph edges
fn hierarchy_to_graph_edges(levels: &[Vec<usize>]) -> Vec<(usize, usize)> {
    let mut edges = Vec::new();

    for i in 0..levels.len() - 1 {
        for &node_i in &levels[i] {
            for &node_j in &levels[i + 1] {
                edges.push((node_i, node_j));
            }
        }
    }

    edges
}

/// Fit isotonic regression with graph constraints using projected gradient descent
fn fit_graph_isotonic(
    y: &Array1<Float>,
    edges: &[(usize, usize)],
    adj_list: &HashMap<usize, Vec<usize>>,
    loss: &LossFunction,
    max_iterations: usize,
    tolerance: Float,
    learning_rate: Float,
) -> Result<Array1<Float>> {
    let n = y.len();
    let mut fitted = y.clone();

    for iteration in 0..max_iterations {
        let prev_fitted = fitted.clone();

        // Compute gradients based on loss function
        let gradients = compute_gradients(&fitted, y, loss)?;

        // Gradient descent step
        for i in 0..n {
            fitted[i] -= learning_rate * gradients[i];
        }

        // Project onto constraint set
        project_onto_graph_constraints(&mut fitted, edges)?;

        // Check convergence
        let change = (&fitted - &prev_fitted).mapv(|x| x.abs()).sum();
        if change < tolerance {
            break;
        }

        if iteration == max_iterations - 1 {
            eprintln!("Warning: Graph isotonic regression reached maximum iterations");
        }
    }

    Ok(fitted)
}

/// Compute gradients for different loss functions
fn compute_gradients(
    fitted: &Array1<Float>,
    y: &Array1<Float>,
    loss: &LossFunction,
) -> Result<Array1<Float>> {
    let mut gradients = Array1::zeros(fitted.len());

    match loss {
        LossFunction::SquaredLoss => {
            gradients = fitted - y; // d/dx (1/2)(x-y)^2 = x-y
        }
        LossFunction::AbsoluteLoss => {
            for i in 0..fitted.len() {
                let diff = fitted[i] - y[i];
                gradients[i] = if diff > 0.0 {
                    1.0
                } else if diff < 0.0 {
                    -1.0
                } else {
                    0.0
                };
            }
        }
        LossFunction::HuberLoss { delta } => {
            for i in 0..fitted.len() {
                let diff = fitted[i] - y[i];
                gradients[i] = if diff.abs() <= *delta {
                    diff
                } else if diff > 0.0 {
                    *delta
                } else {
                    -*delta
                };
            }
        }
        LossFunction::QuantileLoss { quantile } => {
            for i in 0..fitted.len() {
                let diff = fitted[i] - y[i];
                gradients[i] = if diff > 0.0 {
                    *quantile
                } else {
                    quantile - 1.0
                };
            }
        }
    }

    Ok(gradients)
}

/// Project onto graph constraint set using iterative constraint satisfaction
fn project_onto_graph_constraints(
    fitted: &mut Array1<Float>,
    edges: &[(usize, usize)],
) -> Result<()> {
    let max_projection_iterations = 100;
    let projection_tolerance = 1e-8;

    for _ in 0..max_projection_iterations {
        let mut changed = false;

        for &(i, j) in edges {
            if fitted[i] > fitted[j] {
                // Violates constraint i <= j, so adjust both to their average
                let avg = (fitted[i] + fitted[j]) / 2.0;
                fitted[i] = avg;
                fitted[j] = avg;
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    Ok(())
}

// Convenience functions

/// Convenience function for graph-based isotonic regression
pub fn graph_order_isotonic_regression(
    y: &Array1<Float>,
    graph_edges: Vec<(usize, usize)>,
    loss: Option<LossFunction>,
    bounds: Option<(Float, Float)>,
) -> Result<Array1<Float>> {
    let mut graph_iso = GraphOrderIsotonicRegression::new().graph_edges(graph_edges);

    if let Some(loss_fn) = loss {
        graph_iso = graph_iso.loss(loss_fn);
    }

    if let Some((y_min, y_max)) = bounds {
        graph_iso = graph_iso.bounds(Some(y_min), Some(y_max));
    }

    // Create dummy x values since graph constraints don't depend on x
    let x = Array1::from_iter(0..y.len()).mapv(|i| i as Float);

    let fitted = graph_iso.fit(&x, y)?;
    Ok(fitted.fitted_values().clone())
}

/// Convenience function for hierarchical isotonic regression
pub fn hierarchical_isotonic_regression(
    y: &Array1<Float>,
    hierarchy_levels: Vec<Vec<usize>>,
    loss: Option<LossFunction>,
    bounds: Option<(Float, Float)>,
) -> Result<Array1<Float>> {
    let mut hier_iso = HierarchicalIsotonicRegression::new().hierarchy_levels(hierarchy_levels);

    if let Some(loss_fn) = loss {
        hier_iso = hier_iso.loss(loss_fn);
    }

    if let Some((y_min, y_max)) = bounds {
        hier_iso = hier_iso.bounds(Some(y_min), Some(y_max));
    }

    // Create dummy x values since hierarchical constraints don't depend on x
    let x = Array1::from_iter(0..y.len()).mapv(|i| i as Float);

    let fitted = hier_iso.fit(&x, y)?;
    Ok(fitted.fitted_values().clone())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::LossFunction;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_graph_order_isotonic_regression_simple_chain() {
        // Test simple chain: 0 -> 1 -> 2 -> 3
        let y = array![3.0, 1.0, 4.0, 2.0]; // Violates ordering
        let edges = vec![(0, 1), (1, 2), (2, 3)];

        let graph_iso = GraphOrderIsotonicRegression::new().graph_edges(edges);

        let x = array![0.0, 1.0, 2.0, 3.0];
        let fitted = graph_iso.fit(&x, &y).unwrap();
        let fitted_values = fitted.fitted_values();

        // Check that ordering constraints are satisfied
        assert!(fitted_values[0] <= fitted_values[1]);
        assert!(fitted_values[1] <= fitted_values[2]);
        assert!(fitted_values[2] <= fitted_values[3]);
    }

    #[test]
    fn test_graph_order_isotonic_regression_diamond() {
        // Test diamond pattern: 0 -> {1, 2} -> 3
        let y = array![1.0, 3.0, 2.0, 0.5]; // Last value violates constraints
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];

        let graph_iso = GraphOrderIsotonicRegression::new().graph_edges(edges);

        let x = array![0.0, 1.0, 2.0, 3.0];
        let fitted = graph_iso.fit(&x, &y).unwrap();
        let fitted_values = fitted.fitted_values();

        // Check all constraints
        assert!(fitted_values[0] <= fitted_values[1]);
        assert!(fitted_values[0] <= fitted_values[2]);
        assert!(fitted_values[1] <= fitted_values[3]);
        assert!(fitted_values[2] <= fitted_values[3]);
    }

    #[test]
    fn test_graph_order_isotonic_regression_with_bounds() {
        let y = array![1.0, 3.0, 2.0, 4.0];
        let edges = vec![(0, 1), (1, 2), (2, 3)];

        let graph_iso = GraphOrderIsotonicRegression::new()
            .graph_edges(edges)
            .bounds(Some(1.5), Some(3.5));

        let x = array![0.0, 1.0, 2.0, 3.0];
        let fitted = graph_iso.fit(&x, &y).unwrap();
        let fitted_values = fitted.fitted_values();

        // Check bounds are respected
        for &val in fitted_values.iter() {
            assert!(val >= 1.5 - 1e-6);
            assert!(val <= 3.5 + 1e-6);
        }

        // Check ordering constraints
        for i in 0..fitted_values.len() - 1 {
            assert!(fitted_values[i] <= fitted_values[i + 1]);
        }
    }

    #[test]
    fn test_graph_order_isotonic_regression_cycle_detection() {
        // Test cycle detection: 0 -> 1 -> 2 -> 0 (cycle)
        let y = array![1.0, 2.0, 3.0];
        let edges = vec![(0, 1), (1, 2), (2, 0)]; // Contains cycle

        let graph_iso = GraphOrderIsotonicRegression::new().graph_edges(edges);

        let x = array![0.0, 1.0, 2.0];
        let result = graph_iso.fit(&x, &y);

        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn test_graph_order_isotonic_regression_invalid_indices() {
        let y = array![1.0, 2.0, 3.0];
        let edges = vec![(0, 5)]; // Index 5 is out of bounds

        let graph_iso = GraphOrderIsotonicRegression::new().graph_edges(edges);

        let x = array![0.0, 1.0, 2.0];
        let result = graph_iso.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("outside data range"));
    }

    #[test]
    fn test_graph_order_isotonic_regression_different_loss_functions() {
        let y = array![1.0, 4.0, 2.0, 5.0];
        let edges = vec![(0, 1), (1, 2), (2, 3)];
        let x = array![0.0, 1.0, 2.0, 3.0];

        // Test with L1 loss
        let graph_iso_l1 = GraphOrderIsotonicRegression::new()
            .graph_edges(edges.clone())
            .loss(LossFunction::AbsoluteLoss);

        let fitted_l1 = graph_iso_l1.fit(&x, &y).unwrap();
        let fitted_values_l1 = fitted_l1.fitted_values();

        // Check ordering constraints
        for i in 0..fitted_values_l1.len() - 1 {
            assert!(fitted_values_l1[i] <= fitted_values_l1[i + 1]);
        }

        // Test with Huber loss
        let graph_iso_huber = GraphOrderIsotonicRegression::new()
            .graph_edges(edges)
            .loss(LossFunction::HuberLoss { delta: 1.0 });

        let fitted_huber = graph_iso_huber.fit(&x, &y).unwrap();
        let fitted_values_huber = fitted_huber.fitted_values();

        // Check ordering constraints
        for i in 0..fitted_values_huber.len() - 1 {
            assert!(fitted_values_huber[i] <= fitted_values_huber[i + 1]);
        }
    }

    #[test]
    fn test_hierarchical_isotonic_regression_basic() {
        // Test 3-level hierarchy: {0, 1} < {2} < {3, 4}
        let y = array![3.0, 2.0, 1.0, 5.0, 4.0]; // Violates hierarchy
        let levels = vec![vec![0, 1], vec![2], vec![3, 4]];

        let hier_iso = HierarchicalIsotonicRegression::new().hierarchy_levels(levels);

        let x = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let fitted = hier_iso.fit(&x, &y).unwrap();
        let fitted_values = fitted.fitted_values();

        // Check hierarchical constraints
        // Level 0 (indices 0,1) <= Level 1 (index 2) <= Level 2 (indices 3,4)
        assert!(fitted_values[0] <= fitted_values[2]);
        assert!(fitted_values[1] <= fitted_values[2]);
        assert!(fitted_values[2] <= fitted_values[3]);
        assert!(fitted_values[2] <= fitted_values[4]);
    }

    #[test]
    fn test_hierarchical_isotonic_regression_invalid_levels() {
        let y = array![1.0, 2.0, 3.0];

        // Test overlapping levels
        let levels = vec![vec![0, 1], vec![1, 2]]; // Index 1 appears twice

        let hier_iso = HierarchicalIsotonicRegression::new().hierarchy_levels(levels);

        let x = array![0.0, 1.0, 2.0];
        let result = hier_iso.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("multiple hierarchy levels"));
    }

    #[test]
    fn test_hierarchical_isotonic_regression_incomplete_coverage() {
        let y = array![1.0, 2.0, 3.0];

        // Test incomplete coverage (missing index 2)
        let levels = vec![vec![0], vec![1]]; // Index 2 is missing

        let hier_iso = HierarchicalIsotonicRegression::new().hierarchy_levels(levels);

        let x = array![0.0, 1.0, 2.0];
        let result = hier_iso.fit(&x, &y);

        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("cover all data points"));
    }

    #[test]
    fn test_hierarchical_isotonic_regression_with_bounds() {
        let y = array![2.0, 1.0, 4.0, 3.0];
        let levels = vec![vec![0, 1], vec![2, 3]];

        let hier_iso = HierarchicalIsotonicRegression::new()
            .hierarchy_levels(levels)
            .bounds(Some(1.5), Some(3.5));

        let x = array![0.0, 1.0, 2.0, 3.0];
        let fitted = hier_iso.fit(&x, &y).unwrap();
        let fitted_values = fitted.fitted_values();

        // Check bounds are respected
        for &val in fitted_values.iter() {
            assert!(val >= 1.5 - 1e-6);
            assert!(val <= 3.5 + 1e-6);
        }

        // Check hierarchical constraints
        assert!(fitted_values[0] <= fitted_values[2]);
        assert!(fitted_values[0] <= fitted_values[3]);
        assert!(fitted_values[1] <= fitted_values[2]);
        assert!(fitted_values[1] <= fitted_values[3]);
    }

    #[test]
    fn test_convenience_functions() {
        // Test graph order convenience function
        let y = array![3.0, 1.0, 4.0, 2.0];
        let edges = vec![(0, 1), (1, 2), (2, 3)];

        let fitted = graph_order_isotonic_regression(&y, edges, None, None).unwrap();

        // Check ordering constraints
        for i in 0..fitted.len() - 1 {
            assert!(fitted[i] <= fitted[i + 1]);
        }

        // Test hierarchical convenience function
        let levels = vec![vec![0], vec![1], vec![2], vec![3]];
        let fitted_hier = hierarchical_isotonic_regression(&y, levels, None, None).unwrap();

        // Check hierarchical constraints (should be monotonic)
        for i in 0..fitted_hier.len() - 1 {
            assert!(fitted_hier[i] <= fitted_hier[i + 1]);
        }
    }

    #[test]
    fn test_topological_sort() {
        // Test with simple DAG
        let edges = vec![(0, 1), (0, 2), (1, 3), (2, 3)];
        let adj_list = build_adjacency_list(&edges, 4);
        let topo_order = topological_sort(&adj_list, 4).unwrap();

        // Check that the ordering respects all edges
        let position: HashMap<usize, usize> = topo_order
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        for &(from, to) in &edges {
            assert!(position[&from] < position[&to]);
        }
    }

    #[test]
    fn test_build_adjacency_list() {
        let edges = vec![(0, 1), (0, 2), (1, 2)];
        let adj_list = build_adjacency_list(&edges, 3);

        assert_eq!(adj_list[&0], vec![1, 2]);
        assert_eq!(adj_list[&1], vec![2]);
        assert_eq!(adj_list[&2], Vec::<usize>::new());
    }

    #[test]
    fn test_hierarchy_to_graph_edges() {
        let levels = vec![vec![0, 1], vec![2], vec![3, 4]];
        let edges = hierarchy_to_graph_edges(&levels);

        // Should create edges from level i to level i+1
        let expected_edges = vec![
            (0, 2),
            (1, 2), // Level 0 to Level 1
            (2, 3),
            (2, 4), // Level 1 to Level 2
        ];

        assert_eq!(edges.len(), expected_edges.len());
        for edge in expected_edges {
            assert!(edges.contains(&edge));
        }
    }
}

//! Adaptive and lattice-order isotonic regression
//!
//! This module contains lattice-order isotonic regression for complex partial order
//! structures and adaptive weighting schemes for robust isotonic regression that
//! automatically adjust to handle outliers and heteroscedasticity.

use std::marker::PhantomData;
use std::collections::{HashMap, HashSet, VecDeque};
use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::{
    isotonic_regression, apply_global_constraint, IsotonicRegression,
    MonotonicityConstraint, LossFunction,
};

/// Lattice-order isotonic regression for complex partial order structures
///
/// Implements isotonic regression with lattice-structured ordering constraints where
/// elements form a lattice (partial order with unique least upper bounds and greatest
/// lower bounds for any pair of elements).
#[derive(Debug, Clone)]
/// LatticeOrderIsotonicRegression
pub struct LatticeOrderIsotonicRegression<State = Untrained> {
    /// Lattice structure represented as ordering constraints
    /// Each tuple (i, j) means element i ≤ element j in the lattice
    pub ordering_constraints: Vec<(usize, usize)>,
    /// Loss function for robust regression
    pub loss: LossFunction,
    /// Maximum number of iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate for projected gradient descent
    pub learning_rate: Float,
    /// Whether to verify lattice properties (expensive for large lattices)
    pub verify_lattice_properties: bool,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    lattice_structure_: Option<LatticeStructure>,

    _state: PhantomData<State>,
}

/// Internal representation of lattice structure
#[derive(Debug, Clone)]
/// LatticeStructure
pub struct LatticeStructure {
    /// Number of elements in the lattice
    pub n_elements: usize,
    /// Direct ordering relationships as adjacency list
    pub ordering_graph: HashMap<usize, Vec<usize>>,
    /// Transitive closure of ordering relationships
    pub transitive_closure: HashMap<usize, HashSet<usize>>,
    /// Levels in the lattice (for more efficient processing)
    pub levels: Vec<Vec<usize>>,
}

impl LatticeStructure {
    /// Create a new lattice structure from ordering constraints
    pub fn new(n_elements: usize, constraints: &[(usize, usize)]) -> Result<Self> {
        // Build ordering graph
        let mut ordering_graph = HashMap::new();
        for i in 0..n_elements {
            ordering_graph.insert(i, Vec::new());
        }

        for &(i, j) in constraints {
            if i >= n_elements || j >= n_elements {
                return Err(SklearsError::InvalidInput(format!(
                    "Constraint ({}, {}) contains invalid indices for {} elements",
                    i, j, n_elements
                )));
            }
            ordering_graph.get_mut(&i).unwrap().push(j);
        }

        // Compute transitive closure
        let transitive_closure = Self::compute_transitive_closure(&ordering_graph, n_elements)?;

        // Compute levels (topological sorting with level assignment)
        let levels = Self::compute_levels(&ordering_graph, n_elements)?;

        Ok(LatticeStructure {
            n_elements,
            ordering_graph,
            transitive_closure,
            levels,
        })
    }

    /// Compute transitive closure using Floyd-Warshall algorithm
    fn compute_transitive_closure(
        graph: &HashMap<usize, Vec<usize>>,
        n: usize,
    ) -> Result<HashMap<usize, HashSet<usize>>> {
        let mut closure = HashMap::new();

        // Initialize with direct relationships
        for i in 0..n {
            let mut reachable = HashSet::new();
            if let Some(neighbors) = graph.get(&i) {
                for &j in neighbors {
                    reachable.insert(j);
                }
            }
            closure.insert(i, reachable);
        }

        // Floyd-Warshall: find all transitive relationships
        for k in 0..n {
            for i in 0..n {
                let k_reachable: HashSet<usize> = closure[&k].clone();
                if closure[&i].contains(&k) {
                    closure.get_mut(&i).unwrap().extend(k_reachable);
                }
            }
        }

        // Check for cycles (which would make it not a valid partial order)
        for i in 0..n {
            if closure[&i].contains(&i) {
                return Err(SklearsError::InvalidInput(
                    "Ordering constraints contain cycles".to_string(),
                ));
            }
        }

        Ok(closure)
    }

    /// Compute levels using topological sorting
    fn compute_levels(
        graph: &HashMap<usize, Vec<usize>>,
        n: usize,
    ) -> Result<Vec<Vec<usize>>> {
        let mut in_degree = vec![0; n];
        let mut levels = Vec::new();

        // Compute in-degrees
        for (_, neighbors) in graph {
            for &neighbor in neighbors {
                in_degree[neighbor] += 1;
            }
        }

        let mut queue = VecDeque::new();
        for i in 0..n {
            if in_degree[i] == 0 {
                queue.push_back(i);
            }
        }

        while !queue.is_empty() {
            let mut current_level = Vec::new();
            let level_size = queue.len();

            for _ in 0..level_size {
                let node = queue.pop_front().unwrap();
                current_level.push(node);

                if let Some(neighbors) = graph.get(&node) {
                    for &neighbor in neighbors {
                        in_degree[neighbor] -= 1;
                        if in_degree[neighbor] == 0 {
                            queue.push_back(neighbor);
                        }
                    }
                }
            }

            levels.push(current_level);
        }

        // Check if all nodes were processed (no cycles)
        let total_processed: usize = levels.iter().map(|level| level.len()).sum();
        if total_processed != n {
            return Err(SklearsError::InvalidInput(
                "Ordering constraints contain cycles".to_string(),
            ));
        }

        Ok(levels)
    }

    /// Check if this is a valid lattice (every pair has unique lub and glb)
    pub fn verify_lattice_properties(&self) -> Result<bool> {
        // For large lattices, this verification can be expensive
        // For now, we'll do a basic check that the structure is a valid partial order
        // A full lattice verification would require checking unique lub/glb for all pairs

        // Check that the transitive closure is consistent
        for i in 0..self.n_elements {
            for &j in &self.transitive_closure[&i] {
                // If i ≤ j, then for any k where j ≤ k, we should have i ≤ k
                for &k in &self.transitive_closure[&j] {
                    if !self.transitive_closure[&i].contains(&k) {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    /// Check if element i ≤ element j in the lattice
    pub fn is_ordered(&self, i: usize, j: usize) -> bool {
        i == j || self.transitive_closure[&i].contains(&j)
    }

    /// Get all elements that are ≤ given element
    pub fn get_lower_elements(&self, element: usize) -> Vec<usize> {
        let mut lower = Vec::new();
        for i in 0..self.n_elements {
            if self.is_ordered(i, element) {
                lower.push(i);
            }
        }
        lower
    }

    /// Get all elements that are ≥ given element
    pub fn get_upper_elements(&self, element: usize) -> Vec<usize> {
        let mut upper = Vec::new();
        for i in 0..self.n_elements {
            if self.is_ordered(element, i) {
                upper.push(i);
            }
        }
        upper
    }
}

impl LatticeOrderIsotonicRegression<Untrained> {
    /// Create a new lattice-order isotonic regression model
    pub fn new() -> Self {
        Self {
            ordering_constraints: Vec::new(),
            loss: LossFunction::SquaredLoss,
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            verify_lattice_properties: false,
            fitted_values_: None,
            lattice_structure_: None,
            _state: PhantomData,
        }
    }

    /// Set ordering constraints
    pub fn ordering_constraints(mut self, constraints: Vec<(usize, usize)>) -> Self {
        self.ordering_constraints = constraints;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set whether to verify lattice properties
    pub fn verify_lattice_properties(mut self, verify: bool) -> Self {
        self.verify_lattice_properties = verify;
        self
    }

    /// Create from a grid lattice (useful for multidimensional isotonic regression)
    pub fn from_grid(dimensions: &[usize]) -> Self {
        let mut constraints = Vec::new();
        let mut flat_index = 0;
        let total_elements: usize = dimensions.iter().product();

        // Helper function to convert multi-dimensional index to flat index
        let to_flat = |indices: &[usize], dims: &[usize]| -> usize {
            let mut flat = 0;
            let mut stride = 1;
            for i in (0..indices.len()).rev() {
                flat += indices[i] * stride;
                stride *= dims[i];
            }
            flat
        };

        // Generate constraints for grid lattice
        let n_dims = dimensions.len();
        for flat in 0..total_elements {
            // Convert flat index back to multi-dimensional index
            let mut indices = vec![0; n_dims];
            let mut temp = flat;
            for i in (0..n_dims).rev() {
                indices[i] = temp % dimensions[i];
                temp /= dimensions[i];
            }

            // Add constraints for each dimension (i, j, ..., i+1, j, ...)
            for dim in 0..n_dims {
                if indices[dim] + 1 < dimensions[dim] {
                    let mut next_indices = indices.clone();
                    next_indices[dim] += 1;
                    let next_flat = to_flat(&next_indices, dimensions);
                    constraints.push((flat, next_flat));
                }
            }
        }

        Self::new().ordering_constraints(constraints)
    }
}

impl Default for LatticeOrderIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LatticeOrderIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for LatticeOrderIsotonicRegression<Untrained> {
    type Fitted = LatticeOrderIsotonicRegression<Trained>;

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

        // Build lattice structure
        let lattice = LatticeStructure::new(n, &self.ordering_constraints)?;

        // Verify lattice properties if requested
        if self.verify_lattice_properties && !lattice.verify_lattice_properties()? {
            return Err(SklearsError::InvalidInput(
                "Ordering constraints do not form a valid lattice".to_string(),
            ));
        }

        // Solve lattice-order isotonic regression
        let fitted_values = self.solve_lattice_isotonic(y, &lattice)?;

        Ok(LatticeOrderIsotonicRegression {
            ordering_constraints: self.ordering_constraints,
            loss: self.loss,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            verify_lattice_properties: self.verify_lattice_properties,
            fitted_values_: Some(fitted_values),
            lattice_structure_: Some(lattice),
            _state: PhantomData,
        })
    }
}

impl LatticeOrderIsotonicRegression<Untrained> {
    /// Solve lattice-order isotonic regression using projected gradient descent
    fn solve_lattice_isotonic(
        &self,
        y: &Array1<Float>,
        lattice: &LatticeStructure,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut fitted = y.clone();

        for iteration in 0..self.max_iter {
            let prev_fitted = fitted.clone();

            // Gradient descent step
            let gradient = self.compute_gradient(&fitted, y)?;
            for i in 0..n {
                fitted[i] -= self.learning_rate * gradient[i];
            }

            // Project onto lattice constraints
            fitted = self.project_onto_lattice_constraints(&fitted, lattice)?;

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

    /// Compute gradient of the loss function
    fn compute_gradient(&self, fitted: &Array1<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let n = fitted.len();
        let mut gradient = Array1::zeros(n);

        match self.loss {
            LossFunction::SquaredLoss => {
                for i in 0..n {
                    gradient[i] = 2.0 * (fitted[i] - y[i]);
                }
            }
            LossFunction::AbsoluteLoss => {
                for i in 0..n {
                    gradient[i] = if fitted[i] > y[i] {
                        1.0
                    } else if fitted[i] < y[i] {
                        -1.0
                    } else {
                        0.0
                    };
                }
            }
            LossFunction::HuberLoss { delta } => {
                for i in 0..n {
                    let diff = fitted[i] - y[i];
                    if diff.abs() <= delta {
                        gradient[i] = diff;
                    } else {
                        gradient[i] = delta * diff.signum();
                    }
                }
            }
            LossFunction::QuantileLoss { quantile } => {
                for i in 0..n {
                    let diff = fitted[i] - y[i];
                    gradient[i] = if diff > 0.0 { quantile } else { quantile - 1.0 };
                }
            }
        }

        Ok(gradient)
    }

    /// Project values onto lattice constraints using level-wise approach
    fn project_onto_lattice_constraints(
        &self,
        values: &Array1<Float>,
        lattice: &LatticeStructure,
    ) -> Result<Array1<Float>> {
        let mut projected = values.clone();

        // Process each level from bottom to top
        for level in &lattice.levels {
            // Within each level, apply constraints
            for &node in level {
                // Get all elements that should be ≤ this node
                let lower_elements = lattice.get_lower_elements(node);

                // Get all elements that should be ≥ this node
                let upper_elements = lattice.get_upper_elements(node);

                // Find the range of valid values for this node
                let mut min_val = Float::NEG_INFINITY;
                let mut max_val = Float::INFINITY;

                for &lower in &lower_elements {
                    if lower != node {
                        min_val = min_val.max(projected[lower]);
                    }
                }

                for &upper in &upper_elements {
                    if upper != node {
                        max_val = max_val.min(projected[upper]);
                    }
                }

                // Clamp the value to the valid range
                if min_val != Float::NEG_INFINITY || max_val != Float::INFINITY {
                    projected[node] = projected[node].max(min_val).min(max_val);
                }
            }
        }

        // Additional pass to handle any remaining violations using PAV-like approach
        let mut changed = true;
        let mut iterations = 0;
        while changed && iterations < 100 {
            changed = false;
            iterations += 1;

            for (&i, successors) in &lattice.ordering_graph {
                for &j in successors {
                    if projected[i] > projected[j] {
                        // Violation: use pooling
                        let avg = (projected[i] + projected[j]) / 2.0;
                        projected[i] = avg;
                        projected[j] = avg;
                        changed = true;
                    }
                }
            }
        }

        Ok(projected)
    }
}

impl LatticeOrderIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }

    /// Get lattice structure
    pub fn lattice_structure(&self) -> &LatticeStructure {
        self.lattice_structure_.as_ref().unwrap()
    }

    /// Validate that fitted values satisfy lattice constraints
    pub fn validate_constraints(&self) -> bool {
        let fitted = self.fitted_values();
        let lattice = self.lattice_structure();

        for (&i, successors) in &lattice.ordering_graph {
            for &j in successors {
                if fitted[i] > fitted[j] + self.tolerance {
                    return false;
                }
            }
        }

        true
    }
}

impl Predict<Array1<Float>, Array1<Float>> for LatticeOrderIsotonicRegression<Trained> {
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

/// Convenience function for lattice-order isotonic regression
pub fn lattice_order_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    ordering_constraints: Vec<(usize, usize)>,
) -> Result<Array1<Float>> {
    let regressor = LatticeOrderIsotonicRegression::new()
        .ordering_constraints(ordering_constraints);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}

/// Adaptive weighting scheme for robust isotonic regression
#[derive(Debug, Clone, Copy, PartialEq)]
/// AdaptiveWeightingScheme
pub enum AdaptiveWeightingScheme {
    /// Huber-style robust weighting
    Huber { threshold: Float },
    /// Bisquare (Tukey) weighting
    Bisquare { threshold: Float },
    /// Least Absolute Deviation weighting
    LAD,
    /// Iteratively Reweighted Least Squares
    IRLS { power: Float },
    /// Outlier detection based weighting
    OutlierDetection { threshold: Float },
    /// Heteroscedasticity adaptive weighting
    Heteroscedastic,
}

/// Adaptive weighting isotonic regression
///
/// Automatically adjusts sample weights during fitting to improve robustness
/// and handle various data characteristics like outliers and heteroscedasticity.
#[derive(Debug, Clone)]
/// AdaptiveWeightingIsotonicRegression
pub struct AdaptiveWeightingIsotonicRegression<State = Untrained> {
    /// Whether the function should be increasing
    pub increasing: bool,
    /// Adaptive weighting scheme
    pub weighting_scheme: AdaptiveWeightingScheme,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Lower bound on output
    pub y_min: Option<Float>,
    /// Upper bound on output
    pub y_max: Option<Float>,
    /// Out-of-bounds handling
    pub out_of_bounds: String,
    /// Maximum number of iterations for adaptive weighting
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Minimum weight (prevents complete elimination of samples)
    pub min_weight: Float,
    /// Initial weights
    pub initial_weights: Option<Array1<Float>>,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    weights_: Option<Array1<Float>>,
    iteration_weights_: Option<Vec<Array1<Float>>>,

    _state: PhantomData<State>,
}

impl AdaptiveWeightingIsotonicRegression<Untrained> {
    /// Create a new adaptive weighting isotonic regression model
    pub fn new() -> Self {
        Self {
            increasing: true,
            weighting_scheme: AdaptiveWeightingScheme::Huber { threshold: 1.345 },
            fit_intercept: true,
            y_min: None,
            y_max: None,
            out_of_bounds: "nan".to_string(),
            max_iterations: 50,
            tolerance: 1e-6,
            min_weight: 1e-6,
            initial_weights: None,
            x_: None,
            y_: None,
            weights_: None,
            iteration_weights_: None,
            _state: PhantomData,
        }
    }

    /// Set whether the function should be increasing
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.increasing = increasing;
        self
    }

    /// Set adaptive weighting scheme
    pub fn weighting_scheme(mut self, scheme: AdaptiveWeightingScheme) -> Self {
        self.weighting_scheme = scheme;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set lower bound on output
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    /// Set upper bound on output
    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set out-of-bounds handling
    pub fn out_of_bounds(mut self, out_of_bounds: String) -> Self {
        self.out_of_bounds = out_of_bounds;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set minimum weight
    pub fn min_weight(mut self, min_weight: Float) -> Self {
        self.min_weight = min_weight;
        self
    }

    /// Set initial weights
    pub fn initial_weights(mut self, weights: Array1<Float>) -> Self {
        self.initial_weights = Some(weights);
        self
    }

    /// Huber weighting with adaptive threshold
    pub fn huber_adaptive(threshold: Float) -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::Huber { threshold })
    }

    /// Bisquare weighting with adaptive threshold
    pub fn bisquare_adaptive(threshold: Float) -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::Bisquare { threshold })
    }

    /// LAD (Least Absolute Deviation) weighting
    pub fn lad_adaptive() -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::LAD)
    }

    /// IRLS (Iteratively Reweighted Least Squares) weighting
    pub fn irls_adaptive(power: Float) -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::IRLS { power })
    }

    /// Outlier detection based weighting
    pub fn outlier_detection_adaptive(threshold: Float) -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::OutlierDetection { threshold })
    }

    /// Heteroscedasticity adaptive weighting
    pub fn heteroscedastic_adaptive() -> Self {
        Self::new().weighting_scheme(AdaptiveWeightingScheme::Heteroscedastic)
    }
}

impl Default for AdaptiveWeightingIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AdaptiveWeightingIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for AdaptiveWeightingIsotonicRegression<Untrained> {
    type Fitted = AdaptiveWeightingIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "X and y cannot be empty".to_string(),
            ));
        }

        // Sort by x values
        let mut indices: Vec<usize> = (0..x.len()).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let mut x_sorted = Array1::zeros(x.len());
        let mut y_sorted = Array1::zeros(y.len());

        for (i, &idx) in indices.iter().enumerate() {
            x_sorted[i] = x[idx];
            y_sorted[i] = y[idx];
        }

        // Initialize weights
        let mut weights = if let Some(ref initial) = self.initial_weights {
            initial.clone()
        } else {
            Array1::ones(y_sorted.len())
        };

        // Iterative adaptive weighting
        let (final_fitted, final_weights, iteration_weights) =
            self.iterative_adaptive_weighting(&x_sorted, &y_sorted, weights)?;

        Ok(AdaptiveWeightingIsotonicRegression {
            increasing: self.increasing,
            weighting_scheme: self.weighting_scheme,
            fit_intercept: self.fit_intercept,
            y_min: self.y_min,
            y_max: self.y_max,
            out_of_bounds: self.out_of_bounds,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            min_weight: self.min_weight,
            initial_weights: self.initial_weights,
            x_: Some(x_sorted),
            y_: Some(final_fitted),
            weights_: Some(final_weights),
            iteration_weights_: Some(iteration_weights),
            _state: PhantomData,
        })
    }
}

impl AdaptiveWeightingIsotonicRegression<Untrained> {
    /// Iterative adaptive weighting algorithm
    fn iterative_adaptive_weighting(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        mut weights: Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Vec<Array1<Float>>)> {
        let mut fitted = y.clone();
        let mut iteration_weights = Vec::new();
        iteration_weights.push(weights.clone());

        for iteration in 0..self.max_iterations {
            // Fit weighted isotonic regression
            fitted = self.fit_weighted_isotonic(y, &weights)?;

            // Compute residuals
            let residuals = y - &fitted;

            // Update weights based on residuals
            let new_weights = self.compute_adaptive_weights(&residuals, &fitted)?;

            // Check convergence
            let weight_change = weights
                .iter()
                .zip(new_weights.iter())
                .map(|(old, new)| (old - new).abs())
                .fold(0.0_f64, |max: f64, x| max.max(x));

            weights = new_weights;
            iteration_weights.push(weights.clone());

            if weight_change < self.tolerance {
                break;
            }
        }

        // Final fit with converged weights
        fitted = self.fit_weighted_isotonic(y, &weights)?;

        Ok((fitted, weights, iteration_weights))
    }

    /// Fit weighted isotonic regression
    fn fit_weighted_isotonic(
        &self,
        y: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        // Use weighted Pool Adjacent Violators algorithm
        self.weighted_pool_adjacent_violators(y, weights)
    }

    /// Weighted Pool Adjacent Violators algorithm
    fn weighted_pool_adjacent_violators(
        &self,
        y: &Array1<Float>,
        weights: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = y.len();
        let mut result = y.clone();
        let mut w = weights.clone();

        if self.increasing {
            // Increasing case
            let mut i = 0;
            while i < n - 1 {
                if result[i] > result[i + 1] {
                    // Pool adjacent violators
                    let mut j = i + 1;
                    let mut sum_wy = w[i] * result[i] + w[i + 1] * result[i + 1];
                    let mut sum_w = w[i] + w[i + 1];

                    // Find all violators in sequence
                    while j < n - 1 && (sum_wy / sum_w) > result[j + 1] {
                        j += 1;
                        sum_wy += w[j] * result[j];
                        sum_w += w[j];
                    }

                    // Set pooled value
                    let pooled_value = sum_wy / sum_w;
                    let pooled_weight = sum_w;

                    for k in i..=j {
                        result[k] = pooled_value;
                        w[k] = pooled_weight / (j - i + 1) as Float;
                    }

                    // Restart from beginning (could be optimized)
                    i = 0;
                } else {
                    i += 1;
                }
            }
        } else {
            // Decreasing case
            let mut i = 0;
            while i < n - 1 {
                if result[i] < result[i + 1] {
                    // Pool adjacent violators
                    let mut j = i + 1;
                    let mut sum_wy = w[i] * result[i] + w[i + 1] * result[i + 1];
                    let mut sum_w = w[i] + w[i + 1];

                    // Find all violators in sequence
                    while j < n - 1 && (sum_wy / sum_w) < result[j + 1] {
                        j += 1;
                        sum_wy += w[j] * result[j];
                        sum_w += w[j];
                    }

                    // Set pooled value
                    let pooled_value = sum_wy / sum_w;
                    let pooled_weight = sum_w;

                    for k in i..=j {
                        result[k] = pooled_value;
                        w[k] = pooled_weight / (j - i + 1) as Float;
                    }

                    // Restart from beginning
                    i = 0;
                } else {
                    i += 1;
                }
            }
        }

        Ok(result)
    }

    /// Compute adaptive weights based on residuals
    fn compute_adaptive_weights(
        &self,
        residuals: &Array1<Float>,
        fitted: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = residuals.len();
        let mut weights = Array1::zeros(n);

        match self.weighting_scheme {
            AdaptiveWeightingScheme::Huber { threshold } => {
                // Compute MAD (Median Absolute Deviation) for scale
                let mut abs_residuals: Vec<Float> = residuals.iter().map(|&r| r.abs()).collect();
                abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mad = abs_residuals[n / 2] * 1.4826; // Scale factor for normal distribution

                for i in 0..n {
                    let standardized_residual = residuals[i].abs() / mad.max(1e-10);
                    weights[i] = if standardized_residual <= threshold {
                        1.0
                    } else {
                        threshold / standardized_residual
                    };
                    weights[i] = weights[i].max(self.min_weight);
                }
            }

            AdaptiveWeightingScheme::Bisquare { threshold } => {
                // Compute scale (MAD)
                let mut abs_residuals: Vec<Float> = residuals.iter().map(|&r| r.abs()).collect();
                abs_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let mad = abs_residuals[n / 2] * 1.4826;

                for i in 0..n {
                    let standardized_residual = residuals[i].abs() / mad.max(1e-10);
                    weights[i] = if standardized_residual <= threshold {
                        let u = standardized_residual / threshold;
                        (1.0 - u * u).powi(2)
                    } else {
                        0.0
                    };
                    weights[i] = weights[i].max(self.min_weight);
                }
            }

            AdaptiveWeightingScheme::LAD => {
                // LAD weights (inverse of absolute residuals)
                for i in 0..n {
                    weights[i] = 1.0 / residuals[i].abs().max(1e-10);
                    weights[i] = weights[i].max(self.min_weight);
                }
            }

            AdaptiveWeightingScheme::IRLS { power } => {
                // IRLS weights
                for i in 0..n {
                    weights[i] = 1.0 / residuals[i].abs().powf(power).max(1e-10);
                    weights[i] = weights[i].max(self.min_weight);
                }
            }

            AdaptiveWeightingScheme::OutlierDetection { threshold } => {
                // Outlier detection using z-score
                let mean_residual = residuals.mean().unwrap_or(0.0);
                let std_residual = residuals.std(0.0);

                for i in 0..n {
                    let z_score = ((residuals[i] - mean_residual) / std_residual.max(1e-10)).abs();
                    weights[i] = if z_score <= threshold { 1.0 } else { 0.1 };
                    weights[i] = weights[i].max(self.min_weight);
                }
            }

            AdaptiveWeightingScheme::Heteroscedastic => {
                // Weights inversely proportional to local variance
                let window_size = (n / 10).max(3);

                for i in 0..n {
                    let start = i.saturating_sub(window_size / 2);
                    let end = (i + window_size / 2 + 1).min(n);

                    // Local variance estimate
                    let local_residuals = &residuals.slice(s![start..end]);
                    let local_var = local_residuals.var(0.0);

                    weights[i] = 1.0 / local_var.max(1e-10);
                    weights[i] = weights[i].max(self.min_weight);
                }
            }
        }

        // Normalize weights to sum to n (optional, maintains scale)
        let weight_sum = weights.sum();
        if weight_sum > 0.0 {
            weights = weights * (n as Float / weight_sum);
        }

        Ok(weights)
    }
}

impl Predict<Array1<Float>, Array1<Float>> for AdaptiveWeightingIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let x_ = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        let y_ = self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let mut predictions = Array1::zeros(x.len());

        for (i, &xi) in x.iter().enumerate() {
            // Linear interpolation or extrapolation
            if xi <= x_[0] {
                predictions[i] = y_[0];
            } else if xi >= x_[x_.len() - 1] {
                predictions[i] = y_[y_.len() - 1];
            } else {
                // Find the interval
                let mut left_idx = 0;
                for j in 0..x_.len() - 1 {
                    if x_[j] <= xi && xi <= x_[j + 1] {
                        left_idx = j;
                        break;
                    }
                }

                // Linear interpolation
                let x1 = x_[left_idx];
                let x2 = x_[left_idx + 1];
                let y1 = y_[left_idx];
                let y2 = y_[left_idx + 1];

                if (x2 - x1).abs() < 1e-10 {
                    predictions[i] = y1;
                } else {
                    predictions[i] = y1 + (y2 - y1) * (xi - x1) / (x2 - x1);
                }
            }

            // Apply bounds if specified
            if let Some(y_min) = self.y_min {
                predictions[i] = predictions[i].max(y_min);
            }
            if let Some(y_max) = self.y_max {
                predictions[i] = predictions[i].min(y_max);
            }
        }

        Ok(predictions)
    }
}

impl AdaptiveWeightingIsotonicRegression<Trained> {
    /// Get the final weights used
    pub fn weights(&self) -> &Array1<Float> {
        self.weights_.as_ref().unwrap()
    }

    /// Get weights from all iterations
    pub fn iteration_weights(&self) -> &[Array1<Float>] {
        self.iteration_weights_.as_ref().unwrap()
    }

    /// Get the fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.y_.as_ref().unwrap()
    }

    /// Get the training x values
    pub fn x_values(&self) -> &Array1<Float> {
        self.x_.as_ref().unwrap()
    }

    /// Compute influence of each sample on the final fit
    pub fn compute_influence_measures(&self) -> Result<Array1<Float>> {
        let weights = self.weights();
        let n = weights.len();

        // Simple influence measure: weight * leverage
        // For isotonic regression, leverage is roughly 1/n for interior points
        let mut influence = Array1::zeros(n);

        for i in 0..n {
            // Higher weight means more influence
            // Boundary points have higher leverage
            let boundary_factor = if i == 0 || i == n - 1 { 2.0 } else { 1.0 };
            influence[i] = weights[i] * boundary_factor / (n as Float);
        }

        Ok(influence)
    }

    /// Identify outliers based on final weights
    pub fn identify_outliers(&self, threshold: Float) -> Vec<usize> {
        let weights = self.weights();
        let mean_weight = weights.mean().unwrap_or(1.0);

        weights
            .iter()
            .enumerate()
            .filter(|(_, &w)| w < threshold * mean_weight)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Convenience function for adaptive weighting isotonic regression
pub fn adaptive_weighting_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    weighting_scheme: AdaptiveWeightingScheme,
    increasing: bool,
) -> Result<Array1<Float>> {
    let regressor = AdaptiveWeightingIsotonicRegression::new()
        .weighting_scheme(weighting_scheme)
        .increasing(increasing);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}
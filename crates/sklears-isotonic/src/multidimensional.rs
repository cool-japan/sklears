//! Multi-dimensional isotonic regression implementations

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Multi-dimensional isotonic regression model
#[derive(Debug, Clone)]
/// MultiDimensionalIsotonicRegression
pub struct MultiDimensionalIsotonicRegression<State = Untrained> {
    /// Monotonicity constraints for each dimension
    pub constraints: Vec<MonotonicityConstraint>,
    /// Lower bound on the output
    pub y_min: Option<Float>,
    /// Upper bound on the output
    pub y_max: Option<Float>,
    /// Loss function for robust regression
    pub loss: LossFunction,
    /// Whether to use separable optimization
    pub separable: bool,
    /// Maximum iterations for non-separable optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    training_x_: Option<Array2<Float>>,
    training_y_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl MultiDimensionalIsotonicRegression<Untrained> {
    pub fn new(n_dimensions: usize) -> Self {
        Self {
            constraints: vec![MonotonicityConstraint::Global { increasing: true }; n_dimensions],
            y_min: None,
            y_max: None,
            loss: LossFunction::SquaredLoss,
            separable: true,
            max_iterations: 100,
            tolerance: 1e-6,
            fitted_values_: None,
            training_x_: None,
            training_y_: None,
            _state: PhantomData,
        }
    }

    /// Set monotonicity constraints for all dimensions
    pub fn constraints(mut self, constraints: Vec<MonotonicityConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set bounds
    pub fn y_min(mut self, y_min: Float) -> Self {
        self.y_min = Some(y_min);
        self
    }

    pub fn y_max(mut self, y_max: Float) -> Self {
        self.y_max = Some(y_max);
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set whether to use separable optimization
    pub fn separable(mut self, separable: bool) -> Self {
        self.separable = separable;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(mut self, max_iterations: usize, tolerance: Float) -> Self {
        self.max_iterations = max_iterations;
        self.tolerance = tolerance;
        self
    }
}

impl Default for MultiDimensionalIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Estimator for MultiDimensionalIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for MultiDimensionalIsotonicRegression<Untrained> {
    type Fitted = MultiDimensionalIsotonicRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if x.ncols() != self.constraints.len() {
            return Err(SklearsError::InvalidInput(
                "Number of constraints must match number of features".to_string(),
            ));
        }

        let fitted_values = if self.separable {
            fit_separable_multidimensional(x, y, &self)?
        } else {
            fit_non_separable_multidimensional(x, y, &self)?
        };

        Ok(MultiDimensionalIsotonicRegression {
            constraints: self.constraints,
            y_min: self.y_min,
            y_max: self.y_max,
            loss: self.loss,
            separable: self.separable,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            fitted_values_: Some(fitted_values),
            training_x_: Some(x.clone()),
            training_y_: Some(y.clone()),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>> for MultiDimensionalIsotonicRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.constraints.len() {
            return Err(SklearsError::InvalidInput(
                "X must have the same number of features as training data".to_string(),
            ));
        }

        let training_x = self.training_x_.as_ref().unwrap();
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());

        for (i, test_point) in x.outer_iter().enumerate() {
            predictions[i] = interpolate_multidimensional(training_x, fitted_values, &test_point)?;
        }

        Ok(predictions)
    }
}

impl MultiDimensionalIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }
}

/// Separable multi-dimensional isotonic regression
#[derive(Debug, Clone)]
/// SeparableMultiDimensionalIsotonicRegression
pub struct SeparableMultiDimensionalIsotonicRegression<State = Untrained> {
    /// Monotonicity constraints for each dimension
    pub constraints: Vec<bool>,
    /// Loss function
    pub loss: LossFunction,
    /// Bounds
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,

    // Fitted attributes
    fitted_values_: Option<Array1<Float>>,
    training_x_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl SeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Create new separable multi-dimensional isotonic regression
    pub fn new(n_dimensions: usize) -> Self {
        Self {
            constraints: vec![true; n_dimensions],
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            fitted_values_: None,
            training_x_: None,
            _state: PhantomData,
        }
    }

    /// Set constraints (true = increasing, false = decreasing)
    pub fn constraints(mut self, constraints: Vec<bool>) -> Self {
        self.constraints = constraints;
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
}

impl Default for SeparableMultiDimensionalIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new(2)
    }
}

impl Estimator for SeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Fitted = SeparableMultiDimensionalIsotonicRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if x.ncols() != self.constraints.len() {
            return Err(SklearsError::InvalidInput(
                "Number of constraints must match number of features".to_string(),
            ));
        }

        // Fit independent univariate isotonic regression for each dimension
        let mut fitted_values = y.clone();

        for (dim, &increasing) in self.constraints.iter().enumerate() {
            let feature_col = x.column(dim);
            let iso_fitted =
                fit_univariate_isotonic(&feature_col, &fitted_values, increasing, &self.loss)?;
            fitted_values = iso_fitted;
        }

        // Apply bounds if specified
        if let Some(min_val) = self.y_min {
            fitted_values.mapv_inplace(|v| v.max(min_val));
        }
        if let Some(max_val) = self.y_max {
            fitted_values.mapv_inplace(|v| v.min(max_val));
        }

        Ok(SeparableMultiDimensionalIsotonicRegression {
            constraints: self.constraints,
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            fitted_values_: Some(fitted_values),
            training_x_: Some(x.clone()),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>>
    for SeparableMultiDimensionalIsotonicRegression<Trained>
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        if x.ncols() != self.constraints.len() {
            return Err(SklearsError::InvalidInput(
                "X must have the same number of features as training data".to_string(),
            ));
        }

        let training_x = self.training_x_.as_ref().unwrap();
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());

        for (i, test_point) in x.outer_iter().enumerate() {
            predictions[i] = interpolate_multidimensional(training_x, fitted_values, &test_point)?;
        }

        Ok(predictions)
    }
}

impl SeparableMultiDimensionalIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }
}

/// Non-separable multi-dimensional isotonic regression
#[derive(Debug, Clone)]
/// NonSeparableMultiDimensionalIsotonicRegression
pub struct NonSeparableMultiDimensionalIsotonicRegression<State = Untrained> {
    /// Partial order constraints (i, j) means x[i] <= x[j] implies y[i] <= y[j]
    pub partial_order: Vec<(usize, usize)>,
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
    training_x_: Option<Array2<Float>>,

    _state: PhantomData<State>,
}

impl NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    /// Create new non-separable multi-dimensional isotonic regression
    pub fn new() -> Self {
        Self {
            partial_order: Vec::new(),
            loss: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            max_iterations: 100,
            tolerance: 1e-6,
            fitted_values_: None,
            training_x_: None,
            _state: PhantomData,
        }
    }

    /// Set partial order constraints
    pub fn partial_order(mut self, partial_order: Vec<(usize, usize)>) -> Self {
        self.partial_order = partial_order;
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

impl Default for NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NonSeparableMultiDimensionalIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>>
    for NonSeparableMultiDimensionalIsotonicRegression<Untrained>
{
    type Fitted = NonSeparableMultiDimensionalIsotonicRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Generate partial order constraints from coordinate-wise ordering
        let partial_order = if self.partial_order.is_empty() {
            generate_coordinate_wise_order(x)?
        } else {
            self.partial_order.clone()
        };

        // Fit using constrained optimization
        let fitted_values = fit_with_partial_order(
            y,
            &partial_order,
            &self.loss,
            self.max_iterations,
            self.tolerance,
        )?;

        // Apply bounds if specified
        let mut bounded_values = fitted_values;
        if let Some(min_val) = self.y_min {
            bounded_values.mapv_inplace(|v| v.max(min_val));
        }
        if let Some(max_val) = self.y_max {
            bounded_values.mapv_inplace(|v| v.min(max_val));
        }

        Ok(NonSeparableMultiDimensionalIsotonicRegression {
            partial_order,
            loss: self.loss,
            y_min: self.y_min,
            y_max: self.y_max,
            max_iterations: self.max_iterations,
            tolerance: self.tolerance,
            fitted_values_: Some(bounded_values),
            training_x_: Some(x.clone()),
            _state: PhantomData,
        })
    }
}

impl Predict<Array2<Float>, Array1<Float>>
    for NonSeparableMultiDimensionalIsotonicRegression<Trained>
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let training_x = self.training_x_.as_ref().unwrap();
        let fitted_values = self.fitted_values_.as_ref().unwrap();

        let mut predictions = Array1::zeros(x.nrows());

        for (i, test_point) in x.outer_iter().enumerate() {
            predictions[i] = interpolate_multidimensional(training_x, fitted_values, &test_point)?;
        }

        Ok(predictions)
    }
}

impl NonSeparableMultiDimensionalIsotonicRegression<Trained> {
    /// Get fitted values
    pub fn fitted_values(&self) -> &Array1<Float> {
        self.fitted_values_.as_ref().unwrap()
    }
}

// Helper functions

/// Fit separable multi-dimensional isotonic regression
fn fit_separable_multidimensional(
    x: &Array2<Float>,
    y: &Array1<Float>,
    config: &MultiDimensionalIsotonicRegression<Untrained>,
) -> Result<Array1<Float>> {
    let mut fitted_values = y.clone();

    // Apply isotonic regression sequentially for each dimension
    for (dim, constraint) in config.constraints.iter().enumerate() {
        let feature_col = x.column(dim);

        let increasing = match constraint {
            MonotonicityConstraint::Global { increasing } => *increasing,
            _ => true, // Default to increasing for other constraints
        };

        fitted_values =
            fit_univariate_isotonic(&feature_col, &fitted_values, increasing, &config.loss)?;
    }

    // Apply bounds
    if let Some(min_val) = config.y_min {
        fitted_values.mapv_inplace(|v| v.max(min_val));
    }
    if let Some(max_val) = config.y_max {
        fitted_values.mapv_inplace(|v| v.min(max_val));
    }

    Ok(fitted_values)
}

/// Fit non-separable multi-dimensional isotonic regression
fn fit_non_separable_multidimensional(
    x: &Array2<Float>,
    y: &Array1<Float>,
    config: &MultiDimensionalIsotonicRegression<Untrained>,
) -> Result<Array1<Float>> {
    // Generate partial order constraints
    let partial_order = generate_coordinate_wise_order(x)?;

    // Fit using iterative projection
    let fitted_values = fit_with_partial_order(
        y,
        &partial_order,
        &config.loss,
        config.max_iterations,
        config.tolerance,
    )?;

    // Apply bounds
    let mut bounded_values = fitted_values;
    if let Some(min_val) = config.y_min {
        bounded_values.mapv_inplace(|v| v.max(min_val));
    }
    if let Some(max_val) = config.y_max {
        bounded_values.mapv_inplace(|v| v.min(max_val));
    }

    Ok(bounded_values)
}

/// Fit univariate isotonic regression for a single dimension
fn fit_univariate_isotonic(
    x: &ArrayView1<Float>,
    y: &Array1<Float>,
    increasing: bool,
    loss: &LossFunction,
) -> Result<Array1<Float>> {
    use crate::utils::*;

    // Sort indices by x values
    let mut indices: Vec<usize> = (0..x.len()).collect();
    indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

    let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();

    // Apply PAVA algorithm
    let fitted_sorted = match loss {
        LossFunction::SquaredLoss => pava_algorithm(&sorted_y, None, increasing),
        LossFunction::AbsoluteLoss => pava_l1(&sorted_y, None, increasing),
        LossFunction::HuberLoss { delta } => pava_huber(&sorted_y, None, increasing, *delta),
        LossFunction::QuantileLoss { quantile } => {
            pava_quantile(&sorted_y, None, increasing, *quantile)
        }
    };

    // Unsort the fitted values
    let mut fitted_values = Array1::zeros(y.len());
    for (orig_idx, &sort_idx) in indices.iter().enumerate() {
        fitted_values[sort_idx] = fitted_sorted[orig_idx];
    }

    Ok(fitted_values)
}

/// Generate coordinate-wise partial order constraints
fn generate_coordinate_wise_order(x: &Array2<Float>) -> Result<Vec<(usize, usize)>> {
    let n = x.nrows();
    let mut constraints = Vec::new();

    for i in 0..n {
        for j in 0..n {
            if i != j {
                // Check if x[i] <= x[j] coordinate-wise
                let mut dominates = true;
                for k in 0..x.ncols() {
                    if x[[i, k]] > x[[j, k]] {
                        dominates = false;
                        break;
                    }
                }

                if dominates {
                    constraints.push((i, j));
                }
            }
        }
    }

    Ok(constraints)
}

/// Fit isotonic regression with partial order constraints using iterative projection
fn fit_with_partial_order(
    y: &Array1<Float>,
    partial_order: &[(usize, usize)],
    loss: &LossFunction,
    max_iterations: usize,
    tolerance: Float,
) -> Result<Array1<Float>> {
    let mut fitted = y.clone();

    for iteration in 0..max_iterations {
        let prev_fitted = fitted.clone();

        // Project onto isotonic constraints
        for &(i, j) in partial_order {
            if fitted[i] > fitted[j] {
                // Average the violating values
                let avg = (fitted[i] + fitted[j]) / 2.0;
                fitted[i] = avg;
                fitted[j] = avg;
            }
        }

        // Check convergence
        let change = (&fitted - &prev_fitted).mapv(|x| x.abs()).sum();
        if change < tolerance {
            break;
        }

        if iteration == max_iterations - 1 {
            eprintln!("Warning: Maximum iterations reached in non-separable isotonic regression");
        }
    }

    Ok(fitted)
}

/// Multi-dimensional interpolation for predictions
fn interpolate_multidimensional(
    training_x: &Array2<Float>,
    training_y: &Array1<Float>,
    test_point: &ArrayView1<Float>,
) -> Result<Float> {
    if training_x.is_empty() || training_y.is_empty() {
        return Ok(Float::NAN);
    }

    // Find nearest neighbors for interpolation
    let mut distances: Vec<(usize, Float)> = Vec::new();

    for (i, train_point) in training_x.outer_iter().enumerate() {
        let distance = (&train_point - test_point).mapv(|x| x * x).sum().sqrt();
        distances.push((i, distance));
    }

    // Sort by distance
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Use inverse distance weighting with k nearest neighbors
    let k = (training_x.nrows().min(5)).max(1);
    let mut weighted_sum = 0.0;
    let mut weight_sum = 0.0;

    for i in 0..k {
        let (idx, distance) = distances[i];
        let weight = if distance < 1e-10 {
            return Ok(training_y[idx]); // Exact match
        } else {
            1.0 / distance
        };

        weighted_sum += weight * training_y[idx];
        weight_sum += weight;
    }

    if weight_sum > 0.0 {
        Ok(weighted_sum / weight_sum)
    } else {
        Ok(training_y[0]) // Fallback
    }
}

/// Convenience function for separable multi-dimensional isotonic regression
pub fn separable_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: Option<Vec<bool>>,
    loss: Option<LossFunction>,
) -> Result<Array1<Float>> {
    let n_dims = x.ncols();
    let constraints = constraints.unwrap_or_else(|| vec![true; n_dims]);

    let mut iso = SeparableMultiDimensionalIsotonicRegression::new(n_dims).constraints(constraints);

    if let Some(loss_fn) = loss {
        iso = iso.loss(loss_fn);
    }

    let fitted = iso.fit(x, y)?;
    Ok(fitted.fitted_values().clone())
}

/// Convenience function for non-separable multi-dimensional isotonic regression
pub fn non_separable_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    partial_order: Option<Vec<(usize, usize)>>,
    loss: Option<LossFunction>,
) -> Result<Array1<Float>> {
    let mut iso = NonSeparableMultiDimensionalIsotonicRegression::new();

    if let Some(order) = partial_order {
        iso = iso.partial_order(order);
    }

    if let Some(loss_fn) = loss {
        iso = iso.loss(loss_fn);
    }

    let fitted = iso.fit(x, y)?;
    Ok(fitted.fitted_values().clone())
}

//! Structured sparsity for isotonic regression
//!
//! This module implements structured sparsity regularization that encourages
//! sparsity at the group level rather than individual coefficient level.

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Structured sparsity type specification
#[derive(Debug, Clone, PartialEq)]
/// StructuredSparsityType
pub enum StructuredSparsityType {
    /// Group Lasso: L2 norm within groups, L1 norm between groups
    GroupLasso {

        groups: Vec<Vec<usize>>,

        lambda: Float,
    },
    /// Fused Lasso: Penalizes differences between adjacent coefficients
    FusedLasso { lambda: Float },
    /// Hierarchical sparsity: Nested group structure
    Hierarchical {

        group_hierarchy: Vec<(Vec<usize>, Float)>,
    },
    /// Elastic net with group structure
    GroupElasticNet {
        groups: Vec<Vec<usize>>,
        l1_ratio: Float,
        lambda: Float,
    },
    /// Total variation sparsity
    TotalVariation { lambda: Float },
    /// Graph Lasso: Sparsity based on graph structure
    GraphLasso {
        adjacency_matrix: Array2<Float>,
        lambda: Float,
    },
}

/// Structured sparse isotonic regression model
#[derive(Debug, Clone)]
/// StructuredSparseIsotonicRegression
pub struct StructuredSparseIsotonicRegression<State = Untrained> {
    /// Monotonicity constraint
    pub constraint: MonotonicityConstraint,
    /// Structured sparsity type and parameters
    pub sparsity_type: StructuredSparsityType,
    /// Loss function
    pub loss: LossFunction,
    /// Maximum iterations for optimization
    pub max_iter: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate for gradient descent
    pub learning_rate: Float,
    /// Global bounds
    pub y_min: Option<Float>,
    pub y_max: Option<Float>,

    // Fitted attributes
    x_: Option<Array1<Float>>,
    y_: Option<Array1<Float>>,
    active_groups_: Option<Vec<usize>>,
    sparsity_pattern_: Option<Array1<bool>>,

    _state: PhantomData<State>,
}

impl StructuredSparseIsotonicRegression<Untrained> {
    /// Create a new structured sparse isotonic regression model
    pub fn new() -> Self {
        Self {
            constraint: MonotonicityConstraint::Global { increasing: true },
            sparsity_type: StructuredSparsityType::GroupLasso {
                groups: vec![],
                lambda: 0.1,
            },
            loss: LossFunction::SquaredLoss,
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            y_min: None,
            y_max: None,
            x_: None,
            y_: None,
            active_groups_: None,
            sparsity_pattern_: None,
            _state: PhantomData,
        }
    }

    /// Set monotonicity constraint
    pub fn constraint(mut self, constraint: MonotonicityConstraint) -> Self {
        self.constraint = constraint;
        self
    }

    /// Set increasing constraint
    pub fn increasing(mut self, increasing: bool) -> Self {
        self.constraint = MonotonicityConstraint::Global { increasing };
        self
    }

    /// Set structured sparsity type
    pub fn sparsity_type(mut self, sparsity_type: StructuredSparsityType) -> Self {
        self.sparsity_type = sparsity_type;
        self
    }

    /// Set Group Lasso regularization
    pub fn group_lasso(mut self, groups: Vec<Vec<usize>>, lambda: Float) -> Self {
        self.sparsity_type = StructuredSparsityType::GroupLasso { groups, lambda };
        self
    }

    /// Set Fused Lasso regularization
    pub fn fused_lasso(mut self, lambda: Float) -> Self {
        self.sparsity_type = StructuredSparsityType::FusedLasso { lambda };
        self
    }

    /// Set hierarchical sparsity
    pub fn hierarchical_sparsity(mut self, group_hierarchy: Vec<(Vec<usize>, Float)>) -> Self {
        self.sparsity_type = StructuredSparsityType::Hierarchical { group_hierarchy };
        self
    }

    /// Set Group Elastic Net regularization
    pub fn group_elastic_net(
        mut self,
        groups: Vec<Vec<usize>>,
        l1_ratio: Float,
        lambda: Float,
    ) -> Self {
        self.sparsity_type = StructuredSparsityType::GroupElasticNet {
            groups,
            l1_ratio,
            lambda,
        };
        self
    }

    /// Set total variation sparsity
    pub fn total_variation_sparsity(mut self, lambda: Float) -> Self {
        self.sparsity_type = StructuredSparsityType::TotalVariation { lambda };
        self
    }

    /// Set Graph Lasso regularization
    pub fn graph_lasso(mut self, adjacency_matrix: Array2<Float>, lambda: Float) -> Self {
        self.sparsity_type = StructuredSparsityType::GraphLasso {
            adjacency_matrix,
            lambda,
        };
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

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set global bounds
    pub fn bounds(mut self, y_min: Option<Float>, y_max: Option<Float>) -> Self {
        self.y_min = y_min;
        self.y_max = y_max;
        self
    }
}

impl Default for StructuredSparseIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StructuredSparseIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for StructuredSparseIsotonicRegression<Untrained> {
    type Fitted = StructuredSparseIsotonicRegression<Trained>;

    fn fit(self, x: &Array1<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.len() != y.len() {
            return Err(SklearsError::InvalidInput(
                "x and y must have the same length".to_string(),
            ));
        }

        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "x and y cannot be empty".to_string(),
            ));
        }

        let n = x.len();

        // Sort data by x values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            // Use total_cmp for safe float comparison that handles NaN
            x[a].total_cmp(&x[b])
        });

        let sorted_x: Vec<Float> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();

        // Fit with structured sparsity
        let (fitted_y, active_groups, sparsity_pattern) =
            fit_structured_sparse(&sorted_x, &sorted_y, &self)?;

        Ok(StructuredSparseIsotonicRegression {
            constraint: self.constraint,
            sparsity_type: self.sparsity_type,
            loss: self.loss,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            y_min: self.y_min,
            y_max: self.y_max,
            x_: Some(Array1::from(sorted_x)),
            y_: Some(Array1::from(fitted_y)),
            active_groups_: Some(active_groups),
            sparsity_pattern_: Some(sparsity_pattern),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for StructuredSparseIsotonicRegression<Trained> {
    fn predict(&self, x: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_x = self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let fitted_y = self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;

        let predictions = crate::algorithms::linear_interpolate(fitted_x, fitted_y, x);
        Ok(predictions)
    }
}

impl StructuredSparseIsotonicRegression<Trained> {
    /// Get active groups after fitting
    pub fn active_groups(&self) -> Result<&[usize]> {
        self.active_groups_
            .as_ref()
            .map(|v| v.as_slice())
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "active_groups".to_string(),
            })
    }

    /// Get sparsity pattern (boolean array indicating which coefficients are non-zero)
    pub fn sparsity_pattern(&self) -> Result<&Array1<bool>> {
        self.sparsity_pattern_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "sparsity_pattern".to_string(),
            })
    }

    /// Get fitted x values
    pub fn fitted_x(&self) -> Result<&Array1<Float>> {
        self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_x".to_string(),
        })
    }

    /// Get fitted y values
    pub fn fitted_y(&self) -> Result<&Array1<Float>> {
        self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_y".to_string(),
        })
    }

    /// Count number of non-zero groups
    pub fn n_active_groups(&self) -> Result<usize> {
        Ok(self.active_groups()?.len())
    }

    /// Get sparsity ratio (fraction of zero coefficients)
    pub fn sparsity_ratio(&self) -> Result<Float> {
        let pattern = self.sparsity_pattern()?;
        let n_nonzero = pattern.iter().filter(|&&x| x).count();
        Ok(1.0 - (n_nonzero as Float) / (pattern.len() as Float))
    }
}

/// Fit structured sparse isotonic regression
fn fit_structured_sparse(
    x: &[Float],
    y: &[Float],
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    match &config.sparsity_type {
        StructuredSparsityType::GroupLasso { groups, lambda } => {
            fit_group_lasso(&mut fitted_y, y, groups, *lambda, config)
        }
        StructuredSparsityType::FusedLasso { lambda } => {
            fit_fused_lasso(&mut fitted_y, y, *lambda, config)
        }
        StructuredSparsityType::Hierarchical { group_hierarchy } => {
            fit_hierarchical_sparse(&mut fitted_y, y, group_hierarchy, config)
        }
        StructuredSparsityType::GroupElasticNet {
            groups,
            l1_ratio,
            lambda,
        } => fit_group_elastic_net(&mut fitted_y, y, groups, *l1_ratio, *lambda, config),
        StructuredSparsityType::TotalVariation { lambda } => {
            fit_total_variation_sparse(&mut fitted_y, y, *lambda, config)
        }
        StructuredSparsityType::GraphLasso {
            adjacency_matrix,
            lambda,
        } => fit_graph_lasso(&mut fitted_y, y, adjacency_matrix, *lambda, config),
    }
}

/// Fit Group Lasso regularized isotonic regression
fn fit_group_lasso(
    fitted_y: &mut [Float],
    y: &[Float],
    groups: &[Vec<usize>],
    lambda: Float,
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    let n = fitted_y.len();
    let mut active_groups = Vec::new();
    let mut sparsity_pattern = Array1::from(vec![false; n]);

    // Proximal gradient descent with group-wise soft thresholding
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.to_vec();

        // Gradient step for data fitting term
        for i in 0..n {
            let gradient = 2.0 * (fitted_y[i] - y[i]); // L2 loss gradient
            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Group-wise soft thresholding
        for (group_idx, group) in groups.iter().enumerate() {
            let group_norm: Float = group
                .iter()
                .map(|&i| if i < n { fitted_y[i].powi(2) } else { 0.0 })
                .sum::<Float>()
                .sqrt();

            let threshold = config.learning_rate * lambda;

            if group_norm > threshold {
                // Group is active
                let shrinkage_factor = 1.0 - threshold / group_norm;
                for &i in group {
                    if i < n {
                        fitted_y[i] *= shrinkage_factor;
                        sparsity_pattern[i] = true;
                    }
                }
                if !active_groups.contains(&group_idx) {
                    active_groups.push(group_idx);
                }
            } else {
                // Group is set to zero
                for &i in group {
                    if i < n {
                        fitted_y[i] = 0.0;
                    }
                }
            }
        }

        // Apply bounds
        apply_bounds(fitted_y, config);

        // Apply monotonicity constraint
        apply_monotonicity_constraint(fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok((fitted_y.to_vec(), active_groups, sparsity_pattern))
}

/// Fit Fused Lasso regularized isotonic regression
fn fit_fused_lasso(
    fitted_y: &mut [Float],
    y: &[Float],
    lambda: Float,
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    let n = fitted_y.len();

    // Proximal gradient descent for fused lasso
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.to_vec();

        // Gradient step
        for i in 0..n {
            let mut gradient = 2.0 * (fitted_y[i] - y[i]); // Data fitting term

            // Fused lasso penalty gradient
            if i > 0 {
                let diff = fitted_y[i] - fitted_y[i - 1];
                gradient += lambda * diff.signum();
            }
            if i < n - 1 {
                let diff = fitted_y[i + 1] - fitted_y[i];
                gradient -= lambda * diff.signum();
            }

            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Apply bounds
        apply_bounds(fitted_y, config);

        // Apply monotonicity constraint
        apply_monotonicity_constraint(fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    // Determine sparsity pattern (adjacent differences close to zero)
    let mut sparsity_pattern = Array1::from(vec![true; n]);
    let threshold = config.tolerance * 10.0;

    for i in 0..n - 1 {
        if (fitted_y[i + 1] - fitted_y[i]).abs() < threshold {
            sparsity_pattern[i] = false;
        }
    }

    let active_groups = vec![0]; // Single group for fused lasso

    Ok((fitted_y.to_vec(), active_groups, sparsity_pattern))
}

/// Fit hierarchical sparse isotonic regression
fn fit_hierarchical_sparse(
    fitted_y: &mut [Float],
    y: &[Float],
    group_hierarchy: &[(Vec<usize>, Float)],
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    let n = fitted_y.len();
    let mut active_groups = Vec::new();
    let mut sparsity_pattern = Array1::from(vec![false; n]);

    // Apply hierarchical group lasso with different penalties
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.to_vec();

        // Gradient step for data fitting term
        for i in 0..n {
            let gradient = 2.0 * (fitted_y[i] - y[i]);
            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Apply hierarchical group penalties
        for (level, (group, lambda)) in group_hierarchy.iter().enumerate() {
            let group_norm: Float = group
                .iter()
                .map(|&i| if i < n { fitted_y[i].powi(2) } else { 0.0 })
                .sum::<Float>()
                .sqrt();

            let threshold = config.learning_rate * lambda;

            if group_norm > threshold {
                let shrinkage_factor = 1.0 - threshold / group_norm;
                for &i in group {
                    if i < n {
                        fitted_y[i] *= shrinkage_factor;
                        sparsity_pattern[i] = true;
                    }
                }
                if !active_groups.contains(&level) {
                    active_groups.push(level);
                }
            } else {
                for &i in group {
                    if i < n {
                        fitted_y[i] = 0.0;
                    }
                }
            }
        }

        // Apply bounds
        apply_bounds(fitted_y, config);

        // Apply monotonicity constraint
        apply_monotonicity_constraint(fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    Ok((fitted_y.to_vec(), active_groups, sparsity_pattern))
}

/// Fit Group Elastic Net regularized isotonic regression
fn fit_group_elastic_net(
    fitted_y: &mut [Float],
    y: &[Float],
    groups: &[Vec<usize>],
    l1_ratio: Float,
    lambda: Float,
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    // Split regularization between L1 and group L2
    let l1_lambda = l1_ratio * lambda;
    let l2_lambda = (1.0 - l1_ratio) * lambda;

    // Apply group lasso with reduced penalty
    fit_group_lasso(fitted_y, y, groups, l2_lambda, config)
}

/// Fit Total Variation sparse isotonic regression
fn fit_total_variation_sparse(
    fitted_y: &mut [Float],
    y: &[Float],
    lambda: Float,
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    // Total variation is similar to fused lasso but with different penalty structure
    fit_fused_lasso(fitted_y, y, lambda, config)
}

/// Fit Graph Lasso regularized isotonic regression
fn fit_graph_lasso(
    fitted_y: &mut [Float],
    y: &[Float],
    adjacency_matrix: &Array2<Float>,
    lambda: Float,
    config: &StructuredSparseIsotonicRegression<Untrained>,
) -> Result<(Vec<Float>, Vec<usize>, Array1<bool>)> {
    let n = fitted_y.len();

    if adjacency_matrix.nrows() != n || adjacency_matrix.ncols() != n {
        return Err(SklearsError::InvalidInput(
            "Adjacency matrix dimensions must match data length".to_string(),
        ));
    }

    // Proximal gradient descent for graph lasso
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.to_vec();

        // Gradient step
        for i in 0..n {
            let mut gradient = 2.0 * (fitted_y[i] - y[i]); // Data fitting term

            // Graph penalty gradient
            for j in 0..n {
                if i != j && adjacency_matrix[[i, j]] != 0.0 {
                    let diff = fitted_y[i] - fitted_y[j];
                    gradient += lambda * adjacency_matrix[[i, j]] * diff.signum();
                }
            }

            fitted_y[i] -= config.learning_rate * gradient;
        }

        // Apply bounds
        apply_bounds(fitted_y, config);

        // Apply monotonicity constraint
        apply_monotonicity_constraint(fitted_y, &config.constraint)?;

        // Check convergence
        let change: Float = fitted_y
            .iter()
            .zip(old_y.iter())
            .map(|(new, old)| (new - old).powi(2))
            .sum::<Float>()
            .sqrt();

        if change < config.tolerance {
            break;
        }
    }

    // Determine sparsity pattern based on graph structure
    let mut sparsity_pattern = Array1::from(vec![true; n]);
    let threshold = config.tolerance * 10.0;

    for i in 0..n {
        if fitted_y[i].abs() < threshold {
            sparsity_pattern[i] = false;
        }
    }

    let active_groups = vec![0]; // Single group for graph lasso

    Ok((fitted_y.to_vec(), active_groups, sparsity_pattern))
}

/// Apply bounds to fitted values
fn apply_bounds(fitted_y: &mut [Float], config: &StructuredSparseIsotonicRegression<Untrained>) {
    if let Some(y_min) = config.y_min {
        for val in fitted_y.iter_mut() {
            *val = val.max(y_min);
        }
    }
    if let Some(y_max) = config.y_max {
        for val in fitted_y.iter_mut() {
            *val = val.min(y_max);
        }
    }
}

/// Apply monotonicity constraint
fn apply_monotonicity_constraint(
    fitted_y: &mut [Float],
    constraint: &MonotonicityConstraint,
) -> Result<()> {
    match constraint {
        MonotonicityConstraint::Global { increasing: true } => {
            let y_array = Array1::from(fitted_y.to_vec());
            let constrained = crate::algorithms::pool_adjacent_violators_increasing(&y_array);
            copy_array_to_slice(&constrained, fitted_y)?;
        }
        MonotonicityConstraint::Global { increasing: false } => {
            let y_array = Array1::from(fitted_y.to_vec());
            let constrained = crate::algorithms::pool_adjacent_violators_decreasing(&y_array);
            copy_array_to_slice(&constrained, fitted_y)?;
        }
        _ => {
            // For complex constraints, use simple increasing for now
            let y_array = Array1::from(fitted_y.to_vec());
            let constrained = crate::algorithms::pool_adjacent_violators_increasing(&y_array);
            copy_array_to_slice(&constrained, fitted_y)?;
        }
    }
    Ok(())
}

/// Helper function to safely copy Array1 to slice
fn copy_array_to_slice(array: &Array1<Float>, slice: &mut [Float]) -> Result<()> {
    if array.len() != slice.len() {
        return Err(SklearsError::InvalidInput(
            format!("Array length {} does not match slice length {}", array.len(), slice.len()),
        ));
    }

    // Try to use as_slice for efficiency if array is contiguous
    if let Some(arr_slice) = array.as_slice() {
        slice.copy_from_slice(arr_slice);
    } else {
        // Fall back to element-by-element copy if not contiguous
        for (i, &val) in array.iter().enumerate() {
            slice[i] = val;
        }
    }
    Ok(())
}

/// Convenience function for structured sparse isotonic regression
pub fn structured_sparse_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    sparsity_type: StructuredSparsityType,
    increasing: bool,
) -> Result<(Array1<Float>, Array1<Float>, Vec<usize>, Array1<bool>)> {
    let model = StructuredSparseIsotonicRegression::new()
        .sparsity_type(sparsity_type)
        .increasing(increasing);

    let fitted_model = model.fit(x, y)?;

    let fitted_x = fitted_model.fitted_x()?.clone();
    let fitted_y = fitted_model.fitted_y()?.clone();
    let active_groups = fitted_model.active_groups()?.to_vec();
    let sparsity_pattern = fitted_model.sparsity_pattern()?.clone();

    Ok((fitted_x, fitted_y, active_groups, sparsity_pattern))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_group_lasso_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let model = StructuredSparseIsotonicRegression::new()
            .group_lasso(groups, 0.1)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }

        let active_groups = fitted_model.active_groups().unwrap();
        assert!(!active_groups.is_empty());
    }

    #[test]
    fn test_fused_lasso() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 5.0]);

        let model = StructuredSparseIsotonicRegression::new()
            .fused_lasso(0.1)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }

        let sparsity_pattern = fitted_model.sparsity_pattern().unwrap();
        assert_eq!(sparsity_pattern.len(), y.len());
    }

    #[test]
    fn test_hierarchical_sparsity() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let hierarchy = vec![(vec![0, 1], 0.1), (vec![2, 3], 0.15), (vec![4, 5], 0.2)];

        let model = StructuredSparseIsotonicRegression::new()
            .hierarchical_sparsity(hierarchy)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_graph_lasso() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0]);

        // Create a simple adjacency matrix (chain graph)
        let mut adjacency = Array2::zeros((4, 4));
        adjacency[[0, 1]] = 1.0;
        adjacency[[1, 0]] = 1.0;
        adjacency[[1, 2]] = 1.0;
        adjacency[[2, 1]] = 1.0;
        adjacency[[2, 3]] = 1.0;
        adjacency[[3, 2]] = 1.0;

        let model = StructuredSparseIsotonicRegression::new()
            .graph_lasso(adjacency, 0.1)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_sparsity_metrics() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let model = StructuredSparseIsotonicRegression::new()
            .group_lasso(groups, 0.5) // High regularization to induce sparsity
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();

        let n_active_groups = fitted_model.n_active_groups().unwrap();
        let sparsity_ratio = fitted_model.sparsity_ratio().unwrap();

        assert!(n_active_groups <= 2); // At most 2 groups
        assert!(sparsity_ratio >= 0.0 && sparsity_ratio <= 1.0);
    }

    #[test]
    fn test_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0]);

        let sparsity_type = StructuredSparsityType::FusedLasso { lambda: 0.1 };

        let result = structured_sparse_isotonic_regression(&x, &y, sparsity_type, true);
        assert!(result.is_ok());

        let (fitted_x, fitted_y, active_groups, sparsity_pattern) = result.unwrap();
        assert_eq!(fitted_x.len(), 4);
        assert_eq!(fitted_y.len(), 4);
        assert_eq!(sparsity_pattern.len(), 4);

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }

    #[test]
    fn test_group_elastic_net() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let model = StructuredSparseIsotonicRegression::new()
            .group_elastic_net(groups, 0.5, 0.1)
            .increasing(true);

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check monotonicity
        for i in 0..fitted_y.len() - 1 {
            assert!(fitted_y[i] <= fitted_y[i + 1]);
        }
    }
}

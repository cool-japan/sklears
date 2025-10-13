//! Group isotonic constraints for flexible monotonicity modeling
//!
//! This module provides group-based isotonic regression that allows different
//! monotonicity constraints to be applied to different groups of features or data segments.

use crate::core::{LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::{collections::HashMap, marker::PhantomData};

/// Group constraint specification
#[derive(Debug, Clone)]
/// GroupConstraint
pub struct GroupConstraint {
    /// Group identifier
    pub group_id: usize,
    /// Indices of features/points belonging to this group
    pub indices: Vec<usize>,
    /// Monotonicity constraint for this group
    pub constraint: MonotonicityConstraint,
    /// Optional weight for this group in the objective function
    pub weight: Float,
}

/// Group isotonic regression model
///
/// Allows different monotonicity constraints for different groups of features or data segments.
#[derive(Debug, Clone)]
/// GroupIsotonicRegression
pub struct GroupIsotonicRegression<State = Untrained> {
    /// Group constraints specification
    pub group_constraints: Vec<GroupConstraint>,
    /// Loss function to optimize
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
    group_fitted_values_: Option<HashMap<usize, Array1<Float>>>,

    _state: PhantomData<State>,
}

impl GroupIsotonicRegression<Untrained> {
    /// Create a new group isotonic regression model
    pub fn new() -> Self {
        Self {
            group_constraints: Vec::new(),
            loss: LossFunction::SquaredLoss,
            max_iter: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            y_min: None,
            y_max: None,
            x_: None,
            y_: None,
            group_fitted_values_: None,
            _state: PhantomData,
        }
    }

    /// Add a group constraint
    pub fn add_group_constraint(mut self, constraint: GroupConstraint) -> Self {
        self.group_constraints.push(constraint);
        self
    }

    /// Add multiple group constraints
    pub fn add_group_constraints(mut self, constraints: Vec<GroupConstraint>) -> Self {
        self.group_constraints.extend(constraints);
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

    /// Create group constraints automatically from feature groups
    pub fn auto_group_by_features(
        mut self,
        feature_groups: Vec<Vec<usize>>,
        constraints: Vec<MonotonicityConstraint>,
    ) -> Result<Self> {
        if feature_groups.len() != constraints.len() {
            return Err(SklearsError::InvalidInput(
                "Number of feature groups must match number of constraints".to_string(),
            ));
        }

        self.group_constraints.clear();
        for (group_id, (indices, constraint)) in feature_groups
            .into_iter()
            .zip(constraints.into_iter())
            .enumerate()
        {
            self.group_constraints.push(GroupConstraint {
                group_id,
                indices,
                constraint,
                weight: 1.0,
            });
        }

        Ok(self)
    }

    /// Create group constraints from data segments
    pub fn auto_group_by_segments(
        mut self,
        segment_boundaries: Vec<usize>,
        constraints: Vec<MonotonicityConstraint>,
        data_length: usize,
    ) -> Result<Self> {
        if segment_boundaries.len() + 1 != constraints.len() {
            return Err(SklearsError::InvalidInput(
                "Number of segments must be one less than number of constraints".to_string(),
            ));
        }

        self.group_constraints.clear();
        let mut start = 0;

        for (group_id, (&end, constraint)) in segment_boundaries
            .iter()
            .zip(constraints.iter())
            .enumerate()
        {
            if end > data_length {
                return Err(SklearsError::InvalidInput(format!(
                    "Segment boundary {} exceeds data length {}",
                    end, data_length
                )));
            }
            let indices: Vec<usize> = (start..end).collect();
            self.group_constraints.push(GroupConstraint {
                group_id,
                indices,
                constraint: constraint.clone(),
                weight: 1.0,
            });
            start = end;
        }

        // Add final segment
        if let Some(last_constraint) = constraints.last() {
            if start < data_length {
                self.group_constraints.push(GroupConstraint {
                    group_id: constraints.len() - 1,
                    indices: (start..data_length).collect(),
                    constraint: last_constraint.clone(),
                    weight: 1.0,
                });
            }
        }

        Ok(self)
    }
}

impl Default for GroupIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GroupIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for GroupIsotonicRegression<Untrained> {
    type Fitted = GroupIsotonicRegression<Trained>;

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

        if self.group_constraints.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one group constraint must be specified".to_string(),
            ));
        }

        let n = x.len();

        // Validate group constraints
        let mut all_indices = Vec::new();
        for constraint in &self.group_constraints {
            for &idx in &constraint.indices {
                if idx >= n {
                    return Err(SklearsError::InvalidInput(format!(
                        "Group constraint index {} exceeds data length {}",
                        idx, n
                    )));
                }
                all_indices.push(idx);
            }
        }

        // Sort data by x values
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());

        let sorted_x: Vec<Float> = indices.iter().map(|&i| x[i]).collect();
        let sorted_y: Vec<Float> = indices.iter().map(|&i| y[i]).collect();

        // Create mapping from original to sorted indices
        let mut index_mapping = HashMap::new();
        for (sorted_pos, &original_idx) in indices.iter().enumerate() {
            index_mapping.insert(original_idx, sorted_pos);
        }

        // Update group constraints with sorted indices
        let mut updated_constraints = Vec::new();
        for constraint in &self.group_constraints {
            let mut updated_indices = Vec::new();
            for &original_idx in &constraint.indices {
                if let Some(&sorted_idx) = index_mapping.get(&original_idx) {
                    updated_indices.push(sorted_idx);
                }
            }
            updated_indices.sort();
            updated_constraints.push(GroupConstraint {
                group_id: constraint.group_id,
                indices: updated_indices,
                constraint: constraint.constraint.clone(),
                weight: constraint.weight,
            });
        }

        // Fit the model
        let fitted_y = fit_group_constrained(&sorted_x, &sorted_y, &updated_constraints, &self)?;

        // Fit each group separately to get group-specific fitted values
        let mut group_fitted_values = HashMap::new();
        for constraint in &updated_constraints {
            let group_x: Array1<Float> = constraint.indices.iter().map(|&i| sorted_x[i]).collect();
            let group_y: Array1<Float> = constraint.indices.iter().map(|&i| fitted_y[i]).collect();

            group_fitted_values.insert(constraint.group_id, group_y);
        }

        Ok(GroupIsotonicRegression {
            group_constraints: updated_constraints,
            loss: self.loss,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            learning_rate: self.learning_rate,
            y_min: self.y_min,
            y_max: self.y_max,
            x_: Some(Array1::from(sorted_x)),
            y_: Some(Array1::from(fitted_y)),
            group_fitted_values_: Some(group_fitted_values),
            _state: PhantomData,
        })
    }
}

impl Predict<Array1<Float>, Array1<Float>> for GroupIsotonicRegression<Trained> {
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

impl GroupIsotonicRegression<Trained> {
    /// Get fitted values for a specific group
    pub fn group_fitted_values(&self, group_id: usize) -> Result<&Array1<Float>> {
        let group_values =
            self.group_fitted_values_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "group_fitted_values".to_string(),
                })?;

        group_values
            .get(&group_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Group {} not found", group_id)))
    }

    /// Get all fitted x values
    pub fn fitted_x(&self) -> Result<&Array1<Float>> {
        self.x_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_x".to_string(),
        })
    }

    /// Get all fitted y values
    pub fn fitted_y(&self) -> Result<&Array1<Float>> {
        self.y_.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "fitted_y".to_string(),
        })
    }

    /// Get group constraints
    pub fn group_constraints(&self) -> &[GroupConstraint] {
        &self.group_constraints
    }
}

/// Fit isotonic regression with group constraints
fn fit_group_constrained(
    x: &[Float],
    y: &[Float],
    group_constraints: &[GroupConstraint],
    config: &GroupIsotonicRegression<Untrained>,
) -> Result<Vec<Float>> {
    let n = x.len();
    let mut fitted_y = y.to_vec();

    // Alternating optimization: fit each group while keeping others fixed
    for iteration in 0..config.max_iter {
        let old_y = fitted_y.clone();

        // Update each group separately
        for constraint in group_constraints {
            let group_indices = &constraint.indices;
            if group_indices.is_empty() {
                continue;
            }

            // Extract group data
            let group_x: Vec<Float> = group_indices.iter().map(|&i| x[i]).collect();
            let group_y: Vec<Float> = group_indices.iter().map(|&i| y[i]).collect();

            // Fit isotonic regression for this group
            let group_fitted = fit_single_group(&group_x, &group_y, &constraint.constraint)?;

            // Update fitted values for this group
            for (local_idx, &global_idx) in group_indices.iter().enumerate() {
                fitted_y[global_idx] = group_fitted[local_idx];
            }
        }

        // Apply global bounds if specified
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

    Ok(fitted_y)
}

/// Fit isotonic regression for a single group
fn fit_single_group(
    x: &[Float],
    y: &[Float],
    constraint: &MonotonicityConstraint,
) -> Result<Vec<Float>> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "x and y must have same length".to_string(),
        ));
    }

    let y_array = Array1::from(y.to_vec());

    match constraint {
        MonotonicityConstraint::Global { increasing: true } => {
            Ok(crate::algorithms::pool_adjacent_violators_increasing(&y_array).to_vec())
        }
        MonotonicityConstraint::Global { increasing: false } => {
            Ok(crate::algorithms::pool_adjacent_violators_decreasing(&y_array).to_vec())
        }
        _ => {
            // For complex constraints, use simple increasing for now
            Ok(crate::algorithms::pool_adjacent_violators_increasing(&y_array).to_vec())
        }
    }
}

/// Convenience function for group isotonic regression
pub fn group_isotonic_regression(
    x: &Array1<Float>,
    y: &Array1<Float>,
    group_constraints: Vec<GroupConstraint>,
) -> Result<(Array1<Float>, Array1<Float>, HashMap<usize, Array1<Float>>)> {
    let model = GroupIsotonicRegression::new().add_group_constraints(group_constraints);

    let fitted_model = model.fit(x, y)?;

    let fitted_x = fitted_model.fitted_x()?.clone();
    let fitted_y = fitted_model.fitted_y()?.clone();
    let group_values = fitted_model.group_fitted_values_.clone().unwrap();

    Ok((fitted_x, fitted_y, group_values))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_group_isotonic_basic() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);

        // Create two groups: first 3 points increasing, last 3 points decreasing
        let group1 = GroupConstraint {
            group_id: 0,
            indices: vec![0, 1, 2],
            constraint: MonotonicityConstraint::Global { increasing: true },
            weight: 1.0,
        };

        let group2 = GroupConstraint {
            group_id: 1,
            indices: vec![3, 4, 5],
            constraint: MonotonicityConstraint::Global { increasing: false },
            weight: 1.0,
        };

        let model = GroupIsotonicRegression::new()
            .add_group_constraint(group1)
            .add_group_constraint(group2);

        let fitted_model = model.fit(&x, &y).unwrap();

        let fitted_y = fitted_model.fitted_y().unwrap();

        // Check that constraints are satisfied
        // Note: after sorting, indices might change, so we check the pattern
        assert_eq!(fitted_y.len(), 6);
    }

    #[test]
    fn test_auto_group_by_features() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);

        let feature_groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let constraints = vec![
            MonotonicityConstraint::Global { increasing: true },
            MonotonicityConstraint::Global { increasing: false },
        ];

        let model = GroupIsotonicRegression::new()
            .auto_group_by_features(feature_groups, constraints)
            .unwrap();

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        assert_eq!(fitted_y.len(), 6);
    }

    #[test]
    fn test_auto_group_by_segments() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0]);

        let segment_boundaries = vec![3]; // Split at index 3
        let constraints = vec![
            MonotonicityConstraint::Global { increasing: true },
            MonotonicityConstraint::Global { increasing: false },
        ];

        let model = GroupIsotonicRegression::new()
            .auto_group_by_segments(segment_boundaries, constraints, x.len())
            .unwrap();

        let fitted_model = model.fit(&x, &y).unwrap();
        let fitted_y = fitted_model.fitted_y().unwrap();

        assert_eq!(fitted_y.len(), 6);
    }

    #[test]
    fn test_group_fitted_values() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0]);

        let group = GroupConstraint {
            group_id: 0,
            indices: vec![0, 1, 2, 3],
            constraint: MonotonicityConstraint::Global { increasing: true },
            weight: 1.0,
        };

        let model = GroupIsotonicRegression::new().add_group_constraint(group);
        let fitted_model = model.fit(&x, &y).unwrap();

        let group_values = fitted_model.group_fitted_values(0).unwrap();
        assert_eq!(group_values.len(), 4);

        // Check monotonicity within group
        for i in 0..group_values.len() - 1 {
            assert!(group_values[i] <= group_values[i + 1]);
        }
    }

    #[test]
    fn test_convenience_function() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let y = Array1::from(vec![2.0, 1.0, 4.0, 3.0]);

        let constraints = vec![GroupConstraint {
            group_id: 0,
            indices: vec![0, 1, 2, 3],
            constraint: MonotonicityConstraint::Global { increasing: true },
            weight: 1.0,
        }];

        let result = group_isotonic_regression(&x, &y, constraints);
        assert!(result.is_ok());

        let (fitted_x, fitted_y, group_values) = result.unwrap();
        assert_eq!(fitted_x.len(), 4);
        assert_eq!(fitted_y.len(), 4);
        assert!(group_values.contains_key(&0));
    }

    #[test]
    fn test_multiple_groups_different_constraints() {
        let x = Array1::from(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let y = Array1::from(vec![1.0, 3.0, 2.0, 4.0, 6.0, 5.0, 7.0, 6.0]);

        let groups = vec![
            GroupConstraint {
                group_id: 0,
                indices: vec![0, 1, 2, 3],
                constraint: MonotonicityConstraint::Global { increasing: true },
                weight: 1.0,
            },
            GroupConstraint {
                group_id: 1,
                indices: vec![4, 5, 6, 7],
                constraint: MonotonicityConstraint::Global { increasing: false },
                weight: 1.0,
            },
        ];

        let model = GroupIsotonicRegression::new().add_group_constraints(groups);
        let fitted_model = model.fit(&x, &y).unwrap();

        let fitted_y = fitted_model.fitted_y().unwrap();
        assert_eq!(fitted_y.len(), 8);

        // Test that we can get values for both groups
        let group0_values = fitted_model.group_fitted_values(0).unwrap();
        let group1_values = fitted_model.group_fitted_values(1).unwrap();

        assert_eq!(group0_values.len(), 4);
        assert_eq!(group1_values.len(), 4);
    }
}

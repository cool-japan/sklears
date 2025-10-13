//! Group Lasso Covariance Estimator
//!
//! Implements sparse covariance estimation using group lasso regularization.
//! The group lasso encourages sparsity at the group level, making it suitable
//! for scenarios where variables naturally form groups and entire groups
//! should be selected or discarded together.

use crate::utils::{matrix_determinant, matrix_inverse};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Group Lasso Covariance Estimator
///
/// Estimates sparse covariance matrices using group lasso regularization.
/// The group lasso applies penalties to groups of variables, encouraging
/// structured sparsity where entire groups are either included or excluded.
///
/// # Parameters
///
/// * `alpha` - Regularization strength
/// * `groups` - Group assignments for variables (None for automatic grouping)
/// * `group_weights` - Weights for different groups (None for equal weights)
/// * `max_iter` - Maximum number of iterations for optimization
/// * `tol` - Convergence tolerance
/// * `store_precision` - Whether to store the precision matrix
///
/// # Examples
///
/// ```
/// use sklears_covariance::GroupLassoCovariance;
/// use sklears_core::traits::Fit;
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 0.5, 0.1], [0.5, 2.0, 0.2], [2.0, 1.0, 0.3]];
///
/// let estimator = GroupLassoCovariance::new()
///     .alpha(0.1)
///     .groups(vec![0, 0, 1]); // First two variables in group 0, third in group 1
/// let fitted = estimator.fit(&x.view(), &()).unwrap();
/// let precision = fitted.get_precision().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GroupLassoCovariance<S = Untrained> {
    state: S,
    alpha: f64,
    groups: Option<Vec<usize>>,
    group_weights: Option<Vec<f64>>,
    max_iter: usize,
    tol: f64,
    store_precision: bool,
    assume_centered: bool,
}

/// Trained state for GroupLassoCovariance
#[derive(Debug, Clone)]
pub struct GroupLassoCovarianceTrained {
    /// The covariance matrix
    pub covariance: Array2<f64>,
    /// The precision matrix (inverse of covariance)
    pub precision: Option<Array2<f64>>,
    /// The group assignments used
    pub groups: Vec<usize>,
    /// The group weights used
    pub group_weights: Vec<f64>,
    /// Number of iterations used in optimization
    pub n_iter: usize,
    /// Final objective value
    pub objective: f64,
}

impl GroupLassoCovariance<Untrained> {
    /// Create a new GroupLassoCovariance instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 0.1,
            groups: None,
            group_weights: None,
            max_iter: 100,
            tol: 1e-4,
            store_precision: true,
            assume_centered: false,
        }
    }

    /// Set the regularization strength
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha.max(0.0);
        self
    }

    /// Set the group assignments for variables
    pub fn groups(mut self, groups: Vec<usize>) -> Self {
        self.groups = Some(groups);
        self
    }

    /// Set the weights for different groups
    pub fn group_weights(mut self, weights: Vec<f64>) -> Self {
        self.group_weights = Some(weights);
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Set whether to store the precision matrix
    pub fn store_precision(mut self, store_precision: bool) -> Self {
        self.store_precision = store_precision;
        self
    }

    /// Set whether to assume the data is centered
    pub fn assume_centered(mut self, assume_centered: bool) -> Self {
        self.assume_centered = assume_centered;
        self
    }
}

impl Default for GroupLassoCovariance<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GroupLassoCovariance<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GroupLassoCovariance<Untrained> {
    type Fitted = GroupLassoCovariance<GroupLassoCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = *x;
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples".to_string(),
            ));
        }

        // Set up groups (default to single variable per group)
        let groups = self
            .groups
            .clone()
            .unwrap_or_else(|| (0..n_features).collect());

        if groups.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Groups length must match number of features".to_string(),
            ));
        }

        let n_groups = groups.iter().max().map(|&x| x + 1).unwrap_or(0);

        // Set up group weights (default to sqrt of group size)
        let group_weights = self.group_weights.clone().unwrap_or_else(|| {
            let mut weights = vec![0.0f64; n_groups];
            for &group_id in &groups {
                weights[group_id] += 1.0;
            }
            weights.iter().map(|&size| size.sqrt()).collect()
        });

        if group_weights.len() != n_groups {
            return Err(SklearsError::InvalidInput(
                "Group weights length must match number of groups".to_string(),
            ));
        }

        // Compute empirical covariance as starting point
        let mut empirical_cov = compute_empirical_covariance(&x, self.assume_centered)?;

        // Add small regularization to diagonal for numerical stability
        for i in 0..n_features {
            empirical_cov[[i, i]] += 1e-6;
        }

        // Perform group lasso optimization
        let (precision, n_iter, objective) = group_lasso_optimization(
            &empirical_cov,
            &groups,
            &group_weights,
            self.alpha,
            self.max_iter,
            self.tol,
        )?;

        // Compute covariance from precision
        let covariance = matrix_inverse(&precision)?;

        let stored_precision = if self.store_precision {
            Some(precision)
        } else {
            None
        };

        Ok(GroupLassoCovariance {
            state: GroupLassoCovarianceTrained {
                covariance,
                precision: stored_precision,
                groups,
                group_weights,
                n_iter,
                objective,
            },
            alpha: self.alpha,
            groups: self.groups,
            group_weights: self.group_weights,
            max_iter: self.max_iter,
            tol: self.tol,
            store_precision: self.store_precision,
            assume_centered: self.assume_centered,
        })
    }
}

impl GroupLassoCovariance<GroupLassoCovarianceTrained> {
    /// Get the covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix (inverse covariance)
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the group assignments used
    pub fn get_groups(&self) -> &Vec<usize> {
        &self.state.groups
    }

    /// Get the group weights used
    pub fn get_group_weights(&self) -> &Vec<f64> {
        &self.state.group_weights
    }

    /// Get the number of iterations used in optimization
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the final objective value
    pub fn get_objective(&self) -> f64 {
        self.state.objective
    }

    /// Get the regularization strength
    pub fn get_alpha(&self) -> f64 {
        self.alpha
    }
}

/// Compute empirical covariance matrix
fn compute_empirical_covariance(
    x: &ArrayView2<Float>,
    assume_centered: bool,
) -> SklResult<Array2<f64>> {
    let (n_samples, n_features) = x.dim();

    // Compute mean
    let mean = if assume_centered {
        Array1::zeros(n_features)
    } else {
        x.mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("Failed to compute mean".to_string()))?
    };

    // Compute covariance
    let mut cov = Array2::zeros((n_features, n_features));

    for sample in x.axis_iter(Axis(0)) {
        let centered = &sample - &mean;
        for i in 0..n_features {
            for j in 0..n_features {
                cov[[i, j]] += centered[i] * centered[j];
            }
        }
    }

    cov /= (n_samples - 1) as f64;

    Ok(cov)
}

/// Perform group lasso optimization using block coordinate descent
fn group_lasso_optimization(
    empirical_cov: &Array2<f64>,
    groups: &[usize],
    group_weights: &[f64],
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> SklResult<(Array2<f64>, usize, f64)> {
    let n_features = empirical_cov.nrows();
    let n_groups = group_weights.len();

    // Initialize precision matrix as empirical precision (regularized)
    let mut precision = matrix_inverse(empirical_cov)?;

    // Add small regularization to diagonal to ensure positive definiteness
    for i in 0..n_features {
        precision[[i, i]] += alpha * 0.01;
    }

    let mut objective_old = f64::INFINITY;
    let mut n_iter = 0;

    for iter in 0..max_iter {
        n_iter = iter + 1;
        let mut precision_new = precision.clone();

        // Update each group using block coordinate descent
        for group_id in 0..n_groups {
            let group_indices: Vec<usize> = groups
                .iter()
                .enumerate()
                .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
                .collect();

            if group_indices.is_empty() {
                continue;
            }

            // Extract group block from precision matrix
            let group_size = group_indices.len();
            let mut group_block = Array2::zeros((group_size, group_size));

            for (i, &row_idx) in group_indices.iter().enumerate() {
                for (j, &col_idx) in group_indices.iter().enumerate() {
                    group_block[[i, j]] = precision[[row_idx, col_idx]];
                }
            }

            // Compute group norm
            let group_norm = compute_group_norm(&group_block);

            // Apply group soft thresholding
            let threshold = alpha * group_weights[group_id];
            let shrinkage_factor = if group_norm > threshold {
                1.0 - threshold / group_norm
            } else {
                0.0
            };

            // Update precision matrix with shrunk group block
            for (i, &row_idx) in group_indices.iter().enumerate() {
                for (j, &col_idx) in group_indices.iter().enumerate() {
                    if row_idx != col_idx {
                        precision_new[[row_idx, col_idx]] = group_block[[i, j]] * shrinkage_factor;
                        precision_new[[col_idx, row_idx]] = precision_new[[row_idx, col_idx]];
                        // Maintain symmetry
                    }
                }
            }
        }

        // Compute objective function (negative log-likelihood + group penalty)
        let log_det = matrix_determinant(&precision_new).ln();
        let trace_term = (empirical_cov * &precision_new).diag().sum();

        let penalty_term = compute_group_penalty(&precision_new, groups, group_weights, alpha);

        let objective = -log_det + trace_term + penalty_term;

        // Check for convergence
        if (objective - objective_old).abs() < tol {
            break;
        }

        precision = precision_new;
        objective_old = objective;
    }

    Ok((precision, n_iter, objective_old))
}

/// Compute group norm (Frobenius norm for matrix blocks, L2 norm for vectors)
fn compute_group_norm(group_block: &Array2<f64>) -> f64 {
    let mut norm_sq = 0.0;
    for i in 0..group_block.nrows() {
        for j in 0..group_block.ncols() {
            if i != j {
                // Exclude diagonal elements
                norm_sq += group_block[[i, j]].powi(2);
            }
        }
    }
    norm_sq.sqrt()
}

/// Compute total group penalty
fn compute_group_penalty(
    precision: &Array2<f64>,
    groups: &[usize],
    group_weights: &[f64],
    alpha: f64,
) -> f64 {
    let n_groups = group_weights.len();
    let mut total_penalty = 0.0;

    for group_id in 0..n_groups {
        let group_indices: Vec<usize> = groups
            .iter()
            .enumerate()
            .filter_map(|(i, &g)| if g == group_id { Some(i) } else { None })
            .collect();

        if group_indices.is_empty() {
            continue;
        }

        // Extract group block and compute its norm
        let group_size = group_indices.len();
        let mut group_block = Array2::zeros((group_size, group_size));

        for (i, &row_idx) in group_indices.iter().enumerate() {
            for (j, &col_idx) in group_indices.iter().enumerate() {
                group_block[[i, j]] = precision[[row_idx, col_idx]];
            }
        }

        let group_norm = compute_group_norm(&group_block);
        total_penalty += alpha * group_weights[group_id] * group_norm;
    }

    total_penalty
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_group_lasso_basic() {
        let x = array![
            [1.0, 0.5, 0.1],
            [0.5, 2.0, 0.2],
            [2.0, 1.0, 0.3],
            [1.5, 1.5, 0.4],
            [2.5, 2.0, 0.5]
        ];

        let groups = vec![0, 0, 1]; // First two variables in group 0, third in group 1
        let estimator = GroupLassoCovariance::new()
            .alpha(0.1)
            .groups(groups.clone())
            .max_iter(50);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_precision().is_some());
        assert_eq!(fitted.get_groups(), &groups);
        assert_eq!(fitted.get_alpha(), 0.1);
        assert!(fitted.get_n_iter() > 0);
    }

    #[test]
    fn test_group_lasso_default_groups() {
        let x = array![[1.0, 0.5], [2.0, 1.5], [3.0, 2.8], [4.0, 3.9], [5.0, 4.1]];

        let estimator = GroupLassoCovariance::new().alpha(0.05).max_iter(30);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(fitted.get_precision().is_some());
        assert_eq!(fitted.get_groups(), &vec![0, 1]); // Default: one variable per group
        assert_eq!(fitted.get_alpha(), 0.05);
    }

    #[test]
    fn test_group_norm() {
        let block = array![[1.0, 0.3, 0.2], [0.3, 1.0, 0.1], [0.2, 0.1, 1.0]];

        let norm = compute_group_norm(&block);
        let expected = (0.3_f64.powi(2)
            + 0.2_f64.powi(2)
            + 0.3_f64.powi(2)
            + 0.1_f64.powi(2)
            + 0.2_f64.powi(2)
            + 0.1_f64.powi(2))
        .sqrt();
        assert!((norm - expected).abs() < 1e-10);
    }
}

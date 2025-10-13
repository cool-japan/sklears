//! Group Lasso Regularization for Multi-Task Learning
//!
//! Group Lasso applies L1 penalty to groups of features, encouraging sparse selection
//! of feature groups across all tasks. This is particularly useful when features
//! can be naturally grouped and we want to select entire groups rather than
//! individual features.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

use super::simd_ops;

/// Group Lasso Regularization for Multi-Task Learning
///
/// Group Lasso applies L1 penalty to groups of features, encouraging sparse selection
/// of feature groups across all tasks. This is particularly useful when features
/// can be naturally grouped and we want to select entire groups rather than
/// individual features.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::regularization::{GroupLasso, RegularizationStrategy};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0], [3.0, 1.0, 2.0, 3.0]];
/// let mut y_tasks = HashMap::new();
/// y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5]]);
/// y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8]]);
///
/// // Define feature groups: [0,1] and [2,3]
/// let feature_groups = vec![vec![0, 1], vec![2, 3]];
///
/// let group_lasso = GroupLasso::new()
///     .alpha(0.1)
///     .feature_groups(feature_groups)
///     .max_iter(1000)
///     .tolerance(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct GroupLasso<S = Untrained> {
    pub(crate) state: S,
    /// Regularization strength
    pub(crate) alpha: Float,
    /// Feature groups for group lasso
    pub(crate) feature_groups: Vec<Vec<usize>>,
    /// Maximum number of iterations
    pub(crate) max_iter: usize,
    /// Convergence tolerance
    pub(crate) tolerance: Float,
    /// Learning rate for gradient descent
    pub(crate) learning_rate: Float,
    /// Task names and their output dimensions
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Include intercept term
    pub(crate) fit_intercept: bool,
}

/// Trained state for GroupLasso
#[derive(Debug, Clone)]
pub struct GroupLassoTrained {
    /// Coefficients for each task [features x tasks]
    pub(crate) coefficients: HashMap<String, Array2<Float>>,
    /// Intercepts for each task
    pub(crate) intercepts: HashMap<String, Array1<Float>>,
    /// Number of input features
    pub(crate) n_features: usize,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Feature groups
    pub(crate) feature_groups: Vec<Vec<usize>>,
    /// Training iterations performed
    pub(crate) n_iter: usize,
    /// Regularization strength used
    pub(crate) alpha: Float,
}

impl GroupLasso<Untrained> {
    /// Create a new GroupLasso instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 1.0,
            feature_groups: Vec::new(),
            max_iter: 1000,
            tolerance: 1e-4,
            learning_rate: 0.01,
            task_outputs: HashMap::new(),
            fit_intercept: true,
        }
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set feature groups for group lasso
    pub fn feature_groups(mut self, groups: Vec<Vec<usize>>) -> Self {
        self.feature_groups = groups;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, lr: Float) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Configure task outputs
    pub fn task_outputs(mut self, tasks: &[(&str, usize)]) -> Self {
        for (task_name, output_size) in tasks {
            self.task_outputs
                .insert(task_name.to_string(), *output_size);
        }
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for GroupLasso<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GroupLasso<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>> for GroupLasso<Untrained> {
    type Fitted = GroupLassoTrained;

    fn fit(
        self,
        x: &ArrayView2<Float>,
        y: &HashMap<String, Array2<Float>>,
    ) -> SklResult<Self::Fitted> {
        if x.nrows() == 0 || x.ncols() == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if y.is_empty() {
            return Err(SklearsError::InvalidInput("No tasks provided".to_string()));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validate feature groups
        if !self.feature_groups.is_empty() {
            for group in &self.feature_groups {
                for &feature_idx in group {
                    if feature_idx >= n_features {
                        return Err(SklearsError::InvalidInput(format!(
                            "Feature index {} out of range for {} features",
                            feature_idx, n_features
                        )));
                    }
                }
            }
        }

        // Initialize coefficients and intercepts for each task
        let mut coefficients = HashMap::new();
        let mut intercepts = HashMap::new();

        for (task_name, task_targets) in y {
            if task_targets.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{}", n_samples),
                    actual: format!("{}", task_targets.nrows()),
                });
            }

            let n_outputs = task_targets.ncols();
            coefficients.insert(
                task_name.clone(),
                Array2::<Float>::zeros((n_features, n_outputs)),
            );
            intercepts.insert(task_name.clone(), Array1::<Float>::zeros(n_outputs));
        }

        // Proximal gradient descent with group lasso penalty
        for iteration in 0..self.max_iter {
            let mut max_change: Float = 0.0;

            for (task_name, task_targets) in y {
                let task_coefs = coefficients.get_mut(task_name).unwrap();
                let task_intercepts = intercepts.get_mut(task_name).unwrap();

                // Compute predictions
                let mut predictions = x.dot(task_coefs);
                let intercepts_ref = task_intercepts.view();
                for mut row in predictions.rows_mut() {
                    row += &intercepts_ref;
                }

                // Compute residuals
                let residuals = &predictions - task_targets;

                // Compute gradients
                let grad_coefs = x.t().dot(&residuals) / (n_samples as Float);
                let grad_intercepts = residuals.mean_axis(Axis(0)).unwrap();

                // Standard gradient update
                let old_coefs = task_coefs.clone();
                *task_coefs = &*task_coefs - &(self.learning_rate * &grad_coefs);
                *task_intercepts = &*task_intercepts - &(self.learning_rate * &grad_intercepts);

                // Apply group lasso proximal operator
                if !self.feature_groups.is_empty() {
                    self.apply_group_lasso_proximal(task_coefs);
                }

                // Check convergence using SIMD acceleration
                let old_coefs_flat: Vec<f64> = old_coefs.iter().cloned().collect();
                let new_coefs_flat: Vec<f64> = task_coefs.iter().cloned().collect();
                let coef_change = simd_ops::simd_max_change(&old_coefs_flat, &new_coefs_flat);
                max_change = max_change.max(coef_change);
            }

            if max_change < self.tolerance {
                return Ok(GroupLassoTrained {
                    coefficients,
                    intercepts,
                    n_features,
                    task_outputs: self.task_outputs,
                    feature_groups: self.feature_groups,
                    n_iter: iteration + 1,
                    alpha: self.alpha,
                });
            }
        }

        Ok(GroupLassoTrained {
            coefficients,
            intercepts,
            n_features,
            task_outputs: self.task_outputs,
            feature_groups: self.feature_groups,
            n_iter: self.max_iter,
            alpha: self.alpha,
        })
    }
}

impl GroupLasso<Untrained> {
    /// Apply group lasso proximal operator
    fn apply_group_lasso_proximal(&self, coefficients: &mut Array2<Float>) {
        for group in &self.feature_groups {
            for &feature_idx in group {
                if feature_idx < coefficients.nrows() {
                    let mut feature_coefs = coefficients.row_mut(feature_idx);
                    let coef_slice = feature_coefs.as_slice().unwrap();
                    let group_norm = simd_ops::simd_l2_norm(coef_slice);

                    if group_norm > 0.0 {
                        let shrinkage =
                            (1.0 - self.alpha * self.learning_rate / group_norm).max(0.0);
                        feature_coefs *= shrinkage;
                    }
                }
            }
        }
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>> for GroupLassoTrained {
    fn predict(&self, x: &ArrayView2<Float>) -> SklResult<HashMap<String, Array2<Float>>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", self.n_features),
                actual: format!("{}", x.ncols()),
            });
        }

        let mut predictions = HashMap::new();

        for (task_name, task_coefs) in &self.coefficients {
            let task_intercepts = &self.intercepts[task_name];
            let task_pred = x.dot(task_coefs) + task_intercepts;
            predictions.insert(task_name.clone(), task_pred);
        }

        Ok(predictions)
    }
}

impl GroupLassoTrained {
    /// Calculate group sparsity (percentage of zero groups)
    pub fn group_sparsity(&self) -> Float {
        if self.feature_groups.is_empty() {
            return 0.0;
        }

        let mut zero_groups = 0;
        let total_groups = self.feature_groups.len();

        // For each group, check if all coefficients are effectively zero across all tasks
        for group in &self.feature_groups {
            let mut group_is_zero = true;
            for task_coefs in self.coefficients.values() {
                for &feature_idx in group {
                    if feature_idx < task_coefs.nrows() {
                        let coef_row = task_coefs.row(feature_idx);
                        for &coef in coef_row {
                            if coef.abs() > 1e-8 {
                                group_is_zero = false;
                                break;
                            }
                        }
                        if !group_is_zero {
                            break;
                        }
                    }
                }
                if !group_is_zero {
                    break;
                }
            }
            if group_is_zero {
                zero_groups += 1;
            }
        }

        zero_groups as Float / total_groups as Float
    }

    /// Get coefficients for a specific task
    pub fn task_coefficients(&self, task_name: &str) -> Option<&Array2<Float>> {
        self.coefficients.get(task_name)
    }

    /// Get intercepts for a specific task
    pub fn task_intercepts(&self, task_name: &str) -> Option<&Array1<Float>> {
        self.intercepts.get(task_name)
    }

    /// Get the number of training iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

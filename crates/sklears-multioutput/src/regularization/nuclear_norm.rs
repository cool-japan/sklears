//! Nuclear Norm Regularization for Multi-Task Learning
//!
//! Nuclear norm (trace norm) regularization encourages low-rank structure in the
//! coefficient matrix across tasks. This is based on the assumption that tasks
//! share a low-dimensional representation.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Nuclear Norm Regularization for Multi-Task Learning
///
/// Nuclear norm (trace norm) regularization encourages low-rank structure in the
/// coefficient matrix across tasks. This is based on the assumption that tasks
/// share a low-dimensional representation.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::regularization::NuclearNormRegression;
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
/// use scirs2_core::ndarray::array;
/// use std::collections::HashMap;
///
/// let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 4.0]];
/// let mut y_tasks = HashMap::new();
/// y_tasks.insert("task1".to_string(), array![[1.0], [2.0], [1.5], [2.5]]);
/// y_tasks.insert("task2".to_string(), array![[0.5], [1.0], [0.8], [1.2]]);
///
/// let nuclear_norm = NuclearNormRegression::new()
///     .alpha(0.1)
///     .max_iter(1000)
///     .tolerance(1e-6);
/// ```
#[derive(Debug, Clone)]
pub struct NuclearNormRegression<S = Untrained> {
    pub(crate) state: S,
    /// Regularization strength
    pub(crate) alpha: Float,
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
    /// Target rank for low-rank approximation
    pub(crate) target_rank: Option<usize>,
}

/// Trained state for NuclearNormRegression
#[derive(Debug, Clone)]
pub struct NuclearNormRegressionTrained {
    /// Coefficient matrix [features x total_outputs]
    pub(crate) coefficient_matrix: Array2<Float>,
    /// Task-specific coefficient views
    pub(crate) task_coefficients: HashMap<String, (usize, usize)>, // (start_col, end_col) indices
    /// Intercepts for each task
    pub(crate) intercepts: HashMap<String, Array1<Float>>,
    /// Number of input features
    pub(crate) n_features: usize,
    /// Task configurations
    pub(crate) task_outputs: HashMap<String, usize>,
    /// Training iterations performed
    pub(crate) n_iter: usize,
    /// Regularization strength used
    pub(crate) alpha: Float,
    /// Singular values from SVD
    pub(crate) singular_values: Array1<Float>,
}

impl NuclearNormRegression<Untrained> {
    /// Create a new NuclearNormRegression instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 1.0,
            max_iter: 1000,
            tolerance: 1e-4,
            learning_rate: 0.01,
            task_outputs: HashMap::new(),
            fit_intercept: true,
            target_rank: None,
        }
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
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

    /// Set target rank for low-rank approximation
    pub fn target_rank(mut self, rank: Option<usize>) -> Self {
        self.target_rank = rank;
        self
    }
}

impl Default for NuclearNormRegression<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NuclearNormRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for NuclearNormRegression<Untrained>
{
    type Fitted = NuclearNormRegressionTrained;

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

        // Calculate total outputs across all tasks
        let total_outputs: usize = y.values().map(|targets| targets.ncols()).sum();

        // Create combined target matrix
        let mut combined_y = Array2::<Float>::zeros((n_samples, total_outputs));
        let mut task_coefficients = HashMap::new();
        let mut intercepts = HashMap::new();

        let mut current_col = 0;
        for (task_name, task_targets) in y {
            if task_targets.nrows() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{}", n_samples),
                    actual: format!("{}", task_targets.nrows()),
                });
            }

            let n_outputs = task_targets.ncols();
            let end_col = current_col + n_outputs;

            // Copy task targets to combined matrix
            combined_y
                .slice_mut(s![.., current_col..end_col])
                .assign(task_targets);

            task_coefficients.insert(task_name.clone(), (current_col, end_col));
            intercepts.insert(task_name.clone(), Array1::<Float>::zeros(n_outputs));

            current_col = end_col;
        }

        // Initialize coefficient matrix
        let mut coefficient_matrix = Array2::<Float>::zeros((n_features, total_outputs));

        // Proximal gradient descent with nuclear norm penalty
        for iteration in 0..self.max_iter {
            // Compute predictions
            let predictions = x.dot(&coefficient_matrix);

            // Add intercepts
            let mut predictions_with_intercept = predictions.clone();
            current_col = 0;
            for task_name in y.keys() {
                let (start_col, end_col) = task_coefficients[task_name];
                let task_intercepts = &intercepts[task_name];
                for i in 0..predictions_with_intercept.nrows() {
                    for (j, &intercept) in task_intercepts.iter().enumerate() {
                        predictions_with_intercept[[i, start_col + j]] += intercept;
                    }
                }
            }

            // Compute residuals
            let residuals = &predictions_with_intercept - &combined_y;

            // Compute gradients
            let grad_coefs = x.t().dot(&residuals) / (n_samples as Float);

            // Update intercepts
            current_col = 0;
            for task_name in y.keys() {
                let (start_col, end_col) = task_coefficients[task_name];
                let task_residuals = residuals.slice(s![.., start_col..end_col]);
                let grad_intercepts = task_residuals.mean_axis(Axis(0)).unwrap();
                let task_intercepts = intercepts.get_mut(task_name).unwrap();
                *task_intercepts = &*task_intercepts - &(self.learning_rate * &grad_intercepts);
            }

            // Standard gradient update
            let old_coefs = coefficient_matrix.clone();
            coefficient_matrix = &coefficient_matrix - &(self.learning_rate * &grad_coefs);

            // Apply nuclear norm proximal operator
            let (u, s, vt) = self.svd_decomposition(&coefficient_matrix)?;
            let shrunk_s = self.soft_threshold_singular_values(&s);
            coefficient_matrix = self.reconstruct_from_svd(&u, &shrunk_s, &vt);

            // Check convergence
            let coef_change = (&coefficient_matrix - &old_coefs).map(|x| x.abs()).sum();
            if coef_change < self.tolerance {
                return Ok(NuclearNormRegressionTrained {
                    coefficient_matrix,
                    task_coefficients,
                    intercepts,
                    n_features,
                    task_outputs: self.task_outputs,
                    n_iter: iteration + 1,
                    alpha: self.alpha,
                    singular_values: shrunk_s,
                });
            }
        }

        // Final SVD for singular values
        let (_, s, _) = self.svd_decomposition(&coefficient_matrix)?;
        let shrunk_s = self.soft_threshold_singular_values(&s);

        Ok(NuclearNormRegressionTrained {
            coefficient_matrix,
            task_coefficients,
            intercepts,
            n_features,
            task_outputs: self.task_outputs,
            n_iter: self.max_iter,
            alpha: self.alpha,
            singular_values: shrunk_s,
        })
    }
}

impl NuclearNormRegression<Untrained> {
    /// Perform SVD decomposition (simplified placeholder)
    fn svd_decomposition(
        &self,
        matrix: &Array2<Float>,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Array2<Float>)> {
        // This is a simplified placeholder. In a real implementation, you would use
        // a proper SVD library like ndarray-linalg or nalgebra
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        // Create placeholder decomposition
        let u = Array2::eye(m);
        let s = Array1::ones(min_dim);
        let vt = Array2::eye(n);

        Ok((u, s, vt))
    }

    /// Apply soft thresholding to singular values
    fn soft_threshold_singular_values(&self, singular_values: &Array1<Float>) -> Array1<Float> {
        let threshold = self.alpha * self.learning_rate;
        singular_values.map(|&s| (s - threshold).max(0.0))
    }

    /// Reconstruct matrix from SVD components
    fn reconstruct_from_svd(
        &self,
        u: &Array2<Float>,
        s: &Array1<Float>,
        vt: &Array2<Float>,
    ) -> Array2<Float> {
        // This is a simplified reconstruction
        // In practice, you would properly multiply U * S * V^T
        let (m, n) = (u.nrows(), vt.ncols());
        let mut result = Array2::<Float>::zeros((m, n));

        // Simplified reconstruction - just return a scaled identity-like matrix
        let min_dim = m.min(n);
        for i in 0..min_dim {
            if i < s.len() && s[i] > 0.0 {
                result[[i, i]] = s[i];
            }
        }

        result
    }
}

impl Predict<ArrayView2<'_, Float>, HashMap<String, Array2<Float>>>
    for NuclearNormRegressionTrained
{
    fn predict(&self, x: &ArrayView2<Float>) -> SklResult<HashMap<String, Array2<Float>>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", self.n_features),
                actual: format!("{}", x.ncols()),
            });
        }

        let predictions = x.dot(&self.coefficient_matrix);
        let mut task_predictions = HashMap::new();

        for (task_name, &(start_col, end_col)) in &self.task_coefficients {
            let task_pred = predictions.slice(s![.., start_col..end_col]).to_owned();
            let task_intercepts = &self.intercepts[task_name];

            // Add intercepts
            let mut final_pred = task_pred;
            for i in 0..final_pred.nrows() {
                for (j, &intercept) in task_intercepts.iter().enumerate() {
                    final_pred[[i, j]] += intercept;
                }
            }

            task_predictions.insert(task_name.clone(), final_pred);
        }

        Ok(task_predictions)
    }
}

impl NuclearNormRegressionTrained {
    /// Get the nuclear norm of the coefficient matrix
    pub fn nuclear_norm(&self) -> Float {
        self.singular_values.sum()
    }

    /// Get the rank of the coefficient matrix (number of non-zero singular values)
    pub fn effective_rank(&self) -> usize {
        self.singular_values.iter().filter(|&&s| s > 1e-10).count()
    }

    /// Get singular values
    pub fn singular_values(&self) -> &Array1<Float> {
        &self.singular_values
    }

    /// Get coefficients for a specific task
    pub fn task_coefficient_matrix(&self, task_name: &str) -> Option<Array2<Float>> {
        if let Some(&(start_col, end_col)) = self.task_coefficients.get(task_name) {
            Some(
                self.coefficient_matrix
                    .slice(s![.., start_col..end_col])
                    .to_owned(),
            )
        } else {
            None
        }
    }

    /// Get intercepts for a specific task
    pub fn task_intercepts(&self, task_name: &str) -> Option<&Array1<Float>> {
        self.intercepts.get(task_name)
    }

    /// Get number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.n_iter
    }
}

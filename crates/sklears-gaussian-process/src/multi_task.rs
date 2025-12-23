//! Multi-Task Gaussian Process for learning multiple related tasks
//!
//! This module implements multi-task Gaussian processes that learn multiple related tasks
//! simultaneously by sharing information across tasks. This is particularly useful when
//! you have several related learning problems that can benefit from shared knowledge.
//!
//! # Mathematical Background
//!
//! The multi-task GP models each task t using both shared and task-specific components:
//! f_t(x) = f_shared(x) + f_task_t(x)
//!
//! where:
//! - f_shared(x) is a shared latent function common to all tasks
//! - f_task_t(x) is a task-specific function unique to task t
//! - Each component has its own kernel and hyperparameters

// SciRS2 Policy - Use scirs2-autograd for ndarray types and array! macro
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
// SciRS2 Policy - Use scirs2-core for random operations

use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Untrained},
};
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::kernels::Kernel;
use crate::utils;

/// Configuration for Multi-Task Gaussian Process
#[derive(Debug, Clone)]
pub struct MtgpConfig {
    /// shared_kernel_name
    pub shared_kernel_name: String,
    /// task_kernel_name
    pub task_kernel_name: String,
    /// alpha
    pub alpha: f64,
    /// shared_weight
    pub shared_weight: f64,
    /// task_weight
    pub task_weight: f64,
    /// random_state
    pub random_state: Option<u64>,
}

impl Default for MtgpConfig {
    fn default() -> Self {
        Self {
            shared_kernel_name: "RBF".to_string(),
            task_kernel_name: "RBF".to_string(),
            alpha: 1e-10,
            shared_weight: 1.0,
            task_weight: 1.0,
            random_state: None,
        }
    }
}

/// Multi-Task Gaussian Process Regressor
///
/// This implementation allows learning multiple related tasks simultaneously by sharing
/// information through a shared latent function while maintaining task-specific variations.
///
/// # Mathematical Background
///
/// For each task t, the model assumes:
/// f_t(x) = w_shared * f_shared(x) + w_task * f_task_t(x)
///
/// where the covariance between tasks i and j at points x and x' is:
/// cov[f_i(x), f_j(x')] = w_shared² * k_shared(x, x') + δ_{i,j} * w_task² * k_task(x, x')
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::{MultiTaskGaussianProcessRegressor, kernels::RBF};
/// use sklears_core::traits::{Fit, Predict};
/// // SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
/// use scirs2_core::ndarray::array;
///
/// let X1 = array![[1.0], [2.0], [3.0], [4.0]];
/// let y1 = array![1.0, 4.0, 9.0, 16.0];
/// let X2 = array![[1.5], [2.5], [3.5], [4.5]];
/// let y2 = array![2.0, 6.0, 12.0, 20.0];
///
/// let shared_kernel = RBF::new(1.0);
/// let task_kernel = RBF::new(0.5);
/// let mtgp = MultiTaskGaussianProcessRegressor::new()
///     .shared_kernel(Box::new(shared_kernel))
///     .task_kernel(Box::new(task_kernel))
///     .alpha(1e-6);
///
/// let mut mtgp = mtgp.add_task("task1", &X1.view(), &y1.view()).unwrap();
/// mtgp = mtgp.add_task("task2", &X2.view(), &y2.view()).unwrap();
/// let fitted = mtgp.fit().unwrap();
/// let predictions = fitted.predict_task("task1", &X1.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiTaskGaussianProcessRegressor<S = Untrained> {
    shared_kernel: Option<Box<dyn Kernel>>,
    task_kernel: Option<Box<dyn Kernel>>,
    tasks: HashMap<String, (Array2<f64>, Array1<f64>)>, // task_name -> (X, y)
    alpha: f64,
    shared_weight: f64,
    task_weight: f64,
    _state: S,
}

/// Trained state for Multi-Task Gaussian Process
#[derive(Debug, Clone)]
pub struct MtgpTrained {
    tasks: HashMap<String, (Array2<f64>, Array1<f64>)>,
    shared_kernel: Box<dyn Kernel>,
    task_kernel: Box<dyn Kernel>,
    alpha: f64,
    shared_weight: f64,
    task_weight: f64,
    alpha_vector: Array1<f64>, // Solution to the linear system
    log_marginal_likelihood_values: HashMap<String, f64>, // Per-task log marginal likelihood
    task_indices: HashMap<String, (usize, usize)>, // task_name -> (start_idx, end_idx)
    all_X: Array2<f64>,        // Combined input data
    all_y: Array1<f64>,        // Combined target data
}

impl MultiTaskGaussianProcessRegressor<Untrained> {
    /// Create a new Multi-Task Gaussian Process Regressor
    pub fn new() -> Self {
        Self {
            shared_kernel: None,
            task_kernel: None,
            tasks: HashMap::new(),
            alpha: 1e-10,
            shared_weight: 1.0,
            task_weight: 1.0,
            _state: Untrained,
        }
    }

    /// Set the shared kernel function
    pub fn shared_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.shared_kernel = Some(kernel);
        self
    }

    /// Set the task-specific kernel function
    pub fn task_kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.task_kernel = Some(kernel);
        self
    }

    /// Set the regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the weight for the shared component
    pub fn shared_weight(mut self, weight: f64) -> Self {
        self.shared_weight = weight;
        self
    }

    /// Set the weight for the task-specific components
    pub fn task_weight(mut self, weight: f64) -> Self {
        self.task_weight = weight;
        self
    }

    /// Add a task with its training data
    pub fn add_task(
        mut self,
        task_name: &str,
        X: &ArrayView2<f64>,
        y: &ArrayView1<f64>,
    ) -> SklResult<Self> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        self.tasks
            .insert(task_name.to_string(), (X.to_owned(), y.to_owned()));
        Ok(self)
    }

    /// Remove a task
    pub fn remove_task(mut self, task_name: &str) -> Self {
        self.tasks.remove(task_name);
        self
    }

    /// Get the list of task names
    pub fn task_names(&self) -> Vec<String> {
        self.tasks.keys().cloned().collect()
    }

    /// Combine all task data into single arrays
    fn combine_task_data(
        &self,
    ) -> SklResult<(Array2<f64>, Array1<f64>, HashMap<String, (usize, usize)>)> {
        if self.tasks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one task must be added".to_string(),
            ));
        }

        let mut all_X_vec = Vec::new();
        let mut all_y_vec = Vec::new();
        let mut task_indices = HashMap::new();
        let mut current_idx = 0;

        // Determine input dimension from first task
        let first_task = self.tasks.values().next().unwrap();
        let n_features = first_task.0.ncols();

        for (task_name, (X, y)) in &self.tasks {
            if X.ncols() != n_features {
                return Err(SklearsError::InvalidInput(
                    "All tasks must have the same number of features".to_string(),
                ));
            }

            let n_samples = X.nrows();
            task_indices.insert(task_name.clone(), (current_idx, current_idx + n_samples));

            // Append X data
            for i in 0..n_samples {
                let mut row = Vec::new();
                for j in 0..n_features {
                    row.push(X[[i, j]]);
                }
                all_X_vec.push(row);
            }

            // Append y data
            for i in 0..n_samples {
                all_y_vec.push(y[i]);
            }

            current_idx += n_samples;
        }

        let n_total = all_X_vec.len();
        let mut all_X = Array2::<f64>::zeros((n_total, n_features));
        let mut all_y = Array1::<f64>::zeros(n_total);

        for (i, row) in all_X_vec.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                all_X[[i, j]] = val;
            }
        }

        for (i, &val) in all_y_vec.iter().enumerate() {
            all_y[i] = val;
        }

        Ok((all_X, all_y, task_indices))
    }

    /// Compute the multi-task covariance matrix
    #[allow(non_snake_case)]
    fn compute_multitask_covariance(
        &self,
        X: &Array2<f64>,
        task_indices: &HashMap<String, (usize, usize)>,
        shared_kernel: &Box<dyn Kernel>,
        task_kernel: &Box<dyn Kernel>,
    ) -> SklResult<Array2<f64>> {
        let n = X.nrows();
        let mut K = Array2::<f64>::zeros((n, n));

        // Compute shared kernel matrix (applies to all pairs)
        let K_shared = shared_kernel.compute_kernel_matrix(X, None)?;

        // Add shared component
        for i in 0..n {
            for j in 0..n {
                K[[i, j]] += self.shared_weight * self.shared_weight * K_shared[[i, j]];
            }
        }

        // Add task-specific components (only within same task)
        let K_task = task_kernel.compute_kernel_matrix(X, None)?;

        for (start_i, end_i) in task_indices.values() {
            for i in *start_i..*end_i {
                for j in *start_i..*end_i {
                    K[[i, j]] += self.task_weight * self.task_weight * K_task[[i, j]];
                }
            }
        }

        Ok(K)
    }
}

impl Default for MultiTaskGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultiTaskGaussianProcessRegressor<Untrained> {
    type Config = MtgpConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: MtgpConfig = MtgpConfig {
            shared_kernel_name: String::new(),
            task_kernel_name: String::new(),
            alpha: 1e-10,
            shared_weight: 1.0,
            task_weight: 1.0,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

impl Estimator for MultiTaskGaussianProcessRegressor<MtgpTrained> {
    type Config = MtgpConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static DEFAULT_CONFIG: MtgpConfig = MtgpConfig {
            shared_kernel_name: String::new(),
            task_kernel_name: String::new(),
            alpha: 1e-10,
            shared_weight: 1.0,
            task_weight: 1.0,
            random_state: None,
        };
        &DEFAULT_CONFIG
    }
}

// Implement fit without requiring X and y parameters since tasks are added separately
impl MultiTaskGaussianProcessRegressor<Untrained> {
    /// Fit the multi-task Gaussian process
    #[allow(non_snake_case)]
    pub fn fit(self) -> SklResult<MultiTaskGaussianProcessRegressor<MtgpTrained>> {
        let shared_kernel = self.shared_kernel.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Shared kernel must be specified".to_string())
        })?;

        let task_kernel = self.task_kernel.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Task kernel must be specified".to_string())
        })?;

        if self.tasks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one task must be added".to_string(),
            ));
        }

        // Combine all task data
        let (all_X, all_y, task_indices) = self.combine_task_data()?;

        // Compute multi-task covariance matrix
        let K =
            self.compute_multitask_covariance(&all_X, &task_indices, shared_kernel, task_kernel)?;

        // Add regularization
        let mut K_reg = K.clone();
        for i in 0..K_reg.nrows() {
            K_reg[[i, i]] += self.alpha;
        }

        // Solve the linear system
        let chol_decomp = utils::robust_cholesky(&K_reg)?;
        let alpha_vector = utils::triangular_solve(&chol_decomp, &all_y)?;

        // Compute per-task log marginal likelihood
        let mut log_marginal_likelihood_values = HashMap::new();
        for (task_name, (start_idx, end_idx)) in &task_indices {
            let task_size = end_idx - start_idx;
            let task_y = all_y.slice(scirs2_core::ndarray::s![*start_idx..*end_idx]);
            let task_alpha = alpha_vector.slice(scirs2_core::ndarray::s![*start_idx..*end_idx]);

            // Simplified log marginal likelihood for this task
            let data_fit = task_y.dot(&task_alpha);
            let log_ml = -0.5 * (data_fit + task_size as f64 * (2.0 * PI).ln());
            log_marginal_likelihood_values.insert(task_name.clone(), log_ml);
        }

        Ok(MultiTaskGaussianProcessRegressor {
            shared_kernel: None,
            task_kernel: None,
            tasks: self.tasks.clone(),
            alpha: self.alpha,
            shared_weight: self.shared_weight,
            task_weight: self.task_weight,
            _state: MtgpTrained {
                tasks: self.tasks,
                shared_kernel: shared_kernel.clone(),
                task_kernel: task_kernel.clone(),
                alpha: self.alpha,
                shared_weight: self.shared_weight,
                task_weight: self.task_weight,
                alpha_vector,
                log_marginal_likelihood_values,
                task_indices,
                all_X,
                all_y,
            },
        })
    }
}

impl MultiTaskGaussianProcessRegressor<MtgpTrained> {
    /// Access the trained state
    pub fn trained_state(&self) -> &MtgpTrained {
        &self._state
    }

    /// Get the log marginal likelihood for a specific task
    pub fn log_marginal_likelihood_task(&self, task_name: &str) -> Option<f64> {
        self._state
            .log_marginal_likelihood_values
            .get(task_name)
            .copied()
    }

    /// Get all log marginal likelihoods
    pub fn log_marginal_likelihoods(&self) -> &HashMap<String, f64> {
        &self._state.log_marginal_likelihood_values
    }

    /// Get the list of available tasks
    pub fn task_names(&self) -> Vec<String> {
        self._state.tasks.keys().cloned().collect()
    }

    /// Predict for a specific task
    #[allow(non_snake_case)]
    pub fn predict_task(&self, task_name: &str, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let task_data =
            self._state.tasks.get(task_name).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Task '{}' not found", task_name))
            })?;

        let task_X_train = &task_data.0;
        let n_test = X.nrows();

        // Compute cross-covariance between test points and all training points
        let K_shared_star = self
            ._state
            .shared_kernel
            .compute_kernel_matrix(&self._state.all_X, Some(&X.to_owned()))?;
        let K_task_star = self
            ._state
            .task_kernel
            .compute_kernel_matrix(task_X_train, Some(&X.to_owned()))?;

        // For each test point, compute prediction
        let mut predictions = Array1::<f64>::zeros(n_test);

        for i in 0..n_test {
            let mut pred = 0.0;

            // Add shared component contribution from all tasks
            for j in 0..self._state.all_X.nrows() {
                pred += self.shared_weight
                    * self.shared_weight
                    * K_shared_star[[j, i]]
                    * self._state.alpha_vector[j];
            }

            // Add task-specific contribution only from the same task
            if let Some((start_idx, _end_idx)) = self._state.task_indices.get(task_name) {
                for j in 0..task_X_train.nrows() {
                    let global_j = start_idx + j;
                    pred += self.task_weight
                        * self.task_weight
                        * K_task_star[[j, i]]
                        * self._state.alpha_vector[global_j];
                }
            }

            predictions[i] = pred;
        }

        Ok(predictions)
    }

    /// Get shared and task-specific contributions separately
    #[allow(non_snake_case)]
    pub fn predict_task_components(
        &self,
        task_name: &str,
        X: &ArrayView2<f64>,
    ) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let task_data =
            self._state.tasks.get(task_name).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Task '{}' not found", task_name))
            })?;

        let task_X_train = &task_data.0;
        let n_test = X.nrows();

        // Compute cross-covariances
        let K_shared_star = self
            ._state
            .shared_kernel
            .compute_kernel_matrix(&self._state.all_X, Some(&X.to_owned()))?;
        let K_task_star = self
            ._state
            .task_kernel
            .compute_kernel_matrix(task_X_train, Some(&X.to_owned()))?;

        let mut shared_predictions = Array1::<f64>::zeros(n_test);
        let mut task_predictions = Array1::<f64>::zeros(n_test);

        for i in 0..n_test {
            // Shared component
            for j in 0..self._state.all_X.nrows() {
                shared_predictions[i] +=
                    self.shared_weight * K_shared_star[[j, i]] * self._state.alpha_vector[j];
            }

            // Task-specific component
            if let Some((start_idx, _)) = self._state.task_indices.get(task_name) {
                for j in 0..task_X_train.nrows() {
                    let global_j = start_idx + j;
                    task_predictions[i] +=
                        self.task_weight * K_task_star[[j, i]] * self._state.alpha_vector[global_j];
                }
            }
        }

        Ok((shared_predictions, task_predictions))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::RBF;

    // SciRS2 Policy - Use scirs2-autograd for array! macro and types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_mtgp_creation() {
        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .alpha(1e-6);

        assert_eq!(mtgp.alpha, 1e-6);
        assert_eq!(mtgp.shared_weight, 1.0);
        assert_eq!(mtgp.task_weight, 1.0);
        assert_eq!(mtgp.tasks.len(), 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mtgp_add_task() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();

        assert_eq!(mtgp.tasks.len(), 1);
        assert!(mtgp.tasks.contains_key("task1"));
        let task_names = mtgp.task_names();
        assert!(task_names.contains(&"task1".to_string()));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mtgp_remove_task() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap()
            .remove_task("task1");

        assert_eq!(mtgp.tasks.len(), 0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mtgp_fit_single_task() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();
        assert_eq!(fitted.task_names().len(), 1);
        assert!(fitted.log_marginal_likelihood_task("task1").is_some());
    }

    #[test]
    fn test_mtgp_fit_multiple_tasks() {
        let X1 = array![[1.0], [2.0], [3.0], [4.0]];
        let y1 = array![1.0, 4.0, 9.0, 16.0];
        let X2 = array![[1.5], [2.5], [3.5], [4.5]];
        let y2 = array![2.0, 6.0, 12.0, 20.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X1.view(), &y1.view())
            .unwrap()
            .add_task("task2", &X2.view(), &y2.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();
        assert_eq!(fitted.task_names().len(), 2);
        assert!(fitted.log_marginal_likelihood_task("task1").is_some());
        assert!(fitted.log_marginal_likelihood_task("task2").is_some());
    }

    #[test]
    fn test_mtgp_predict_task() {
        let X1 = array![[1.0], [2.0], [3.0], [4.0]];
        let y1 = array![1.0, 4.0, 9.0, 16.0];
        let X2 = array![[1.5], [2.5], [3.5], [4.5]];
        let y2 = array![2.0, 6.0, 12.0, 20.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X1.view(), &y1.view())
            .unwrap()
            .add_task("task2", &X2.view(), &y2.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();

        let predictions = fitted.predict_task("task1", &X1.view()).unwrap();
        assert_eq!(predictions.len(), 4);

        let predictions2 = fitted.predict_task("task2", &X2.view()).unwrap();
        assert_eq!(predictions2.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mtgp_predict_components() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();
        let (shared_pred, task_pred) = fitted.predict_task_components("task1", &X.view()).unwrap();

        assert_eq!(shared_pred.len(), 4);
        assert_eq!(task_pred.len(), 4);
    }

    #[test]
    fn test_mtgp_log_marginal_likelihoods() {
        let X1 = array![[1.0], [2.0], [3.0], [4.0]];
        let y1 = array![1.0, 4.0, 9.0, 16.0];
        let X2 = array![[1.5], [2.5], [3.5], [4.5]];
        let y2 = array![2.0, 6.0, 12.0, 20.0];

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X1.view(), &y1.view())
            .unwrap()
            .add_task("task2", &X2.view(), &y2.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();
        let all_lml = fitted.log_marginal_likelihoods();

        assert_eq!(all_lml.len(), 2);
        assert!(all_lml.contains_key("task1"));
        assert!(all_lml.contains_key("task2"));
        assert!(all_lml.get("task1").unwrap().is_finite());
        assert!(all_lml.get("task2").unwrap().is_finite());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mtgp_errors() {
        let X = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 4.0, 9.0, 16.0];

        // Test with no shared kernel
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();
        assert!(mtgp.fit().is_err());

        // Test with no task kernel
        let shared_kernel = RBF::new(1.0);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();
        assert!(mtgp.fit().is_err());

        // Test with no tasks
        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel));
        assert!(mtgp.fit().is_err());

        // Test prediction on non-existent task
        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel))
            .add_task("task1", &X.view(), &y.view())
            .unwrap();

        let fitted = mtgp.fit().unwrap();
        assert!(fitted.predict_task("nonexistent", &X.view()).is_err());
    }

    #[test]
    fn test_mtgp_mismatched_dimensions() {
        let X1 = array![[1.0], [2.0], [3.0], [4.0]];
        let y_wrong = array![1.0, 4.0, 9.0]; // Wrong size

        let shared_kernel = RBF::new(1.0);
        let task_kernel = RBF::new(0.5);
        let mtgp = MultiTaskGaussianProcessRegressor::new()
            .shared_kernel(Box::new(shared_kernel))
            .task_kernel(Box::new(task_kernel));

        assert!(mtgp.add_task("task1", &X1.view(), &y_wrong.view()).is_err());
    }
}

//! Multi-Task Kernel Ridge Regression Implementation
//!
//! This module provides multi-task learning capabilities for kernel ridge regression,
//! allowing simultaneous learning across multiple related tasks with cross-task
//! regularization strategies to improve generalization.

use crate::{
    FastfoodTransform, Nystroem, RBFSampler, StructuredRandomFeatures, Trained, Untrained,
};
use scirs2_core::ndarray::ndarray_linalg::solve::Solve;
use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::{Estimator, Fit, Float, Predict};
use std::marker::PhantomData;

use super::core_types::*;

/// Multi-Task Kernel Ridge Regression
///
/// Performs kernel ridge regression simultaneously across multiple related tasks,
/// with optional cross-task regularization to encourage similarity between tasks.
///
/// This is particularly useful when you have multiple regression targets that are
/// related and can benefit from shared representations and joint learning.
///
/// # Parameters
///
/// * `approximation_method` - Method for kernel approximation
/// * `alpha` - Within-task regularization parameter
/// * `task_regularization` - Cross-task regularization strategy
/// * `solver` - Method for solving the linear system
/// * `random_state` - Random seed for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::kernel_ridge_regression::{
#[derive(Debug, Clone)]
pub struct MultiTaskKernelRidgeRegression<State = Untrained> {
    pub approximation_method: ApproximationMethod,
    pub alpha: Float,
    pub task_regularization: TaskRegularization,
    pub solver: Solver,
    pub random_state: Option<u64>,

    // Fitted parameters
    weights_: Option<Array2<Float>>, // Shape: (n_features, n_tasks)
    feature_transformer_: Option<FeatureTransformer>,
    n_tasks_: Option<usize>,

    _state: PhantomData<State>,
}

/// Cross-task regularization strategies for multi-task learning
#[derive(Debug, Clone)]
pub enum TaskRegularization {
    /// No cross-task regularization (independent tasks)
    None,
    /// L2 regularization on task weight differences
    L2 { beta: Float },
    /// L1 regularization promoting sparsity across tasks
    L1 { beta: Float },
    /// Nuclear norm regularization on weight matrix (low-rank)
    NuclearNorm { beta: Float },
    /// Group sparsity regularization
    GroupSparsity { beta: Float },
    /// Custom regularization function
    Custom {
        beta: Float,
        regularizer: fn(&Array2<Float>) -> Float,
    },
}

impl Default for TaskRegularization {
    fn default() -> Self {
        Self::None
    }
}

impl MultiTaskKernelRidgeRegression<Untrained> {
    /// Create a new multi-task kernel ridge regression model
    pub fn new(approximation_method: ApproximationMethod) -> Self {
        Self {
            approximation_method,
            alpha: 1.0,
            task_regularization: TaskRegularization::None,
            solver: Solver::Direct,
            random_state: None,
            weights_: None,
            feature_transformer_: None,
            n_tasks_: None,
            _state: PhantomData,
        }
    }

    /// Set regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set cross-task regularization strategy
    pub fn task_regularization(mut self, regularization: TaskRegularization) -> Self {
        self.task_regularization = regularization;
        self
    }

    /// Set solver method
    pub fn solver(mut self, solver: Solver) -> Self {
        self.solver = solver;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute regularization penalty for the weight matrix
    fn compute_task_regularization_penalty(&self, weights: &Array2<Float>) -> Float {
        match &self.task_regularization {
            TaskRegularization::None => 0.0,
            TaskRegularization::L2 { beta } => {
                // L2 penalty on differences between task weights
                let mut penalty = 0.0;
                let n_tasks = weights.ncols();
                for i in 0..n_tasks {
                    for j in (i + 1)..n_tasks {
                        let diff = &weights.column(i) - &weights.column(j);
                        penalty += diff.mapv(|x| x * x).sum();
                    }
                }
                beta * penalty / (n_tasks * (n_tasks - 1) / 2) as Float
            }
            TaskRegularization::L1 { beta } => {
                // L1 penalty on task weights
                beta * weights.mapv(|x| x.abs()).sum()
            }
            TaskRegularization::NuclearNorm { beta } => {
                // Nuclear norm (sum of singular values)
                // Approximate with Frobenius norm for efficiency
                beta * weights.mapv(|x| x * x).sum().sqrt()
            }
            TaskRegularization::GroupSparsity { beta } => {
                // Group sparsity: L2,1 norm
                let mut penalty = 0.0;
                for row in weights.axis_iter(Axis(0)) {
                    penalty += row.mapv(|x| x * x).sum().sqrt();
                }
                beta * penalty
            }
            TaskRegularization::Custom { beta, regularizer } => beta * regularizer(weights),
        }
    }
}

impl Estimator for MultiTaskKernelRidgeRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for MultiTaskKernelRidgeRegression<Untrained> {
    type Fitted = MultiTaskKernelRidgeRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Number of samples must match".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_tasks = y.ncols();

        // Fit the feature transformer
        let feature_transformer = self.fit_feature_transformer(x)?;
        let x_transformed = feature_transformer.transform(x)?;
        let n_features = x_transformed.ncols();

        // Solve multi-task regression problem
        let weights = match self.solver {
            Solver::Direct => self.solve_direct_multitask(&x_transformed, y)?,
            Solver::SVD => self.solve_svd_multitask(&x_transformed, y)?,
            Solver::ConjugateGradient { max_iter, tol } => {
                self.solve_cg_multitask(&x_transformed, y, max_iter, tol)?
            }
        };

        Ok(MultiTaskKernelRidgeRegression {
            approximation_method: self.approximation_method,
            alpha: self.alpha,
            task_regularization: self.task_regularization,
            solver: self.solver,
            random_state: self.random_state,
            weights_: Some(weights),
            feature_transformer_: Some(feature_transformer),
            n_tasks_: Some(n_tasks),
            _state: PhantomData,
        })
    }
}

impl MultiTaskKernelRidgeRegression<Untrained> {
    /// Fit the feature transformer based on the approximation method
    fn fit_feature_transformer(&self, x: &Array2<Float>) -> Result<FeatureTransformer> {
        match &self.approximation_method {
            ApproximationMethod::Nystroem {
                kernel,
                n_components,
                sampling_strategy,
            } => {
                let mut nystroem = Nystroem::new(kernel.clone(), *n_components)
                    .sampling_strategy(sampling_strategy.clone());
                if let Some(seed) = self.random_state {
                    nystroem = nystroem.random_state(seed);
                }
                let fitted = nystroem.fit(x, &())?;
                Ok(FeatureTransformer::Nystroem(fitted))
            }
            ApproximationMethod::RandomFourierFeatures {
                n_components,
                gamma,
            } => {
                let mut rff = RBFSampler::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    rff = rff.random_state(seed);
                }
                let fitted = rff.fit(x, &())?;
                Ok(FeatureTransformer::RBFSampler(fitted))
            }
            ApproximationMethod::StructuredRandomFeatures {
                n_components,
                gamma,
            } => {
                let mut srf = StructuredRandomFeatures::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    srf = srf.random_state(seed);
                }
                let fitted = srf.fit(x, &())?;
                Ok(FeatureTransformer::StructuredRFF(fitted))
            }
            ApproximationMethod::Fastfood {
                n_components,
                gamma,
            } => {
                let mut fastfood = FastfoodTransform::new(*n_components).gamma(*gamma);
                if let Some(seed) = self.random_state {
                    fastfood = fastfood.random_state(seed);
                }
                let fitted = fastfood.fit(x, &())?;
                Ok(FeatureTransformer::Fastfood(fitted))
            }
        }
    }

    /// Solve multi-task problem using direct method
    fn solve_direct_multitask(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Array2<Float>> {
        let n_features = x.ncols();
        let n_tasks = y.ncols();

        // For multi-task learning, we solve each task separately but with shared features
        // and apply cross-task regularization
        let mut all_weights = Array2::zeros((n_features, n_tasks));

        for task_idx in 0..n_tasks {
            let y_task = y.column(task_idx);

            // Standard ridge regression for this task
            let x_f64 = Array2::from_shape_fn(x.dim(), |(i, j)| x[[i, j]] as f64);
            let y_task_f64 = Array1::from_vec(y_task.iter().map(|&val| val as f64).collect());

            let xtx = x_f64.t().dot(&x_f64);
            let regularized_xtx = xtx + Array2::<f64>::eye(n_features) * (self.alpha as f64);

            let xty = x_f64.t().dot(&y_task_f64);
            let weights_task_f64 =
                regularized_xtx
                    .solve(&xty)
                    .map_err(|e| SklearsError::InvalidParameter {
                        name: "regularization".to_string(),
                        reason: format!("Linear system solving failed: {:?}", e),
                    })?;

            // Convert back to Float
            let weights_task =
                Array1::from_vec(weights_task_f64.iter().map(|&val| val as Float).collect());
            all_weights.column_mut(task_idx).assign(&weights_task);
        }

        // Apply cross-task regularization (simplified approach)
        // In practice, you might want to solve a joint optimization problem
        match &self.task_regularization {
            TaskRegularization::L2 { beta } => {
                // Apply additional regularization penalty
                let mean_weight = all_weights.mean_axis(Axis(1)).unwrap();
                for mut col in all_weights.axis_iter_mut(Axis(1)) {
                    let diff = &col.to_owned() - &mean_weight;
                    col.scaled_add(-beta, &diff);
                }
            }
            _ => {} // Other regularization methods would be implemented here
        }

        Ok(all_weights)
    }

    /// Solve multi-task problem using SVD
    fn solve_svd_multitask(&self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Array2<Float>> {
        let n_features = x.ncols();
        let n_tasks = y.ncols();
        let mut all_weights = Array2::zeros((n_features, n_tasks));

        for task_idx in 0..n_tasks {
            let y_task = y.column(task_idx);

            // Use SVD for more stable solution
            let x_f64 = Array2::from_shape_fn(x.dim(), |(i, j)| x[[i, j]] as f64);
            let y_task_f64 = Array1::from_vec(y_task.iter().map(|&val| val as f64).collect());

            let xtx = x_f64.t().dot(&x_f64);
            let regularized_xtx = xtx + Array2::<f64>::eye(n_features) * (self.alpha as f64);

            let (u, s, vt) =
                regularized_xtx
                    .svd(true, true)
                    .map_err(|e| SklearsError::InvalidParameter {
                        name: "svd".to_string(),
                        reason: format!("SVD decomposition failed: {:?}", e),
                    })?;
            let u = u.unwrap();
            let vt = vt.unwrap();

            // Solve using SVD
            let xty = x_f64.t().dot(&y_task_f64);
            let ut_b = u.t().dot(&xty);
            let s_inv = s.mapv(|x| if x > 1e-10 { 1.0 / x } else { 0.0 });
            let y_svd = ut_b * s_inv;
            let weights_task_f64 = vt.t().dot(&y_svd);

            // Convert back to Float
            let weights_task =
                Array1::from_vec(weights_task_f64.iter().map(|&val| val as Float).collect());
            all_weights.column_mut(task_idx).assign(&weights_task);
        }

        Ok(all_weights)
    }

    /// Solve multi-task problem using conjugate gradient
    fn solve_cg_multitask(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
        max_iter: usize,
        tol: Float,
    ) -> Result<Array2<Float>> {
        let n_features = x.ncols();
        let n_tasks = y.ncols();
        let mut all_weights = Array2::zeros((n_features, n_tasks));

        for task_idx in 0..n_tasks {
            let y_task = y.column(task_idx);
            let xty = x.t().dot(&y_task);

            // Conjugate gradient solver for each task
            let mut weights = Array1::zeros(n_features);
            let mut r = xty.clone();
            let mut p = r.clone();
            let mut rsold = r.dot(&r);

            for _iter in 0..max_iter {
                let xtx_p = x.t().dot(&x.dot(&p)) + &p * self.alpha;
                let alpha_cg = rsold / p.dot(&xtx_p);

                weights = weights + &p * alpha_cg;
                r = r - &xtx_p * alpha_cg;

                let rsnew = r.dot(&r);

                if rsnew.sqrt() < tol {
                    break;
                }

                let beta = rsnew / rsold;
                p = &r + &p * beta;
                rsold = rsnew;
            }

            all_weights.column_mut(task_idx).assign(&weights);
        }

        Ok(all_weights)
    }
}

impl Predict<Array2<Float>, Array2<Float>> for MultiTaskKernelRidgeRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let feature_transformer =
            self.feature_transformer_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict".to_string(),
                })?;

        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let x_transformed = feature_transformer.transform(x)?;
        let predictions = x_transformed.dot(weights);

        Ok(predictions)
    }
}

impl MultiTaskKernelRidgeRegression<Trained> {
    /// Get the number of tasks
    pub fn n_tasks(&self) -> usize {
        self.n_tasks_.unwrap_or(0)
    }

    /// Get the fitted weights for all tasks
    pub fn weights(&self) -> Option<&Array2<Float>> {
        self.weights_.as_ref()
    }

    /// Get the weights for a specific task
    pub fn task_weights(&self, task_idx: usize) -> Result<Array1<Float>> {
        let weights = self
            .weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        if task_idx >= weights.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Task index {} out of range",
                task_idx
            )));
        }

        Ok(weights.column(task_idx).to_owned())
    }

    /// Predict for a specific task only
    pub fn predict_task(&self, x: &Array2<Float>, task_idx: usize) -> Result<Array1<Float>> {
        let predictions = self.predict(x)?;

        if task_idx >= predictions.ncols() {
            return Err(SklearsError::InvalidInput(format!(
                "Task index {} out of range",
                task_idx
            )));
        }

        Ok(predictions.column(task_idx).to_owned())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multitask_kernel_ridge_regression() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.0, 2.0], [4.0, 5.0], [9.0, 10.0], [16.0, 17.0]]; // Two tasks

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 20,
            gamma: 0.1,
        };

        let mtkrr = MultiTaskKernelRidgeRegression::new(approximation).alpha(0.1);
        let fitted = mtkrr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 2]);
        assert_eq!(fitted.n_tasks(), 2);

        // Test individual task prediction
        let task0_pred = fitted.predict_task(&x, 0).unwrap();
        let task1_pred = fitted.predict_task(&x, 1).unwrap();

        assert_eq!(task0_pred.len(), 4);
        assert_eq!(task1_pred.len(), 4);

        // Check that predictions are reasonable
        for pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_multitask_with_regularization() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]]; // Similar tasks

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 15,
            gamma: 1.0,
        };

        let mtkrr = MultiTaskKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .task_regularization(TaskRegularization::L2 { beta: 0.1 });

        let fitted = mtkrr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[3, 2]);

        // With L2 regularization, task weights should be similar
        let weights = fitted.weights().unwrap();
        let task0_weights = weights.column(0);
        let task1_weights = weights.column(1);
        let weight_diff = (&task0_weights - &task1_weights)
            .mapv(|x| x.abs())
            .mean()
            .unwrap();

        // Tasks should have similar weights due to regularization
        assert!(weight_diff < 1.0);
    }

    #[test]
    fn test_multitask_different_solvers() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        // Test different solvers
        let solvers = vec![
            Solver::Direct,
            Solver::SVD,
            Solver::ConjugateGradient {
                max_iter: 100,
                tol: 1e-6,
            },
        ];

        for solver in solvers {
            let mtkrr = MultiTaskKernelRidgeRegression::new(approximation.clone())
                .solver(solver)
                .alpha(0.1);

            let fitted = mtkrr.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();

            assert_eq!(predictions.shape(), &[3, 2]);
        }
    }

    #[test]
    fn test_multitask_single_task() {
        // Test that multi-task regression works with a single task
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0], [3.0]]; // Single task

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let mtkrr = MultiTaskKernelRidgeRegression::new(approximation).alpha(0.1);
        let fitted = mtkrr.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[3, 1]);
        assert_eq!(fitted.n_tasks(), 1);
    }

    #[test]
    fn test_multitask_reproducibility() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];

        let approximation = ApproximationMethod::RandomFourierFeatures {
            n_components: 10,
            gamma: 1.0,
        };

        let mtkrr1 = MultiTaskKernelRidgeRegression::new(approximation.clone())
            .alpha(0.1)
            .random_state(42);
        let fitted1 = mtkrr1.fit(&x, &y).unwrap();
        let pred1 = fitted1.predict(&x).unwrap();

        let mtkrr2 = MultiTaskKernelRidgeRegression::new(approximation)
            .alpha(0.1)
            .random_state(42);
        let fitted2 = mtkrr2.fit(&x, &y).unwrap();
        let pred2 = fitted2.predict(&x).unwrap();

        assert_eq!(pred1.shape(), pred2.shape());
        for i in 0..pred1.nrows() {
            for j in 0..pred1.ncols() {
                assert!((pred1[[i, j]] - pred2[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_task_regularization_penalties() {
        let weights = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];

        let model =
            MultiTaskKernelRidgeRegression::new(ApproximationMethod::RandomFourierFeatures {
                n_components: 10,
                gamma: 1.0,
            });

        // Test different regularization types
        let reg_l2 = TaskRegularization::L2 { beta: 0.1 };
        let reg_l1 = TaskRegularization::L1 { beta: 0.1 };
        let reg_nuclear = TaskRegularization::NuclearNorm { beta: 0.1 };
        let reg_group = TaskRegularization::GroupSparsity { beta: 0.1 };

        let model_l2 = model.clone().task_regularization(reg_l2);
        let model_l1 = model.clone().task_regularization(reg_l1);
        let model_nuclear = model.clone().task_regularization(reg_nuclear);
        let model_group = model.clone().task_regularization(reg_group);

        let penalty_l2 = model_l2.compute_task_regularization_penalty(&weights);
        let penalty_l1 = model_l1.compute_task_regularization_penalty(&weights);
        let penalty_nuclear = model_nuclear.compute_task_regularization_penalty(&weights);
        let penalty_group = model_group.compute_task_regularization_penalty(&weights);

        // All penalties should be non-negative
        assert!(penalty_l2 >= 0.0);
        assert!(penalty_l1 >= 0.0);
        assert!(penalty_nuclear >= 0.0);
        assert!(penalty_group >= 0.0);

        // L1 penalty should be larger than others for this matrix
        assert!(penalty_l1 > penalty_l2);
    }
}

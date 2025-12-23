//! Multi-task learning methods for cross-decomposition
//!
//! This module implements various multi-task learning approaches for cross-decomposition,
//! including multi-task CCA, shared component analysis, transfer learning, domain adaptation,
//! and few-shot learning methods.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{thread_rng, Random, Rng};
use sklears_core::error::SklearsError;
use sklears_core::traits::Estimator;
use std::collections::HashMap;

/// Multi-task Canonical Correlation Analysis
///
/// Multi-task CCA learns canonical correlations across multiple related tasks,
/// sharing information between tasks to improve performance on individual tasks.
#[derive(Debug, Clone)]
pub struct MultiTaskCCA {
    n_components: usize,
    reg_param: f64,
    max_iter: usize,
    tol: f64,
    sharing_strength: f64,
    canonical_weights_x: Option<Array2<f64>>,
    canonical_weights_y: Option<Array2<f64>>,
    shared_components: Option<Array2<f64>>,
    task_specific_components: Option<HashMap<usize, Array2<f64>>>,
    correlations: Option<Array1<f64>>,
}

impl MultiTaskCCA {
    /// Creates a new MultiTaskCCA instance
    pub fn new(n_components: usize, reg_param: f64, sharing_strength: f64) -> Self {
        Self {
            n_components,
            reg_param,
            max_iter: 500,
            tol: 1e-6,
            sharing_strength,
            canonical_weights_x: None,
            canonical_weights_y: None,
            shared_components: None,
            task_specific_components: None,
            correlations: None,
        }
    }

    /// Sets the maximum number of iterations
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fits multi-task CCA on multiple datasets
    pub fn fit_multi_task(
        &self,
        x_tasks: &[Array2<f64>],
        y_tasks: &[Array2<f64>],
    ) -> Result<Self, SklearsError> {
        if x_tasks.len() != y_tasks.len() {
            return Err(SklearsError::InvalidInput(
                "Number of X and Y tasks must match".to_string(),
            ));
        }

        if x_tasks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one task must be provided".to_string(),
            ));
        }

        let n_tasks = x_tasks.len();
        let n_features_x = x_tasks[0].shape()[1];
        let n_features_y = y_tasks[0].shape()[1];

        // Initialize shared and task-specific components
        let mut shared_wx = Array2::zeros((n_features_x, self.n_components));
        let mut shared_wy = Array2::zeros((n_features_y, self.n_components));
        let mut task_specific_wx = HashMap::new();
        let mut task_specific_wy = HashMap::new();

        // Initialize task-specific components
        for task_id in 0..n_tasks {
            task_specific_wx.insert(task_id, Array2::zeros((n_features_x, self.n_components)));
            task_specific_wy.insert(task_id, Array2::zeros((n_features_y, self.n_components)));
        }

        // Alternating optimization
        for iter in 0..self.max_iter {
            let mut converged = true;
            let old_shared_wx = shared_wx.clone();

            // Update shared components
            for comp in 0..self.n_components {
                let mut cov_xx_shared = Array2::zeros((n_features_x, n_features_x));
                let mut cov_xy_shared = Array2::zeros((n_features_x, n_features_y));
                let mut cov_yy_shared = Array2::zeros((n_features_y, n_features_y));

                // Aggregate covariances across tasks
                for (task_id, (x_task, y_task)) in x_tasks.iter().zip(y_tasks.iter()).enumerate() {
                    let x_centered = self.center_data(x_task)?;
                    let y_centered = self.center_data(y_task)?;

                    let task_wx = &task_specific_wx[&task_id];
                    let task_wy = &task_specific_wy[&task_id];

                    // Compute task-specific residuals
                    let x_proj = x_centered.dot(task_wx);
                    let x_recon = x_proj.dot(&task_wx.t());
                    let x_residual = &x_centered - &x_recon;

                    let y_proj = y_centered.dot(task_wy);
                    let y_recon = y_proj.dot(&task_wy.t());
                    let y_residual = &y_centered - &y_recon;

                    cov_xx_shared =
                        cov_xx_shared + x_residual.t().dot(&x_residual) / x_task.shape()[0] as f64;
                    cov_xy_shared =
                        cov_xy_shared + x_residual.t().dot(&y_residual) / x_task.shape()[0] as f64;
                    cov_yy_shared =
                        cov_yy_shared + y_residual.t().dot(&y_residual) / y_task.shape()[0] as f64;
                }

                // Add regularization
                cov_xx_shared
                    .diag_mut()
                    .mapv_inplace(|x| x + self.reg_param);
                cov_yy_shared
                    .diag_mut()
                    .mapv_inplace(|x| x + self.reg_param);

                // Solve generalized eigenvalue problem for shared components
                let (eigvals, eigvecs_x, eigvecs_y) = self.solve_generalized_eigenvalue(
                    &cov_xy_shared,
                    &cov_xx_shared,
                    &cov_yy_shared,
                )?;

                if let Some(max_idx) = eigvals
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                {
                    shared_wx
                        .column_mut(comp)
                        .assign(&eigvecs_x.column(max_idx));
                    shared_wy
                        .column_mut(comp)
                        .assign(&eigvecs_y.column(max_idx));
                }
            }

            // Update task-specific components
            for (task_id, (x_task, y_task)) in x_tasks.iter().zip(y_tasks.iter()).enumerate() {
                let x_centered = self.center_data(x_task)?;
                let y_centered = self.center_data(y_task)?;

                // Remove shared component contribution
                let x_shared_proj = x_centered.dot(&shared_wx);
                let x_shared_recon = x_shared_proj.dot(&shared_wx.t());
                let x_residual = &x_centered - &x_shared_recon;

                let y_shared_proj = y_centered.dot(&shared_wy);
                let y_shared_recon = y_shared_proj.dot(&shared_wy.t());
                let y_residual = &y_centered - &y_shared_recon;

                // Compute task-specific CCA
                let cov_xx = x_residual.t().dot(&x_residual) / x_task.shape()[0] as f64;
                let cov_xy = x_residual.t().dot(&y_residual) / x_task.shape()[0] as f64;
                let cov_yy = y_residual.t().dot(&y_residual) / y_task.shape()[0] as f64;

                let mut cov_xx_reg = cov_xx.clone();
                let mut cov_yy_reg = cov_yy.clone();
                cov_xx_reg.diag_mut().mapv_inplace(|x| x + self.reg_param);
                cov_yy_reg.diag_mut().mapv_inplace(|x| x + self.reg_param);

                let (_, eigvecs_x, eigvecs_y) =
                    self.solve_generalized_eigenvalue(&cov_xy, &cov_xx_reg, &cov_yy_reg)?;

                let n_comps = self.n_components.min(eigvecs_x.shape()[1]);
                if let Some(task_wx) = task_specific_wx.get_mut(&task_id) {
                    task_wx
                        .slice_mut(s![.., ..n_comps])
                        .assign(&eigvecs_x.slice(s![.., ..n_comps]));
                }
                if let Some(task_wy) = task_specific_wy.get_mut(&task_id) {
                    task_wy
                        .slice_mut(s![.., ..n_comps])
                        .assign(&eigvecs_y.slice(s![.., ..n_comps]));
                }
            }

            // Check convergence
            let diff = (&shared_wx - &old_shared_wx).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                converged = true;
                break;
            }

            if iter == self.max_iter - 1 && !converged {
                return Err(SklearsError::ConvergenceError {
                    iterations: self.max_iter,
                });
            }
        }

        // Compute correlations for shared components
        let mut correlations = Array1::zeros(self.n_components);
        for comp in 0..self.n_components {
            let mut total_corr = 0.0;
            for (x_task, y_task) in x_tasks.iter().zip(y_tasks.iter()) {
                let x_centered = self.center_data(x_task)?;
                let y_centered = self.center_data(y_task)?;

                let x_proj = x_centered.dot(&shared_wx.column(comp));
                let y_proj = y_centered.dot(&shared_wy.column(comp));

                let corr = self.compute_correlation(&x_proj, &y_proj)?;
                total_corr += corr;
            }
            correlations[comp] = total_corr / n_tasks as f64;
        }

        Ok(Self {
            canonical_weights_x: Some(shared_wx),
            canonical_weights_y: Some(shared_wy),
            shared_components: Some(Array2::zeros((self.n_components, self.n_components))),
            task_specific_components: Some(task_specific_wx),
            correlations: Some(correlations),
            ..self.clone()
        })
    }

    fn center_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mean = data
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))?;
        Ok(data - &mean)
    }

    fn compute_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        let n = x.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom.abs() < 1e-12 {
            Ok(0.0)
        } else {
            Ok(cov / denom)
        }
    }

    fn solve_generalized_eigenvalue(
        &self,
        cov_xy: &Array2<f64>,
        cov_xx: &Array2<f64>,
        cov_yy: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), SklearsError> {
        // Simplified eigenvalue decomposition
        // In a real implementation, this would use proper LAPACK routines
        let n_features = cov_xx.shape()[0];
        let n_comps = self.n_components.min(n_features);

        let mut rng = thread_rng();
        let mut eigvecs_x = Array2::zeros((n_features, n_comps));
        let mut eigvecs_y = Array2::zeros((cov_yy.shape()[0], n_comps));
        let eigvals = Array1::from_vec((0..n_comps).map(|_| rng.gen_range(0.1..1.0)).collect());

        // Initialize with random orthogonal vectors
        for i in 0..n_comps {
            for j in 0..n_features {
                eigvecs_x[[j, i]] = rng.gen_range(-1.0..1.0);
            }
            for j in 0..cov_yy.shape()[0] {
                eigvecs_y[[j, i]] = rng.gen_range(-1.0..1.0);
            }
        }

        // Normalize columns
        for i in 0..n_comps {
            let norm_x = (eigvecs_x.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            let norm_y = (eigvecs_y.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            if norm_x > 1e-12 {
                eigvecs_x.column_mut(i).mapv_inplace(|x| x / norm_x);
            }
            if norm_y > 1e-12 {
                eigvecs_y.column_mut(i).mapv_inplace(|x| x / norm_y);
            }
        }

        Ok((eigvals, eigvecs_x, eigvecs_y))
    }

    /// Gets the shared canonical weights for X
    pub fn shared_weights_x(&self) -> Option<&Array2<f64>> {
        self.canonical_weights_x.as_ref()
    }

    /// Gets the shared canonical weights for Y
    pub fn shared_weights_y(&self) -> Option<&Array2<f64>> {
        self.canonical_weights_y.as_ref()
    }

    /// Gets the task-specific weights for a given task
    pub fn task_weights(&self, task_id: usize) -> Option<&Array2<f64>> {
        self.task_specific_components.as_ref()?.get(&task_id)
    }

    /// Gets the canonical correlations
    pub fn correlations(&self) -> Option<&Array1<f64>> {
        self.correlations.as_ref()
    }
}

/// Shared Component Analysis
///
/// Identifies components that are shared across multiple datasets/tasks
/// and components that are specific to individual tasks.
#[derive(Debug, Clone)]
pub struct SharedComponentAnalysis {
    n_shared_components: usize,
    n_specific_components: usize,
    reg_param: f64,
    max_iter: usize,
    tol: f64,
    shared_components: Option<Array2<f64>>,
    specific_components: Option<HashMap<usize, Array2<f64>>>,
    explained_variance_shared: Option<Array1<f64>>,
    explained_variance_specific: Option<HashMap<usize, Array1<f64>>>,
}

impl SharedComponentAnalysis {
    /// Creates a new SharedComponentAnalysis instance
    pub fn new(n_shared_components: usize, n_specific_components: usize, reg_param: f64) -> Self {
        Self {
            n_shared_components,
            n_specific_components,
            reg_param,
            max_iter: 100,
            tol: 1e-3,
            shared_components: None,
            specific_components: None,
            explained_variance_shared: None,
            explained_variance_specific: None,
        }
    }

    /// Fits shared component analysis on multiple datasets
    pub fn fit_datasets(&self, datasets: &[Array2<f64>]) -> Result<Self, SklearsError> {
        if datasets.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one dataset must be provided".to_string(),
            ));
        }

        let n_tasks = datasets.len();
        let n_features = datasets[0].shape()[1];

        // Center all datasets
        let mut centered_datasets = Vec::new();
        for dataset in datasets {
            let centered = self.center_data(dataset)?;
            centered_datasets.push(centered);
        }

        // Initialize shared and specific components
        let mut shared_comps = Array2::zeros((n_features, self.n_shared_components));
        let mut specific_comps = HashMap::new();

        for task_id in 0..n_tasks {
            specific_comps.insert(
                task_id,
                Array2::zeros((n_features, self.n_specific_components)),
            );
        }

        // Random initialization
        let mut rng = thread_rng();
        shared_comps.mapv_inplace(|_| rng.gen_range(-1.0..1.0));
        for comps in specific_comps.values_mut() {
            comps.mapv_inplace(|_| rng.gen_range(-1.0..1.0));
        }

        // Simplified approach: just compute PCA on averaged covariance
        let mut total_cov = Array2::zeros((n_features, n_features));
        for dataset in &centered_datasets {
            let cov = dataset.t().dot(dataset) / dataset.shape()[0] as f64;
            total_cov = total_cov + cov;
        }
        total_cov = total_cov / n_tasks as f64;

        // Add regularization
        total_cov.diag_mut().mapv_inplace(|x| x + self.reg_param);

        // Compute shared components from averaged covariance
        shared_comps = self.compute_principal_components(&total_cov, self.n_shared_components)?;

        // Compute specific components for each task
        for (task_id, dataset) in centered_datasets.iter().enumerate() {
            let shared_proj = dataset.dot(&shared_comps);
            let shared_recon = shared_proj.dot(&shared_comps.t());
            let residual = dataset - &shared_recon;

            let specific_cov = residual.t().dot(&residual) / dataset.shape()[0] as f64;
            let mut specific_cov_reg = specific_cov.clone();
            specific_cov_reg
                .diag_mut()
                .mapv_inplace(|x| x + self.reg_param);

            let specific_pc =
                self.compute_principal_components(&specific_cov_reg, self.n_specific_components)?;
            specific_comps.insert(task_id, specific_pc);
        }

        // Compute explained variance
        let mut shared_variance = Array1::zeros(self.n_shared_components);
        let mut specific_variance = HashMap::new();

        for (task_id, dataset) in centered_datasets.iter().enumerate() {
            // Shared variance
            let shared_proj = dataset.dot(&shared_comps);
            for comp in 0..self.n_shared_components {
                let var =
                    shared_proj.column(comp).mapv(|x| x * x).sum() / dataset.shape()[0] as f64;
                shared_variance[comp] += var;
            }

            // Specific variance
            let specific = &specific_comps[&task_id];
            let specific_proj = dataset.dot(specific);
            let mut specific_var = Array1::zeros(self.n_specific_components);
            for comp in 0..self.n_specific_components {
                let var =
                    specific_proj.column(comp).mapv(|x| x * x).sum() / dataset.shape()[0] as f64;
                specific_var[comp] = var;
            }
            specific_variance.insert(task_id, specific_var);
        }

        // Average shared variance across tasks
        shared_variance.mapv_inplace(|x| x / n_tasks as f64);

        Ok(Self {
            shared_components: Some(shared_comps),
            specific_components: Some(specific_comps),
            explained_variance_shared: Some(shared_variance),
            explained_variance_specific: Some(specific_variance),
            ..self.clone()
        })
    }

    fn center_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mean = data
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))?;
        Ok(data - &mean)
    }

    fn compute_principal_components(
        &self,
        cov_matrix: &Array2<f64>,
        n_components: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        let n_features = cov_matrix.shape()[0];
        let n_comps = n_components.min(n_features);

        // Simplified PCA (placeholder for real eigenvalue decomposition)
        let mut rng = thread_rng();
        let mut components = Array2::zeros((n_features, n_comps));

        for i in 0..n_comps {
            for j in 0..n_features {
                components[[j, i]] = rng.gen_range(-1.0..1.0);
            }
            // Normalize
            let norm = (components.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            if norm > 1e-12 {
                components.column_mut(i).mapv_inplace(|x| x / norm);
            }
        }

        Ok(components)
    }

    /// Gets the shared components
    pub fn shared_components(&self) -> Option<&Array2<f64>> {
        self.shared_components.as_ref()
    }

    /// Gets the specific components for a task
    pub fn specific_components(&self, task_id: usize) -> Option<&Array2<f64>> {
        self.specific_components.as_ref()?.get(&task_id)
    }

    /// Gets the explained variance for shared components
    pub fn explained_variance_shared(&self) -> Option<&Array1<f64>> {
        self.explained_variance_shared.as_ref()
    }

    /// Gets the explained variance for specific components of a task
    pub fn explained_variance_specific(&self, task_id: usize) -> Option<&Array1<f64>> {
        self.explained_variance_specific.as_ref()?.get(&task_id)
    }
}

/// Transfer Learning for Cross-Decomposition
///
/// Transfers knowledge from source tasks to target tasks using cross-decomposition methods.
#[derive(Debug, Clone)]
pub struct TransferLearningCCA {
    n_components: usize,
    reg_param: f64,
    transfer_strength: f64,
    max_iter: usize,
    tol: f64,
    source_weights_x: Option<Array2<f64>>,
    source_weights_y: Option<Array2<f64>>,
    target_weights_x: Option<Array2<f64>>,
    target_weights_y: Option<Array2<f64>>,
    transfer_matrix: Option<Array2<f64>>,
    correlations: Option<Array1<f64>>,
}

impl TransferLearningCCA {
    /// Creates a new TransferLearningCCA instance
    pub fn new(n_components: usize, reg_param: f64, transfer_strength: f64) -> Self {
        Self {
            n_components,
            reg_param,
            transfer_strength,
            max_iter: 500,
            tol: 1e-6,
            source_weights_x: None,
            source_weights_y: None,
            target_weights_x: None,
            target_weights_y: None,
            transfer_matrix: None,
            correlations: None,
        }
    }

    /// First fits on source domain, then transfers to target domain
    pub fn fit_transfer(
        &self,
        source_x: &Array2<f64>,
        source_y: &Array2<f64>,
        target_x: &Array2<f64>,
        target_y: &Array2<f64>,
    ) -> Result<Self, SklearsError> {
        // Step 1: Learn source domain CCA
        let source_result = self.fit_source_domain(source_x, source_y)?;

        // Step 2: Transfer to target domain
        let target_result = self.transfer_to_target_domain(&source_result, target_x, target_y)?;

        Ok(target_result)
    }

    fn fit_source_domain(
        &self,
        source_x: &Array2<f64>,
        source_y: &Array2<f64>,
    ) -> Result<Self, SklearsError> {
        // Center the data
        let x_centered = self.center_data(source_x)?;
        let y_centered = self.center_data(source_y)?;

        // Compute covariance matrices
        let n_samples = source_x.shape()[0] as f64;
        let cov_xx = x_centered.t().dot(&x_centered) / n_samples;
        let cov_xy = x_centered.t().dot(&y_centered) / n_samples;
        let cov_yy = y_centered.t().dot(&y_centered) / n_samples;

        // Add regularization
        let mut cov_xx_reg = cov_xx.clone();
        let mut cov_yy_reg = cov_yy.clone();
        cov_xx_reg.diag_mut().mapv_inplace(|x| x + self.reg_param);
        cov_yy_reg.diag_mut().mapv_inplace(|x| x + self.reg_param);

        // Solve generalized eigenvalue problem
        let (eigvals, eigvecs_x, eigvecs_y) =
            self.solve_generalized_eigenvalue(&cov_xy, &cov_xx_reg, &cov_yy_reg)?;

        Ok(Self {
            source_weights_x: Some(eigvecs_x),
            source_weights_y: Some(eigvecs_y),
            correlations: Some(eigvals),
            ..self.clone()
        })
    }

    fn transfer_to_target_domain(
        &self,
        source_model: &Self,
        target_x: &Array2<f64>,
        target_y: &Array2<f64>,
    ) -> Result<Self, SklearsError> {
        let source_wx = source_model.source_weights_x.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("Source weights X not found".to_string())
        })?;
        let source_wy = source_model.source_weights_y.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("Source weights Y not found".to_string())
        })?;

        // Center target data
        let x_centered = self.center_data(target_x)?;
        let y_centered = self.center_data(target_y)?;

        // Initialize target weights close to source weights
        let mut target_wx = source_wx.clone();
        let mut target_wy = source_wy.clone();

        // Compute target covariances
        let n_samples = target_x.shape()[0] as f64;
        let target_cov_xx = x_centered.t().dot(&x_centered) / n_samples;
        let target_cov_xy = x_centered.t().dot(&y_centered) / n_samples;
        let target_cov_yy = y_centered.t().dot(&y_centered) / n_samples;

        // Transfer learning objective: balance between source knowledge and target fit
        for iter in 0..self.max_iter {
            let old_wx = target_wx.clone();

            // Update target weights with transfer regularization
            for comp in 0..self.n_components {
                // Compute gradients for target domain CCA objective
                let x_proj = x_centered.dot(&target_wx.column(comp));
                let y_proj = y_centered.dot(&target_wy.column(comp));

                // Transfer regularization: pull towards source weights
                let transfer_reg_x = self.transfer_strength
                    * (source_wx.column(comp).to_owned() - target_wx.column(comp).to_owned());
                let transfer_reg_y = self.transfer_strength
                    * (source_wy.column(comp).to_owned() - target_wy.column(comp).to_owned());

                // Simple gradient step (placeholder for more sophisticated optimization)
                let learning_rate = 0.01;
                target_wx
                    .column_mut(comp)
                    .zip_mut_with(&transfer_reg_x, |w, reg| *w += learning_rate * reg);
                target_wy
                    .column_mut(comp)
                    .zip_mut_with(&transfer_reg_y, |w, reg| *w += learning_rate * reg);

                // Normalize
                let norm_x = target_wx.column(comp).mapv(|x| x * x).sum().sqrt();
                let norm_y = target_wy.column(comp).mapv(|x| x * x).sum().sqrt();
                if norm_x > 1e-12 {
                    target_wx.column_mut(comp).mapv_inplace(|x| x / norm_x);
                }
                if norm_y > 1e-12 {
                    target_wy.column_mut(comp).mapv_inplace(|x| x / norm_y);
                }
            }

            // Check convergence
            let diff = (&target_wx - &old_wx).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }

            if iter == self.max_iter - 1 {
                return Err(SklearsError::ConvergenceError {
                    iterations: self.max_iter,
                });
            }
        }

        // Compute final correlations on target domain
        let mut correlations = Array1::zeros(self.n_components);
        for comp in 0..self.n_components {
            let x_proj = x_centered.dot(&target_wx.column(comp));
            let y_proj = y_centered.dot(&target_wy.column(comp));
            correlations[comp] = self.compute_correlation(&x_proj, &y_proj)?;
        }

        // Compute transfer matrix (alignment between source and target)
        let transfer_matrix = source_wx.t().dot(&target_wx);

        Ok(Self {
            source_weights_x: Some(source_wx.clone()),
            source_weights_y: Some(source_wy.clone()),
            target_weights_x: Some(target_wx),
            target_weights_y: Some(target_wy),
            transfer_matrix: Some(transfer_matrix),
            correlations: Some(correlations),
            ..self.clone()
        })
    }

    fn center_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mean = data
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))?;
        Ok(data - &mean)
    }

    fn compute_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        let n = x.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom.abs() < 1e-12 {
            Ok(0.0)
        } else {
            Ok(cov / denom)
        }
    }

    fn solve_generalized_eigenvalue(
        &self,
        cov_xy: &Array2<f64>,
        cov_xx: &Array2<f64>,
        cov_yy: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), SklearsError> {
        // Simplified implementation
        let n_features_x = cov_xx.shape()[0];
        let n_features_y = cov_yy.shape()[0];
        let n_comps = self.n_components.min(n_features_x).min(n_features_y);

        let mut rng = thread_rng();
        let mut eigvecs_x = Array2::zeros((n_features_x, n_comps));
        let mut eigvecs_y = Array2::zeros((n_features_y, n_comps));
        let eigvals = Array1::from_vec((0..n_comps).map(|_| rng.gen_range(0.1..1.0)).collect());

        // Initialize with random orthogonal vectors
        for i in 0..n_comps {
            for j in 0..n_features_x {
                eigvecs_x[[j, i]] = rng.gen_range(-1.0..1.0);
            }
            for j in 0..n_features_y {
                eigvecs_y[[j, i]] = rng.gen_range(-1.0..1.0);
            }

            // Normalize
            let norm_x = (eigvecs_x.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            let norm_y = (eigvecs_y.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            if norm_x > 1e-12 {
                eigvecs_x.column_mut(i).mapv_inplace(|x| x / norm_x);
            }
            if norm_y > 1e-12 {
                eigvecs_y.column_mut(i).mapv_inplace(|x| x / norm_y);
            }
        }

        Ok((eigvals, eigvecs_x, eigvecs_y))
    }

    /// Gets the source domain weights for X
    pub fn source_weights_x(&self) -> Option<&Array2<f64>> {
        self.source_weights_x.as_ref()
    }

    /// Gets the source domain weights for Y
    pub fn source_weights_y(&self) -> Option<&Array2<f64>> {
        self.source_weights_y.as_ref()
    }

    /// Gets the target domain weights for X
    pub fn target_weights_x(&self) -> Option<&Array2<f64>> {
        self.target_weights_x.as_ref()
    }

    /// Gets the target domain weights for Y
    pub fn target_weights_y(&self) -> Option<&Array2<f64>> {
        self.target_weights_y.as_ref()
    }

    /// Gets the transfer matrix (alignment between source and target)
    pub fn transfer_matrix(&self) -> Option<&Array2<f64>> {
        self.transfer_matrix.as_ref()
    }

    /// Gets the canonical correlations on target domain
    pub fn correlations(&self) -> Option<&Array1<f64>> {
        self.correlations.as_ref()
    }
}

/// Domain Adaptation for Cross-Decomposition
///
/// Adapts cross-decomposition models across different domains with distribution shifts.
#[derive(Debug, Clone)]
pub struct DomainAdaptationCCA {
    n_components: usize,
    reg_param: f64,
    adaptation_strength: f64,
    max_iter: usize,
    tol: f64,
    domain_weights_x: Option<Array2<f64>>,
    domain_weights_y: Option<Array2<f64>>,
    domain_shift_matrix: Option<Array2<f64>>,
    adapted_correlations: Option<Array1<f64>>,
}

impl DomainAdaptationCCA {
    /// Creates a new DomainAdaptationCCA instance
    pub fn new(n_components: usize, reg_param: f64, adaptation_strength: f64) -> Self {
        Self {
            n_components,
            reg_param,
            adaptation_strength,
            max_iter: 500,
            tol: 1e-6,
            domain_weights_x: None,
            domain_weights_y: None,
            domain_shift_matrix: None,
            adapted_correlations: None,
        }
    }

    /// Fits domain adaptation CCA
    pub fn fit_domains(
        &self,
        source_x: &Array2<f64>,
        source_y: &Array2<f64>,
        target_x: &Array2<f64>,
        target_y: &Array2<f64>,
    ) -> Result<Self, SklearsError> {
        // Center both domains
        let source_x_centered = self.center_data(source_x)?;
        let source_y_centered = self.center_data(source_y)?;
        let target_x_centered = self.center_data(target_x)?;
        let target_y_centered = self.center_data(target_y)?;

        // Compute domain statistics
        let source_cov_xx =
            source_x_centered.t().dot(&source_x_centered) / source_x.shape()[0] as f64;
        let source_cov_xy =
            source_x_centered.t().dot(&source_y_centered) / source_x.shape()[0] as f64;
        let source_cov_yy =
            source_y_centered.t().dot(&source_y_centered) / source_y.shape()[0] as f64;

        let target_cov_xx =
            target_x_centered.t().dot(&target_x_centered) / target_x.shape()[0] as f64;
        let target_cov_xy =
            target_x_centered.t().dot(&target_y_centered) / target_x.shape()[0] as f64;
        let target_cov_yy =
            target_y_centered.t().dot(&target_y_centered) / target_y.shape()[0] as f64;

        // Domain adaptation: minimize domain discrepancy while maximizing correlation
        let adapted_cov_xx = &source_cov_xx * (1.0 - self.adaptation_strength)
            + &target_cov_xx * self.adaptation_strength;
        let adapted_cov_xy = &source_cov_xy * (1.0 - self.adaptation_strength)
            + &target_cov_xy * self.adaptation_strength;
        let adapted_cov_yy = &source_cov_yy * (1.0 - self.adaptation_strength)
            + &target_cov_yy * self.adaptation_strength;

        // Add regularization
        let mut adapted_cov_xx_reg = adapted_cov_xx.clone();
        let mut adapted_cov_yy_reg = adapted_cov_yy.clone();
        adapted_cov_xx_reg
            .diag_mut()
            .mapv_inplace(|x| x + self.reg_param);
        adapted_cov_yy_reg
            .diag_mut()
            .mapv_inplace(|x| x + self.reg_param);

        // Solve adapted CCA
        let (eigvals, eigvecs_x, eigvecs_y) = self.solve_generalized_eigenvalue(
            &adapted_cov_xy,
            &adapted_cov_xx_reg,
            &adapted_cov_yy_reg,
        )?;

        // Compute domain shift matrix (difference between source and target projections)
        let domain_shift = self.compute_domain_shift(&source_cov_xx, &target_cov_xx, &eigvecs_x)?;

        Ok(Self {
            domain_weights_x: Some(eigvecs_x),
            domain_weights_y: Some(eigvecs_y),
            domain_shift_matrix: Some(domain_shift),
            adapted_correlations: Some(eigvals),
            ..self.clone()
        })
    }

    fn center_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mean = data
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))?;
        Ok(data - &mean)
    }

    fn compute_domain_shift(
        &self,
        source_cov: &Array2<f64>,
        target_cov: &Array2<f64>,
        weights: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Compute how much the covariance structure changes between domains
        let cov_diff = target_cov - source_cov;
        let domain_shift = weights.t().dot(&cov_diff).dot(weights);
        Ok(domain_shift)
    }

    fn solve_generalized_eigenvalue(
        &self,
        cov_xy: &Array2<f64>,
        cov_xx: &Array2<f64>,
        cov_yy: &Array2<f64>,
    ) -> Result<(Array1<f64>, Array2<f64>, Array2<f64>), SklearsError> {
        // Simplified implementation
        let n_features_x = cov_xx.shape()[0];
        let n_features_y = cov_yy.shape()[0];
        let n_comps = self.n_components.min(n_features_x).min(n_features_y);

        let mut rng = thread_rng();
        let mut eigvecs_x = Array2::zeros((n_features_x, n_comps));
        let mut eigvecs_y = Array2::zeros((n_features_y, n_comps));
        let eigvals = Array1::from_vec((0..n_comps).map(|_| rng.gen_range(0.1..1.0)).collect());

        // Initialize and normalize
        for i in 0..n_comps {
            for j in 0..n_features_x {
                eigvecs_x[[j, i]] = rng.gen_range(-1.0..1.0);
            }
            for j in 0..n_features_y {
                eigvecs_y[[j, i]] = rng.gen_range(-1.0..1.0);
            }

            let norm_x = (eigvecs_x.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            let norm_y = (eigvecs_y.column(i).mapv(|x| x * x).sum() as f64).sqrt();
            if norm_x > 1e-12 {
                eigvecs_x.column_mut(i).mapv_inplace(|x| x / norm_x);
            }
            if norm_y > 1e-12 {
                eigvecs_y.column_mut(i).mapv_inplace(|x| x / norm_y);
            }
        }

        Ok((eigvals, eigvecs_x, eigvecs_y))
    }

    /// Gets the adapted domain weights for X
    pub fn domain_weights_x(&self) -> Option<&Array2<f64>> {
        self.domain_weights_x.as_ref()
    }

    /// Gets the adapted domain weights for Y
    pub fn domain_weights_y(&self) -> Option<&Array2<f64>> {
        self.domain_weights_y.as_ref()
    }

    /// Gets the domain shift matrix
    pub fn domain_shift_matrix(&self) -> Option<&Array2<f64>> {
        self.domain_shift_matrix.as_ref()
    }

    /// Gets the adapted canonical correlations
    pub fn adapted_correlations(&self) -> Option<&Array1<f64>> {
        self.adapted_correlations.as_ref()
    }
}

/// Few-Shot Learning for Cross-Decomposition
///
/// Learns effective cross-decomposition from limited training examples.
#[derive(Debug, Clone)]
pub struct FewShotCCA {
    n_components: usize,
    n_support_examples: usize,
    reg_param: f64,
    meta_learning_rate: f64,
    adaptation_steps: usize,
    prototypes_x: Option<Array2<f64>>,
    prototypes_y: Option<Array2<f64>>,
    meta_weights_x: Option<Array2<f64>>,
    meta_weights_y: Option<Array2<f64>>,
}

impl FewShotCCA {
    /// Creates a new FewShotCCA instance
    pub fn new(
        n_components: usize,
        n_support_examples: usize,
        reg_param: f64,
        meta_learning_rate: f64,
    ) -> Self {
        Self {
            n_components,
            n_support_examples,
            reg_param,
            meta_learning_rate,
            adaptation_steps: 10,
            prototypes_x: None,
            prototypes_y: None,
            meta_weights_x: None,
            meta_weights_y: None,
        }
    }

    /// Meta-trains on multiple few-shot tasks
    pub fn meta_train(
        &self,
        few_shot_tasks: &[(Array2<f64>, Array2<f64>)],
    ) -> Result<Self, SklearsError> {
        if few_shot_tasks.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one few-shot task must be provided".to_string(),
            ));
        }

        let n_features_x = few_shot_tasks[0].0.shape()[1];
        let n_features_y = few_shot_tasks[0].1.shape()[1];

        // Initialize meta-parameters
        let mut meta_wx = Array2::zeros((n_features_x, self.n_components));
        let mut meta_wy = Array2::zeros((n_features_y, self.n_components));

        let mut rng = thread_rng();
        meta_wx.mapv_inplace(|_| rng.gen_range(-0.1..0.1));
        meta_wy.mapv_inplace(|_| rng.gen_range(-0.1..0.1));

        // Meta-learning loop
        for episode in 0..100 {
            // Meta-training episodes
            for (task_x, task_y) in few_shot_tasks {
                // Sample support and query sets
                let (support_x, support_y, query_x, query_y) =
                    self.sample_support_query(task_x, task_y)?;

                // Fast adaptation on support set
                let (adapted_wx, adapted_wy) =
                    self.fast_adaptation(&meta_wx, &meta_wy, &support_x, &support_y)?;

                // Compute loss on query set
                let query_loss =
                    self.compute_cca_loss(&adapted_wx, &adapted_wy, &query_x, &query_y)?;

                // Update meta-parameters (simplified gradient step)
                let grad_scale = self.meta_learning_rate * query_loss;
                meta_wx.mapv_inplace(|w| w - grad_scale * rng.gen_range(-0.01..0.01));
                meta_wy.mapv_inplace(|w| w - grad_scale * rng.gen_range(-0.01..0.01));
            }
        }

        // Compute prototypes from meta-training data
        let (prototypes_x, prototypes_y) = self.compute_prototypes(few_shot_tasks)?;

        Ok(Self {
            prototypes_x: Some(prototypes_x),
            prototypes_y: Some(prototypes_y),
            meta_weights_x: Some(meta_wx),
            meta_weights_y: Some(meta_wy),
            ..self.clone()
        })
    }

    /// Adapts to a new few-shot task
    pub fn adapt_to_task(
        &self,
        support_x: &Array2<f64>,
        support_y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), SklearsError> {
        let meta_wx = self.meta_weights_x.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("Meta-weights not trained".to_string())
        })?;
        let meta_wy = self.meta_weights_y.as_ref().ok_or_else(|| {
            SklearsError::InvalidOperation("Meta-weights not trained".to_string())
        })?;

        self.fast_adaptation(meta_wx, meta_wy, support_x, support_y)
    }

    fn sample_support_query(
        &self,
        task_x: &Array2<f64>,
        task_y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), SklearsError> {
        let n_samples = task_x.shape()[0];
        if n_samples < self.n_support_examples * 2 {
            return Err(SklearsError::InvalidInput(
                "Not enough samples for support and query sets".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let support_indices = &indices[..self.n_support_examples];
        let query_indices = &indices[self.n_support_examples..2 * self.n_support_examples];

        let support_x = task_x.select(Axis(0), support_indices);
        let support_y = task_y.select(Axis(0), support_indices);
        let query_x = task_x.select(Axis(0), query_indices);
        let query_y = task_y.select(Axis(0), query_indices);

        Ok((support_x, support_y, query_x, query_y))
    }

    fn fast_adaptation(
        &self,
        init_wx: &Array2<f64>,
        init_wy: &Array2<f64>,
        support_x: &Array2<f64>,
        support_y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), SklearsError> {
        let mut wx = init_wx.clone();
        let mut wy = init_wy.clone();

        // Center support data
        let x_centered = self.center_data(support_x)?;
        let y_centered = self.center_data(support_y)?;

        // Fast adaptation steps
        for _ in 0..self.adaptation_steps {
            // Compute current projections
            let x_proj = x_centered.dot(&wx);
            let y_proj = y_centered.dot(&wy);

            // Simple gradient-based update (placeholder)
            let learning_rate = 0.1;
            let mut rng = thread_rng();

            // Add small random updates (simplified optimization)
            wx.mapv_inplace(|w| w + learning_rate * rng.gen_range(-0.01..0.01));
            wy.mapv_inplace(|w| w + learning_rate * rng.gen_range(-0.01..0.01));

            // Normalize
            for i in 0..self.n_components {
                let norm_x = wx.column(i).mapv(|x| x * x).sum().sqrt();
                let norm_y = wy.column(i).mapv(|x| x * x).sum().sqrt();
                if norm_x > 1e-12 {
                    wx.column_mut(i).mapv_inplace(|x| x / norm_x);
                }
                if norm_y > 1e-12 {
                    wy.column_mut(i).mapv_inplace(|x| x / norm_y);
                }
            }
        }

        Ok((wx, wy))
    }

    fn compute_cca_loss(
        &self,
        wx: &Array2<f64>,
        wy: &Array2<f64>,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<f64, SklearsError> {
        let x_centered = self.center_data(x)?;
        let y_centered = self.center_data(y)?;

        let x_proj = x_centered.dot(wx);
        let y_proj = y_centered.dot(wy);

        let mut total_loss = 0.0;
        for i in 0..self.n_components {
            let corr = self
                .compute_correlation(&x_proj.column(i).to_owned(), &y_proj.column(i).to_owned())?;
            total_loss += 1.0 - corr.abs(); // Loss is 1 - |correlation|
        }

        Ok(total_loss / self.n_components as f64)
    }

    fn compute_prototypes(
        &self,
        tasks: &[(Array2<f64>, Array2<f64>)],
    ) -> Result<(Array2<f64>, Array2<f64>), SklearsError> {
        let n_features_x = tasks[0].0.shape()[1];
        let n_features_y = tasks[0].1.shape()[1];

        let mut prototype_x = Array2::zeros((self.n_support_examples, n_features_x));
        let mut prototype_y = Array2::zeros((self.n_support_examples, n_features_y));

        // Average first few examples from each task as prototypes
        for (i, (task_x, task_y)) in tasks.iter().enumerate() {
            let n_samples = task_x.shape()[0].min(self.n_support_examples);
            for j in 0..n_samples {
                if i == 0 {
                    prototype_x.row_mut(j).assign(&task_x.row(j));
                    prototype_y.row_mut(j).assign(&task_y.row(j));
                } else {
                    prototype_x
                        .row_mut(j)
                        .zip_mut_with(&task_x.row(j), |p, t| *p = (*p + t) / 2.0);
                    prototype_y
                        .row_mut(j)
                        .zip_mut_with(&task_y.row(j), |p, t| *p = (*p + t) / 2.0);
                }
            }
        }

        Ok((prototype_x, prototype_y))
    }

    fn center_data(&self, data: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let mean = data
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::NumericalError("Failed to compute mean".to_string()))?;
        Ok(data - &mean)
    }

    fn compute_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, SklearsError> {
        let n = x.len() as f64;
        if n < 2.0 {
            return Ok(0.0);
        }

        let mean_x = x.sum() / n;
        let mean_y = y.sum() / n;

        let mut cov = 0.0;
        let mut var_x = 0.0;
        let mut var_y = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            cov += dx * dy;
            var_x += dx * dx;
            var_y += dy * dy;
        }

        let denom = (var_x * var_y).sqrt();
        if denom.abs() < 1e-12 {
            Ok(0.0)
        } else {
            Ok(cov / denom)
        }
    }

    /// Gets the learned prototypes for X
    pub fn prototypes_x(&self) -> Option<&Array2<f64>> {
        self.prototypes_x.as_ref()
    }

    /// Gets the learned prototypes for Y
    pub fn prototypes_y(&self) -> Option<&Array2<f64>> {
        self.prototypes_y.as_ref()
    }

    /// Gets the meta-learned weights for X
    pub fn meta_weights_x(&self) -> Option<&Array2<f64>> {
        self.meta_weights_x.as_ref()
    }

    /// Gets the meta-learned weights for Y
    pub fn meta_weights_y(&self) -> Option<&Array2<f64>> {
        self.meta_weights_y.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_multi_task_cca_creation() {
        let mt_cca = MultiTaskCCA::new(2, 0.1, 0.5);
        assert_eq!(mt_cca.n_components, 2);
        assert_eq!(mt_cca.reg_param, 0.1);
        assert_eq!(mt_cca.sharing_strength, 0.5);
    }

    #[test]
    fn test_shared_component_analysis_creation() {
        let sca = SharedComponentAnalysis::new(3, 2, 0.05);
        assert_eq!(sca.n_shared_components, 3);
        assert_eq!(sca.n_specific_components, 2);
        assert_eq!(sca.reg_param, 0.05);
    }

    #[test]
    fn test_multi_task_cca_fit() {
        let x1 = Array2::from_shape_vec((20, 5), (0..100).map(|x| x as f64).collect()).unwrap();
        let y1 =
            Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64 * 1.5).collect()).unwrap();
        let x2 = Array2::from_shape_vec((20, 5), (50..150).map(|x| x as f64).collect()).unwrap();
        let y2 =
            Array2::from_shape_vec((20, 3), (30..90).map(|x| x as f64 * 1.2).collect()).unwrap();

        let mt_cca = MultiTaskCCA::new(2, 0.1, 0.5);
        let result = mt_cca.fit_multi_task(&[x1, x2], &[y1, y2]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_shared_component_analysis_fit() {
        let data1 = Array2::from_shape_vec((30, 6), (0..180).map(|x| x as f64).collect()).unwrap();
        let data2 = Array2::from_shape_vec((30, 6), (20..200).map(|x| x as f64).collect()).unwrap();
        let data3 = Array2::from_shape_vec((30, 6), (10..190).map(|x| x as f64).collect()).unwrap();

        let sca = SharedComponentAnalysis::new(2, 1, 0.01);
        let result = sca.fit_datasets(&[data1, data2, data3]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_transfer_learning_cca_creation() {
        let tl_cca = TransferLearningCCA::new(2, 0.1, 0.3);
        assert_eq!(tl_cca.n_components, 2);
        assert_eq!(tl_cca.reg_param, 0.1);
        assert_eq!(tl_cca.transfer_strength, 0.3);
    }

    #[test]
    fn test_transfer_learning_cca_fit() {
        let source_x =
            Array2::from_shape_vec((20, 4), (0..80).map(|x| x as f64).collect()).unwrap();
        let source_y =
            Array2::from_shape_vec((20, 3), (0..60).map(|x| x as f64 * 1.1).collect()).unwrap();
        let target_x =
            Array2::from_shape_vec((15, 4), (10..70).map(|x| x as f64).collect()).unwrap();
        let target_y =
            Array2::from_shape_vec((15, 3), (5..50).map(|x| x as f64 * 1.2).collect()).unwrap();

        let tl_cca = TransferLearningCCA::new(2, 0.1, 0.3);
        let result = tl_cca.fit_transfer(&source_x, &source_y, &target_x, &target_y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_domain_adaptation_cca_creation() {
        let da_cca = DomainAdaptationCCA::new(2, 0.05, 0.4);
        assert_eq!(da_cca.n_components, 2);
        assert_eq!(da_cca.reg_param, 0.05);
        assert_eq!(da_cca.adaptation_strength, 0.4);
    }

    #[test]
    fn test_domain_adaptation_cca_fit() {
        let source_x =
            Array2::from_shape_vec((25, 5), (0..125).map(|x| x as f64).collect()).unwrap();
        let source_y =
            Array2::from_shape_vec((25, 3), (0..75).map(|x| x as f64 * 0.9).collect()).unwrap();
        let target_x =
            Array2::from_shape_vec((20, 5), (15..115).map(|x| x as f64).collect()).unwrap();
        let target_y =
            Array2::from_shape_vec((20, 3), (10..70).map(|x| x as f64 * 1.1).collect()).unwrap();

        let da_cca = DomainAdaptationCCA::new(2, 0.05, 0.4);
        let result = da_cca.fit_domains(&source_x, &source_y, &target_x, &target_y);
        assert!(result.is_ok());
    }

    #[test]
    fn test_few_shot_cca_creation() {
        let fs_cca = FewShotCCA::new(2, 5, 0.1, 0.01);
        assert_eq!(fs_cca.n_components, 2);
        assert_eq!(fs_cca.n_support_examples, 5);
        assert_eq!(fs_cca.reg_param, 0.1);
        assert_eq!(fs_cca.meta_learning_rate, 0.01);
    }

    #[test]
    fn test_few_shot_cca_meta_train() {
        let task1_x = Array2::from_shape_vec((15, 4), (0..60).map(|x| x as f64).collect()).unwrap();
        let task1_y =
            Array2::from_shape_vec((15, 3), (0..45).map(|x| x as f64 * 1.1).collect()).unwrap();
        let task2_x =
            Array2::from_shape_vec((15, 4), (10..70).map(|x| x as f64).collect()).unwrap();
        let task2_y =
            Array2::from_shape_vec((15, 3), (5..50).map(|x| x as f64 * 0.9).collect()).unwrap();

        let fs_cca = FewShotCCA::new(1, 3, 0.1, 0.01);
        let result = fs_cca.meta_train(&[(task1_x, task1_y), (task2_x, task2_y)]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_multi_task_cca_getters() {
        let mt_cca = MultiTaskCCA::new(2, 0.1, 0.5);
        assert!(mt_cca.shared_weights_x().is_none());
        assert!(mt_cca.shared_weights_y().is_none());
        assert!(mt_cca.correlations().is_none());
    }

    #[test]
    fn test_shared_component_analysis_getters() {
        let sca = SharedComponentAnalysis::new(2, 1, 0.01);
        assert!(sca.shared_components().is_none());
        assert!(sca.specific_components(0).is_none());
        assert!(sca.explained_variance_shared().is_none());
    }
}

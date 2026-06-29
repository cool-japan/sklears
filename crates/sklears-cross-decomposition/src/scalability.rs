//! Scalability and performance optimizations for cross-decomposition
//!
//! This module implements memory-efficient and distributed versions of
//! cross-decomposition algorithms designed for large-scale data processing
//! with bounded memory usage and parallel computation support.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_linalg::compat::{eigh, svd, UPLO};
use scirs2_linalg::LinalgError;
use sklears_core::error::SklearsError;
use std::sync::Arc;
use std::thread;

/// Memory-efficient CCA using stochastic optimization
///
/// Implements CCA using stochastic gradient descent and other memory-efficient
/// algorithms that can handle datasets larger than available memory through
/// out-of-core processing and incremental updates.
///
/// # Mathematical Background
///
/// Uses stochastic approximation to the CCA objective:
/// - Mini-batch updates to canonical weights
/// - Adaptive learning rates and momentum
/// - Regularization for numerical stability
/// - Optional dimensionality reduction
///
/// # Examples
///
/// ```rust
/// use sklears_cross_decomposition::MemoryEfficientCCA;
/// use scirs2_core::ndarray::array;
///
/// let mut cca = MemoryEfficientCCA::new(1)
///     .batch_size(32)
///     .learning_rate(0.01)
///     .max_iter(1000);
///
/// // Process data in chunks
/// let x_batch = array![[1.0, 2.0], [2.0, 3.0]];
/// let y_batch = array![[1.5, 2.5], [2.5, 3.5]];
/// cca.partial_fit(&x_batch, &y_batch).expect("operation should succeed");
///
/// let correlations = cca.canonical_correlations();
/// ```
#[derive(Debug, Clone)]
pub struct MemoryEfficientCCA {
    n_components: usize,
    batch_size: usize,
    learning_rate: f64,
    momentum: f64,
    regularization: f64,
    max_iter: usize,
    tolerance: f64,
    random_state: Option<u64>,

    // Internal state
    wx: Option<Array2<f64>>,
    wy: Option<Array2<f64>>,
    wx_momentum: Option<Array2<f64>>,
    wy_momentum: Option<Array2<f64>>,
    running_mean_x: Option<Array1<f64>>,
    running_mean_y: Option<Array1<f64>>,
    running_var_x: Option<Array1<f64>>,
    running_var_y: Option<Array1<f64>>,
    /// Exponentially-smoothed canonical correlations measured on the most recent
    /// batches using the current weights (real estimate, not a constant).
    running_correlations: Option<Array1<f64>>,
    n_samples_seen: usize,
    iteration: usize,
}

impl MemoryEfficientCCA {
    /// Create a new MemoryEfficientCCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            batch_size: 32,
            learning_rate: 0.01,
            momentum: 0.9,
            regularization: 1e-6,
            max_iter: 1000,
            tolerance: 1e-8,
            random_state: None,
            wx: None,
            wy: None,
            wx_momentum: None,
            wy_momentum: None,
            running_mean_x: None,
            running_mean_y: None,
            running_var_x: None,
            running_var_y: None,
            running_correlations: None,
            n_samples_seen: 0,
            iteration: 0,
        }
    }

    /// Set the mini-batch size
    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set momentum parameter
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Incrementally fit the model with a batch of data
    pub fn partial_fit(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), SklearsError> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Empty input".to_string()));
        }

        // Initialize parameters if this is the first batch
        if self.wx.is_none() {
            self.initialize_parameters(x.ncols(), y.ncols())?;
        }

        // Update running statistics
        self.update_running_statistics(x, y);

        // Normalize data using running statistics
        let x_norm = self.normalize_x(x)?;
        let y_norm = self.normalize_y(y)?;

        // Process in mini-batches
        let n_samples = x_norm.nrows();
        let mut batch_start = 0;

        while batch_start < n_samples {
            let batch_end = (batch_start + self.batch_size).min(n_samples);
            let x_batch = x_norm.slice(s![batch_start..batch_end, ..]);
            let y_batch = y_norm.slice(s![batch_start..batch_end, ..]);

            self.sgd_step(&x_batch.to_owned(), &y_batch.to_owned())?;

            batch_start = batch_end;
            self.iteration += 1;

            if self.iteration >= self.max_iter {
                break;
            }
        }

        self.n_samples_seen += n_samples;

        // Measure the real per-component canonical correlations achieved by the
        // current weights on this (normalized) batch and fold them into an
        // exponentially-smoothed running estimate.
        self.update_running_correlations(&x_norm, &y_norm);
        Ok(())
    }

    /// Update the running canonical-correlation estimate from the latest batch.
    fn update_running_correlations(&mut self, x_norm: &Array2<f64>, y_norm: &Array2<f64>) {
        let (wx, wy) = match (&self.wx, &self.wy) {
            (Some(wx), Some(wy)) => (wx, wy),
            _ => return,
        };
        if x_norm.nrows() < 2 {
            return;
        }
        let u = x_norm.dot(wx);
        let v = y_norm.dot(wy);

        let batch_correlations: Array1<f64> = (0..self.n_components)
            .map(|k| Self::compute_correlation_static(&u.column(k), &v.column(k)).abs())
            .collect();

        self.running_correlations = Some(match self.running_correlations.take() {
            Some(prev) => {
                let alpha = 0.1;
                &prev * (1.0 - alpha) + &batch_correlations * alpha
            }
            None => batch_correlations,
        });
    }

    fn initialize_parameters(&mut self, p_x: usize, p_y: usize) -> Result<(), SklearsError> {
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            let mut entropy_rng = thread_rng();
            StdRng::from_rng(&mut entropy_rng)
        };

        use scirs2_core::random::{Distribution, RandNormal as Normal, SeedableRng};

        let normal = Normal::new(0.0, 0.1)
            .map_err(|e| SklearsError::InvalidInput(format!("invalid Normal params: {}", e)))?;

        let mut wx = Array2::zeros((p_x, self.n_components));
        let mut wy = Array2::zeros((p_y, self.n_components));

        for i in 0..p_x {
            for j in 0..self.n_components {
                wx[[i, j]] = normal.sample(&mut rng);
            }
        }

        for i in 0..p_y {
            for j in 0..self.n_components {
                wy[[i, j]] = normal.sample(&mut rng);
            }
        }

        self.wx = Some(wx);
        self.wy = Some(wy);
        self.wx_momentum = Some(Array2::zeros((p_x, self.n_components)));
        self.wy_momentum = Some(Array2::zeros((p_y, self.n_components)));
        self.running_mean_x = Some(Array1::zeros(p_x));
        self.running_mean_y = Some(Array1::zeros(p_y));
        self.running_var_x = Some(Array1::ones(p_x));
        self.running_var_y = Some(Array1::ones(p_y));

        Ok(())
    }

    fn update_running_statistics(&mut self, x: &Array2<f64>, y: &Array2<f64>) {
        let alpha = if self.n_samples_seen == 0 { 1.0 } else { 0.01 };

        let batch_mean_x = x
            .mean_axis(Axis(0))
            .expect("mean_axis requires non-empty array");
        let batch_mean_y = y
            .mean_axis(Axis(0))
            .expect("mean_axis requires non-empty array");
        let batch_var_x = x.var_axis(Axis(0), 1.0);
        let batch_var_y = y.var_axis(Axis(0), 1.0);

        if let (
            Some(ref mut mean_x),
            Some(ref mut mean_y),
            Some(ref mut var_x),
            Some(ref mut var_y),
        ) = (
            &mut self.running_mean_x,
            &mut self.running_mean_y,
            &mut self.running_var_x,
            &mut self.running_var_y,
        ) {
            *mean_x = &*mean_x * (1.0 - alpha) + &batch_mean_x * alpha;
            *mean_y = &*mean_y * (1.0 - alpha) + &batch_mean_y * alpha;
            *var_x = &*var_x * (1.0 - alpha) + &batch_var_x * alpha;
            *var_y = &*var_y * (1.0 - alpha) + &batch_var_y * alpha;
        }
    }

    fn normalize_x(&self, x: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        if let (Some(ref mean_x), Some(ref var_x)) = (&self.running_mean_x, &self.running_var_x) {
            let std_x = var_x.mapv(|v| v.sqrt().max(1e-8));
            Ok((x - &mean_x.clone().insert_axis(Axis(0))) / &std_x.clone().insert_axis(Axis(0)))
        } else {
            Ok(x.clone())
        }
    }

    fn normalize_y(&self, y: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        if let (Some(ref mean_y), Some(ref var_y)) = (&self.running_mean_y, &self.running_var_y) {
            let std_y = var_y.mapv(|v| v.sqrt().max(1e-8));
            Ok((y - &mean_y.clone().insert_axis(Axis(0))) / &std_y.clone().insert_axis(Axis(0)))
        } else {
            Ok(y.clone())
        }
    }

    fn sgd_step(&mut self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), SklearsError> {
        let batch_size = x.nrows() as f64;
        let lr = self.learning_rate / (1.0 + 0.0001 * self.iteration as f64); // Adaptive learning rate
        let momentum = self.momentum;
        let regularization = self.regularization;
        let n_components = self.n_components;

        if let (Some(ref mut wx), Some(ref mut wy), Some(ref mut wx_mom), Some(ref mut wy_mom)) = (
            &mut self.wx,
            &mut self.wy,
            &mut self.wx_momentum,
            &mut self.wy_momentum,
        ) {
            // Compute canonical variables
            let u = x.dot(wx);
            let v = y.dot(wy);

            // Compute gradients using simplified CCA objective
            for k in 0..n_components {
                let u_k = u.column(k);
                let v_k = v.column(k);

                // Compute correlation and gradients
                let correlation = Self::compute_correlation_static(&u_k, &v_k);

                // Gradients for canonical correlation maximization
                let grad_wx_k =
                    x.t().dot(&(v_k.to_owned() - u_k.to_owned() * correlation)) / batch_size;
                let grad_wy_k =
                    y.t().dot(&(u_k.to_owned() - v_k.to_owned() * correlation)) / batch_size;

                // Add regularization
                let reg_grad_wx = wx.column(k).to_owned() * regularization;
                let reg_grad_wy = wy.column(k).to_owned() * regularization;

                let total_grad_wx = grad_wx_k - reg_grad_wx;
                let total_grad_wy = grad_wy_k - reg_grad_wy;

                // Update momentum
                let mut wx_mom_k = wx_mom.column_mut(k);
                let mut wy_mom_k = wy_mom.column_mut(k);
                wx_mom_k *= momentum;
                wy_mom_k *= momentum;
                wx_mom_k += &(total_grad_wx * lr);
                wy_mom_k += &(total_grad_wy * lr);

                // Update weights
                let mut wx_k = wx.column_mut(k);
                let mut wy_k = wy.column_mut(k);
                wx_k += &wx_mom_k;
                wy_k += &wy_mom_k;
            }

            // Orthogonalize weights (Gram-Schmidt)
            Self::orthogonalize_weights_static(wx, wy, n_components);
        }

        Ok(())
    }

    fn compute_correlation_static(
        u: &scirs2_core::ndarray::ArrayView1<f64>,
        v: &scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let n = u.len() as f64;
        let mean_u = u.sum() / n;
        let mean_v = v.sum() / n;

        let mut num = 0.0;
        let mut den_u = 0.0;
        let mut den_v = 0.0;

        for i in 0..u.len() {
            let du = u[i] - mean_u;
            let dv = v[i] - mean_v;
            num += du * dv;
            den_u += du * du;
            den_v += dv * dv;
        }

        if den_u == 0.0 || den_v == 0.0 {
            0.0
        } else {
            num / (den_u * den_v).sqrt()
        }
    }

    fn orthogonalize_weights_static(
        wx: &mut Array2<f64>,
        wy: &mut Array2<f64>,
        n_components: usize,
    ) {
        // Simple Gram-Schmidt orthogonalization
        for i in 0..n_components {
            // Normalize current vectors
            let wx_norm = wx.column(i).dot(&wx.column(i)).sqrt();
            let wy_norm = wy.column(i).dot(&wy.column(i)).sqrt();

            if wx_norm > 1e-10 {
                let mut wx_i = wx.column_mut(i);
                wx_i /= wx_norm;
            }

            if wy_norm > 1e-10 {
                let mut wy_i = wy.column_mut(i);
                wy_i /= wy_norm;
            }

            // Orthogonalize against previous components
            for j in 0..i {
                let proj_x = wx.column(j).dot(&wx.column(i));
                let proj_y = wy.column(j).dot(&wy.column(i));

                let wx_j = wx.column(j).to_owned();
                let wy_j = wy.column(j).to_owned();
                let mut wx_i = wx.column_mut(i);
                let mut wy_i = wy.column_mut(i);
                wx_i -= &(wx_j * proj_x);
                wy_i -= &(wy_j * proj_y);
            }
        }
    }

    /// Get current canonical correlations.
    ///
    /// Returns the exponentially-smoothed canonical correlations measured on the
    /// batches seen so far using the current weights (a real running estimate),
    /// or `None` if the model has not been fitted yet.
    pub fn canonical_correlations(&self) -> Option<Array1<f64>> {
        self.running_correlations.clone()
    }

    /// Get current canonical weights
    pub fn canonical_weights(&self) -> Option<(Array2<f64>, Array2<f64>)> {
        if let (Some(ref wx), Some(ref wy)) = (&self.wx, &self.wy) {
            Some((wx.clone(), wy.clone()))
        } else {
            None
        }
    }

    /// Transform new data using current weights
    pub fn transform(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), SklearsError> {
        if let (Some(ref wx), Some(ref wy)) = (&self.wx, &self.wy) {
            let x_norm = self.normalize_x(x)?;
            let y_norm = self.normalize_y(y)?;

            Ok((x_norm.dot(wx), y_norm.dot(wy)))
        } else {
            Err(SklearsError::InvalidInput("Model not fitted".to_string()))
        }
    }

    /// Get number of samples processed
    pub fn n_samples_seen(&self) -> usize {
        self.n_samples_seen
    }

    /// Get current iteration count
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Reset the model state
    pub fn reset(&mut self) {
        self.wx = None;
        self.wy = None;
        self.wx_momentum = None;
        self.wy_momentum = None;
        self.running_mean_x = None;
        self.running_mean_y = None;
        self.running_var_x = None;
        self.running_var_y = None;
        self.running_correlations = None;
        self.n_samples_seen = 0;
        self.iteration = 0;
    }
}

/// Distributed CCA using parallel computation
///
/// Implements distributed canonical correlation analysis that can split
/// computation across multiple threads or processes for large-scale data.
/// Uses data parallelism and model averaging techniques.
#[derive(Debug, Clone)]
pub struct DistributedCCA {
    n_components: usize,
    n_workers: usize,
    regularization: f64,
    aggregation_strategy: AggregationStrategy,
}

/// Strategy for aggregating results from distributed workers
#[derive(Debug, Clone)]
pub enum AggregationStrategy {
    /// Simple averaging of canonical weights
    Average,
    /// Weighted averaging based on data size
    WeightedAverage,
    /// Use the best performing worker's results
    BestWorker,
}

impl DistributedCCA {
    /// Create a new DistributedCCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            n_workers: num_cpus::get(),
            regularization: 1e-6,
            aggregation_strategy: AggregationStrategy::Average,
        }
    }

    /// Set the number of parallel workers
    pub fn n_workers(mut self, n_workers: usize) -> Self {
        self.n_workers = n_workers;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set aggregation strategy
    pub fn aggregation_strategy(mut self, strategy: AggregationStrategy) -> Self {
        self.aggregation_strategy = strategy;
        self
    }

    /// Fit the distributed CCA model
    pub fn fit(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<DistributedCCAResults, SklearsError> {
        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        let n_samples = x.nrows();
        if n_samples < self.n_workers {
            return Err(SklearsError::InvalidInput(
                "Need at least as many samples as workers".to_string(),
            ));
        }

        // Split data across workers
        let chunk_size = n_samples / self.n_workers;
        let mut worker_results = Vec::new();
        let mut handles = Vec::new();

        let x_arc = Arc::new(x.clone());
        let y_arc = Arc::new(y.clone());

        // Launch workers
        for worker_id in 0..self.n_workers {
            let start_idx = worker_id * chunk_size;
            let end_idx = if worker_id == self.n_workers - 1 {
                n_samples
            } else {
                (worker_id + 1) * chunk_size
            };

            let x_worker = x_arc.clone();
            let y_worker = y_arc.clone();
            let n_components = self.n_components;
            let regularization = self.regularization;

            let handle = thread::spawn(move || {
                let x_chunk = x_worker.slice(s![start_idx..end_idx, ..]).to_owned();
                let y_chunk = y_worker.slice(s![start_idx..end_idx, ..]).to_owned();

                // Fit CCA on this chunk
                Self::fit_worker_cca(&x_chunk, &y_chunk, n_components, regularization)
            });

            handles.push(handle);
        }

        // Collect results from workers
        for handle in handles {
            match handle.join() {
                Ok(Ok(result)) => worker_results.push(result),
                Ok(Err(e)) => return Err(e),
                Err(_) => {
                    return Err(SklearsError::InvalidInput(
                        "Worker thread panicked".to_string(),
                    ))
                }
            }
        }

        // Aggregate results
        let aggregated = self.aggregate_results(&worker_results)?;

        Ok(DistributedCCAResults {
            wx: aggregated.0,
            wy: aggregated.1,
            correlations: aggregated.2,
            n_workers: self.n_workers,
            worker_results,
        })
    }

    /// Fit canonical correlation analysis on a single worker's data shard.
    ///
    /// Solves the standard CCA generalized-eigenproblem via the symmetric
    /// whitening / SVD formulation. Given centered `X` (n x p_x) and
    /// `Y` (n x p_y):
    ///
    /// * `Cxx = Xᵀ X / (n-1) + reg·I`
    /// * `Cyy = Yᵀ Y / (n-1) + reg·I`
    /// * `Cxy = Xᵀ Y / (n-1)`
    /// * `M = Cxx^(-1/2) · Cxy · Cyy^(-1/2)`, then `SVD(M) = U S Vᵀ`.
    ///
    /// The canonical correlations are the singular values `S` (clamped to
    /// `[0, 1]`); the x-weights are `Cxx^(-1/2) · U[:, :n_comp]` and the
    /// y-weights are `Cyy^(-1/2) · V[:, :n_comp]` with `V = Vt.t()`.
    fn fit_worker_cca(
        x: &Array2<f64>,
        y: &Array2<f64>,
        n_components: usize,
        regularization: f64,
    ) -> Result<WorkerCCAResult, SklearsError> {
        let n_samples = x.nrows();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "fit_worker_cca requires at least two samples to estimate covariance".to_string(),
            ));
        }
        let n = n_samples as f64;

        // Center the data.
        let x_mean = x
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("X shard is empty".to_string()))?;
        let y_mean = y
            .mean_axis(Axis(0))
            .ok_or_else(|| SklearsError::InvalidInput("Y shard is empty".to_string()))?;
        let x_centered = x - &x_mean.insert_axis(Axis(0));
        let y_centered = y - &y_mean.insert_axis(Axis(0));

        let p_x = x.ncols();
        let p_y = y.ncols();

        // Regularized auto-covariance and cross-covariance matrices.
        let cxx = &(x_centered.t().dot(&x_centered) / (n - 1.0))
            + &(Array2::<f64>::eye(p_x) * regularization);
        let cyy = &(y_centered.t().dot(&y_centered) / (n - 1.0))
            + &(Array2::<f64>::eye(p_y) * regularization);
        let cxy = x_centered.t().dot(&y_centered) / (n - 1.0);

        // Symmetric inverse square roots used to whiten each view.
        let cxx_inv_sqrt = Self::symmetric_inverse_sqrt(&cxx)?;
        let cyy_inv_sqrt = Self::symmetric_inverse_sqrt(&cyy)?;

        // Whitened cross-covariance: M = Cxx^(-1/2) · Cxy · Cyy^(-1/2).
        let whitened = cxx_inv_sqrt.dot(&cxy).dot(&cyy_inv_sqrt);

        // SVD of the whitened cross-covariance yields canonical directions.
        let (u, s, vt) = svd(&whitened, true)
            .map_err(|e: LinalgError| SklearsError::InvalidInput(e.to_string()))?;
        let v = vt.t();

        let n_comp = n_components.min(p_x).min(p_y).min(s.len());
        if n_comp == 0 {
            return Err(SklearsError::InvalidInput(
                "No canonical components could be extracted (zero-width input)".to_string(),
            ));
        }

        // Canonical correlations are the leading singular values, clamped to
        // the theoretically valid range [0, 1].
        let correlations = s.slice(s![..n_comp]).mapv(|sv| sv.clamp(0.0, 1.0));

        // Map whitened singular vectors back to the original feature spaces.
        let u_top = u.slice(s![.., ..n_comp]);
        let v_top = v.slice(s![.., ..n_comp]);
        let wx = cxx_inv_sqrt.dot(&u_top);
        let wy = cyy_inv_sqrt.dot(&v_top);

        Ok(WorkerCCAResult {
            wx,
            wy,
            correlations,
            n_samples,
        })
    }

    /// Compute the symmetric inverse square root `C^(-1/2)` of a symmetric
    /// positive semi-definite matrix via its eigendecomposition.
    ///
    /// With `eigh(C) = (w, Q)` (eigenvalues `w`, eigenvectors as columns of
    /// `Q`), `C^(-1/2) = Q · diag(1 / sqrt(max(w, eps))) · Qᵀ`. A small `eps`
    /// floor guards against division by zero for (near-)singular directions;
    /// the caller's regularization keeps conditioning reasonable.
    fn symmetric_inverse_sqrt(matrix: &Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        const EPS: f64 = 1e-10;
        let (eigenvalues, eigenvectors) = eigh(matrix, UPLO::Upper)
            .map_err(|e: LinalgError| SklearsError::InvalidInput(e.to_string()))?;

        let inv_sqrt_eigenvalues = eigenvalues.mapv(|w| 1.0 / w.max(EPS).sqrt());

        // Scale each eigenvector column by its inverse-sqrt eigenvalue, then
        // reassemble: Q · diag(d) · Qᵀ.
        let scaled = &eigenvectors * &inv_sqrt_eigenvalues.view().insert_axis(Axis(0));
        Ok(scaled.dot(&eigenvectors.t()))
    }

    #[allow(clippy::type_complexity)] // returns (x_weights, y_weights, correlations) triple
    fn aggregate_results(
        &self,
        results: &[WorkerCCAResult],
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>), SklearsError> {
        if results.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No worker results to aggregate".to_string(),
            ));
        }

        let n_features_x = results[0].wx.nrows();
        let n_features_y = results[0].wy.nrows();

        match self.aggregation_strategy {
            AggregationStrategy::Average => {
                let mut wx_sum = Array2::zeros((n_features_x, self.n_components));
                let mut wy_sum = Array2::zeros((n_features_y, self.n_components));
                let mut corr_sum = Array1::zeros(self.n_components);

                for result in results {
                    wx_sum += &result.wx;
                    wy_sum += &result.wy;
                    corr_sum += &result.correlations;
                }

                let n_workers = results.len() as f64;
                Ok((wx_sum / n_workers, wy_sum / n_workers, corr_sum / n_workers))
            }

            AggregationStrategy::WeightedAverage => {
                let total_samples: usize = results.iter().map(|r| r.n_samples).sum();
                let mut wx_weighted = Array2::zeros((n_features_x, self.n_components));
                let mut wy_weighted = Array2::zeros((n_features_y, self.n_components));
                let mut corr_weighted = Array1::zeros(self.n_components);

                for result in results {
                    let weight = result.n_samples as f64 / total_samples as f64;
                    wx_weighted += &(&result.wx * weight);
                    wy_weighted += &(&result.wy * weight);
                    corr_weighted += &(&result.correlations * weight);
                }

                Ok((wx_weighted, wy_weighted, corr_weighted))
            }

            AggregationStrategy::BestWorker => {
                // Find worker with highest average correlation
                let mut best_idx = 0;
                let mut best_avg_corr = 0.0;

                for (i, result) in results.iter().enumerate() {
                    let avg_corr = result.correlations.sum() / result.correlations.len() as f64;
                    if avg_corr > best_avg_corr {
                        best_avg_corr = avg_corr;
                        best_idx = i;
                    }
                }

                let best_result = &results[best_idx];
                Ok((
                    best_result.wx.clone(),
                    best_result.wy.clone(),
                    best_result.correlations.clone(),
                ))
            }
        }
    }
}

/// Results from a single worker in distributed CCA
#[derive(Debug, Clone)]
struct WorkerCCAResult {
    wx: Array2<f64>,
    wy: Array2<f64>,
    correlations: Array1<f64>,
    n_samples: usize,
}

/// Results from distributed CCA analysis
#[derive(Debug, Clone)]
pub struct DistributedCCAResults {
    wx: Array2<f64>,
    wy: Array2<f64>,
    correlations: Array1<f64>,
    n_workers: usize,
    worker_results: Vec<WorkerCCAResult>,
}

impl DistributedCCAResults {
    /// Get aggregated canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<f64> {
        &self.correlations
    }

    /// Get aggregated canonical weights
    pub fn canonical_weights(&self) -> (&Array2<f64>, &Array2<f64>) {
        (&self.wx, &self.wy)
    }

    /// Get number of workers used
    pub fn n_workers(&self) -> usize {
        self.n_workers
    }

    /// Get variance in results across workers
    pub fn worker_variance(&self) -> (Array2<f64>, Array2<f64>, Array1<f64>) {
        let n_workers = self.worker_results.len() as f64;
        let n_features_x = self.wx.nrows();
        let n_features_y = self.wy.nrows();
        let n_components = self.correlations.len();

        let mut wx_var = Array2::zeros((n_features_x, n_components));
        let mut wy_var = Array2::zeros((n_features_y, n_components));
        let mut corr_var = Array1::zeros(n_components);

        // Compute variance across workers
        for i in 0..n_features_x {
            for j in 0..n_components {
                let values: Vec<f64> = self.worker_results.iter().map(|r| r.wx[[i, j]]).collect();
                let mean = values.iter().sum::<f64>() / n_workers;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_workers;
                wx_var[[i, j]] = variance;
            }
        }

        for i in 0..n_features_y {
            for j in 0..n_components {
                let values: Vec<f64> = self.worker_results.iter().map(|r| r.wy[[i, j]]).collect();
                let mean = values.iter().sum::<f64>() / n_workers;
                let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_workers;
                wy_var[[i, j]] = variance;
            }
        }

        for j in 0..n_components {
            let values: Vec<f64> = self
                .worker_results
                .iter()
                .map(|r| r.correlations[j])
                .collect();
            let mean = values.iter().sum::<f64>() / n_workers;
            let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_workers;
            corr_var[j] = variance;
        }

        (wx_var, wy_var, corr_var)
    }

    /// Transform data using aggregated weights
    pub fn transform(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), SklearsError> {
        Ok((x.dot(&self.wx), y.dot(&self.wy)))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_memory_efficient_cca_basic() {
        let mut cca = MemoryEfficientCCA::new(1)
            .batch_size(2)
            .learning_rate(0.1)
            .max_iter(50)
            .random_state(42);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5], [4.5, 5.5]];

        cca.partial_fit(&x, &y).expect("operation should succeed");

        assert_eq!(cca.n_samples_seen(), 4);
        assert!(cca.iteration() > 0);

        let correlations = cca.canonical_correlations();
        assert!(correlations.is_some());
        assert_eq!(correlations.expect("operation should succeed").len(), 1);

        let weights = cca.canonical_weights();
        assert!(weights.is_some());
        let (wx, wy) = weights.expect("operation should succeed");
        assert_eq!(wx.shape(), &[2, 1]);
        assert_eq!(wy.shape(), &[2, 1]);
    }

    #[test]
    fn test_memory_efficient_cca_multiple_batches() {
        let mut cca = MemoryEfficientCCA::new(1)
            .batch_size(2)
            .learning_rate(0.05)
            .random_state(42);

        // First batch
        let x1 = array![[1.0, 2.0], [2.0, 3.0]];
        let y1 = array![[1.5, 2.5], [2.5, 3.5]];
        cca.partial_fit(&x1, &y1).expect("operation should succeed");

        // Second batch
        let x2 = array![[3.0, 4.0], [4.0, 5.0]];
        let y2 = array![[3.5, 4.5], [4.5, 5.5]];
        cca.partial_fit(&x2, &y2).expect("operation should succeed");

        assert_eq!(cca.n_samples_seen(), 4);

        let (u, v) = cca.transform(&x1, &y1).expect("transform should succeed");
        assert_eq!(u.shape(), &[2, 1]);
        assert_eq!(v.shape(), &[2, 1]);
    }

    #[test]
    fn test_memory_efficient_cca_real_correlations() {
        // Strongly correlated streaming data: the reported canonical correlation
        // must be a real value in [0, 1] (never the old hardcoded 0.8), and the
        // running estimate should track the genuine high correlation.
        let mut cca = MemoryEfficientCCA::new(1)
            .batch_size(16)
            .learning_rate(0.05)
            .max_iter(2000)
            .random_state(7);

        let n = 60usize;
        let mut x = Array2::<f64>::zeros((n, 2));
        let mut y = Array2::<f64>::zeros((n, 2));
        for i in 0..n {
            let t = i as f64 * 0.2;
            x[[i, 0]] = t.sin();
            x[[i, 1]] = (0.5 * t).cos();
            // Y is an almost-deterministic linear image of X.
            y[[i, 0]] = 0.9 * x[[i, 0]] + 0.1 * x[[i, 1]];
            y[[i, 1]] = -0.2 * x[[i, 0]] + 0.8 * x[[i, 1]];
        }

        for _ in 0..30 {
            cca.partial_fit(&x, &y).expect("partial_fit should succeed");
        }

        let correlations = cca
            .canonical_correlations()
            .expect("correlations should exist after fitting");
        assert_eq!(correlations.len(), 1);
        let c = correlations[0];
        assert!((0.0..=1.0).contains(&c), "correlation out of range: {c}");
        // A near-deterministic relationship -> strong recovered correlation.
        assert!(c > 0.5, "expected high canonical correlation, got {c}");
    }

    #[test]
    fn test_memory_efficient_cca_reset() {
        let mut cca = MemoryEfficientCCA::new(1);

        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5]];
        cca.partial_fit(&x, &y).expect("operation should succeed");

        assert_eq!(cca.n_samples_seen(), 2);

        cca.reset();
        assert_eq!(cca.n_samples_seen(), 0);
        assert_eq!(cca.iteration(), 0);
        assert!(cca.canonical_correlations().is_none());
    }

    #[test]
    fn test_distributed_cca_basic() {
        let x = array![
            [1.0, 2.0, 1.5],
            [2.0, 3.0, 2.5],
            [3.0, 4.0, 3.5],
            [4.0, 5.0, 4.5],
            [5.0, 6.0, 5.5],
            [6.0, 7.0, 6.5]
        ];

        let y = array![
            [1.2, 1.8, 1.1],
            [2.2, 2.8, 2.1],
            [3.2, 3.8, 3.1],
            [4.2, 4.8, 4.1],
            [5.2, 5.8, 5.1],
            [6.2, 6.8, 6.1]
        ];

        let dcca = DistributedCCA::new(2)
            .n_workers(2)
            .aggregation_strategy(AggregationStrategy::Average);

        let result = dcca.fit(&x, &y).expect("fit should succeed");

        assert_eq!(result.n_workers(), 2);

        let correlations = result.canonical_correlations();
        assert_eq!(correlations.len(), 2);

        let (wx, wy) = result.canonical_weights();
        assert_eq!(wx.shape(), &[3, 2]);
        assert_eq!(wy.shape(), &[3, 2]);
    }

    #[test]
    fn test_distributed_cca_aggregation_strategies() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];

        let y = array![
            [1.2, 1.8],
            [2.2, 2.8],
            [3.2, 3.8],
            [4.2, 4.8],
            [5.2, 5.8],
            [6.2, 6.8]
        ];

        // Test different aggregation strategies
        let strategies = vec![
            AggregationStrategy::Average,
            AggregationStrategy::WeightedAverage,
            AggregationStrategy::BestWorker,
        ];

        for strategy in strategies {
            let dcca = DistributedCCA::new(1)
                .n_workers(2)
                .aggregation_strategy(strategy);

            let result = dcca.fit(&x, &y).expect("fit should succeed");

            let correlations = result.canonical_correlations();
            assert_eq!(correlations.len(), 1);
            assert!(correlations[0] >= 0.0);
        }
    }

    #[test]
    fn test_distributed_cca_transform() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.2, 1.8], [2.2, 2.8], [3.2, 3.8], [4.2, 4.8]];

        let dcca = DistributedCCA::new(1).n_workers(2);
        let result = dcca.fit(&x, &y).expect("fit should succeed");

        let (u, v) = result.transform(&x, &y).expect("transform should succeed");
        assert_eq!(u.shape(), &[4, 1]);
        assert_eq!(v.shape(), &[4, 1]);
    }

    #[test]
    fn test_distributed_cca_worker_variance() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let y = array![[1.2, 1.8], [2.2, 2.8], [3.2, 3.8], [4.2, 4.8]];

        let dcca = DistributedCCA::new(1).n_workers(2);
        let result = dcca.fit(&x, &y).expect("fit should succeed");

        let (wx_var, wy_var, corr_var) = result.worker_variance();
        assert_eq!(wx_var.shape(), &[2, 1]);
        assert_eq!(wy_var.shape(), &[2, 1]);
        assert_eq!(corr_var.len(), 1);

        // Variance should be non-negative
        assert!(wx_var.iter().all(|&x| x >= 0.0));
        assert!(wy_var.iter().all(|&x| x >= 0.0));
        assert!(corr_var.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_memory_efficient_cca_error_cases() {
        let mut cca = MemoryEfficientCCA::new(1);

        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];

        // Mismatched sample sizes
        let result = cca.partial_fit(&x, &y);
        assert!(result.is_err());

        // Empty input
        let x_empty = Array2::zeros((0, 2));
        let y_empty = Array2::zeros((0, 2));
        let result2 = cca.partial_fit(&x_empty, &y_empty);
        assert!(result2.is_err());

        // Transform before fitting
        let result3 = cca.transform(&x, &x);
        assert!(result3.is_err());
    }

    #[test]
    fn test_distributed_cca_error_cases() {
        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![[1.5, 2.5], [2.5, 3.5], [3.5, 4.5]];

        // Mismatched sample sizes
        let dcca = DistributedCCA::new(1);
        let result = dcca.fit(&x, &y);
        assert!(result.is_err());

        // Too few samples for workers
        let x_small = array![[1.0, 2.0]];
        let y_small = array![[1.5, 2.5]];
        let dcca_many_workers = DistributedCCA::new(1).n_workers(5);
        let result2 = dcca_many_workers.fit(&x_small, &y_small);
        assert!(result2.is_err());
    }

    #[test]
    fn test_fit_worker_cca_recovers_linear_relationship() {
        // Build X with two informative, near-uncorrelated features and define
        // Y as a known linear function of X plus a tiny deterministic
        // perturbation. A correctly implemented CCA must recover a leading
        // canonical correlation very close to 1.0 for such data.
        let n_samples = 60usize;
        let mut x = Array2::<f64>::zeros((n_samples, 2));
        for i in 0..n_samples {
            let t = i as f64;
            // Two distinct, well-conditioned directions.
            x[[i, 0]] = (t * 0.37).sin() * 3.0 + t * 0.05;
            x[[i, 1]] = (t * 0.21).cos() * 2.0 - t * 0.03;
        }

        // Mixing matrix B (p_x x p_y) defines Y = X · B (+ tiny noise).
        let b = array![[1.5, -0.7, 0.2], [0.4, 1.1, -1.3]];
        let mut y = x.dot(&b);
        // Deterministic small perturbation so the relationship is strong but
        // not perfectly degenerate.
        for i in 0..n_samples {
            for j in 0..y.ncols() {
                y[[i, j]] += (i as f64 * 0.13 + j as f64).sin() * 1e-3;
            }
        }

        let n_components = 2usize;
        let result = DistributedCCA::fit_worker_cca(&x, &y, n_components, 1e-8)
            .expect("fit_worker_cca should succeed on well-conditioned data");

        // Shapes: wx is (p_x, n_comp), wy is (p_y, n_comp), correlations len n_comp.
        let n_comp = n_components.min(x.ncols()).min(y.ncols());
        assert_eq!(result.wx.shape(), &[x.ncols(), n_comp]);
        assert_eq!(result.wy.shape(), &[y.ncols(), n_comp]);
        assert_eq!(result.correlations.len(), n_comp);

        // Canonical correlations are in [0, 1].
        assert!(result
            .correlations
            .iter()
            .all(|&c| (0.0..=1.0).contains(&c)));

        // The top canonical correlation must be near 1.0 because Y is an
        // (almost) exact linear image of X.
        let top = result.correlations.iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            top > 0.9,
            "expected top canonical correlation > 0.9, got {top}"
        );

        // Y is a rank-2 linear image of X's two-dimensional column space, so
        // BOTH canonical correlations must be strong. Check order-independently
        // via the minimum extracted correlation.
        let weakest = result.correlations.iter().cloned().fold(f64::MAX, f64::min);
        assert!(
            weakest > 0.9,
            "expected every canonical correlation > 0.9, got min {weakest}"
        );
    }
}

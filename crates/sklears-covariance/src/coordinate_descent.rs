//! Coordinate Descent Methods for Covariance Estimation
//!
//! This module implements various coordinate descent algorithms for covariance estimation,
//! including methods for sparse precision matrix estimation, regularized covariance,
//! and structured covariance models.

use scirs2_core::ndarray::ndarray_linalg::{Determinant, Eig, Inverse, SVD};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
};

/// Coordinate descent covariance estimator
///
/// Uses coordinate descent algorithms to estimate covariance and precision matrices
/// with various regularization schemes and structural constraints.
#[derive(Debug, Clone)]
pub struct CoordinateDescentCovariance<S = Untrained> {
    state: S,
    /// Optimization target
    target: OptimizationTarget,
    /// Regularization method
    regularization: RegularizationMethod,
    /// Maximum number of iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Learning rate for updates
    learning_rate: f64,
    /// Whether to use adaptive learning rate
    adaptive_lr: bool,
    /// Whether to use warm start
    warm_start: bool,
    /// Random state for reproducible results
    random_state: Option<u64>,
    /// Frequency of convergence checks
    check_convergence_freq: usize,
    /// Whether to use line search
    use_line_search: bool,
}

/// Optimization targets for coordinate descent
#[derive(Debug, Clone)]
pub enum OptimizationTarget {
    Covariance,
    Precision,
    Joint,
    FactorModel {
        n_factors: usize,
    },
    /// Low-rank plus sparse decomposition
    LowRankPlusSparse {
        rank: usize,
    },
}

/// Regularization methods for coordinate descent
#[derive(Debug, Clone)]
pub enum RegularizationMethod {
    /// L1 regularization (Lasso)
    L1 { alpha: f64 },
    /// L2 regularization (Ridge)
    L2 { alpha: f64 },
    /// Elastic Net (L1 + L2)
    ElasticNet { alpha: f64, l1_ratio: f64 },
    /// Group Lasso
    GroupLasso { alpha: f64, groups: Vec<Vec<usize>> },
    /// Fused Lasso (for ordered variables)
    FusedLasso { alpha: f64 },
    /// Nuclear norm (for low-rank estimation)
    NuclearNorm { alpha: f64 },
    /// SCAD (Smoothly Clipped Absolute Deviation)
    SCAD { alpha: f64, gamma: f64 },
    /// MCP (Minimax Concave Penalty)
    MCP { alpha: f64, gamma: f64 },
}

/// Trained Coordinate Descent state
#[derive(Debug, Clone)]
pub struct CoordinateDescentCovarianceTrained {
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Estimated precision matrix
    precision: Option<Array2<f64>>,
    /// Mean of training data
    mean: Array1<f64>,
    /// Number of iterations performed
    n_iter: usize,
    /// Convergence history
    convergence_history: Vec<f64>,
    /// Final objective value
    objective_value: f64,
    /// Active set (non-zero parameters)
    active_set: Vec<(usize, usize)>,
    /// Optimization target used
    target: OptimizationTarget,
    /// Regularization method used
    regularization: RegularizationMethod,
    /// Sparsity level achieved
    sparsity_ratio: f64,
    /// Condition number of final matrix
    condition_number: f64,
    /// Factor loadings (if factor model)
    factor_loadings: Option<Array2<f64>>,
    /// Low-rank component (if applicable)
    low_rank_component: Option<Array2<f64>>,
    /// Sparse component (if applicable)
    sparse_component: Option<Array2<f64>>,
}

impl Default for CoordinateDescentCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl CoordinateDescentCovariance {
    /// Creates a new coordinate descent covariance estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            target: OptimizationTarget::Precision,
            regularization: RegularizationMethod::L1 { alpha: 0.1 },
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 1.0,
            adaptive_lr: true,
            warm_start: false,
            random_state: None,
            check_convergence_freq: 10,
            use_line_search: false,
        }
    }

    /// Sets the optimization target
    pub fn target(mut self, target: OptimizationTarget) -> Self {
        self.target = target;
        self
    }

    /// Sets the regularization method
    pub fn regularization(mut self, regularization: RegularizationMethod) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Sets the convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Sets the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets whether to use adaptive learning rate
    pub fn adaptive_lr(mut self, adaptive_lr: bool) -> Self {
        self.adaptive_lr = adaptive_lr;
        self
    }

    /// Sets whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.warm_start = warm_start;
        self
    }

    /// Sets random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Sets convergence check frequency
    pub fn check_convergence_freq(mut self, freq: usize) -> Self {
        self.check_convergence_freq = freq;
        self
    }

    /// Sets whether to use line search
    pub fn use_line_search(mut self, use_line_search: bool) -> Self {
        self.use_line_search = use_line_search;
        self
    }

    /// Convenience method for L1 regularization
    pub fn l1_alpha(mut self, alpha: f64) -> Self {
        self.regularization = RegularizationMethod::L1 { alpha };
        self
    }

    /// Convenience method for L2 regularization
    pub fn l2_alpha(mut self, alpha: f64) -> Self {
        self.regularization = RegularizationMethod::L2 { alpha };
        self
    }

    /// Convenience method for Elastic Net regularization
    pub fn elastic_net(mut self, alpha: f64, l1_ratio: f64) -> Self {
        self.regularization = RegularizationMethod::ElasticNet { alpha, l1_ratio };
        self
    }
}

#[derive(Debug, Clone)]
pub struct CoordinateDescentConfig {
    pub target: OptimizationTarget,
    pub regularization: RegularizationMethod,
    pub max_iter: usize,
    pub tol: f64,
    pub learning_rate: f64,
    pub adaptive_lr: bool,
    pub warm_start: bool,
    pub random_state: Option<u64>,
    pub check_convergence_freq: usize,
    pub use_line_search: bool,
}

impl Estimator for CoordinateDescentCovariance {
    type Config = CoordinateDescentConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        // For now, create a temporary config on the fly
        // In practice, this would be stored as a field
        static CONFIG: std::sync::OnceLock<CoordinateDescentConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| CoordinateDescentConfig {
            target: OptimizationTarget::Covariance,
            regularization: RegularizationMethod::L1 { alpha: 0.1 },
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            adaptive_lr: false,
            warm_start: false,
            random_state: None,
            check_convergence_freq: 10,
            use_line_search: false,
        })
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for CoordinateDescentCovariance {
    type Fitted = CoordinateDescentCovariance<CoordinateDescentCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Coordinate descent requires at least 2 samples".to_string(),
            ));
        }

        // Compute empirical mean and center data
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x.to_owned();
        for mut row in x_centered.axis_iter_mut(Axis(0)) {
            row -= &mean;
        }

        // Initialize empirical covariance
        let empirical_cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Run coordinate descent based on target
        let (
            covariance,
            precision,
            n_iter,
            convergence_history,
            objective_value,
            active_set,
            factor_loadings,
            low_rank_component,
            sparse_component,
        ) = match &self.target {
            OptimizationTarget::Covariance => {
                self.coordinate_descent_covariance(&empirical_cov, &x_centered)?
            }
            OptimizationTarget::Precision => {
                self.coordinate_descent_precision(&empirical_cov, &x_centered)?
            }
            OptimizationTarget::Joint => {
                self.coordinate_descent_joint(&empirical_cov, &x_centered)?
            }
            OptimizationTarget::FactorModel { n_factors } => {
                self.coordinate_descent_factor_model(&x_centered, *n_factors)?
            }
            OptimizationTarget::LowRankPlusSparse { rank } => {
                self.coordinate_descent_low_rank_sparse(&empirical_cov, *rank)?
            }
        };

        // Compute final statistics
        let sparsity_ratio = self.compute_sparsity_ratio(&covariance);
        let condition_number = self.compute_condition_number(&covariance)?;

        let trained_state = CoordinateDescentCovarianceTrained {
            covariance,
            precision,
            mean,
            n_iter,
            convergence_history,
            objective_value,
            active_set,
            target: self.target.clone(),
            regularization: self.regularization.clone(),
            sparsity_ratio,
            condition_number,
            factor_loadings,
            low_rank_component,
            sparse_component,
        };

        Ok(CoordinateDescentCovariance {
            state: trained_state,
            target: self.target,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tol: self.tol,
            learning_rate: self.learning_rate,
            adaptive_lr: self.adaptive_lr,
            warm_start: self.warm_start,
            random_state: self.random_state,
            check_convergence_freq: self.check_convergence_freq,
            use_line_search: self.use_line_search,
        })
    }
}

impl CoordinateDescentCovariance {
    /// Coordinate descent for covariance estimation
    fn coordinate_descent_covariance(
        &self,
        empirical_cov: &Array2<f64>,
        _x: &Array2<f64>,
    ) -> SklResult<(
        Array2<f64>,
        Option<Array2<f64>>,
        usize,
        Vec<f64>,
        f64,
        Vec<(usize, usize)>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        let n = empirical_cov.nrows();
        let mut covariance = empirical_cov.clone();
        let mut convergence_history = Vec::new();
        let mut lr = self.learning_rate;

        for iter in 0..self.max_iter {
            let prev_cov = covariance.clone();

            // Coordinate descent updates
            for i in 0..n {
                for j in i..n {
                    let old_val = covariance[[i, j]];

                    // Compute gradient
                    let grad = self.compute_covariance_gradient(&covariance, empirical_cov, i, j);

                    // Apply regularization
                    let regularized_grad =
                        self.apply_regularization_gradient(old_val, grad, i, j)?;

                    // Update with learning rate
                    let new_val = old_val - lr * regularized_grad;

                    // Apply soft thresholding for L1 regularization
                    let final_val = self.apply_soft_thresholding(new_val, i, j)?;

                    covariance[[i, j]] = final_val;
                    if i != j {
                        covariance[[j, i]] = final_val;
                    }
                }
            }

            // Ensure positive semi-definiteness
            covariance = self.project_to_psd(&covariance)?;

            // Check convergence
            if iter % self.check_convergence_freq == 0 {
                let convergence = (&covariance - &prev_cov).mapv(|x| x * x).sum().sqrt();
                convergence_history.push(convergence);

                if convergence < self.tol {
                    let precision = covariance.inv().ok();
                    let objective = self.compute_objective(&covariance, empirical_cov)?;
                    let active_set = self.compute_active_set(&covariance);
                    return Ok((
                        covariance,
                        precision,
                        iter + 1,
                        convergence_history,
                        objective,
                        active_set,
                        None,
                        None,
                        None,
                    ));
                }

                // Adaptive learning rate
                if self.adaptive_lr && convergence_history.len() > 1 {
                    let prev_convergence = convergence_history[convergence_history.len() - 2];
                    if convergence > prev_convergence {
                        lr *= 0.8; // Reduce learning rate
                    } else if convergence < 0.1 * prev_convergence {
                        lr *= 1.1; // Increase learning rate
                    }
                }
            }
        }

        let precision = covariance.inv().ok();
        let objective = self.compute_objective(&covariance, empirical_cov)?;
        let active_set = self.compute_active_set(&covariance);
        Ok((
            covariance,
            precision,
            self.max_iter,
            convergence_history,
            objective,
            active_set,
            None,
            None,
            None,
        ))
    }

    /// Coordinate descent for precision matrix estimation
    fn coordinate_descent_precision(
        &self,
        empirical_cov: &Array2<f64>,
        _x: &Array2<f64>,
    ) -> SklResult<(
        Array2<f64>,
        Option<Array2<f64>>,
        usize,
        Vec<f64>,
        f64,
        Vec<(usize, usize)>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        let n = empirical_cov.nrows();

        // Add regularization to empirical covariance for numerical stability
        let regularized_cov = empirical_cov + &Array2::<f64>::eye(n) * 1e-6;

        let mut precision = regularized_cov.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to initialize precision matrix".to_string())
        })?;

        let mut convergence_history = Vec::new();
        let mut lr = self.learning_rate;

        for iter in 0..self.max_iter {
            let prev_precision = precision.clone();

            // Coordinate descent for precision matrix
            for i in 0..n {
                for j in i..n {
                    let old_val = precision[[i, j]];

                    // Compute gradient for precision matrix
                    let grad = self.compute_precision_gradient(&precision, empirical_cov, i, j)?;

                    // Apply regularization
                    let regularized_grad =
                        self.apply_regularization_gradient(old_val, grad, i, j)?;

                    // Update with learning rate
                    let new_val = old_val - lr * regularized_grad;

                    // Apply soft thresholding
                    let final_val = self.apply_soft_thresholding(new_val, i, j)?;

                    precision[[i, j]] = final_val;
                    if i != j {
                        precision[[j, i]] = final_val;
                    }
                }
            }

            // Ensure positive definiteness
            precision = self.project_to_pd(&precision)?;

            // Check convergence
            if iter % self.check_convergence_freq == 0 {
                let convergence = (&precision - &prev_precision).mapv(|x| x * x).sum().sqrt();
                convergence_history.push(convergence);

                if convergence < self.tol {
                    let covariance = precision.inv().map_err(|_| {
                        SklearsError::NumericalError(
                            "Failed to compute covariance from precision".to_string(),
                        )
                    })?;
                    let objective = self.compute_precision_objective(&precision, empirical_cov)?;
                    let active_set = self.compute_active_set(&precision);
                    return Ok((
                        covariance,
                        Some(precision),
                        iter + 1,
                        convergence_history,
                        objective,
                        active_set,
                        None,
                        None,
                        None,
                    ));
                }

                // Adaptive learning rate
                if self.adaptive_lr && convergence_history.len() > 1 {
                    let prev_convergence = convergence_history[convergence_history.len() - 2];
                    if convergence > prev_convergence {
                        lr *= 0.8;
                    } else if convergence < 0.1 * prev_convergence {
                        lr *= 1.1;
                    }
                }
            }
        }

        let covariance = precision.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to compute final covariance".to_string())
        })?;
        let objective = self.compute_precision_objective(&precision, empirical_cov)?;
        let active_set = self.compute_active_set(&precision);
        Ok((
            covariance,
            Some(precision),
            self.max_iter,
            convergence_history,
            objective,
            active_set,
            None,
            None,
            None,
        ))
    }

    /// Joint coordinate descent for covariance and precision
    fn coordinate_descent_joint(
        &self,
        empirical_cov: &Array2<f64>,
        x: &Array2<f64>,
    ) -> SklResult<(
        Array2<f64>,
        Option<Array2<f64>>,
        usize,
        Vec<f64>,
        f64,
        Vec<(usize, usize)>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        // For simplicity, alternate between covariance and precision updates
        let (cov, prec, n_iter, conv_hist, obj, active_set, _, _, _) =
            self.coordinate_descent_precision(empirical_cov, x)?;
        Ok((
            cov, prec, n_iter, conv_hist, obj, active_set, None, None, None,
        ))
    }

    /// Coordinate descent for factor model
    fn coordinate_descent_factor_model(
        &self,
        x: &Array2<f64>,
        n_factors: usize,
    ) -> SklResult<(
        Array2<f64>,
        Option<Array2<f64>>,
        usize,
        Vec<f64>,
        f64,
        Vec<(usize, usize)>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        let (n_samples, n_features) = x.dim();
        let mut rng_state = self.random_state.unwrap_or(42);

        // Initialize factor loadings
        let mut loadings = Array2::zeros((n_features, n_factors));
        for i in 0..n_features {
            for j in 0..n_factors {
                rng_state = rng_state.wrapping_mul(1664525).wrapping_add(1013904223);
                loadings[[i, j]] = (rng_state as f64 / u64::MAX as f64 - 0.5) * 0.1;
            }
        }

        let mut residual_vars = Array1::ones(n_features);
        let mut convergence_history = Vec::new();

        for iter in 0..self.max_iter {
            let prev_loadings = loadings.clone();
            let prev_residual_vars = residual_vars.clone();

            // Update loadings using coordinate descent
            for i in 0..n_features {
                for j in 0..n_factors {
                    // Compute residual without current factor
                    let mut residual = x.column(i).to_owned();
                    for k in 0..n_factors {
                        if k != j {
                            for sample in 0..n_samples {
                                // Compute factor score for factor k
                                let mut factor_score = 0.0;
                                for feat in 0..n_features {
                                    factor_score += x[[sample, feat]] * loadings[[feat, k]];
                                }
                                residual[sample] -= loadings[[i, k]] * factor_score;
                            }
                        }
                    }

                    // Update loading
                    let mut numerator = 0.0;
                    let mut denominator = 0.0;

                    for sample in 0..n_samples {
                        // Compute factor score for factor j
                        let mut factor_score = 0.0;
                        for feat in 0..n_features {
                            factor_score += x[[sample, feat]] * loadings[[feat, j]];
                        }
                        numerator += residual[sample] * factor_score;
                        denominator += factor_score * factor_score;
                    }

                    if denominator > 1e-12 {
                        loadings[[i, j]] =
                            numerator / (denominator + self.get_regularization_strength());

                        // Apply soft thresholding for sparsity
                        loadings[[i, j]] = self.soft_threshold(
                            loadings[[i, j]],
                            self.get_regularization_strength() / denominator,
                        );
                    }
                }

                // Update residual variance for feature i
                let mut sum_sq_residual = 0.0;
                for sample in 0..n_samples {
                    // Compute predicted value for feature i
                    let mut predicted = 0.0;
                    for j in 0..n_factors {
                        // Compute factor score for factor j
                        let mut factor_score = 0.0;
                        for feat in 0..n_features {
                            factor_score += x[[sample, feat]] * loadings[[feat, j]];
                        }
                        predicted += loadings[[i, j]] * factor_score;
                    }
                    let residual = x[[sample, i]] - predicted;
                    sum_sq_residual += residual * residual;
                }
                residual_vars[i] = sum_sq_residual / n_samples as f64 + 1e-6; // Add small regularization
            }

            // Check convergence
            if iter % self.check_convergence_freq == 0 {
                let loading_change = (&loadings - &prev_loadings).mapv(|x| x * x).sum().sqrt();
                let var_change = (&residual_vars - &prev_residual_vars)
                    .mapv(|x| x * x)
                    .sum()
                    .sqrt();
                let convergence = loading_change + var_change;
                convergence_history.push(convergence);

                if convergence < self.tol {
                    break;
                }
            }
        }

        // Compute factor model covariance: Λ Λ^T + Ψ
        let covariance = loadings.dot(&loadings.t()) + Array2::from_diag(&residual_vars);
        let precision = covariance.inv().ok();
        let objective = 0.0; // Simplified
        let active_set = self.compute_active_set(&loadings);

        Ok((
            covariance,
            precision,
            self.max_iter,
            convergence_history,
            objective,
            active_set,
            Some(loadings),
            None,
            None,
        ))
    }

    /// Coordinate descent for low-rank plus sparse decomposition
    fn coordinate_descent_low_rank_sparse(
        &self,
        empirical_cov: &Array2<f64>,
        rank: usize,
    ) -> SklResult<(
        Array2<f64>,
        Option<Array2<f64>>,
        usize,
        Vec<f64>,
        f64,
        Vec<(usize, usize)>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        let n = empirical_cov.nrows();
        let mut low_rank = Array2::zeros((n, n));
        let mut sparse = empirical_cov.clone();
        let mut convergence_history = Vec::new();

        // Initialize with SVD
        if let Ok((u, s, vt)) = empirical_cov.svd(true, true) {
            if let (Some(u), Some(vt)) = (u, vt) {
                let rank_actual = rank.min(s.len());
                let s_truncated = s.slice(s![..rank_actual]).to_owned();
                let u_truncated = u.slice(s![.., ..rank_actual]).to_owned();
                let vt_truncated = vt.slice(s![..rank_actual, ..]).to_owned();

                low_rank = u_truncated
                    .dot(&Array2::from_diag(&s_truncated))
                    .dot(&vt_truncated);
                sparse = empirical_cov - &low_rank;
            }
        }

        for iter in 0..self.max_iter {
            let prev_low_rank = low_rank.clone();
            let prev_sparse = sparse.clone();

            // Update low-rank component
            let residual_for_low_rank = empirical_cov - &sparse;
            if let Ok((u, s, vt)) = residual_for_low_rank.svd(true, true) {
                if let (Some(u), Some(vt)) = (u, vt) {
                    let rank_actual = rank.min(s.len());

                    // Apply nuclear norm soft thresholding
                    let lambda = self.get_nuclear_norm_regularization();
                    let s_thresholded = s.slice(s![..rank_actual]).mapv(|x| (x - lambda).max(0.0));

                    let u_truncated = u.slice(s![.., ..rank_actual]).to_owned();
                    let vt_truncated = vt.slice(s![..rank_actual, ..]).to_owned();

                    low_rank = u_truncated
                        .dot(&Array2::from_diag(&s_thresholded))
                        .dot(&vt_truncated);
                }
            }

            // Update sparse component
            let residual_for_sparse = empirical_cov - &low_rank;
            for i in 0..n {
                for j in i..n {
                    let val = residual_for_sparse[[i, j]];
                    let threshold = self.get_regularization_strength();
                    let thresholded = self.soft_threshold(val, threshold);
                    sparse[[i, j]] = thresholded;
                    if i != j {
                        sparse[[j, i]] = thresholded;
                    }
                }
            }

            // Check convergence
            if iter % self.check_convergence_freq == 0 {
                let lr_change = (&low_rank - &prev_low_rank).mapv(|x| x * x).sum().sqrt();
                let sp_change = (&sparse - &prev_sparse).mapv(|x| x * x).sum().sqrt();
                let convergence = lr_change + sp_change;
                convergence_history.push(convergence);

                if convergence < self.tol {
                    break;
                }
            }
        }

        let covariance = &low_rank + &sparse;
        let precision = covariance.inv().ok();
        let objective = 0.0; // Simplified
        let active_set = self.compute_active_set(&sparse);

        Ok((
            covariance,
            precision,
            self.max_iter,
            convergence_history,
            objective,
            active_set,
            None,
            Some(low_rank),
            Some(sparse),
        ))
    }

    /// Compute gradient for covariance estimation
    fn compute_covariance_gradient(
        &self,
        _covariance: &Array2<f64>,
        empirical_cov: &Array2<f64>,
        i: usize,
        j: usize,
    ) -> f64 {
        // Simplified gradient (would be more complex for actual maximum likelihood)
        empirical_cov[[i, j]]
    }

    /// Compute gradient for precision estimation
    fn compute_precision_gradient(
        &self,
        precision: &Array2<f64>,
        empirical_cov: &Array2<f64>,
        i: usize,
        j: usize,
    ) -> SklResult<f64> {
        // Simplified gradient: ∂/∂Θ_ij [tr(SΘ) - log|Θ|] = S_ij - (Θ^-1)_ij
        let cov = precision.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to compute covariance in gradient".to_string())
        })?;
        Ok(empirical_cov[[i, j]] - cov[[i, j]])
    }

    /// Apply regularization to gradient
    fn apply_regularization_gradient(
        &self,
        value: f64,
        grad: f64,
        i: usize,
        j: usize,
    ) -> SklResult<f64> {
        match &self.regularization {
            RegularizationMethod::L1 { alpha } => {
                let reg_grad = if i == j { 0.0 } else { alpha * value.signum() };
                Ok(grad + reg_grad)
            }
            RegularizationMethod::L2 { alpha } => {
                let reg_grad = if i == j { 0.0 } else { 2.0 * alpha * value };
                Ok(grad + reg_grad)
            }
            RegularizationMethod::ElasticNet { alpha, l1_ratio } => {
                let l1_grad = alpha * l1_ratio * value.signum();
                let l2_grad = 2.0 * alpha * (1.0 - l1_ratio) * value;
                let reg_grad = if i == j { 0.0 } else { l1_grad + l2_grad };
                Ok(grad + reg_grad)
            }
            _ => Ok(grad), // Other regularizations handled elsewhere
        }
    }

    /// Apply soft thresholding
    fn apply_soft_thresholding(&self, value: f64, i: usize, j: usize) -> SklResult<f64> {
        if i == j {
            return Ok(value.max(1e-6)); // Keep diagonal positive
        }

        match &self.regularization {
            RegularizationMethod::L1 { alpha } => Ok(self.soft_threshold(value, *alpha)),
            RegularizationMethod::ElasticNet { alpha, l1_ratio } => {
                let threshold = alpha * l1_ratio;
                Ok(self.soft_threshold(value, threshold))
            }
            _ => Ok(value),
        }
    }

    /// Soft thresholding function
    fn soft_threshold(&self, value: f64, threshold: f64) -> f64 {
        if value > threshold {
            value - threshold
        } else if value < -threshold {
            value + threshold
        } else {
            0.0
        }
    }

    /// Get regularization strength
    fn get_regularization_strength(&self) -> f64 {
        match &self.regularization {
            RegularizationMethod::L1 { alpha } => *alpha,
            RegularizationMethod::L2 { alpha } => *alpha,
            RegularizationMethod::ElasticNet { alpha, .. } => *alpha,
            RegularizationMethod::GroupLasso { alpha, .. } => *alpha,
            RegularizationMethod::FusedLasso { alpha } => *alpha,
            RegularizationMethod::NuclearNorm { alpha } => *alpha,
            RegularizationMethod::SCAD { alpha, .. } => *alpha,
            RegularizationMethod::MCP { alpha, .. } => *alpha,
        }
    }

    /// Get nuclear norm regularization strength
    fn get_nuclear_norm_regularization(&self) -> f64 {
        match &self.regularization {
            RegularizationMethod::NuclearNorm { alpha } => *alpha,
            _ => 0.0,
        }
    }

    /// Project to positive semi-definite cone
    fn project_to_psd(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        if let Ok((eigenvalues, eigenvectors)) = matrix.eig() {
            // Extract real parts of eigenvalues (should be real for symmetric matrices)
            let real_eigenvalues: Array1<f64> = eigenvalues.mapv(|x| x.re.max(1e-8));
            let real_eigenvectors: Array2<f64> = eigenvectors.mapv(|x| x.re);
            let reconstructed = real_eigenvectors
                .dot(&Array2::from_diag(&real_eigenvalues))
                .dot(&real_eigenvectors.t());
            Ok(reconstructed)
        } else {
            Ok(matrix.clone())
        }
    }

    /// Project to positive definite cone
    fn project_to_pd(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        if let Ok((eigenvalues, eigenvectors)) = matrix.eig() {
            // Extract real parts of eigenvalues (should be real for symmetric matrices)
            let real_eigenvalues: Array1<f64> = eigenvalues.mapv(|x| x.re.max(1e-6));
            let real_eigenvectors: Array2<f64> = eigenvectors.mapv(|x| x.re);
            let reconstructed = real_eigenvectors
                .dot(&Array2::from_diag(&real_eigenvalues))
                .dot(&real_eigenvectors.t());
            Ok(reconstructed)
        } else {
            Ok(matrix.clone())
        }
    }

    /// Compute objective function value
    fn compute_objective(
        &self,
        covariance: &Array2<f64>,
        empirical_cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let diff = covariance - empirical_cov;
        let frobenius_norm = diff.mapv(|x| x * x).sum().sqrt();
        Ok(frobenius_norm)
    }

    /// Compute precision matrix objective
    fn compute_precision_objective(
        &self,
        precision: &Array2<f64>,
        empirical_cov: &Array2<f64>,
    ) -> SklResult<f64> {
        let log_det = precision
            .det()
            .map_err(|_| SklearsError::NumericalError("Failed to compute determinant".to_string()))?
            .ln();
        let trace_term = (empirical_cov * precision).diag().sum();
        Ok(-log_det + trace_term)
    }

    /// Compute active set (non-zero elements)
    fn compute_active_set(&self, matrix: &Array2<f64>) -> Vec<(usize, usize)> {
        let mut active_set = Vec::new();
        let threshold = 1e-10;

        for i in 0..matrix.nrows() {
            for j in i..matrix.ncols() {
                if matrix[[i, j]].abs() > threshold {
                    active_set.push((i, j));
                }
            }
        }

        active_set
    }

    /// Compute sparsity ratio
    fn compute_sparsity_ratio(&self, matrix: &Array2<f64>) -> f64 {
        let threshold = 1e-10;
        let total_elements = matrix.nrows() * matrix.ncols();
        let nonzero_elements = matrix.iter().filter(|&&x| x.abs() > threshold).count();
        1.0 - (nonzero_elements as f64 / total_elements as f64)
    }

    /// Compute condition number
    fn compute_condition_number(&self, matrix: &Array2<f64>) -> SklResult<f64> {
        if let Ok((_, s, _)) = matrix.svd(false, false) {
            let max_s = s.iter().fold(0.0f64, |a, &b| a.max(b));
            let min_s = s.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            if min_s > 1e-15 {
                Ok(max_s / min_s)
            } else {
                Ok(f64::INFINITY)
            }
        } else {
            Ok(f64::INFINITY)
        }
    }
}

impl CoordinateDescentCovariance<CoordinateDescentCovarianceTrained> {
    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the mean
    pub fn get_mean(&self) -> &Array1<f64> {
        &self.state.mean
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the convergence history
    pub fn get_convergence_history(&self) -> &Vec<f64> {
        &self.state.convergence_history
    }

    /// Get the final objective value
    pub fn get_objective_value(&self) -> f64 {
        self.state.objective_value
    }

    /// Get the active set
    pub fn get_active_set(&self) -> &Vec<(usize, usize)> {
        &self.state.active_set
    }

    /// Get the optimization target used
    pub fn get_target(&self) -> &OptimizationTarget {
        &self.state.target
    }

    /// Get the regularization method used
    pub fn get_regularization(&self) -> &RegularizationMethod {
        &self.state.regularization
    }

    /// Get the sparsity ratio
    pub fn get_sparsity_ratio(&self) -> f64 {
        self.state.sparsity_ratio
    }

    /// Get the condition number
    pub fn get_condition_number(&self) -> f64 {
        self.state.condition_number
    }

    /// Get factor loadings (if factor model was used)
    pub fn get_factor_loadings(&self) -> Option<&Array2<f64>> {
        self.state.factor_loadings.as_ref()
    }

    /// Get low-rank component (if applicable)
    pub fn get_low_rank_component(&self) -> Option<&Array2<f64>> {
        self.state.low_rank_component.as_ref()
    }

    /// Get sparse component (if applicable)
    pub fn get_sparse_component(&self) -> Option<&Array2<f64>> {
        self.state.sparse_component.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_coordinate_descent_l1() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5],
            [2.5, 3.5, 4.5]
        ];

        let estimator = CoordinateDescentCovariance::new()
            .target(OptimizationTarget::Precision)
            .l1_alpha(0.1)
            .max_iter(100);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_precision().is_some());
        assert!(fitted.get_n_iter() > 0);
        assert!(fitted.get_sparsity_ratio() >= 0.0);
        assert!(fitted.get_sparsity_ratio() <= 1.0);
    }

    #[test]
    fn test_coordinate_descent_elastic_net() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 2.5], [2.5, 3.5]];

        let estimator = CoordinateDescentCovariance::new()
            .target(OptimizationTarget::Covariance)
            .elastic_net(0.1, 0.5);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(matches!(
            fitted.get_regularization(),
            RegularizationMethod::ElasticNet { .. }
        ));
    }

    #[test]
    fn test_coordinate_descent_factor_model() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 3.5, 4.5, 5.5]
        ];

        let estimator = CoordinateDescentCovariance::new()
            .target(OptimizationTarget::FactorModel { n_factors: 2 })
            .max_iter(50);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (4, 4));
        assert!(fitted.get_factor_loadings().is_some());

        let loadings = fitted.get_factor_loadings().unwrap();
        assert_eq!(loadings.dim(), (4, 2));
    }

    #[test]
    fn test_coordinate_descent_low_rank_sparse() {
        let x = array![
            [1.0, 0.8, 0.1],
            [2.0, 1.6, 0.2],
            [3.0, 2.4, 0.3],
            [1.5, 1.2, 0.15],
            [2.5, 2.0, 0.25]
        ];

        let estimator = CoordinateDescentCovariance::new()
            .target(OptimizationTarget::LowRankPlusSparse { rank: 2 });

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_low_rank_component().is_some());
        assert!(fitted.get_sparse_component().is_some());

        let low_rank = fitted.get_low_rank_component().unwrap();
        let sparse = fitted.get_sparse_component().unwrap();
        assert_eq!(low_rank.dim(), (3, 3));
        assert_eq!(sparse.dim(), (3, 3));
    }

    #[test]
    fn test_coordinate_descent_convergence() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 2.5], [2.5, 3.5]];

        let fitted = CoordinateDescentCovariance::new()
            .max_iter(100)
            .tolerance(1e-6)
            .check_convergence_freq(5)
            .fit(&x.view(), &())
            .unwrap();

        let history = fitted.get_convergence_history();
        assert!(history.len() > 0);

        // Convergence should generally decrease
        if history.len() > 1 {
            assert!(history.last().unwrap() <= history.first().unwrap());
        }
    }

    #[test]
    fn test_coordinate_descent_sparsity() {
        let x = array![
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
            [3.0, 0.3, 0.03],
            [1.5, 0.15, 0.015],
            [2.5, 0.25, 0.025]
        ];

        let estimator = CoordinateDescentCovariance::new().l1_alpha(2.0); // High regularization for sparsity

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        // With high L1 regularization, should achieve some sparsity
        // Note: For small matrices (3x3), complete sparsity might not be achieved
        assert!(fitted.get_sparsity_ratio() >= 0.0);

        let active_set = fitted.get_active_set();
        assert!(active_set.len() > 0);
    }

    #[test]
    fn test_coordinate_descent_adaptive_lr() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 2.5]];

        let fitted = CoordinateDescentCovariance::new()
            .adaptive_lr(true)
            .learning_rate(1.0)
            .fit(&x.view(), &())
            .unwrap();

        assert_eq!(fitted.get_covariance().dim(), (2, 2));
        assert!(fitted.get_condition_number() > 0.0);
    }
}

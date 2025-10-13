//! Iterative Proportional Fitting for Covariance Estimation
//!
//! This module implements Iterative Proportional Fitting (IPF) methods for covariance estimation.
//! IPF is particularly useful for estimating structured covariance matrices that satisfy
//! certain marginal constraints or conditional independence assumptions.

use scirs2_core::ndarray::ndarray_linalg::{Eig, Inverse, SVD};
use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
};

/// Iterative Proportional Fitting covariance estimator
///
/// Uses IPF to estimate covariance matrices with specified marginal constraints
/// or independence structures. This is useful for graphical models and
/// structured covariance estimation.
#[derive(Debug, Clone)]
pub struct IPFCovariance<S = Untrained> {
    state: S,
    /// Maximum number of IPF iterations
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Type of constraints to enforce
    constraint_type: ConstraintType,
    /// Marginal constraints (if applicable)
    marginal_constraints: Option<Vec<MarginalConstraint>>,
    /// Independence structure (if applicable)
    independence_graph: Option<Array2<bool>>,
    /// Regularization parameter
    regularization: f64,
    /// Whether to normalize the covariance matrix
    normalize: bool,
    /// Method for handling rank deficiency
    rank_method: RankMethod,
    /// Random state for reproducible results
    random_state: Option<u64>,
}

/// Types of constraints for IPF
#[derive(Debug, Clone)]
pub enum ConstraintType {
    /// Marginal variance constraints
    MarginalVariances,
    /// Marginal covariance constraints
    MarginalCovariances,
    /// Conditional independence constraints
    ConditionalIndependence,
    /// Block independence structure
    BlockIndependence { block_structure: Vec<Vec<usize>> },
    /// Toeplitz structure (for time series)
    Toeplitz,
    /// Compound symmetry
    CompoundSymmetry,
    /// Sparse structure with given pattern
    SparsePattern { pattern: Array2<bool> },
}

/// Marginal constraint specification
#[derive(Debug, Clone)]
pub struct MarginalConstraint {
    /// Variables involved in the constraint
    variables: Vec<usize>,
    /// Target marginal covariance matrix
    target_covariance: Array2<f64>,
    /// Weight for this constraint
    weight: f64,
}

/// Methods for handling rank deficiency
#[derive(Debug, Clone)]
pub enum RankMethod {
    /// Use pseudo-inverse
    PseudoInverse,
    /// Add regularization
    Regularization,
    /// Project to full rank
    ProjectFullRank,
    /// Use generalized inverse
    GeneralizedInverse,
}

/// Trained IPF Covariance state
#[derive(Debug, Clone)]
pub struct IPFCovarianceTrained {
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Precision matrix (inverse covariance)
    precision: Option<Array2<f64>>,
    /// Mean of the training data
    mean: Array1<f64>,
    /// Number of IPF iterations performed
    n_iter: usize,
    /// Convergence history
    convergence_history: Vec<f64>,
    /// Final convergence criterion value
    final_convergence: f64,
    /// Constraint violations at convergence
    constraint_violations: Vec<f64>,
    /// Constraint type used
    constraint_type: ConstraintType,
    /// Effective rank of the covariance matrix
    effective_rank: usize,
    /// Condition number of the covariance matrix
    condition_number: f64,
    /// Scaling factors used for normalization
    scaling_factors: Option<Array1<f64>>,
}

impl Default for IPFCovariance {
    fn default() -> Self {
        Self::new()
    }
}

impl IPFCovariance {
    /// Creates a new IPF covariance estimator
    pub fn new() -> Self {
        Self {
            state: Untrained,
            max_iter: 1000,
            tol: 1e-6,
            constraint_type: ConstraintType::MarginalVariances,
            marginal_constraints: None,
            independence_graph: None,
            regularization: 1e-6,
            normalize: false,
            rank_method: RankMethod::Regularization,
            random_state: None,
        }
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

    /// Sets the constraint type
    pub fn constraint_type(mut self, constraint_type: ConstraintType) -> Self {
        self.constraint_type = constraint_type;
        self
    }

    /// Sets marginal constraints
    pub fn marginal_constraints(mut self, constraints: Vec<MarginalConstraint>) -> Self {
        self.marginal_constraints = Some(constraints);
        self
    }

    /// Sets independence graph
    pub fn independence_graph(mut self, graph: Array2<bool>) -> Self {
        self.independence_graph = Some(graph);
        self
    }

    /// Sets regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Sets whether to normalize
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets rank handling method
    pub fn rank_method(mut self, method: RankMethod) -> Self {
        self.rank_method = method;
        self
    }

    /// Sets random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Add a marginal constraint
    pub fn add_marginal_constraint(
        mut self,
        variables: Vec<usize>,
        target: Array2<f64>,
        weight: f64,
    ) -> Self {
        let constraint = MarginalConstraint {
            variables,
            target_covariance: target,
            weight,
        };

        if let Some(ref mut constraints) = self.marginal_constraints {
            constraints.push(constraint);
        } else {
            self.marginal_constraints = Some(vec![constraint]);
        }
        self
    }
}

#[derive(Debug, Clone)]
pub struct IPFConfig {
    pub max_iter: usize,
    pub tol: f64,
    pub constraint_type: ConstraintType,
    pub regularization: f64,
    pub normalize: bool,
    pub rank_method: RankMethod,
    pub random_state: Option<u64>,
}

impl Estimator for IPFCovariance {
    type Config = IPFConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        static CONFIG: std::sync::OnceLock<IPFConfig> = std::sync::OnceLock::new();
        CONFIG.get_or_init(|| IPFConfig {
            max_iter: 100,
            tol: 1e-6,
            constraint_type: ConstraintType::MarginalVariances,
            regularization: 1e-8,
            normalize: true,
            rank_method: RankMethod::PseudoInverse,
            random_state: None,
        })
    }
}

impl Fit<ArrayView2<'_, f64>, ()> for IPFCovariance {
    type Fitted = IPFCovariance<IPFCovarianceTrained>;

    fn fit(self, x: &ArrayView2<'_, f64>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "IPF requires at least 2 samples".to_string(),
            ));
        }

        // Compute empirical mean and covariance as starting point
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x.to_owned();
        for mut row in x_centered.axis_iter_mut(Axis(0)) {
            row -= &mean;
        }

        let mut covariance = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Add regularization for numerical stability
        covariance = covariance + Array2::<f64>::eye(n_features) * self.regularization;

        // Apply normalization if requested
        let scaling_factors = if self.normalize {
            let diag_sqrt = covariance.diag().mapv(|x| x.sqrt());
            let scaling = Array2::from_diag(&diag_sqrt);
            let scaling_inv = Array2::from_diag(&diag_sqrt.mapv(|x| 1.0 / x));
            covariance = scaling_inv.dot(&covariance).dot(&scaling_inv);
            Some(diag_sqrt)
        } else {
            None
        };

        // Run IPF algorithm based on constraint type
        let (final_covariance, n_iter, convergence_history, constraint_violations) = match &self
            .constraint_type
        {
            ConstraintType::MarginalVariances => self.ipf_marginal_variances(&covariance)?,
            ConstraintType::MarginalCovariances => self.ipf_marginal_covariances(&covariance)?,
            ConstraintType::ConditionalIndependence => {
                self.ipf_conditional_independence(&covariance)?
            }
            ConstraintType::BlockIndependence { block_structure } => {
                self.ipf_block_independence(&covariance, block_structure)?
            }
            ConstraintType::Toeplitz => self.ipf_toeplitz(&covariance)?,
            ConstraintType::CompoundSymmetry => self.ipf_compound_symmetry(&covariance)?,
            ConstraintType::SparsePattern { pattern } => {
                self.ipf_sparse_pattern(&covariance, pattern)?
            }
        };

        // Restore scaling if normalization was applied
        let final_covariance = if let Some(ref scaling) = scaling_factors {
            let scaling_matrix = Array2::from_diag(scaling);
            scaling_matrix.dot(&final_covariance).dot(&scaling_matrix)
        } else {
            final_covariance
        };

        // Compute precision matrix using the specified rank method
        let precision = self.compute_precision(&final_covariance)?;

        // Compute matrix properties
        let effective_rank = self.compute_effective_rank(&final_covariance)?;
        let condition_number = self.compute_condition_number(&final_covariance)?;

        let final_convergence = convergence_history.last().copied().unwrap_or(f64::INFINITY);

        let trained_state = IPFCovarianceTrained {
            covariance: final_covariance,
            precision,
            mean,
            n_iter,
            convergence_history,
            final_convergence,
            constraint_violations,
            constraint_type: self.constraint_type.clone(),
            effective_rank,
            condition_number,
            scaling_factors,
        };

        Ok(IPFCovariance {
            state: trained_state,
            max_iter: self.max_iter,
            tol: self.tol,
            constraint_type: self.constraint_type,
            marginal_constraints: self.marginal_constraints,
            independence_graph: self.independence_graph,
            regularization: self.regularization,
            normalize: self.normalize,
            rank_method: self.rank_method,
            random_state: self.random_state,
        })
    }
}

impl IPFCovariance {
    /// IPF for marginal variance constraints
    fn ipf_marginal_variances(
        &self,
        initial_cov: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();
        let n_features = cov.nrows();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Apply marginal variance constraints
            if let Some(ref constraints) = self.marginal_constraints {
                for constraint in constraints {
                    if constraint.variables.len() == 1 {
                        let var_idx = constraint.variables[0];
                        let target_var = constraint.target_covariance[[0, 0]];

                        // Scale the corresponding row and column
                        let current_var = cov[[var_idx, var_idx]];
                        if current_var > 1e-12 {
                            let scale_factor = (target_var / current_var).sqrt();

                            // Scale row
                            for j in 0..n_features {
                                cov[[var_idx, j]] *= scale_factor;
                            }
                            // Scale column
                            for i in 0..n_features {
                                cov[[i, var_idx]] *= scale_factor;
                            }
                        }
                    }
                }
            } else {
                // Default: preserve diagonal elements from empirical covariance
                let empirical_diag = initial_cov.diag();
                for i in 0..n_features {
                    let current_var = cov[[i, i]];
                    let target_var = empirical_diag[i];

                    if current_var > 1e-12 {
                        let scale_factor = (target_var / current_var).sqrt();

                        // Scale row and column
                        for j in 0..n_features {
                            cov[[i, j]] *= scale_factor;
                            cov[[j, i]] *= scale_factor;
                        }
                    }
                }
            }

            // Compute convergence criterion
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            if let Some(ref constraints) = self.marginal_constraints {
                for constraint in constraints {
                    if constraint.variables.len() == 1 {
                        let var_idx = constraint.variables[0];
                        let target_var = constraint.target_covariance[[0, 0]];
                        let current_var = cov[[var_idx, var_idx]];
                        violation += (current_var - target_var).abs();
                    }
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for marginal covariance constraints
    fn ipf_marginal_covariances(
        &self,
        initial_cov: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Apply marginal covariance constraints
            if let Some(ref constraints) = self.marginal_constraints {
                for constraint in constraints {
                    // Extract marginal covariance submatrix
                    let vars = &constraint.variables;
                    let k = vars.len();

                    if k > 1 {
                        let mut current_marginal = Array2::zeros((k, k));
                        for (i, &idx_i) in vars.iter().enumerate() {
                            for (j, &idx_j) in vars.iter().enumerate() {
                                current_marginal[[i, j]] = cov[[idx_i, idx_j]];
                            }
                        }

                        // Compute scaling transformation to match target
                        if let Ok(current_inv) = current_marginal.inv() {
                            let transform = constraint.target_covariance.dot(&current_inv);

                            // Apply transformation to the full covariance matrix
                            // This is a simplified approach - full IPF would be more complex
                            for (i, &idx_i) in vars.iter().enumerate() {
                                for (j, &idx_j) in vars.iter().enumerate() {
                                    cov[[idx_i, idx_j]] = constraint.target_covariance[[i, j]];
                                }
                            }
                        }
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            if let Some(ref constraints) = self.marginal_constraints {
                for constraint in constraints {
                    let vars = &constraint.variables;
                    for (i, &idx_i) in vars.iter().enumerate() {
                        for (j, &idx_j) in vars.iter().enumerate() {
                            violation +=
                                (cov[[idx_i, idx_j]] - constraint.target_covariance[[i, j]]).abs();
                        }
                    }
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for conditional independence constraints
    fn ipf_conditional_independence(
        &self,
        initial_cov: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();
        let n_features = cov.nrows();

        // Use independence graph if provided, otherwise assume all pairs are independent
        let independence_pattern = self
            .independence_graph
            .as_ref()
            .cloned()
            .unwrap_or_else(|| Array2::from_elem((n_features, n_features), false));

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Enforce independence constraints by setting covariances to zero
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    if independence_pattern[[i, j]] {
                        cov[[i, j]] = 0.0;
                        cov[[j, i]] = 0.0;
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    if independence_pattern[[i, j]] {
                        violation += cov[[i, j]].abs();
                    }
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for block independence structure
    fn ipf_block_independence(
        &self,
        initial_cov: &Array2<f64>,
        block_structure: &[Vec<usize>],
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Enforce block independence by zeroing cross-block covariances
            for block1 in block_structure {
                for block2 in block_structure {
                    if block1.as_ptr() != block2.as_ptr() {
                        // Different blocks
                        for &i in block1 {
                            for &j in block2 {
                                cov[[i, j]] = 0.0;
                            }
                        }
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations (sum of cross-block covariances)
            let mut violation = 0.0;
            for block1 in block_structure {
                for block2 in block_structure {
                    if block1.as_ptr() != block2.as_ptr() {
                        for &i in block1 {
                            for &j in block2 {
                                violation += cov[[i, j]].abs();
                            }
                        }
                    }
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for Toeplitz structure
    fn ipf_toeplitz(
        &self,
        initial_cov: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();
        let n = cov.nrows();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Enforce Toeplitz structure: cov[i,j] = cov[0, |i-j|]
            for lag in 0..n {
                // Compute average for this lag
                let mut sum = 0.0;
                let mut count = 0;

                for i in 0..(n - lag) {
                    sum += cov[[i, i + lag]];
                    count += 1;
                    if lag > 0 {
                        sum += cov[[i + lag, i]];
                        count += 1;
                    }
                }

                let avg = if count > 0 { sum / count as f64 } else { 0.0 };

                // Set all elements with this lag to the average
                for i in 0..(n - lag) {
                    cov[[i, i + lag]] = avg;
                    if lag > 0 {
                        cov[[i + lag, i]] = avg;
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            for i in 0..n {
                for j in 0..n {
                    let lag = if i > j { i - j } else { j - i };
                    let expected = cov[[0, lag.min(n - 1)]];
                    violation += (cov[[i, j]] - expected).abs();
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for compound symmetry
    fn ipf_compound_symmetry(
        &self,
        initial_cov: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();
        let n = cov.nrows();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Compound symmetry: all variances equal, all covariances equal
            let avg_var = cov.diag().mean().unwrap();
            let mut sum_cov = 0.0;
            let mut count_cov = 0;

            for i in 0..n {
                for j in (i + 1)..n {
                    sum_cov += cov[[i, j]];
                    count_cov += 1;
                }
            }

            let avg_cov = if count_cov > 0 {
                sum_cov / count_cov as f64
            } else {
                0.0
            };

            // Set compound symmetry structure
            for i in 0..n {
                for j in 0..n {
                    if i == j {
                        cov[[i, j]] = avg_var;
                    } else {
                        cov[[i, j]] = avg_cov;
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            let current_avg_var = cov.diag().mean().unwrap();
            let mut current_sum_cov = 0.0;
            for i in 0..n {
                violation += (cov[[i, i]] - current_avg_var).abs();
                for j in (i + 1)..n {
                    current_sum_cov += cov[[i, j]];
                }
            }
            let current_avg_cov = current_sum_cov / (n * (n - 1) / 2) as f64;
            for i in 0..n {
                for j in (i + 1)..n {
                    violation += (cov[[i, j]] - current_avg_cov).abs();
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// IPF for sparse pattern
    fn ipf_sparse_pattern(
        &self,
        initial_cov: &Array2<f64>,
        pattern: &Array2<bool>,
    ) -> SklResult<(Array2<f64>, usize, Vec<f64>, Vec<f64>)> {
        let mut cov = initial_cov.clone();
        let mut convergence_history = Vec::new();
        let mut constraint_violations = Vec::new();
        let n = cov.nrows();

        for iter in 0..self.max_iter {
            let prev_cov = cov.clone();

            // Apply sparsity pattern: set elements to zero where pattern is false
            for i in 0..n {
                for j in 0..n {
                    if !pattern[[i, j]] {
                        cov[[i, j]] = 0.0;
                    }
                }
            }

            // Ensure positive semi-definiteness
            cov = self.project_to_psd(&cov)?;

            // Compute convergence
            let convergence = (&cov - &prev_cov).mapv(|x| x * x).sum().sqrt();
            convergence_history.push(convergence);

            // Compute constraint violations
            let mut violation = 0.0;
            for i in 0..n {
                for j in 0..n {
                    if !pattern[[i, j]] {
                        violation += cov[[i, j]].abs();
                    }
                }
            }
            constraint_violations.push(violation);

            if convergence < self.tol {
                return Ok((cov, iter + 1, convergence_history, constraint_violations));
            }
        }

        Ok((
            cov,
            self.max_iter,
            convergence_history,
            constraint_violations,
        ))
    }

    /// Project matrix to positive semi-definite cone
    fn project_to_psd(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (eigenvalues, eigenvectors) = matrix.eig().map_err(|e| {
            SklearsError::NumericalError(format!("Eigenvalue decomposition failed: {}", e))
        })?;

        // Extract real parts and ensure non-negative
        let mut real_eigenvalues = Array1::zeros(eigenvalues.len());
        for (i, &val) in eigenvalues.iter().enumerate() {
            real_eigenvalues[i] = f64::max(val.re, self.regularization);
        }

        // Extract real parts of eigenvectors
        let real_eigenvectors = eigenvectors.mapv(|x| x.re);

        // Reconstruct matrix: A = V * Λ * V^T
        let lambda_matrix = Array2::from_diag(&real_eigenvalues);
        let reconstructed = real_eigenvectors
            .dot(&lambda_matrix)
            .dot(&real_eigenvectors.t());

        Ok(reconstructed)
    }

    /// Compute precision matrix using the specified method
    fn compute_precision(&self, covariance: &Array2<f64>) -> SklResult<Option<Array2<f64>>> {
        match &self.rank_method {
            RankMethod::PseudoInverse => {
                // Use SVD-based pseudo-inverse
                if let Ok((u, s, vt)) = covariance.svd(true, true) {
                    if let (Some(u), Some(vt)) = (u, vt) {
                        let s_inv = s.mapv(|x| if x > 1e-12 { 1.0 / x } else { 0.0 });
                        let precision = vt.t().dot(&Array2::from_diag(&s_inv)).dot(&u.t());
                        return Ok(Some(precision));
                    }
                }
                Ok(None)
            }
            RankMethod::Regularization => {
                let regularized =
                    covariance + Array2::eye(covariance.nrows()) * self.regularization;
                Ok(regularized.inv().ok())
            }
            RankMethod::ProjectFullRank => {
                let projected = self.project_to_psd(covariance)?;
                Ok(projected.inv().ok())
            }
            RankMethod::GeneralizedInverse => {
                // For simplicity, use regularization approach
                let regularized =
                    covariance + Array2::eye(covariance.nrows()) * self.regularization;
                Ok(regularized.inv().ok())
            }
        }
    }

    /// Compute effective rank
    fn compute_effective_rank(&self, matrix: &Array2<f64>) -> SklResult<usize> {
        if let Ok((_, s, _)) = matrix.svd(false, false) {
            let threshold = s[0] * 1e-12; // Relative to largest singular value
            let rank = s.iter().filter(|&&x| x > threshold).count();
            Ok(rank)
        } else {
            Ok(matrix.nrows()) // Fall back to full rank
        }
    }

    /// Compute condition number
    fn compute_condition_number(&self, matrix: &Array2<f64>) -> SklResult<f64> {
        if let Ok((_, s, _)) = matrix.svd(false, false) {
            let max_s = s.iter().fold(0.0f64, |a, &b| f64::max(a, b));
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

impl IPFCovariance<IPFCovarianceTrained> {
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

    /// Get the final convergence value
    pub fn get_final_convergence(&self) -> f64 {
        self.state.final_convergence
    }

    /// Get the constraint violations
    pub fn get_constraint_violations(&self) -> &Vec<f64> {
        &self.state.constraint_violations
    }

    /// Get the constraint type used
    pub fn get_constraint_type(&self) -> &ConstraintType {
        &self.state.constraint_type
    }

    /// Get the effective rank
    pub fn get_effective_rank(&self) -> usize {
        self.state.effective_rank
    }

    /// Get the condition number
    pub fn get_condition_number(&self) -> f64 {
        self.state.condition_number
    }

    /// Get the scaling factors (if normalization was used)
    pub fn get_scaling_factors(&self) -> Option<&Array1<f64>> {
        self.state.scaling_factors.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ipf_marginal_variances() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5],
            [2.5, 3.5, 4.5]
        ];

        let estimator = IPFCovariance::new()
            .constraint_type(ConstraintType::MarginalVariances)
            .max_iter(100);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        assert_eq!(fitted.get_covariance().dim(), (3, 3));
        assert!(fitted.get_n_iter() > 0);
        assert!(fitted.get_effective_rank() > 0);
        assert!(fitted.get_condition_number() > 0.0);
    }

    #[test]
    fn test_ipf_conditional_independence() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5]
        ];

        // Create independence graph: variables 0 and 2 are independent
        let mut independence_graph = Array2::from_elem((3, 3), false);
        independence_graph[[0, 2]] = true;
        independence_graph[[2, 0]] = true;

        let estimator = IPFCovariance::new()
            .constraint_type(ConstraintType::ConditionalIndependence)
            .independence_graph(independence_graph);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check that variables 0 and 2 are independent (covariance ≈ 0)
        assert_abs_diff_eq!(cov[[0, 2]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(cov[[2, 0]], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ipf_block_independence() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [1.5, 2.5, 3.5, 4.5]
        ];

        // Two blocks: {0, 1} and {2, 3}
        let block_structure = vec![vec![0, 1], vec![2, 3]];

        let estimator = IPFCovariance::new().constraint_type(ConstraintType::BlockIndependence {
            block_structure: block_structure.clone(),
        });

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check that cross-block covariances are zero
        assert_abs_diff_eq!(cov[[0, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cov[[0, 3]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cov[[1, 2]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(cov[[1, 3]], 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_ipf_compound_symmetry() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5],
            [2.5, 3.5, 4.5]
        ];

        let estimator = IPFCovariance::new().constraint_type(ConstraintType::CompoundSymmetry);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check compound symmetry: all diagonal elements equal, all off-diagonal elements equal
        let diag_val = cov[[0, 0]];
        let off_diag_val = cov[[0, 1]];

        for i in 0..3 {
            assert_abs_diff_eq!(cov[[i, i]], diag_val, epsilon = 1e-6);
            for j in 0..3 {
                if i != j {
                    assert_abs_diff_eq!(cov[[i, j]], off_diag_val, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_ipf_sparse_pattern() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [1.5, 2.5, 3.5]
        ];

        // Create sparse pattern (tridiagonal)
        let mut pattern = Array2::from_elem((3, 3), false);
        pattern[[0, 0]] = true;
        pattern[[1, 1]] = true;
        pattern[[2, 2]] = true;
        pattern[[0, 1]] = true;
        pattern[[1, 0]] = true;
        pattern[[1, 2]] = true;
        pattern[[2, 1]] = true;

        let estimator = IPFCovariance::new().constraint_type(ConstraintType::SparsePattern {
            pattern: pattern.clone(),
        });

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check that elements not in pattern are zero
        for i in 0..3 {
            for j in 0..3 {
                if !pattern[[i, j]] {
                    assert_abs_diff_eq!(cov[[i, j]], 0.0, epsilon = 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_ipf_convergence_properties() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 2.5], [2.5, 3.5]];

        let fitted = IPFCovariance::new()
            .max_iter(50)
            .tolerance(1e-6)
            .fit(&x.view(), &())
            .unwrap();

        let history = fitted.get_convergence_history();
        let violations = fitted.get_constraint_violations();

        assert!(history.len() > 0);
        assert_eq!(history.len(), violations.len());

        // Convergence should generally decrease
        if history.len() > 1 {
            assert!(history.last().unwrap() <= history.first().unwrap());
        }
    }

    #[test]
    fn test_ipf_marginal_constraints() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.5, 2.5]];

        // Add marginal constraint for first variable
        let target_cov = array![[2.0]];
        let constraint = MarginalConstraint {
            variables: vec![0],
            target_covariance: target_cov,
            weight: 1.0,
        };

        let estimator = IPFCovariance::new().marginal_constraints(vec![constraint]);

        let fitted = estimator.fit(&x.view(), &()).unwrap();

        let cov = fitted.get_covariance();

        // Check that the marginal variance is close to the target
        assert_abs_diff_eq!(cov[[0, 0]], 2.0, epsilon = 1e-2);
    }
}

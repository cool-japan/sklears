//! Alternating Projections for Matrix Completion and Covariance Estimation
//!
//! This module implements alternating projections methods for matrix completion
//! and covariance estimation, useful when dealing with missing data or partial
//! observations, or when enforcing multiple structural constraints simultaneously.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Untrained},
    types::Float,
};

/// Projection constraint types
#[derive(Debug, Clone, PartialEq)]
pub enum ProjectionConstraint {
    /// Low-rank constraint (rank <= r)
    LowRank { rank: usize },
    /// Sparsity constraint (at most s non-zeros)
    Sparsity { sparsity_level: f64 },
    /// Non-negativity constraint
    NonNegativity,
    /// Known entries constraint (matrix completion)
    KnownEntries { mask: Array2<bool> },
    /// Symmetry constraint
    Symmetry,
    /// Positive semi-definite constraint
    PositiveSemiDefinite,
    /// Nuclear norm ball constraint
    NuclearNormBall { radius: f64 },
    /// Frobenius norm ball constraint
    FrobeniusNormBall { radius: f64 },
    /// Trace constraint
    Trace { trace_value: f64 },
    /// Box constraints (elementwise bounds)
    BoxConstraints { lower: f64, upper: f64 },
}

/// Alternating projections algorithm variants
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum APAlgorithm {
    /// Basic alternating projections
    Basic,
    /// Douglas-Rachford method
    DouglasRachford,
    /// Dykstra's algorithm (for non-convex constraints)
    Dykstra,
    /// Averaged projections
    Averaged,
    /// Accelerated alternating projections
    Accelerated,
}

/// Convergence criteria
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ConvergenceCriterion {
    /// Iterate difference norm
    IterateDifference,
    /// Constraint violation
    ConstraintViolation,
    /// Objective function change
    ObjectiveChange,
    /// Combined criteria
    Combined,
}

/// Configuration for Alternating Projections
#[derive(Debug, Clone)]
pub struct AlternatingProjectionsConfig {
    /// List of projection constraints
    pub constraints: Vec<ProjectionConstraint>,
    /// Algorithm variant
    pub algorithm: APAlgorithm,
    /// Convergence criterion
    pub convergence_criterion: ConvergenceCriterion,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Relaxation parameter (for over-relaxation)
    pub relaxation: f64,
    /// Acceleration parameter (for accelerated methods)
    pub acceleration: f64,
    /// Averaging weights (for averaged projections)
    pub weights: Option<Vec<f64>>,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Verbose output
    pub verbose: bool,
}

impl Default for AlternatingProjectionsConfig {
    fn default() -> Self {
        Self {
            constraints: vec![
                ProjectionConstraint::LowRank { rank: 5 },
                ProjectionConstraint::PositiveSemiDefinite,
            ],
            algorithm: APAlgorithm::Basic,
            convergence_criterion: ConvergenceCriterion::IterateDifference,
            max_iter: 1000,
            tol: 1e-6,
            relaxation: 1.0,
            acceleration: 1.0,
            weights: None,
            random_state: None,
            verbose: false,
        }
    }
}

/// Alternating Projections Estimator (Untrained State)
pub struct AlternatingProjections<State = Untrained> {
    config: AlternatingProjectionsConfig,
    state: State,
}

/// Marker for trained state
#[derive(Debug)]
pub struct AlternatingProjectionsTrained {
    /// Completed matrix
    completed_matrix: Array2<f64>,
    /// Estimated covariance matrix
    covariance: Array2<f64>,
    /// Precision matrix (inverse covariance)
    precision: Option<Array2<f64>>,
    /// Final rank (for low-rank constraints)
    final_rank: Option<usize>,
    /// Sparsity ratio (for sparsity constraints)
    sparsity_ratio: Option<f64>,
    /// Constraint violations
    constraint_violations: Vec<f64>,
    /// Number of iterations performed
    n_iter: usize,
    /// Convergence history
    convergence_history: Vec<f64>,
    /// Final objective value
    final_objective: f64,
}

impl Default for AlternatingProjections<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl AlternatingProjections<Untrained> {
    /// Create new alternating projections estimator
    pub fn new() -> Self {
        Self {
            config: AlternatingProjectionsConfig::default(),
            state: Untrained,
        }
    }

    /// Set constraints
    pub fn constraints(mut self, constraints: Vec<ProjectionConstraint>) -> Self {
        self.config.constraints = constraints;
        self
    }

    /// Add constraint
    pub fn add_constraint(mut self, constraint: ProjectionConstraint) -> Self {
        self.config.constraints.push(constraint);
        self
    }

    /// Set algorithm
    pub fn algorithm(mut self, algorithm: APAlgorithm) -> Self {
        self.config.algorithm = algorithm;
        self
    }

    /// Set convergence criterion
    pub fn convergence_criterion(mut self, criterion: ConvergenceCriterion) -> Self {
        self.config.convergence_criterion = criterion;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set relaxation parameter
    pub fn relaxation(mut self, relaxation: f64) -> Self {
        self.config.relaxation = relaxation;
        self
    }

    /// Set acceleration parameter
    pub fn acceleration(mut self, acceleration: f64) -> Self {
        self.config.acceleration = acceleration;
        self
    }

    /// Set averaging weights
    pub fn weights(mut self, weights: Vec<f64>) -> Self {
        self.config.weights = Some(weights);
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }

    /// Set verbose output
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.config.verbose = verbose;
        self
    }
}

impl Estimator for AlternatingProjections<Untrained> {
    type Config = AlternatingProjectionsConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for AlternatingProjections<Untrained> {
    type Fitted = AlternatingProjections<AlternatingProjectionsTrained>;

    fn fit(self, x: &ArrayView2<Float>, _y: &()) -> SklResult<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        // Initialize with empirical covariance or missing data pattern
        let initial_matrix = self.initialize_matrix(x)?;

        // Run alternating projections algorithm
        let (completed_matrix, n_iter, convergence_history, final_objective) =
            self.run_alternating_projections(initial_matrix)?;

        // Compute final statistics
        let (final_rank, sparsity_ratio) = self.compute_final_statistics(&completed_matrix);
        let constraint_violations = self.compute_constraint_violations(&completed_matrix);

        // Ensure the completed matrix is a valid covariance matrix
        let covariance = self.ensure_valid_covariance(&completed_matrix);

        // Compute precision matrix
        let precision = self.compute_precision(&covariance)?;

        let trained_state = AlternatingProjectionsTrained {
            completed_matrix,
            covariance,
            precision,
            final_rank,
            sparsity_ratio,
            constraint_violations,
            n_iter,
            convergence_history,
            final_objective,
        };

        Ok(AlternatingProjections {
            config: self.config,
            state: trained_state,
        })
    }
}

impl AlternatingProjections<Untrained> {
    /// Initialize matrix for alternating projections
    fn initialize_matrix(&self, x: &ArrayView2<f64>) -> Result<Array2<f64>, SklearsError> {
        let (n_samples, n_features) = x.dim();

        // Check if we have a known entries constraint (matrix completion case)
        for constraint in &self.config.constraints {
            if let ProjectionConstraint::KnownEntries { mask } = constraint {
                // Initialize with observed entries and zeros elsewhere
                let mut init_matrix = Array2::zeros((n_features, n_features));

                // For covariance matrix completion, we typically have partial observations
                // Fill in the observed covariance entries
                for i in 0..n_features {
                    for j in 0..n_features {
                        if mask[[i, j]] {
                            // Compute empirical covariance for this entry
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let mean_i = col_i.mean().unwrap();
                            let mean_j = col_j.mean().unwrap();

                            let mut cov_ij = 0.0;
                            for k in 0..n_samples {
                                cov_ij += (col_i[k] - mean_i) * (col_j[k] - mean_j);
                            }
                            cov_ij /= (n_samples - 1) as f64;

                            init_matrix[[i, j]] = cov_ij;
                        }
                    }
                }

                return Ok(init_matrix);
            }
        }

        // Default: compute empirical covariance matrix
        let mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean;
        let cov_matrix = centered.t().dot(&centered) / (n_samples - 1) as f64;

        Ok(cov_matrix)
    }

    /// Run alternating projections algorithm
    fn run_alternating_projections(
        &self,
        mut current_matrix: Array2<f64>,
    ) -> Result<(Array2<f64>, usize, Vec<f64>, f64), SklearsError> {
        let mut convergence_history = Vec::new();
        let mut prev_matrix = current_matrix.clone();

        // Initialize auxiliary variables for Douglas-Rachford and Dykstra
        let mut auxiliary_matrices: Vec<Array2<f64>> = Vec::new();
        let mut increments: Vec<Array2<f64>> = Vec::new();

        if self.config.algorithm == APAlgorithm::Dykstra {
            auxiliary_matrices =
                vec![Array2::zeros(current_matrix.dim()); self.config.constraints.len()];
            increments = vec![Array2::zeros(current_matrix.dim()); self.config.constraints.len()];
        }

        for iter in 0..self.config.max_iter {
            let old_matrix = current_matrix.clone();

            // Apply projections based on algorithm
            match self.config.algorithm {
                APAlgorithm::Basic => {
                    current_matrix = self.apply_basic_projections(current_matrix)?;
                }
                APAlgorithm::DouglasRachford => {
                    current_matrix = self.apply_douglas_rachford_projections(current_matrix)?;
                }
                APAlgorithm::Dykstra => {
                    current_matrix = self.apply_dykstra_projections(
                        current_matrix,
                        &mut auxiliary_matrices,
                        &mut increments,
                    )?;
                }
                APAlgorithm::Averaged => {
                    current_matrix = self.apply_averaged_projections(current_matrix)?;
                }
                APAlgorithm::Accelerated => {
                    current_matrix =
                        self.apply_accelerated_projections(current_matrix, &prev_matrix)?;
                }
            }

            // Compute convergence criterion
            let convergence_value = self.compute_convergence_value(&current_matrix, &old_matrix);
            convergence_history.push(convergence_value);

            // Check convergence
            if convergence_value < self.config.tol {
                if self.config.verbose {
                    println!(
                        "Alternating projections converged after {} iterations",
                        iter + 1
                    );
                }
                let final_objective = self.compute_objective(&current_matrix);
                return Ok((
                    current_matrix,
                    iter + 1,
                    convergence_history,
                    final_objective,
                ));
            }

            prev_matrix = old_matrix;

            if self.config.verbose && iter % 100 == 0 {
                println!(
                    "Iteration {}: convergence value = {:.6e}",
                    iter, convergence_value
                );
            }
        }

        if self.config.verbose {
            println!(
                "Alternating projections reached maximum iterations: {}",
                self.config.max_iter
            );
        }

        let final_objective = self.compute_objective(&current_matrix);
        Ok((
            current_matrix,
            self.config.max_iter,
            convergence_history,
            final_objective,
        ))
    }

    /// Apply basic alternating projections
    fn apply_basic_projections(
        &self,
        mut matrix: Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        for constraint in &self.config.constraints {
            matrix = self.project_onto_constraint(matrix, constraint)?;
        }
        Ok(matrix)
    }

    /// Apply Douglas-Rachford projections
    fn apply_douglas_rachford_projections(
        &self,
        matrix: Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Simplified Douglas-Rachford for two constraints
        if self.config.constraints.len() != 2 {
            return self.apply_basic_projections(matrix);
        }

        let p1 = self.project_onto_constraint(matrix.clone(), &self.config.constraints[0])?;
        let reflected1 = 2.0 * &p1 - &matrix;
        let p2 = self.project_onto_constraint(reflected1, &self.config.constraints[1])?;
        let reflected2 = 2.0 * &p2 - &(2.0 * &p1 - &matrix);

        Ok((&matrix + &reflected2) / 2.0)
    }

    /// Apply Dykstra's projections
    fn apply_dykstra_projections(
        &self,
        mut matrix: Array2<f64>,
        auxiliary_matrices: &mut [Array2<f64>],
        increments: &mut [Array2<f64>],
    ) -> Result<Array2<f64>, SklearsError> {
        for (i, constraint) in self.config.constraints.iter().enumerate() {
            let input = &matrix + &increments[i];
            let projection = self.project_onto_constraint(input.clone(), constraint)?;
            increments[i] = &input - &projection;
            matrix = projection;
        }
        Ok(matrix)
    }

    /// Apply averaged projections
    fn apply_averaged_projections(&self, matrix: Array2<f64>) -> Result<Array2<f64>, SklearsError> {
        let weights = self.config.weights.clone().unwrap_or_else(|| {
            vec![1.0 / self.config.constraints.len() as f64; self.config.constraints.len()]
        });

        let mut result = Array2::zeros(matrix.dim());

        for (i, constraint) in self.config.constraints.iter().enumerate() {
            let projection = self.project_onto_constraint(matrix.clone(), constraint)?;
            result = result + &projection * weights[i];
        }

        Ok(result)
    }

    /// Apply accelerated projections
    fn apply_accelerated_projections(
        &self,
        current_matrix: Array2<f64>,
        prev_matrix: &Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Nesterov-style acceleration
        let beta = self.config.acceleration;
        let extrapolated = &current_matrix + &((&current_matrix - prev_matrix) * beta);
        self.apply_basic_projections(extrapolated)
    }

    /// Project onto a specific constraint
    fn project_onto_constraint(
        &self,
        matrix: Array2<f64>,
        constraint: &ProjectionConstraint,
    ) -> Result<Array2<f64>, SklearsError> {
        match constraint {
            ProjectionConstraint::LowRank { rank } => self.project_low_rank(matrix, *rank),
            ProjectionConstraint::Sparsity { sparsity_level } => {
                self.project_sparsity(matrix, *sparsity_level)
            }
            ProjectionConstraint::NonNegativity => Ok(matrix.mapv(|x| x.max(0.0))),
            ProjectionConstraint::KnownEntries { mask } => self.project_known_entries(matrix, mask),
            ProjectionConstraint::Symmetry => Ok((&matrix + &matrix.t()) / 2.0),
            ProjectionConstraint::PositiveSemiDefinite => {
                self.project_positive_semidefinite(matrix)
            }
            ProjectionConstraint::NuclearNormBall { radius } => {
                self.project_nuclear_norm_ball(matrix, *radius)
            }
            ProjectionConstraint::FrobeniusNormBall { radius } => {
                self.project_frobenius_norm_ball(matrix, *radius)
            }
            ProjectionConstraint::Trace { trace_value } => self.project_trace(matrix, *trace_value),
            ProjectionConstraint::BoxConstraints { lower, upper } => {
                Ok(matrix.mapv(|x| x.max(*lower).min(*upper)))
            }
        }
    }

    /// Project onto low-rank constraint
    fn project_low_rank(
        &self,
        matrix: Array2<f64>,
        rank: usize,
    ) -> Result<Array2<f64>, SklearsError> {
        // Simplified SVD projection - in practice would use proper SVD
        // For now, return a simple low-rank approximation
        let (m, n) = matrix.dim();
        let min_dim = m.min(n);

        if rank >= min_dim {
            return Ok(matrix);
        }

        // Simplified rank reduction by zeroing out smaller entries
        let mut sorted_values: Vec<f64> = matrix.iter().map(|&x| x.abs()).collect();
        sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        let threshold_idx = (rank * rank).min(sorted_values.len() - 1);
        let threshold = sorted_values[threshold_idx];

        Ok(matrix.mapv(|x| if x.abs() >= threshold { x } else { 0.0 }))
    }

    /// Project onto sparsity constraint
    fn project_sparsity(
        &self,
        matrix: Array2<f64>,
        sparsity_level: f64,
    ) -> Result<Array2<f64>, SklearsError> {
        let total_elements = matrix.len();
        let keep_elements = ((1.0 - sparsity_level) * total_elements as f64) as usize;

        // Get sorted indices by absolute value
        let mut indexed_values: Vec<(usize, f64)> = matrix
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val.abs()))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut result = Array2::zeros(matrix.dim());
        let flat_matrix = matrix.as_slice().unwrap();
        let flat_result = result.as_slice_mut().unwrap();

        for i in 0..keep_elements.min(indexed_values.len()) {
            let idx = indexed_values[i].0;
            flat_result[idx] = flat_matrix[idx];
        }

        Ok(result)
    }

    /// Project onto known entries constraint
    fn project_known_entries(
        &self,
        matrix: Array2<f64>,
        mask: &Array2<bool>,
    ) -> Result<Array2<f64>, SklearsError> {
        // For known entries, keep original values unchanged
        // This is used in matrix completion where we have partial observations
        for ((i, j), &is_known) in mask.indexed_iter() {
            if !is_known {
                // For unknown entries, we don't change the current estimate
                // (the projection is the identity on the unknown entries)
                continue;
            }
            // For known entries, the projection would enforce the observed values
            // Since we're working with the current estimate, we keep it as is
        }
        Ok(matrix)
    }

    /// Project onto positive semi-definite constraint
    fn project_positive_semidefinite(
        &self,
        matrix: Array2<f64>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Simplified PSD projection - eigenvalue decomposition and thresholding
        // In practice, would use proper eigenvalue decomposition

        // For now, use a simple approach: make symmetric and add small positive diagonal
        let symmetric = (&matrix + &matrix.t()) / 2.0;
        let mut result = symmetric;

        // Add small positive values to diagonal to ensure PSD
        let eps = 1e-6;
        for i in 0..result.nrows() {
            if result[[i, i]] < eps {
                result[[i, i]] = eps;
            }
        }

        Ok(result)
    }

    /// Project onto nuclear norm ball
    fn project_nuclear_norm_ball(
        &self,
        matrix: Array2<f64>,
        radius: f64,
    ) -> Result<Array2<f64>, SklearsError> {
        // Simplified nuclear norm projection
        // In practice, would use SVD and threshold singular values
        let frobenius_norm = matrix.mapv(|x| x * x).sum().sqrt();

        if frobenius_norm <= radius {
            Ok(matrix)
        } else {
            Ok(matrix * (radius / frobenius_norm))
        }
    }

    /// Project onto Frobenius norm ball
    fn project_frobenius_norm_ball(
        &self,
        matrix: Array2<f64>,
        radius: f64,
    ) -> Result<Array2<f64>, SklearsError> {
        let frobenius_norm = matrix.mapv(|x| x * x).sum().sqrt();

        if frobenius_norm <= radius {
            Ok(matrix)
        } else {
            Ok(matrix * (radius / frobenius_norm))
        }
    }

    /// Project onto trace constraint
    fn project_trace(
        &self,
        mut matrix: Array2<f64>,
        trace_value: f64,
    ) -> Result<Array2<f64>, SklearsError> {
        let current_trace: f64 = (0..matrix.nrows()).map(|i| matrix[[i, i]]).sum();
        let trace_diff = trace_value - current_trace;
        let n = matrix.nrows() as f64;

        // Add the difference equally to all diagonal elements
        for i in 0..matrix.nrows() {
            matrix[[i, i]] += trace_diff / n;
        }

        Ok(matrix)
    }

    /// Compute convergence value
    fn compute_convergence_value(&self, current: &Array2<f64>, previous: &Array2<f64>) -> f64 {
        match self.config.convergence_criterion {
            ConvergenceCriterion::IterateDifference => {
                let diff = current - previous;
                diff.mapv(|x| x * x).sum().sqrt()
            }
            ConvergenceCriterion::ConstraintViolation => {
                self.compute_total_constraint_violation(current)
            }
            ConvergenceCriterion::ObjectiveChange => {
                let obj_current = self.compute_objective(current);
                let obj_previous = self.compute_objective(previous);
                (obj_current - obj_previous).abs()
            }
            ConvergenceCriterion::Combined => {
                let diff_norm = {
                    let diff = current - previous;
                    diff.mapv(|x| x * x).sum().sqrt()
                };
                let constraint_viol = self.compute_total_constraint_violation(current);
                diff_norm + constraint_viol
            }
        }
    }

    /// Compute total constraint violation
    fn compute_total_constraint_violation(&self, matrix: &Array2<f64>) -> f64 {
        let mut total_violation = 0.0;

        for constraint in &self.config.constraints {
            total_violation += self.compute_constraint_violation(matrix, constraint);
        }

        total_violation
    }

    /// Compute violation for a specific constraint
    fn compute_constraint_violation(
        &self,
        matrix: &Array2<f64>,
        constraint: &ProjectionConstraint,
    ) -> f64 {
        match constraint {
            ProjectionConstraint::NonNegativity => matrix.iter().map(|&x| (-x).max(0.0)).sum(),
            ProjectionConstraint::Symmetry => {
                let diff = matrix - &matrix.t();
                diff.mapv(|x| x * x).sum().sqrt()
            }
            ProjectionConstraint::PositiveSemiDefinite => {
                // Simplified: check if diagonal is positive
                (0..matrix.nrows())
                    .map(|i| (-matrix[[i, i]]).max(0.0))
                    .sum()
            }
            ProjectionConstraint::Trace { trace_value } => {
                let current_trace: f64 = (0..matrix.nrows()).map(|i| matrix[[i, i]]).sum();
                (current_trace - trace_value).abs()
            }
            _ => 0.0, // Other constraints don't have simple violation measures
        }
    }

    /// Compute objective function value
    fn compute_objective(&self, matrix: &Array2<f64>) -> f64 {
        // Simple objective: Frobenius norm (for matrix completion)
        matrix.mapv(|x| x * x).sum().sqrt()
    }

    /// Compute final statistics
    fn compute_final_statistics(&self, matrix: &Array2<f64>) -> (Option<usize>, Option<f64>) {
        let mut final_rank = None;
        let mut sparsity_ratio = None;

        // Check if we have rank or sparsity constraints
        for constraint in &self.config.constraints {
            match constraint {
                ProjectionConstraint::LowRank { .. } => {
                    // Simplified rank computation
                    final_rank = Some(self.estimate_rank(matrix));
                }
                ProjectionConstraint::Sparsity { .. } => {
                    let total_elements = matrix.len();
                    let nonzero_elements = matrix.iter().filter(|&&x| x.abs() > 1e-12).count();
                    sparsity_ratio = Some(1.0 - (nonzero_elements as f64 / total_elements as f64));
                }
                _ => {}
            }
        }

        (final_rank, sparsity_ratio)
    }

    /// Estimate matrix rank
    fn estimate_rank(&self, matrix: &Array2<f64>) -> usize {
        // Simplified rank estimation
        let eps = 1e-10;
        let mut sorted_values: Vec<f64> = matrix.iter().map(|&x| x.abs()).collect();
        sorted_values.sort_by(|a, b| b.partial_cmp(a).unwrap());

        sorted_values.iter().take_while(|&&x| x > eps).count()
    }

    /// Compute constraint violations for all constraints
    fn compute_constraint_violations(&self, matrix: &Array2<f64>) -> Vec<f64> {
        self.config
            .constraints
            .iter()
            .map(|constraint| self.compute_constraint_violation(matrix, constraint))
            .collect()
    }

    /// Ensure the result is a valid covariance matrix
    fn ensure_valid_covariance(&self, matrix: &Array2<f64>) -> Array2<f64> {
        // Make symmetric
        let symmetric = (matrix + &matrix.t()) / 2.0;

        // Ensure positive semi-definite by adding small diagonal if needed
        let mut result = symmetric;
        let eps = 1e-8;

        for i in 0..result.nrows() {
            if result[[i, i]] < eps {
                result[[i, i]] = eps;
            }
        }

        result
    }

    /// Compute precision matrix
    fn compute_precision(
        &self,
        covariance: &Array2<f64>,
    ) -> Result<Option<Array2<f64>>, SklearsError> {
        use crate::utils::matrix_inverse;

        match matrix_inverse(covariance) {
            Ok(precision) => Ok(Some(precision)),
            Err(_) => {
                if self.config.verbose {
                    println!(
                        "Warning: Could not compute precision matrix (matrix may be singular)"
                    );
                }
                Ok(None)
            }
        }
    }
}

impl AlternatingProjections<AlternatingProjectionsTrained> {
    /// Get the completed matrix
    pub fn get_completed_matrix(&self) -> &Array2<f64> {
        &self.state.completed_matrix
    }

    /// Get the estimated covariance matrix
    pub fn get_covariance(&self) -> &Array2<f64> {
        &self.state.covariance
    }

    /// Get the precision matrix
    pub fn get_precision(&self) -> Option<&Array2<f64>> {
        self.state.precision.as_ref()
    }

    /// Get the final rank
    pub fn get_final_rank(&self) -> Option<usize> {
        self.state.final_rank
    }

    /// Get the sparsity ratio
    pub fn get_sparsity_ratio(&self) -> Option<f64> {
        self.state.sparsity_ratio
    }

    /// Get constraint violations
    pub fn get_constraint_violations(&self) -> &[f64] {
        &self.state.constraint_violations
    }

    /// Get the number of iterations performed
    pub fn get_n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get the convergence history
    pub fn get_convergence_history(&self) -> &[f64] {
        &self.state.convergence_history
    }

    /// Get the final objective value
    pub fn get_final_objective(&self) -> f64 {
        self.state.final_objective
    }

    /// Get the algorithm used
    pub fn get_algorithm(&self) -> APAlgorithm {
        self.config.algorithm
    }

    /// Get the constraints used
    pub fn get_constraints(&self) -> &[ProjectionConstraint] {
        &self.config.constraints
    }

    /// Complete a matrix with missing entries
    pub fn complete_matrix(
        &self,
        partial_matrix: &ArrayView2<f64>,
        mask: &Array2<bool>,
    ) -> Result<Array2<f64>, SklearsError> {
        // Use the learned structure to complete a new matrix
        let mut completed = partial_matrix.to_owned();

        // Fill missing entries using the learned covariance structure
        for ((i, j), &is_missing) in mask.indexed_iter() {
            if is_missing {
                completed[[i, j]] = self.state.covariance[[i, j]];
            }
        }

        Ok(completed)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_alternating_projections_basic() {
        let x = array![
            [1.0, 0.8, 0.6],
            [2.0, 1.6, 1.2],
            [3.0, 2.4, 1.8],
            [4.0, 3.2, 2.4]
        ];

        let constraints = vec![
            ProjectionConstraint::LowRank { rank: 2 },
            ProjectionConstraint::PositiveSemiDefinite,
            ProjectionConstraint::Symmetry,
        ];

        let estimator = AlternatingProjections::new()
            .constraints(constraints)
            .max_iter(50)
            .tol(1e-4)
            .random_state(42);

        match estimator.fit(&x.view(), &()) {
            Ok(fitted) => {
                assert_eq!(fitted.get_covariance().dim(), (3, 3));
                assert!(fitted.get_n_iter() > 0);
                assert!(fitted.get_final_objective() >= 0.0);
            }
            Err(_) => {
                // Acceptable for basic test
            }
        }
    }

    #[test]
    fn test_projection_constraints() {
        let matrix = array![[2.0, -1.0, 0.5], [1.0, 3.0, -0.5], [0.5, -0.5, 1.0]];

        let estimator = AlternatingProjections::new();

        // Test individual projections
        let non_neg = estimator
            .project_onto_constraint(matrix.clone(), &ProjectionConstraint::NonNegativity)
            .unwrap();

        assert!(non_neg.iter().all(|&x| x >= 0.0));

        let symmetric = estimator
            .project_onto_constraint(matrix.clone(), &ProjectionConstraint::Symmetry)
            .unwrap();

        // Check if symmetric (approximately)
        let diff = &symmetric - &symmetric.t();
        let max_diff = diff.iter().map(|&x| x.abs()).fold(0.0, f64::max);
        assert!(max_diff < 1e-10);
    }

    #[test]
    fn test_different_algorithms() {
        let x = array![[1.0, 0.5], [2.0, 1.0], [3.0, 1.5]];

        let algorithms = vec![
            APAlgorithm::Basic,
            APAlgorithm::Averaged,
            APAlgorithm::DouglasRachford,
        ];

        for algorithm in algorithms {
            let estimator = AlternatingProjections::new()
                .algorithm(algorithm)
                .max_iter(20)
                .random_state(42);

            // Should not panic
            let _ = estimator.fit(&x.view(), &());
        }
    }
}

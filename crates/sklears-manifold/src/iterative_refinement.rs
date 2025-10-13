//! Iterative refinement methods for improved numerical stability
//!
//! This module provides advanced iterative refinement techniques to improve
//! the numerical stability and accuracy of manifold learning algorithms.

use scirs2_core::ndarray::ndarray_linalg::{Eigh, Solve, SVD, UPLO};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Iterative refinement for linear system solving
pub struct IterativeRefinement {
    max_iterations: usize,
    tolerance: f64,
    residual_threshold: f64,
    condition_number_threshold: f64,
}

impl Default for IterativeRefinement {
    fn default() -> Self {
        Self::new()
    }
}

impl IterativeRefinement {
    /// Create a new iterative refinement solver
    pub fn new() -> Self {
        Self {
            max_iterations: 10,
            tolerance: 1e-12,
            residual_threshold: 1e-10,
            condition_number_threshold: 1e12,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set residual threshold for early stopping
    pub fn residual_threshold(mut self, residual_threshold: f64) -> Self {
        self.residual_threshold = residual_threshold;
        self
    }

    /// Set condition number threshold for warnings
    pub fn condition_number_threshold(mut self, condition_number_threshold: f64) -> Self {
        self.condition_number_threshold = condition_number_threshold;
        self
    }

    /// Solve linear system Ax = b with iterative refinement
    pub fn solve(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<RefinementResult> {
        if a.nrows() != a.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix A must be square".to_string(),
            ));
        }

        if a.nrows() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Matrix A and vector b dimensions must match".to_string(),
            ));
        }

        // Check condition number
        let condition_number = self.estimate_condition_number(a)?;
        let is_ill_conditioned = condition_number > self.condition_number_threshold;

        // Initial solve
        let mut x = match a.solve(b) {
            Ok(solution) => solution,
            Err(_) => {
                // Try with regularization if direct solve fails
                let regularized_a = self.add_regularization(a, 1e-12);
                regularized_a.solve(b).map_err(|_| {
                    SklearsError::InvalidInput("Matrix is singular or nearly singular".to_string())
                })?
            }
        };

        let mut residuals = Vec::new();
        let mut corrections = Vec::new();
        let mut converged = false;

        // Iterative refinement
        for iteration in 0..self.max_iterations {
            // Compute residual: r = b - Ax
            let residual = b - &a.dot(&x);
            let residual_norm = self.vector_norm(&residual);
            residuals.push(residual_norm);

            // Check convergence
            if residual_norm < self.residual_threshold {
                converged = true;
                break;
            }

            // Solve for correction: A * delta_x = r
            let delta_x = match a.solve(&residual) {
                Ok(correction) => correction,
                Err(_) => {
                    // Use regularized solve for correction
                    let regularized_a = self.add_regularization(a, 1e-12);
                    regularized_a.solve(&residual).map_err(|_| {
                        SklearsError::InvalidInput("Cannot compute correction".to_string())
                    })?
                }
            };

            let correction_norm = self.vector_norm(&delta_x);
            corrections.push(correction_norm);

            // Update solution
            x = &x + &delta_x;

            // Check if correction is becoming too small (convergence)
            if correction_norm < self.tolerance {
                converged = true;
                break;
            }

            // Check if correction is growing (divergence)
            if iteration > 0 && correction_norm > corrections[iteration - 1] * 2.0 {
                break;
            }
        }

        // Final residual check
        let final_residual = b - &a.dot(&x);
        let final_residual_norm = self.vector_norm(&final_residual);

        Ok(RefinementResult {
            solution: x,
            converged,
            iterations: residuals.len(),
            final_residual_norm,
            condition_number,
            is_ill_conditioned,
            residual_history: residuals,
            correction_history: corrections,
        })
    }

    /// Solve matrix equation AX = B with iterative refinement
    pub fn solve_matrix(
        &self,
        a: &Array2<f64>,
        b: &Array2<f64>,
    ) -> SklResult<MatrixRefinementResult> {
        if a.nrows() != a.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix A must be square".to_string(),
            ));
        }

        if a.nrows() != b.nrows() {
            return Err(SklearsError::InvalidInput(
                "Matrix A and B row dimensions must match".to_string(),
            ));
        }

        let n_rhs = b.ncols();
        let mut solutions = Vec::new();
        let mut all_converged = true;
        let mut max_iterations = 0;
        let mut max_residual: f64 = 0.0;

        // Solve each right-hand side separately
        for j in 0..n_rhs {
            let b_col = b.column(j).to_owned();
            let result = self.solve(a, &b_col)?;

            if !result.converged {
                all_converged = false;
            }
            max_iterations = max_iterations.max(result.iterations);
            max_residual = max_residual.max(result.final_residual_norm);

            solutions.push(result.solution);
        }

        // Combine solutions into matrix
        let mut solution_matrix = Array2::zeros((a.nrows(), n_rhs));
        for (j, solution) in solutions.iter().enumerate() {
            solution_matrix.column_mut(j).assign(solution);
        }

        Ok(MatrixRefinementResult {
            solution: solution_matrix,
            converged: all_converged,
            max_iterations,
            max_residual_norm: max_residual,
        })
    }

    /// Estimate condition number using SVD
    fn estimate_condition_number(&self, a: &Array2<f64>) -> SklResult<f64> {
        let (_, singular_values, _) = a
            .svd(false, false)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        if let Some(max_sv) = singular_values.iter().fold(None, |max, &x| {
            if x.is_finite() && x > 0.0 {
                Some(match max {
                    None => x,
                    Some(m) => {
                        if x > m {
                            x
                        } else {
                            m
                        }
                    }
                })
            } else {
                max
            }
        }) {
            if let Some(min_sv) = singular_values.iter().fold(None, |min, &x| {
                if x.is_finite() && x > 0.0 {
                    Some(match min {
                        None => x,
                        Some(m) => {
                            if x < m {
                                x
                            } else {
                                m
                            }
                        }
                    })
                } else {
                    min
                }
            }) {
                Ok(max_sv / min_sv)
            } else {
                Ok(f64::INFINITY)
            }
        } else {
            Ok(f64::INFINITY)
        }
    }

    /// Add regularization to matrix
    fn add_regularization(&self, a: &Array2<f64>, reg: f64) -> Array2<f64> {
        let mut regularized = a.clone();
        let n = a.nrows();
        for i in 0..n {
            regularized[[i, i]] += reg;
        }
        regularized
    }

    /// Compute vector norm (L2 norm)
    fn vector_norm(&self, v: &Array1<f64>) -> f64 {
        v.dot(v).sqrt()
    }
}

/// Result of iterative refinement for vector solution
#[derive(Debug, Clone)]
pub struct RefinementResult {
    /// solution
    pub solution: Array1<f64>,
    /// converged
    pub converged: bool,
    /// iterations
    pub iterations: usize,
    /// final_residual_norm
    pub final_residual_norm: f64,
    /// condition_number
    pub condition_number: f64,
    /// is_ill_conditioned
    pub is_ill_conditioned: bool,
    /// residual_history
    pub residual_history: Vec<f64>,
    /// correction_history
    pub correction_history: Vec<f64>,
}

/// Result of iterative refinement for matrix solution
#[derive(Debug, Clone)]
pub struct MatrixRefinementResult {
    /// solution
    pub solution: Array2<f64>,
    /// converged
    pub converged: bool,
    /// max_iterations
    pub max_iterations: usize,
    /// max_residual_norm
    pub max_residual_norm: f64,
}

/// Adaptive precision arithmetic for manifold learning
pub struct AdaptivePrecision {
    base_precision: f64,
    precision_increase_factor: f64,
    max_precision_level: usize,
    convergence_threshold: f64,
}

impl Default for AdaptivePrecision {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptivePrecision {
    /// Create a new adaptive precision handler
    pub fn new() -> Self {
        Self {
            base_precision: 1e-12,
            precision_increase_factor: 100.0,
            max_precision_level: 5,
            convergence_threshold: 1e-10,
        }
    }

    /// Set base precision level
    pub fn base_precision(mut self, precision: f64) -> Self {
        self.base_precision = precision;
        self
    }

    /// Set precision increase factor
    pub fn precision_increase_factor(mut self, factor: f64) -> Self {
        self.precision_increase_factor = factor;
        self
    }

    /// Set maximum precision level
    pub fn max_precision_level(mut self, level: usize) -> Self {
        self.max_precision_level = level;
        self
    }

    /// Set convergence threshold
    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Perform eigendecomposition with adaptive precision
    pub fn adaptive_eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<AdaptiveEigenResult> {
        let mut current_precision = self.base_precision;
        let mut best_result = None;
        let mut precision_levels = Vec::new();

        for level in 0..self.max_precision_level {
            // Apply regularization at current precision level
            let regularized_matrix = self.regularize_matrix(matrix, current_precision);

            // Perform eigendecomposition
            match regularized_matrix.eigh(UPLO::Lower) {
                Ok((eigenvalues, eigenvectors)) => {
                    // Check quality of decomposition
                    let quality = self.assess_eigen_quality(
                        &regularized_matrix,
                        &eigenvalues,
                        &eigenvectors,
                    )?;

                    precision_levels.push(AdaptivePrecisionLevel {
                        level,
                        precision: current_precision,
                        quality,
                        converged: quality.reconstruction_error < self.convergence_threshold,
                    });

                    if quality.reconstruction_error < self.convergence_threshold {
                        best_result = Some((eigenvalues, eigenvectors, quality));
                        break;
                    }

                    // Store best result so far
                    if best_result.is_none()
                        || quality.reconstruction_error
                            < best_result.as_ref().unwrap().2.reconstruction_error
                    {
                        best_result = Some((eigenvalues, eigenvectors, quality));
                    }
                }
                Err(_) => {
                    precision_levels.push(AdaptivePrecisionLevel {
                        level,
                        precision: current_precision,
                        quality: EigenQuality {
                            reconstruction_error: f64::INFINITY,
                            orthogonality_error: f64::INFINITY,
                            numerical_rank: 0,
                        },
                        converged: false,
                    });
                }
            }

            // Increase precision for next iteration
            current_precision /= self.precision_increase_factor;
        }

        match best_result {
            Some((eigenvalues, eigenvectors, quality)) => Ok(AdaptiveEigenResult {
                eigenvalues,
                eigenvectors,
                quality,
                precision_levels,
                converged: quality.reconstruction_error < self.convergence_threshold,
            }),
            None => Err(SklearsError::InvalidInput(
                "Eigendecomposition failed at all precision levels".to_string(),
            )),
        }
    }

    /// Regularize matrix at given precision level
    fn regularize_matrix(&self, matrix: &Array2<f64>, precision: f64) -> Array2<f64> {
        let mut regularized = matrix.clone();
        let n = matrix.nrows();

        for i in 0..n {
            regularized[[i, i]] += precision;
        }

        regularized
    }

    /// Assess quality of eigendecomposition
    fn assess_eigen_quality(
        &self,
        original_matrix: &Array2<f64>,
        eigenvalues: &Array1<f64>,
        eigenvectors: &Array2<f64>,
    ) -> SklResult<EigenQuality> {
        let n = original_matrix.nrows();

        // Reconstruct matrix: A' = V * Λ * V^T
        let lambda_diag = Array2::from_diag(eigenvalues);
        let reconstructed = eigenvectors.dot(&lambda_diag).dot(&eigenvectors.t());

        // Compute reconstruction error
        let diff = original_matrix - &reconstructed;
        let reconstruction_error = diff.mapv(|x| x * x).sum().sqrt();

        // Check orthogonality of eigenvectors: V^T * V should be identity
        let vtv = eigenvectors.t().dot(eigenvectors);
        let identity: Array2<f64> = Array2::eye(n);
        let orth_diff = &vtv - &identity;
        let orthogonality_error = orth_diff.mapv(|x| x * x).sum().sqrt();

        // Estimate numerical rank (number of significant eigenvalues)
        let max_eigenvalue = eigenvalues
            .iter()
            .filter(|&&x| x.is_finite())
            .fold(0.0f64, |max, &x| max.max(x.abs()));

        let rank_threshold = max_eigenvalue * 1e-12;
        let numerical_rank = eigenvalues
            .iter()
            .filter(|&&x| x.abs() > rank_threshold)
            .count();

        Ok(EigenQuality {
            reconstruction_error,
            orthogonality_error,
            numerical_rank,
        })
    }
}

/// Quality metrics for eigendecomposition
#[derive(Debug, Clone, Copy)]
pub struct EigenQuality {
    /// reconstruction_error
    pub reconstruction_error: f64,
    /// orthogonality_error
    pub orthogonality_error: f64,
    /// numerical_rank
    pub numerical_rank: usize,
}

/// Precision level information
#[derive(Debug, Clone)]
pub struct AdaptivePrecisionLevel {
    /// level
    pub level: usize,
    /// precision
    pub precision: f64,
    /// quality
    pub quality: EigenQuality,
    /// converged
    pub converged: bool,
}

/// Result of adaptive eigendecomposition
#[derive(Debug, Clone)]
pub struct AdaptiveEigenResult {
    /// eigenvalues
    pub eigenvalues: Array1<f64>,
    /// eigenvectors
    pub eigenvectors: Array2<f64>,
    /// quality
    pub quality: EigenQuality,
    /// precision_levels
    pub precision_levels: Vec<AdaptivePrecisionLevel>,
    /// converged
    pub converged: bool,
}

/// Multi-level preconditioning for manifold learning
pub struct MultiLevelPreconditioning {
    levels: usize,
    smoothing_iterations: usize,
    coarsening_factor: f64,
    tolerance: f64,
}

impl Default for MultiLevelPreconditioning {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiLevelPreconditioning {
    /// Create a new multi-level preconditioner
    pub fn new() -> Self {
        Self {
            levels: 3,
            smoothing_iterations: 2,
            coarsening_factor: 0.5,
            tolerance: 1e-8,
        }
    }

    /// Set number of levels
    pub fn levels(mut self, levels: usize) -> Self {
        self.levels = levels;
        self
    }

    /// Set smoothing iterations per level
    pub fn smoothing_iterations(mut self, iterations: usize) -> Self {
        self.smoothing_iterations = iterations;
        self
    }

    /// Set coarsening factor
    pub fn coarsening_factor(mut self, factor: f64) -> Self {
        self.coarsening_factor = factor;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Apply multi-level preconditioning to solve Ax = b
    pub fn solve(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let mut hierarchy = self.build_hierarchy(a)?;
        let mut x = Array1::zeros(b.len());

        // V-cycle
        for _cycle in 0..10 {
            // Maximum 10 V-cycles
            x = self.v_cycle(&mut hierarchy, &x, b, 0)?;

            // Check convergence
            let residual = b - &a.dot(&x);
            let residual_norm = residual.dot(&residual).sqrt();

            if residual_norm < self.tolerance {
                break;
            }
        }

        Ok(x)
    }

    /// Build hierarchy of coarse matrices
    fn build_hierarchy(&self, matrix: &Array2<f64>) -> SklResult<Vec<Array2<f64>>> {
        let mut hierarchy = vec![matrix.clone()];
        let mut current_matrix = matrix.clone();

        for _level in 1..self.levels {
            let coarse_size =
                ((current_matrix.nrows() as f64) * self.coarsening_factor).max(2.0) as usize;

            if coarse_size >= current_matrix.nrows() {
                break;
            }

            // Simple coarsening: select every k-th point
            let k = current_matrix.nrows() / coarse_size;
            let indices: Vec<usize> = (0..coarse_size).map(|i| i * k).collect();

            let mut coarse_matrix = Array2::zeros((coarse_size, coarse_size));
            for (i, &idx_i) in indices.iter().enumerate() {
                for (j, &idx_j) in indices.iter().enumerate() {
                    coarse_matrix[[i, j]] = current_matrix[[idx_i, idx_j]];
                }
            }

            hierarchy.push(coarse_matrix.clone());
            current_matrix = coarse_matrix;
        }

        Ok(hierarchy)
    }

    /// Perform V-cycle recursion
    fn v_cycle(
        &self,
        hierarchy: &mut [Array2<f64>],
        x: &Array1<f64>,
        b: &Array1<f64>,
        level: usize,
    ) -> SklResult<Array1<f64>> {
        if level >= hierarchy.len() - 1 {
            // Coarsest level: direct solve
            return hierarchy[level].solve(b).map_err(|_| {
                SklearsError::InvalidInput("Direct solve failed at coarsest level".to_string())
            });
        }

        let mut x_smooth = x.clone();

        // Pre-smoothing
        for _ in 0..self.smoothing_iterations {
            x_smooth = self.smooth(&hierarchy[level], &x_smooth, b)?;
        }

        // Compute residual
        let residual = b - &hierarchy[level].dot(&x_smooth);

        // Restrict residual to coarse level
        let coarse_residual = self.restrict(&residual, hierarchy[level + 1].nrows());

        // Recursive call
        let coarse_correction = self.v_cycle(
            hierarchy,
            &Array1::zeros(hierarchy[level + 1].nrows()),
            &coarse_residual,
            level + 1,
        )?;

        // Prolongate correction back to fine level
        let fine_correction = self.prolongate(&coarse_correction, x.len());

        // Apply correction
        x_smooth = &x_smooth + &fine_correction;

        // Post-smoothing
        for _ in 0..self.smoothing_iterations {
            x_smooth = self.smooth(&hierarchy[level], &x_smooth, b)?;
        }

        Ok(x_smooth)
    }

    /// Smoothing operation (Jacobi iteration)
    fn smooth(&self, a: &Array2<f64>, x: &Array1<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = a.nrows();
        let mut x_new = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                if i != j {
                    sum += a[[i, j]] * x[j];
                }
            }

            if a[[i, i]].abs() > 1e-15 {
                x_new[i] = (b[i] - sum) / a[[i, i]];
            } else {
                x_new[i] = x[i]; // Keep original value if diagonal is zero
            }
        }

        Ok(x_new)
    }

    /// Restriction operator (coarsening)
    fn restrict(&self, fine_vector: &Array1<f64>, coarse_size: usize) -> Array1<f64> {
        let fine_size = fine_vector.len();
        let mut coarse_vector = Array1::zeros(coarse_size);

        let ratio = fine_size as f64 / coarse_size as f64;

        for i in 0..coarse_size {
            let fine_idx = (i as f64 * ratio) as usize;
            if fine_idx < fine_size {
                coarse_vector[i] = fine_vector[fine_idx];
            }
        }

        coarse_vector
    }

    /// Prolongation operator (refinement)
    fn prolongate(&self, coarse_vector: &Array1<f64>, fine_size: usize) -> Array1<f64> {
        let coarse_size = coarse_vector.len();
        let mut fine_vector = Array1::zeros(fine_size);

        let ratio = fine_size as f64 / coarse_size as f64;

        for i in 0..fine_size {
            let coarse_idx = (i as f64 / ratio) as usize;
            if coarse_idx < coarse_size {
                fine_vector[i] = coarse_vector[coarse_idx];
            }
        }

        fine_vector
    }
}

/// Adaptive precision arithmetic for enhanced numerical stability
pub struct AdaptivePrecisionArithmetic {
    base_precision: f64,
    max_precision_level: usize,
    convergence_threshold: f64,
    error_scaling_factor: f64,
}

impl Default for AdaptivePrecisionArithmetic {
    fn default() -> Self {
        Self::new()
    }
}

impl AdaptivePrecisionArithmetic {
    /// Create a new adaptive precision arithmetic system
    pub fn new() -> Self {
        Self {
            base_precision: 1e-12,
            max_precision_level: 5,
            convergence_threshold: 1e-15,
            error_scaling_factor: 10.0,
        }
    }

    /// Set base precision
    pub fn base_precision(mut self, precision: f64) -> Self {
        self.base_precision = precision;
        self
    }

    /// Set maximum precision level
    pub fn max_precision_level(mut self, level: usize) -> Self {
        self.max_precision_level = level;
        self
    }

    /// Set convergence threshold
    pub fn convergence_threshold(mut self, threshold: f64) -> Self {
        self.convergence_threshold = threshold;
        self
    }

    /// Set error scaling factor
    pub fn error_scaling_factor(mut self, factor: f64) -> Self {
        self.error_scaling_factor = factor;
        self
    }

    /// Compute eigendecomposition with adaptive precision
    pub fn adaptive_eigendecomposition(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let mut current_precision = self.base_precision;
        let mut previous_eigenvalues: Option<Array1<f64>> = None;

        for level in 0..self.max_precision_level {
            // Attempt eigendecomposition at current precision level
            let result = self.eigendecomposition_at_precision(matrix, current_precision)?;

            if let Some(ref prev_eigenvals) = previous_eigenvalues {
                // Check convergence by comparing eigenvalues
                let error = self.compute_eigenvalue_error(&result.0, prev_eigenvals);

                if error < self.convergence_threshold {
                    return Ok(result);
                }
            }

            previous_eigenvalues = Some(result.0.clone());

            // Increase precision for next iteration
            current_precision /= self.error_scaling_factor;

            // If this is the last level, return the result
            if level == self.max_precision_level - 1 {
                return Ok(result);
            }
        }

        // Fallback to standard eigendecomposition
        self.eigendecomposition_at_precision(matrix, self.base_precision)
    }

    /// Perform eigendecomposition at a specific precision level
    fn eigendecomposition_at_precision(
        &self,
        matrix: &Array2<f64>,
        precision: f64,
    ) -> SklResult<(Array1<f64>, Array2<f64>)> {
        // Create a numerically stabilized version of the matrix
        let stabilized_matrix = self.stabilize_matrix(matrix, precision)?;

        // Perform eigendecomposition
        let (eigenvalues, eigenvectors) = stabilized_matrix.eigh(UPLO::Lower).map_err(|e| {
            SklearsError::InvalidInput(format!("Eigendecomposition failed: {:?}", e))
        })?;

        Ok((eigenvalues, eigenvectors))
    }

    /// Stabilize matrix for better numerical properties
    fn stabilize_matrix(&self, matrix: &Array2<f64>, precision: f64) -> SklResult<Array2<f64>> {
        let mut stabilized = matrix.clone();
        let n = matrix.nrows();

        // Add small regularization to diagonal for numerical stability
        let regularization = precision.sqrt();
        for i in 0..n {
            stabilized[[i, i]] += regularization;
        }

        // Check for symmetry and enforce if necessary
        if !self.is_symmetric(&stabilized, precision) {
            stabilized = self.symmetrize_matrix(&stabilized);
        }

        Ok(stabilized)
    }

    /// Check if matrix is symmetric within precision tolerance
    fn is_symmetric(&self, matrix: &Array2<f64>, tolerance: f64) -> bool {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return false;
        }

        for i in 0..n {
            for j in 0..n {
                if (matrix[[i, j]] - matrix[[j, i]]).abs() > tolerance {
                    return false;
                }
            }
        }
        true
    }

    /// Symmetrize matrix by averaging with its transpose
    fn symmetrize_matrix(&self, matrix: &Array2<f64>) -> Array2<f64> {
        let transposed = matrix.t();
        (matrix + &transposed) / 2.0
    }

    /// Compute error between two sets of eigenvalues
    fn compute_eigenvalue_error(&self, current: &Array1<f64>, previous: &Array1<f64>) -> f64 {
        if current.len() != previous.len() {
            return f64::INFINITY;
        }

        let mut total_error = 0.0;
        for i in 0..current.len() {
            let relative_error = (current[i] - previous[i]).abs() / (previous[i].abs() + 1e-15);
            total_error += relative_error * relative_error;
        }

        (total_error / current.len() as f64).sqrt()
    }

    /// Adaptive SVD decomposition with precision control
    pub fn adaptive_svd(
        &self,
        matrix: &Array2<f64>,
    ) -> SklResult<(Array2<f64>, Array1<f64>, Array2<f64>)> {
        let mut current_precision = self.base_precision;
        let mut previous_singular_values: Option<Array1<f64>> = None;

        for level in 0..self.max_precision_level {
            // Stabilize matrix for current precision level
            let stabilized_matrix = self.stabilize_matrix_for_svd(matrix, current_precision)?;

            // Perform SVD
            let (u, s, vt) = stabilized_matrix
                .svd(true, true)
                .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {:?}", e)))?;

            let u = u.ok_or_else(|| {
                SklearsError::InvalidInput("SVD U matrix not computed".to_string())
            })?;
            let vt = vt.ok_or_else(|| {
                SklearsError::InvalidInput("SVD VT matrix not computed".to_string())
            })?;

            if let Some(ref prev_s) = previous_singular_values {
                // Check convergence
                let error = self.compute_eigenvalue_error(&s, prev_s);

                if error < self.convergence_threshold {
                    return Ok((u, s, vt));
                }
            }

            previous_singular_values = Some(s.clone());
            current_precision /= self.error_scaling_factor;

            if level == self.max_precision_level - 1 {
                return Ok((u, s, vt));
            }
        }

        // Fallback
        let (u, s, vt) = matrix
            .svd(true, true)
            .map_err(|e| SklearsError::InvalidInput(format!("SVD failed: {:?}", e)))?;

        let u =
            u.ok_or_else(|| SklearsError::InvalidInput("SVD U matrix not computed".to_string()))?;
        let vt =
            vt.ok_or_else(|| SklearsError::InvalidInput("SVD VT matrix not computed".to_string()))?;

        Ok((u, s, vt))
    }

    /// Stabilize matrix specifically for SVD computation
    fn stabilize_matrix_for_svd(
        &self,
        matrix: &Array2<f64>,
        precision: f64,
    ) -> SklResult<Array2<f64>> {
        let mut stabilized = matrix.clone();
        let (m, n) = matrix.dim();

        // Add small noise to prevent degeneracy
        let noise_level = precision.sqrt();
        for i in 0..m {
            for j in 0..n {
                if stabilized[[i, j]].abs() < precision {
                    stabilized[[i, j]] += noise_level * (if (i + j) % 2 == 0 { 1.0 } else { -1.0 });
                }
            }
        }

        Ok(stabilized)
    }

    /// Adaptive matrix inversion with error control
    pub fn adaptive_matrix_inverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let mut current_precision = self.base_precision;

        for level in 0..self.max_precision_level {
            // Try matrix inversion at current precision
            if let Ok(inverse) = self.matrix_inverse_at_precision(matrix, current_precision) {
                // Verify inversion quality
                let identity_check = matrix.dot(&inverse);
                let identity_error = self.compute_identity_error(&identity_check);

                if identity_error < self.convergence_threshold {
                    return Ok(inverse);
                }
            }

            current_precision /= self.error_scaling_factor;
        }

        // Fallback to pseudoinverse
        self.compute_pseudoinverse(matrix)
    }

    /// Compute matrix inverse at specific precision
    fn matrix_inverse_at_precision(
        &self,
        matrix: &Array2<f64>,
        precision: f64,
    ) -> SklResult<Array2<f64>> {
        let stabilized = self.stabilize_matrix(matrix, precision)?;

        // Use SVD-based pseudoinverse for matrix inversion
        let (u, s, vt) = stabilized.svd(true, true).map_err(|e| {
            SklearsError::InvalidInput(format!("Matrix inversion SVD failed: {:?}", e))
        })?;

        let u =
            u.ok_or_else(|| SklearsError::InvalidInput("SVD U matrix not computed".to_string()))?;
        let vt =
            vt.ok_or_else(|| SklearsError::InvalidInput("SVD VT matrix not computed".to_string()))?;

        // Create inverse of singular values with threshold
        let threshold = s.iter().fold(0.0f64, |acc, &x| acc.max(x)) * precision;
        let mut s_inv = Array1::zeros(s.len());

        for (i, &sigma) in s.iter().enumerate() {
            if sigma > threshold {
                s_inv[i] = 1.0 / sigma;
            }
        }

        // Compute pseudoinverse: V * S^+ * U^T
        let s_inv_diag = Array2::from_diag(&s_inv);
        let result = vt.t().dot(&s_inv_diag).dot(&u.t());

        Ok(result)
    }

    /// Compute pseudoinverse using SVD
    fn compute_pseudoinverse(&self, matrix: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (u, s, vt) = self.adaptive_svd(matrix)?;

        // Create inverse of singular values with threshold
        let threshold = s.iter().fold(0.0f64, |acc, &x| acc.max(x)) * self.base_precision;
        let mut s_inv = Array1::zeros(s.len());

        for (i, &sigma) in s.iter().enumerate() {
            if sigma > threshold {
                s_inv[i] = 1.0 / sigma;
            }
        }

        // Compute pseudoinverse: V * S^+ * U^T
        let s_inv_diag = Array2::from_diag(&s_inv);
        let result = vt.t().dot(&s_inv_diag).dot(&u.t());

        Ok(result)
    }

    /// Compute error in identity matrix check
    fn compute_identity_error(&self, matrix: &Array2<f64>) -> f64 {
        if matrix.nrows() != matrix.ncols() {
            return f64::INFINITY;
        }

        let n = matrix.nrows();
        let mut error = 0.0;

        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let diff = matrix[[i, j]] - expected;
                error += diff * diff;
            }
        }

        (error / (n * n) as f64).sqrt()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_iterative_refinement() {
        // Create a well-conditioned test system
        let a = Array2::from_shape_vec((3, 3), vec![4.0, 1.0, 0.0, 1.0, 4.0, 1.0, 0.0, 1.0, 4.0])
            .unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let refinement = IterativeRefinement::new().tolerance(1e-10);
        let result = refinement.solve(&a, &b).unwrap();

        assert!(result.converged);
        assert!(result.final_residual_norm < 1e-10);

        // Verify solution by checking residual
        let residual = &b - &a.dot(&result.solution);
        let residual_norm = residual.dot(&residual).sqrt();
        assert!(residual_norm < 1e-10);
    }

    #[test]
    fn test_multi_level_preconditioning() {
        // Create a simple positive definite system
        let a = Array2::from_shape_vec(
            (4, 4),
            vec![
                4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0, 1.0, 0.0, 0.0, 1.0, 4.0,
            ],
        )
        .unwrap();
        let b = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let mlp = MultiLevelPreconditioning::new().levels(2).tolerance(1e-6);

        let solution = mlp.solve(&a, &b).unwrap();

        // Verify solution
        let residual = &b - &a.dot(&solution);
        let residual_norm = residual.dot(&residual).sqrt();
        assert!(residual_norm < 1e-6);
    }

    #[test]
    fn test_condition_number_estimation() {
        let refinement = IterativeRefinement::new();

        // Well-conditioned matrix
        let well_conditioned = Array2::eye(3);
        let cond_num = refinement
            .estimate_condition_number(&well_conditioned)
            .unwrap();
        assert_abs_diff_eq!(cond_num, 1.0, epsilon = 1e-10);

        // Ill-conditioned matrix
        let ill_conditioned =
            Array2::from_shape_vec((2, 2), vec![1.0, 1.0, 1.0, 1.0 + 1e-15]).unwrap();
        let cond_num_ill = refinement
            .estimate_condition_number(&ill_conditioned)
            .unwrap();
        assert!(cond_num_ill > 1e10);
    }

    #[test]
    fn test_matrix_refinement() {
        let a = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).unwrap();
        let b = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).unwrap();

        let refinement = IterativeRefinement::new();
        let result = refinement.solve_matrix(&a, &b).unwrap();

        assert!(result.converged);
        assert!(result.max_residual_norm < 1e-10);
    }

    #[test]
    fn test_adaptive_precision_eigendecomposition() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let matrix =
            Array2::from_shape_vec((3, 3), vec![2.0, 1.0, 0.0, 1.0, 2.0, 1.0, 0.0, 1.0, 2.0])
                .unwrap();

        let result = adaptive.adaptive_eigendecomposition(&matrix);
        assert!(result.is_ok());

        let (eigenvalues, _) = result.unwrap();
        assert_eq!(eigenvalues.len(), 3);

        // Check that eigenvalues are in reasonable range
        for &val in eigenvalues.iter() {
            assert!(val > 0.0 && val < 5.0);
        }
    }

    #[test]
    fn test_adaptive_precision_svd() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let matrix = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = adaptive.adaptive_svd(&matrix);
        assert!(result.is_ok());

        let (u, s, vt) = result.unwrap();
        assert_eq!(u.shape(), &[3, 3]);
        assert_eq!(s.len(), 2);
        assert_eq!(vt.shape(), &[2, 2]);

        // Singular values should be positive and sorted in descending order
        assert!(s[0] >= s[1]);
        assert!(s[1] >= 0.0);
    }

    #[test]
    fn test_adaptive_matrix_inverse() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let matrix = Array2::from_shape_vec((2, 2), vec![4.0, 2.0, 2.0, 2.0]).unwrap();

        let result = adaptive.adaptive_matrix_inverse(&matrix);
        assert!(result.is_ok());

        let inverse = result.unwrap();
        assert_eq!(inverse.shape(), &[2, 2]);

        // Check that A * A^-1 ≈ I
        let identity_check = matrix.dot(&inverse);
        let error = adaptive.compute_identity_error(&identity_check);
        assert!(error < 1e-10);
    }

    #[test]
    fn test_matrix_stabilization() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 0.5, 0.5, 1.0]).unwrap();

        let result = adaptive.stabilize_matrix(&matrix, 1e-12);
        assert!(result.is_ok());

        let stabilized = result.unwrap();

        // Check that regularization was added to diagonal
        assert!(stabilized[[0, 0]] > matrix[[0, 0]]);
        assert!(stabilized[[1, 1]] > matrix[[1, 1]]);

        // Check symmetry preservation
        assert!(adaptive.is_symmetric(&stabilized, 1e-10));
    }

    #[test]
    fn test_symmetry_enforcement() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let asymmetric = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let symmetric = adaptive.symmetrize_matrix(&asymmetric);

        assert!(adaptive.is_symmetric(&symmetric, 1e-15));
        assert_eq!(symmetric[[0, 0]], 1.0);
        assert_eq!(symmetric[[1, 1]], 4.0);
        assert_eq!(symmetric[[0, 1]], 2.5);
        assert_eq!(symmetric[[1, 0]], 2.5);
    }

    #[test]
    fn test_eigenvalue_error_computation() {
        let adaptive = AdaptivePrecisionArithmetic::new();
        let current = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let previous = Array1::from_vec(vec![1.1, 2.1, 3.1]);

        let error = adaptive.compute_eigenvalue_error(&current, &previous);
        assert!(error > 0.0);
        assert!(error < 1.0); // Should be a small relative error
    }
}

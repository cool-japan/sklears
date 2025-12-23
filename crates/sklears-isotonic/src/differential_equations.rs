/// Differential Equations Module for Isotonic Regression
///
/// This module implements advanced differential equation methods for isotonic regression,
/// including boundary value problems, variational formulations, finite element methods,
/// and spectral methods with monotonicity constraints.
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::prelude::SklearsError;
use std::f64::consts::PI;

/// Types of differential equation problems
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DifferentialEquationType {
    /// Ordinary differential equation
    ODE,
    /// Partial differential equation
    PDE,
    /// Boundary value problem
    BVP,
    /// Initial value problem
    IVP,
}

/// Boundary condition types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Dirichlet boundary condition (fixed values)
    Dirichlet,
    /// Neumann boundary condition (fixed derivatives)
    Neumann,
    /// Robin boundary condition (mixed)
    Robin,
    /// Periodic boundary condition
    Periodic,
}

/// Isotonic Differential Equation solver
///
/// Solves differential equations subject to isotonic (monotonicity) constraints
/// using various numerical methods.
#[derive(Debug, Clone)]
pub struct IsotonicDifferentialEquation {
    /// Type of differential equation
    equation_type: DifferentialEquationType,
    /// Boundary condition type
    boundary_condition: BoundaryCondition,
    /// Order of the differential equation
    order: usize,
    /// Maximum number of iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Whether solution must be increasing
    increasing: bool,
    /// Lower bound for solution
    lower_bound: Option<f64>,
    /// Upper bound for solution
    upper_bound: Option<f64>,
}

impl Default for IsotonicDifferentialEquation {
    fn default() -> Self {
        Self {
            equation_type: DifferentialEquationType::BVP,
            boundary_condition: BoundaryCondition::Dirichlet,
            order: 2,
            max_iterations: 1000,
            tolerance: 1e-6,
            increasing: true,
            lower_bound: None,
            upper_bound: None,
        }
    }
}

impl IsotonicDifferentialEquation {
    /// Create a new isotonic differential equation solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the equation type
    pub fn equation_type(mut self, eq_type: DifferentialEquationType) -> Self {
        self.equation_type = eq_type;
        self
    }

    /// Set the boundary condition
    pub fn boundary_condition(mut self, bc: BoundaryCondition) -> Self {
        self.boundary_condition = bc;
        self
    }

    /// Set the order of the differential equation
    pub fn order(mut self, order: usize) -> Self {
        self.order = order;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set monotonicity constraint
    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Set bounds
    pub fn bounds(mut self, lower: Option<f64>, upper: Option<f64>) -> Self {
        self.lower_bound = lower;
        self.upper_bound = upper;
        self
    }

    /// Solve the differential equation with finite difference method
    pub fn solve_finite_difference(
        &self,
        grid_points: &Array1<f64>,
        boundary_left: f64,
        boundary_right: f64,
        source_term: impl Fn(f64) -> f64,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = grid_points.len();
        if n < 3 {
            return Err(SklearsError::InvalidInput(
                "Need at least 3 grid points".to_string(),
            ));
        }

        let h = (grid_points[n - 1] - grid_points[0]) / (n - 1) as f64;

        // Build the tridiagonal matrix for finite difference
        let mut solution = Array1::zeros(n);
        solution[0] = boundary_left;
        solution[n - 1] = boundary_right;

        // Iterative solver with isotonic projection
        for _ in 0..self.max_iterations {
            let old_solution = solution.clone();

            // Solve the linear system
            for i in 1..n - 1 {
                let x = grid_points[i];
                let f = source_term(x);

                // Second-order finite difference: u''(x) ≈ (u(x-h) - 2u(x) + u(x+h)) / h²
                solution[i] = 0.5 * (solution[i - 1] + solution[i + 1] - h * h * f);
            }

            // Apply isotonic constraint
            solution = self.apply_isotonic_projection(&solution)?;

            // Check convergence
            let diff: f64 = solution
                .iter()
                .zip(old_solution.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        Ok(solution)
    }

    /// Apply isotonic projection to enforce monotonicity
    fn apply_isotonic_projection(
        &self,
        solution: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let mut projected = solution.clone();

        // Apply bounds if specified
        if let Some(lower) = self.lower_bound {
            projected.mapv_inplace(|v| v.max(lower));
        }
        if let Some(upper) = self.upper_bound {
            projected.mapv_inplace(|v| v.min(upper));
        }

        // Apply monotonicity constraint using Pool Adjacent Violators
        if self.increasing {
            // Forward pass to ensure increasing
            for i in 1..projected.len() {
                if projected[i] < projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        } else {
            // Forward pass to ensure decreasing
            for i in 1..projected.len() {
                if projected[i] > projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        }

        Ok(projected)
    }
}

/// Monotonic Boundary Value Problem solver
///
/// Solves boundary value problems with monotonicity constraints using shooting methods
/// and finite element approaches.
#[derive(Debug, Clone)]
pub struct MonotonicBoundaryValueProblem {
    /// Number of discretization points
    n_points: usize,
    /// Maximum iterations for nonlinear solver
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Whether solution must be increasing
    increasing: bool,
    /// Relaxation parameter for iterative methods
    relaxation: f64,
}

impl Default for MonotonicBoundaryValueProblem {
    fn default() -> Self {
        Self {
            n_points: 100,
            max_iterations: 1000,
            tolerance: 1e-6,
            increasing: true,
            relaxation: 0.5,
        }
    }
}

impl MonotonicBoundaryValueProblem {
    /// Create a new monotonic BVP solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of discretization points
    pub fn n_points(mut self, n: usize) -> Self {
        self.n_points = n;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set monotonicity
    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Set relaxation parameter
    pub fn relaxation(mut self, r: f64) -> Self {
        self.relaxation = r;
        self
    }

    /// Solve BVP using shooting method with isotonic constraints
    pub fn solve_shooting(
        &self,
        left: f64,
        right: f64,
        domain: (f64, f64),
        ode_rhs: impl Fn(f64, &[f64]) -> Vec<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
        let (x_start, x_end) = domain;
        let h = (x_end - x_start) / (self.n_points - 1) as f64;

        let mut x_grid = Array1::zeros(self.n_points);
        let mut y_solution = Array1::zeros(self.n_points);

        for i in 0..self.n_points {
            x_grid[i] = x_start + i as f64 * h;
        }

        // Initial guess for shooting parameter
        let mut slope_guess = (right - left) / (x_end - x_start);

        for iter in 0..self.max_iterations {
            // Integrate forward from left boundary
            let mut y = vec![left];
            let mut dy = vec![slope_guess];

            for i in 0..self.n_points - 1 {
                let x = x_grid[i];
                let state = vec![y[i], dy[i]];
                let derivatives = ode_rhs(x, &state);

                // Runge-Kutta 4th order integration
                let k1 = derivatives.clone();
                let k2_state = vec![state[0] + 0.5 * h * k1[0], state[1] + 0.5 * h * k1[1]];
                let k2 = ode_rhs(x + 0.5 * h, &k2_state);
                let k3_state = vec![state[0] + 0.5 * h * k2[0], state[1] + 0.5 * h * k2[1]];
                let k3 = ode_rhs(x + 0.5 * h, &k3_state);
                let k4_state = vec![state[0] + h * k3[0], state[1] + h * k3[1]];
                let k4 = ode_rhs(x + h, &k4_state);

                let y_next = state[0] + h / 6.0 * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]);
                let dy_next = state[1] + h / 6.0 * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]);

                y.push(y_next);
                dy.push(dy_next);
            }

            // Check if we hit the right boundary condition
            let error = y[self.n_points - 1] - right;

            if error.abs() < self.tolerance {
                // Copy solution to output
                for i in 0..self.n_points {
                    y_solution[i] = y[i];
                }

                // Apply isotonic constraint
                if self.increasing {
                    for i in 1..self.n_points {
                        if y_solution[i] < y_solution[i - 1] {
                            y_solution[i] = y_solution[i - 1];
                        }
                    }
                } else {
                    for i in 1..self.n_points {
                        if y_solution[i] > y_solution[i - 1] {
                            y_solution[i] = y_solution[i - 1];
                        }
                    }
                }

                return Ok((x_grid, y_solution));
            }

            // Adjust slope guess using secant method
            let derivative_approx = if iter == 0 { 1.0 } else { error / h };
            slope_guess -= self.relaxation * error / (derivative_approx + 1e-10);
        }

        Err(SklearsError::ConvergenceError {
            iterations: self.max_iterations,
        })
    }
}

/// Variational Formulation for Isotonic Problems
///
/// Solves isotonic regression problems using variational principles and
/// energy minimization with monotonicity constraints.
#[derive(Debug, Clone)]
pub struct VariationalIsotonicFormulation {
    /// Number of basis functions
    n_basis: usize,
    /// Regularization parameter
    regularization: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Convergence tolerance
    tolerance: f64,
    /// Whether solution must be increasing
    increasing: bool,
}

impl Default for VariationalIsotonicFormulation {
    fn default() -> Self {
        Self {
            n_basis: 20,
            regularization: 1e-3,
            max_iterations: 1000,
            tolerance: 1e-6,
            increasing: true,
        }
    }
}

impl VariationalIsotonicFormulation {
    /// Create a new variational formulation solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of basis functions
    pub fn n_basis(mut self, n: usize) -> Self {
        self.n_basis = n;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set monotonicity
    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Solve using Galerkin method with monotonicity constraints
    ///
    /// Minimizes: ∫(u' - f)² dx + λ∫(u'')² dx subject to monotonicity
    pub fn solve_galerkin(
        &self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
    ) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
        let n = x_data.len();
        if n != y_data.len() {
            return Err(SklearsError::InvalidInput(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        // Create basis functions (Legendre polynomials)
        let basis_matrix = self.create_basis_matrix(x_data)?;

        // Solve least squares with regularization
        let mut coefficients = self.solve_regularized_least_squares(&basis_matrix, y_data)?;

        // Iteratively project to satisfy monotonicity
        for _ in 0..self.max_iterations {
            let old_coefficients = coefficients.clone();

            // Reconstruct solution
            let solution = basis_matrix.dot(&coefficients);

            // Apply isotonic projection
            let projected_solution = self.project_isotonic(&solution)?;

            // Solve least squares again with projected target
            coefficients =
                self.solve_regularized_least_squares(&basis_matrix, &projected_solution)?;

            // Check convergence
            let diff: f64 = coefficients
                .iter()
                .zip(old_coefficients.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        let final_solution = basis_matrix.dot(&coefficients);

        Ok((x_data.clone(), final_solution))
    }

    /// Create basis function matrix
    fn create_basis_matrix(&self, x: &Array1<f64>) -> Result<Array2<f64>, SklearsError> {
        let n = x.len();
        let mut basis = Array2::zeros((n, self.n_basis));

        // Normalize x to [-1, 1] for Legendre polynomials
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_range = x_max - x_min;

        for i in 0..n {
            let x_norm = 2.0 * (x[i] - x_min) / x_range - 1.0;

            for j in 0..self.n_basis {
                basis[[i, j]] = self.legendre_polynomial(j, x_norm);
            }
        }

        Ok(basis)
    }

    /// Compute Legendre polynomial of degree n at point x
    fn legendre_polynomial(&self, n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => x,
            _ => {
                let mut p0 = 1.0;
                let mut p1 = x;
                let mut p2;

                for k in 2..=n {
                    p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                p1
            }
        }
    }

    /// Solve regularized least squares
    fn solve_regularized_least_squares(
        &self,
        basis: &Array2<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        // Compute A^T A + λI
        let ata = basis.t().dot(basis);
        let mut regularized = ata.clone();

        for i in 0..self.n_basis {
            regularized[[i, i]] += self.regularization;
        }

        // Compute A^T b
        let atb = basis.t().dot(target);

        // Solve using simple Gaussian elimination with partial pivoting
        self.solve_linear_system(&regularized, &atb)
    }

    /// Simple linear system solver
    fn solve_linear_system(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        // Create augmented matrix [A|b]
        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination with partial pivoting
        for i in 0..n {
            // Find pivot
            let mut max_row = i;
            for k in i + 1..n {
                if aug[[k, i]].abs() > aug[[max_row, i]].abs() {
                    max_row = k;
                }
            }

            // Swap rows
            if max_row != i {
                for j in 0..=n {
                    let temp = aug[[i, j]];
                    aug[[i, j]] = aug[[max_row, j]];
                    aug[[max_row, j]] = temp;
                }
            }

            // Check for singularity
            if aug[[i, i]].abs() < 1e-10 {
                return Err(SklearsError::NumericalError(
                    "Singular matrix in linear solve".to_string(),
                ));
            }

            // Eliminate
            for k in i + 1..n {
                let factor = aug[[k, i]] / aug[[i, i]];
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in i + 1..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }

    /// Apply isotonic projection
    fn project_isotonic(&self, y: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut projected = y.clone();

        if self.increasing {
            for i in 1..projected.len() {
                if projected[i] < projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        } else {
            for i in 1..projected.len() {
                if projected[i] > projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        }

        Ok(projected)
    }
}

/// Finite Element Method for Isotonic Problems
///
/// Solves isotonic regression using finite element discretization with
/// piecewise linear or higher-order basis functions.
#[derive(Debug, Clone)]
pub struct FiniteElementIsotonic {
    /// Number of elements
    n_elements: usize,
    /// Polynomial degree of basis functions
    polynomial_degree: usize,
    /// Regularization parameter
    regularization: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Whether solution must be increasing
    increasing: bool,
}

impl Default for FiniteElementIsotonic {
    fn default() -> Self {
        Self {
            n_elements: 50,
            polynomial_degree: 1,
            regularization: 1e-4,
            max_iterations: 1000,
            tolerance: 1e-6,
            increasing: true,
        }
    }
}

impl FiniteElementIsotonic {
    /// Create a new finite element solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set number of elements
    pub fn n_elements(mut self, n: usize) -> Self {
        self.n_elements = n;
        self
    }

    /// Set polynomial degree
    pub fn polynomial_degree(mut self, deg: usize) -> Self {
        self.polynomial_degree = deg;
        self
    }

    /// Set regularization
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set monotonicity
    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Solve using finite element method with piecewise linear elements
    pub fn solve_fem(
        &self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = x_data.len();
        if n != y_data.len() {
            return Err(SklearsError::InvalidInput(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        let n_nodes = self.n_elements + 1;

        // Create mesh
        let x_min = x_data.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x_data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let h = (x_max - x_min) / self.n_elements as f64;

        let mut mesh = Array1::zeros(n_nodes);
        for i in 0..n_nodes {
            mesh[i] = x_min + i as f64 * h;
        }

        // Build stiffness matrix and load vector
        let (stiffness, load) = self.assemble_system(x_data, y_data, &mesh)?;

        // Solve with isotonic constraint projection
        let mut solution = Array1::zeros(n_nodes);

        for _ in 0..self.max_iterations {
            let old_solution = solution.clone();

            // Solve linear system
            solution = self.solve_fem_system(&stiffness, &load)?;

            // Apply isotonic projection
            if self.increasing {
                for i in 1..n_nodes {
                    if solution[i] < solution[i - 1] {
                        solution[i] = solution[i - 1];
                    }
                }
            } else {
                for i in 1..n_nodes {
                    if solution[i] > solution[i - 1] {
                        solution[i] = solution[i - 1];
                    }
                }
            }

            // Check convergence
            let diff: f64 = solution
                .iter()
                .zip(old_solution.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        // Interpolate solution to original data points
        let interpolated = self.interpolate_solution(x_data, &mesh, &solution)?;

        Ok(interpolated)
    }

    /// Assemble finite element system
    fn assemble_system(
        &self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
        mesh: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>), SklearsError> {
        let n_nodes = mesh.len();
        let mut stiffness = Array2::zeros((n_nodes, n_nodes));
        let mut load = Array1::zeros(n_nodes);

        let h = mesh[1] - mesh[0];

        // Build stiffness matrix for piecewise linear elements
        // K_ij = ∫ φ_i' φ_j' dx + λ ∫ φ_i φ_j dx
        for i in 0..n_nodes {
            // Diagonal entries
            if i == 0 || i == n_nodes - 1 {
                stiffness[[i, i]] = 1.0 / h + self.regularization * h / 3.0;
            } else {
                stiffness[[i, i]] = 2.0 / h + self.regularization * 2.0 * h / 3.0;
            }

            // Off-diagonal entries
            if i > 0 {
                stiffness[[i, i - 1]] = -1.0 / h + self.regularization * h / 6.0;
                stiffness[[i - 1, i]] = -1.0 / h + self.regularization * h / 6.0;
            }
        }

        // Build load vector by projecting data onto basis functions
        for (x, y) in x_data.iter().zip(y_data.iter()) {
            // Find which element contains x
            let elem_idx = ((x - mesh[0]) / h).floor() as usize;
            let elem_idx = elem_idx.min(n_nodes - 2);

            // Local coordinates
            let xi = (x - mesh[elem_idx]) / h;

            // Contribution to load vector
            load[elem_idx] += (1.0 - xi) * y;
            load[elem_idx + 1] += xi * y;
        }

        Ok((stiffness, load))
    }

    /// Solve FEM linear system
    fn solve_fem_system(
        &self,
        stiffness: &Array2<f64>,
        load: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = stiffness.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = stiffness[[i, j]];
            }
            aug[[i, n]] = load[i];
        }

        // Gaussian elimination
        for i in 0..n {
            let pivot = aug[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err(SklearsError::NumericalError(
                    "Singular stiffness matrix".to_string(),
                ));
            }

            for k in i + 1..n {
                let factor = aug[[k, i]] / pivot;
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in i + 1..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }

    /// Interpolate FEM solution to data points
    fn interpolate_solution(
        &self,
        x_data: &Array1<f64>,
        mesh: &Array1<f64>,
        solution: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let h = mesh[1] - mesh[0];
        let n_nodes = mesh.len();
        let mut interpolated = Array1::zeros(x_data.len());

        for (i, x) in x_data.iter().enumerate() {
            let elem_idx = ((x - mesh[0]) / h).floor() as usize;
            let elem_idx = elem_idx.min(n_nodes - 2);

            let xi = (x - mesh[elem_idx]) / h;
            interpolated[i] = (1.0 - xi) * solution[elem_idx] + xi * solution[elem_idx + 1];
        }

        Ok(interpolated)
    }
}

/// Spectral Methods for Isotonic Problems
///
/// Solves isotonic regression using spectral methods with Fourier or Chebyshev
/// basis functions and monotonicity constraints.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SpectralBasisType {
    /// Fourier basis functions
    Fourier,
    /// Chebyshev polynomials
    Chebyshev,
    /// Legendre polynomials
    Legendre,
}

#[derive(Debug, Clone)]
pub struct SpectralMethodIsotonic {
    /// Type of spectral basis
    basis_type: SpectralBasisType,
    /// Number of basis functions
    n_modes: usize,
    /// Regularization parameter
    regularization: f64,
    /// Maximum iterations
    max_iterations: usize,
    /// Tolerance
    tolerance: f64,
    /// Whether solution must be increasing
    increasing: bool,
}

impl Default for SpectralMethodIsotonic {
    fn default() -> Self {
        Self {
            basis_type: SpectralBasisType::Chebyshev,
            n_modes: 20,
            regularization: 1e-4,
            max_iterations: 1000,
            tolerance: 1e-6,
            increasing: true,
        }
    }
}

impl SpectralMethodIsotonic {
    /// Create a new spectral method solver
    pub fn new() -> Self {
        Self::default()
    }

    /// Set basis type
    pub fn basis_type(mut self, basis: SpectralBasisType) -> Self {
        self.basis_type = basis;
        self
    }

    /// Set number of modes
    pub fn n_modes(mut self, n: usize) -> Self {
        self.n_modes = n;
        self
    }

    /// Set regularization
    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = max_iter;
        self
    }

    /// Set tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tolerance = tol;
        self
    }

    /// Set monotonicity
    pub fn increasing(mut self, inc: bool) -> Self {
        self.increasing = inc;
        self
    }

    /// Solve using spectral collocation with monotonicity constraints
    pub fn solve_spectral(
        &self,
        x_data: &Array1<f64>,
        y_data: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = x_data.len();
        if n != y_data.len() {
            return Err(SklearsError::InvalidInput(
                "x_data and y_data must have same length".to_string(),
            ));
        }

        // Create spectral basis matrix
        let basis = self.create_spectral_basis(x_data)?;

        // Solve with iterative isotonic projection
        let mut coefficients = Array1::zeros(self.n_modes);

        for _ in 0..self.max_iterations {
            let old_coefficients = coefficients.clone();

            // Reconstruct solution
            let solution = basis.dot(&coefficients);

            // Apply isotonic projection
            let projected = self.project_isotonic(&solution)?;

            // Solve least squares with regularization
            coefficients = self.solve_spectral_least_squares(&basis, &projected)?;

            // Check convergence
            let diff: f64 = coefficients
                .iter()
                .zip(old_coefficients.iter())
                .map(|(new, old)| (new - old).powi(2))
                .sum();

            if diff.sqrt() < self.tolerance {
                break;
            }
        }

        let final_solution = basis.dot(&coefficients);

        Ok(final_solution)
    }

    /// Create spectral basis matrix
    fn create_spectral_basis(&self, x: &Array1<f64>) -> Result<Array2<f64>, SklearsError> {
        let n = x.len();
        let mut basis = Array2::zeros((n, self.n_modes));

        // Normalize x to appropriate domain
        let x_min = x.iter().cloned().fold(f64::INFINITY, f64::min);
        let x_max = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let x_range = x_max - x_min;

        for i in 0..n {
            let x_norm = match self.basis_type {
                SpectralBasisType::Fourier => 2.0 * PI * (x[i] - x_min) / x_range,
                _ => 2.0 * (x[i] - x_min) / x_range - 1.0, // [-1, 1] for Chebyshev/Legendre
            };

            for j in 0..self.n_modes {
                basis[[i, j]] = match self.basis_type {
                    SpectralBasisType::Fourier => {
                        if j % 2 == 0 {
                            (j as f64 / 2.0 * x_norm).cos()
                        } else {
                            ((j as f64 + 1.0) / 2.0 * x_norm).sin()
                        }
                    }
                    SpectralBasisType::Chebyshev => self.chebyshev_polynomial(j, x_norm),
                    SpectralBasisType::Legendre => self.legendre_polynomial(j, x_norm),
                };
            }
        }

        Ok(basis)
    }

    /// Compute Chebyshev polynomial
    fn chebyshev_polynomial(&self, n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => x,
            _ => {
                let mut t0 = 1.0;
                let mut t1 = x;
                let mut t2;

                for _ in 2..=n {
                    t2 = 2.0 * x * t1 - t0;
                    t0 = t1;
                    t1 = t2;
                }
                t1
            }
        }
    }

    /// Compute Legendre polynomial
    fn legendre_polynomial(&self, n: usize, x: f64) -> f64 {
        match n {
            0 => 1.0,
            1 => x,
            _ => {
                let mut p0 = 1.0;
                let mut p1 = x;
                let mut p2;

                for k in 2..=n {
                    p2 = ((2 * k - 1) as f64 * x * p1 - (k - 1) as f64 * p0) / k as f64;
                    p0 = p1;
                    p1 = p2;
                }
                p1
            }
        }
    }

    /// Solve spectral least squares with regularization
    fn solve_spectral_least_squares(
        &self,
        basis: &Array2<f64>,
        target: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = basis.nrows();
        let m = basis.ncols();

        // Compute B^T B + λI
        let mut btb = Array2::zeros((m, m));
        for i in 0..m {
            for j in 0..m {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += basis[[k, i]] * basis[[k, j]];
                }
                btb[[i, j]] = sum;
            }
        }

        for i in 0..m {
            btb[[i, i]] += self.regularization;
        }

        // Compute B^T y
        let bty = basis.t().dot(target);

        // Solve using Cholesky decomposition (assuming positive definite)
        self.solve_linear_cholesky(&btb, &bty)
    }

    /// Solve using Cholesky decomposition
    fn solve_linear_cholesky(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = a.nrows();
        let mut l = Array2::zeros((n, n));

        // Cholesky decomposition: A = L L^T
        for i in 0..n {
            for j in 0..=i {
                let mut sum = 0.0;
                for k in 0..j {
                    sum += l[[i, k]] * l[[j, k]];
                }

                if i == j {
                    let val = a[[i, i]] - sum;
                    if val <= 0.0 {
                        // Fall back to simple Gaussian elimination
                        return self.solve_linear_gauss(a, b);
                    }
                    l[[i, j]] = val.sqrt();
                } else {
                    l[[i, j]] = (a[[i, j]] - sum) / l[[j, j]];
                }
            }
        }

        // Forward substitution: L y = b
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / l[[i, i]];
        }

        // Back substitution: L^T x = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in i + 1..n {
                sum += l[[j, i]] * x[j];
            }
            x[i] = (y[i] - sum) / l[[i, i]];
        }

        Ok(x)
    }

    /// Fallback linear solver using Gaussian elimination
    fn solve_linear_gauss(
        &self,
        a: &Array2<f64>,
        b: &Array1<f64>,
    ) -> Result<Array1<f64>, SklearsError> {
        let n = a.nrows();
        let mut aug = Array2::zeros((n, n + 1));

        for i in 0..n {
            for j in 0..n {
                aug[[i, j]] = a[[i, j]];
            }
            aug[[i, n]] = b[i];
        }

        // Forward elimination
        for i in 0..n {
            let pivot = aug[[i, i]];
            if pivot.abs() < 1e-10 {
                return Err(SklearsError::NumericalError("Singular matrix".to_string()));
            }

            for k in i + 1..n {
                let factor = aug[[k, i]] / pivot;
                for j in i..=n {
                    aug[[k, j]] -= factor * aug[[i, j]];
                }
            }
        }

        // Back substitution
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = aug[[i, n]];
            for j in i + 1..n {
                sum -= aug[[i, j]] * x[j];
            }
            x[i] = sum / aug[[i, i]];
        }

        Ok(x)
    }

    /// Apply isotonic projection
    fn project_isotonic(&self, y: &Array1<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut projected = y.clone();

        if self.increasing {
            for i in 1..projected.len() {
                if projected[i] < projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        } else {
            for i in 1..projected.len() {
                if projected[i] > projected[i - 1] {
                    projected[i] = projected[i - 1];
                }
            }
        }

        Ok(projected)
    }
}

// ============================================================================
// Function APIs
// ============================================================================

/// Solve isotonic differential equation using finite difference method
pub fn isotonic_differential_equation(
    grid_points: &Array1<f64>,
    boundary_left: f64,
    boundary_right: f64,
    source_term: impl Fn(f64) -> f64,
    increasing: bool,
) -> Result<Array1<f64>, SklearsError> {
    IsotonicDifferentialEquation::new()
        .increasing(increasing)
        .solve_finite_difference(grid_points, boundary_left, boundary_right, source_term)
}

/// Solve monotonic boundary value problem using shooting method
pub fn monotonic_boundary_value_problem(
    left: f64,
    right: f64,
    domain: (f64, f64),
    ode_rhs: impl Fn(f64, &[f64]) -> Vec<f64>,
    increasing: bool,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    MonotonicBoundaryValueProblem::new()
        .increasing(increasing)
        .solve_shooting(left, right, domain, ode_rhs)
}

/// Solve using variational formulation
pub fn variational_isotonic_regression(
    x_data: &Array1<f64>,
    y_data: &Array1<f64>,
    increasing: bool,
) -> Result<(Array1<f64>, Array1<f64>), SklearsError> {
    VariationalIsotonicFormulation::new()
        .increasing(increasing)
        .solve_galerkin(x_data, y_data)
}

/// Solve using finite element method
pub fn finite_element_isotonic_regression(
    x_data: &Array1<f64>,
    y_data: &Array1<f64>,
    increasing: bool,
) -> Result<Array1<f64>, SklearsError> {
    FiniteElementIsotonic::new()
        .increasing(increasing)
        .solve_fem(x_data, y_data)
}

/// Solve using spectral methods
pub fn spectral_isotonic_regression(
    x_data: &Array1<f64>,
    y_data: &Array1<f64>,
    basis_type: SpectralBasisType,
    increasing: bool,
) -> Result<Array1<f64>, SklearsError> {
    SpectralMethodIsotonic::new()
        .basis_type(basis_type)
        .increasing(increasing)
        .solve_spectral(x_data, y_data)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_isotonic_differential_equation() {
        let n = 20;
        let grid = Array1::from_vec((0..n).map(|i| i as f64 / (n - 1) as f64).collect());

        // Solve u'' = 1, u(0) = 0, u(1) = 1 with increasing constraint
        // This gives a concave up parabola that naturally increases
        let source = |_x: f64| 1.0;

        let solution = isotonic_differential_equation(&grid, 0.0, 1.0, source, true).unwrap();

        assert_eq!(solution.len(), n);
        assert_abs_diff_eq!(solution[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(solution[n - 1], 1.0, epsilon = 1e-3);

        // Check monotonicity (with increasing constraint, solution should be non-decreasing)
        for i in 1..n {
            assert!(
                solution[i] >= solution[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_variational_formulation() {
        // Simple test data
        let x = Array1::from_vec(vec![0.0, 0.25, 0.5, 0.75, 1.0]);
        let y = Array1::from_vec(vec![0.0, 0.3, 0.6, 0.8, 1.0]);

        let (x_out, y_out) = variational_isotonic_regression(&x, &y, true).unwrap();

        assert_eq!(x_out.len(), x.len());
        assert_eq!(y_out.len(), y.len());

        // Check monotonicity
        for i in 1..y_out.len() {
            assert!(
                y_out[i] >= y_out[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_finite_element_method() {
        let x = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let y = Array1::from_vec(vec![0.0, 0.2, 0.5, 0.7, 0.9, 1.0]);

        let solution = finite_element_isotonic_regression(&x, &y, true).unwrap();

        assert_eq!(solution.len(), x.len());

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(
                solution[i] >= solution[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_spectral_method_chebyshev() {
        let x = Array1::from_vec((0..10).map(|i| i as f64 / 9.0).collect::<Vec<_>>());
        let y = Array1::from_vec((0..10).map(|i| (i as f64 / 9.0).sqrt()).collect::<Vec<_>>());

        let solution =
            spectral_isotonic_regression(&x, &y, SpectralBasisType::Chebyshev, true).unwrap();

        assert_eq!(solution.len(), x.len());

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(
                solution[i] >= solution[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_spectral_method_fourier() {
        let x = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let y = Array1::from_vec(vec![0.0, 0.1, 0.3, 0.5, 0.7, 1.0]);

        let solution =
            spectral_isotonic_regression(&x, &y, SpectralBasisType::Fourier, true).unwrap();

        assert_eq!(solution.len(), x.len());

        // Check monotonicity
        for i in 1..solution.len() {
            assert!(
                solution[i] >= solution[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_monotonic_bvp_simple() {
        // Test simple second-order ODE: y'' = 0, y(0) = 0, y(1) = 1
        let ode_rhs = |_x: f64, state: &[f64]| vec![state[1], 0.0];

        let result = monotonic_boundary_value_problem(0.0, 1.0, (0.0, 1.0), ode_rhs, true);

        assert!(result.is_ok());
        let (_x_grid, y_solution) = result.unwrap();

        assert_abs_diff_eq!(y_solution[0], 0.0, epsilon = 1e-3);
        assert_abs_diff_eq!(y_solution[y_solution.len() - 1], 1.0, epsilon = 1e-3);

        // Check monotonicity
        for i in 1..y_solution.len() {
            assert!(
                y_solution[i] >= y_solution[i - 1] - 1e-6,
                "Monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_decreasing_constraint() {
        let x = Array1::from_vec(vec![0.0, 0.2, 0.4, 0.6, 0.8, 1.0]);
        let y = Array1::from_vec(vec![1.0, 0.9, 0.7, 0.5, 0.3, 0.0]);

        let solution = finite_element_isotonic_regression(&x, &y, false).unwrap();

        // Check decreasing monotonicity
        for i in 1..solution.len() {
            assert!(
                solution[i] <= solution[i - 1] + 1e-6,
                "Decreasing monotonicity violated at index {}",
                i
            );
        }
    }

    #[test]
    fn test_differential_equation_builder() {
        let solver = IsotonicDifferentialEquation::new()
            .equation_type(DifferentialEquationType::BVP)
            .boundary_condition(BoundaryCondition::Dirichlet)
            .order(2)
            .max_iterations(500)
            .tolerance(1e-5)
            .increasing(true)
            .bounds(Some(0.0), Some(1.0));

        assert_eq!(solver.equation_type, DifferentialEquationType::BVP);
        assert_eq!(solver.boundary_condition, BoundaryCondition::Dirichlet);
        assert_eq!(solver.order, 2);
        assert_eq!(solver.max_iterations, 500);
        assert_abs_diff_eq!(solver.tolerance, 1e-5);
        assert_eq!(solver.increasing, true);
        assert_eq!(solver.lower_bound, Some(0.0));
        assert_eq!(solver.upper_bound, Some(1.0));
    }
}

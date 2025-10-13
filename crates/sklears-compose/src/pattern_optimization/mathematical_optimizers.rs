//! Mathematical optimization solvers for classical programming problems
//!
//! This module provides solvers for well-defined mathematical programming problems including:
//! - Linear programming (LP) with simplex and interior point methods
//! - Quadratic programming (QP) with active set and interior point methods
//! - Nonlinear programming (NLP) with gradient-based and derivative-free methods
//! - Convex optimization with specialized algorithms and guarantees
//! - Integer programming with branch-and-bound and cutting plane methods
//! - Problem transformation and solution post-processing utilities

use std::collections::HashMap;
use std::time::Duration;

use scirs2_core::ndarray::{Array1, Array2};
use crate::core::SklResult;
use super::optimization_core::{ConstraintType, Solution};

/// Mathematical optimizer for classical programming problems
///
/// Coordinates various specialized solvers for different problem types
/// and provides automatic solver selection based on problem characteristics.
#[derive(Debug)]
pub struct MathematicalOptimizer {
    /// Unique optimizer identifier
    pub optimizer_id: String,
    /// Linear programming solvers
    pub linear_solvers: HashMap<String, Box<dyn LinearSolver>>,
    /// Quadratic programming solvers
    pub quadratic_solvers: HashMap<String, Box<dyn QuadraticSolver>>,
    /// Nonlinear programming solvers
    pub nonlinear_solvers: HashMap<String, Box<dyn NonlinearSolver>>,
    /// Convex optimization solvers
    pub convex_solvers: HashMap<String, Box<dyn ConvexSolver>>,
    /// Integer programming solvers
    pub integer_solvers: HashMap<String, Box<dyn IntegerSolver>>,
    /// Automatic solver selection utility
    pub solver_selector: SolverSelector,
    /// Problem transformation utilities
    pub problem_transformer: ProblemTransformer,
    /// Solution post-processing utilities
    pub solution_postprocessor: SolutionPostprocessor,
}

/// Linear programming solver trait
///
/// Defines interface for LP solvers such as simplex, dual simplex,
/// primal-dual interior point, and network flow algorithms.
pub trait LinearSolver: Send + Sync {
    /// Solve linear programming problem
    fn solve_linear(&self, problem: &LinearProblem) -> SklResult<LinearSolution>;

    /// Get solver name/identifier
    fn get_solver_name(&self) -> &str;

    /// Get solver-specific parameters
    fn get_solver_parameters(&self) -> HashMap<String, f64>;

    /// Check if solver supports this problem
    fn supports_problem(&self, problem: &LinearProblem) -> bool;

    /// Estimate solution time
    fn estimate_solve_time(&self, problem: &LinearProblem) -> Duration;
}

/// Quadratic programming solver trait
///
/// Defines interface for QP solvers including active set methods,
/// interior point methods, and conjugate gradient approaches.
pub trait QuadraticSolver: Send + Sync {
    /// Solve quadratic programming problem
    fn solve_quadratic(&self, problem: &QuadraticProblem) -> SklResult<QuadraticSolution>;

    /// Get solver name/identifier
    fn get_solver_name(&self) -> &str;

    /// Check if solver supports this problem
    fn supports_problem(&self, problem: &QuadraticProblem) -> bool;

    /// Get optimality conditions for solution
    fn get_optimality_conditions(&self, solution: &QuadraticSolution) -> OptimalityConditions;
}

/// Nonlinear programming solver trait
///
/// Defines interface for NLP solvers including sequential quadratic programming,
/// trust region methods, and derivative-free optimization.
pub trait NonlinearSolver: Send + Sync {
    /// Solve nonlinear programming problem
    fn solve_nonlinear(&self, problem: &NonlinearProblem) -> SklResult<NonlinearSolution>;

    /// Get solver name/identifier
    fn get_solver_name(&self) -> &str;

    /// Get convergence criteria
    fn get_convergence_criteria(&self) -> ConvergenceCriteria;

    /// Set convergence criteria
    fn set_convergence_criteria(&mut self, criteria: ConvergenceCriteria) -> SklResult<()>;

    /// Check if solver supports constraints
    fn supports_constraints(&self) -> bool;

    /// Check if solver requires derivatives
    fn supports_derivatives(&self) -> bool;
}

/// Convex optimization solver trait
///
/// Defines interface for specialized convex solvers with theoretical guarantees
/// including interior point methods, first-order methods, and conic programming.
pub trait ConvexSolver: Send + Sync {
    /// Solve convex optimization problem
    fn solve_convex(&self, problem: &ConvexProblem) -> SklResult<ConvexSolution>;

    /// Verify problem convexity
    fn verify_convexity(&self, problem: &ConvexProblem) -> SklResult<bool>;

    /// Get duality gap for solution
    fn get_duality_gap(&self, solution: &ConvexSolution) -> SklResult<f64>;

    /// Get convergence guarantees
    fn get_convergence_guarantees(&self) -> ConvergenceGuarantees;
}

/// Integer programming solver trait
///
/// Defines interface for IP solvers including branch-and-bound,
/// cutting plane methods, and heuristic approaches.
pub trait IntegerSolver: Send + Sync {
    /// Solve integer programming problem
    fn solve_integer(&self, problem: &IntegerProblem) -> SklResult<IntegerSolution>;

    /// Get relaxation solution (continuous relaxation)
    fn get_relaxation_solution(&self, problem: &IntegerProblem) -> SklResult<Solution>;

    /// Get branch-and-bound tree structure
    fn get_branch_and_bound_tree(&self) -> SklResult<BranchAndBoundTree>;
}

/// Linear programming problem definition
#[derive(Debug, Clone)]
pub struct LinearProblem {
    /// Objective function coefficients
    pub objective_coefficients: Array1<f64>,
    /// Constraint matrix A in Ax ≤ b
    pub constraint_matrix: Array2<f64>,
    /// Right-hand side bounds
    pub constraint_bounds: Array1<f64>,
    /// Variable bounds (lower, upper)
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Problem type (minimize/maximize)
    pub problem_type: LinearProblemType,
}

/// Linear programming problem type
#[derive(Debug, Clone)]
pub enum LinearProblemType {
    /// Minimization problem
    Minimize,
    /// Maximization problem
    Maximize,
    /// Feasibility problem
    Feasibility,
}

/// Linear programming solution
#[derive(Debug, Clone)]
pub struct LinearSolution {
    /// Primal solution variables
    pub solution: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64,
    /// Dual solution (shadow prices)
    pub dual_solution: Array1<f64>,
    /// Solution status
    pub status: SolutionStatus,
    /// Number of iterations
    pub iterations: u32,
    /// Solution time
    pub solve_time: Duration,
}

/// Quadratic programming problem definition
#[derive(Debug, Clone)]
pub struct QuadraticProblem {
    /// Quadratic term matrix Q in ½x'Qx
    pub quadratic_matrix: Array2<f64>,
    /// Linear coefficients c in c'x
    pub linear_coefficients: Array1<f64>,
    /// Linear constraint matrix
    pub constraint_matrix: Option<Array2<f64>>,
    /// Constraint bounds
    pub constraint_bounds: Option<Array1<f64>>,
    /// Variable bounds
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
}

/// Quadratic programming solution
#[derive(Debug, Clone)]
pub struct QuadraticSolution {
    /// Primal solution variables
    pub solution: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64,
    /// Lagrange multipliers
    pub lagrange_multipliers: Array1<f64>,
    /// Solution status
    pub status: SolutionStatus,
    /// KKT residual measure
    pub kkt_residual: f64,
}

/// Nonlinear programming problem definition
#[derive(Debug, Clone)]
pub struct NonlinearProblem {
    /// Objective function expression
    pub objective_function: String,
    /// Constraint function expressions
    pub constraint_functions: Vec<String>,
    /// Variable bounds
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Initial starting point
    pub initial_point: Array1<f64>,
    /// Problem properties
    pub problem_properties: NonlinearProblemProperties,
}

/// Properties of nonlinear programming problems
#[derive(Debug, Clone)]
pub struct NonlinearProblemProperties {
    /// Whether problem is convex
    pub is_convex: bool,
    /// Whether gradients are available
    pub has_gradients: bool,
    /// Whether Hessians are available
    pub has_hessians: bool,
    /// Types of constraints present
    pub constraint_types: Vec<ConstraintType>,
    /// Smoothness degree of functions
    pub smoothness: SmoothnessDegree,
}

/// Smoothness degree enumeration
#[derive(Debug, Clone)]
pub enum SmoothnessDegree {
    /// Discontinuous functions
    Discontinuous,
    /// Continuous but not differentiable
    Continuous,
    /// Continuously differentiable (C¹)
    C1,
    /// Twice continuously differentiable (C²)
    C2,
    /// Infinitely differentiable (C∞)
    CInfinity,
}

/// Nonlinear programming solution
#[derive(Debug, Clone)]
pub struct NonlinearSolution {
    /// Primal solution variables
    pub solution: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64,
    /// Constraint violation measures
    pub constraint_violations: Array1<f64>,
    /// Lagrange multipliers
    pub lagrange_multipliers: Array1<f64>,
    /// Gradient at solution
    pub gradient: Array1<f64>,
    /// Hessian at solution
    pub hessian: Array2<f64>,
    /// Solution status
    pub status: SolutionStatus,
}

/// Convex optimization problem definition
#[derive(Debug, Clone)]
pub struct ConvexProblem {
    /// Problem dimension
    pub dimension: usize,
    /// Objective function description
    pub objective_function: String,
    /// Constraint descriptions
    pub constraints: Vec<String>,
    /// Variable bounds
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Convexity certificate
    pub convexity_certificate: ConvexityCertificate,
}

/// Certificate verifying problem convexity
#[derive(Debug, Clone)]
pub struct ConvexityCertificate {
    /// Whether convexity is verified
    pub is_verified: bool,
    /// Verification method used
    pub verification_method: String,
    /// Confidence in convexity
    pub confidence: f64,
}

/// Convex optimization solution
#[derive(Debug, Clone)]
pub struct ConvexSolution {
    /// Primal solution variables
    pub solution: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64,
    /// Dual solution
    pub dual_solution: Array1<f64>,
    /// Duality gap
    pub duality_gap: f64,
    /// Solution status
    pub status: SolutionStatus,
    /// Optimality certificate
    pub optimality_certificate: OptimalityCertificate,
}

/// Integer programming problem definition
#[derive(Debug, Clone)]
pub struct IntegerProblem {
    /// Linear objective coefficients
    pub objective_coefficients: Array1<f64>,
    /// Constraint matrix
    pub constraint_matrix: Array2<f64>,
    /// Constraint bounds
    pub constraint_bounds: Array1<f64>,
    /// Variable bounds
    pub variable_bounds: Option<(Array1<f64>, Array1<f64>)>,
    /// Integer variable indices
    pub integer_variables: Vec<usize>,
    /// Binary variable indices
    pub binary_variables: Vec<usize>,
}

/// Integer programming solution
#[derive(Debug, Clone)]
pub struct IntegerSolution {
    /// Primal integer solution
    pub solution: Array1<f64>,
    /// Optimal objective value
    pub objective_value: f64>,
    /// Relaxation objective value
    pub relaxation_bound: f64,
    /// Optimality gap
    pub optimality_gap: f64,
    /// Solution status
    pub status: SolutionStatus,
    /// Number of branch-and-bound nodes
    pub nodes_explored: u64,
}

/// Branch-and-bound tree structure
#[derive(Debug, Clone)]
pub struct BranchAndBoundTree {
    /// Root node of the tree
    pub root: BranchNode,
    /// Active nodes in the tree
    pub active_nodes: Vec<BranchNode>,
    /// Best integer solution found
    pub incumbent: Option<Solution>,
    /// Global lower bound
    pub global_lower_bound: f64,
}

/// Individual node in branch-and-bound tree
#[derive(Debug, Clone)]
pub struct BranchNode {
    /// Node identifier
    pub node_id: u64,
    /// Parent node identifier
    pub parent_id: Option<u64>,
    /// Variable bounds for this node
    pub variable_bounds: (Array1<f64>, Array1<f64>),
    /// Relaxation solution at this node
    pub relaxation_solution: Option<Solution>,
    /// Lower bound at this node
    pub lower_bound: f64,
    /// Whether node is fathomed
    pub is_fathomed: bool,
}

/// Solution status enumeration
#[derive(Debug, Clone)]
pub enum SolutionStatus {
    /// Optimal solution found
    Optimal,
    /// Feasible solution found
    Feasible,
    /// Problem is infeasible
    Infeasible,
    /// Problem is unbounded
    Unbounded,
    /// Numerical difficulties encountered
    NumericalError,
    /// Maximum iterations reached
    IterationLimit,
    /// Maximum time reached
    TimeLimit,
    /// User terminated
    UserTerminated,
}

/// Optimality conditions for solutions
#[derive(Debug, Clone)]
pub struct OptimalityConditions {
    /// Gradient norm (first-order optimality)
    pub gradient_norm: f64,
    /// KKT residual
    pub kkt_residual: f64,
    /// Complementarity gap
    pub complementarity_gap: f64,
    /// Dual feasibility measure
    pub dual_feasibility: f64,
    /// Primal feasibility measure
    pub primal_feasibility: f64,
}

/// Convergence criteria for iterative solvers
#[derive(Debug, Clone)]
pub struct ConvergenceCriteria {
    /// Gradient tolerance
    pub gradient_tolerance: f64,
    /// Step size tolerance
    pub step_tolerance: f64,
    /// Function value tolerance
    pub function_tolerance: f64,
    /// Constraint violation tolerance
    pub constraint_tolerance: f64,
    /// Maximum iterations
    pub max_iterations: u32,
    /// Whether to use relative tolerances
    pub relative_tolerance: bool,
}

/// Optimality certificate for convex problems
#[derive(Debug, Clone)]
pub struct OptimalityCertificate {
    pub is_certified: bool,
    pub certification_method: String,
    pub confidence: f64,
}

/// Convergence guarantees for convex solvers
#[derive(Debug, Clone)]
pub struct ConvergenceGuarantees {
    /// Theoretical convergence rate
    pub convergence_rate: f64,
    /// Number of iterations to ε-optimality
    pub iterations_to_epsilon: f64,
    /// Whether global optimum is guaranteed
    pub global_optimum_guaranteed: bool,
}

/// Automatic solver selector
#[derive(Debug, Default)]
pub struct SolverSelector;

/// Problem transformation utilities
#[derive(Debug, Default)]
pub struct ProblemTransformer;

/// Solution post-processing utilities
#[derive(Debug, Default)]
pub struct SolutionPostprocessor;

impl Default for MathematicalOptimizer {
    fn default() -> Self {
        Self {
            optimizer_id: format!("math_opt_{}", std::time::SystemTime::now()
                .duration_since(std::time::SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_millis()),
            linear_solvers: HashMap::new(),
            quadratic_solvers: HashMap::new(),
            nonlinear_solvers: HashMap::new(),
            convex_solvers: HashMap::new(),
            integer_solvers: HashMap::new(),
            solver_selector: SolverSelector::default(),
            problem_transformer: ProblemTransformer::default(),
            solution_postprocessor: SolutionPostprocessor::default(),
        }
    }
}

impl MathematicalOptimizer {
    /// Create a new mathematical optimizer
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a linear solver
    pub fn register_linear_solver(&mut self, name: String, solver: Box<dyn LinearSolver>) {
        self.linear_solvers.insert(name, solver);
    }

    /// Register a quadratic solver
    pub fn register_quadratic_solver(&mut self, name: String, solver: Box<dyn QuadraticSolver>) {
        self.quadratic_solvers.insert(name, solver);
    }

    /// Register a nonlinear solver
    pub fn register_nonlinear_solver(&mut self, name: String, solver: Box<dyn NonlinearSolver>) {
        self.nonlinear_solvers.insert(name, solver);
    }

    /// Get available solver names by type
    pub fn get_available_solvers(&self) -> HashMap<String, Vec<String>> {
        let mut solvers = HashMap::new();
        solvers.insert("linear".to_string(), self.linear_solvers.keys().cloned().collect());
        solvers.insert("quadratic".to_string(), self.quadratic_solvers.keys().cloned().collect());
        solvers.insert("nonlinear".to_string(), self.nonlinear_solvers.keys().cloned().collect());
        solvers.insert("convex".to_string(), self.convex_solvers.keys().cloned().collect());
        solvers.insert("integer".to_string(), self.integer_solvers.keys().cloned().collect());
        solvers
    }
}

impl Default for ConvergenceCriteria {
    fn default() -> Self {
        Self {
            gradient_tolerance: 1e-6,
            step_tolerance: 1e-8,
            function_tolerance: 1e-9,
            constraint_tolerance: 1e-6,
            max_iterations: 1000,
            relative_tolerance: true,
        }
    }
}

impl Default for OptimalityConditions {
    fn default() -> Self {
        Self {
            gradient_norm: f64::INFINITY,
            kkt_residual: f64::INFINITY,
            complementarity_gap: f64::INFINITY,
            dual_feasibility: f64::INFINITY,
            primal_feasibility: f64::INFINITY,
        }
    }
}
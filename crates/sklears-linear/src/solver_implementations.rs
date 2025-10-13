//! Trait-based Solver Implementations
//!
//! This module implements various optimization solvers that work with the modular framework.
//! All solvers implement the OptimizationSolver trait for consistency and pluggability.

use crate::modular_framework::{
    Objective, ObjectiveData, OptimizationSolver, SolverInfo, SolverRecommendations,
};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Configuration for gradient descent solvers
#[derive(Debug, Clone)]
pub struct GradientDescentConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate (step size)
    pub learning_rate: Float,
    /// Whether to use line search
    pub use_line_search: bool,
    /// Line search parameters
    pub line_search_config: LineSearchConfig,
    /// Whether to enable verbose output
    pub verbose: bool,
}

impl Default for GradientDescentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            use_line_search: false,
            line_search_config: LineSearchConfig::default(),
            verbose: false,
        }
    }
}

/// Configuration for line search
#[derive(Debug, Clone)]
pub struct LineSearchConfig {
    /// Armijo condition parameter (c1)
    pub c1: Float,
    /// Curvature condition parameter (c2, for strong Wolfe conditions)
    pub c2: Float,
    /// Maximum number of line search iterations
    pub max_line_search_iterations: usize,
    /// Initial step size scaling factor
    pub initial_step_scale: Float,
    /// Step size reduction factor
    pub step_reduction_factor: Float,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            max_line_search_iterations: 20,
            initial_step_scale: 1.0,
            step_reduction_factor: 0.5,
        }
    }
}

/// Result from gradient descent optimization
#[derive(Debug, Clone)]
pub struct GradientDescentResult {
    /// Final coefficient values
    pub coefficients: Array1<Float>,
    /// Final objective value
    pub objective_value: Float,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Convergence history
    pub convergence_history: Array1<Float>,
    /// Gradient norm history
    pub gradient_norm_history: Array1<Float>,
    /// Final gradient norm
    pub final_gradient_norm: Float,
}

/// Standard Gradient Descent solver
#[derive(Debug)]
pub struct GradientDescentSolver;

impl OptimizationSolver for GradientDescentSolver {
    type Config = GradientDescentConfig;
    type Result = GradientDescentResult;

    fn solve(
        &self,
        objective: &dyn Objective,
        initial_guess: &Array1<Float>,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        let mut coefficients = initial_guess.clone();
        let mut convergence_history = Vec::new();
        let mut gradient_norm_history = Vec::new();
        let mut converged = false;

        // Create dummy data for objective computation (this is a limitation of the current design)
        // In practice, the objective would need to store its own data
        let dummy_data = ObjectiveData {
            features: Array2::zeros((1, coefficients.len())),
            targets: Array1::zeros(1),
            sample_weights: None,
            metadata: Default::default(),
        };

        for iteration in 0..config.max_iterations {
            // Compute objective value and gradient
            let (obj_value, gradient) = objective.value_and_gradient(&coefficients, &dummy_data)?;
            let gradient_norm = gradient.mapv(|x| x * x).sum().sqrt();

            convergence_history.push(obj_value);
            gradient_norm_history.push(gradient_norm);

            if config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: obj={:.6}, ||grad||={:.6}",
                    iteration, obj_value, gradient_norm
                );
            }

            // Check convergence
            if gradient_norm < config.tolerance {
                converged = true;
                if config.verbose {
                    println!("Converged after {} iterations", iteration);
                }
                break;
            }

            // Compute step size
            let step_size = if config.use_line_search {
                self.line_search(
                    objective,
                    &coefficients,
                    &gradient,
                    &dummy_data,
                    &config.line_search_config,
                )?
            } else {
                config.learning_rate
            };

            // Update coefficients
            coefficients = &coefficients - step_size * &gradient;
        }

        let final_objective = objective.value(&coefficients, &dummy_data)?;
        let final_gradient = objective.gradient(&coefficients, &dummy_data)?;
        let final_gradient_norm = final_gradient.mapv(|x| x * x).sum().sqrt();

        Ok(GradientDescentResult {
            coefficients,
            objective_value: final_objective,
            n_iterations: convergence_history.len(),
            converged,
            convergence_history: Array1::from_vec(convergence_history),
            gradient_norm_history: Array1::from_vec(gradient_norm_history),
            final_gradient_norm,
        })
    }

    fn supports_objective(&self, _objective: &dyn Objective) -> bool {
        true // Gradient descent works with any differentiable objective
    }

    fn name(&self) -> &'static str {
        "GradientDescent"
    }

    fn get_recommendations(&self, data: &ObjectiveData) -> SolverRecommendations {
        let n_samples = data.features.nrows();
        let n_features = data.features.ncols();

        // Heuristic recommendations based on problem size
        let max_iter = if n_samples > 10000 { 100 } else { 1000 };
        let tolerance = if n_features > 1000 { 1e-4 } else { 1e-6 };
        let learning_rate = 1.0 / (n_samples as Float).sqrt();

        SolverRecommendations {
            max_iterations: Some(max_iter),
            tolerance: Some(tolerance),
            step_size: Some(learning_rate),
            use_line_search: Some(n_features > 100),
            notes: vec![
                format!(
                    "Problem size: {} samples, {} features",
                    n_samples, n_features
                ),
                "Consider using line search for better convergence".to_string(),
            ],
        }
    }
}

impl GradientDescentSolver {
    /// Perform backtracking line search to find appropriate step size
    fn line_search(
        &self,
        objective: &dyn Objective,
        x: &Array1<Float>,
        direction: &Array1<Float>,
        data: &ObjectiveData,
        config: &LineSearchConfig,
    ) -> Result<Float> {
        let f0 = objective.value(x, data)?;
        let grad0 = objective.gradient(x, data)?;
        let slope = grad0.dot(direction);

        let mut step_size = config.initial_step_scale;

        for _ in 0..config.max_line_search_iterations {
            let x_new = x - step_size * direction;
            let f_new = objective.value(&x_new, data)?;

            // Armijo condition: f(x + α*p) ≤ f(x) + c1*α*∇f(x)ᵀp
            if f_new <= f0 + config.c1 * step_size * slope {
                return Ok(step_size);
            }

            step_size *= config.step_reduction_factor;
        }

        // If line search fails, return a small step size
        Ok(step_size)
    }
}

/// Configuration for coordinate descent solver
#[derive(Debug, Clone)]
pub struct CoordinateDescentConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to use random coordinate selection
    pub random_selection: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to enable verbose output
    pub verbose: bool,
}

impl Default for CoordinateDescentConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            random_selection: false,
            random_seed: None,
            verbose: false,
        }
    }
}

/// Result from coordinate descent optimization
#[derive(Debug, Clone)]
pub struct CoordinateDescentResult {
    /// Final coefficient values
    pub coefficients: Array1<Float>,
    /// Final objective value
    pub objective_value: Float,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Convergence history
    pub convergence_history: Array1<Float>,
    /// Number of coordinate updates performed
    pub n_coordinate_updates: usize,
}

/// Coordinate Descent solver (good for L1-regularized problems)
#[derive(Debug)]
pub struct CoordinateDescentSolver;

impl OptimizationSolver for CoordinateDescentSolver {
    type Config = CoordinateDescentConfig;
    type Result = CoordinateDescentResult;

    fn solve(
        &self,
        objective: &dyn Objective,
        initial_guess: &Array1<Float>,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        let mut coefficients = initial_guess.clone();
        let n_features = coefficients.len();
        let mut convergence_history = Vec::new();
        let mut converged = false;
        let mut coordinate_updates = 0;

        // Create coordinate selection order
        let coord_order: Vec<usize> = if config.random_selection {
            // TODO: Implement random permutation using random_seed
            (0..n_features).collect()
        } else {
            (0..n_features).collect()
        };

        let dummy_data = ObjectiveData {
            features: Array2::zeros((1, n_features)),
            targets: Array1::zeros(1),
            sample_weights: None,
            metadata: Default::default(),
        };

        for iteration in 0..config.max_iterations {
            let obj_value_start = objective.value(&coefficients, &dummy_data)?;
            let mut max_coordinate_change: f64 = 0.0;

            // Update each coordinate
            for &coord_idx in &coord_order {
                let old_value = coefficients[coord_idx];

                // For coordinate descent, we would typically compute the optimal update
                // for this coordinate. This is a simplified implementation.
                let gradient = objective.gradient(&coefficients, &dummy_data)?;
                let coord_gradient = gradient[coord_idx];

                // Simple gradient step for this coordinate
                let learning_rate = 0.01; // This should be adaptive
                let new_value = old_value - learning_rate * coord_gradient;

                coefficients[coord_idx] = new_value;
                coordinate_updates += 1;

                let change = (new_value - old_value).abs();
                max_coordinate_change = max_coordinate_change.max(change);
            }

            let obj_value_end = objective.value(&coefficients, &dummy_data)?;
            convergence_history.push(obj_value_end);

            if config.verbose && iteration % 100 == 0 {
                println!(
                    "Iteration {}: obj={:.6}, max_change={:.6}",
                    iteration, obj_value_end, max_coordinate_change
                );
            }

            // Check convergence
            if max_coordinate_change < config.tolerance {
                converged = true;
                if config.verbose {
                    println!("Converged after {} iterations", iteration);
                }
                break;
            }
        }

        let final_objective = objective.value(&coefficients, &dummy_data)?;

        Ok(CoordinateDescentResult {
            coefficients,
            objective_value: final_objective,
            n_iterations: convergence_history.len(),
            converged,
            convergence_history: Array1::from_vec(convergence_history),
            n_coordinate_updates: coordinate_updates,
        })
    }

    fn supports_objective(&self, objective: &dyn Objective) -> bool {
        // Coordinate descent works well with separable objectives
        // For L1 regularization, it's particularly effective
        true
    }

    fn name(&self) -> &'static str {
        "CoordinateDescent"
    }

    fn get_recommendations(&self, data: &ObjectiveData) -> SolverRecommendations {
        let n_features = data.features.ncols();

        SolverRecommendations {
            max_iterations: Some(if n_features > 1000 { 100 } else { 1000 }),
            tolerance: Some(1e-6),
            step_size: None, // Not applicable for coordinate descent
            use_line_search: Some(false),
            notes: vec![
                "Coordinate descent is particularly effective for L1-regularized problems"
                    .to_string(),
                "Consider random coordinate selection for large problems".to_string(),
            ],
        }
    }
}

/// Configuration for proximal gradient solver
#[derive(Debug, Clone)]
pub struct ProximalGradientConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Initial step size
    pub initial_step_size: Float,
    /// Whether to use adaptive step size
    pub adaptive_step_size: bool,
    /// Backtracking parameters
    pub backtracking_config: BacktrackingConfig,
    /// Whether to enable verbose output
    pub verbose: bool,
}

impl Default for ProximalGradientConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            initial_step_size: 1.0,
            adaptive_step_size: true,
            backtracking_config: BacktrackingConfig::default(),
            verbose: false,
        }
    }
}

/// Configuration for backtracking in proximal gradient
#[derive(Debug, Clone)]
pub struct BacktrackingConfig {
    /// Backtracking parameter (β)
    pub beta: Float,
    /// Sufficient decrease parameter
    pub sigma: Float,
    /// Maximum backtracking iterations
    pub max_backtrack_iterations: usize,
}

impl Default for BacktrackingConfig {
    fn default() -> Self {
        Self {
            beta: 0.5,
            sigma: 0.01,
            max_backtrack_iterations: 50,
        }
    }
}

/// Result from proximal gradient optimization
#[derive(Debug, Clone)]
pub struct ProximalGradientResult {
    /// Final coefficient values
    pub coefficients: Array1<Float>,
    /// Final objective value
    pub objective_value: Float,
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Convergence history
    pub convergence_history: Array1<Float>,
    /// Step size history
    pub step_size_history: Array1<Float>,
}

/// Proximal Gradient solver (for non-smooth regularization)
#[derive(Debug)]
pub struct ProximalGradientSolver;

impl OptimizationSolver for ProximalGradientSolver {
    type Config = ProximalGradientConfig;
    type Result = ProximalGradientResult;

    fn solve(
        &self,
        objective: &dyn Objective,
        initial_guess: &Array1<Float>,
        config: &Self::Config,
    ) -> Result<Self::Result> {
        // NOTE: This is a simplified implementation. In practice, proximal gradient
        // methods require separating the smooth and non-smooth parts of the objective.
        // The current Objective trait doesn't directly support this separation.

        Err(SklearsError::InvalidOperation(
            "Proximal gradient solver requires objective decomposition not yet implemented"
                .to_string(),
        ))
    }

    fn supports_objective(&self, _objective: &dyn Objective) -> bool {
        // This would check if the objective has a decomposable structure
        false
    }

    fn name(&self) -> &'static str {
        "ProximalGradient"
    }

    fn get_recommendations(&self, data: &ObjectiveData) -> SolverRecommendations {
        SolverRecommendations {
            max_iterations: Some(1000),
            tolerance: Some(1e-6),
            step_size: Some(1.0),
            use_line_search: Some(false),
            notes: vec![
                "Proximal gradient is ideal for problems with non-smooth regularization"
                    .to_string(),
                "Requires objective decomposition into smooth + non-smooth parts".to_string(),
            ],
        }
    }
}

/// Factory for creating solver instances
pub struct SolverFactory;

impl SolverFactory {
    /// Create a gradient descent solver
    pub fn gradient_descent(
    ) -> Box<dyn OptimizationSolver<Config = GradientDescentConfig, Result = GradientDescentResult>>
    {
        Box::new(GradientDescentSolver)
    }

    /// Create a coordinate descent solver
    pub fn coordinate_descent() -> Box<
        dyn OptimizationSolver<Config = CoordinateDescentConfig, Result = CoordinateDescentResult>,
    > {
        Box::new(CoordinateDescentSolver)
    }

    /// Create a proximal gradient solver
    pub fn proximal_gradient(
    ) -> Box<dyn OptimizationSolver<Config = ProximalGradientConfig, Result = ProximalGradientResult>>
    {
        Box::new(ProximalGradientSolver)
    }
}

/// Utility function to convert from framework result to standard format
pub fn convert_solver_result_to_standard(
    result: &dyn std::fmt::Debug,
    solver_name: &str,
) -> crate::modular_framework::OptimizationResult {
    // This is a placeholder for result conversion
    // In practice, each solver result type would implement a conversion trait
    crate::modular_framework::OptimizationResult {
        coefficients: Array1::zeros(1), // Placeholder
        intercept: None,
        objective_value: 0.0,
        n_iterations: 0,
        converged: false,
        solver_info: SolverInfo {
            solver_name: solver_name.to_string(),
            metrics: HashMap::new(),
            warnings: Vec::new(),
            convergence_history: None,
        },
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::loss_functions::SquaredLoss;
    use crate::modular_framework::CompositeObjective;
    use crate::regularization_schemes::L2Regularization;

    // Test helper: create a simple quadratic objective
    fn create_test_objective() -> CompositeObjective<'static> {
        let loss = Box::leak(Box::new(SquaredLoss));
        let reg = Box::leak(Box::new(L2Regularization::new(0.1).unwrap()));
        CompositeObjective::new(loss, Some(reg))
    }

    #[test]
    fn test_gradient_descent_config() {
        let config = GradientDescentConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert_eq!(config.learning_rate, 0.01);
        assert!(!config.use_line_search);
    }

    #[test]
    fn test_coordinate_descent_config() {
        let config = CoordinateDescentConfig::default();
        assert_eq!(config.max_iterations, 1000);
        assert_eq!(config.tolerance, 1e-6);
        assert!(!config.random_selection);
    }

    #[test]
    fn test_solver_names() {
        let gd_solver = GradientDescentSolver;
        assert_eq!(gd_solver.name(), "GradientDescent");

        let cd_solver = CoordinateDescentSolver;
        assert_eq!(cd_solver.name(), "CoordinateDescent");

        let pg_solver = ProximalGradientSolver;
        assert_eq!(pg_solver.name(), "ProximalGradient");
    }

    #[test]
    fn test_solver_factory() {
        let gd = SolverFactory::gradient_descent();
        assert_eq!(gd.name(), "GradientDescent");

        let cd = SolverFactory::coordinate_descent();
        assert_eq!(cd.name(), "CoordinateDescent");

        let pg = SolverFactory::proximal_gradient();
        assert_eq!(pg.name(), "ProximalGradient");
    }

    #[test]
    fn test_solver_recommendations() {
        let solver = GradientDescentSolver;
        let data = ObjectiveData {
            features: Array2::zeros((100, 10)),
            targets: Array1::zeros(100),
            sample_weights: None,
            metadata: Default::default(),
        };

        let recommendations = solver.get_recommendations(&data);
        assert!(recommendations.max_iterations.is_some());
        assert!(recommendations.tolerance.is_some());
        assert!(recommendations.step_size.is_some());
    }

    #[test]
    fn test_line_search_config() {
        let config = LineSearchConfig::default();
        assert_eq!(config.c1, 1e-4);
        assert_eq!(config.c2, 0.9);
        assert_eq!(config.max_line_search_iterations, 20);
    }
}

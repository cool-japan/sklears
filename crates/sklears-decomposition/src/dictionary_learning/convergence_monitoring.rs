//! Convergence monitoring and optimization utilities

use scirs2_core::ndarray::Array1;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::types::Float;

/// Convergence configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceConfig {
    pub tol: Float,
    pub max_iter: usize,
    pub window_size: usize,
}

impl Default for ConvergenceConfig {
    fn default() -> Self {
        Self {
            tol: 1e-6,
            max_iter: 1000,
            window_size: 10,
        }
    }
}

/// Convergence result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceResult {
    pub converged: bool,
    pub n_iter: usize,
    pub final_error: Float,
}

/// Optimization metrics
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationMetrics {
    pub objective_values: Array1<Float>,
    pub gradient_norms: Array1<Float>,
    pub step_sizes: Array1<Float>,
}

/// Learning curve data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LearningCurve {
    pub iterations: Array1<Float>,
    pub training_errors: Array1<Float>,
    pub validation_errors: Option<Array1<Float>>,
}

/// Convergence monitor
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConvergenceMonitor {
    config: ConvergenceConfig,
    error_history: Vec<Float>,
}

impl ConvergenceMonitor {
    pub fn new(config: ConvergenceConfig) -> Self {
        Self {
            config,
            error_history: Vec::new(),
        }
    }

    pub fn check_convergence(&mut self, error: Float, iteration: usize) -> ConvergenceResult {
        self.error_history.push(error);

        let converged = if self.error_history.len() > 1 {
            let prev_error = self.error_history[self.error_history.len() - 2];
            (prev_error - error).abs() < self.config.tol
        } else {
            false
        };

        ConvergenceResult {
            converged: converged || iteration >= self.config.max_iter,
            n_iter: iteration,
            final_error: error,
        }
    }
}

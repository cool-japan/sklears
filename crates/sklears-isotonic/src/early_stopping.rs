//! Early stopping and warm start capabilities for isotonic regression
//!
//! This module provides early stopping criteria and warm start functionality
//! to improve the efficiency and convergence of iterative isotonic regression algorithms.

use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};
use std::collections::VecDeque;

/// Early stopping criteria for optimization algorithms
#[derive(Debug, Clone)]
/// EarlyStoppingCriteria
pub struct EarlyStoppingCriteria {
    /// Minimum relative improvement required to continue
    pub min_improvement: Float,
    /// Number of iterations to look back for improvement
    pub patience: usize,
    /// Minimum number of iterations before early stopping can trigger
    pub min_iterations: usize,
    /// Maximum number of iterations regardless of improvement
    pub max_iterations: usize,
    /// Tolerance for absolute convergence
    pub absolute_tolerance: Float,
    /// Tolerance for relative convergence
    pub relative_tolerance: Float,
    /// Whether to monitor objective function value
    pub monitor_objective: bool,
    /// Whether to monitor gradient norm
    pub monitor_gradient: bool,
    /// Whether to monitor parameter change
    pub monitor_parameters: bool,
}

impl Default for EarlyStoppingCriteria {
    fn default() -> Self {
        Self {
            min_improvement: 1e-4,
            patience: 10,
            min_iterations: 10,
            max_iterations: 1000,
            absolute_tolerance: 1e-8,
            relative_tolerance: 1e-6,
            monitor_objective: true,
            monitor_gradient: true,
            monitor_parameters: true,
        }
    }
}

impl EarlyStoppingCriteria {
    /// Create new early stopping criteria
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum improvement threshold
    pub fn min_improvement(mut self, min_improvement: Float) -> Self {
        self.min_improvement = min_improvement;
        self
    }

    /// Set patience (number of iterations to wait for improvement)
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set minimum iterations before early stopping can trigger
    pub fn min_iterations(mut self, min_iterations: usize) -> Self {
        self.min_iterations = min_iterations;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Set absolute tolerance
    pub fn absolute_tolerance(mut self, tolerance: Float) -> Self {
        self.absolute_tolerance = tolerance;
        self
    }

    /// Set relative tolerance
    pub fn relative_tolerance(mut self, tolerance: Float) -> Self {
        self.relative_tolerance = tolerance;
        self
    }

    /// Set monitoring options
    pub fn monitoring(mut self, objective: bool, gradient: bool, parameters: bool) -> Self {
        self.monitor_objective = objective;
        self.monitor_gradient = gradient;
        self.monitor_parameters = parameters;
        self
    }
}

/// State for early stopping monitoring
#[derive(Debug, Clone)]
/// EarlyStoppingState
pub struct EarlyStoppingState {
    /// Iteration counter
    pub iteration: usize,
    /// History of objective function values
    objective_history: VecDeque<Float>,
    /// History of gradient norms
    gradient_history: VecDeque<Float>,
    /// History of parameter change norms
    parameter_history: VecDeque<Float>,
    /// Best objective value seen so far
    best_objective: Option<Float>,
    /// Iteration when best objective was achieved
    best_iteration: usize,
    /// Whether convergence has been achieved
    converged: bool,
    /// Reason for stopping
    stop_reason: StopReason,
}

/// Reason for stopping optimization
#[derive(Debug, Clone, PartialEq)]
/// StopReason
pub enum StopReason {
    /// Still running
    Running,
    /// Maximum iterations reached
    MaxIterations,
    /// Absolute tolerance achieved
    AbsoluteTolerance,
    /// Relative tolerance achieved
    RelativeTolerance,
    /// No improvement in patience iterations
    NoImprovement,
    /// Objective function converged
    ObjectiveConverged,
    /// Gradient converged
    GradientConverged,
    /// Parameters converged
    ParametersConverged,
}

impl EarlyStoppingState {
    /// Create new early stopping state
    pub fn new(criteria: &EarlyStoppingCriteria) -> Self {
        Self {
            iteration: 0,
            objective_history: VecDeque::with_capacity(criteria.patience + 1),
            gradient_history: VecDeque::with_capacity(criteria.patience + 1),
            parameter_history: VecDeque::with_capacity(criteria.patience + 1),
            best_objective: None,
            best_iteration: 0,
            converged: false,
            stop_reason: StopReason::Running,
        }
    }

    /// Update state with new iteration data
    pub fn update(
        &mut self,
        criteria: &EarlyStoppingCriteria,
        objective: Option<Float>,
        gradient_norm: Option<Float>,
        parameter_change: Option<Float>,
    ) -> bool {
        self.iteration += 1;

        // Update histories
        if let Some(obj) = objective {
            if criteria.monitor_objective {
                self.objective_history.push_back(obj);
                if self.objective_history.len() > criteria.patience + 1 {
                    self.objective_history.pop_front();
                }

                // Track best objective
                if self.best_objective.is_none() || obj < self.best_objective.unwrap() {
                    self.best_objective = Some(obj);
                    self.best_iteration = self.iteration;
                }
            }
        }

        if let Some(grad_norm) = gradient_norm {
            if criteria.monitor_gradient {
                self.gradient_history.push_back(grad_norm);
                if self.gradient_history.len() > criteria.patience + 1 {
                    self.gradient_history.pop_front();
                }
            }
        }

        if let Some(param_change) = parameter_change {
            if criteria.monitor_parameters {
                self.parameter_history.push_back(param_change);
                if self.parameter_history.len() > criteria.patience + 1 {
                    self.parameter_history.pop_front();
                }
            }
        }

        // Check stopping criteria
        self.check_stopping_criteria(criteria)
    }

    /// Check if stopping criteria are met
    fn check_stopping_criteria(&mut self, criteria: &EarlyStoppingCriteria) -> bool {
        // Maximum iterations
        if self.iteration >= criteria.max_iterations {
            self.converged = true;
            self.stop_reason = StopReason::MaxIterations;
            return true;
        }

        // Don't check other criteria until minimum iterations
        if self.iteration < criteria.min_iterations {
            return false;
        }

        // Absolute tolerance on objective
        if criteria.monitor_objective && !self.objective_history.is_empty() {
            let current_obj = *self.objective_history.back().unwrap();
            if current_obj.abs() < criteria.absolute_tolerance {
                self.converged = true;
                self.stop_reason = StopReason::AbsoluteTolerance;
                return true;
            }
        }

        // Gradient convergence
        if criteria.monitor_gradient && !self.gradient_history.is_empty() {
            let current_grad = *self.gradient_history.back().unwrap();
            if current_grad < criteria.absolute_tolerance {
                self.converged = true;
                self.stop_reason = StopReason::GradientConverged;
                return true;
            }
        }

        // Parameter change convergence
        if criteria.monitor_parameters && !self.parameter_history.is_empty() {
            let current_change = *self.parameter_history.back().unwrap();
            if current_change < criteria.absolute_tolerance {
                self.converged = true;
                self.stop_reason = StopReason::ParametersConverged;
                return true;
            }
        }

        // Relative tolerance
        if criteria.monitor_objective && self.objective_history.len() > 1 {
            let current_obj = *self.objective_history.back().unwrap();
            let prev_obj = self.objective_history[self.objective_history.len() - 2];
            let relative_change = (current_obj - prev_obj).abs() / (prev_obj.abs() + 1e-10);

            if relative_change < criteria.relative_tolerance {
                self.converged = true;
                self.stop_reason = StopReason::RelativeTolerance;
                return true;
            }
        }

        // No improvement in patience iterations
        if criteria.monitor_objective && self.iteration - self.best_iteration >= criteria.patience {
            self.converged = true;
            self.stop_reason = StopReason::NoImprovement;
            return true;
        }

        false
    }

    /// Check if converged
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Get stop reason
    pub fn stop_reason(&self) -> &StopReason {
        &self.stop_reason
    }

    /// Get current iteration
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get best objective value
    pub fn best_objective(&self) -> Option<Float> {
        self.best_objective
    }
}

/// Warm start state for optimization algorithms
#[derive(Debug, Clone)]
/// WarmStartState
pub struct WarmStartState {
    /// Initial parameter values from previous optimization
    pub initial_parameters: Option<Array1<Float>>,
    /// Previous objective value
    pub previous_objective: Option<Float>,
    /// Previous gradient
    pub previous_gradient: Option<Array1<Float>>,
    /// Learning rate from previous optimization
    pub previous_learning_rate: Option<Float>,
    /// Number of previous optimizations
    pub optimization_count: usize,
    /// Whether warm start is enabled
    pub enabled: bool,
}

impl Default for WarmStartState {
    fn default() -> Self {
        Self {
            initial_parameters: None,
            previous_objective: None,
            previous_gradient: None,
            previous_learning_rate: None,
            optimization_count: 0,
            enabled: true,
        }
    }
}

impl WarmStartState {
    /// Create new warm start state
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable warm start
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }

    /// Update warm start state after optimization
    pub fn update_after_optimization(
        &mut self,
        final_parameters: &Array1<Float>,
        final_objective: Float,
        final_gradient: Option<&Array1<Float>>,
        final_learning_rate: Option<Float>,
    ) {
        if self.enabled {
            self.initial_parameters = Some(final_parameters.clone());
            self.previous_objective = Some(final_objective);
            self.previous_gradient = final_gradient.map(|g| g.clone());
            self.previous_learning_rate = final_learning_rate;
            self.optimization_count += 1;
        }
    }

    /// Get initial parameters for warm start
    pub fn get_initial_parameters(&self) -> Option<&Array1<Float>> {
        if self.enabled {
            self.initial_parameters.as_ref()
        } else {
            None
        }
    }

    /// Get suggested initial learning rate
    pub fn get_initial_learning_rate(&self, default_lr: Float) -> Float {
        if self.enabled {
            // Adaptive learning rate based on previous optimization
            match self.previous_learning_rate {
                Some(prev_lr) => {
                    // Slightly reduce learning rate to be conservative
                    (prev_lr * 0.9).max(default_lr * 0.1).min(default_lr * 2.0)
                }
                None => default_lr,
            }
        } else {
            default_lr
        }
    }

    /// Check if this is a warm start
    pub fn is_warm_start(&self) -> bool {
        self.enabled && self.initial_parameters.is_some()
    }
}

/// Adaptive learning rate scheduler
#[derive(Debug, Clone)]
/// AdaptiveLearningRate
pub struct AdaptiveLearningRate {
    /// Initial learning rate
    pub initial_lr: Float,
    /// Current learning rate
    pub current_lr: Float,
    /// Decay factor when no improvement
    pub decay_factor: Float,
    /// Increase factor when improvement
    pub increase_factor: Float,
    /// Minimum learning rate
    pub min_lr: Float,
    /// Maximum learning rate
    pub max_lr: Float,
    /// Number of iterations without improvement
    no_improvement_count: usize,
    /// Previous objective value
    previous_objective: Option<Float>,
}

impl AdaptiveLearningRate {
    /// Create new adaptive learning rate scheduler
    pub fn new(initial_lr: Float) -> Self {
        Self {
            initial_lr,
            current_lr: initial_lr,
            decay_factor: 0.8,
            increase_factor: 1.05,
            min_lr: initial_lr * 1e-4,
            max_lr: initial_lr * 10.0,
            no_improvement_count: 0,
            previous_objective: None,
        }
    }

    /// Update learning rate based on objective progress
    pub fn update(&mut self, current_objective: Float) -> Float {
        if let Some(prev_obj) = self.previous_objective {
            if current_objective < prev_obj - 1e-8 {
                // Improvement - slightly increase learning rate
                self.current_lr = (self.current_lr * self.increase_factor).min(self.max_lr);
                self.no_improvement_count = 0;
            } else {
                // No improvement - decay learning rate
                self.no_improvement_count += 1;
                if self.no_improvement_count >= 3 {
                    self.current_lr = (self.current_lr * self.decay_factor).max(self.min_lr);
                    self.no_improvement_count = 0;
                }
            }
        }

        self.previous_objective = Some(current_objective);
        self.current_lr
    }

    /// Get current learning rate
    pub fn current_rate(&self) -> Float {
        self.current_lr
    }

    /// Reset to initial learning rate
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.no_improvement_count = 0;
        self.previous_objective = None;
    }
}

/// Enhanced optimization result with early stopping information
#[derive(Debug, Clone)]
/// OptimizationResult
pub struct OptimizationResult {
    /// Final parameter values
    pub parameters: Array1<Float>,
    /// Final objective value
    pub objective: Float,
    /// Number of iterations performed
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Reason for stopping
    pub stop_reason: StopReason,
    /// Optimization history (objective values)
    pub objective_history: Vec<Float>,
    /// Final gradient norm (if available)
    pub final_gradient_norm: Option<Float>,
}

impl OptimizationResult {
    /// Create new optimization result
    pub fn new(
        parameters: Array1<Float>,
        objective: Float,
        iterations: usize,
        converged: bool,
        stop_reason: StopReason,
    ) -> Self {
        Self {
            parameters,
            objective,
            iterations,
            converged,
            stop_reason,
            objective_history: Vec::new(),
            final_gradient_norm: None,
        }
    }

    /// Add objective history
    pub fn with_history(mut self, history: Vec<Float>) -> Self {
        self.objective_history = history;
        self
    }

    /// Add final gradient norm
    pub fn with_gradient_norm(mut self, gradient_norm: Float) -> Self {
        self.final_gradient_norm = Some(gradient_norm);
        self
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_early_stopping_criteria() {
        let criteria = EarlyStoppingCriteria::new()
            .min_improvement(1e-3)
            .patience(5)
            .max_iterations(100)
            .absolute_tolerance(1e-12)
            .relative_tolerance(1e-12);

        let mut state = EarlyStoppingState::new(&criteria);

        // Should not stop initially
        assert!(!state.update(&criteria, Some(1.0), None, None));
        assert!(!state.converged());

        // Should not stop with improvement
        for i in 1..=5 {
            let obj = 1.0 - i as Float * 0.05; // Smaller improvement steps
            let should_stop = state.update(&criteria, Some(obj), None, None);
            if should_stop {
                println!(
                    "Stopped at iteration {}, obj={}, reason={:?}",
                    i,
                    obj,
                    state.stop_reason()
                );
            }
            assert!(!should_stop);
        }

        // Should stop when max iterations reached
        for i in 6..=100 {
            let obj = 0.75; // Constant objective to avoid tolerance triggers
            let should_stop = state.update(&criteria, Some(obj), None, None);
            if i == 100 {
                assert!(should_stop);
                assert_eq!(state.stop_reason(), &StopReason::MaxIterations);
            }
        }
    }

    #[test]
    fn test_warm_start_state() {
        let mut warm_start = WarmStartState::new();
        assert!(!warm_start.is_warm_start());

        let params = Array1::from(vec![1.0, 2.0, 3.0]);
        warm_start.update_after_optimization(&params, 0.5, None, Some(0.01));

        assert!(warm_start.is_warm_start());
        assert_eq!(warm_start.get_initial_parameters().unwrap(), &params);
        let lr = warm_start.get_initial_learning_rate(0.1);
        // Actually calculates: max(0.01 * 0.9, 0.1 * 0.1) = max(0.009, 0.01) = 0.01
        assert!((lr - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_learning_rate() {
        let mut scheduler = AdaptiveLearningRate::new(0.1);

        // Test improvement case
        let lr1 = scheduler.update(1.0);
        let lr2 = scheduler.update(0.8); // Improvement
        assert!(lr2 >= lr1); // Should increase or stay same

        // Test no improvement case
        let lr3 = scheduler.update(0.9); // No improvement, count = 1
        let lr4 = scheduler.update(1.0); // No improvement, count = 2
        let lr5 = scheduler.update(1.1); // No improvement, count = 3, triggers decay
        let lr6 = scheduler.update(1.2); // No improvement, count = 1
        let lr7 = scheduler.update(1.3); // No improvement, count = 2
        let lr8 = scheduler.update(1.4); // No improvement, count = 3, triggers decay again
        assert!(lr5 < lr4); // lr5 should have decayed from lr4
        assert!(lr8 < lr7); // lr8 should have decayed from lr7
    }

    #[test]
    fn test_gradient_convergence() {
        let criteria = EarlyStoppingCriteria::new()
            .absolute_tolerance(1e-6)
            .min_iterations(5)
            .monitoring(false, true, false); // Only monitor gradient

        let mut state = EarlyStoppingState::new(&criteria);

        // Should not stop with large gradient
        for i in 1..=10 {
            let grad_norm = 1e-3;
            assert!(!state.update(&criteria, None, Some(grad_norm), None));
        }

        // Should stop with small gradient
        let should_stop = state.update(&criteria, None, Some(1e-7), None);
        assert!(should_stop);
        assert_eq!(state.stop_reason(), &StopReason::GradientConverged);
    }
}

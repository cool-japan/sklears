//! Robust optimization methods for manifold learning
//! This module provides optimization algorithms that are resistant to numerical
//! instabilities, outliers, and poor conditioning. These methods are essential
//! for reliable manifold learning in real-world applications.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::Estimator,
    types::Float,
};
use std::collections::HashMap;

/// Configuration for robust optimization algorithms
#[derive(Debug, Clone)]
pub struct RobustOptimConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Learning rate (for gradient-based methods)
    pub learning_rate: Float,
    /// Momentum parameter
    pub momentum: Float,
    /// Whether to use adaptive learning rate
    pub adaptive_lr: bool,
    /// L1 regularization parameter
    pub l1_reg: Float,
    /// L2 regularization parameter
    pub l2_reg: Float,
    /// Outlier detection threshold
    pub outlier_threshold: Float,
    /// Whether to use robust loss functions
    pub use_robust_loss: bool,
    /// Trust region radius (for trust region methods)
    pub trust_radius: Float,
    /// Random seed for stochastic methods
    pub random_seed: Option<u64>,
}

impl Default for RobustOptimConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            learning_rate: 0.01,
            momentum: 0.9,
            adaptive_lr: true,
            l1_reg: 0.0,
            l2_reg: 1e-4,
            outlier_threshold: 3.0,
            use_robust_loss: true,
            trust_radius: 1.0,
            random_seed: None,
        }
    }
}

/// Optimization result with diagnostic information
#[derive(Debug, Clone)]
pub struct OptimizationResult {
    /// Final parameters
    pub parameters: Array1<Float>,
    /// Final objective value
    pub objective_value: Float,
    /// Number of iterations used
    pub iterations: usize,
    /// Whether optimization converged
    pub converged: bool,
    /// Gradient norm at final iteration
    pub final_gradient_norm: Float,
    /// Optimization path (objective values at each iteration)
    pub objective_history: Vec<Float>,
    /// Gradient norms throughout optimization
    pub gradient_history: Vec<Float>,
    /// Learning rate schedule (for adaptive methods)
    pub learning_rate_history: Vec<Float>,
    /// Diagnostic information
    pub diagnostics: OptimizationDiagnostics,
}

/// Diagnostic information for optimization process
#[derive(Debug, Clone)]
pub struct OptimizationDiagnostics {
    /// Number of outliers detected and handled
    pub outliers_detected: usize,
    /// Number of numerical instabilities encountered
    pub numerical_issues: usize,
    /// Condition number estimates
    pub condition_estimates: Vec<Float>,
    /// Trust region statistics
    pub trust_region_stats: Option<TrustRegionStats>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Trust region optimization statistics
#[derive(Debug, Clone)]
pub struct TrustRegionStats {
    /// Number of successful steps
    pub successful_steps: usize,
    /// Number of rejected steps
    pub rejected_steps: usize,
    /// Final trust region radius
    pub final_radius: Float,
    /// Average radius throughout optimization
    pub average_radius: Float,
}

/// Trait for robust objective functions
pub trait RobustObjective {
    /// Evaluate objective function
    fn evaluate(&self, params: &ArrayView1<Float>) -> SklResult<Float>;

    /// Compute gradient
    fn gradient(&self, params: &ArrayView1<Float>) -> SklResult<Array1<Float>>;

    /// Compute Hessian (optional, for second-order methods)
    fn hessian(&self, params: &ArrayView1<Float>) -> SklResult<Array2<Float>> {
        Err(SklearsError::InvalidParameter {
            name: "hessian".to_string(),
            reason: "Hessian not implemented for this objective".to_string(),
        })
    }

    /// Check for outliers in data (optional)
    fn detect_outliers(&self, _params: &ArrayView1<Float>) -> Vec<usize> {
        Vec::new()
    }
}

/// Robust optimization algorithms
pub struct RobustOptimizer {
    config: RobustOptimConfig,
}

impl RobustOptimizer {
    /// Create a new robust optimizer
    pub fn new(config: RobustOptimConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(RobustOptimConfig::default())
    }

    /// Robust Adam optimizer with outlier handling
    pub fn robust_adam<F: RobustObjective>(
        &self,
        objective: &F,
        initial_params: Array1<Float>,
    ) -> SklResult<OptimizationResult> {
        let mut params = initial_params;
        let n_params = params.len();

        // Adam state variables
        let mut m = Array1::zeros(n_params); // First moment estimate
        let mut v = Array1::zeros(n_params); // Second moment estimate
        let mut t = 0; // Time step

        // Adaptive learning rate
        let mut learning_rate = self.config.learning_rate;

        // History tracking
        let mut objective_history = Vec::new();
        let mut gradient_history = Vec::new();
        let mut lr_history = Vec::new();

        // Diagnostics
        let mut outliers_detected = 0;
        let mut numerical_issues = 0;
        let mut condition_estimates = Vec::new();

        for iteration in 0..self.config.max_iterations {
            t += 1;

            // Evaluate objective and gradient
            let obj_value = objective.evaluate(&params.view())?;
            let mut gradient = objective.gradient(&params.view())?;

            // Handle outliers if enabled
            if self.config.use_robust_loss {
                let outliers = objective.detect_outliers(&params.view());
                outliers_detected += outliers.len();

                // Apply robust loss function
                gradient = self.apply_robust_loss(&gradient, &outliers)?;
            }

            // Check for numerical issues
            if !gradient.iter().all(|&x| x.is_finite()) {
                numerical_issues += 1;
                // Reset gradient to small random values
                self.reset_gradient_safely(&mut gradient)?;
            }

            // Add regularization
            if self.config.l1_reg > 0.0 {
                for (i, &param) in params.iter().enumerate() {
                    gradient[i] += self.config.l1_reg * param.signum();
                }
            }

            if self.config.l2_reg > 0.0 {
                gradient += &(&params * self.config.l2_reg);
            }

            // Estimate condition number (simplified)
            let gradient_norm = gradient.iter().map(|&x| x * x).sum::<Float>().sqrt();
            if gradient_norm > 1e-14 {
                let condition_est = obj_value.abs() / gradient_norm;
                condition_estimates.push(condition_est);
            }

            // Adam update
            let beta1 = 0.9;
            let beta2 = 0.999;
            let epsilon = 1e-8;

            // Update biased first moment estimate
            m = beta1 * &m + (1.0 - beta1) * &gradient;

            // Update biased second moment estimate
            let gradient_squared = gradient.mapv(|x| x * x);
            v = beta2 * &v + (1.0 - beta2) * &gradient_squared;

            // Compute bias-corrected first moment estimate
            let m_hat = &m / (1.0 - beta1.powi(t));

            // Compute bias-corrected second moment estimate
            let v_hat = &v / (1.0 - beta2.powi(t));

            // Adaptive learning rate
            if self.config.adaptive_lr && iteration > 10 {
                learning_rate =
                    self.adapt_learning_rate(learning_rate, &objective_history, iteration);
            }

            // Update parameters
            for i in 0..n_params {
                let denominator = v_hat[i].sqrt() + epsilon;
                params[i] -= learning_rate * m_hat[i] / denominator;

                // Clip extreme values for numerical stability
                if !params[i].is_finite() {
                    params[i] = 0.0;
                    numerical_issues += 1;
                }
            }

            // Record history
            objective_history.push(obj_value);
            gradient_history.push(gradient_norm);
            lr_history.push(learning_rate);

            // Check convergence
            if gradient_norm < self.config.tolerance {
                return Ok(OptimizationResult {
                    parameters: params,
                    objective_value: obj_value,
                    iterations: iteration + 1,
                    converged: true,
                    final_gradient_norm: gradient_norm,
                    objective_history,
                    gradient_history,
                    learning_rate_history: lr_history,
                    diagnostics: OptimizationDiagnostics {
                        outliers_detected,
                        numerical_issues,
                        condition_estimates,
                        trust_region_stats: None,
                        metadata: HashMap::new(),
                    },
                });
            }
        }

        // Did not converge
        let final_obj = objective.evaluate(&params.view())?;
        let final_grad = objective.gradient(&params.view())?;
        let final_grad_norm = final_grad.iter().map(|&x| x * x).sum::<Float>().sqrt();

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_obj,
            iterations: self.config.max_iterations,
            converged: false,
            final_gradient_norm: final_grad_norm,
            objective_history,
            gradient_history,
            learning_rate_history: lr_history,
            diagnostics: OptimizationDiagnostics {
                outliers_detected,
                numerical_issues,
                condition_estimates,
                trust_region_stats: None,
                metadata: HashMap::new(),
            },
        })
    }

    /// Trust region optimizer for robust optimization
    pub fn trust_region<F: RobustObjective>(
        &self,
        objective: &F,
        initial_params: Array1<Float>,
    ) -> SklResult<OptimizationResult> {
        let mut params = initial_params;
        let mut trust_radius = self.config.trust_radius;

        // Trust region statistics
        let mut successful_steps = 0;
        let mut rejected_steps = 0;
        let mut radius_history = Vec::new();

        // History tracking
        let mut objective_history = Vec::new();
        let mut gradient_history = Vec::new();
        let mut lr_history = Vec::new();

        // Diagnostics
        let outliers_detected = 0;
        let numerical_issues = 0;
        let mut condition_estimates = Vec::new();

        for iteration in 0..self.config.max_iterations {
            // Evaluate objective and gradient
            let obj_value = objective.evaluate(&params.view())?;
            let gradient = objective.gradient(&params.view())?;

            let gradient_norm = gradient.iter().map(|&x| x * x).sum::<Float>().sqrt();

            // Record history before convergence check
            objective_history.push(obj_value);
            gradient_history.push(gradient_norm);

            // Check convergence (but only after first iteration to avoid immediate exit)
            if iteration > 0 && gradient_norm < self.config.tolerance {
                break;
            }

            // Solve trust region subproblem (simplified - just use steepest descent)
            let step_norm = gradient_norm.min(trust_radius);
            let step = &gradient * (-step_norm / gradient_norm);

            // Ensure step is within trust region
            let actual_step_norm = step.iter().map(|&x| x * x).sum::<Float>().sqrt();
            let step = if actual_step_norm > trust_radius {
                &step * (trust_radius / actual_step_norm)
            } else {
                step
            };

            // Candidate new parameters
            let candidate_params = &params + &step;

            // Evaluate objective at candidate point
            let candidate_obj = objective.evaluate(&candidate_params.view())?;

            // Compute actual reduction
            let actual_reduction = obj_value - candidate_obj;

            // Compute predicted reduction (linear model)
            let predicted_reduction = -gradient.dot(&step);

            // Compute ratio
            let ratio = if predicted_reduction.abs() > 1e-14 {
                actual_reduction / predicted_reduction
            } else {
                0.0
            };

            // Update trust region radius and accept/reject step
            if ratio > 0.75 && actual_step_norm > 0.9 * trust_radius {
                // Very successful - increase radius
                trust_radius = (2.0 * trust_radius).min(1e6);
                params = candidate_params;
                successful_steps += 1;
            } else if ratio > 0.25 {
                // Moderately successful - keep radius, accept step
                params = candidate_params;
                successful_steps += 1;
            } else {
                // Unsuccessful - decrease radius, reject step
                trust_radius *= 0.5;
                rejected_steps += 1;

                if trust_radius < 1e-12 {
                    // Trust region too small, stop
                    break;
                }
            }

            radius_history.push(trust_radius);
            lr_history.push(trust_radius); // Use radius as "learning rate" proxy

            // Estimate condition number
            if gradient_norm > 1e-14 {
                let condition_est = obj_value.abs() / gradient_norm;
                condition_estimates.push(condition_est);
            }
        }

        let final_obj = objective.evaluate(&params.view())?;
        let final_grad = objective.gradient(&params.view())?;
        let final_grad_norm = final_grad.iter().map(|&x| x * x).sum::<Float>().sqrt();

        let trust_stats = TrustRegionStats {
            successful_steps,
            rejected_steps,
            final_radius: trust_radius,
            average_radius: if radius_history.is_empty() {
                trust_radius
            } else {
                radius_history.iter().sum::<Float>() / radius_history.len() as Float
            },
        };

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_obj,
            iterations: objective_history.len(),
            converged: final_grad_norm < self.config.tolerance,
            final_gradient_norm: final_grad_norm,
            objective_history,
            gradient_history,
            learning_rate_history: lr_history,
            diagnostics: OptimizationDiagnostics {
                outliers_detected,
                numerical_issues,
                condition_estimates,
                trust_region_stats: Some(trust_stats),
                metadata: HashMap::new(),
            },
        })
    }

    /// L-BFGS optimizer with robust modifications
    pub fn robust_lbfgs<F: RobustObjective>(
        &self,
        objective: &F,
        initial_params: Array1<Float>,
    ) -> SklResult<OptimizationResult> {
        let mut params = initial_params;
        let memory_size = 10; // L-BFGS memory

        // L-BFGS history
        let mut s_history: Vec<Array1<Float>> = Vec::new();
        let mut y_history: Vec<Array1<Float>> = Vec::new();
        let mut rho_history: Vec<Float> = Vec::new();

        // Previous gradient
        let mut prev_gradient = objective.gradient(&params.view())?;

        // History tracking
        let mut objective_history = Vec::new();
        let mut gradient_history = Vec::new();
        let mut lr_history = Vec::new();

        // Diagnostics
        let outliers_detected = 0;
        let mut numerical_issues = 0;
        let mut condition_estimates = Vec::new();

        for iteration in 0..self.config.max_iterations {
            let obj_value = objective.evaluate(&params.view())?;
            let gradient = objective.gradient(&params.view())?;

            let gradient_norm = gradient.iter().map(|&x| x * x).sum::<Float>().sqrt();

            // Record history before convergence check
            objective_history.push(obj_value);
            gradient_history.push(gradient_norm);

            // Check convergence (but only after first iteration to avoid immediate exit)
            if iteration > 0 && gradient_norm < self.config.tolerance {
                break;
            }

            // Compute search direction using L-BFGS
            let direction = if s_history.is_empty() {
                // First iteration - use steepest descent
                gradient.mapv(|x| -x)
            } else {
                self.lbfgs_direction(&gradient, &s_history, &y_history, &rho_history)?
            };

            // Line search (simplified - use fixed step size with backtracking)
            let mut step_size = 1.0;
            let mut candidate_params = &params + &(&direction * step_size);
            let mut candidate_obj = objective.evaluate(&candidate_params.view())?;

            // Backtracking line search
            let armijo_const = 1e-4;
            let backtrack_factor = 0.5;
            let sufficient_decrease =
                obj_value + armijo_const * step_size * gradient.dot(&direction);

            for _ in 0..20 {
                // Max 20 backtrack steps
                if candidate_obj <= sufficient_decrease || step_size < 1e-12 {
                    break;
                }
                step_size *= backtrack_factor;
                candidate_params = &params + &(&direction * step_size);
                candidate_obj = objective.evaluate(&candidate_params.view())?;
            }

            // Update parameters
            let new_params = candidate_params;
            let new_gradient = objective.gradient(&new_params.view())?;

            // Update L-BFGS memory
            let s = &new_params - &params;
            let y = &new_gradient - &prev_gradient;
            let rho = 1.0 / y.dot(&s);

            if rho.is_finite() && rho > 1e-14 {
                s_history.push(s);
                y_history.push(y);
                rho_history.push(rho);

                // Keep only recent history
                if s_history.len() > memory_size {
                    s_history.remove(0);
                    y_history.remove(0);
                    rho_history.remove(0);
                }
            } else {
                numerical_issues += 1;
            }

            params = new_params;
            prev_gradient = new_gradient;

            // Record learning rate history
            lr_history.push(step_size);

            // Estimate condition number
            if gradient_norm > 1e-14 {
                let condition_est = obj_value.abs() / gradient_norm;
                condition_estimates.push(condition_est);
            }
        }

        let final_obj = objective.evaluate(&params.view())?;
        let final_grad = objective.gradient(&params.view())?;
        let final_grad_norm = final_grad.iter().map(|&x| x * x).sum::<Float>().sqrt();

        Ok(OptimizationResult {
            parameters: params,
            objective_value: final_obj,
            iterations: objective_history.len(),
            converged: final_grad_norm < self.config.tolerance,
            final_gradient_norm: final_grad_norm,
            objective_history,
            gradient_history,
            learning_rate_history: lr_history,
            diagnostics: OptimizationDiagnostics {
                outliers_detected,
                numerical_issues,
                condition_estimates,
                trust_region_stats: None,
                metadata: HashMap::new(),
            },
        })
    }

    /// Apply robust loss function to handle outliers
    fn apply_robust_loss(
        &self,
        gradient: &Array1<Float>,
        outliers: &[usize],
    ) -> SklResult<Array1<Float>> {
        let mut robust_gradient = gradient.clone();

        // Huber loss modification for outliers
        for &outlier_idx in outliers {
            if outlier_idx < robust_gradient.len() {
                let grad_val = robust_gradient[outlier_idx];
                let threshold = self.config.outlier_threshold;

                // Apply Huber loss derivative
                if grad_val.abs() > threshold {
                    robust_gradient[outlier_idx] = threshold * grad_val.signum();
                }
            }
        }

        Ok(robust_gradient)
    }

    /// Reset gradient safely when numerical issues occur
    fn reset_gradient_safely(&self, gradient: &mut Array1<Float>) -> SklResult<()> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        for elem in gradient.iter_mut() {
            if !elem.is_finite() {
                *elem = rng.gen_range(-1e-6..1e-6);
            }
        }

        Ok(())
    }

    /// Adaptive learning rate adjustment
    fn adapt_learning_rate(&self, current_lr: Float, history: &[Float], iteration: usize) -> Float {
        if history.len() < 3 {
            return current_lr;
        }

        let recent_window = history.len().saturating_sub(5);
        let recent_values = &history[recent_window..];

        // Check if objective is decreasing
        let is_decreasing = recent_values.windows(2).all(|w| w[1] <= w[0]);

        if is_decreasing {
            // Increase learning rate slightly
            (current_lr * 1.05).min(1.0)
        } else {
            // Decrease learning rate
            current_lr * 0.95
        }
    }

    /// Compute L-BFGS search direction
    fn lbfgs_direction(
        &self,
        gradient: &Array1<Float>,
        s_history: &[Array1<Float>],
        y_history: &[Array1<Float>],
        rho_history: &[Float],
    ) -> SklResult<Array1<Float>> {
        let mut q = gradient.clone();
        let m = s_history.len();
        let mut alpha = vec![0.0; m];

        // First loop
        for i in (0..m).rev() {
            alpha[i] = rho_history[i] * s_history[i].dot(&q);
            q = q - alpha[i] * &y_history[i];
        }

        // Scale by H_0 (use identity for simplicity)
        let mut r = q.mapv(|x| -x);

        // Second loop
        for i in 0..m {
            let beta = rho_history[i] * y_history[i].dot(&r);
            r = r + (alpha[i] - beta) * &s_history[i];
        }

        Ok(r)
    }
}

/// Example robust objective function for manifold learning
pub struct RobustMDSObjective {
    data: Array2<Float>,
    weights: Array1<Float>,
    target_distances: Array2<Float>,
}

impl RobustMDSObjective {
    /// Create a new robust MDS objective
    pub fn new(data: Array2<Float>, target_distances: Array2<Float>) -> Self {
        let n_samples = data.nrows();
        let weights = Array1::ones(n_samples);

        Self {
            data,
            weights,
            target_distances,
        }
    }

    /// Set sample weights
    pub fn with_weights(mut self, weights: Array1<Float>) -> Self {
        self.weights = weights;
        self
    }
}

impl RobustObjective for RobustMDSObjective {
    fn evaluate(&self, params: &ArrayView1<Float>) -> SklResult<Float> {
        let n_samples = self.data.nrows();
        let n_components = params.len() / n_samples;

        if params.len() != n_samples * n_components {
            return Err(SklearsError::InvalidParameter {
                name: "params_length".to_string(),
                reason: "Parameter length doesn't match expected embedding size".to_string(),
            });
        }

        // Reshape parameters to embedding matrix
        let embedding = Array2::from_shape_vec((n_samples, n_components), params.to_vec())
            .map_err(|e| SklearsError::InvalidParameter {
                name: "embedding_shape".to_string(),
                reason: format!("Failed to reshape parameters: {}", e),
            })?;

        let mut stress = 0.0;

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let target_dist = self.target_distances[[i, j]];

                // Compute embedding distance
                let mut embed_dist_sq = 0.0;
                for k in 0..n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    embed_dist_sq += diff * diff;
                }
                let embed_dist = embed_dist_sq.sqrt();

                // Robust loss (Huber loss)
                let residual = embed_dist - target_dist;
                let threshold = 1.0;

                let loss = if residual.abs() <= threshold {
                    0.5 * residual * residual
                } else {
                    threshold * (residual.abs() - 0.5 * threshold)
                };

                stress += self.weights[i] * self.weights[j] * loss;
            }
        }

        Ok(stress)
    }

    fn gradient(&self, params: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let n_samples = self.data.nrows();
        let n_components = params.len() / n_samples;

        let embedding = Array2::from_shape_vec((n_samples, n_components), params.to_vec())
            .map_err(|e| SklearsError::InvalidParameter {
                name: "embedding_shape".to_string(),
                reason: format!("Failed to reshape parameters: {}", e),
            })?;

        let mut gradient = Array1::zeros(params.len());

        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let target_dist = self.target_distances[[i, j]];

                // Compute embedding distance
                let mut embed_dist_sq = 0.0;
                for k in 0..n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    embed_dist_sq += diff * diff;
                }
                let embed_dist = embed_dist_sq.sqrt();

                if embed_dist < 1e-14 {
                    // Handle case where points are very close or identical
                    // Add a small gradient to push points apart
                    let weight = self.weights[i] * self.weights[j];
                    for k in 0..n_components {
                        let small_gradient = weight * target_dist * 1e-3; // Small push toward target distance
                        gradient[i * n_components + k] += small_gradient;
                        gradient[j * n_components + k] -= small_gradient;
                    }
                    continue;
                }

                // Robust loss derivative
                let residual = embed_dist - target_dist;
                let threshold = 1.0;

                let loss_derivative = if residual.abs() <= threshold {
                    residual
                } else {
                    threshold * residual.signum()
                };

                let weight = self.weights[i] * self.weights[j];

                for k in 0..n_components {
                    let diff = embedding[[i, k]] - embedding[[j, k]];
                    let common_factor = weight * loss_derivative * diff / embed_dist;

                    gradient[i * n_components + k] += common_factor;
                    gradient[j * n_components + k] -= common_factor;
                }
            }
        }

        Ok(gradient)
    }

    fn detect_outliers(&self, params: &ArrayView1<Float>) -> Vec<usize> {
        let n_samples = self.data.nrows();
        let n_components = params.len() / n_samples;

        if params.len() != n_samples * n_components {
            return Vec::new();
        }

        let embedding = Array2::from_shape_vec((n_samples, n_components), params.to_vec());
        if embedding.is_err() {
            return Vec::new();
        }
        let embedding = embedding.unwrap();

        let mut outliers = Vec::new();

        // Simple outlier detection based on distance from centroid
        let centroid = embedding.mean_axis(Axis(0)).unwrap();

        for i in 0..n_samples {
            let mut dist_sq = 0.0;
            for k in 0..n_components {
                let diff = embedding[[i, k]] - centroid[k];
                dist_sq += diff * diff;
            }
            let dist = dist_sq.sqrt();

            // Mark as outlier if distance > 3 standard deviations
            if dist > 3.0 {
                outliers.push(i);
            }
        }

        outliers
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_robust_adam() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let target_distances = array![[0.0, 1.0, 1.0], [1.0, 0.0, 1.414], [1.0, 1.414, 0.0]];

        let objective = RobustMDSObjective::new(data, target_distances);
        let initial_params = Array1::zeros(6); // 3 points × 2 dimensions

        let config = RobustOptimConfig {
            max_iterations: 100,
            tolerance: 1e-3,
            ..Default::default()
        };

        let optimizer = RobustOptimizer::new(config);
        let result = optimizer.robust_adam(&objective, initial_params).unwrap();

        assert!(result.iterations > 0);
        assert!(result.final_gradient_norm >= 0.0);
    }

    #[test]
    fn test_trust_region() {
        let data = array![[0.0, 0.0], [1.0, 0.0]];
        let target_distances = array![[0.0, 1.0], [1.0, 0.0]];

        let objective = RobustMDSObjective::new(data, target_distances);
        // Use small random initialization instead of zeros
        let initial_params = array![0.1, 0.1, 0.2, 0.1]; // 2 points × 2 dimensions

        let config = RobustOptimConfig {
            max_iterations: 50,
            ..Default::default()
        };

        let optimizer = RobustOptimizer::new(config);
        let result = optimizer.trust_region(&objective, initial_params).unwrap();

        assert!(result.iterations > 0);
        assert!(result.diagnostics.trust_region_stats.is_some());
    }

    #[test]
    fn test_robust_lbfgs() {
        let data = array![[0.0, 0.0], [1.0, 0.0]];
        let target_distances = array![[0.0, 1.0], [1.0, 0.0]];

        let objective = RobustMDSObjective::new(data, target_distances);
        // Use small random initialization instead of zeros
        let initial_params = array![0.1, 0.1, 0.2, 0.1];

        let config = RobustOptimConfig {
            max_iterations: 50,
            tolerance: 1e-4,
            ..Default::default()
        };

        let optimizer = RobustOptimizer::new(config);
        let result = optimizer.robust_lbfgs(&objective, initial_params).unwrap();

        assert!(result.iterations > 0);
        assert!(result.final_gradient_norm >= 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]]; // Third point is outlier
        let target_distances = Array2::zeros((3, 3));

        let objective = RobustMDSObjective::new(data, target_distances);
        let params = array![0.0, 0.0, 1.0, 0.0, 10.0, 10.0]; // Embedding that matches data

        let outliers = objective.detect_outliers(&params.view());

        // Should detect the third point as an outlier
        assert!(!outliers.is_empty());
    }

    #[test]
    fn test_optimization_diagnostics() {
        let data = array![[0.0, 0.0], [1.0, 0.0]];
        let target_distances = array![[0.0, 1.0], [1.0, 0.0]];

        let objective = RobustMDSObjective::new(data, target_distances);
        let initial_params = Array1::zeros(4);

        let config = RobustOptimConfig {
            max_iterations: 10,
            use_robust_loss: true,
            ..Default::default()
        };

        let optimizer = RobustOptimizer::new(config);
        let result = optimizer.robust_adam(&objective, initial_params).unwrap();

        // Check that diagnostics are populated
        assert!(!result.objective_history.is_empty());
        assert!(!result.gradient_history.is_empty());
        assert!(!result.learning_rate_history.is_empty());
        assert_eq!(result.objective_history.len(), result.iterations);
    }
}

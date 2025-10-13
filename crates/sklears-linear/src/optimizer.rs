//! Optimization algorithms for linear models

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use sklears_core::types::Float;

/// L-BFGS optimizer for unconstrained optimization
pub struct LbfgsOptimizer {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Number of corrections to approximate the Hessian
    pub m: usize,
    /// Line search max iterations
    pub max_ls: usize,
}

impl Default for LbfgsOptimizer {
    fn default() -> Self {
        Self {
            max_iter: 100,
            tol: 1e-4,
            m: 10,
            max_ls: 20,
        }
    }
}

impl LbfgsOptimizer {
    /// Minimize a differentiable function using L-BFGS
    pub fn minimize<F, G>(
        &self,
        f: F,
        grad_f: G,
        x0: Array1<Float>,
    ) -> Result<Array1<Float>, String>
    where
        F: Fn(&Array1<Float>) -> Float,
        G: Fn(&Array1<Float>) -> Array1<Float>,
    {
        let _n = x0.len();
        let mut x = x0;
        let mut f_k = f(&x);
        let mut g_k = grad_f(&x);

        // History for L-BFGS
        let mut s_list: Vec<Array1<Float>> = Vec::with_capacity(self.m);
        let mut y_list: Vec<Array1<Float>> = Vec::with_capacity(self.m);
        let mut rho_list: Vec<Float> = Vec::with_capacity(self.m);

        for _iter in 0..self.max_iter {
            // Check convergence
            let g_norm = g_k.mapv(Float::abs).sum();
            if g_norm < self.tol {
                return Ok(x);
            }

            // Compute search direction using L-BFGS
            let p = self.compute_direction(&g_k, &s_list, &y_list, &rho_list);

            // Line search
            let alpha = self.line_search(&f, &x, &p, f_k, &g_k)?;

            // Update x
            let x_new = &x + alpha * &p;
            let f_new = f(&x_new);
            let g_new = grad_f(&x_new);

            // Update history
            let s = &x_new - &x;
            let y = &g_new - &g_k;

            let rho = 1.0 / s.dot(&y).max(1e-10);

            // Maintain history size
            if s_list.len() == self.m {
                s_list.remove(0);
                y_list.remove(0);
                rho_list.remove(0);
            }

            s_list.push(s);
            y_list.push(y);
            rho_list.push(rho);

            // Update for next iteration
            x = x_new;
            f_k = f_new;
            g_k = g_new;
        }

        Ok(x)
    }

    /// Compute L-BFGS search direction
    fn compute_direction(
        &self,
        g: &Array1<Float>,
        s_list: &[Array1<Float>],
        y_list: &[Array1<Float>],
        rho_list: &[Float],
    ) -> Array1<Float> {
        let mut q = g.clone();
        let k = s_list.len();
        let mut alpha = vec![0.0; k];

        // First loop
        for i in (0..k).rev() {
            alpha[i] = rho_list[i] * s_list[i].dot(&q);
            q = q - alpha[i] * &y_list[i];
        }

        // Scale initial direction
        let mut r = if k > 0 {
            let gamma = s_list[k - 1].dot(&y_list[k - 1]) / y_list[k - 1].dot(&y_list[k - 1]);
            gamma * q
        } else {
            q
        };

        // Second loop
        for i in 0..k {
            let beta = rho_list[i] * y_list[i].dot(&r);
            r = r + (alpha[i] - beta) * &s_list[i];
        }

        -r
    }

    /// Backtracking line search
    fn line_search<F>(
        &self,
        f: &F,
        x: &Array1<Float>,
        p: &Array1<Float>,
        f_k: Float,
        g_k: &Array1<Float>,
    ) -> Result<Float, String>
    where
        F: Fn(&Array1<Float>) -> Float,
    {
        let c1 = 1e-4;
        let rho = 0.5;
        let mut alpha = 1.0;

        let gp = g_k.dot(p);

        for _ in 0..self.max_ls {
            let x_new = x + alpha * p;
            let f_new = f(&x_new);

            // Armijo condition
            if f_new <= f_k + c1 * alpha * gp {
                return Ok(alpha);
            }

            alpha *= rho;
        }

        Err("Line search failed to find suitable step size".to_string())
    }
}

/// Stochastic Average Gradient (SAG) optimizer
pub struct SagOptimizer {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate
    pub learning_rate: Option<Float>,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for SagOptimizer {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            tol: 1e-4,
            learning_rate: None,
            random_state: None,
        }
    }
}

impl SagOptimizer {
    /// Minimize using SAG for finite-sum problems
    /// f_i: individual loss function for sample i
    /// grad_f_i: gradient of individual loss function
    pub fn minimize<F, G>(
        &self,
        _f_i: F,
        grad_f_i: G,
        x0: Array1<Float>,
        n_samples: usize,
    ) -> Result<Array1<Float>, String>
    where
        F: Fn(&Array1<Float>, usize) -> Float,
        G: Fn(&Array1<Float>, usize) -> Array1<Float>,
    {
        let n_features = x0.len();
        let mut x = x0;

        // Initialize gradient memory
        let mut gradient_memory = Array2::zeros((n_samples, n_features));
        let mut gradient_sum = Array1::zeros(n_features);
        let mut seen = vec![false; n_samples];

        // Learning rate (can be auto-tuned based on Lipschitz constant)
        let alpha = self.learning_rate.unwrap_or(0.01);

        // Random number generator
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        for _epoch in 0..self.max_epochs {
            // Random permutation for this epoch
            let mut indices: Vec<usize> = (0..n_samples).collect();
            use scirs2_core::random::SliceRandomExt;
            indices.shuffle(&mut rng);

            for &i in &indices {
                // Compute gradient for sample i
                let grad_i = grad_f_i(&x, i);

                // Update gradient sum
                if seen[i] {
                    gradient_sum = &gradient_sum - &gradient_memory.row(i) + &grad_i;
                } else {
                    gradient_sum = &gradient_sum + &grad_i;
                    seen[i] = true;
                }

                // Store gradient in memory
                gradient_memory.row_mut(i).assign(&grad_i);

                // Update parameters
                x = &x - alpha * &gradient_sum / n_samples as Float;
            }

            // Check convergence
            let avg_grad_norm = gradient_sum.mapv(Float::abs).sum() / n_samples as Float;
            if avg_grad_norm < self.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }
}

/// SAGA optimizer (improved SAG with support for non-smooth penalties)
pub struct SagaOptimizer {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate
    pub learning_rate: Option<Float>,
    /// Random seed
    pub random_state: Option<u64>,
}

impl Default for SagaOptimizer {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            tol: 1e-4,
            learning_rate: None,
            random_state: None,
        }
    }
}

impl SagaOptimizer {
    /// Minimize using SAGA for composite objectives: f(x) + g(x)
    /// where f is smooth (finite-sum) and g is possibly non-smooth (e.g., L1 penalty)
    pub fn minimize<F, GradF, ProxG>(
        &self,
        _f_i: F,
        grad_f_i: GradF,
        prox_g: ProxG,
        x0: Array1<Float>,
        n_samples: usize,
    ) -> Result<Array1<Float>, String>
    where
        F: Fn(&Array1<Float>, usize) -> Float,
        GradF: Fn(&Array1<Float>, usize) -> Array1<Float>,
        ProxG: Fn(&Array1<Float>, Float) -> Array1<Float>,
    {
        let n_features = x0.len();
        let mut x = x0;

        // Initialize gradient memory and average
        let mut gradient_memory = Array2::zeros((n_samples, n_features));
        let mut gradient_avg = Array1::zeros(n_features);

        // Initialize all gradients
        for i in 0..n_samples {
            let grad_i = grad_f_i(&x, i);
            gradient_memory.row_mut(i).assign(&grad_i);
            gradient_avg = &gradient_avg + &grad_i / n_samples as Float;
        }

        // Learning rate
        let alpha = self.learning_rate.unwrap_or(0.01);

        // Random number generator
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        for _epoch in 0..self.max_epochs {
            // Random permutation
            let mut indices: Vec<usize> = (0..n_samples).collect();
            use scirs2_core::random::SliceRandomExt;
            indices.shuffle(&mut rng);

            for &i in &indices {
                // Store old gradient
                let grad_old = gradient_memory.row(i).to_owned();

                // Compute new gradient
                let grad_new = grad_f_i(&x, i);

                // Update gradient memory and average
                gradient_memory.row_mut(i).assign(&grad_new);
                gradient_avg = &gradient_avg + (&grad_new - &grad_old) / n_samples as Float;

                // Gradient step
                let v = &grad_new - &grad_old + &gradient_avg;
                let x_intermediate = &x - alpha * &v;

                // Proximal step (handles non-smooth penalty)
                x = prox_g(&x_intermediate, alpha);
            }

            // Check convergence
            let grad_norm = gradient_avg.mapv(Float::abs).sum();
            if grad_norm < self.tol {
                return Ok(x);
            }
        }

        Ok(x)
    }
}

/// Proximal Gradient Method optimizer for composite objectives
pub struct ProximalGradientOptimizer {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Initial step size
    pub step_size: Option<Float>,
    /// Whether to use line search for step size
    pub use_line_search: bool,
    /// Whether to use acceleration (FISTA)
    pub accelerated: bool,
    /// Backtracking line search parameters
    pub beta: Float,
    pub sigma: Float,
}

impl Default for ProximalGradientOptimizer {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            step_size: None,
            use_line_search: true,
            accelerated: false,
            beta: 0.5,
            sigma: 0.01,
        }
    }
}

impl ProximalGradientOptimizer {
    /// Create accelerated version (FISTA)
    pub fn accelerated() -> Self {
        Self {
            accelerated: true,
            ..Default::default()
        }
    }

    /// Minimize composite objective f(x) + g(x) using proximal gradient method
    /// where f is smooth and g is convex (possibly non-smooth)
    pub fn minimize<F, GradF, ProxG>(
        &self,
        f: F,
        grad_f: GradF,
        prox_g: ProxG,
        x0: Array1<Float>,
    ) -> Result<Array1<Float>, String>
    where
        F: Fn(&Array1<Float>) -> Float,
        GradF: Fn(&Array1<Float>) -> Array1<Float>,
        ProxG: Fn(&Array1<Float>, Float) -> Array1<Float>,
    {
        let mut x = x0.clone();
        let mut y = x0; // For acceleration
        let mut t = 1.0; // Acceleration parameter

        // Initial step size
        let mut step_size = self.step_size.unwrap_or(1.0);

        let mut f_prev = f(&x);

        for iter in 0..self.max_iter {
            // Use y for accelerated version, x for regular version
            let current_point = if self.accelerated { &y } else { &x };

            // Compute gradient at current point
            let grad = grad_f(current_point);

            // Line search for step size if enabled
            if self.use_line_search {
                step_size = self.backtracking_line_search(
                    &f,
                    &grad_f,
                    &prox_g,
                    current_point,
                    &grad,
                    step_size,
                )?;
            }

            // Proximal gradient step
            let x_new = prox_g(&(current_point - step_size * &grad), step_size);

            // Check convergence
            let diff_norm = (&x_new - &x).mapv(Float::abs).sum();
            if diff_norm < self.tol {
                return Ok(x_new);
            }

            // Update for acceleration
            if self.accelerated {
                let t_new: Float = (1.0_f64 + (1.0_f64 + 4.0_f64 * t * t).sqrt()) / 2.0_f64;
                let beta = (t - 1.0) / t_new;
                y = &x_new + beta * (&x_new - &x);
                t = t_new;
            }

            x = x_new;

            // Optional: track objective value for convergence
            let f_curr = f(&x);
            if (f_curr - f_prev).abs() < self.tol * f_prev.abs().max(1.0) {
                if iter > 10 {
                    // Avoid early termination
                    return Ok(x);
                }
            }
            f_prev = f_curr;
        }

        Ok(x)
    }

    /// Backtracking line search for proximal gradient
    fn backtracking_line_search<F, GradF, ProxG>(
        &self,
        f: &F,
        _grad_f: &GradF,
        prox_g: &ProxG,
        x: &Array1<Float>,
        grad: &Array1<Float>,
        mut step_size: Float,
    ) -> Result<Float, String>
    where
        F: Fn(&Array1<Float>) -> Float,
        GradF: Fn(&Array1<Float>) -> Array1<Float>,
        ProxG: Fn(&Array1<Float>, Float) -> Array1<Float>,
    {
        let f_x = f(x);
        let max_iter = 50;

        for _ in 0..max_iter {
            // Proximal step
            let x_new = prox_g(&(x - step_size * grad), step_size);
            let f_new = f(&x_new);

            // Quadratic approximation condition
            let diff = &x_new - x;
            let quad_approx = f_x + grad.dot(&diff) + diff.dot(&diff) / (2.0 * step_size);

            if f_new <= quad_approx + self.sigma * step_size * grad.dot(grad) {
                return Ok(step_size);
            }

            step_size *= self.beta;

            if step_size < 1e-16 {
                return Err("Step size became too small in line search".to_string());
            }
        }

        Ok(step_size) // Return current step size if line search doesn't converge
    }
}

/// Accelerated Proximal Gradient Method (FISTA)
pub type FistaOptimizer = ProximalGradientOptimizer;

/// Accelerated Gradient Descent with Nesterov momentum
pub struct NesterovAcceleratedGradient {
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Learning rate
    pub learning_rate: Float,
    /// Whether to use adaptive learning rate
    pub adaptive_lr: bool,
    /// Momentum parameter (typically 0.9)
    pub momentum: Float,
    /// Learning rate decay factor
    pub lr_decay: Float,
}

impl Default for NesterovAcceleratedGradient {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
            learning_rate: 0.01,
            adaptive_lr: false,
            momentum: 0.9,
            lr_decay: 0.999,
        }
    }
}

impl NesterovAcceleratedGradient {
    /// Create optimizer with adaptive learning rate
    pub fn adaptive() -> Self {
        Self {
            adaptive_lr: true,
            ..Default::default()
        }
    }

    /// Minimize smooth objective using Nesterov accelerated gradient descent
    pub fn minimize<F, GradF>(
        &self,
        f: F,
        grad_f: GradF,
        x0: Array1<Float>,
    ) -> Result<Array1<Float>, String>
    where
        F: Fn(&Array1<Float>) -> Float,
        GradF: Fn(&Array1<Float>) -> Array1<Float>,
    {
        let mut x = x0.clone();
        let mut v = Array1::zeros(x0.len()); // velocity
        let mut lr = self.learning_rate;

        let mut f_prev = f(&x);

        for iter in 0..self.max_iter {
            // Nesterov look-ahead point
            let y = &x + self.momentum * &v;

            // Compute gradient at look-ahead point
            let grad = grad_f(&y);

            // Check convergence
            let grad_norm = grad.mapv(Float::abs).sum();
            if grad_norm < self.tol {
                return Ok(x);
            }

            // Update velocity and position
            v = self.momentum * &v - lr * &grad;
            x = &x + &v;

            // Adaptive learning rate
            if self.adaptive_lr {
                let f_curr = f(&x);
                if f_curr > f_prev + 1e-10 {
                    // If objective increased, reduce learning rate and reset
                    lr *= 0.9;
                    x = &x - &v; // undo the step
                    v = Array1::zeros(x0.len()); // reset velocity for stability
                    continue;
                } else if f_curr < f_prev - 0.01 * lr * grad_norm {
                    // If we're making good progress, increase learning rate slightly
                    lr = (lr * 1.005).min(self.learning_rate * 1.5); // Very conservative increase
                }

                // Check objective-based convergence every iteration for adaptive mode
                if (f_curr - f_prev).abs() < self.tol * f_prev.abs().max(1.0) {
                    return Ok(x);
                }
                f_prev = f_curr;
            } else {
                // Fixed decay schedule
                lr *= self.lr_decay;

                // Check objective-based convergence every 10 iterations for non-adaptive mode
                if iter % 10 == 0 {
                    let f_curr = f(&x);
                    if (f_curr - f_prev).abs() < self.tol * f_prev.abs().max(1.0) {
                        return Ok(x);
                    }
                    f_prev = f_curr;
                }
            }
        }

        Ok(x)
    }
}

/// Proximal operators for common penalties
pub mod proximal {
    use super::*;

    /// Proximal operator for L1 penalty: prox_{α*λ*||.||_1}
    pub fn prox_l1(x: &Array1<Float>, alpha_lambda: Float) -> Array1<Float> {
        x.mapv(|xi| soft_threshold(xi, alpha_lambda))
    }

    /// Proximal operator for L2 penalty: prox_{α*λ/2*||.||²}
    pub fn prox_l2(x: &Array1<Float>, alpha_lambda: Float) -> Array1<Float> {
        x / (1.0 + alpha_lambda)
    }

    /// Proximal operator for Elastic Net
    pub fn prox_elastic_net(x: &Array1<Float>, alpha: Float, l1_ratio: Float) -> Array1<Float> {
        let l1_prox = prox_l1(x, alpha * l1_ratio);
        prox_l2(&l1_prox, alpha * (1.0 - l1_ratio))
    }

    /// Soft thresholding function
    #[inline]
    fn soft_threshold(x: Float, lambda: Float) -> Float {
        if x > lambda {
            x - lambda
        } else if x < -lambda {
            x + lambda
        } else {
            0.0
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_lbfgs_quadratic() {
        // Minimize f(x) = 0.5 * x^T * A * x - b^T * x
        // where A = [[2, 0], [0, 4]], b = [1, 2]
        // Solution: x* = [0.5, 0.5]

        let f = |x: &Array1<Float>| -> Float {
            0.5 * (2.0 * x[0] * x[0] + 4.0 * x[1] * x[1]) - x[0] - 2.0 * x[1]
        };

        let grad_f =
            |x: &Array1<Float>| -> Array1<Float> { array![2.0 * x[0] - 1.0, 4.0 * x[1] - 2.0] };

        let optimizer = LbfgsOptimizer::default();
        let x0 = array![0.0, 0.0];

        let result = optimizer.minimize(f, grad_f, x0).unwrap();

        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(result[1], 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_proximal_operators() {
        use proximal::*;

        let x = array![2.0, -1.5, 0.5, -0.3];

        // Test L1 proximal
        let prox_x = prox_l1(&x, 0.5);
        assert_abs_diff_eq!(prox_x[0], 1.5);
        assert_abs_diff_eq!(prox_x[1], -1.0);
        assert_abs_diff_eq!(prox_x[2], 0.0);
        assert_abs_diff_eq!(prox_x[3], 0.0);

        // Test L2 proximal
        let prox_x = prox_l2(&x, 0.5);
        assert_abs_diff_eq!(prox_x[0], 2.0 / 1.5);
        assert_abs_diff_eq!(prox_x[1], -1.5 / 1.5);
    }

    #[test]
    fn test_proximal_gradient_lasso() {
        use proximal::*;

        // Test Lasso problem: min 0.5 * ||Ax - b||^2 + lambda * ||x||_1
        // A = [[1, 0], [0, 1]], b = [2, 1], lambda = 0.5
        // Solution should be approximately [1.5, 0.5]

        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![2.0, 1.0];
        let lambda = 0.5;

        let f = |x: &Array1<Float>| -> Float {
            let residual = a.dot(x) - &b;
            0.5 * residual.dot(&residual)
        };

        let grad_f = |x: &Array1<Float>| -> Array1<Float> {
            let residual = a.dot(x) - &b;
            a.t().dot(&residual)
        };

        let prox_g = |x: &Array1<Float>, t: Float| -> Array1<Float> { prox_l1(x, lambda * t) };

        let optimizer = ProximalGradientOptimizer::default();
        let x0 = array![0.0, 0.0];

        let result = optimizer.minimize(f, grad_f, prox_g, x0).unwrap();

        // Check that solution is reasonable
        assert!(result[0] > 1.0 && result[0] < 2.0);
        assert!(result[1] > 0.0 && result[1] < 1.0);
    }

    #[test]
    fn test_fista_accelerated() {
        use proximal::*;

        // Same problem as above but with FISTA acceleration
        let a = array![[1.0, 0.0], [0.0, 1.0]];
        let b = array![2.0, 1.0];
        let lambda = 0.5;

        let f = |x: &Array1<Float>| -> Float {
            let residual = a.dot(x) - &b;
            0.5 * residual.dot(&residual)
        };

        let grad_f = |x: &Array1<Float>| -> Array1<Float> {
            let residual = a.dot(x) - &b;
            a.t().dot(&residual)
        };

        let prox_g = |x: &Array1<Float>, t: Float| -> Array1<Float> { prox_l1(x, lambda * t) };

        let optimizer = ProximalGradientOptimizer::accelerated();
        let x0 = array![0.0, 0.0];

        let result = optimizer.minimize(f, grad_f, prox_g, x0).unwrap();

        // Check that solution is reasonable
        assert!(result[0] > 1.0 && result[0] < 2.0);
        assert!(result[1] > 0.0 && result[1] < 1.0);
    }

    #[test]
    fn test_nesterov_accelerated_gradient() {
        // Test Nesterov AGD on quadratic function: f(x) = 0.5 * x^T * A * x - b^T * x
        // where A = [[4, 0], [0, 1]], b = [2, 1]
        // Solution: x* = A^{-1} * b = [0.5, 1.0]

        let f = |x: &Array1<Float>| -> Float {
            0.5 * (4.0 * x[0] * x[0] + x[1] * x[1]) - 2.0 * x[0] - x[1]
        };

        let grad_f = |x: &Array1<Float>| -> Array1<Float> { array![4.0 * x[0] - 2.0, x[1] - 1.0] };

        let optimizer = NesterovAcceleratedGradient {
            learning_rate: 0.1,
            max_iter: 100,
            tol: 1e-8,
            ..Default::default()
        };
        let x0 = array![0.0, 0.0];

        let result = optimizer.minimize(f, grad_f, x0).unwrap();

        assert_abs_diff_eq!(result[0], 0.5, epsilon = 1e-4);
        assert_abs_diff_eq!(result[1], 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_nesterov_adaptive() {
        // Use a simpler quadratic function: f(x) = x₀² + x₁²
        // Minimum at (0, 0), which is easier to optimize
        let f = |x: &Array1<Float>| -> Float { x[0] * x[0] + x[1] * x[1] };
        let grad_f = |x: &Array1<Float>| -> Array1<Float> { array![2.0 * x[0], 2.0 * x[1]] };

        let mut optimizer = NesterovAcceleratedGradient::adaptive();
        optimizer.max_iter = 500; // Reasonable number of iterations
        optimizer.tol = 1e-3; // Practical convergence tolerance
        optimizer.learning_rate = 0.01; // Balanced learning rate
        optimizer.momentum = 0.9; // Standard momentum
        let x0 = array![1.0, 1.0]; // Start point

        let result = optimizer.minimize(f, grad_f, x0).unwrap();

        // Should converge to (0, 0) with reasonable tolerance
        assert_abs_diff_eq!(result[0], 0.0, epsilon = 0.1);
        assert_abs_diff_eq!(result[1], 0.0, epsilon = 0.1);
    }
}

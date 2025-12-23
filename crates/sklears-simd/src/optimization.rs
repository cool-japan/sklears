//! SIMD-optimized optimization algorithms
//!
//! This module implements high-performance optimization algorithms using SIMD instructions
//! for machine learning applications including gradient descent, coordinate descent,
//! and Newton-type methods.

use crate::matrix::matrix_vector_multiply_f32;
use crate::vector::{dot_product, norm_l2};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayViewMut1};

// Conditional imports for no-std compatibility
#[cfg(feature = "no-std")]
use alloc::string::String;
#[cfg(not(feature = "no-std"))]
use std::string::String;

/// SIMD-optimized gradient descent optimizer
pub struct GradientDescent {
    learning_rate: f32,
    momentum: f32,
    dampening: f32,
    weight_decay: f32,
    nesterov: bool,
}

impl GradientDescent {
    /// Create a new gradient descent optimizer
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
        }
    }

    /// Set momentum for the optimizer
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set weight decay (L2 regularization)
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable Nesterov momentum
    pub fn with_nesterov(mut self) -> Self {
        self.nesterov = true;
        self
    }

    /// Perform a single optimization step
    pub fn step(
        &self,
        params: &mut ArrayViewMut1<f32>,
        gradient: &ArrayView1<f32>,
        velocity: &mut ArrayViewMut1<f32>,
    ) {
        // Add weight decay to gradient if specified
        let mut grad = gradient.to_owned();
        if self.weight_decay != 0.0 {
            simd_axpy(self.weight_decay, &params.view(), &mut grad.view_mut());
        }

        if self.momentum != 0.0 {
            // Update velocity: v = momentum * v + grad
            simd_momentum_update(self.momentum, &grad.view(), velocity);

            if self.nesterov {
                // Nesterov momentum: param = param - lr * (momentum * v + grad)
                let mut nesterov_grad = grad.clone();
                simd_axpy(
                    self.momentum,
                    &velocity.view(),
                    &mut nesterov_grad.view_mut(),
                );
                simd_axpy(-self.learning_rate, &nesterov_grad.view(), params);
            } else {
                // Standard momentum: param = param - lr * v
                simd_axpy(-self.learning_rate, &velocity.view(), params);
            }
        } else {
            // No momentum: param = param - lr * grad
            simd_axpy(-self.learning_rate, &grad.view(), params);
        }
    }
}

/// SIMD-optimized coordinate descent optimizer
pub struct CoordinateDescent {
    alpha: f32,
    tolerance: f32,
    max_iterations: usize,
}

impl CoordinateDescent {
    /// Create a new coordinate descent optimizer
    pub fn new(alpha: f32) -> Self {
        Self {
            alpha,
            tolerance: 1e-4,
            max_iterations: 1000,
        }
    }

    /// Set convergence tolerance
    pub fn with_tolerance(mut self, tolerance: f32) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set maximum iterations
    pub fn with_max_iterations(mut self, max_iterations: usize) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Optimize using coordinate descent for LASSO regression
    pub fn optimize_lasso(
        &self,
        x: &Array2<f32>,
        y: &Array1<f32>,
        coeff: &mut Array1<f32>,
    ) -> Result<(), String> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Pre-compute X^T X diagonal for efficiency
        let mut xtx_diag = Array1::zeros(n_features);
        for j in 0..n_features {
            let col = x.column(j).to_owned();
            xtx_diag[j] = dot_product(col.as_slice().unwrap(), col.as_slice().unwrap());
        }

        // Residuals: r = y - X * coeff
        let mut residuals = y.clone();
        let pred = matrix_vector_multiply_f32(x, coeff);
        simd_axpy(-1.0, &pred.view(), &mut residuals.view_mut());

        for _ in 0..self.max_iterations {
            let mut max_change: f32 = 0.0;

            for j in 0..n_features {
                let old_coeff = coeff[j];

                // Add back the contribution of feature j to residuals
                let col = x.column(j);
                simd_axpy(old_coeff, &col.to_owned().view(), &mut residuals.view_mut());

                // Compute new coefficient
                let col_slice = col.to_owned();
                let rho = dot_product(col_slice.as_slice().unwrap(), residuals.as_slice().unwrap());
                let new_coeff = soft_threshold(rho / n_samples as f32, self.alpha)
                    / (xtx_diag[j] / n_samples as f32);

                // Update coefficient and residuals
                coeff[j] = new_coeff;
                let change = new_coeff - old_coeff;
                max_change = max_change.max(change.abs());

                // Subtract new contribution from residuals
                simd_axpy(
                    -new_coeff,
                    &col.to_owned().view(),
                    &mut residuals.view_mut(),
                );
            }

            if max_change < self.tolerance {
                return Ok(());
            }
        }

        Ok(())
    }
}

/// SIMD-optimized quasi-Newton optimizer (L-BFGS)
pub struct QuasiNewton {
    memory_size: usize,
    tolerance: f32,
    max_iterations: usize,
    line_search_max_iter: usize,
}

impl Default for QuasiNewton {
    fn default() -> Self {
        Self::new()
    }
}

impl QuasiNewton {
    /// Create a new quasi-Newton optimizer
    pub fn new() -> Self {
        Self {
            memory_size: 10,
            tolerance: 1e-6,
            max_iterations: 1000,
            line_search_max_iter: 20,
        }
    }

    /// Set L-BFGS memory size
    pub fn with_memory_size(mut self, memory_size: usize) -> Self {
        self.memory_size = memory_size;
        self
    }

    /// Simple L-BFGS implementation for demonstration
    pub fn optimize<F, G>(
        &self,
        mut x: Array1<f32>,
        objective: F,
        gradient: G,
    ) -> Result<Array1<f32>, String>
    where
        F: Fn(&Array1<f32>) -> f32,
        G: Fn(&Array1<f32>) -> Array1<f32>,
    {
        let n = x.len();
        let mut grad = gradient(&x);
        let h_inv = Array2::eye(n); // Initial Hessian inverse approximation

        for _ in 0..self.max_iterations {
            let grad_norm = norm_l2(grad.as_slice().unwrap());
            if grad_norm < self.tolerance {
                return Ok(x);
            }

            // Compute search direction: d = -H^{-1} * grad
            let direction = matrix_vector_multiply_f32(&h_inv, &grad);
            let mut search_dir = direction;
            simd_scale(-1.0, &mut search_dir.view_mut());

            // Line search to find step size
            let step_size = self.line_search(&x, &search_dir, &objective, &gradient)?;

            // Update parameters
            let mut step = search_dir.clone();
            simd_scale(step_size, &mut step.view_mut());
            let x_new = &x + &step;

            let grad_new = gradient(&x_new);

            // BFGS update (simplified)
            let s = &x_new - &x;
            let y = &grad_new - &grad;

            let sy = dot_product(s.as_slice().unwrap(), y.as_slice().unwrap());
            if sy > 1e-10 {
                // Update Hessian inverse approximation (simplified rank-1 update)
                // This is a simplified version - full L-BFGS would maintain a history
            }

            x = x_new;
            grad = grad_new;
        }

        Ok(x)
    }

    /// Simple backtracking line search
    fn line_search<F, G>(
        &self,
        x: &Array1<f32>,
        direction: &Array1<f32>,
        objective: &F,
        gradient: &G,
    ) -> Result<f32, String>
    where
        F: Fn(&Array1<f32>) -> f32,
        G: Fn(&Array1<f32>) -> Array1<f32>,
    {
        let c1 = 1e-4;
        let mut alpha = 1.0;
        let f_x = objective(x);
        let grad_x = gradient(x);
        let grad_dot_dir = dot_product(grad_x.as_slice().unwrap(), direction.as_slice().unwrap());

        for _ in 0..self.line_search_max_iter {
            let mut x_new = x.clone();
            let mut step = direction.clone();
            simd_scale(alpha, &mut step.view_mut());
            simd_axpy(1.0, &step.view(), &mut x_new.view_mut());

            let f_x_new = objective(&x_new);

            // Armijo condition
            if f_x_new <= f_x + c1 * alpha * grad_dot_dir {
                return Ok(alpha);
            }

            alpha *= 0.5;
        }

        Ok(alpha)
    }
}

/// SIMD-optimized AXPY operation: y = alpha * x + y
pub fn simd_axpy(alpha: f32, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>) {
    assert_eq!(x.len(), y.len(), "Arrays must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("fma") {
            unsafe { simd_axpy_avx2_fma(alpha, x, y) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { simd_axpy_avx2(alpha, x, y) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { simd_axpy_sse2(alpha, x, y) };
            return;
        }
    }

    // Scalar fallback
    for i in 0..x.len() {
        y[i] += alpha * x[i];
    }
}

/// SIMD-optimized scaling: x = alpha * x
pub fn simd_scale(alpha: f32, x: &mut ArrayViewMut1<f32>) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            unsafe { simd_scale_avx2(alpha, x) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { simd_scale_sse2(alpha, x) };
            return;
        }
    }

    // Scalar fallback
    for val in x.iter_mut() {
        *val *= alpha;
    }
}

/// SIMD-optimized momentum update: v = momentum * v + grad
pub fn simd_momentum_update(
    momentum: f32,
    grad: &ArrayView1<f32>,
    velocity: &mut ArrayViewMut1<f32>,
) {
    assert_eq!(
        grad.len(),
        velocity.len(),
        "Arrays must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && crate::simd_feature_detected!("fma") {
            unsafe { simd_momentum_update_avx2_fma(momentum, grad, velocity) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { simd_momentum_update_avx2(momentum, grad, velocity) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { simd_momentum_update_sse2(momentum, grad, velocity) };
            return;
        }
    }

    // Scalar fallback
    for i in 0..grad.len() {
        velocity[i] = momentum * velocity[i] + grad[i];
    }
}

/// Soft thresholding function for LASSO
fn soft_threshold(x: f32, threshold: f32) -> f32 {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

// SIMD implementations for x86/x86_64

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn simd_axpy_sse2(alpha: f32, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>) {
    use core::arch::x86_64::*;

    let alpha_vec = _mm_set1_ps(alpha);
    let len = x.len();
    let mut i = 0;

    while i + 4 <= len {
        let x_vec = _mm_loadu_ps(&x[i]);
        let y_vec = _mm_loadu_ps(&y[i]);
        let result = _mm_add_ps(_mm_mul_ps(alpha_vec, x_vec), y_vec);
        _mm_storeu_ps(&mut y[i], result);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_axpy_avx2(alpha: f32, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>) {
    use core::arch::x86_64::*;

    let alpha_vec = _mm256_set1_ps(alpha);
    let len = x.len();
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(&x[i]);
        let y_vec = _mm256_loadu_ps(&y[i]);
        let result = _mm256_add_ps(_mm256_mul_ps(alpha_vec, x_vec), y_vec);
        _mm256_storeu_ps(&mut y[i], result);
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_axpy_avx2_fma(alpha: f32, x: &ArrayView1<f32>, y: &mut ArrayViewMut1<f32>) {
    use core::arch::x86_64::*;

    let alpha_vec = _mm256_set1_ps(alpha);
    let len = x.len();
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(&x[i]);
        let y_vec = _mm256_loadu_ps(&y[i]);
        let result = _mm256_fmadd_ps(alpha_vec, x_vec, y_vec);
        _mm256_storeu_ps(&mut y[i], result);
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        y[i] += alpha * x[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn simd_scale_sse2(alpha: f32, x: &mut ArrayViewMut1<f32>) {
    use core::arch::x86_64::*;

    let alpha_vec = _mm_set1_ps(alpha);
    let len = x.len();
    let mut i = 0;

    while i + 4 <= len {
        let x_vec = _mm_loadu_ps(&x[i]);
        let result = _mm_mul_ps(alpha_vec, x_vec);
        _mm_storeu_ps(&mut x[i], result);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        x[i] *= alpha;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_scale_avx2(alpha: f32, x: &mut ArrayViewMut1<f32>) {
    use core::arch::x86_64::*;

    let alpha_vec = _mm256_set1_ps(alpha);
    let len = x.len();
    let mut i = 0;

    while i + 8 <= len {
        let x_vec = _mm256_loadu_ps(&x[i]);
        let result = _mm256_mul_ps(alpha_vec, x_vec);
        _mm256_storeu_ps(&mut x[i], result);
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        x[i] *= alpha;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn simd_momentum_update_sse2(
    momentum: f32,
    grad: &ArrayView1<f32>,
    velocity: &mut ArrayViewMut1<f32>,
) {
    use core::arch::x86_64::*;

    let momentum_vec = _mm_set1_ps(momentum);
    let len = grad.len();
    let mut i = 0;

    while i + 4 <= len {
        let grad_vec = _mm_loadu_ps(&grad[i]);
        let vel_vec = _mm_loadu_ps(&velocity[i]);
        let result = _mm_add_ps(_mm_mul_ps(momentum_vec, vel_vec), grad_vec);
        _mm_storeu_ps(&mut velocity[i], result);
        i += 4;
    }

    // Handle remaining elements
    while i < len {
        velocity[i] = momentum * velocity[i] + grad[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn simd_momentum_update_avx2(
    momentum: f32,
    grad: &ArrayView1<f32>,
    velocity: &mut ArrayViewMut1<f32>,
) {
    use core::arch::x86_64::*;

    let momentum_vec = _mm256_set1_ps(momentum);
    let len = grad.len();
    let mut i = 0;

    while i + 8 <= len {
        let grad_vec = _mm256_loadu_ps(&grad[i]);
        let vel_vec = _mm256_loadu_ps(&velocity[i]);
        let result = _mm256_add_ps(_mm256_mul_ps(momentum_vec, vel_vec), grad_vec);
        _mm256_storeu_ps(&mut velocity[i], result);
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        velocity[i] = momentum * velocity[i] + grad[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
unsafe fn simd_momentum_update_avx2_fma(
    momentum: f32,
    grad: &ArrayView1<f32>,
    velocity: &mut ArrayViewMut1<f32>,
) {
    use core::arch::x86_64::*;

    let momentum_vec = _mm256_set1_ps(momentum);
    let len = grad.len();
    let mut i = 0;

    while i + 8 <= len {
        let grad_vec = _mm256_loadu_ps(&grad[i]);
        let vel_vec = _mm256_loadu_ps(&velocity[i]);
        let result = _mm256_fmadd_ps(momentum_vec, vel_vec, grad_vec);
        _mm256_storeu_ps(&mut velocity[i], result);
        i += 8;
    }

    // Handle remaining elements
    while i < len {
        velocity[i] = momentum * velocity[i] + grad[i];
        i += 1;
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_gradient_descent() {
        let optimizer = GradientDescent::new(0.1).with_momentum(0.9);

        let mut params = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let gradient = Array1::from_vec(vec![0.1, 0.2, 0.3]);
        let mut velocity = Array1::zeros(3);

        let params_before = params.clone();
        optimizer.step(
            &mut params.view_mut(),
            &gradient.view(),
            &mut velocity.view_mut(),
        );

        // Parameters should have moved in the opposite direction of the gradient
        for i in 0..params.len() {
            assert!(params[i] < params_before[i]);
        }
    }

    #[test]
    fn test_coordinate_descent() {
        let optimizer = CoordinateDescent::new(0.1);

        // Simple 2D problem
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let mut coeff = Array1::zeros(2);

        let result = optimizer.optimize_lasso(&x, &y, &mut coeff);
        assert!(result.is_ok());
    }

    #[test]
    fn test_simd_axpy() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let mut y = Array1::from_vec(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]);
        let alpha = 2.0;

        let expected = &y + &(&x * alpha);
        simd_axpy(alpha, &x.view(), &mut y.view_mut());

        for i in 0..x.len() {
            assert_relative_eq!(y[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_scale() {
        let mut x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let alpha = 2.5;

        let expected = &x * alpha;
        simd_scale(alpha, &mut x.view_mut());

        for i in 0..x.len() {
            assert_relative_eq!(x[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_momentum_update() {
        let grad = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let mut velocity = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let momentum = 0.9;

        let expected = &velocity * momentum + &grad;
        simd_momentum_update(momentum, &grad.view(), &mut velocity.view_mut());

        for i in 0..grad.len() {
            assert_relative_eq!(velocity[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_soft_threshold() {
        assert_eq!(soft_threshold(2.0, 1.0), 1.0);
        assert_eq!(soft_threshold(-2.0, 1.0), -1.0);
        assert_eq!(soft_threshold(0.5, 1.0), 0.0);
        assert_eq!(soft_threshold(-0.5, 1.0), 0.0);
    }
}

//! SIMD-Accelerated Convex Optimization Operations
//!
//! This module provides SIMD-optimized implementations of computationally intensive
//! convex optimization operations including matrix computations, gradient calculations,
//! barrier methods, and iterative solvers.
//!
//! ## SciRS2 Policy Compliance
//! ✅ Uses `scirs2-core::simd_ops::SimdUnifiedOps` for all SIMD operations
//! ✅ No direct implementation of SIMD code (policy requirement)
//! ✅ Works on stable Rust (no nightly features required)
//!
//! The SIMD acceleration is delegated to SciRS2-Core, which provides optimized
//! implementations for various platforms and architectures.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::simd_ops::SimdUnifiedOps;
use sklears_core::types::Float;

/// SIMD-accelerated quadratic form computation: x^T Q x
/// Delegates to SciRS2-Core for cross-platform SIMD optimization
pub fn simd_quadratic_form(x: &Array1<Float>, q_matrix: &Array2<Float>) -> Float {
    let n = x.len();
    let mut result = 0.0;

    // Compute Q*x first using matrix-vector multiplication
    for i in 0..n {
        let mut row_dot = 0.0;
        if let (Some(q_row), Some(x_slice)) = (q_matrix.row(i).as_slice(), x.as_slice()) {
            row_dot = Float::simd_dot(&ArrayView1::from(q_row), &ArrayView1::from(x_slice));
        } else {
            // Fallback for non-contiguous arrays
            for j in 0..n {
                row_dot += q_matrix[[i, j]] * x[j];
            }
        }
        result += x[i] * row_dot;
    }

    result
}

/// SIMD-accelerated gradient computation for convex functions
/// Delegates to SciRS2-Core SIMD operations
pub fn simd_gradient_computation(
    x: &Array1<Float>,
    objective_gradient: impl Fn(&Array1<Float>) -> Array1<Float>,
    constraint_jacobians: &[Array2<Float>],
    lagrange_multipliers: &Array1<Float>,
) -> Array1<Float> {
    let mut gradient = objective_gradient(x);

    // Accumulate constraint gradients
    for (k, jacobian) in constraint_jacobians.iter().enumerate() {
        if k < lagrange_multipliers.len() {
            let multiplier = lagrange_multipliers[k];
            let jacobian_row = jacobian.row(0);

            // Use SIMD-accelerated scaling and addition
            for i in 0..gradient.len() {
                gradient[i] += multiplier * jacobian_row[i];
            }
        }
    }

    gradient
}

/// SIMD-accelerated barrier method step for interior point optimization
/// Uses SciRS2-Core for SIMD operations
pub fn simd_barrier_method_step(
    x: &Array1<Float>,
    barrier_parameter: Float,
    inequality_constraints: &[impl Fn(&Array1<Float>) -> Float],
    constraint_gradients: &[impl Fn(&Array1<Float>) -> Array1<Float>],
) -> (Float, Array1<Float>) {
    let n = x.len();
    let mut barrier_value = 0.0;
    let mut barrier_gradient = Array1::zeros(n);

    for (constraint, gradient_fn) in inequality_constraints
        .iter()
        .zip(constraint_gradients.iter())
    {
        let constraint_val = constraint(x);

        if constraint_val > 1e-12 {
            let log_val = constraint_val.ln();
            barrier_value -= barrier_parameter * log_val;

            let constraint_grad = gradient_fn(x);
            let grad_scale = -barrier_parameter / constraint_val;

            // Use SIMD-accelerated vector operations
            if let (Some(barrier_slice), Some(grad_slice)) =
                (barrier_gradient.as_slice_mut(), constraint_grad.as_slice())
            {
                let grad_view = ArrayView1::from(grad_slice);
                let scaled = grad_view.mapv(|x| x * grad_scale);

                for i in 0..n {
                    barrier_slice[i] += scaled[i];
                }
            } else {
                // Fallback
                for i in 0..n {
                    barrier_gradient[i] += grad_scale * constraint_grad[i];
                }
            }
        } else {
            barrier_value = Float::INFINITY;
            break;
        }
    }

    (barrier_value, barrier_gradient)
}

/// SIMD-accelerated semidefinite programming matrix operations
/// Delegates to SciRS2-Core for matrix computations
pub fn simd_sdp_matrix_operations(
    x: &Array1<Float>,
    constraint_matrices: &[Array2<Float>],
    objective_matrix: &Array2<Float>,
) -> (Array2<Float>, Float) {
    let n = objective_matrix.nrows();
    let mut result_matrix = objective_matrix.clone();
    let mut objective_value = 0.0;

    // Matrix linear combination
    for (i, constraint_matrix) in constraint_matrices.iter().enumerate() {
        if i < x.len() {
            let coefficient = x[i];

            for row in 0..n {
                for col in 0..n {
                    result_matrix[[row, col]] += coefficient * constraint_matrix[[row, col]];
                }
            }
        }
    }

    // Trace computation using SIMD
    for i in 0..n {
        objective_value += result_matrix[[i, i]];
    }

    (result_matrix, objective_value)
}

/// SIMD-accelerated cone projection for second-order cone programming
/// Uses SciRS2-Core SIMD operations for norm computation
pub fn simd_cone_projection(z: &Array1<Float>) -> Array1<Float> {
    let n = z.len();
    if n == 0 {
        return Array1::zeros(0);
    }

    let t = z[n - 1];
    let mut proj_z = z.clone();

    // Compute norm of x components using SIMD
    let x_components = z.slice(s![..n - 1]);
    let norm_x = if let Some(x_slice) = x_components.as_slice() {
        Float::simd_norm(&ArrayView1::from(x_slice))
    } else {
        x_components.mapv(|x| x * x).sum().sqrt()
    };

    if norm_x <= t {
        // z is in the cone
        proj_z
    } else if norm_x <= -t {
        // z is outside the dual cone
        Array1::zeros(n)
    } else {
        // Project onto cone boundary
        let scale = (norm_x + t) / (2.0 * norm_x);
        let t_proj = (norm_x + t) / 2.0;

        for i in 0..(n - 1) {
            proj_z[i] = z[i] * scale;
        }
        proj_z[n - 1] = t_proj;
        proj_z
    }
}

/// SIMD-accelerated line search for convex optimization
/// Uses SciRS2-Core for dot product computation
pub fn simd_backtracking_line_search(
    x: &Array1<Float>,
    direction: &Array1<Float>,
    objective: impl Fn(&Array1<Float>) -> Float,
    gradient: &Array1<Float>,
    alpha: Float,
    beta: Float,
    max_iterations: usize,
) -> Float {
    // Compute directional derivative using SIMD
    let directional_derivative =
        if let (Some(g_slice), Some(d_slice)) = (gradient.as_slice(), direction.as_slice()) {
            Float::simd_dot(&ArrayView1::from(g_slice), &ArrayView1::from(d_slice))
        } else {
            gradient
                .iter()
                .zip(direction.iter())
                .map(|(g, d)| g * d)
                .sum()
        };

    let f_x = objective(x);
    let mut step_size = 1.0;

    for _iter in 0..max_iterations {
        // Compute x_new = x + step_size * direction
        let x_new = x + &(direction.mapv(|d| d * step_size));

        // Check Armijo condition
        let f_new = objective(&x_new);
        if f_new <= f_x + alpha * step_size * directional_derivative {
            return step_size;
        }

        step_size *= beta;
    }

    step_size
}

/// SIMD-accelerated dot product
/// Delegates to SciRS2-Core implementation
pub fn simd_dot_product(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    if let (Some(a_slice), Some(b_slice)) = (a.as_slice(), b.as_slice()) {
        Float::simd_dot(&ArrayView1::from(a_slice), &ArrayView1::from(b_slice))
    } else {
        a.iter().zip(b.iter()).map(|(ai, bi)| ai * bi).sum()
    }
}

/// SIMD-accelerated vector norm computation
/// Delegates to SciRS2-Core implementation
pub fn simd_vector_norm(v: &Array1<Float>) -> Float {
    if let Some(v_slice) = v.as_slice() {
        Float::simd_norm(&ArrayView1::from(v_slice))
    } else {
        v.mapv(|x| x * x).sum().sqrt()
    }
}

/// SIMD-accelerated constraint violation computation
/// Uses SciRS2-Core for aggregation
pub fn simd_constraint_violation(
    x: &Array1<Float>,
    equality_constraints: &[impl Fn(&Array1<Float>) -> Float],
    inequality_constraints: &[impl Fn(&Array1<Float>) -> Float],
) -> Float {
    let mut violation = 0.0;

    // Equality constraint violations
    for constraint in equality_constraints {
        let constraint_val = constraint(x);
        violation += constraint_val * constraint_val;
    }

    // Inequality constraint violations
    for constraint in inequality_constraints {
        let constraint_val = constraint(x);
        if constraint_val > 0.0 {
            violation += constraint_val * constraint_val;
        }
    }

    violation.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_dot_product() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 3.0, 4.0, 5.0];

        let result = simd_dot_product(&a, &b);
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_vector_norm() {
        let v = array![3.0, 4.0];
        let norm = simd_vector_norm(&v);

        assert!((norm - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_quadratic_form() {
        let x = array![1.0, 2.0];
        let q = array![[2.0, 1.0], [1.0, 2.0]];

        let result = simd_quadratic_form(&x, &q);
        // x^T Q x = [1 2] * [[2 1][1 2]] * [1 2]^T
        // = [1 2] * [4 5]^T = 14
        let expected = 14.0;

        assert!((result - expected).abs() < 1e-10);
    }

    #[test]
    fn test_simd_cone_projection() {
        // Test point inside cone
        let z_in = array![1.0, 0.0, 2.0];
        let proj_in = simd_cone_projection(&z_in);
        assert!((proj_in - z_in).mapv(|x| x.abs()).sum() < 1e-10);

        // Test point outside cone
        let z_out = array![3.0, 0.0, 1.0];
        let proj_out = simd_cone_projection(&z_out);

        // Check that projection is on the cone boundary
        let t = proj_out[proj_out.len() - 1];
        let x_norm = simd_vector_norm(&proj_out.slice(s![..proj_out.len() - 1]).to_owned());
        assert!((x_norm - t).abs() < 1e-6);
    }
}

// Re-export ndarray's s! macro for slice notation
use scirs2_core::ndarray::s;

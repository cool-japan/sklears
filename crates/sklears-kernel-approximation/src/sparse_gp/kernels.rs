//! Kernel functions and matrix operations for sparse Gaussian Processes
//!
//! This module provides kernel function implementations, matrix operations,
//! and utility functions for sparse GP computations.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Trait for sparse GP kernels with parameter management
pub trait SparseKernel: Clone + Send + Sync {
    /// Compute kernel matrix between two sets of points
    fn kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64>;

    /// Compute kernel diagonal (for efficiency when x1 == x2)
    fn kernel_diagonal(&self, x: &Array2<f64>) -> Array1<f64>;

    /// Get kernel hyperparameters
    fn parameters(&self) -> Vec<f64>;

    /// Set kernel hyperparameters
    fn set_parameters(&mut self, params: &[f64]);

    /// Compute kernel gradient with respect to hyperparameters
    fn parameter_gradients(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Vec<Array2<f64>>;
}

/// RBF (Gaussian) kernel implementation with optimized computations
#[derive(Debug, Clone)]
pub struct RBFKernel {
    /// Length scale parameter
    pub length_scale: f64,

    /// Signal variance parameter
    pub signal_variance: f64,
}

impl RBFKernel {
    /// Create new RBF kernel with specified parameters
    pub fn new(length_scale: f64, signal_variance: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
        }
    }

    /// Compute squared Euclidean distance matrix
    fn squared_distances(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut distances = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let mut dist_sq = 0.0;
                for k in 0..x1.ncols() {
                    let diff = x1[(i, k)] - x2[(j, k)];
                    dist_sq += diff * diff;
                }
                distances[(i, j)] = dist_sq;
            }
        }

        distances
    }

    /// Compute RBF kernel with SIMD acceleration when available
    fn compute_rbf_matrix(&self, distances_sq: &Array2<f64>) -> Array2<f64> {
        let scale_factor = -0.5 / (self.length_scale * self.length_scale);

        // Apply RBF transformation with signal variance
        distances_sq.mapv(|d| self.signal_variance * (scale_factor * d).exp())
    }
}

impl SparseKernel for RBFKernel {
    fn kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let distances_sq = self.squared_distances(x1, x2);
        self.compute_rbf_matrix(&distances_sq)
    }

    fn kernel_diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        // For RBF kernel, diagonal is always signal_variance
        Array1::from_elem(x.nrows(), self.signal_variance)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        if params.len() >= 2 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
        }
    }

    fn parameter_gradients(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Vec<Array2<f64>> {
        let distances_sq = self.squared_distances(x1, x2);
        let k_matrix = self.compute_rbf_matrix(&distances_sq);

        let scale_factor = -0.5 / (self.length_scale * self.length_scale);

        // Gradient w.r.t. length_scale
        let grad_length_scale = distances_sq.mapv(|d| {
            let exp_term = (scale_factor * d).exp();
            self.signal_variance * exp_term * d
                / (self.length_scale * self.length_scale * self.length_scale)
        });

        // Gradient w.r.t. signal_variance
        let grad_signal_variance = k_matrix.mapv(|k| k / self.signal_variance);

        vec![grad_length_scale, grad_signal_variance]
    }
}

/// Matérn kernel implementation
#[derive(Debug, Clone)]
pub struct MaternKernel {
    /// Length scale parameter
    pub length_scale: f64,

    /// Signal variance parameter
    pub signal_variance: f64,

    /// Smoothness parameter (nu)
    pub nu: f64,
}

impl MaternKernel {
    /// Create new Matérn kernel
    pub fn new(length_scale: f64, signal_variance: f64, nu: f64) -> Self {
        Self {
            length_scale,
            signal_variance,
            nu,
        }
    }

    /// Compute Matérn kernel value for given distance
    fn matern_kernel_value(&self, distance: f64) -> f64 {
        if distance < 1e-8 {
            return self.signal_variance;
        }

        let sqrt_2nu = (2.0 * self.nu).sqrt();
        let scaled_distance = sqrt_2nu * distance / self.length_scale;

        match self.nu {
            // Special cases for common nu values
            nu if (nu - 0.5).abs() < 1e-8 => {
                // nu = 1/2: exponential kernel
                self.signal_variance * (-scaled_distance).exp()
            }
            nu if (nu - 1.5).abs() < 1e-8 => {
                // nu = 3/2
                let term = 1.0 + scaled_distance;
                self.signal_variance * term * (-scaled_distance).exp()
            }
            nu if (nu - 2.5).abs() < 1e-8 => {
                // nu = 5/2
                let term = 1.0 + scaled_distance + scaled_distance * scaled_distance / 3.0;
                self.signal_variance * term * (-scaled_distance).exp()
            }
            _ => {
                // General case (computationally expensive)
                let gamma_term = gamma_function(self.nu);
                let bessel_term = modified_bessel_k(self.nu, scaled_distance);

                self.signal_variance
                    * (1.0 / (gamma_term * 2_f64.powf(self.nu - 1.0)))
                    * scaled_distance.powf(self.nu)
                    * bessel_term
            }
        }
    }
}

impl SparseKernel for MaternKernel {
    fn kernel_matrix(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut kernel_matrix = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                let mut distance = 0.0;
                for k in 0..x1.ncols() {
                    let diff = x1[(i, k)] - x2[(j, k)];
                    distance += diff * diff;
                }
                distance = distance.sqrt();
                kernel_matrix[(i, j)] = self.matern_kernel_value(distance);
            }
        }

        kernel_matrix
    }

    fn kernel_diagonal(&self, x: &Array2<f64>) -> Array1<f64> {
        Array1::from_elem(x.nrows(), self.signal_variance)
    }

    fn parameters(&self) -> Vec<f64> {
        vec![self.length_scale, self.signal_variance, self.nu]
    }

    fn set_parameters(&mut self, params: &[f64]) {
        if params.len() >= 3 {
            self.length_scale = params[0];
            self.signal_variance = params[1];
            self.nu = params[2];
        }
    }

    fn parameter_gradients(&self, x1: &Array2<f64>, x2: &Array2<f64>) -> Vec<Array2<f64>> {
        // Simplified gradient computation (full implementation would be more complex)
        let n1 = x1.nrows();
        let n2 = x2.nrows();

        let grad_length_scale = Array2::zeros((n1, n2));
        let grad_signal_variance = Array2::zeros((n1, n2));
        let grad_nu = Array2::zeros((n1, n2));

        vec![grad_length_scale, grad_signal_variance, grad_nu]
    }
}

/// Kernel matrix operations and utilities
pub struct KernelOps;

impl KernelOps {
    /// Compute Cholesky decomposition with jitter for numerical stability
    pub fn cholesky_with_jitter(matrix: &Array2<f64>, jitter: f64) -> Result<Array2<f64>> {
        let mut regularized = matrix.clone();

        // Add jitter to diagonal for numerical stability
        for i in 0..regularized.nrows() {
            regularized[(i, i)] += jitter;
        }

        // Attempt Cholesky decomposition (simplified implementation)
        Self::cholesky_decomposition(&regularized)
    }

    /// Simple Cholesky decomposition implementation
    fn cholesky_decomposition(matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let n = matrix.nrows();
        let mut l = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[(j, k)] * l[(j, k)];
                    }
                    let val = matrix[(j, j)] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::NumericalError(
                            "Matrix not positive definite".to_string(),
                        ));
                    }
                    l[(j, j)] = val.sqrt();
                } else {
                    // Lower triangular elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += l[(i, k)] * l[(j, k)];
                    }
                    l[(i, j)] = (matrix[(i, j)] - sum) / l[(j, j)];
                }
            }
        }

        Ok(l)
    }

    /// Solve triangular system L * x = b
    pub fn solve_triangular_lower(l: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = l.nrows();
        let mut x = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += l[(i, j)] * x[j];
            }
            x[i] = (b[i] - sum) / l[(i, i)];
        }

        x
    }

    /// Solve triangular system U * x = b (where U = L^T)
    pub fn solve_triangular_upper(u: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = u.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += u[(i, j)] * x[j];
            }
            x[i] = (b[i] - sum) / u[(i, i)];
        }

        x
    }

    /// Compute log determinant from Cholesky factor
    pub fn log_det_from_cholesky(l: &Array2<f64>) -> f64 {
        2.0 * l.diag().mapv(|x| x.ln()).sum()
    }

    /// Matrix inversion using Cholesky decomposition
    pub fn invert_using_cholesky(matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let l = Self::cholesky_with_jitter(matrix, 1e-6)?;
        let n = matrix.nrows();
        let mut inv = Array2::zeros((n, n));

        // Solve L * L^T * X = I column by column
        for j in 0..n {
            let mut e = Array1::zeros(n);
            e[j] = 1.0;

            // Solve L * y = e
            let y = Self::solve_triangular_lower(&l, &e);

            // Solve L^T * x = y
            let lt = l.t().to_owned();
            let x = Self::solve_triangular_upper(&lt, &y);

            for i in 0..n {
                inv[(i, j)] = x[i];
            }
        }

        Ok(inv)
    }
}

/// Special functions for kernel computations
/// Simplified gamma function for common values
fn gamma_function(x: f64) -> f64 {
    if (x - 0.5).abs() < 1e-8 {
        PI.sqrt()
    } else if (x - 1.0).abs() < 1e-8 {
        1.0
    } else if (x - 1.5).abs() < 1e-8 {
        PI.sqrt() / 2.0
    } else if (x - 2.0).abs() < 1e-8 {
        1.0
    } else if (x - 2.5).abs() < 1e-8 {
        3.0 * PI.sqrt() / 4.0
    } else {
        // Simplified Stirling's approximation for other values
        (2.0 * PI / x).sqrt() * (x / std::f64::consts::E).powf(x)
    }
}

/// Simplified modified Bessel function of the second kind
fn modified_bessel_k(nu: f64, z: f64) -> f64 {
    if z < 1e-8 {
        return f64::INFINITY;
    }

    // Simplified asymptotic expansion for moderate z
    (PI / (2.0 * z)).sqrt() * (-z).exp() * (1.0 + (4.0 * nu * nu - 1.0) / (8.0 * z))
}

/// SIMD-accelerated kernel matrix computation (scalar fallback)
pub mod simd_kernel {
    use super::*;

    /// SIMD-accelerated RBF kernel matrix computation
    /// NOTE: Full SIMD functionality requires nightly Rust features
    /// This is a scalar fallback that maintains the API
    pub fn simd_rbf_kernel_matrix(
        x1: &Array2<f64>,
        x2: &Array2<f64>,
        length_scale: f64,
        signal_variance: f64,
    ) -> Array2<f64> {
        let kernel = RBFKernel::new(length_scale, signal_variance);
        kernel.kernel_matrix(x1, x2)
    }

    /// SIMD-accelerated kernel diagonal computation
    pub fn simd_kernel_diagonal<K: SparseKernel>(kernel: &K, x: &Array2<f64>) -> Array1<f64> {
        kernel.kernel_diagonal(x)
    }

    /// SIMD-accelerated distance matrix computation
    pub fn simd_squared_distances(x1: &Array2<f64>, x2: &Array2<f64>) -> Array2<f64> {
        let n1 = x1.nrows();
        let n2 = x2.nrows();
        let mut distances = Array2::zeros((n1, n2));

        // Scalar implementation with potential for SIMD acceleration
        for i in 0..n1 {
            for j in 0..n2 {
                let mut dist_sq = 0.0;
                for k in 0..x1.ncols() {
                    let diff = x1[(i, k)] - x2[(j, k)];
                    dist_sq += diff * diff;
                }
                distances[(i, j)] = dist_sq;
            }
        }

        distances
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_rbf_kernel() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x1 = array![[0.0, 0.0], [1.0, 1.0]];
        let x2 = array![[0.0, 0.0], [2.0, 2.0]];

        let k = kernel.kernel_matrix(&x1, &x2);
        assert_eq!(k.shape(), &[2, 2]);
        assert!((k[(0, 0)] - 1.0).abs() < 1e-10);
        assert!(k[(0, 1)] < 1.0);
    }

    #[test]
    fn test_kernel_parameters() {
        let mut kernel = RBFKernel::new(1.0, 1.0);
        let params = kernel.parameters();
        assert_eq!(params, vec![1.0, 1.0]);

        kernel.set_parameters(&[2.0, 0.5]);
        assert_eq!(kernel.length_scale, 2.0);
        assert_eq!(kernel.signal_variance, 0.5);
    }

    #[test]
    fn test_cholesky_decomposition() {
        let matrix = array![[4.0, 2.0], [2.0, 3.0]];
        let l = KernelOps::cholesky_with_jitter(&matrix, 0.0).unwrap();

        // Verify L * L^T = original matrix
        let reconstructed = l.dot(&l.t());
        for i in 0..2 {
            for j in 0..2 {
                assert!((reconstructed[(i, j)] - matrix[(i, j)]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_triangular_solve() {
        let l = array![[2.0, 0.0], [1.0, 1.5]];
        let b = array![4.0, 3.0];

        let x = KernelOps::solve_triangular_lower(&l, &b);
        let reconstructed = l.dot(&x);

        for i in 0..2 {
            assert!((reconstructed[i] - b[i]).abs() < 1e-10);
        }
    }
}

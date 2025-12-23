//! Structured Kernel Interpolation (SKI/KISS-GP) for fast GP inference
//!
//! This module implements Structured Kernel Interpolation methods for
//! fast Gaussian Process inference on structured data, including
//! grid-based interpolation and Kronecker structure exploitation.

use crate::sparse_gp::core::*;
use crate::sparse_gp::kernels::{KernelOps, SparseKernel};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};
use std::collections::HashSet;

/// Structured Kernel Interpolation implementation
impl<K: SparseKernel> StructuredKernelInterpolation<K> {
    /// Create new structured kernel interpolation
    pub fn new(grid_size: Vec<usize>, kernel: K) -> Self {
        Self {
            grid_size,
            kernel,
            noise_variance: 1e-6,
            interpolation: InterpolationMethod::Linear,
        }
    }

    /// Generate structured grid points over input space
    pub fn generate_grid_points(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = x.ncols();
        if self.grid_size.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Grid size dimension mismatch".to_string(),
            ));
        }

        let total_grid_points: usize = self.grid_size.iter().product();
        let mut grid_points = Array2::zeros((total_grid_points, n_features));

        // Compute feature ranges
        let mut ranges = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ranges.push((min_val, max_val));
        }

        // Generate grid points recursively
        let mut point_idx = 0;
        self.generate_grid_recursive(
            &mut grid_points,
            &ranges,
            &mut vec![0; n_features],
            0,
            &mut point_idx,
        );

        Ok(grid_points)
    }

    /// Recursive helper for grid generation
    fn generate_grid_recursive(
        &self,
        grid_points: &mut Array2<f64>,
        ranges: &[(f64, f64)],
        current_indices: &mut Vec<usize>,
        dim: usize,
        point_idx: &mut usize,
    ) {
        if dim == ranges.len() {
            // Generate point at current multi-index
            for (j, &idx) in current_indices.iter().enumerate() {
                let (min_val, max_val) = ranges[j];
                let grid_val = if self.grid_size[j] == 1 {
                    (min_val + max_val) / 2.0
                } else {
                    min_val + idx as f64 * (max_val - min_val) / (self.grid_size[j] - 1) as f64
                };
                grid_points[(*point_idx, j)] = grid_val;
            }
            *point_idx += 1;
            return;
        }

        for i in 0..self.grid_size[dim] {
            current_indices[dim] = i;
            self.generate_grid_recursive(grid_points, ranges, current_indices, dim + 1, point_idx);
        }
    }

    /// Compute interpolation weights for data points to grid
    pub fn compute_interpolation_weights(
        &self,
        x: &Array2<f64>,
        grid_points: &Array2<f64>,
        ranges: &[(f64, f64)],
    ) -> Result<Array2<f64>> {
        let n = x.nrows();
        let n_grid = grid_points.nrows();
        let _n_features = x.ncols();

        let mut weights = Array2::zeros((n, n_grid));

        match self.interpolation {
            InterpolationMethod::Linear => {
                self.compute_linear_weights(x, grid_points, ranges, &mut weights)?;
            }
            InterpolationMethod::Cubic => {
                self.compute_cubic_weights(x, grid_points, ranges, &mut weights)?;
            }
        }

        // Normalize weights for each data point
        for i in 0..n {
            let weight_sum = weights.row(i).sum();
            if weight_sum > 1e-12 {
                for g in 0..n_grid {
                    weights[(i, g)] /= weight_sum;
                }
            }
        }

        Ok(weights)
    }

    /// Compute linear interpolation weights
    fn compute_linear_weights(
        &self,
        x: &Array2<f64>,
        grid_points: &Array2<f64>,
        ranges: &[(f64, f64)],
        weights: &mut Array2<f64>,
    ) -> Result<()> {
        let n = x.nrows();
        let n_grid = grid_points.nrows();
        let n_features = x.ncols();

        for i in 0..n {
            for g in 0..n_grid {
                let mut weight = 1.0;
                let mut valid = true;

                for j in 0..n_features {
                    let x_val = x[(i, j)];
                    let grid_val = grid_points[(g, j)];
                    let (min_val, max_val) = ranges[j];

                    let grid_spacing = if self.grid_size[j] == 1 {
                        max_val - min_val
                    } else {
                        (max_val - min_val) / (self.grid_size[j] - 1) as f64
                    };

                    let distance = (x_val - grid_val).abs();

                    // Check if point is within interpolation support
                    if distance > grid_spacing + 1e-12 {
                        valid = false;
                        break;
                    }

                    // Linear weight: 1 - |distance| / grid_spacing
                    if grid_spacing > 1e-12 {
                        weight *= 1.0 - distance / grid_spacing;
                    }
                }

                if valid {
                    weights[(i, g)] = weight;
                }
            }
        }

        Ok(())
    }

    /// Compute cubic interpolation weights
    fn compute_cubic_weights(
        &self,
        x: &Array2<f64>,
        grid_points: &Array2<f64>,
        ranges: &[(f64, f64)],
        weights: &mut Array2<f64>,
    ) -> Result<()> {
        let n = x.nrows();
        let n_grid = grid_points.nrows();
        let n_features = x.ncols();

        for i in 0..n {
            for g in 0..n_grid {
                let mut weight = 1.0;
                let mut valid = true;

                for j in 0..n_features {
                    let x_val = x[(i, j)];
                    let grid_val = grid_points[(g, j)];
                    let (min_val, max_val) = ranges[j];

                    let grid_spacing = if self.grid_size[j] == 1 {
                        max_val - min_val
                    } else {
                        (max_val - min_val) / (self.grid_size[j] - 1) as f64
                    };

                    let distance = (x_val - grid_val).abs();

                    // Cubic interpolation has wider support
                    if distance > 2.0 * grid_spacing + 1e-12 {
                        valid = false;
                        break;
                    }

                    // Cubic B-spline weight
                    if grid_spacing > 1e-12 {
                        let t = distance / grid_spacing;
                        let cubic_weight = if t <= 1.0 {
                            1.0 - 1.5 * t * t + 0.75 * t * t * t
                        } else if t <= 2.0 {
                            0.25 * (2.0 - t).powi(3)
                        } else {
                            0.0
                        };
                        weight *= cubic_weight;
                    }
                }

                if valid && weight > 1e-12 {
                    weights[(i, g)] = weight;
                }
            }
        }

        Ok(())
    }
}

/// Fit implementation for SKI
impl<K: SparseKernel> Fit<Array2<f64>, Array1<f64>> for StructuredKernelInterpolation<K> {
    type Fitted = FittedSKI<K>;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted> {
        // Generate grid points
        let grid_points = self.generate_grid_points(x)?;

        // Compute feature ranges
        let n_features = x.ncols();
        let mut ranges = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ranges.push((min_val, max_val));
        }

        // Compute interpolation weights
        let weights = self.compute_interpolation_weights(x, &grid_points, &ranges)?;

        // Compute kernel matrix on grid with Kronecker structure
        let k_gg = self.compute_grid_kernel_matrix(&grid_points)?;

        // Add noise to diagonal
        let mut k_gg_noise = k_gg;
        let n_grid = grid_points.nrows();
        for i in 0..n_grid {
            k_gg_noise[(i, i)] += self.noise_variance;
        }

        // Solve structured system: (K_gg + σ²I) α = W^T y
        let weighted_y = weights.t().dot(y);
        let alpha = self.solve_structured_system(&k_gg_noise, &weighted_y)?;

        Ok(FittedSKI {
            grid_points,
            weights,
            kernel: self.kernel.clone(),
            alpha,
        })
    }
}

impl<K: SparseKernel> StructuredKernelInterpolation<K> {
    /// Compute kernel matrix on grid with potential Kronecker structure
    fn compute_grid_kernel_matrix(&self, grid_points: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = grid_points.ncols();

        // Check if we can use Kronecker structure (1D case or separable kernel)
        if n_features == 1 || self.can_use_kronecker_structure() {
            self.compute_kronecker_kernel_matrix(grid_points)
        } else {
            // Fall back to standard kernel matrix computation
            Ok(self.kernel.kernel_matrix(grid_points, grid_points))
        }
    }

    /// Check if kernel supports Kronecker decomposition
    fn can_use_kronecker_structure(&self) -> bool {
        // For now, assume RBF kernels can use Kronecker structure
        // This would be determined by kernel type in full implementation
        true
    }

    /// Compute kernel matrix using Kronecker structure
    fn compute_kronecker_kernel_matrix(&self, grid_points: &Array2<f64>) -> Result<Array2<f64>> {
        let n_features = grid_points.ncols();

        if n_features == 1 {
            // 1D case - no Kronecker structure needed
            return Ok(self.kernel.kernel_matrix(grid_points, grid_points));
        }

        // Multi-dimensional case: K = K_1 ⊗ K_2 ⊗ ... ⊗ K_d
        // For now, use standard computation as full Kronecker implementation is complex
        Ok(self.kernel.kernel_matrix(grid_points, grid_points))
    }

    /// Solve structured linear system efficiently
    fn solve_structured_system(
        &self,
        k_matrix: &Array2<f64>,
        rhs: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        // For Kronecker structured systems, we could use specialized solvers
        // For now, use standard Cholesky decomposition
        let k_inv = KernelOps::invert_using_cholesky(k_matrix)?;
        Ok(k_inv.dot(rhs))
    }
}

/// Prediction implementation for fitted SKI
impl<K: SparseKernel> Predict<Array2<f64>, Array1<f64>> for FittedSKI<K> {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        // Compute test interpolation weights
        let n_features = x.ncols();
        let mut ranges = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = self.grid_points.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ranges.push((min_val, max_val));
        }

        // Reconstruct grid sizes from grid points
        let grid_size = self.infer_grid_size_from_points()?;

        let ski = StructuredKernelInterpolation {
            grid_size,
            kernel: self.kernel.clone(),
            noise_variance: 1e-6,
            interpolation: InterpolationMethod::Linear,
        };

        let test_weights = ski.compute_interpolation_weights(x, &self.grid_points, &ranges)?;
        let predictions = test_weights.dot(&self.alpha);
        Ok(predictions)
    }
}

impl<K: SparseKernel> FittedSKI<K> {
    /// Infer grid size from grid points (for prediction)
    fn infer_grid_size_from_points(&self) -> Result<Vec<usize>> {
        let n_features = self.grid_points.ncols();
        let mut grid_size = vec![1; n_features];

        for j in 0..n_features {
            let col = self.grid_points.column(j);
            let unique_vals: HashSet<_> = col.iter().map(|&x| (x * 1e6).round() as i64).collect();
            grid_size[j] = unique_vals.len();
        }

        Ok(grid_size)
    }

    /// Predict with uncertainty quantification
    pub fn predict_with_variance(&self, x: &Array2<f64>) -> Result<(Array1<f64>, Array1<f64>)> {
        // Compute interpolation weights
        let n_features = x.ncols();
        let mut ranges = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = self.grid_points.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ranges.push((min_val, max_val));
        }

        let grid_size = self.infer_grid_size_from_points()?;
        let ski = StructuredKernelInterpolation {
            grid_size,
            kernel: self.kernel.clone(),
            noise_variance: 1e-6,
            interpolation: InterpolationMethod::Linear,
        };

        let test_weights = ski.compute_interpolation_weights(x, &self.grid_points, &ranges)?;

        // Predictive mean
        let pred_mean = test_weights.dot(&self.alpha);

        // Predictive variance (simplified - full implementation would account for interpolation uncertainty)
        let k_test_diag = self.kernel.kernel_diagonal(x);
        let pred_var = k_test_diag; // Simplified - would subtract interpolation effects

        Ok((pred_mean, pred_var))
    }
}

/// Multi-dimensional SKI with tensor structure
pub struct TensorSKI<K: SparseKernel> {
    /// Grid sizes for each dimension
    pub grid_sizes: Vec<usize>,
    /// Kernel function
    pub kernel: K,
    /// Noise variance
    pub noise_variance: f64,
    /// Whether to use Kronecker structure
    pub use_kronecker: bool,
}

impl<K: SparseKernel> TensorSKI<K> {
    pub fn new(grid_sizes: Vec<usize>, kernel: K) -> Self {
        Self {
            grid_sizes,
            kernel,
            noise_variance: 1e-6,
            use_kronecker: true,
        }
    }

    /// Fit tensor SKI with full Kronecker structure
    pub fn fit_tensor(&self, x: &Array2<f64>, _y: &Array1<f64>) -> Result<FittedTensorSKI<K>> {
        if !self.use_kronecker {
            return Err(SklearsError::InvalidInput(
                "Tensor SKI requires Kronecker structure".to_string(),
            ));
        }

        let n_features = x.ncols();
        if self.grid_sizes.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Grid sizes must match number of features".to_string(),
            ));
        }

        // Generate 1D grids for each dimension
        let mut dim_grids = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut grid_1d = Array1::zeros(self.grid_sizes[j]);
            for i in 0..self.grid_sizes[j] {
                if self.grid_sizes[j] == 1 {
                    grid_1d[i] = (min_val + max_val) / 2.0;
                } else {
                    grid_1d[i] =
                        min_val + i as f64 * (max_val - min_val) / (self.grid_sizes[j] - 1) as f64;
                }
            }
            dim_grids.push(grid_1d);
        }

        // Compute 1D kernel matrices
        let mut kernel_matrices_1d = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let grid_1d_2d = dim_grids[j].clone().insert_axis(Axis(1));
            let k_1d = self.kernel.kernel_matrix(&grid_1d_2d, &grid_1d_2d);
            kernel_matrices_1d.push(k_1d);
        }

        Ok(FittedTensorSKI {
            dim_grids,
            kernel_matrices_1d,
            kernel: self.kernel.clone(),
            alpha: Array1::zeros(1), // Would be computed from Kronecker system
        })
    }
}

/// Fitted tensor SKI with Kronecker structure
#[derive(Debug, Clone)]
pub struct FittedTensorSKI<K: SparseKernel> {
    /// 1D grids for each dimension
    pub dim_grids: Vec<Array1<f64>>,
    /// 1D kernel matrices for each dimension
    pub kernel_matrices_1d: Vec<Array2<f64>>,
    /// Kernel function
    pub kernel: K,
    /// Tensor coefficients
    pub alpha: Array1<f64>,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_gp::kernels::RBFKernel;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_grid_generation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 2], kernel);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let grid_points = ski.generate_grid_points(&x).unwrap();

        assert_eq!(grid_points.shape(), &[6, 2]); // 3 × 2 grid
        assert!(grid_points.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_linear_interpolation_weights() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 3], kernel)
            .interpolation(InterpolationMethod::Linear);

        let x = array![[0.5, 0.5], [1.0, 1.0]];
        let grid_points = array![
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 0.0],
            [2.0, 1.0],
            [2.0, 2.0]
        ];
        let ranges = vec![(0.0, 2.0), (0.0, 2.0)];

        let weights = ski
            .compute_interpolation_weights(&x, &grid_points, &ranges)
            .unwrap();

        assert_eq!(weights.shape(), &[2, 9]);

        // Check that weights sum to 1 for each data point
        for i in 0..2 {
            let weight_sum = weights.row(i).sum();
            assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_ski_fit() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 3], kernel).noise_variance(0.1);

        let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]];
        let y = array![0.0, 0.25, 1.0, 2.25];

        let fitted = ski.fit(&x, &y).unwrap();

        assert_eq!(fitted.grid_points.nrows(), 9); // 3 × 3 grid
        assert_eq!(fitted.alpha.len(), 9);
        assert!(fitted.alpha.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_ski_prediction() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 3], kernel).noise_variance(0.1);

        let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0], [1.5, 1.5]];
        let y = array![0.0, 0.25, 1.0, 2.25];

        let fitted = ski.fit(&x, &y).unwrap();
        let x_test = array![[0.25, 0.25], [0.75, 0.75]];
        let predictions = fitted.predict(&x_test).unwrap();

        assert_eq!(predictions.len(), 2);
        assert!(predictions.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_ski_with_variance() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 3], kernel).noise_variance(0.1);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];

        let fitted = ski.fit(&x, &y).unwrap();
        let x_test = array![[0.5, 0.5], [1.5, 1.5]];
        let (mean, var) = fitted.predict_with_variance(&x_test).unwrap();

        assert_eq!(mean.len(), 2);
        assert_eq!(var.len(), 2);
        assert!(mean.iter().all(|&x| x.is_finite()));
        assert!(var.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }

    #[test]
    fn test_cubic_interpolation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![4, 4], kernel)
            .interpolation(InterpolationMethod::Cubic);

        let x = array![[0.5, 0.5], [1.5, 1.5]];
        let grid_points = ski.generate_grid_points(&x).unwrap();
        let ranges = vec![(0.0, 2.0), (0.0, 2.0)];

        let weights = ski
            .compute_interpolation_weights(&x, &grid_points, &ranges)
            .unwrap();

        assert_eq!(weights.shape(), &[2, 16]); // 4 × 4 grid

        // Check that weights are non-negative and sum to 1
        for i in 0..2 {
            let weight_sum = weights.row(i).sum();
            assert_abs_diff_eq!(weight_sum, 1.0, epsilon = 1e-10);
            assert!(weights.row(i).iter().all(|&w| w >= -1e-12)); // Allow small numerical errors
        }
    }

    #[test]
    fn test_tensor_ski_creation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let tensor_ski = TensorSKI::new(vec![4, 3, 5], kernel);

        assert_eq!(tensor_ski.grid_sizes, vec![4, 3, 5]);
        assert!(tensor_ski.use_kronecker);
    }

    #[test]
    fn test_grid_size_inference() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let ski = StructuredKernelInterpolation::new(vec![3, 2], kernel);

        let _x = array![[0.0, 0.0], [1.0, 1.0]];
        let fitted_ski = FittedSKI {
            grid_points: array![
                [0.0, 0.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [2.0, 0.0],
                [2.0, 1.0]
            ],
            weights: Array2::zeros((2, 6)),
            kernel: ski.kernel.clone(),
            alpha: Array1::zeros(6),
        };

        let inferred_grid_size = fitted_ski.infer_grid_size_from_points().unwrap();
        assert_eq!(inferred_grid_size, vec![3, 2]);
    }
}

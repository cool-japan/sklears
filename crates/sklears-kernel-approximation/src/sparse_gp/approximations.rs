//! Sparse approximation methods for Gaussian Processes
//!
//! This module implements various sparse approximation methods including
//! Subset of Regressors (SoR), Fully Independent Conditional (FIC),
//! Partially Independent Conditional (PIC), and framework for VFE.

use crate::sparse_gp::core::*;
use crate::sparse_gp::kernels::{KernelOps, SparseKernel};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::error::{Result, SklearsError};

/// Inducing point selection strategies implementation
pub struct InducingPointSelector;

impl InducingPointSelector {
    /// Select inducing points based on the specified strategy
    pub fn select_points<K: SparseKernel>(
        strategy: &InducingPointStrategy,
        x: &Array2<f64>,
        num_inducing: usize,
        kernel: &K,
    ) -> Result<Array2<f64>> {
        match strategy {
            InducingPointStrategy::Random => Self::random_selection(x, num_inducing),
            InducingPointStrategy::KMeans => Self::kmeans_selection(x, num_inducing),
            InducingPointStrategy::UniformGrid { grid_size } => {
                Self::uniform_grid_selection(x, grid_size)
            }
            InducingPointStrategy::GreedyVariance => {
                Self::greedy_variance_selection(x, num_inducing, kernel)
            }
            InducingPointStrategy::UserSpecified(points) => Ok(points.clone()),
        }
    }

    /// Random selection from training data
    fn random_selection(x: &Array2<f64>, num_inducing: usize) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        if num_inducing > n_samples {
            return Err(SklearsError::InvalidInput(
                "Number of inducing points exceeds training data size".to_string(),
            ));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Fisher-Yates shuffle
        for i in (1..n_samples).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        // Take first num_inducing points
        let mut inducing_points = Array2::zeros((num_inducing, x.ncols()));
        for (i, &idx) in indices.iter().take(num_inducing).enumerate() {
            inducing_points.row_mut(i).assign(&x.row(idx));
        }

        Ok(inducing_points)
    }

    /// K-means clustering for inducing point selection
    fn kmeans_selection(x: &Array2<f64>, num_inducing: usize) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if num_inducing >= n_samples {
            return Ok(x.clone());
        }

        // Initialize centroids randomly
        let mut rng = thread_rng();
        let mut centroids = Array2::zeros((num_inducing, n_features));
        for i in 0..num_inducing {
            let idx = rng.gen_range(0..n_samples);
            centroids.row_mut(i).assign(&x.row(idx));
        }

        let max_iter = 100;
        let tol = 1e-6;

        for _iter in 0..max_iter {
            let old_centroids = centroids.clone();

            // Assign points to clusters
            let mut assignments = vec![0; n_samples];
            for i in 0..n_samples {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for j in 0..num_inducing {
                    let mut dist = 0.0;
                    for k in 0..n_features {
                        let diff = x[(i, k)] - centroids[(j, k)];
                        dist += diff * diff;
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        best_cluster = j;
                    }
                }
                assignments[i] = best_cluster;
            }

            // Update centroids
            centroids.fill(0.0);
            let mut cluster_counts = vec![0; num_inducing];

            for i in 0..n_samples {
                let cluster = assignments[i];
                cluster_counts[cluster] += 1;
                for j in 0..n_features {
                    centroids[(cluster, j)] += x[(i, j)];
                }
            }

            for i in 0..num_inducing {
                if cluster_counts[i] > 0 {
                    for j in 0..n_features {
                        centroids[(i, j)] /= cluster_counts[i] as f64;
                    }
                }
            }

            // Check convergence
            let mut max_change: f64 = 0.0;
            for i in 0..num_inducing {
                for j in 0..n_features {
                    let change = (centroids[(i, j)] - old_centroids[(i, j)]).abs();
                    max_change = max_change.max(change);
                }
            }

            if max_change < tol {
                break;
            }
        }

        Ok(centroids)
    }

    /// Uniform grid selection
    fn uniform_grid_selection(x: &Array2<f64>, grid_size: &[usize]) -> Result<Array2<f64>> {
        let n_features = x.ncols();
        if grid_size.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Grid size must match number of features".to_string(),
            ));
        }

        let total_points: usize = grid_size.iter().product();
        let mut grid_points = Array2::zeros((total_points, n_features));

        // Compute feature ranges
        let mut ranges = Vec::with_capacity(n_features);
        for j in 0..n_features {
            let col = x.column(j);
            let min_val = col.fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = col.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            ranges.push((min_val, max_val));
        }

        // Generate grid points
        let mut point_idx = 0;
        Self::generate_grid_recursive(
            &mut grid_points,
            &ranges,
            grid_size,
            &mut vec![0; n_features],
            0,
            &mut point_idx,
        );

        Ok(grid_points)
    }

    /// Recursive helper for grid generation
    fn generate_grid_recursive(
        grid_points: &mut Array2<f64>,
        ranges: &[(f64, f64)],
        grid_size: &[usize],
        current_indices: &mut Vec<usize>,
        dim: usize,
        point_idx: &mut usize,
    ) {
        if dim == ranges.len() {
            // Generate point
            for (j, &idx) in current_indices.iter().enumerate() {
                let (min_val, max_val) = ranges[j];
                let grid_val = if grid_size[j] == 1 {
                    (min_val + max_val) / 2.0
                } else {
                    min_val + idx as f64 * (max_val - min_val) / (grid_size[j] - 1) as f64
                };
                grid_points[(*point_idx, j)] = grid_val;
            }
            *point_idx += 1;
            return;
        }

        for i in 0..grid_size[dim] {
            current_indices[dim] = i;
            Self::generate_grid_recursive(
                grid_points,
                ranges,
                grid_size,
                current_indices,
                dim + 1,
                point_idx,
            );
        }
    }

    /// Greedy variance-based selection
    fn greedy_variance_selection<K: SparseKernel>(
        x: &Array2<f64>,
        num_inducing: usize,
        kernel: &K,
    ) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if num_inducing >= n_samples {
            return Ok(x.clone());
        }

        let mut selected_indices = Vec::new();
        let mut remaining_indices: Vec<usize> = (0..n_samples).collect();

        // Select first point randomly
        let mut rng = thread_rng();
        let first_idx = rng.gen_range(0..remaining_indices.len());
        selected_indices.push(remaining_indices.remove(first_idx));

        // Greedily select remaining points
        for _ in 1..num_inducing {
            let mut best_idx = 0;
            let mut best_variance = -f64::INFINITY;

            for (i, &candidate_idx) in remaining_indices.iter().enumerate() {
                // Compute predictive variance at candidate point
                let variance = Self::compute_predictive_variance(
                    &x.row(candidate_idx).to_owned(),
                    x,
                    &selected_indices,
                    kernel,
                );

                if variance > best_variance {
                    best_variance = variance;
                    best_idx = i;
                }
            }

            selected_indices.push(remaining_indices.remove(best_idx));
        }

        // Create inducing points matrix
        let mut inducing_points = Array2::zeros((num_inducing, n_features));
        for (i, &idx) in selected_indices.iter().enumerate() {
            inducing_points.row_mut(i).assign(&x.row(idx));
        }

        Ok(inducing_points)
    }

    /// Compute predictive variance for greedy selection
    fn compute_predictive_variance<K: SparseKernel>(
        x_star: &Array1<f64>,
        x_train: &Array2<f64>,
        selected_indices: &[usize],
        kernel: &K,
    ) -> f64 {
        if selected_indices.is_empty() {
            return 1.0; // Prior variance
        }

        // Create inducing points from selected indices
        let num_selected = selected_indices.len();
        let mut inducing_points = Array2::zeros((num_selected, x_train.ncols()));
        for (i, &idx) in selected_indices.iter().enumerate() {
            inducing_points.row_mut(i).assign(&x_train.row(idx));
        }

        // Compute kernel matrices
        let k_mm = kernel.kernel_matrix(&inducing_points, &inducing_points);
        let x_star_2d = x_star.clone().insert_axis(Axis(0));
        let k_star_m = kernel.kernel_matrix(&x_star_2d, &inducing_points);
        let k_star_star = kernel.kernel_diagonal(&x_star_2d)[0];

        // Compute predictive variance: k(x*,x*) - k(x*,X_m) K_mm^(-1) k(X_m,x*)
        if let Ok(k_mm_inv) = KernelOps::invert_using_cholesky(&k_mm) {
            let temp = k_star_m.dot(&k_mm_inv);
            let quad_form = temp.dot(&k_star_m.t())[(0, 0)];
            (k_star_star - quad_form).max(0.0)
        } else {
            1.0 // Return prior variance if inversion fails
        }
    }
}

/// Sparse approximation method implementations
pub struct SparseApproximationMethods;

impl SparseApproximationMethods {
    /// Fit Subset of Regressors (SoR) approximation
    pub fn fit_sor<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let m = inducing_points.nrows();

        // Compute kernel matrix K_mm
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);

        // Add noise for numerical stability
        let mut k_mm_noise = k_mm.clone();
        for i in 0..m {
            k_mm_noise[(i, i)] += noise_variance;
        }

        // Compute K_nm (cross-covariance between data and inducing points)
        let _k_nm = kernel.kernel_matrix(x, inducing_points);

        // For SoR: use only inducing point subset
        // Solve (K_mm + σ²I) α = K_mn y_subset
        let inducing_targets = if x.nrows() == inducing_points.nrows() {
            y.clone()
        } else {
            // If not exact match, use nearest neighbor mapping (simplified)
            Self::map_targets_to_inducing(y, x, inducing_points, kernel)?
        };

        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm_noise)?;
        let alpha = k_mm_inv.dot(&inducing_targets);

        Ok((alpha, k_mm_inv))
    }

    /// Fit Fully Independent Conditional (FIC) approximation
    pub fn fit_fic<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let _n = x.nrows();
        let _m = inducing_points.nrows();

        // Compute kernel matrices
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);
        let k_nm = kernel.kernel_matrix(x, inducing_points);
        let k_diag = kernel.kernel_diagonal(x);

        // FIC approximation: Q_nn + diag(K_nn - diag(Q_nn)) + σ²I
        // where Q_nn = K_nm K_mm^(-1) K_mn

        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm)?;
        let q_nn_diag = Self::compute_q_diagonal(&k_nm, &k_mm_inv);

        // Diagonal correction: diag(K_nn - Q_nn)
        let diag_correction: Array1<f64> = k_diag
            .iter()
            .zip(q_nn_diag.iter())
            .map(|(&k_ii, &q_ii)| k_ii - q_ii)
            .collect();

        // Build the FIC system: (Q_nn + diag_correction + σ²I) α = y
        // Use Woodbury matrix identity for efficient solution
        let lambda = &diag_correction + noise_variance;
        let lambda_inv_diag = lambda.mapv(|x| if x > 1e-12 { 1.0 / x } else { 1e12 });

        // Woodbury: (Q + Λ)^(-1) = Λ^(-1) - Λ^(-1) K_nm (K_mm + K_mn Λ^(-1) K_nm)^(-1) K_mn Λ^(-1)
        let lambda_inv_k_nm = &k_nm * &lambda_inv_diag.clone().insert_axis(Axis(1));
        let woodbury_middle = &k_mm + k_nm.t().dot(&lambda_inv_k_nm);
        let woodbury_middle_inv = KernelOps::invert_using_cholesky(&woodbury_middle)?;

        // Solve for alpha using Woodbury identity
        let lambda_inv_y = &lambda_inv_diag * y;
        let temp1 = k_nm.t().dot(&lambda_inv_y);
        let temp2 = woodbury_middle_inv.dot(&temp1);
        let temp3 = lambda_inv_k_nm.dot(&temp2);
        let alpha = &lambda_inv_y - temp3;

        Ok((alpha, k_mm_inv))
    }

    /// Fit Partially Independent Conditional (PIC) approximation
    pub fn fit_pic<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
        block_size: usize,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let n = x.nrows();
        let _m = inducing_points.nrows();

        if block_size >= n {
            // If block size >= n, PIC reduces to standard GP
            return Self::fit_full_gp(x, y, kernel, noise_variance);
        }

        // Compute kernel matrices
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);
        let _k_nm = kernel.kernel_matrix(x, inducing_points);

        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm)?;

        // Build PIC system with block diagonal structure
        let num_blocks = (n + block_size - 1) / block_size;
        let mut alpha = Array1::zeros(n);

        // Process each block independently
        for block_idx in 0..num_blocks {
            let start_idx = block_idx * block_size;
            let end_idx = ((block_idx + 1) * block_size).min(n);
            let block_indices: Vec<usize> = (start_idx..end_idx).collect();

            if block_indices.is_empty() {
                continue;
            }

            // Extract block data
            let mut x_block = Array2::zeros((block_indices.len(), x.ncols()));
            let mut y_block = Array1::zeros(block_indices.len());
            for (i, &idx) in block_indices.iter().enumerate() {
                x_block.row_mut(i).assign(&x.row(idx));
                y_block[i] = y[idx];
            }

            // Compute block kernel matrix and solve
            let k_block = kernel.kernel_matrix(&x_block, &x_block);
            let k_block_m = kernel.kernel_matrix(&x_block, inducing_points);

            // PIC block system
            let q_block = k_block_m.dot(&k_mm_inv).dot(&k_block_m.t());
            let pic_correction = &k_block - &q_block;

            let mut pic_system = pic_correction;
            for i in 0..pic_system.nrows() {
                pic_system[(i, i)] += noise_variance;
            }

            let pic_inv = KernelOps::invert_using_cholesky(&pic_system)?;
            let alpha_block = pic_inv.dot(&y_block);

            // Store block solution
            for (i, &idx) in block_indices.iter().enumerate() {
                alpha[idx] = alpha_block[i];
            }
        }

        Ok((alpha, k_mm_inv))
    }

    /// Helper: Compute Q diagonal for FIC
    fn compute_q_diagonal(k_nm: &Array2<f64>, k_mm_inv: &Array2<f64>) -> Array1<f64> {
        let temp = k_nm.dot(k_mm_inv);
        let mut q_diag = Array1::zeros(k_nm.nrows());

        for i in 0..k_nm.nrows() {
            q_diag[i] = k_nm.row(i).dot(&temp.row(i));
        }

        q_diag
    }

    /// Helper: Map targets to inducing points for SoR
    fn map_targets_to_inducing<K: SparseKernel>(
        y: &Array1<f64>,
        x: &Array2<f64>,
        inducing_points: &Array2<f64>,
        _kernel: &K,
    ) -> Result<Array1<f64>> {
        let n = x.nrows();
        let m = inducing_points.nrows();
        let mut inducing_targets = Array1::zeros(m);

        // Simple approach: find nearest inducing point for each data point
        for i in 0..n {
            let mut min_dist = f64::INFINITY;
            let mut nearest_idx = 0;

            for j in 0..m {
                let mut dist = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[(i, k)] - inducing_points[(j, k)];
                    dist += diff * diff;
                }

                if dist < min_dist {
                    min_dist = dist;
                    nearest_idx = j;
                }
            }

            inducing_targets[nearest_idx] += y[i];
        }

        Ok(inducing_targets)
    }

    /// Full GP fit for comparison (when PIC block size is large)
    fn fit_full_gp<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        kernel: &K,
        noise_variance: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let k_nn = kernel.kernel_matrix(x, x);
        let mut k_noise = k_nn;

        for i in 0..k_noise.nrows() {
            k_noise[(i, i)] += noise_variance;
        }

        let k_inv = KernelOps::invert_using_cholesky(&k_noise)?;
        let alpha = k_inv.dot(y);

        Ok((alpha, k_inv))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_gp::kernels::RBFKernel;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_random_inducing_selection() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];

        let inducing = InducingPointSelector::random_selection(&x, 3).unwrap();
        assert_eq!(inducing.shape(), &[3, 2]);
        assert!(inducing.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kmeans_inducing_selection() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 5.1], [10.0, 10.0]];

        let inducing = InducingPointSelector::kmeans_selection(&x, 2).unwrap();
        assert_eq!(inducing.shape(), &[2, 2]);
        assert!(inducing.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_uniform_grid_selection() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0]];
        let grid_size = vec![2, 2];

        let inducing = InducingPointSelector::uniform_grid_selection(&x, &grid_size).unwrap();
        assert_eq!(inducing.shape(), &[4, 2]);
        assert!(inducing.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_sor_approximation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];
        let inducing_points = array![[0.0, 0.0], [2.0, 2.0]];

        let (alpha, k_mm_inv) =
            SparseApproximationMethods::fit_sor(&x, &y, &inducing_points, &kernel, 0.1).unwrap();

        assert_eq!(alpha.len(), 2);
        assert_eq!(k_mm_inv.shape(), &[2, 2]);
        assert!(alpha.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_fic_approximation() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];
        let inducing_points = array![[0.0, 0.0], [2.0, 2.0]];

        let (alpha, k_mm_inv) =
            SparseApproximationMethods::fit_fic(&x, &y, &inducing_points, &kernel, 0.1).unwrap();

        assert_eq!(alpha.len(), 3);
        assert_eq!(k_mm_inv.shape(), &[2, 2]);
        assert!(alpha.iter().all(|&x| x.is_finite()));
    }
}

//! Variational methods for sparse Gaussian Process approximation
//!
//! This module implements Variational Free Energy (VFE) approximation,
//! Evidence Lower BOund (ELBO) computation, and variational optimization
//! algorithms for sparse Gaussian Processes.

use crate::sparse_gp::core::*;
use crate::sparse_gp::kernels::{KernelOps, SparseKernel};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Variational Free Energy implementation for sparse GP
pub struct VariationalFreeEnergy;

impl VariationalFreeEnergy {
    /// Fit sparse GP using Variational Free Energy approximation
    pub fn fit<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
        whitened: bool,
        natural_gradients: bool,
        max_iter: usize,
        tol: f64,
    ) -> Result<(Array1<f64>, Array2<f64>, VariationalParams)> {
        let n = x.nrows();
        let m = inducing_points.nrows();

        // Initialize variational parameters
        let mut variational_mean = Array1::zeros(m);
        let mut variational_cov_factor = Array2::eye(m);

        // Compute fixed kernel matrices
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);
        let k_nm = kernel.kernel_matrix(x, inducing_points);
        let k_diag = kernel.kernel_diagonal(x);

        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm)?;

        // Optimization loop
        let mut best_elbo = f64::NEG_INFINITY;
        let mut best_params = None;

        for iter in 0..max_iter {
            let old_mean = variational_mean.clone();
            let old_cov_factor = variational_cov_factor.clone();

            // Compute ELBO and gradients
            let elbo_result = Self::compute_elbo_and_gradients(
                y,
                &k_nm,
                &k_diag,
                &k_mm,
                &k_mm_inv,
                &variational_mean,
                &variational_cov_factor,
                noise_variance,
                whitened,
            )?;

            if elbo_result.elbo > best_elbo {
                best_elbo = elbo_result.elbo;
                best_params = Some((
                    variational_mean.clone(),
                    variational_cov_factor.clone(),
                    elbo_result.clone(),
                ));
            }

            // Update variational parameters
            if natural_gradients {
                Self::natural_gradient_update(
                    &mut variational_mean,
                    &mut variational_cov_factor,
                    &elbo_result,
                    0.01, // Learning rate
                )?;
            } else {
                Self::standard_gradient_update(
                    &mut variational_mean,
                    &mut variational_cov_factor,
                    &elbo_result,
                    0.01, // Learning rate
                )?;
            }

            // Check convergence
            let mean_change = (&variational_mean - &old_mean).mapv(|x| x * x).sum().sqrt();
            let cov_change = (&variational_cov_factor - &old_cov_factor)
                .mapv(|x| x * x)
                .sum()
                .sqrt();

            if mean_change < tol && cov_change < tol {
                break;
            }
        }

        let (best_mean, best_cov_factor, best_elbo_result) = best_params
            .ok_or_else(|| SklearsError::NumericalError("VFE optimization failed".to_string()))?;

        // Compute final alpha for prediction
        let alpha = if whitened {
            Self::compute_alpha_whitened(&k_nm, &k_mm_inv, &best_mean, noise_variance)?
        } else {
            Self::compute_alpha_standard(&k_mm_inv, &best_mean)?
        };

        let vfe_params = VariationalParams {
            mean: best_mean,
            cov_factor: best_cov_factor,
            elbo: best_elbo_result.elbo,
            kl_divergence: best_elbo_result.kl_divergence,
            log_likelihood: best_elbo_result.log_likelihood,
        };

        Ok((alpha, k_mm_inv, vfe_params))
    }

    /// Compute ELBO and its gradients
    fn compute_elbo_and_gradients(
        y: &Array1<f64>,
        k_nm: &Array2<f64>,
        k_diag: &Array1<f64>,
        k_mm: &Array2<f64>,
        k_mm_inv: &Array2<f64>,
        variational_mean: &Array1<f64>,
        variational_cov_factor: &Array2<f64>,
        noise_variance: f64,
        whitened: bool,
    ) -> Result<ELBOResult> {
        let n = y.len();
        let m = variational_mean.len();

        // Compute variational covariance S = L L^T
        let variational_cov = variational_cov_factor.dot(&variational_cov_factor.t());

        // Expected log likelihood term
        let (log_likelihood, ll_grad_mean, ll_grad_cov) = Self::compute_expected_log_likelihood(
            y,
            k_nm,
            k_diag,
            variational_mean,
            &variational_cov,
            noise_variance,
        )?;

        // KL divergence term
        let (kl_divergence, kl_grad_mean, kl_grad_cov) = if whitened {
            Self::compute_kl_divergence_whitened(variational_mean, &variational_cov)?
        } else {
            Self::compute_kl_divergence_standard(
                variational_mean,
                &variational_cov,
                k_mm,
                k_mm_inv,
            )?
        };

        // ELBO = E[log p(y|f)] - KL[q(u) || p(u)]
        let elbo = log_likelihood - kl_divergence;

        // Gradients
        let grad_mean = &ll_grad_mean - &kl_grad_mean;
        let grad_cov = &ll_grad_cov - &kl_grad_cov;

        Ok(ELBOResult {
            elbo,
            log_likelihood,
            kl_divergence,
            grad_mean,
            grad_cov,
        })
    }

    /// Compute expected log likelihood and gradients
    fn compute_expected_log_likelihood(
        y: &Array1<f64>,
        k_nm: &Array2<f64>,
        k_diag: &Array1<f64>,
        variational_mean: &Array1<f64>,
        variational_cov: &Array2<f64>,
        noise_variance: f64,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let n = y.len();

        // Predictive mean: μ_f = K_nm m
        let pred_mean = k_nm.dot(variational_mean);

        // Predictive variance: diag(K_nn - K_nm (K_mm - S)^(-1) K_mn)
        let k_nm_s_k_mn = k_nm.dot(variational_cov).dot(&k_nm.t());
        let mut pred_var_diag = k_diag.clone();

        for i in 0..n {
            pred_var_diag[i] -= k_nm_s_k_mn[(i, i)];
            pred_var_diag[i] = pred_var_diag[i].max(1e-6); // Ensure positive
        }

        // Log likelihood: -0.5 * sum(log(2π(σ² + σ_f²)) + (y - μ_f)² / (σ² + σ_f²))
        let residuals = y - &pred_mean;
        let total_var = &pred_var_diag + noise_variance;

        let log_likelihood = -0.5
            * (n as f64 * (2.0 * PI).ln()
                + total_var.mapv(|x| x.ln()).sum()
                + residuals
                    .iter()
                    .zip(total_var.iter())
                    .map(|(r, v)| r * r / v)
                    .sum::<f64>());

        // Gradients (simplified)
        let grad_mean = k_nm.t().dot(&(y - &pred_mean)) / noise_variance;
        let grad_cov = Array2::zeros(variational_cov.dim());

        Ok((log_likelihood, grad_mean, grad_cov))
    }

    /// Compute KL divergence for standard parameterization
    fn compute_kl_divergence_standard(
        variational_mean: &Array1<f64>,
        variational_cov: &Array2<f64>,
        k_mm: &Array2<f64>,
        k_mm_inv: &Array2<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let m = variational_mean.len();

        // KL[q(u) || p(u)] = 0.5 * [tr(K_mm^(-1) S) + m^T K_mm^(-1) m - m - log|S| + log|K_mm|]

        // Trace term
        let trace_term = (k_mm_inv * variational_cov).diag().sum();

        // Quadratic term
        let quad_term = variational_mean.dot(&k_mm_inv.dot(variational_mean));

        // Log determinant terms (simplified using Cholesky)
        let log_det_s = Self::log_det_from_cholesky_factor(variational_cov)?;
        let log_det_k_mm =
            KernelOps::log_det_from_cholesky(&KernelOps::cholesky_with_jitter(k_mm, 1e-6)?);

        let kl = 0.5 * (trace_term + quad_term - m as f64 - log_det_s + log_det_k_mm);

        // Gradients (simplified)
        let grad_mean = k_mm_inv.dot(variational_mean);
        let grad_cov = k_mm_inv.clone();

        Ok((kl, grad_mean, grad_cov))
    }

    /// Compute KL divergence for whitened parameterization
    fn compute_kl_divergence_whitened(
        variational_mean: &Array1<f64>,
        variational_cov: &Array2<f64>,
    ) -> Result<(f64, Array1<f64>, Array2<f64>)> {
        let m = variational_mean.len();

        // For whitened variables: KL[q(v) || N(0,I)] = 0.5 * [tr(S) + m^T m - m - log|S|]
        let trace_term = variational_cov.diag().sum();
        let quad_term = variational_mean.dot(variational_mean);
        let log_det_s = Self::log_det_from_cholesky_factor(variational_cov)?;

        let kl = 0.5 * (trace_term + quad_term - m as f64 - log_det_s);

        // Gradients
        let grad_mean = variational_mean.clone();
        let grad_cov = Array2::eye(m);

        Ok((kl, grad_mean, grad_cov))
    }

    /// Natural gradient update for variational parameters
    fn natural_gradient_update(
        variational_mean: &mut Array1<f64>,
        variational_cov_factor: &mut Array2<f64>,
        elbo_result: &ELBOResult,
        learning_rate: f64,
    ) -> Result<()> {
        // Natural gradients transform the gradient by the inverse Fisher information matrix
        // For Gaussian variational distribution, this simplifies the updates

        // Update mean (natural gradient is the same as standard gradient for mean)
        *variational_mean = &*variational_mean + learning_rate * &elbo_result.grad_mean;

        // Update covariance factor (simplified natural gradient)
        let cov_update = learning_rate * &elbo_result.grad_cov;
        *variational_cov_factor = &*variational_cov_factor + &cov_update;

        Ok(())
    }

    /// Standard gradient update for variational parameters
    fn standard_gradient_update(
        variational_mean: &mut Array1<f64>,
        variational_cov_factor: &mut Array2<f64>,
        elbo_result: &ELBOResult,
        learning_rate: f64,
    ) -> Result<()> {
        // Standard gradient updates
        *variational_mean = &*variational_mean + learning_rate * &elbo_result.grad_mean;
        *variational_cov_factor = &*variational_cov_factor + learning_rate * &elbo_result.grad_cov;

        Ok(())
    }

    /// Compute alpha for whitened parameterization
    fn compute_alpha_whitened(
        k_nm: &Array2<f64>,
        k_mm_inv: &Array2<f64>,
        variational_mean: &Array1<f64>,
        noise_variance: f64,
    ) -> Result<Array1<f64>> {
        // Transform whitened variables back to standard space
        // Simplified version - assume k_mm_inv is already the inverse
        let l_mm = KernelOps::cholesky_with_jitter(k_mm_inv, 1e-6)?;

        let alpha = k_mm_inv.dot(&l_mm.dot(variational_mean));
        Ok(alpha)
    }

    /// Compute alpha for standard parameterization
    fn compute_alpha_standard(
        k_mm_inv: &Array2<f64>,
        variational_mean: &Array1<f64>,
    ) -> Result<Array1<f64>> {
        Ok(k_mm_inv.dot(variational_mean))
    }

    /// Compute log determinant from Cholesky factor (simplified)
    fn log_det_from_cholesky_factor(matrix: &Array2<f64>) -> Result<f64> {
        // For positive definite matrix S = L*L^T, log|S| = 2*sum(log(diag(L)))
        let eigenvals = matrix.diag();
        let log_det = eigenvals.mapv(|x| x.abs().max(1e-12).ln()).sum();
        Ok(log_det)
    }
}

/// Structure to hold ELBO computation results
#[derive(Debug, Clone)]
pub struct ELBOResult {
    /// elbo
    pub elbo: f64,
    /// log_likelihood
    pub log_likelihood: f64,
    /// kl_divergence
    pub kl_divergence: f64,
    /// grad_mean
    pub grad_mean: Array1<f64>,
    /// grad_cov
    pub grad_cov: Array2<f64>,
}

/// Stochastic Variational Inference for large-scale datasets
pub struct StochasticVariationalInference;

impl StochasticVariationalInference {
    /// Fit sparse GP using stochastic variational inference with mini-batches
    pub fn fit<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
        batch_size: usize,
        max_iter: usize,
        learning_rate: f64,
    ) -> Result<(Array1<f64>, Array2<f64>, VariationalParams)> {
        let n = x.nrows();
        let m = inducing_points.nrows();

        // Initialize variational parameters
        let mut variational_mean = Array1::zeros(m);
        let mut variational_cov_factor = Array2::eye(m);

        // Compute fixed kernel matrices
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);
        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm)?;

        let mut rng = thread_rng();

        // Stochastic optimization loop
        for iter in 0..max_iter {
            // Sample mini-batch
            let batch_indices = Self::sample_batch(&mut rng, n, batch_size);
            let (x_batch, y_batch) = Self::extract_batch(x, y, &batch_indices);

            // Compute batch-specific kernel matrices
            let k_batch_m = kernel.kernel_matrix(&x_batch, inducing_points);
            let k_batch_diag = kernel.kernel_diagonal(&x_batch);

            // Compute stochastic ELBO and gradients
            let elbo_result = VariationalFreeEnergy::compute_elbo_and_gradients(
                &y_batch,
                &k_batch_m,
                &k_batch_diag,
                &k_mm,
                &k_mm_inv,
                &variational_mean,
                &variational_cov_factor,
                noise_variance,
                false, // Use standard parameterization
            )?;

            // Scale gradients by batch size factor
            let scale_factor = n as f64 / batch_size as f64;
            let scaled_grad_mean = &elbo_result.grad_mean * scale_factor;
            let scaled_grad_cov = &elbo_result.grad_cov * scale_factor;

            // Update parameters
            variational_mean = &variational_mean + learning_rate * &scaled_grad_mean;
            variational_cov_factor = &variational_cov_factor + learning_rate * &scaled_grad_cov;

            // Decay learning rate
            if iter % 100 == 0 {
                // learning_rate *= 0.99;
            }
        }

        // Compute final alpha and parameters
        let alpha = VariationalFreeEnergy::compute_alpha_standard(&k_mm_inv, &variational_mean)?;

        let variational_cov = variational_cov_factor.dot(&variational_cov_factor.t());

        // Compute final ELBO on full dataset
        let k_nm = kernel.kernel_matrix(x, inducing_points);
        let k_diag = kernel.kernel_diagonal(x);

        let final_elbo = VariationalFreeEnergy::compute_elbo_and_gradients(
            y,
            &k_nm,
            &k_diag,
            &k_mm,
            &k_mm_inv,
            &variational_mean,
            &variational_cov_factor,
            noise_variance,
            false,
        )?;

        let vfe_params = VariationalParams {
            mean: variational_mean,
            cov_factor: variational_cov_factor,
            elbo: final_elbo.elbo,
            kl_divergence: final_elbo.kl_divergence,
            log_likelihood: final_elbo.log_likelihood,
        };

        Ok((alpha, k_mm_inv, vfe_params))
    }

    /// Sample mini-batch indices
    fn sample_batch(rng: &mut impl Rng, n: usize, batch_size: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..n).collect();

        // Fisher-Yates shuffle
        for i in (1..n).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        indices.into_iter().take(batch_size).collect()
    }

    /// Extract mini-batch from data
    fn extract_batch(
        x: &Array2<f64>,
        y: &Array1<f64>,
        indices: &[usize],
    ) -> (Array2<f64>, Array1<f64>) {
        let batch_size = indices.len();
        let n_features = x.ncols();

        let mut x_batch = Array2::zeros((batch_size, n_features));
        let mut y_batch = Array1::zeros(batch_size);

        for (i, &idx) in indices.iter().enumerate() {
            x_batch.row_mut(i).assign(&x.row(idx));
            y_batch[i] = y[idx];
        }

        (x_batch, y_batch)
    }
}

/// Utility functions for variational inference
pub mod variational_utils {
    use super::*;

    /// Initialize variational parameters using data
    pub fn initialize_variational_params<K: SparseKernel>(
        x: &Array2<f64>,
        y: &Array1<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        noise_variance: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let m = inducing_points.nrows();

        // Initialize mean using simple linear regression-like approach
        let k_mm = kernel.kernel_matrix(inducing_points, inducing_points);
        let k_nm = kernel.kernel_matrix(x, inducing_points);

        let k_mm_inv = KernelOps::invert_using_cholesky(&k_mm)?;

        // Simple initialization: m = (K_mm + σ²I)^(-1) K_mn y
        let mut k_mm_reg = k_mm.clone();
        for i in 0..m {
            k_mm_reg[(i, i)] += noise_variance;
        }

        let k_mm_reg_inv = KernelOps::invert_using_cholesky(&k_mm_reg)?;
        let initial_mean = k_mm_reg_inv.dot(&k_nm.t()).dot(y);

        // Initialize covariance as scaled identity
        let initial_cov_factor = Array2::eye(m) * 0.1;

        Ok((initial_mean, initial_cov_factor))
    }

    /// Compute predictive distribution moments
    pub fn predictive_moments<K: SparseKernel>(
        x_test: &Array2<f64>,
        inducing_points: &Array2<f64>,
        kernel: &K,
        vfe_params: &VariationalParams,
        noise_variance: f64,
    ) -> Result<(Array1<f64>, Array1<f64>)> {
        let k_star_m = kernel.kernel_matrix(x_test, inducing_points);
        let k_star_star = kernel.kernel_diagonal(x_test);

        // Predictive mean
        let pred_mean = k_star_m.dot(&vfe_params.mean);

        // Predictive variance (including epistemic uncertainty)
        let variational_cov = vfe_params.cov_factor.dot(&vfe_params.cov_factor.t());
        let epistemic_var = compute_epistemic_variance(&k_star_m, &variational_cov);

        let pred_var = &k_star_star - &epistemic_var + noise_variance;

        Ok((pred_mean, pred_var))
    }

    /// Compute epistemic variance contribution
    fn compute_epistemic_variance(
        k_star_m: &Array2<f64>,
        variational_cov: &Array2<f64>,
    ) -> Array1<f64> {
        let temp = k_star_m.dot(variational_cov);
        let mut epistemic_var = Array1::zeros(k_star_m.nrows());

        for i in 0..k_star_m.nrows() {
            epistemic_var[i] = k_star_m.row(i).dot(&temp.row(i));
        }

        epistemic_var
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_gp::kernels::RBFKernel;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_vfe_initialization() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0.0, 1.0, 4.0];
        let inducing_points = array![[0.0, 0.0], [2.0, 2.0]];

        let (mean, cov_factor) = variational_utils::initialize_variational_params(
            &x,
            &y,
            &inducing_points,
            &kernel,
            0.1,
        )
        .unwrap();

        assert_eq!(mean.len(), 2);
        assert_eq!(cov_factor.shape(), &[2, 2]);
        assert!(mean.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_kl_divergence_whitened() {
        let mean = array![0.5, -0.3];
        let cov_factor = Array2::eye(2) * 0.8;
        let cov = cov_factor.dot(&cov_factor.t());

        let (kl, grad_mean, grad_cov) =
            VariationalFreeEnergy::compute_kl_divergence_whitened(&mean, &cov).unwrap();

        assert!(kl >= 0.0); // KL divergence should be non-negative
        assert_eq!(grad_mean.len(), 2);
        assert_eq!(grad_cov.shape(), &[2, 2]);
    }

    #[test]
    fn test_expected_log_likelihood() {
        let y = array![0.0, 1.0, 2.0];
        let k_nm = array![[1.0, 0.5], [0.8, 0.3], [0.6, 0.9]];
        let k_diag = array![1.0, 1.0, 1.0];
        let mean = array![0.5, 0.3];
        let cov = Array2::eye(2) * 0.1;

        let (ll, grad_mean, grad_cov) = VariationalFreeEnergy::compute_expected_log_likelihood(
            &y, &k_nm, &k_diag, &mean, &cov, 0.1,
        )
        .unwrap();

        assert!(ll.is_finite());
        assert_eq!(grad_mean.len(), 2);
        assert_eq!(grad_cov.shape(), &[2, 2]);
    }

    #[test]
    fn test_stochastic_batch_sampling() {
        let mut rng = thread_rng();
        let batch_indices = StochasticVariationalInference::sample_batch(&mut rng, 10, 3);

        assert_eq!(batch_indices.len(), 3);
        assert!(batch_indices.iter().all(|&i| i < 10));

        // Check uniqueness
        let mut sorted_indices = batch_indices.clone();
        sorted_indices.sort();
        sorted_indices.dedup();
        assert_eq!(sorted_indices.len(), batch_indices.len());
    }

    #[test]
    fn test_predictive_moments() {
        let kernel = RBFKernel::new(1.0, 1.0);
        let x_test = array![[0.5, 0.5], [1.5, 1.5]];
        let inducing_points = array![[0.0, 0.0], [2.0, 2.0]];

        let vfe_params = VariationalParams {
            mean: array![0.5, 0.3],
            cov_factor: Array2::eye(2) * 0.1,
            elbo: -10.5,
            kl_divergence: 2.3,
            log_likelihood: -12.8,
        };

        let (pred_mean, pred_var) = variational_utils::predictive_moments(
            &x_test,
            &inducing_points,
            &kernel,
            &vfe_params,
            0.1,
        )
        .unwrap();

        assert_eq!(pred_mean.len(), 2);
        assert_eq!(pred_var.len(), 2);
        assert!(pred_mean.iter().all(|&x| x.is_finite()));
        assert!(pred_var.iter().all(|&x| x >= 0.0 && x.is_finite()));
    }
}

//! EM Algorithm Implementation for Gaussian Mixture Models
//!
//! This module provides the core Expectation-Maximization algorithm implementation
//! with SIMD acceleration, supporting various covariance types and initialization
//! strategies for both classical and Bayesian GMM variants.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
use sklears_core::{error::Result, types::Float};

use super::simd_operations::*;
use super::types_config::{CovarianceType, WeightInit};

/// Core EM Algorithm implementation with SIMD acceleration
pub struct EMAlgorithm {
    pub max_iter: usize,
    pub tol: Float,
    pub reg_covar: Float,
    pub covariance_type: CovarianceType,
    pub init_params: WeightInit,
    pub random_state: Option<u64>,
}

impl EMAlgorithm {
    /// Create a new EM algorithm instance
    pub fn new(max_iter: usize, tol: Float, reg_covar: Float) -> Self {
        Self {
            max_iter,
            tol,
            reg_covar,
            covariance_type: CovarianceType::Full,
            init_params: WeightInit::KMeans,
            random_state: None,
        }
    }

    /// Set covariance type
    pub fn covariance_type(mut self, covariance_type: CovarianceType) -> Self {
        self.covariance_type = covariance_type;
        self
    }

    /// Set initialization method
    pub fn init_params(mut self, init_params: WeightInit) -> Self {
        self.init_params = init_params;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Run the complete EM algorithm
    pub fn fit(&self, x: &ArrayView2<Float>, n_components: usize) -> Result<EMResult> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize parameters using SIMD-accelerated operations
        let (mut weights, mut means, mut covariances) =
            self.initialize_parameters(x, n_components)?;

        let mut log_likelihood = Float::NEG_INFINITY;
        let mut converged = false;
        let mut n_iter = 0;

        // Main EM algorithm loop with SIMD acceleration
        for iteration in 0..self.max_iter {
            n_iter = iteration + 1;

            // E-step: Compute responsibilities using SIMD operations
            let responsibilities = self.e_step_simd(x, &weights, &means, &covariances)?;

            // M-step: Update parameters using SIMD operations
            let (new_weights, new_means, new_covariances) =
                self.m_step_simd(x, &responsibilities)?;

            // Compute log-likelihood with SIMD acceleration
            let new_log_likelihood =
                self.compute_log_likelihood_simd(x, &new_weights, &new_means, &new_covariances)?;

            // Check convergence using SIMD operations
            if simd_check_convergence(log_likelihood, new_log_likelihood, self.tol) {
                converged = true;
                weights = new_weights;
                means = new_means;
                covariances = new_covariances;
                log_likelihood = new_log_likelihood;
                break;
            }

            weights = new_weights;
            means = new_means;
            covariances = new_covariances;
            log_likelihood = new_log_likelihood;
        }

        // Apply final regularization using SIMD operations
        for cov in &mut covariances {
            simd_regularize_covariance(cov, self.reg_covar);
        }

        Ok(EMResult {
            weights,
            means,
            covariances,
            converged,
            n_iter,
            log_likelihood,
        })
    }

    /// Initialize GMM parameters with SIMD-accelerated operations
    pub fn initialize_parameters(
        &self,
        x: &ArrayView2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>, Vec<Array2<Float>>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Initialize weights uniformly
        let weights = Array1::from_elem(n_components, 1.0 / n_components as Float);

        // Initialize means using specified strategy
        let means = self.initialize_means(x, n_components)?;

        // Initialize covariances based on covariance type
        let covariances = self.initialize_covariances(x, &means, n_components)?;

        Ok((weights, means, covariances))
    }

    /// Initialize means using K-means++ or random strategy
    fn initialize_means(
        &self,
        x: &ArrayView2<Float>,
        n_components: usize,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let mut means = Array2::zeros((n_components, n_features));

        match self.init_params {
            WeightInit::KMeans => {
                // K-means++ initialization for better convergence
                let mut selected = Vec::new();

                // Choose first center randomly or deterministically
                let first_idx = if let Some(seed) = self.random_state {
                    (seed as usize) % n_samples
                } else {
                    0
                };
                means.row_mut(0).assign(&x.row(first_idx));
                selected.push(first_idx);

                // Choose remaining centers with probability proportional to squared distance
                for i in 1..n_components {
                    let mut distances = Array1::zeros(n_samples);

                    // Compute distances to nearest existing center using SIMD
                    for (j, sample) in x.outer_iter().enumerate() {
                        let mut min_dist = Float::INFINITY;
                        for &sel_idx in &selected {
                            let center = x.row(sel_idx);
                            let dist = simd_euclidean_distance_squared(&sample, &center);
                            if dist < min_dist {
                                min_dist = dist;
                            }
                        }
                        distances[j] = min_dist;
                    }

                    // Select center with probability proportional to squared distance
                    let sum_dist = distances.sum();
                    if sum_dist > 0.0 {
                        let mut cumsum = 0.0;
                        let target = self.generate_random_float(i as u64) * sum_dist;

                        for (j, &dist) in distances.iter().enumerate() {
                            cumsum += dist;
                            if cumsum >= target {
                                means.row_mut(i).assign(&x.row(j));
                                selected.push(j);
                                break;
                            }
                        }
                    } else {
                        // Fallback to uniform sampling
                        let idx = i % n_samples;
                        means.row_mut(i).assign(&x.row(idx));
                        selected.push(idx);
                    }
                }
            }
            WeightInit::Random => {
                // Simple random initialization
                for i in 0..n_components {
                    let idx = if let Some(seed) = self.random_state {
                        ((seed as usize) + i) % n_samples
                    } else {
                        i * n_samples / n_components
                    };
                    means.row_mut(i).assign(&x.row(idx));
                }
            }
        }

        Ok(means)
    }

    /// Initialize covariances based on covariance type
    fn initialize_covariances(
        &self,
        x: &ArrayView2<Float>,
        means: &Array2<Float>,
        n_components: usize,
    ) -> Result<Vec<Array2<Float>>> {
        let n_features = x.ncols();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                // Initialize with scaled identity matrices
                let data_var = x.var_axis(Axis(0), 0.0);
                let scale = data_var.mean().unwrap_or(1.0);

                for _ in 0..n_components {
                    let mut cov = Array2::eye(n_features) * scale;
                    simd_regularize_covariance(&mut cov, self.reg_covar);
                    covariances.push(cov);
                }
            }
            CovarianceType::Diagonal => {
                let data_var = x.var_axis(Axis(0), 0.0);

                for _ in 0..n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = data_var[i].max(self.reg_covar);
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                // All components share the same covariance
                let data_cov = self.compute_sample_covariance(x);
                let mut cov = data_cov;
                simd_regularize_covariance(&mut cov, self.reg_covar);

                for _ in 0..n_components {
                    covariances.push(cov.clone());
                }
            }
            CovarianceType::Spherical => {
                let data_var = x.var_axis(Axis(0), 0.0);
                let avg_var = data_var.mean().unwrap_or(1.0);

                for _ in 0..n_components {
                    let mut cov = Array2::zeros((n_features, n_features));
                    for i in 0..n_features {
                        cov[[i, i]] = avg_var.max(self.reg_covar);
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// E-step: Compute responsibilities using SIMD acceleration
    pub fn e_step_simd(
        &self,
        x: &ArrayView2<Float>,
        weights: &Array1<Float>,
        means: &Array2<Float>,
        covariances: &[Array2<Float>],
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let n_components = weights.len();
        let mut responsibilities = Array2::zeros((n_samples, n_components));

        for (i, sample) in x.outer_iter().enumerate() {
            let mut log_probs = Array1::zeros(n_components);

            for k in 0..n_components {
                let mean = means.row(k);
                let cov = &covariances[k];

                // Extract diagonal for SIMD multivariate normal computation
                let inv_diag = self.extract_diagonal_inverse(cov)?;
                let log_det = simd_log_determinant(&cov.view());

                // Compute log probability density using SIMD operations
                let log_prob = weights[k].ln()
                    + simd_multivariate_normal_log_density(
                        &sample,
                        &mean,
                        &inv_diag.view(),
                        log_det,
                    );

                log_probs[k] = log_prob;
            }

            // Normalize using SIMD log-sum-exp trick for numerical stability
            let log_sum = simd_log_sum_exp(&log_probs.view());
            for k in 0..n_components {
                responsibilities[[i, k]] = (log_probs[k] - log_sum).exp();
            }
        }

        Ok(responsibilities)
    }

    /// M-step: Update parameters using SIMD acceleration
    pub fn m_step_simd(
        &self,
        x: &ArrayView2<Float>,
        responsibilities: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>, Vec<Array2<Float>>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_components = responsibilities.ncols();

        // Compute effective number of samples for each component using SIMD
        let nk: Array1<Float> = responsibilities.sum_axis(Axis(0));

        // Update weights using SIMD acceleration
        let weights = &nk / n_samples as Float;

        // Update means using SIMD weighted sum operations
        let mut means = Array2::zeros((n_components, n_features));
        for k in 0..n_components {
            if nk[k] > 1e-12 {
                // Use SIMD-accelerated weighted sum
                for i in 0..n_samples {
                    means
                        .row_mut(k)
                        .scaled_add(responsibilities[[i, k]], &x.row(i));
                }
                means.row_mut(k).mapv_inplace(|x| x / nk[k]);
            }
        }

        // Update covariances using SIMD operations based on covariance type
        let covariances = self.update_covariances_simd(x, &means, responsibilities, &nk)?;

        Ok((weights, means, covariances))
    }

    /// Update covariances with SIMD acceleration
    fn update_covariances_simd(
        &self,
        x: &ArrayView2<Float>,
        means: &Array2<Float>,
        responsibilities: &Array2<Float>,
        nk: &Array1<Float>,
    ) -> Result<Vec<Array2<Float>>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let n_components = means.nrows();
        let mut covariances = Vec::new();

        match self.covariance_type {
            CovarianceType::Full => {
                for k in 0..n_components {
                    let cov = if nk[k] > 1e-12 {
                        simd_covariance_matrix(
                            &x.view(),
                            &responsibilities.column(k).view(),
                            &means.row(k).view(),
                        )
                    } else {
                        Array2::eye(n_features)
                    };

                    let mut regularized_cov = cov;
                    simd_regularize_covariance(&mut regularized_cov, self.reg_covar);
                    covariances.push(regularized_cov);
                }
            }
            CovarianceType::Diagonal => {
                for k in 0..n_components {
                    let diag_cov = if nk[k] > 1e-12 {
                        simd_diagonal_covariance(
                            &x.view(),
                            &responsibilities.column(k).view(),
                            &means.row(k).view(),
                        )
                    } else {
                        Array1::from_elem(n_features, self.reg_covar)
                    };

                    let mut cov = Array2::zeros((n_features, n_features));
                    for j in 0..n_features {
                        cov[[j, j]] = diag_cov[j].max(self.reg_covar);
                    }
                    covariances.push(cov);
                }
            }
            CovarianceType::Tied => {
                // Compute shared covariance using SIMD operations
                let mut shared_cov = Array2::zeros((n_features, n_features));

                for k in 0..n_components {
                    if nk[k] > 1e-12 {
                        let component_cov = simd_covariance_matrix(
                            &x.view(),
                            &responsibilities.column(k).view(),
                            &means.row(k).view(),
                        );
                        shared_cov.scaled_add(nk[k] / n_samples as Float, &component_cov);
                    }
                }

                simd_regularize_covariance(&mut shared_cov, self.reg_covar);

                for _ in 0..n_components {
                    covariances.push(shared_cov.clone());
                }
            }
            CovarianceType::Spherical => {
                for k in 0..n_components {
                    let avg_var = if nk[k] > 1e-12 {
                        let diag_cov = simd_diagonal_covariance(
                            &x.view(),
                            &responsibilities.column(k).view(),
                            &means.row(k).view(),
                        );
                        diag_cov
                            .mean()
                            .unwrap_or(self.reg_covar)
                            .max(self.reg_covar)
                    } else {
                        self.reg_covar
                    };

                    let mut cov = Array2::zeros((n_features, n_features));
                    for j in 0..n_features {
                        cov[[j, j]] = avg_var;
                    }
                    covariances.push(cov);
                }
            }
        }

        Ok(covariances)
    }

    /// Compute log-likelihood using SIMD acceleration
    pub fn compute_log_likelihood_simd(
        &self,
        x: &ArrayView2<Float>,
        weights: &Array1<Float>,
        means: &Array2<Float>,
        covariances: &[Array2<Float>],
    ) -> Result<Float> {
        let mut log_likelihood = 0.0;

        for sample in x.outer_iter() {
            let mut sample_likelihood = 0.0;

            for k in 0..weights.len() {
                let mean = means.row(k);
                let cov = &covariances[k];

                let inv_diag = self.extract_diagonal_inverse(cov)?;
                let log_det = simd_log_determinant(&cov.view());

                let log_prob =
                    simd_multivariate_normal_log_density(&sample, &mean, &inv_diag.view(), log_det);

                sample_likelihood += weights[k] * log_prob.exp();
            }

            if sample_likelihood > 1e-12 {
                log_likelihood += sample_likelihood.ln();
            }
        }

        Ok(log_likelihood)
    }

    /// Extract diagonal inverse for SIMD operations
    fn extract_diagonal_inverse(&self, cov: &Array2<Float>) -> Result<Array1<Float>> {
        let mut inv_diag = Array1::zeros(cov.nrows());
        for i in 0..cov.nrows() {
            let diag_val = cov[[i, i]];
            if diag_val <= 1e-12 {
                return Err(sklears_core::error::SklearsError::Other(
                    "Singular covariance matrix".to_string(),
                ));
            }
            inv_diag[i] = 1.0 / diag_val;
        }
        Ok(inv_diag)
    }

    /// Compute sample covariance matrix
    fn compute_sample_covariance(&self, x: &ArrayView2<Float>) -> Array2<Float> {
        let n_features = x.ncols();
        let mean = x.mean_axis(Axis(0)).unwrap();
        let mut cov = Array2::zeros((n_features, n_features));

        for sample in x.outer_iter() {
            let diff = &sample - &mean;
            for i in 0..n_features {
                for j in 0..n_features {
                    cov[[i, j]] += diff[i] * diff[j];
                }
            }
        }

        cov /= (x.nrows() - 1) as Float;
        cov
    }

    /// Generate pseudo-random float for deterministic randomness
    fn generate_random_float(&self, seed_offset: u64) -> Float {
        if let Some(base_seed) = self.random_state {
            let seed = base_seed.wrapping_add(seed_offset);
            // Simple linear congruential generator for deterministic randomness
            let a = 1103515245_u64;
            let c = 12345_u64;
            let m = 2_u64.pow(31);
            let next = (a.wrapping_mul(seed).wrapping_add(c)) % m;
            next as Float / m as Float
        } else {
            0.5 // Default deterministic value
        }
    }
}

/// Result of EM algorithm execution
#[derive(Debug, Clone)]
pub struct EMResult {
    pub weights: Array1<Float>,
    pub means: Array2<Float>,
    pub covariances: Vec<Array2<Float>>,
    pub converged: bool,
    pub n_iter: usize,
    pub log_likelihood: Float,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_em_algorithm() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let em = EMAlgorithm::new(100, 1e-3, 1e-6)
            .covariance_type(CovarianceType::Diagonal)
            .init_params(WeightInit::KMeans);

        let result = em.fit(&x.view(), 2).unwrap();

        assert_eq!(result.weights.len(), 2);
        assert_eq!(result.means.nrows(), 2);
        assert_eq!(result.covariances.len(), 2);
        assert!(result.n_iter > 0);
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_parameter_initialization() {
        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [3.0, 3.0],];

        let em = EMAlgorithm::new(100, 1e-3, 1e-6)
            .covariance_type(CovarianceType::Full)
            .init_params(WeightInit::KMeans);

        let (weights, means, covariances) = em.initialize_parameters(&x.view(), 2).unwrap();

        assert_eq!(weights.len(), 2);
        assert_eq!(means.nrows(), 2);
        assert_eq!(means.ncols(), 2);
        assert_eq!(covariances.len(), 2);

        // Check that weights sum to 1
        assert!((weights.sum() - 1.0).abs() < 1e-10);

        // Check covariance matrix properties
        for cov in &covariances {
            assert_eq!(cov.nrows(), 2);
            assert_eq!(cov.ncols(), 2);
            // Check positive definiteness (diagonal elements should be positive)
            for i in 0..2 {
                assert!(cov[[i, i]] > 0.0);
            }
        }
    }

    #[test]
    fn test_convergence_check() {
        let old_ll = -100.0;
        let new_ll = -99.999;
        let tol = 1e-3;

        assert!(simd_check_convergence(old_ll, new_ll, tol));

        let new_ll_no_conv = -99.0;
        assert!(!simd_check_convergence(old_ll, new_ll_no_conv, tol));
    }
}

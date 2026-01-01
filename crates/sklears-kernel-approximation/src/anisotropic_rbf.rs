//! Anisotropic RBF Kernel Approximations
//!
//! This module implements anisotropic RBF kernels and their approximations.
//! Anisotropic kernels use different length scales for different dimensions,
//! allowing the kernel to adapt to the varying importance of features.
//!
//! # Key Features
//!
//! - **Anisotropic RBF Sampler**: Random features for anisotropic RBF kernels
//! - **Automatic Relevance Determination (ARD)**: Learn feature relevance
//! - **Mahalanobis Distance**: Use learned covariance matrix
//! - **Robust Anisotropic RBF**: Outlier-resistant anisotropic kernels
//! - **Adaptive Length Scales**: Automatic length scale optimization
//!
//! # Mathematical Background
//!
//! Anisotropic RBF kernel:
//! k(x, x') = σ² exp(-0.5 * (x - x')ᵀ Λ⁻¹ (x - x'))
//!
//! Where Λ = diag(l₁², l₂², ..., lₐ²) is the diagonal matrix of squared length scales.
//!
//! # References

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::Rng;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_core::StandardNormal;
use scirs2_linalg::compat::{Eig, Inverse, SVD};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Transform};
use std::f64::consts::PI;

/// Anisotropic RBF kernel sampler using random Fourier features
#[derive(Debug, Clone)]
/// AnisotropicRBFSampler
pub struct AnisotropicRBFSampler {
    /// Number of random features
    n_components: usize,
    /// Length scales for each dimension (ARD)
    length_scales: Vec<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Whether to learn length scales automatically
    learn_length_scales: bool,
    /// Maximum iterations for length scale optimization
    max_iter: usize,
    /// Convergence tolerance
    tol: f64,
    /// Random seed
    random_state: Option<u64>,
}

/// Fitted anisotropic RBF sampler
#[derive(Debug, Clone)]
/// FittedAnisotropicRBF
pub struct FittedAnisotropicRBF {
    /// Random frequencies
    random_weights: Array2<f64>,
    /// Random biases
    random_biases: Array1<f64>,
    /// Learned length scales
    length_scales: Array1<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Number of features
    n_features: usize,
}

impl AnisotropicRBFSampler {
    /// Create new anisotropic RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            length_scales: vec![1.0],
            signal_variance: 1.0,
            learn_length_scales: true,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
        }
    }

    /// Set length scales manually (disables automatic learning)
    pub fn length_scales(mut self, length_scales: Vec<f64>) -> Self {
        self.length_scales = length_scales;
        self.learn_length_scales = false;
        self
    }

    /// Set signal variance
    pub fn signal_variance(mut self, signal_variance: f64) -> Self {
        self.signal_variance = signal_variance;
        self
    }

    /// Enable/disable automatic length scale learning
    pub fn learn_length_scales(mut self, learn: bool) -> Self {
        self.learn_length_scales = learn;
        self
    }

    /// Set optimization parameters
    pub fn optimization_params(mut self, max_iter: usize, tol: f64) -> Self {
        self.max_iter = max_iter;
        self.tol = tol;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Learn length scales using maximum likelihood
    fn learn_length_scales_ml(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
        let n_features = x.ncols();
        let mut length_scales = Array1::ones(n_features);

        // Initialize with data variance per dimension
        for j in 0..n_features {
            let col = x.column(j);
            let mean = col.mean().unwrap_or(0.0);
            let var = col.mapv(|v| (v - mean).powi(2)).mean().unwrap_or(1.0);
            length_scales[j] = var.sqrt();
        }

        // Simple optimization using coordinate descent
        for _iter in 0..self.max_iter {
            let mut improved = false;

            for j in 0..n_features {
                let old_scale = length_scales[j];
                let mut best_scale = old_scale;
                let mut best_ll = self.compute_log_likelihood(x, &length_scales)?;

                // Try different scale values
                for factor in [0.5, 0.8, 1.2, 2.0] {
                    length_scales[j] = old_scale * factor;
                    let ll = self.compute_log_likelihood(x, &length_scales)?;
                    if ll > best_ll {
                        best_ll = ll;
                        best_scale = length_scales[j];
                        improved = true;
                    }
                }

                length_scales[j] = best_scale;
            }

            if !improved {
                break;
            }
        }

        Ok(length_scales)
    }

    /// Compute log likelihood for length scale optimization
    fn compute_log_likelihood(&self, x: &Array2<f64>, length_scales: &Array1<f64>) -> Result<f64> {
        let n = x.nrows();
        let mut k = Array2::zeros((n, n));

        // Compute kernel matrix
        for i in 0..n {
            for j in 0..n {
                let mut dist_sq = 0.0;
                for d in 0..x.ncols() {
                    let diff = x[(i, d)] - x[(j, d)];
                    dist_sq += diff * diff / (length_scales[d] * length_scales[d]);
                }
                k[(i, j)] = self.signal_variance * (-0.5 * dist_sq).exp();
            }
        }

        // Add noise for numerical stability
        for i in 0..n {
            k[(i, i)] += 1e-6;
        }

        // Compute log determinant and log likelihood
        // Use SVD for numerical stability instead of cholesky
        let (u, s, vt) = k.svd(true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD decomposition failed: {:?}", e))
        })?;
        let s_inv = s.mapv(|x| if x > 1e-10 { 1.0 / x.sqrt() } else { 0.0 });
        let s_inv_diag = Array2::from_diag(&s_inv);
        let _k_inv_sqrt = u.dot(&s_inv_diag).dot(&vt);
        let log_det = s.mapv(|x| if x > 1e-10 { x.ln() } else { -23.0 }).sum(); // log(1e-10) ≈ -23
        let log_likelihood = -0.5 * (log_det + n as f64 * (2.0 * PI).ln());

        Ok(log_likelihood)
    }
}

impl Fit<Array2<f64>, ()> for AnisotropicRBFSampler {
    type Fitted = FittedAnisotropicRBF;
    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();

        // Determine length scales
        let length_scales = if self.learn_length_scales {
            self.learn_length_scales_ml(x)?
        } else if self.length_scales.len() == 1 {
            Array1::from_elem(n_features, self.length_scales[0])
        } else if self.length_scales.len() == n_features {
            Array1::from_vec(self.length_scales.clone())
        } else {
            return Err(SklearsError::InvalidInput(
                "Length scales must be either scalar or match number of features".to_string(),
            ));
        };

        // Generate random frequencies from scaled normal distribution
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let mut random_weights = Array2::zeros((self.n_components, n_features));
        for j in 0..n_features {
            let normal = RandNormal::new(0.0, 1.0 / length_scales[j]).unwrap();
            for i in 0..self.n_components {
                random_weights[(i, j)] = rng.sample(normal);
            }
        }

        // Generate random biases
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_biases = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(FittedAnisotropicRBF {
            random_weights,
            random_biases,
            length_scales,
            signal_variance: self.signal_variance,
            n_features,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedAnisotropicRBF {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_components = self.random_weights.nrows();
        let normalization = (2.0 * self.signal_variance / n_components as f64).sqrt();

        let mut features = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            for j in 0..n_components {
                let mut dot_product = 0.0;
                for k in 0..self.n_features {
                    dot_product += x[(i, k)] * self.random_weights[(j, k)];
                }
                features[(i, j)] = normalization * (dot_product + self.random_biases[j]).cos();
            }
        }

        Ok(features)
    }
}

/// Mahalanobis distance-based RBF sampler
#[derive(Debug, Clone)]
/// MahalanobisRBFSampler
pub struct MahalanobisRBFSampler {
    /// Number of random features
    n_components: usize,
    /// Signal variance
    signal_variance: f64,
    /// Regularization parameter for covariance matrix
    reg_param: f64,
    /// Random seed
    random_state: Option<u64>,
}

/// Fitted Mahalanobis RBF sampler
#[derive(Debug, Clone)]
/// FittedMahalanobisRBF
pub struct FittedMahalanobisRBF {
    /// Random frequencies (pre-whitened)
    random_weights: Array2<f64>,
    /// Random biases
    random_biases: Array1<f64>,
    /// Whitening transformation matrix
    whitening_matrix: Array2<f64>,
    /// Data mean
    mean: Array1<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Number of features
    n_features: usize,
}

impl MahalanobisRBFSampler {
    /// Create new Mahalanobis RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            signal_variance: 1.0,
            reg_param: 1e-6,
            random_state: None,
        }
    }

    /// Set signal variance
    pub fn signal_variance(mut self, signal_variance: f64) -> Self {
        self.signal_variance = signal_variance;
        self
    }

    /// Set regularization parameter
    pub fn reg_param(mut self, reg_param: f64) -> Self {
        self.reg_param = reg_param;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, ()> for MahalanobisRBFSampler {
    type Fitted = FittedMahalanobisRBF;
    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();
        let n_samples = x.nrows();

        // Compute mean
        let mean = x.mean_axis(Axis(0)).unwrap();

        // Center data
        let x_centered = x - &mean.clone().insert_axis(Axis(0));

        // Compute covariance matrix
        let cov = x_centered.t().dot(&x_centered) / (n_samples - 1) as f64;

        // Add regularization
        let mut cov_reg = cov;
        for i in 0..n_features {
            cov_reg[(i, i)] += self.reg_param;
        }

        // Compute whitening matrix (inverse square root of covariance)
        let (eigenvals_complex, eigenvecs_complex) = cov_reg.eig().map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {:?}", e))
        })?;

        // Extract real parts (eigenvalues should be real for symmetric matrix)
        let eigenvals = eigenvals_complex.mapv(|x| x.re);
        let eigenvecs = eigenvecs_complex.mapv(|x| x.re);

        // Create inverse square root matrix
        let mut inv_sqrt_eigenvals = Array1::zeros(n_features);
        for i in 0..n_features {
            inv_sqrt_eigenvals[i] = 1.0 / eigenvals[i].sqrt();
        }

        let whitening_matrix = eigenvecs
            .dot(&Array2::from_diag(&inv_sqrt_eigenvals))
            .dot(&eigenvecs.t());

        // Generate random frequencies from standard normal
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        let mut random_weights = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            for j in 0..n_features {
                random_weights[(i, j)] = rng.sample(StandardNormal);
            }
        }

        // Generate random biases
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_biases = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(FittedMahalanobisRBF {
            random_weights,
            random_biases,
            whitening_matrix,
            mean,
            signal_variance: self.signal_variance,
            n_features,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedMahalanobisRBF {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_components = self.random_weights.nrows();
        let normalization = (2.0 * self.signal_variance / n_components as f64).sqrt();

        // Center and whiten data
        let x_centered = x - &self.mean.clone().insert_axis(Axis(0));
        let x_whitened = x_centered.dot(&self.whitening_matrix);

        let mut features = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            for j in 0..n_components {
                let dot_product = x_whitened.row(i).dot(&self.random_weights.row(j));
                features[(i, j)] = normalization * (dot_product + self.random_biases[j]).cos();
            }
        }

        Ok(features)
    }
}

/// Robust anisotropic RBF sampler with outlier resistance
#[derive(Debug, Clone)]
/// RobustAnisotropicRBFSampler
pub struct RobustAnisotropicRBFSampler {
    /// Number of random features
    n_components: usize,
    /// Robust estimator type
    robust_estimator: RobustEstimator,
    /// Signal variance
    signal_variance: f64,
    /// Random seed
    random_state: Option<u64>,
}

/// Types of robust estimators for covariance
#[derive(Debug, Clone)]
/// RobustEstimator
pub enum RobustEstimator {
    /// Minimum Covariance Determinant
    MCD { support_fraction: f64 },
    /// Minimum Volume Ellipsoid
    MVE,
    /// Huber's M-estimator
    Huber { c: f64 },
}

/// Fitted robust anisotropic RBF sampler
#[derive(Debug, Clone)]
/// FittedRobustAnisotropicRBF
pub struct FittedRobustAnisotropicRBF {
    /// Random frequencies
    random_weights: Array2<f64>,
    /// Random biases
    random_biases: Array1<f64>,
    /// Robust covariance matrix
    robust_cov: Array2<f64>,
    /// Robust mean
    robust_mean: Array1<f64>,
    /// Signal variance
    signal_variance: f64,
    /// Number of features
    n_features: usize,
}

impl RobustAnisotropicRBFSampler {
    /// Create new robust anisotropic RBF sampler
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            robust_estimator: RobustEstimator::MCD {
                support_fraction: 0.8,
            },
            signal_variance: 1.0,
            random_state: None,
        }
    }

    /// Set robust estimator
    pub fn robust_estimator(mut self, estimator: RobustEstimator) -> Self {
        self.robust_estimator = estimator;
        self
    }

    /// Set signal variance
    pub fn signal_variance(mut self, signal_variance: f64) -> Self {
        self.signal_variance = signal_variance;
        self
    }

    /// Set random seed
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute robust covariance using MCD estimator
    fn compute_mcd_covariance(
        &self,
        x: &Array2<f64>,
        support_fraction: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let h = ((n_samples as f64 * support_fraction).floor() as usize).max(n_features + 1);

        let mut best_det = f64::INFINITY;
        let mut best_mean = Array1::zeros(n_features);
        let mut best_cov = Array2::eye(n_features);

        let mut rng = thread_rng();

        // Try multiple random subsets
        for _ in 0..50 {
            // Select random subset
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);
            indices.truncate(h);

            let subset = x.select(Axis(0), &indices);

            // Compute mean and covariance of subset
            let mean = subset.mean_axis(Axis(0)).unwrap();
            let centered = &subset - &mean.clone().insert_axis(Axis(0));
            let cov = centered.t().dot(&centered) / (h - 1) as f64;

            // Add regularization for numerical stability
            let mut cov_reg = cov;
            let trace = cov_reg.diag().sum();
            let reg = trace * 1e-6 / n_features as f64;
            for i in 0..n_features {
                cov_reg[(i, i)] += reg;
            }

            // Compute determinant
            if let Ok((_, s, _)) = cov_reg.svd(false) {
                let log_det: f64 = s.mapv(|x| if x > 1e-10 { x.ln() } else { -23.0 }).sum();
                let det = log_det.exp();

                if det < best_det {
                    best_det = det;
                    best_mean = mean;
                    best_cov = cov_reg;
                }
            }
        }

        Ok((best_mean, best_cov))
    }

    /// Compute robust covariance using Huber's M-estimator
    fn compute_huber_covariance(
        &self,
        x: &Array2<f64>,
        c: f64,
    ) -> Result<(Array1<f64>, Array2<f64>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Start with sample mean and covariance
        let mut mean = x.mean_axis(Axis(0)).unwrap();
        let centered = x - &mean.clone().insert_axis(Axis(0));
        let mut cov = centered.t().dot(&centered) / (n_samples - 1) as f64;

        // Iteratively reweighted estimation
        for _ in 0..20 {
            // Compute Mahalanobis distances
            let cov_inv = cov.inv().map_err(|_| {
                SklearsError::InvalidInput("Covariance matrix not positive definite".to_string())
            })?;

            let mut weights = Array1::zeros(n_samples);
            for i in 0..n_samples {
                let diff = &x.row(i) - &mean;
                let mahal_dist = diff.dot(&cov_inv).dot(&diff).sqrt();
                weights[i] = if mahal_dist <= c { 1.0 } else { c / mahal_dist };
            }

            // Update mean
            let weight_sum = weights.sum();
            let mut new_mean = Array1::zeros(n_features);
            for i in 0..n_samples {
                for j in 0..n_features {
                    new_mean[j] += weights[i] * x[(i, j)];
                }
            }
            new_mean /= weight_sum;

            // Update covariance
            let mut new_cov = Array2::zeros((n_features, n_features));
            for i in 0..n_samples {
                let diff = &x.row(i) - &new_mean;
                let outer =
                    Array2::from_shape_fn((n_features, n_features), |(j, k)| diff[j] * diff[k]);
                new_cov = new_cov + weights[i] * outer;
            }
            new_cov /= weight_sum;

            // Check convergence
            let mean_diff = (&new_mean - &mean).mapv(|x| x.abs()).sum();
            if mean_diff < 1e-6 {
                break;
            }

            mean = new_mean;
            cov = new_cov;
        }

        Ok((mean, cov))
    }
}

impl Fit<Array2<f64>, ()> for RobustAnisotropicRBFSampler {
    type Fitted = FittedRobustAnisotropicRBF;
    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let n_features = x.ncols();

        // Compute robust mean and covariance
        let (robust_mean, robust_cov) = match &self.robust_estimator {
            RobustEstimator::MCD { support_fraction } => {
                self.compute_mcd_covariance(x, *support_fraction)?
            }
            RobustEstimator::MVE => {
                // Simplified MVE - use MCD with smaller support fraction
                self.compute_mcd_covariance(x, 0.5)?
            }
            RobustEstimator::Huber { c } => self.compute_huber_covariance(x, *c)?,
        };

        // Compute precision matrix (inverse covariance)
        let precision = robust_cov.inv().map_err(|_| {
            SklearsError::InvalidInput("Robust covariance matrix not positive definite".to_string())
        })?;

        // Generate random frequencies from multivariate normal with precision matrix
        let mut rng = if let Some(seed) = self.random_state {
            RealStdRng::seed_from_u64(seed)
        } else {
            RealStdRng::from_seed(thread_rng().gen())
        };

        // Cholesky decomposition of precision matrix for sampling
        // Use SVD decomposition for sampling
        let (u, s, _vt) = precision.svd(true).map_err(|e| {
            SklearsError::NumericalError(format!("SVD decomposition failed: {:?}", e))
        })?;
        let s_sqrt = s.mapv(|x| x.sqrt());
        let precision_sqrt = u.dot(&Array2::from_diag(&s_sqrt));

        let mut random_weights = Array2::zeros((self.n_components, n_features));
        for i in 0..self.n_components {
            let mut z = Array1::zeros(n_features);
            for j in 0..n_features {
                z[j] = rng.sample(StandardNormal);
            }
            let w = precision_sqrt.t().dot(&z);
            random_weights.row_mut(i).assign(&w);
        }

        // Generate random biases
        let uniform = RandUniform::new(0.0, 2.0 * PI).unwrap();
        let random_biases = Array1::from_shape_fn(self.n_components, |_| rng.sample(uniform));

        Ok(FittedRobustAnisotropicRBF {
            random_weights,
            random_biases,
            robust_cov,
            robust_mean,
            signal_variance: self.signal_variance,
            n_features,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedRobustAnisotropicRBF {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features {
            return Err(SklearsError::InvalidInput(
                "Feature dimension mismatch".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_components = self.random_weights.nrows();
        let normalization = (2.0 * self.signal_variance / n_components as f64).sqrt();

        // Center data
        let x_centered = x - &self.robust_mean.clone().insert_axis(Axis(0));

        let mut features = Array2::zeros((n_samples, n_components));

        for i in 0..n_samples {
            for j in 0..n_components {
                let dot_product = x_centered.row(i).dot(&self.random_weights.row(j));
                features[(i, j)] = normalization * (dot_product + self.random_biases[j]).cos();
            }
        }

        Ok(features)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_anisotropic_rbf_sampler() {
        let sampler = AnisotropicRBFSampler::new(100)
            .length_scales(vec![1.0, 2.0])
            .signal_variance(1.5);

        let x = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [-1.0, -1.0]];
        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[4, 100]);
        assert!(features.iter().all(|&x| x.is_finite()));

        // Check that features have approximately zero mean
        let mean = features.mean_axis(Axis(0)).unwrap();
        for &m in mean.iter() {
            assert!(m.abs() < 0.5);
        }
    }

    #[test]
    fn test_anisotropic_rbf_learned_scales() {
        let sampler = AnisotropicRBFSampler::new(50)
            .learn_length_scales(true)
            .signal_variance(1.0);

        // Data with different scales in each dimension
        let x = array![
            [0.0, 0.0],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 0.4],
            [5.0, 0.5],
            [-1.0, -0.1],
            [-2.0, -0.2]
        ];

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[8, 50]);
        assert!(features.iter().all(|&x| x.is_finite()));

        // First dimension should have larger length scale than second
        assert!(fitted.length_scales[0] > fitted.length_scales[1]);
    }

    #[test]
    fn test_mahalanobis_rbf_sampler() {
        let sampler = MahalanobisRBFSampler::new(80)
            .signal_variance(2.0)
            .reg_param(1e-4);

        let x = array![
            [1.0, 2.0],
            [2.0, 4.0],
            [3.0, 6.0],
            [1.5, 3.0],
            [2.5, 5.0],
            [0.5, 1.0],
            [3.5, 7.0],
            [4.0, 8.0]
        ];

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[8, 80]);
        assert!(features.iter().all(|&x| x.is_finite()));

        // Test with new data
        let x_test = array![[2.0, 4.0], [1.0, 2.0]];
        let features_test = fitted.transform(&x_test).unwrap();
        assert_eq!(features_test.shape(), &[2, 80]);
        assert!(features_test.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_robust_anisotropic_rbf_mcd() {
        let sampler = RobustAnisotropicRBFSampler::new(60)
            .robust_estimator(RobustEstimator::MCD {
                support_fraction: 0.7,
            })
            .signal_variance(1.0);

        // Data with outliers
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [0.9, 0.9],
            [1.2, 1.2],
            [0.8, 0.8],
            [1.3, 1.3],
            [10.0, 10.0], // outlier
            [1.05, 1.05],
            [0.95, 0.95],
            [1.15, 1.15]
        ];

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[10, 60]);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_robust_anisotropic_rbf_huber() {
        let sampler = RobustAnisotropicRBFSampler::new(40)
            .robust_estimator(RobustEstimator::Huber { c: 1.345 })
            .signal_variance(0.5);

        let x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [1.5, 0.5],
            [0.5, 1.5],
            [5.0, 5.0] // outlier
        ];

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[8, 40]);
        assert!(features.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_anisotropic_rbf_reproducibility() {
        let sampler1 = AnisotropicRBFSampler::new(30)
            .random_state(42)
            .length_scales(vec![1.0, 2.0]);

        let sampler2 = AnisotropicRBFSampler::new(30)
            .random_state(42)
            .length_scales(vec![1.0, 2.0]);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let features1 = sampler1.fit(&x, &()).unwrap().transform(&x).unwrap();
        let features2 = sampler2.fit(&x, &()).unwrap().transform(&x).unwrap();

        for i in 0..features1.nrows() {
            for j in 0..features1.ncols() {
                assert_abs_diff_eq!(features1[(i, j)], features2[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_length_scale_learning() {
        let sampler = AnisotropicRBFSampler::new(20)
            .learn_length_scales(true)
            .optimization_params(10, 1e-4);

        // Create data where first feature varies much more than second
        let x = array![
            [0.0, 0.0],
            [10.0, 0.1],
            [20.0, 0.2],
            [30.0, 0.3],
            [-10.0, -0.1],
            [-20.0, -0.2],
            [15.0, 0.15],
            [25.0, 0.25]
        ];

        let fitted = sampler.fit(&x, &()).unwrap();

        // First dimension should have much larger length scale
        assert!(fitted.length_scales[0] > 5.0 * fitted.length_scales[1]);
    }
}

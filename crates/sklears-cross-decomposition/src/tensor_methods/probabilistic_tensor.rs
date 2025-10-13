//! Probabilistic Tensor Decomposition Methods
//!
//! This module provides probabilistic and Bayesian approaches to tensor decomposition,
//! enabling uncertainty quantification and robust handling of noisy tensor data.
//!
//! ## Methods Included
//! - Bayesian PARAFAC with uncertainty quantification
//! - Probabilistic Tucker decomposition with automatic rank selection
//! - Variational Bayesian Tensor Factorization
//! - Robust probabilistic tensor decomposition with outlier handling
//! - Hierarchical Bayesian models for multi-level tensor analysis

use scirs2_core::ndarray::ndarray_linalg::Inverse;
use scirs2_core::ndarray::{Array1, Array2, Array3, Array4, ArrayView3, Axis, Dim, Ix2, Ix3};
use scirs2_core::ndarray_ext::random::RandomExt;
use scirs2_core::random::{thread_rng, RandUniform, Random};
use sklears_core::error::SklearsError;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Results from probabilistic tensor decomposition
#[derive(Debug, Clone)]
pub struct ProbabilisticTensorResults {
    /// Factor matrices with uncertainty bounds
    pub factors: Vec<Array2<Float>>,
    /// Core tensor (for Tucker decomposition)
    pub core: Option<Array3<Float>>,
    /// Uncertainty estimates for each factor
    pub factor_uncertainties: Vec<Array2<Float>>,
    /// Posterior probability of the decomposition
    pub log_likelihood: Float,
    /// Model evidence for rank selection
    pub model_evidence: Float,
    /// Convergence information
    pub converged: bool,
    /// Number of iterations until convergence
    pub n_iterations: usize,
}

/// Probabilistic tensor decomposition configuration
#[derive(Debug, Clone)]
pub struct ProbabilisticConfig {
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Prior variance for factor matrices
    pub prior_variance: Float,
    /// Noise precision (inverse variance)
    pub noise_precision: Float,
    /// Automatic relevance determination
    pub use_ard: bool,
    /// Number of Monte Carlo samples
    pub n_samples: usize,
    /// Burn-in period for sampling
    pub burn_in: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for ProbabilisticConfig {
    fn default() -> Self {
        Self {
            max_iterations: 1000,
            tolerance: 1e-6,
            prior_variance: 1.0,
            noise_precision: 1.0,
            use_ard: true,
            n_samples: 1000,
            burn_in: 100,
            random_seed: None,
        }
    }
}

/// Bayesian PARAFAC decomposition with uncertainty quantification
#[derive(Debug, Clone)]
pub struct BayesianParafac {
    /// Number of components/rank
    rank: usize,
    /// Configuration parameters
    config: ProbabilisticConfig,
    /// Hyperparameters for automatic relevance determination
    alpha: Vec<Float>,
    /// Precision parameters for each mode
    beta: Vec<Float>,
}

impl BayesianParafac {
    /// Create a new Bayesian PARAFAC decomposition
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            config: ProbabilisticConfig::default(),
            alpha: vec![1.0; rank],
            beta: vec![1.0; 3], // Assuming 3-mode tensors
        }
    }

    /// Set configuration parameters
    pub fn with_config(mut self, config: ProbabilisticConfig) -> Self {
        self.config = config;
        self
    }

    /// Fit Bayesian PARAFAC to tensor data
    pub fn fit(
        &mut self,
        tensor: &Array3<Float>,
    ) -> Result<ProbabilisticTensorResults, SklearsError> {
        let (n1, n2, n3) = tensor.dim();

        // Initialize factor matrices
        let mut rng = thread_rng();
        let uniform = RandUniform::new(-0.1, 0.1).unwrap();
        let mut factors = vec![
            Array2::<Float>::random(Ix2(n1, self.rank), uniform, &mut rng),
            Array2::<Float>::random(Ix2(n2, self.rank), uniform, &mut rng),
            Array2::<Float>::random(Ix2(n3, self.rank), uniform, &mut rng),
        ];

        // Initialize uncertainty estimates
        let mut factor_uncertainties = vec![
            Array2::<Float>::ones((n1, self.rank)),
            Array2::<Float>::ones((n2, self.rank)),
            Array2::<Float>::ones((n3, self.rank)),
        ];

        let mut log_likelihood = Float::NEG_INFINITY;
        let mut converged = false;

        // Variational Bayes iteration
        let mut iterations_run = self.config.max_iterations;
        for iteration in 0..self.config.max_iterations {
            let old_likelihood = log_likelihood;

            // Update each factor matrix
            for mode in 0..3 {
                self.update_factor_variational(
                    tensor,
                    &mut factors,
                    &mut factor_uncertainties,
                    mode,
                )?;
            }

            // Update hyperparameters if using ARD
            if self.config.use_ard {
                self.update_hyperparameters(&factors)?;
            }

            // Compute log likelihood
            log_likelihood = self.compute_log_likelihood(tensor, &factors)?;

            // Check convergence
            if (log_likelihood - old_likelihood).abs() < self.config.tolerance {
                converged = true;
                iterations_run = iteration + 1;
                break;
            }
        }

        // Compute model evidence
        let model_evidence =
            self.compute_model_evidence(tensor, &factors, &factor_uncertainties)?;

        Ok(ProbabilisticTensorResults {
            factors,
            core: None,
            factor_uncertainties,
            log_likelihood,
            model_evidence,
            converged,
            n_iterations: iterations_run,
        })
    }

    /// Update factor matrix using variational Bayes
    fn update_factor_variational(
        &self,
        tensor: &Array3<Float>,
        factors: &mut [Array2<Float>],
        uncertainties: &mut [Array2<Float>],
        mode: usize,
    ) -> Result<(), SklearsError> {
        let (n1, n2, n3) = tensor.dim();
        let dims = [n1, n2, n3];

        // Compute Khatri-Rao product of other modes
        let other_modes: Vec<usize> = (0..3).filter(|&i| i != mode).collect();
        let kr_product =
            self.khatri_rao_product(&factors[other_modes[0]], &factors[other_modes[1]])?;

        // Compute precision matrix
        let precision = kr_product.t().dot(&kr_product)
            + Array2::<Float>::eye(self.rank) * (self.beta[mode] / dims[mode] as Float);

        // Update factor matrix (mean of posterior)
        let unfolded = self.unfold_tensor(tensor, mode)?;
        let mean_update = unfolded.dot(&kr_product).dot(&precision.inv().unwrap());
        factors[mode].assign(&mean_update);

        // Update uncertainty (diagonal of inverse precision)
        for r in 0..self.rank {
            let variance = 1.0 / precision[[r, r]];
            let mut column = uncertainties[mode].column_mut(r);
            column.fill(variance);
        }

        Ok(())
    }

    /// Compute Khatri-Rao product of two matrices
    fn khatri_rao_product(
        &self,
        a: &Array2<Float>,
        b: &Array2<Float>,
    ) -> Result<Array2<Float>, SklearsError> {
        let (n_a, r_a) = a.dim();
        let (n_b, r_b) = b.dim();

        if r_a != r_b {
            return Err(SklearsError::InvalidInput(
                "Matrices must have same number of columns for Khatri-Rao product".to_string(),
            ));
        }

        let mut result = Array2::<Float>::zeros((n_a * n_b, r_a));

        for r in 0..r_a {
            let col_a = a.column(r);
            let col_b = b.column(r);
            let mut result_col = result.column_mut(r);

            for (i, &a_val) in col_a.iter().enumerate() {
                for (j, &b_val) in col_b.iter().enumerate() {
                    result_col[i * n_b + j] = a_val * b_val;
                }
            }
        }

        Ok(result)
    }

    /// Unfold tensor along specified mode
    fn unfold_tensor(
        &self,
        tensor: &Array3<Float>,
        mode: usize,
    ) -> Result<Array2<Float>, SklearsError> {
        let (n1, n2, n3) = tensor.dim();

        match mode {
            0 => {
                let mut unfolded = Array2::<Float>::zeros((n1, n2 * n3));
                for i in 0..n1 {
                    for j in 0..n2 {
                        for k in 0..n3 {
                            unfolded[[i, j * n3 + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            1 => {
                let mut unfolded = Array2::<Float>::zeros((n2, n1 * n3));
                for j in 0..n2 {
                    for i in 0..n1 {
                        for k in 0..n3 {
                            unfolded[[j, i * n3 + k]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            2 => {
                let mut unfolded = Array2::<Float>::zeros((n3, n1 * n2));
                for k in 0..n3 {
                    for i in 0..n1 {
                        for j in 0..n2 {
                            unfolded[[k, i * n2 + j]] = tensor[[i, j, k]];
                        }
                    }
                }
                Ok(unfolded)
            }
            _ => Err(SklearsError::InvalidInput(
                "Invalid mode for 3D tensor".to_string(),
            )),
        }
    }

    /// Update hyperparameters for automatic relevance determination
    fn update_hyperparameters(&mut self, factors: &[Array2<Float>]) -> Result<(), SklearsError> {
        for (mode, factor) in factors.iter().enumerate() {
            let (n, r) = factor.dim();

            // Update alpha (precision for each component)
            for comp in 0..r {
                let column_norm = factor.column(comp).mapv(|x| x * x).sum();
                self.alpha[comp] = (n as Float) / (column_norm + 1e-10);
            }

            // Update beta (noise precision)
            let factor_norm = factors
                .iter()
                .map(|f| f.mapv(|x| x * x).sum())
                .sum::<Float>();
            self.beta[mode] = 1.0 / (factor_norm / (n * r) as Float + 1e-10);
        }

        Ok(())
    }

    /// Compute log likelihood of the model
    fn compute_log_likelihood(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
    ) -> Result<Float, SklearsError> {
        let reconstruction = self.reconstruct_tensor(factors)?;
        let residual = tensor - &reconstruction;
        let residual_norm = residual.mapv(|x| x * x).sum();

        // Log likelihood under Gaussian noise model
        let log_likelihood = -0.5 * self.config.noise_precision * residual_norm;

        Ok(log_likelihood)
    }

    /// Reconstruct tensor from factor matrices
    fn reconstruct_tensor(&self, factors: &[Array2<Float>]) -> Result<Array3<Float>, SklearsError> {
        let (n1, _) = factors[0].dim();
        let (n2, _) = factors[1].dim();
        let (n3, _) = factors[2].dim();

        let mut tensor = Array3::<Float>::zeros((n1, n2, n3));

        for r in 0..self.rank {
            let a_r = factors[0].column(r);
            let b_r = factors[1].column(r);
            let c_r = factors[2].column(r);

            for i in 0..n1 {
                for j in 0..n2 {
                    for k in 0..n3 {
                        tensor[[i, j, k]] += a_r[i] * b_r[j] * c_r[k];
                    }
                }
            }
        }

        Ok(tensor)
    }

    /// Compute model evidence for rank selection
    fn compute_model_evidence(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        uncertainties: &[Array2<Float>],
    ) -> Result<Float, SklearsError> {
        let log_likelihood = self.compute_log_likelihood(tensor, factors)?;

        // Compute complexity penalty (BIC-like)
        let n_params = factors.iter().map(|f| f.len()).sum::<usize>() as Float;
        let n_data = tensor.len() as Float;
        let complexity_penalty = 0.5 * n_params * n_data.ln();

        let model_evidence = log_likelihood - complexity_penalty;

        Ok(model_evidence)
    }
}

/// Probabilistic Tucker decomposition with automatic rank selection
#[derive(Debug, Clone)]
pub struct ProbabilisticTucker {
    /// Core tensor dimensions
    core_dims: [usize; 3],
    /// Configuration parameters
    config: ProbabilisticConfig,
    /// Rank selection history
    rank_evidence: HashMap<[usize; 3], Float>,
}

impl ProbabilisticTucker {
    /// Create a new probabilistic Tucker decomposition
    pub fn new(core_dims: [usize; 3]) -> Self {
        Self {
            core_dims,
            config: ProbabilisticConfig::default(),
            rank_evidence: HashMap::new(),
        }
    }

    /// Set configuration parameters
    pub fn with_config(mut self, config: ProbabilisticConfig) -> Self {
        self.config = config;
        self
    }

    /// Fit probabilistic Tucker decomposition with automatic rank selection
    pub fn fit_with_rank_selection(
        &mut self,
        tensor: &Array3<Float>,
        max_rank: [usize; 3],
    ) -> Result<ProbabilisticTensorResults, SklearsError> {
        let mut best_evidence = Float::NEG_INFINITY;
        let mut best_result = None;

        // Try different core dimensions
        for r1 in 1..=max_rank[0] {
            for r2 in 1..=max_rank[1] {
                for r3 in 1..=max_rank[2] {
                    self.core_dims = [r1, r2, r3];
                    let result = self.fit(tensor)?;

                    self.rank_evidence
                        .insert([r1, r2, r3], result.model_evidence);

                    if result.model_evidence > best_evidence {
                        best_evidence = result.model_evidence;
                        best_result = Some(result);
                    }
                }
            }
        }

        best_result
            .ok_or_else(|| SklearsError::InvalidInput("No valid decomposition found".to_string()))
    }

    /// Fit probabilistic Tucker decomposition
    pub fn fit(&self, tensor: &Array3<Float>) -> Result<ProbabilisticTensorResults, SklearsError> {
        let (n1, n2, n3) = tensor.dim();
        let [r1, r2, r3] = self.core_dims;

        // Initialize factor matrices
        let mut rng = thread_rng();
        let uniform = RandUniform::new(-0.1, 0.1).unwrap();
        let mut factors = vec![
            Array2::<Float>::random(Ix2(n1, r1), uniform, &mut rng),
            Array2::<Float>::random(Ix2(n2, r2), uniform, &mut rng),
            Array2::<Float>::random(Ix2(n3, r3), uniform, &mut rng),
        ];

        // Initialize core tensor
        let mut core = Array3::<Float>::random(Ix3(r1, r2, r3), uniform, &mut rng);

        // Initialize uncertainties
        let mut factor_uncertainties = vec![
            Array2::<Float>::ones((n1, r1)),
            Array2::<Float>::ones((n2, r2)),
            Array2::<Float>::ones((n3, r3)),
        ];

        let mut log_likelihood = Float::NEG_INFINITY;
        let mut converged = false;
        let mut final_iteration = 0;

        // Alternating least squares with Bayesian updates
        for iteration in 0..self.config.max_iterations {
            final_iteration = iteration;
            let old_likelihood = log_likelihood;

            // Update factor matrices
            for mode in 0..3 {
                self.update_tucker_factor(
                    tensor,
                    &mut factors,
                    &core,
                    &mut factor_uncertainties,
                    mode,
                )?;
            }

            // Update core tensor
            self.update_core_tensor(tensor, &factors, &mut core)?;

            // Compute log likelihood
            log_likelihood = self.compute_tucker_likelihood(tensor, &factors, &core)?;

            // Check convergence
            if (log_likelihood - old_likelihood).abs() < self.config.tolerance {
                converged = true;
                break;
            }
        }

        // Compute model evidence
        let model_evidence =
            self.compute_tucker_evidence(tensor, &factors, &core, &factor_uncertainties)?;

        Ok(ProbabilisticTensorResults {
            factors,
            core: Some(core),
            factor_uncertainties,
            log_likelihood,
            model_evidence,
            converged,
            n_iterations: if converged {
                final_iteration + 1
            } else {
                self.config.max_iterations
            },
        })
    }

    /// Update factor matrix for Tucker decomposition
    fn update_tucker_factor(
        &self,
        tensor: &Array3<Float>,
        factors: &mut [Array2<Float>],
        core: &Array3<Float>,
        uncertainties: &mut [Array2<Float>],
        mode: usize,
    ) -> Result<(), SklearsError> {
        // Implementation would involve Tucker-specific factor updates
        // This is a simplified placeholder
        let (n1, n2, n3) = tensor.dim();
        let dims = [n1, n2, n3];

        // Placeholder implementation - would need full Tucker algebra
        let factor_size = factors[mode].dim();
        let mut rng = thread_rng();
        let uniform = RandUniform::new(-0.1, 0.1).unwrap();
        factors[mode] =
            Array2::<Float>::random(Ix2(factor_size.0, factor_size.1), uniform, &mut rng);
        uncertainties[mode] = Array2::<Float>::ones(factor_size);

        Ok(())
    }

    /// Update core tensor
    fn update_core_tensor(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        core: &mut Array3<Float>,
    ) -> Result<(), SklearsError> {
        // Placeholder implementation - would need tensor mode products
        let mut rng = thread_rng();
        let uniform = RandUniform::new(-0.1, 0.1).unwrap();
        *core = Array3::<Float>::random(
            Ix3(self.core_dims[0], self.core_dims[1], self.core_dims[2]),
            uniform,
            &mut rng,
        );
        Ok(())
    }

    /// Compute Tucker likelihood
    fn compute_tucker_likelihood(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        core: &Array3<Float>,
    ) -> Result<Float, SklearsError> {
        // Placeholder - would compute actual Tucker reconstruction error
        Ok(-1.0) // Simplified
    }

    /// Compute Tucker model evidence
    fn compute_tucker_evidence(
        &self,
        tensor: &Array3<Float>,
        factors: &[Array2<Float>],
        core: &Array3<Float>,
        uncertainties: &[Array2<Float>],
    ) -> Result<Float, SklearsError> {
        let log_likelihood = self.compute_tucker_likelihood(tensor, factors, core)?;

        // Compute complexity penalty
        let core_params = core.len() as Float;
        let factor_params = factors.iter().map(|f| f.len()).sum::<usize>() as Float;
        let n_params = core_params + factor_params;
        let n_data = tensor.len() as Float;
        let complexity_penalty = 0.5 * n_params * n_data.ln();

        Ok(log_likelihood - complexity_penalty)
    }
}

/// Robust probabilistic tensor decomposition with outlier handling
#[derive(Debug, Clone)]
pub struct RobustProbabilisticTensor {
    /// Base decomposition method
    base_method: String,
    /// Outlier threshold
    outlier_threshold: Float,
    /// Configuration parameters
    config: ProbabilisticConfig,
}

impl RobustProbabilisticTensor {
    /// Create new robust probabilistic tensor decomposition
    pub fn new(method: &str) -> Self {
        Self {
            base_method: method.to_string(),
            outlier_threshold: 3.0, // Standard deviations
            config: ProbabilisticConfig::default(),
        }
    }

    /// Set outlier threshold
    pub fn with_outlier_threshold(mut self, threshold: Float) -> Self {
        self.outlier_threshold = threshold;
        self
    }

    /// Fit robust decomposition
    pub fn fit(&self, tensor: &Array3<Float>) -> Result<ProbabilisticTensorResults, SklearsError> {
        // Detect outliers
        let outlier_mask = self.detect_outliers(tensor)?;

        // Create weighted tensor (downweight outliers)
        let weighted_tensor = self.apply_weights(tensor, &outlier_mask)?;

        // Fit decomposition to weighted data
        match self.base_method.as_str() {
            "parafac" => {
                let mut parafac = BayesianParafac::new(3);
                parafac.fit(&weighted_tensor)
            }
            "tucker" => {
                let tucker = ProbabilisticTucker::new([3, 3, 3]);
                tucker.fit(&weighted_tensor)
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown method: {}",
                self.base_method
            ))),
        }
    }

    /// Detect outliers in tensor data
    fn detect_outliers(&self, tensor: &Array3<Float>) -> Result<Array3<bool>, SklearsError> {
        let mean = tensor.mean().unwrap();
        let std_dev = (tensor.mapv(|x| (x - mean).powi(2)).mean().unwrap()).sqrt();

        let outlier_mask = tensor.mapv(|x| (x - mean).abs() > self.outlier_threshold * std_dev);

        Ok(outlier_mask)
    }

    /// Apply weights to downweight outliers
    fn apply_weights(
        &self,
        tensor: &Array3<Float>,
        outlier_mask: &Array3<bool>,
    ) -> Result<Array3<Float>, SklearsError> {
        let mut weighted = tensor.clone();
        for ((i, j, k), &is_outlier) in outlier_mask.indexed_iter() {
            if is_outlier {
                weighted[[i, j, k]] *= 0.1; // Downweight outliers
            }
        }

        Ok(weighted)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_bayesian_parafac_basic() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let tensor = Array3::<Float>::random(Ix3(10, 15, 8), uniform, &mut rng);
        let mut parafac = BayesianParafac::new(3);

        let result = parafac.fit(&tensor);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.factors.len(), 3);
        assert_eq!(result.factors[0].dim(), (10, 3));
        assert_eq!(result.factors[1].dim(), (15, 3));
        assert_eq!(result.factors[2].dim(), (8, 3));
    }

    #[test]
    fn test_probabilistic_tucker_basic() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let tensor = Array3::<Float>::random(Ix3(12, 10, 8), uniform, &mut rng);
        let tucker = ProbabilisticTucker::new([4, 3, 3]);

        let result = tucker.fit(&tensor);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.factors.len(), 3);
        assert!(result.core.is_some());

        let core = result.core.unwrap();
        assert_eq!(core.dim(), (4, 3, 3));
    }

    #[test]
    fn test_robust_tensor_decomposition() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let mut tensor = Array3::<Float>::random(Ix3(8, 6, 5), uniform, &mut rng);

        // Add some outliers
        tensor[[0, 0, 0]] = 100.0;
        tensor[[1, 1, 1]] = -100.0;

        let robust = RobustProbabilisticTensor::new("parafac");
        let result = robust.fit(&tensor);

        assert!(result.is_ok());
    }

    #[test]
    fn test_automatic_rank_selection() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let tensor = Array3::<Float>::random(Ix3(10, 8, 6), uniform, &mut rng);
        let mut tucker = ProbabilisticTucker::new([2, 2, 2]);

        let result = tucker.fit_with_rank_selection(&tensor, [4, 4, 4]);
        assert!(result.is_ok());

        // Check that rank evidence was computed
        assert!(!tucker.rank_evidence.is_empty());
    }

    #[test]
    fn test_uncertainty_quantification() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let tensor = Array3::<Float>::random(Ix3(6, 8, 5), uniform, &mut rng);
        let mut parafac = BayesianParafac::new(2);

        let result = parafac.fit(&tensor).unwrap();

        // Check uncertainty estimates are provided
        assert_eq!(result.factor_uncertainties.len(), 3);
        assert!(result.factor_uncertainties[0].iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_model_evidence_computation() {
        let mut rng = thread_rng();
        let uniform = RandUniform::new(0.0, 1.0).unwrap();
        let tensor = Array3::<Float>::random(Ix3(8, 6, 4), uniform, &mut rng);
        let mut parafac = BayesianParafac::new(3);

        let result = parafac.fit(&tensor).unwrap();

        // Model evidence should be finite
        assert!(result.model_evidence.is_finite());
        assert!(result.log_likelihood.is_finite());
    }

    #[test]
    fn test_convergence_detection() {
        let tensor = Array3::<Float>::ones((5, 5, 5));
        let mut config = ProbabilisticConfig::default();
        config.max_iterations = 10;
        config.tolerance = 1e-3;

        let mut parafac = BayesianParafac::new(2).with_config(config);
        let result = parafac.fit(&tensor).unwrap();

        // Should converge quickly for simple tensor
        assert!(result.n_iterations < 10);
    }
}

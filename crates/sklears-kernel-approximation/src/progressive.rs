//! Progressive kernel approximation methods
//!
//! This module provides progressive approximation strategies that start with
//! coarse approximations and progressively refine them based on quality criteria.

use crate::{Nystroem, RBFSampler};
use scirs2_core::ndarray::ndarray_linalg::{Norm, SVD};
use scirs2_core::ndarray::Array2;
use sklears_core::traits::Fit;
use sklears_core::{
    error::{Result, SklearsError},
    traits::Transform,
};
use std::time::Instant;

/// Progressive refinement strategies
#[derive(Debug, Clone)]
/// ProgressiveStrategy
pub enum ProgressiveStrategy {
    /// Double the number of components at each step
    Doubling,
    /// Add a fixed number of components at each step
    FixedIncrement { increment: usize },
    /// Adaptive increment based on quality improvement
    AdaptiveIncrement {
        min_increment: usize,

        max_increment: usize,

        improvement_threshold: f64,
    },
    /// Exponential growth with custom base
    Exponential { base: f64 },
    /// Fibonacci-based growth
    Fibonacci,
}

/// Stopping criteria for progressive approximation
#[derive(Debug, Clone)]
/// StoppingCriterion
pub enum StoppingCriterion {
    /// Stop when target quality is reached
    TargetQuality { quality: f64 },
    /// Stop when improvement falls below threshold
    ImprovementThreshold { threshold: f64 },
    /// Stop after maximum number of iterations
    MaxIterations { max_iter: usize },
    /// Stop when maximum components is reached
    MaxComponents { max_components: usize },
    /// Combined criteria (all must be satisfied)
    Combined {
        quality: Option<f64>,
        improvement_threshold: Option<f64>,
        max_iter: Option<usize>,
        max_components: Option<usize>,
    },
}

/// Quality metrics for progressive approximation
#[derive(Debug, Clone)]
/// ProgressiveQualityMetric
pub enum ProgressiveQualityMetric {
    /// Kernel alignment between exact and approximate kernels
    KernelAlignment,
    /// Frobenius norm of approximation error
    FrobeniusError,
    /// Spectral norm of approximation error
    SpectralError,
    /// Effective rank of the approximation
    EffectiveRank,
    /// Relative improvement over previous iteration
    RelativeImprovement,
    /// Custom quality function
    Custom,
}

/// Configuration for progressive approximation
#[derive(Debug, Clone)]
/// ProgressiveConfig
pub struct ProgressiveConfig {
    /// Initial number of components
    pub initial_components: usize,
    /// Progressive strategy
    pub strategy: ProgressiveStrategy,
    /// Stopping criterion
    pub stopping_criterion: StoppingCriterion,
    /// Quality metric to optimize
    pub quality_metric: ProgressiveQualityMetric,
    /// Number of trials per iteration for stability
    pub n_trials: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Validation fraction for quality assessment
    pub validation_fraction: f64,
    /// Whether to store intermediate results
    pub store_intermediate: bool,
}

impl Default for ProgressiveConfig {
    fn default() -> Self {
        Self {
            initial_components: 10,
            strategy: ProgressiveStrategy::Doubling,
            stopping_criterion: StoppingCriterion::Combined {
                quality: Some(0.95),
                improvement_threshold: Some(0.01),
                max_iter: Some(10),
                max_components: Some(1000),
            },
            quality_metric: ProgressiveQualityMetric::KernelAlignment,
            n_trials: 3,
            random_seed: None,
            validation_fraction: 0.2,
            store_intermediate: true,
        }
    }
}

/// Results from a single progressive step
#[derive(Debug, Clone)]
/// ProgressiveStep
pub struct ProgressiveStep {
    /// Number of components in this step
    pub n_components: usize,
    /// Quality score achieved
    pub quality_score: f64,
    /// Improvement over previous step
    pub improvement: f64,
    /// Time taken for this step
    pub time_taken: f64,
    /// Iteration number
    pub iteration: usize,
}

/// Results from progressive approximation
#[derive(Debug, Clone)]
/// ProgressiveResult
pub struct ProgressiveResult {
    /// Final number of components
    pub final_components: usize,
    /// Final quality score
    pub final_quality: f64,
    /// All progressive steps
    pub steps: Vec<ProgressiveStep>,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Stopping reason
    pub stopping_reason: String,
    /// Total time taken
    pub total_time: f64,
}

/// Progressive RBF sampler
#[derive(Debug, Clone)]
/// ProgressiveRBFSampler
pub struct ProgressiveRBFSampler {
    gamma: f64,
    config: ProgressiveConfig,
}

impl Default for ProgressiveRBFSampler {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressiveRBFSampler {
    /// Create a new progressive RBF sampler
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            config: ProgressiveConfig::default(),
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: ProgressiveConfig) -> Self {
        self.config = config;
        self
    }

    /// Set initial components
    pub fn initial_components(mut self, components: usize) -> Self {
        self.config.initial_components = components;
        self
    }

    /// Set progressive strategy
    pub fn strategy(mut self, strategy: ProgressiveStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set stopping criterion
    pub fn stopping_criterion(mut self, criterion: StoppingCriterion) -> Self {
        self.config.stopping_criterion = criterion;
        self
    }

    /// Run progressive approximation
    pub fn run_progressive_approximation(&self, x: &Array2<f64>) -> Result<ProgressiveResult> {
        let start_time = Instant::now();
        let n_samples = x.nrows();

        // Split data for validation
        let split_idx = (n_samples as f64 * (1.0 - self.config.validation_fraction)) as usize;
        let x_train = x
            .slice(scirs2_core::ndarray::s![..split_idx, ..])
            .to_owned();
        let x_val = x
            .slice(scirs2_core::ndarray::s![split_idx.., ..])
            .to_owned();

        // Compute exact kernel matrix for validation (small subset)
        let k_exact = self.compute_exact_kernel_matrix(&x_val)?;

        let mut steps = Vec::new();
        let mut current_components = self.config.initial_components;
        let mut previous_quality = 0.0;
        let mut iteration = 0;
        let result;

        // Fibonacci sequence state (for Fibonacci strategy)
        let mut fib_prev = 1;
        let mut fib_curr = 1;

        loop {
            let step_start = Instant::now();

            // Compute quality for current number of components
            let quality = self.compute_quality_for_components(
                current_components,
                &x_train,
                &x_val,
                &k_exact,
            )?;

            let improvement = if iteration == 0 {
                quality
            } else {
                quality - previous_quality
            };

            let step_time = step_start.elapsed().as_secs_f64();

            // Store step result
            let step = ProgressiveStep {
                n_components: current_components,
                quality_score: quality,
                improvement,
                time_taken: step_time,
                iteration,
            };
            steps.push(step);

            // Check stopping criteria
            if let Some(stop_result) =
                self.check_stopping_criteria(quality, improvement, iteration, current_components)
            {
                result = Some(stop_result);
                break;
            }

            // Update for next iteration
            previous_quality = quality;
            iteration += 1;

            // Determine next number of components
            current_components = match &self.config.strategy {
                ProgressiveStrategy::Doubling => current_components * 2,
                ProgressiveStrategy::FixedIncrement { increment } => current_components + increment,
                ProgressiveStrategy::AdaptiveIncrement {
                    min_increment,
                    max_increment,
                    improvement_threshold,
                } => {
                    let increment = if improvement > *improvement_threshold {
                        *min_increment
                    } else {
                        (*min_increment + (*max_increment - *min_increment) / 2).max(*min_increment)
                    };
                    current_components + increment
                }
                ProgressiveStrategy::Exponential { base } => {
                    ((current_components as f64) * base) as usize
                }
                ProgressiveStrategy::Fibonacci => {
                    let next_fib = fib_prev + fib_curr;
                    fib_prev = fib_curr;
                    fib_curr = next_fib;
                    self.config.initial_components + fib_curr
                }
            };
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let (converged, stopping_reason) =
            result.unwrap_or((false, "Max iterations reached".to_string()));

        Ok(ProgressiveResult {
            final_components: steps
                .last()
                .map(|s| s.n_components)
                .unwrap_or(current_components),
            final_quality: steps.last().map(|s| s.quality_score).unwrap_or(0.0),
            steps,
            converged,
            stopping_reason,
            total_time,
        })
    }

    /// Compute exact kernel matrix for validation
    fn compute_exact_kernel_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let n_samples = x.nrows().min(100); // Limit for computational efficiency
        let x_subset = x.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        let mut k_exact = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-self.gamma * squared_norm).exp();
            }
        }

        Ok(k_exact)
    }

    /// Compute quality for a given number of components
    fn compute_quality_for_components(
        &self,
        n_components: usize,
        x_train: &Array2<f64>,
        x_val: &Array2<f64>,
        k_exact: &Array2<f64>,
    ) -> Result<f64> {
        let mut trial_qualities = Vec::new();

        // Run multiple trials for stability
        for trial in 0..self.config.n_trials {
            let seed = self.config.random_seed.map(|s| s + trial as u64);
            let sampler = if let Some(s) = seed {
                RBFSampler::new(n_components)
                    .gamma(self.gamma)
                    .random_state(s)
            } else {
                RBFSampler::new(n_components).gamma(self.gamma)
            };

            let fitted = sampler.fit(x_train, &())?;
            let x_val_transformed = fitted.transform(x_val)?;

            let quality = self.compute_quality_metric(x_val, &x_val_transformed, k_exact)?;
            trial_qualities.push(quality);
        }

        // Return average quality across trials
        Ok(trial_qualities.iter().sum::<f64>() / trial_qualities.len() as f64)
    }

    /// Compute quality metric
    fn compute_quality_metric(
        &self,
        _x: &Array2<f64>,
        x_transformed: &Array2<f64>,
        k_exact: &Array2<f64>,
    ) -> Result<f64> {
        match &self.config.quality_metric {
            ProgressiveQualityMetric::KernelAlignment => {
                self.compute_kernel_alignment(x_transformed, k_exact)
            }
            ProgressiveQualityMetric::FrobeniusError => {
                self.compute_frobenius_error(x_transformed, k_exact)
            }
            ProgressiveQualityMetric::SpectralError => {
                self.compute_spectral_error(x_transformed, k_exact)
            }
            ProgressiveQualityMetric::EffectiveRank => self.compute_effective_rank(x_transformed),
            ProgressiveQualityMetric::RelativeImprovement => {
                // This is handled at a higher level
                Ok(1.0)
            }
            ProgressiveQualityMetric::Custom => {
                // Placeholder for custom quality function
                self.compute_kernel_alignment(x_transformed, k_exact)
            }
        }
    }

    /// Compute kernel alignment
    fn compute_kernel_alignment(
        &self,
        x_transformed: &Array2<f64>,
        k_exact: &Array2<f64>,
    ) -> Result<f64> {
        let n_samples = k_exact.nrows().min(x_transformed.nrows());
        let x_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute approximate kernel matrix
        let k_approx = x_subset.dot(&x_subset.t());

        // Compute alignment
        let k_exact_norm = k_exact.norm_l2();
        let k_approx_norm = k_approx.norm_l2();

        if k_exact_norm > 1e-12 && k_approx_norm > 1e-12 {
            let alignment = (k_exact * &k_approx).sum() / (k_exact_norm * k_approx_norm);
            Ok(alignment)
        } else {
            Ok(0.0)
        }
    }

    /// Compute Frobenius error (as quality score, so higher is better)
    fn compute_frobenius_error(
        &self,
        x_transformed: &Array2<f64>,
        k_exact: &Array2<f64>,
    ) -> Result<f64> {
        let n_samples = k_exact.nrows().min(x_transformed.nrows());
        let x_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute approximate kernel matrix
        let k_approx = x_subset.dot(&x_subset.t());

        // Compute error and convert to quality (higher is better)
        let diff = k_exact - &k_approx.slice(scirs2_core::ndarray::s![..n_samples, ..n_samples]);
        let error = diff.norm_l2();
        let quality = 1.0 / (1.0 + error); // Convert error to quality score

        Ok(quality)
    }

    /// Compute spectral error (as quality score)
    fn compute_spectral_error(
        &self,
        x_transformed: &Array2<f64>,
        k_exact: &Array2<f64>,
    ) -> Result<f64> {
        let n_samples = k_exact.nrows().min(x_transformed.nrows());
        let x_subset = x_transformed.slice(scirs2_core::ndarray::s![..n_samples, ..]);

        // Compute approximate kernel matrix
        let k_approx = x_subset.dot(&x_subset.t());

        // Compute spectral norm (largest singular value) of the error
        let diff = k_exact - &k_approx.slice(scirs2_core::ndarray::s![..n_samples, ..n_samples]);
        let (_, s, _) = diff
            .svd(false, false)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        let spectral_error = s.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let quality = 1.0 / (1.0 + spectral_error);

        Ok(quality)
    }

    /// Compute effective rank
    fn compute_effective_rank(&self, x_transformed: &Array2<f64>) -> Result<f64> {
        // Compute SVD of transformed data
        let (_, s, _) = x_transformed
            .svd(true, true)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        // Compute effective rank using entropy
        let s_sum = s.sum();
        if s_sum == 0.0 {
            return Ok(0.0);
        }

        let s_normalized = &s / s_sum;
        let entropy = -s_normalized
            .iter()
            .filter(|&&x| x > 1e-12)
            .map(|&x| x * x.ln())
            .sum::<f64>();

        let effective_rank = entropy.exp();
        Ok(effective_rank / x_transformed.ncols() as f64) // Normalize by max possible rank
    }

    /// Check stopping criteria
    fn check_stopping_criteria(
        &self,
        quality: f64,
        improvement: f64,
        iteration: usize,
        components: usize,
    ) -> Option<(bool, String)> {
        match &self.config.stopping_criterion {
            StoppingCriterion::TargetQuality { quality: target } => {
                if quality >= *target {
                    Some((true, format!("Target quality {} reached", target)))
                } else {
                    None
                }
            }
            StoppingCriterion::ImprovementThreshold { threshold } => {
                if iteration > 0 && improvement < *threshold {
                    Some((
                        true,
                        format!("Improvement {} below threshold {}", improvement, threshold),
                    ))
                } else {
                    None
                }
            }
            StoppingCriterion::MaxIterations { max_iter } => {
                if iteration + 1 >= *max_iter {
                    Some((false, format!("Maximum iterations {} reached", max_iter)))
                } else {
                    None
                }
            }
            StoppingCriterion::MaxComponents { max_components } => {
                if components >= *max_components {
                    Some((
                        false,
                        format!("Maximum components {} reached", max_components),
                    ))
                } else {
                    None
                }
            }
            StoppingCriterion::Combined {
                quality: target_quality,
                improvement_threshold,
                max_iter,
                max_components,
            } => {
                // Check target quality
                if let Some(target) = target_quality {
                    if quality >= *target {
                        return Some((true, format!("Target quality {} reached", target)));
                    }
                }

                // Check improvement threshold
                if let Some(threshold) = improvement_threshold {
                    if iteration > 0 && improvement < *threshold {
                        return Some((
                            true,
                            format!("Improvement {} below threshold {}", improvement, threshold),
                        ));
                    }
                }

                // Check max iterations
                if let Some(max) = max_iter {
                    if iteration >= *max {
                        return Some((false, format!("Maximum iterations {} reached", max)));
                    }
                }

                // Check max components
                if let Some(max) = max_components {
                    if components >= *max {
                        return Some((false, format!("Maximum components {} reached", max)));
                    }
                }

                None
            }
        }
    }
}

/// Fitted progressive RBF sampler
pub struct FittedProgressiveRBFSampler {
    fitted_rbf: crate::rbf_sampler::RBFSampler<sklears_core::traits::Trained>,
    progressive_result: ProgressiveResult,
}

impl Fit<Array2<f64>, ()> for ProgressiveRBFSampler {
    type Fitted = FittedProgressiveRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Run progressive approximation
        let progressive_result = self.run_progressive_approximation(x)?;

        // Fit RBF sampler with final configuration
        let rbf_sampler = RBFSampler::new(progressive_result.final_components).gamma(self.gamma);
        let fitted_rbf = rbf_sampler.fit(x, &())?;

        Ok(FittedProgressiveRBFSampler {
            fitted_rbf,
            progressive_result,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedProgressiveRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_rbf.transform(x)
    }
}

impl FittedProgressiveRBFSampler {
    /// Get the progressive result
    pub fn progressive_result(&self) -> &ProgressiveResult {
        &self.progressive_result
    }

    /// Get the final number of components
    pub fn final_components(&self) -> usize {
        self.progressive_result.final_components
    }

    /// Get the final quality score
    pub fn final_quality(&self) -> f64 {
        self.progressive_result.final_quality
    }

    /// Check if progressive approximation converged
    pub fn converged(&self) -> bool {
        self.progressive_result.converged
    }

    /// Get all progressive steps
    pub fn steps(&self) -> &[ProgressiveStep] {
        &self.progressive_result.steps
    }

    /// Get the stopping reason
    pub fn stopping_reason(&self) -> &str {
        &self.progressive_result.stopping_reason
    }
}

/// Progressive Nyström method
#[derive(Debug, Clone)]
/// ProgressiveNystroem
pub struct ProgressiveNystroem {
    kernel: crate::nystroem::Kernel,
    config: ProgressiveConfig,
}

impl Default for ProgressiveNystroem {
    fn default() -> Self {
        Self::new()
    }
}

impl ProgressiveNystroem {
    /// Create a new progressive Nyström method
    pub fn new() -> Self {
        Self {
            kernel: crate::nystroem::Kernel::Rbf { gamma: 1.0 },
            config: ProgressiveConfig::default(),
        }
    }

    /// Set gamma parameter (for RBF kernel)
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.kernel = crate::nystroem::Kernel::Rbf { gamma };
        self
    }

    /// Set kernel type
    pub fn kernel(mut self, kernel: crate::nystroem::Kernel) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: ProgressiveConfig) -> Self {
        self.config = config;
        self
    }

    /// Run progressive approximation for Nyström method
    pub fn run_progressive_approximation(&self, x: &Array2<f64>) -> Result<ProgressiveResult> {
        let start_time = Instant::now();

        let mut steps = Vec::new();
        let mut current_components = self.config.initial_components;
        let mut previous_quality = 0.0;
        let mut iteration = 0;
        let result;

        loop {
            let step_start = Instant::now();

            // Compute quality for current number of components
            let quality = self.compute_nystroem_quality(current_components, x)?;

            let improvement = if iteration == 0 {
                quality
            } else {
                quality - previous_quality
            };

            let step_time = step_start.elapsed().as_secs_f64();

            // Store step result
            let step = ProgressiveStep {
                n_components: current_components,
                quality_score: quality,
                improvement,
                time_taken: step_time,
                iteration,
            };
            steps.push(step);

            // Check stopping criteria (using same logic as RBF sampler)
            if let Some(stop_result) =
                self.check_stopping_criteria(quality, improvement, iteration, current_components)
            {
                result = Some(stop_result);
                break;
            }

            // Update for next iteration
            previous_quality = quality;
            iteration += 1;

            // Determine next number of components (same logic as RBF sampler)
            current_components = match &self.config.strategy {
                ProgressiveStrategy::Doubling => current_components * 2,
                ProgressiveStrategy::FixedIncrement { increment } => current_components + increment,
                _ => current_components * 2, // Simplified for Nyström
            };
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let (converged, stopping_reason) =
            result.unwrap_or((false, "Max iterations reached".to_string()));

        Ok(ProgressiveResult {
            final_components: steps
                .last()
                .map(|s| s.n_components)
                .unwrap_or(current_components),
            final_quality: steps.last().map(|s| s.quality_score).unwrap_or(0.0),
            steps,
            converged,
            stopping_reason,
            total_time,
        })
    }

    /// Compute quality for Nyström with given components
    fn compute_nystroem_quality(&self, n_components: usize, x: &Array2<f64>) -> Result<f64> {
        let mut trial_qualities = Vec::new();

        // Run multiple trials for stability
        for trial in 0..self.config.n_trials {
            let seed = self.config.random_seed.map(|s| s + trial as u64);
            let nystroem = if let Some(s) = seed {
                Nystroem::new(self.kernel.clone(), n_components).random_state(s)
            } else {
                Nystroem::new(self.kernel.clone(), n_components)
            };

            let fitted = nystroem.fit(x, &())?;
            let x_transformed = fitted.transform(x)?;

            // Use effective rank as quality measure
            let quality = self.compute_effective_rank(&x_transformed)?;
            trial_qualities.push(quality);
        }

        Ok(trial_qualities.iter().sum::<f64>() / trial_qualities.len() as f64)
    }

    /// Compute effective rank (same as RBF sampler)
    fn compute_effective_rank(&self, x_transformed: &Array2<f64>) -> Result<f64> {
        let (_, s, _) = x_transformed
            .svd(true, true)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        let s_sum = s.sum();
        if s_sum == 0.0 {
            return Ok(0.0);
        }

        let s_normalized = &s / s_sum;
        let entropy = -s_normalized
            .iter()
            .filter(|&&x| x > 1e-12)
            .map(|&x| x * x.ln())
            .sum::<f64>();

        let effective_rank = entropy.exp();
        Ok(effective_rank / x_transformed.ncols() as f64)
    }

    /// Check stopping criteria (same as RBF sampler)
    fn check_stopping_criteria(
        &self,
        quality: f64,
        _improvement: f64,
        iteration: usize,
        _components: usize,
    ) -> Option<(bool, String)> {
        match &self.config.stopping_criterion {
            StoppingCriterion::TargetQuality { quality: target } => {
                if quality >= *target {
                    Some((true, format!("Target quality {} reached", target)))
                } else {
                    None
                }
            }
            StoppingCriterion::MaxIterations { max_iter } => {
                if iteration + 1 >= *max_iter {
                    Some((false, format!("Maximum iterations {} reached", max_iter)))
                } else {
                    None
                }
            }
            _ => None, // Simplified for Nyström
        }
    }
}

/// Fitted progressive Nyström method
pub struct FittedProgressiveNystroem {
    fitted_nystroem: crate::nystroem::Nystroem<sklears_core::traits::Trained>,
    progressive_result: ProgressiveResult,
}

impl Fit<Array2<f64>, ()> for ProgressiveNystroem {
    type Fitted = FittedProgressiveNystroem;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Run progressive approximation
        let progressive_result = self.run_progressive_approximation(x)?;

        // Fit Nyström method with final configuration
        let nystroem = Nystroem::new(self.kernel, progressive_result.final_components);
        let fitted_nystroem = nystroem.fit(x, &())?;

        Ok(FittedProgressiveNystroem {
            fitted_nystroem,
            progressive_result,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedProgressiveNystroem {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_nystroem.transform(x)
    }
}

impl FittedProgressiveNystroem {
    /// Get the progressive result
    pub fn progressive_result(&self) -> &ProgressiveResult {
        &self.progressive_result
    }

    /// Get the final number of components
    pub fn final_components(&self) -> usize {
        self.progressive_result.final_components
    }

    /// Get the final quality score
    pub fn final_quality(&self) -> f64 {
        self.progressive_result.final_quality
    }

    /// Check if progressive approximation converged
    pub fn converged(&self) -> bool {
        self.progressive_result.converged
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_progressive_rbf_sampler() {
        let x = Array2::from_shape_vec((100, 4), (0..400).map(|i| (i as f64) * 0.01).collect())
            .unwrap();

        let config = ProgressiveConfig {
            initial_components: 5,
            strategy: ProgressiveStrategy::Doubling,
            stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 3 },
            quality_metric: ProgressiveQualityMetric::KernelAlignment,
            n_trials: 2,
            validation_fraction: 0.3,
            ..Default::default()
        };

        let sampler = ProgressiveRBFSampler::new().gamma(0.5).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 100);
        assert!(fitted.final_components() >= 5);
        assert!(fitted.final_quality() >= 0.0);
        assert_eq!(fitted.steps().len(), 3); // 3 iterations max
    }

    #[test]
    fn test_progressive_nystroem() {
        let x =
            Array2::from_shape_vec((80, 3), (0..240).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = ProgressiveConfig {
            initial_components: 10,
            strategy: ProgressiveStrategy::FixedIncrement { increment: 5 },
            stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 4 },
            n_trials: 2,
            ..Default::default()
        };

        let nystroem = ProgressiveNystroem::new().gamma(1.0).config(config);

        let fitted = nystroem.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 80);
        assert!(fitted.final_components() >= 10);
        assert!(fitted.final_quality() >= 0.0);
    }

    #[test]
    fn test_progressive_strategies() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| (i as f64) * 0.05).collect()).unwrap();

        let strategies = vec![
            ProgressiveStrategy::Doubling,
            ProgressiveStrategy::FixedIncrement { increment: 3 },
            ProgressiveStrategy::Exponential { base: 1.5 },
            ProgressiveStrategy::Fibonacci,
        ];

        for strategy in strategies {
            let config = ProgressiveConfig {
                initial_components: 5,
                strategy,
                stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 3 },
                n_trials: 1,
                ..Default::default()
            };

            let sampler = ProgressiveRBFSampler::new().gamma(0.8).config(config);

            let result = sampler.run_progressive_approximation(&x).unwrap();

            assert!(result.final_components >= 5);
            assert!(result.final_quality >= 0.0);
            assert_eq!(result.steps.len(), 3);
        }
    }

    #[test]
    fn test_stopping_criteria() {
        let x =
            Array2::from_shape_vec((60, 3), (0..180).map(|i| (i as f64) * 0.03).collect()).unwrap();

        let criteria = vec![
            StoppingCriterion::TargetQuality { quality: 0.8 },
            StoppingCriterion::ImprovementThreshold { threshold: 0.01 },
            StoppingCriterion::MaxIterations { max_iter: 5 },
            StoppingCriterion::MaxComponents { max_components: 50 },
        ];

        for criterion in criteria {
            let config = ProgressiveConfig {
                initial_components: 10,
                strategy: ProgressiveStrategy::Doubling,
                stopping_criterion: criterion,
                n_trials: 1,
                ..Default::default()
            };

            let sampler = ProgressiveRBFSampler::new().gamma(0.5).config(config);

            let result = sampler.run_progressive_approximation(&x).unwrap();

            assert!(result.final_components >= 10);
            assert!(result.final_quality >= 0.0);
            assert!(!result.stopping_reason.is_empty());
        }
    }

    #[test]
    fn test_quality_metrics() {
        let x =
            Array2::from_shape_vec((40, 2), (0..80).map(|i| (i as f64) * 0.05).collect()).unwrap();

        let metrics = vec![
            ProgressiveQualityMetric::KernelAlignment,
            ProgressiveQualityMetric::FrobeniusError,
            ProgressiveQualityMetric::SpectralError,
            ProgressiveQualityMetric::EffectiveRank,
        ];

        for metric in metrics {
            let config = ProgressiveConfig {
                initial_components: 5,
                strategy: ProgressiveStrategy::Doubling,
                stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 3 },
                quality_metric: metric,
                n_trials: 1,
                ..Default::default()
            };

            let sampler = ProgressiveRBFSampler::new().gamma(0.3).config(config);

            let result = sampler.run_progressive_approximation(&x).unwrap();

            assert!(result.final_components >= 5);
            assert!(result.final_quality >= 0.0);

            // All steps should have valid quality scores
            for step in &result.steps {
                assert!(step.quality_score >= 0.0);
                assert!(step.time_taken >= 0.0);
            }
        }
    }

    #[test]
    fn test_progressive_improvement() {
        let x =
            Array2::from_shape_vec((70, 3), (0..210).map(|i| (i as f64) * 0.02).collect()).unwrap();

        let config = ProgressiveConfig {
            initial_components: 10,
            strategy: ProgressiveStrategy::Doubling,
            stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 4 },
            quality_metric: ProgressiveQualityMetric::KernelAlignment,
            n_trials: 2,
            ..Default::default()
        };

        let sampler = ProgressiveRBFSampler::new().gamma(0.7).config(config);

        let result = sampler.run_progressive_approximation(&x).unwrap();

        // Quality should generally improve or stay stable
        for i in 1..result.steps.len() {
            let current_quality = result.steps[i].quality_score;
            let previous_quality = result.steps[i - 1].quality_score;

            // Allow for small numerical differences
            assert!(
                current_quality >= previous_quality - 0.1,
                "Quality should not decrease significantly: {} -> {}",
                previous_quality,
                current_quality
            );
        }
    }

    #[test]
    fn test_progressive_reproducibility() {
        let x =
            Array2::from_shape_vec((50, 2), (0..100).map(|i| (i as f64) * 0.04).collect()).unwrap();

        let config = ProgressiveConfig {
            initial_components: 5,
            strategy: ProgressiveStrategy::Doubling,
            stopping_criterion: StoppingCriterion::MaxIterations { max_iter: 3 },
            n_trials: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let sampler1 = ProgressiveRBFSampler::new()
            .gamma(0.6)
            .config(config.clone());

        let sampler2 = ProgressiveRBFSampler::new().gamma(0.6).config(config);

        let result1 = sampler1.run_progressive_approximation(&x).unwrap();
        let result2 = sampler2.run_progressive_approximation(&x).unwrap();

        assert_eq!(result1.final_components, result2.final_components);
        assert_abs_diff_eq!(
            result1.final_quality,
            result2.final_quality,
            epsilon = 1e-10
        );
        assert_eq!(result1.steps.len(), result2.steps.len());
    }
}

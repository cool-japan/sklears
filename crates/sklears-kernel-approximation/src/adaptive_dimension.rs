//! Adaptive feature dimension selection for kernel approximations
//!
//! This module provides methods to automatically determine the optimal number of features
//! for kernel approximation methods based on approximation quality metrics.

use crate::RBFSampler;
use scirs2_core::ndarray::ndarray_linalg::{Norm, SVD};
use scirs2_core::ndarray::{concatenate, s, Array2, Axis};
use sklears_core::traits::Fit;
use sklears_core::{
    error::{Result, SklearsError},
    traits::Transform,
};
use std::collections::HashMap;

/// Quality metrics for approximation assessment
#[derive(Debug, Clone)]
/// QualityMetric
pub enum QualityMetric {
    /// Frobenius norm of the approximation error
    FrobeniusNorm,
    /// Spectral norm (largest singular value) of the error
    SpectralNorm,
    /// Nuclear norm (sum of singular values) of the error
    NuclearNorm,
    /// Relative Frobenius error
    RelativeFrobeniusNorm,
    /// Kernel alignment score
    KernelAlignment,
    /// Effective rank of the approximation
    EffectiveRank,
    /// Cross-validation score
    CrossValidation,
    /// Approximation trace
    Trace,
}

/// Selection strategy for adaptive dimension selection
#[derive(Debug, Clone)]
/// SelectionStrategy
pub enum SelectionStrategy {
    /// Select dimension that meets error tolerance
    ErrorTolerance { tolerance: f64 },
    /// Select dimension with best quality/dimension trade-off
    QualityEfficiency { efficiency_threshold: f64 },
    /// Select dimension using elbow method
    ElbowMethod { sensitivity: f64 },
    /// Select dimension with cross-validation
    CrossValidation { n_folds: usize },
    /// Select dimension based on information criteria
    InformationCriteria { criterion: String },
    /// Select dimension using early stopping
    EarlyStopping {
        patience: usize,
        min_improvement: f64,
    },
}

/// Configuration for adaptive dimension selection
#[derive(Debug, Clone)]
/// AdaptiveDimensionConfig
pub struct AdaptiveDimensionConfig {
    /// Minimum number of features to test
    pub min_features: usize,
    /// Maximum number of features to test
    pub max_features: usize,
    /// Step size for feature testing
    pub step_size: usize,
    /// Quality metric to optimize
    pub quality_metric: QualityMetric,
    /// Selection strategy
    pub selection_strategy: SelectionStrategy,
    /// Number of random trials for each dimension
    pub n_trials: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Validation fraction for quality assessment
    pub validation_fraction: f64,
}

impl Default for AdaptiveDimensionConfig {
    fn default() -> Self {
        Self {
            min_features: 10,
            max_features: 1000,
            step_size: 10,
            quality_metric: QualityMetric::KernelAlignment,
            selection_strategy: SelectionStrategy::ErrorTolerance { tolerance: 0.1 },
            n_trials: 3,
            random_seed: None,
            validation_fraction: 0.2,
        }
    }
}

/// Results from adaptive dimension selection
#[derive(Debug, Clone)]
/// DimensionSelectionResult
pub struct DimensionSelectionResult {
    /// Selected optimal dimension
    pub optimal_dimension: usize,
    /// Quality scores for all tested dimensions
    pub quality_scores: HashMap<usize, f64>,
    /// Approximation errors for all tested dimensions
    pub approximation_errors: HashMap<usize, f64>,
    /// Computational times for all tested dimensions
    pub computation_times: HashMap<usize, f64>,
    /// Memory usage for all tested dimensions
    pub memory_usage: HashMap<usize, usize>,
}

/// Adaptive RBF sampler with automatic dimension selection
#[derive(Debug, Clone)]
/// AdaptiveRBFSampler
pub struct AdaptiveRBFSampler {
    gamma: f64,
    config: AdaptiveDimensionConfig,
}

impl AdaptiveRBFSampler {
    /// Create a new adaptive RBF sampler
    pub fn new() -> Self {
        Self {
            gamma: 1.0,
            config: AdaptiveDimensionConfig::default(),
        }
    }

    /// Set gamma parameter
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set configuration
    pub fn config(mut self, config: AdaptiveDimensionConfig) -> Self {
        self.config = config;
        self
    }

    /// Perform adaptive dimension selection
    pub fn select_dimension(&self, x: &Array2<f64>) -> Result<DimensionSelectionResult> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Split data for validation
        let split_idx = (n_samples as f64 * (1.0 - self.config.validation_fraction)) as usize;
        let x_train = x.slice(s![..split_idx, ..]).to_owned();
        let x_val = x.slice(s![split_idx.., ..]).to_owned();

        let mut quality_scores = HashMap::new();
        let mut approximation_errors = HashMap::new();
        let mut computation_times = HashMap::new();
        let mut memory_usage = HashMap::new();

        // Test different dimensions
        let dimensions: Vec<usize> = (self.config.min_features..=self.config.max_features)
            .step_by(self.config.step_size)
            .collect();

        for &n_components in &dimensions {
            let mut trial_scores = Vec::new();
            let mut trial_errors = Vec::new();
            let mut trial_times = Vec::new();

            // Run multiple trials for each dimension
            for trial in 0..self.config.n_trials {
                let start_time = std::time::Instant::now();

                // Create RBF sampler with current dimension
                let seed = self.config.random_seed.map(|s| s + trial as u64);
                let sampler = if let Some(s) = seed {
                    RBFSampler::new(n_components)
                        .gamma(self.gamma)
                        .random_state(s)
                } else {
                    RBFSampler::new(n_components).gamma(self.gamma)
                };

                // Fit and transform
                let fitted = sampler.fit(&x_train, &())?;
                let x_train_transformed = fitted.transform(&x_train)?;
                let x_val_transformed = fitted.transform(&x_val)?;

                let elapsed = start_time.elapsed().as_secs_f64();
                trial_times.push(elapsed);

                // Compute quality metrics
                let quality_score = self.compute_quality_score(
                    &x_train,
                    &x_val,
                    &x_train_transformed,
                    &x_val_transformed,
                    &fitted,
                )?;

                let approximation_error = self.compute_approximation_error(
                    &x_train,
                    &x_val,
                    &x_train_transformed,
                    &x_val_transformed,
                )?;

                trial_scores.push(quality_score);
                trial_errors.push(approximation_error);
            }

            // Average across trials
            let avg_score = trial_scores.iter().sum::<f64>() / trial_scores.len() as f64;
            let avg_error = trial_errors.iter().sum::<f64>() / trial_errors.len() as f64;
            let avg_time = trial_times.iter().sum::<f64>() / trial_times.len() as f64;

            quality_scores.insert(n_components, avg_score);
            approximation_errors.insert(n_components, avg_error);
            computation_times.insert(n_components, avg_time);
            memory_usage.insert(n_components, n_components * n_features * 8); // Rough estimate
        }

        // Select optimal dimension based on strategy
        let optimal_dimension =
            self.select_optimal_dimension(&dimensions, &quality_scores, &approximation_errors)?;

        Ok(DimensionSelectionResult {
            optimal_dimension,
            quality_scores,
            approximation_errors,
            computation_times,
            memory_usage,
        })
    }

    fn compute_quality_score(
        &self,
        x_train: &Array2<f64>,
        x_val: &Array2<f64>,
        x_train_transformed: &Array2<f64>,
        x_val_transformed: &Array2<f64>,
        fitted_sampler: &crate::rbf_sampler::RBFSampler<sklears_core::traits::Trained>,
    ) -> Result<f64> {
        match &self.config.quality_metric {
            QualityMetric::KernelAlignment => {
                self.compute_kernel_alignment(x_train, x_train_transformed)
            }
            QualityMetric::EffectiveRank => self.compute_effective_rank(x_train_transformed),
            QualityMetric::FrobeniusNorm => {
                self.compute_frobenius_approximation_quality(x_train, x_train_transformed)
            }
            QualityMetric::RelativeFrobeniusNorm => {
                self.compute_relative_frobenius_quality(x_train, x_train_transformed)
            }
            QualityMetric::CrossValidation => {
                self.compute_cross_validation_score(x_train, x_train_transformed)
            }
            _ => {
                // Default to kernel alignment for unsupported metrics
                self.compute_kernel_alignment(x_train, x_train_transformed)
            }
        }
    }

    fn compute_kernel_alignment(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
    ) -> Result<f64> {
        // Compute exact kernel matrix (small subset for efficiency)
        let n_samples = x.nrows().min(100); // Limit for computational efficiency
        let x_subset = x.slice(s![..n_samples, ..]);

        let mut k_exact = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-self.gamma * squared_norm).exp();
            }
        }

        // Compute approximate kernel matrix
        let x_transformed_subset = x_transformed.slice(s![..n_samples, ..]);
        let k_approx = x_transformed_subset.dot(&x_transformed_subset.t());

        // Compute alignment
        let k_exact_frobenius = k_exact.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let k_approx_frobenius = k_approx.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let k_product = (&k_exact * &k_approx).sum();

        let alignment = k_product / (k_exact_frobenius * k_approx_frobenius);
        Ok(alignment)
    }

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
        Ok(effective_rank)
    }

    fn compute_frobenius_approximation_quality(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
    ) -> Result<f64> {
        // This is a simplified version - in practice, you'd want to compute
        // the actual kernel approximation error
        let reconstruction_error = self.compute_reconstruction_error(x, x_transformed)?;
        Ok(1.0 / (1.0 + reconstruction_error)) // Convert error to quality score
    }

    fn compute_relative_frobenius_quality(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
    ) -> Result<f64> {
        let reconstruction_error = self.compute_reconstruction_error(x, x_transformed)?;
        let original_norm = x.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let relative_error = reconstruction_error / original_norm;
        Ok(1.0 / (1.0 + relative_error))
    }

    fn compute_cross_validation_score(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified cross-validation score based on feature stability
        let n_samples = x.nrows();
        let fold_size = n_samples / 5; // 5-fold CV
        let mut cv_scores = Vec::new();

        for fold in 0..5 {
            let start = fold * fold_size;
            let end = if fold == 4 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            let val_features = x_transformed.slice(s![start..end, ..]);
            let train_features = if start == 0 {
                x_transformed.slice(s![end.., ..]).to_owned()
            } else if end == n_samples {
                x_transformed.slice(s![..start, ..]).to_owned()
            } else {
                let part1 = x_transformed.slice(s![..start, ..]);
                let part2 = x_transformed.slice(s![end.., ..]);
                concatenate![Axis(0), part1, part2]
            };

            // Compute similarity between validation and training features
            let train_mean = train_features.mean_axis(Axis(0)).unwrap();
            let val_mean = val_features.mean_axis(Axis(0)).unwrap();
            let diff = &train_mean - &val_mean;
            let similarity = 1.0 / (1.0 + diff.dot(&diff).sqrt());

            cv_scores.push(similarity);
        }

        Ok(cv_scores.iter().sum::<f64>() / cv_scores.len() as f64)
    }

    fn compute_reconstruction_error(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
    ) -> Result<f64> {
        // Simplified reconstruction error - assumes we can approximate original features
        // This is a placeholder for more sophisticated error computation
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let x_transformed_projected =
            x_transformed.sum_axis(Axis(1)) / x_transformed.ncols() as f64;

        let mut error = 0.0;
        for (i, &projected_val) in x_transformed_projected.iter().enumerate() {
            let original_norm = x.row(i).dot(&x_mean);
            error += (projected_val - original_norm).powi(2);
        }

        Ok(error.sqrt())
    }

    fn compute_approximation_error(
        &self,
        x_train: &Array2<f64>,
        x_val: &Array2<f64>,
        x_train_transformed: &Array2<f64>,
        x_val_transformed: &Array2<f64>,
    ) -> Result<f64> {
        // Compute approximation error based on kernel matrix approximation
        // This is a simplified version that computes the Frobenius norm of the error

        // For simplicity, compute the validation error as the norm difference
        let val_norm = x_val.norm_l2();
        let transformed_norm = x_val_transformed.norm_l2();

        // Relative error
        let error = if val_norm > 1e-12 {
            (val_norm - transformed_norm).abs() / val_norm
        } else {
            (val_norm - transformed_norm).abs()
        };

        Ok(error)
    }

    fn select_optimal_dimension(
        &self,
        dimensions: &[usize],
        quality_scores: &HashMap<usize, f64>,
        approximation_errors: &HashMap<usize, f64>,
    ) -> Result<usize> {
        match &self.config.selection_strategy {
            SelectionStrategy::ErrorTolerance { tolerance } => {
                // Find first dimension that meets error tolerance
                for &dim in dimensions {
                    if let Some(&error) = approximation_errors.get(&dim) {
                        if error <= *tolerance {
                            return Ok(dim);
                        }
                    }
                }
                // If no dimension meets tolerance, return best performing
                self.select_best_quality_dimension(dimensions, quality_scores)
            }
            SelectionStrategy::QualityEfficiency {
                efficiency_threshold,
            } => {
                // Find dimension with best quality/dimension ratio above threshold
                let mut best_efficiency = 0.0;
                let mut best_dim = dimensions[0];

                for &dim in dimensions {
                    if let Some(&quality) = quality_scores.get(&dim) {
                        let efficiency = quality / dim as f64;
                        if efficiency >= *efficiency_threshold && efficiency > best_efficiency {
                            best_efficiency = efficiency;
                            best_dim = dim;
                        }
                    }
                }
                Ok(best_dim)
            }
            SelectionStrategy::ElbowMethod { sensitivity } => {
                self.select_elbow_dimension(dimensions, quality_scores, *sensitivity)
            }
            _ => {
                // Default to best quality
                self.select_best_quality_dimension(dimensions, quality_scores)
            }
        }
    }

    fn select_best_quality_dimension(
        &self,
        dimensions: &[usize],
        quality_scores: &HashMap<usize, f64>,
    ) -> Result<usize> {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_dim = dimensions[0];

        for &dim in dimensions {
            if let Some(&score) = quality_scores.get(&dim) {
                if score > best_score {
                    best_score = score;
                    best_dim = dim;
                }
            }
        }

        Ok(best_dim)
    }

    fn select_elbow_dimension(
        &self,
        dimensions: &[usize],
        quality_scores: &HashMap<usize, f64>,
        sensitivity: f64,
    ) -> Result<usize> {
        if dimensions.len() < 3 {
            return Ok(dimensions[0]);
        }

        // Convert to sorted vectors
        let mut dim_score_pairs: Vec<_> = dimensions
            .iter()
            .filter_map(|&dim| quality_scores.get(&dim).map(|&score| (dim, score)))
            .collect();
        dim_score_pairs.sort_by(|a, b| a.0.cmp(&b.0));

        // Find elbow using second derivative
        let mut best_elbow_idx = 1;
        let mut max_curvature = 0.0;

        for i in 1..(dim_score_pairs.len() - 1) {
            let (d1, s1) = dim_score_pairs[i - 1];
            let (d2, s2) = dim_score_pairs[i];
            let (d3, s3) = dim_score_pairs[i + 1];

            // Compute second derivative (curvature)
            let first_deriv1 = (s2 - s1) / (d2 - d1) as f64;
            let first_deriv2 = (s3 - s2) / (d3 - d2) as f64;
            let second_deriv = (first_deriv2 - first_deriv1) / ((d3 - d1) as f64 / 2.0);

            let curvature = second_deriv.abs();
            if curvature > max_curvature && curvature > sensitivity {
                max_curvature = curvature;
                best_elbow_idx = i;
            }
        }

        Ok(dim_score_pairs[best_elbow_idx].0)
    }
}

/// Fitted adaptive RBF sampler
pub struct FittedAdaptiveRBFSampler {
    fitted_rbf: crate::rbf_sampler::RBFSampler<sklears_core::traits::Trained>,
    selection_result: DimensionSelectionResult,
}

impl Fit<Array2<f64>, ()> for AdaptiveRBFSampler {
    type Fitted = FittedAdaptiveRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        // Perform dimension selection
        let selection_result = self.select_dimension(x)?;

        // Fit RBF sampler with optimal dimension
        let rbf_sampler = RBFSampler::new(selection_result.optimal_dimension).gamma(self.gamma);

        let fitted_rbf = rbf_sampler.fit(x, &())?;

        Ok(FittedAdaptiveRBFSampler {
            fitted_rbf,
            selection_result,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedAdaptiveRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        self.fitted_rbf.transform(x)
    }
}

impl FittedAdaptiveRBFSampler {
    /// Get the dimension selection result
    pub fn selection_result(&self) -> &DimensionSelectionResult {
        &self.selection_result
    }

    /// Get the optimal dimension
    pub fn optimal_dimension(&self) -> usize {
        self.selection_result.optimal_dimension
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_adaptive_rbf_sampler_basic() {
        let x =
            Array2::from_shape_vec((50, 5), (0..250).map(|i| i as f64 * 0.1).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 10,
            max_features: 50,
            step_size: 10,
            n_trials: 2,
            ..Default::default()
        };

        let sampler = AdaptiveRBFSampler::new().gamma(0.5).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.nrows(), 50);
        assert!(transformed.ncols() >= 10);
        assert!(transformed.ncols() <= 50);

        // Check that dimension is reasonable
        let optimal_dim = fitted.optimal_dimension();
        assert!(optimal_dim >= 10);
        assert!(optimal_dim <= 50);
    }

    #[test]
    fn test_dimension_selection_error_tolerance() {
        let x =
            Array2::from_shape_vec((40, 4), (0..160).map(|i| i as f64 * 0.05).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 5,
            max_features: 25,
            step_size: 5,
            selection_strategy: SelectionStrategy::ErrorTolerance { tolerance: 0.2 },
            n_trials: 1,
            validation_fraction: 0.3,
            ..Default::default()
        };

        let sampler = AdaptiveRBFSampler::new().gamma(1.0).config(config);

        let result = sampler.select_dimension(&x).unwrap();

        assert!(result.optimal_dimension >= 5);
        assert!(result.optimal_dimension <= 25);
        assert!(!result.quality_scores.is_empty());
        assert!(!result.approximation_errors.is_empty());
    }

    #[test]
    fn test_dimension_selection_quality_efficiency() {
        let x = Array2::from_shape_vec((30, 3), (0..90).map(|i| i as f64 * 0.1).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 5,
            max_features: 20,
            step_size: 5,
            selection_strategy: SelectionStrategy::QualityEfficiency {
                efficiency_threshold: 0.01,
            },
            quality_metric: QualityMetric::EffectiveRank,
            n_trials: 1,
            ..Default::default()
        };

        let sampler = AdaptiveRBFSampler::new().gamma(0.8).config(config);

        let result = sampler.select_dimension(&x).unwrap();

        assert!(result.optimal_dimension >= 5);
        assert!(result.optimal_dimension <= 20);

        // Check that quality scores are computed
        for &dim in &[5, 10, 15, 20] {
            assert!(result.quality_scores.contains_key(&dim));
        }
    }

    #[test]
    fn test_dimension_selection_elbow_method() {
        let x =
            Array2::from_shape_vec((60, 6), (0..360).map(|i| i as f64 * 0.02).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 10,
            max_features: 40,
            step_size: 10,
            selection_strategy: SelectionStrategy::ElbowMethod { sensitivity: 0.01 },
            quality_metric: QualityMetric::KernelAlignment,
            n_trials: 1,
            ..Default::default()
        };

        let sampler = AdaptiveRBFSampler::new().gamma(0.3).config(config);

        let result = sampler.select_dimension(&x).unwrap();

        assert!(result.optimal_dimension >= 10);
        assert!(result.optimal_dimension <= 40);

        // Verify that computation times are recorded
        assert!(!result.computation_times.is_empty());
        assert!(!result.memory_usage.is_empty());
    }

    #[test]
    fn test_quality_metrics() {
        let x =
            Array2::from_shape_vec((25, 4), (0..100).map(|i| i as f64 * 0.1).collect()).unwrap();

        let sampler = AdaptiveRBFSampler::new().gamma(1.0);

        // Test kernel alignment
        let rbf = RBFSampler::new(15).gamma(1.0);
        let fitted_rbf = rbf.fit(&x, &()).unwrap();
        let x_transformed = fitted_rbf.transform(&x).unwrap();

        let alignment = sampler
            .compute_kernel_alignment(&x, &x_transformed)
            .unwrap();
        assert!(alignment >= 0.0);
        assert!(alignment <= 1.0);

        // Test effective rank
        let eff_rank = sampler.compute_effective_rank(&x_transformed).unwrap();
        assert!(eff_rank > 0.0);
        assert!(eff_rank <= x_transformed.ncols() as f64);

        // Test reconstruction error
        let recon_error = sampler
            .compute_reconstruction_error(&x, &x_transformed)
            .unwrap();
        assert!(recon_error >= 0.0);
    }

    #[test]
    fn test_adaptive_sampler_reproducibility() {
        let x =
            Array2::from_shape_vec((40, 5), (0..200).map(|i| i as f64 * 0.08).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 10,
            max_features: 30,
            step_size: 10,
            n_trials: 2,
            random_seed: Some(42),
            ..Default::default()
        };

        let sampler1 = AdaptiveRBFSampler::new().gamma(0.5).config(config.clone());

        let sampler2 = AdaptiveRBFSampler::new().gamma(0.5).config(config);

        let result1 = sampler1.select_dimension(&x).unwrap();
        let result2 = sampler2.select_dimension(&x).unwrap();

        assert_eq!(result1.optimal_dimension, result2.optimal_dimension);

        // Quality scores should be similar (allowing for small numerical differences)
        for (&dim, &score1) in &result1.quality_scores {
            if let Some(&score2) = result2.quality_scores.get(&dim) {
                assert_abs_diff_eq!(score1, score2, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_dimension_selection_result() {
        let x =
            Array2::from_shape_vec((35, 3), (0..105).map(|i| i as f64 * 0.1).collect()).unwrap();

        let config = AdaptiveDimensionConfig {
            min_features: 5,
            max_features: 15,
            step_size: 5,
            n_trials: 1,
            ..Default::default()
        };

        let sampler = AdaptiveRBFSampler::new().gamma(0.7).config(config);

        let fitted = sampler.fit(&x, &()).unwrap();
        let result = fitted.selection_result();

        // Verify all required fields are present
        assert!(result.optimal_dimension >= 5);
        assert!(result.optimal_dimension <= 15);
        assert_eq!(result.quality_scores.len(), 3); // 5, 10, 15
        assert_eq!(result.approximation_errors.len(), 3);
        assert_eq!(result.computation_times.len(), 3);
        assert_eq!(result.memory_usage.len(), 3);

        // Verify memory usage is reasonable
        for (&dim, &memory) in &result.memory_usage {
            assert_eq!(memory, dim * 3 * 8); // features * dimensions * sizeof(f64)
        }
    }
}

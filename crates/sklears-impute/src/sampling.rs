//! Sampling-based imputation methods
//!
//! This module provides imputation methods based on various sampling techniques
//! including importance sampling, stratified sampling, and adaptive sampling.

// ✅ SciRS2 Policy compliant imports
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random, Rng};
// use scirs2_core::parallel::{ParallelExecutor, ChunkStrategy}; // Note: not available

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Configuration for sampling-based imputation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    /// Number of samples to draw
    pub n_samples: usize,
    /// Sampling strategy to use
    pub strategy: SamplingStrategy,
    /// Use importance sampling
    pub importance_sampling: bool,
    /// Weight function for importance sampling
    pub weight_function: WeightFunction,
    /// Stratification variables for stratified sampling
    pub stratify_by: Option<Vec<usize>>,
    /// Number of strata for stratified sampling
    pub n_strata: usize,
    /// Use quasi-random sequences (low-discrepancy)
    pub use_quasi_random: bool,
    /// Sequence type for quasi-random sampling
    pub quasi_sequence_type: QuasiSequenceType,
    /// Enable adaptive sampling
    pub adaptive_sampling: bool,
    /// Target confidence level for adaptive sampling
    pub confidence_level: f64,
    /// Maximum sampling iterations
    pub max_iterations: usize,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            strategy: SamplingStrategy::Simple,
            importance_sampling: false,
            weight_function: WeightFunction::Uniform,
            stratify_by: None,
            n_strata: 5,
            use_quasi_random: false,
            quasi_sequence_type: QuasiSequenceType::Halton,
            adaptive_sampling: false,
            confidence_level: 0.95,
            max_iterations: 100,
        }
    }
}

/// Sampling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SamplingStrategy {
    /// Simple random sampling
    Simple,
    /// Stratified sampling
    Stratified,
    /// Cluster sampling
    Cluster,
    /// Systematic sampling
    Systematic,
    /// Importance sampling
    Importance,
    /// Latin hypercube sampling
    LatinHypercube,
    /// Bootstrap sampling
    Bootstrap,
    /// Reservoir sampling
    Reservoir,
}

/// Weight functions for importance sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightFunction {
    /// Uniform weights (equivalent to simple random sampling)
    Uniform,
    /// Inverse probability weighting
    InverseProbability,
    /// Density-based weighting
    DensityBased,
    /// Distance-based weighting
    DistanceBased,
    /// Custom weights provided by user
    Custom(Vec<f64>),
}

/// Quasi-random sequence types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuasiSequenceType {
    /// Halton sequence
    Halton,
    /// Sobol sequence
    Sobol,
    /// Faure sequence
    Faure,
    /// Niederreiter sequence
    Niederreiter,
}

/// Sampling-based Simple Imputer
#[derive(Debug)]
pub struct SamplingSimpleImputer<S = Untrained> {
    state: S,
    strategy: String,
    missing_values: f64,
    config: SamplingConfig,
}

/// Trained state for sampling-based simple imputer
#[derive(Debug)]
pub struct SamplingSimpleImputerTrained {
    sample_statistics_: Array1<f64>,
    sample_distributions_: Vec<SampleDistribution>,
    n_features_in_: usize,
    config: SamplingConfig,
}

/// Sample distribution for a feature
#[derive(Debug, Clone)]
pub struct SampleDistribution {
    /// values
    pub values: Vec<f64>,
    /// weights
    pub weights: Vec<f64>,
    /// cumulative_weights
    pub cumulative_weights: Vec<f64>,
    /// distribution_type
    pub distribution_type: DistributionType,
}

/// Distribution types for sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    /// Empirical distribution (discrete)
    Empirical,
    /// Kernel density estimate (continuous)
    KernelDensity,
    /// Parametric distribution
    Parametric(ParametricDistribution),
}

/// Parametric distribution types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParametricDistribution {
    /// Normal
    Normal { mean: f64, std: f64 },
    /// LogNormal
    LogNormal { mean_log: f64, std_log: f64 },
    /// Exponential
    Exponential { rate: f64 },
    /// Gamma
    Gamma { shape: f64, rate: f64 },
    /// Beta
    Beta { alpha: f64, beta: f64 },
    /// Uniform
    Uniform { low: f64, high: f64 },
}

/// Stratified Sampling Imputer
#[derive(Debug)]
pub struct StratifiedSamplingImputer<S = Untrained> {
    state: S,
    missing_values: f64,
    config: SamplingConfig,
    stratification_features: Vec<usize>,
}

/// Trained state for stratified sampling imputer
#[derive(Debug)]
pub struct StratifiedSamplingImputerTrained {
    strata_distributions_: HashMap<Vec<usize>, HashMap<usize, SampleDistribution>>,
    feature_strata_: Vec<Array1<f64>>, // Stratification boundaries for each feature
    n_features_in_: usize,
    config: SamplingConfig,
}

/// Importance Sampling Imputer
#[derive(Debug)]
pub struct ImportanceSamplingImputer<S = Untrained> {
    state: S,
    missing_values: f64,
    config: SamplingConfig,
    proposal_distribution: ProposalDistribution,
}

/// Trained state for importance sampling imputer
#[derive(Debug)]
pub struct ImportanceSamplingImputerTrained {
    importance_weights_: Array2<f64>, // [feature, sample]
    proposal_samples_: Array2<f64>,
    target_density_: Array1<f64>,
    n_features_in_: usize,
    config: SamplingConfig,
}

/// Proposal distributions for importance sampling
#[derive(Debug, Clone)]
pub enum ProposalDistribution {
    /// Use empirical distribution as proposal
    Empirical,
    /// Use Gaussian mixture model
    GaussianMixture { n_components: usize },
    /// Use kernel density estimate
    KernelDensity { bandwidth: f64 },
}

/// Adaptive Sampling Imputer
#[derive(Debug)]
pub struct AdaptiveSamplingImputer<S = Untrained> {
    state: S,
    missing_values: f64,
    config: SamplingConfig,
    convergence_threshold: f64,
}

/// Trained state for adaptive sampling imputer
#[derive(Debug)]
pub struct AdaptiveSamplingImputerTrained {
    adaptive_samples_: Vec<Array1<f64>>, // Samples for each feature
    convergence_history_: Vec<f64>,
    final_estimates_: Array1<f64>,
    confidence_intervals_: Array2<f64>,
    n_features_in_: usize,
    config: SamplingConfig,
}

impl SamplingSimpleImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            strategy: "mean".to_string(),
            missing_values: f64::NAN,
            config: SamplingConfig::default(),
        }
    }

    pub fn strategy(mut self, strategy: String) -> Self {
        self.strategy = strategy;
        self
    }

    pub fn sampling_config(mut self, config: SamplingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.config.n_samples = n_samples;
        self
    }

    pub fn sampling_strategy(mut self, strategy: SamplingStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    pub fn weight_function(mut self, weight_function: WeightFunction) -> Self {
        self.config.weight_function = weight_function;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for SamplingSimpleImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SamplingSimpleImputer<Untrained> {
    type Config = SamplingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SamplingSimpleImputer<Untrained> {
    type Fitted = SamplingSimpleImputer<SamplingSimpleImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        let (sample_statistics, sample_distributions) = self.compute_sample_statistics(&X)?;

        Ok(SamplingSimpleImputer {
            state: SamplingSimpleImputerTrained {
                sample_statistics_: sample_statistics,
                sample_distributions_: sample_distributions,
                n_features_in_: n_features,
                config: self.config,
            },
            strategy: self.strategy,
            missing_values: self.missing_values,
            config: Default::default(),
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for SamplingSimpleImputer<SamplingSimpleImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        // Apply sampling-based imputation
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    let imputed_value = self.sample_imputed_value(j)?;
                    X_imputed[[i, j]] = imputed_value;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl SamplingSimpleImputer<Untrained> {
    /// Compute sample statistics based on sampling strategy
    fn compute_sample_statistics(
        &self,
        X: &Array2<f64>,
    ) -> Result<(Array1<f64>, Vec<SampleDistribution>), SklearsError> {
        let (_, n_features) = X.dim();
        let mut sample_statistics = Array1::<f64>::zeros(n_features);
        let mut sample_distributions = Vec::new();

        for j in 0..n_features {
            let column = X.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if valid_values.is_empty() {
                sample_statistics[j] = 0.0;
                sample_distributions.push(SampleDistribution {
                    values: vec![0.0],
                    weights: vec![1.0],
                    cumulative_weights: vec![1.0],
                    distribution_type: DistributionType::Empirical,
                });
                continue;
            }

            // Create sample distribution based on strategy
            let distribution = match self.config.strategy {
                SamplingStrategy::Simple => {
                    self.create_simple_sample_distribution(&valid_values)?
                }
                SamplingStrategy::Importance => {
                    self.create_importance_sample_distribution(&valid_values)?
                }
                SamplingStrategy::Bootstrap => {
                    self.create_bootstrap_sample_distribution(&valid_values)?
                }
                SamplingStrategy::LatinHypercube => {
                    self.create_latin_hypercube_distribution(&valid_values)?
                }
                _ => self.create_simple_sample_distribution(&valid_values)?,
            };

            // Compute primary statistic
            sample_statistics[j] = match self.strategy.as_str() {
                "mean" => self.compute_weighted_mean(&distribution),
                "median" => self.compute_weighted_median(&distribution),
                "mode" => self.compute_weighted_mode(&distribution),
                _ => self.compute_weighted_mean(&distribution),
            };

            sample_distributions.push(distribution);
        }

        Ok((sample_statistics, sample_distributions))
    }

    /// Create simple sample distribution
    fn create_simple_sample_distribution(
        &self,
        values: &[f64],
    ) -> Result<SampleDistribution, SklearsError> {
        let n_samples = self.config.n_samples.min(values.len());
        let mut rng = Random::default();

        // Simple random sampling
        let mut sampled_values = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..n_samples {
            let idx = rng.gen_range(0..values.len());
            sampled_values.push(values[idx]);
            weights.push(1.0 / n_samples as f64);
        }

        // Compute cumulative weights
        let mut cumulative_weights = Vec::new();
        let mut cumsum = 0.0;
        for &weight in &weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        Ok(SampleDistribution {
            values: sampled_values,
            weights,
            cumulative_weights,
            distribution_type: DistributionType::Empirical,
        })
    }

    /// Create importance sample distribution
    fn create_importance_sample_distribution(
        &self,
        values: &[f64],
    ) -> Result<SampleDistribution, SklearsError> {
        let n_samples = self.config.n_samples.min(values.len());

        // Compute importance weights based on density
        let mut weighted_values = Vec::new();
        let mut importance_weights = Vec::new();

        // Use kernel density estimation for importance weights
        let bandwidth = self.compute_bandwidth(values);

        for &value in values.iter().take(n_samples) {
            let density = self.kernel_density_estimate(value, values, bandwidth);
            let importance_weight = 1.0 / (density + 1e-8); // Avoid division by zero

            weighted_values.push(value);
            importance_weights.push(importance_weight);
        }

        // Normalize weights
        let total_weight: f64 = importance_weights.iter().sum();
        for weight in &mut importance_weights {
            *weight /= total_weight;
        }

        // Compute cumulative weights
        let mut cumulative_weights = Vec::new();
        let mut cumsum = 0.0;
        for &weight in &importance_weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        Ok(SampleDistribution {
            values: weighted_values,
            weights: importance_weights,
            cumulative_weights,
            distribution_type: DistributionType::KernelDensity,
        })
    }

    /// Create bootstrap sample distribution
    fn create_bootstrap_sample_distribution(
        &self,
        values: &[f64],
    ) -> Result<SampleDistribution, SklearsError> {
        let n_bootstrap = 100;
        let n_samples_per_bootstrap = values.len();
        let mut bootstrap_estimates = Vec::new();

        let mut rng = Random::default();

        for _ in 0..n_bootstrap {
            let mut bootstrap_sample = Vec::new();

            // Sample with replacement
            for _ in 0..n_samples_per_bootstrap {
                let idx = rng.gen_range(0..values.len());
                bootstrap_sample.push(values[idx]);
            }

            // Compute statistic for this bootstrap sample
            let estimate = match self.strategy.as_str() {
                "mean" => bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64,
                "median" => {
                    let mut sorted = bootstrap_sample.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len() % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                _ => bootstrap_sample.iter().sum::<f64>() / bootstrap_sample.len() as f64,
            };

            bootstrap_estimates.push(estimate);
        }

        let uniform_weight = 1.0 / bootstrap_estimates.len() as f64;
        let weights = vec![uniform_weight; bootstrap_estimates.len()];

        // Compute cumulative weights
        let mut cumulative_weights = Vec::new();
        let mut cumsum = 0.0;
        for &weight in &weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        Ok(SampleDistribution {
            values: bootstrap_estimates,
            weights,
            cumulative_weights,
            distribution_type: DistributionType::Empirical,
        })
    }

    /// Create Latin hypercube sample distribution
    fn create_latin_hypercube_distribution(
        &self,
        values: &[f64],
    ) -> Result<SampleDistribution, SklearsError> {
        let n_samples = self.config.n_samples.min(values.len());

        // Sort values to create quantile-based sampling
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut lhs_values = Vec::new();
        let mut rng = Random::default();

        // Generate Latin hypercube samples
        for i in 0..n_samples {
            let lower_bound = i as f64 / n_samples as f64;
            let upper_bound = (i + 1) as f64 / n_samples as f64;
            let uniform_sample = rng.gen::<f64>();
            let stratified_sample = lower_bound + uniform_sample * (upper_bound - lower_bound);

            // Map to quantile
            let quantile_idx = (stratified_sample * (sorted_values.len() - 1) as f64) as usize;
            let quantile_idx = quantile_idx.min(sorted_values.len() - 1);

            lhs_values.push(sorted_values[quantile_idx]);
        }

        let uniform_weight = 1.0 / lhs_values.len() as f64;
        let weights = vec![uniform_weight; lhs_values.len()];

        // Compute cumulative weights
        let mut cumulative_weights = Vec::new();
        let mut cumsum = 0.0;
        for &weight in &weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        Ok(SampleDistribution {
            values: lhs_values,
            weights,
            cumulative_weights,
            distribution_type: DistributionType::Empirical,
        })
    }

    /// Compute bandwidth for kernel density estimation
    fn compute_bandwidth(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 1.0;
        }

        // Silverman's rule of thumb
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        std_dev * (4.0 / (3.0 * n)).powf(0.2)
    }

    /// Kernel density estimate at a point
    fn kernel_density_estimate(&self, x: f64, values: &[f64], bandwidth: f64) -> f64 {
        let n = values.len() as f64;
        let sum: f64 = values
            .iter()
            .map(|&xi| {
                let u = (x - xi) / bandwidth;
                (-0.5 * u * u).exp() // Gaussian kernel
            })
            .sum();

        sum / (n * bandwidth * (2.0 * std::f64::consts::PI).sqrt())
    }

    /// Compute weighted mean
    fn compute_weighted_mean(&self, distribution: &SampleDistribution) -> f64 {
        distribution
            .values
            .iter()
            .zip(distribution.weights.iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    }

    /// Compute weighted median
    fn compute_weighted_median(&self, distribution: &SampleDistribution) -> f64 {
        // Find the value where cumulative weight crosses 0.5
        for (i, &cum_weight) in distribution.cumulative_weights.iter().enumerate() {
            if cum_weight >= 0.5 {
                return distribution.values[i];
            }
        }

        // Fallback to last value
        distribution.values.last().copied().unwrap_or(0.0)
    }

    /// Compute weighted mode (most frequent value)
    fn compute_weighted_mode(&self, distribution: &SampleDistribution) -> f64 {
        let mut value_weights = HashMap::new();

        for (&value, &weight) in distribution.values.iter().zip(distribution.weights.iter()) {
            let key = (value * 1e6) as i64; // Handle floating point precision
            *value_weights.entry(key).or_insert(0.0) += weight;
        }

        value_weights
            .into_iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(key, _)| key as f64 / 1e6)
            .unwrap_or(0.0)
    }
}

impl SamplingSimpleImputer<SamplingSimpleImputerTrained> {
    /// Sample an imputed value from the learned distribution
    fn sample_imputed_value(&self, feature_idx: usize) -> Result<f64, SklearsError> {
        let distribution = &self.state.sample_distributions_[feature_idx];
        let mut rng = Random::default();
        let random_value = rng.gen::<f64>();

        // Find the corresponding value using cumulative weights
        for (i, &cum_weight) in distribution.cumulative_weights.iter().enumerate() {
            if random_value <= cum_weight {
                return Ok(distribution.values[i]);
            }
        }

        // Fallback to last value
        Ok(distribution.values.last().copied().unwrap_or(0.0))
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Get the sample distribution for a feature
    pub fn distribution(&self, feature_idx: usize) -> Option<&SampleDistribution> {
        self.state.sample_distributions_.get(feature_idx)
    }

    /// Get sample statistics
    pub fn statistics(&self) -> &Array1<f64> {
        &self.state.sample_statistics_
    }
}

// Implement Stratified Sampling Imputer
impl StratifiedSamplingImputer<Untrained> {
    pub fn new() -> Self {
        Self {
            state: Untrained,
            missing_values: f64::NAN,
            config: SamplingConfig::default(),
            stratification_features: Vec::new(),
        }
    }

    pub fn sampling_config(mut self, config: SamplingConfig) -> Self {
        self.config = config;
        self
    }

    pub fn stratify_by(mut self, features: Vec<usize>) -> Self {
        self.stratification_features = features.clone();
        self.config.stratify_by = Some(features);
        self
    }

    pub fn n_strata(mut self, n_strata: usize) -> Self {
        self.config.n_strata = n_strata;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for StratifiedSamplingImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for StratifiedSamplingImputer<Untrained> {
    type Config = SamplingConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for StratifiedSamplingImputer<Untrained> {
    type Fitted = StratifiedSamplingImputer<StratifiedSamplingImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if self.stratification_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Stratification features must be specified".to_string(),
            ));
        }

        let (strata_distributions, feature_strata) = self.compute_stratified_distributions(&X)?;

        Ok(StratifiedSamplingImputer {
            state: StratifiedSamplingImputerTrained {
                strata_distributions_: strata_distributions,
                feature_strata_: feature_strata,
                n_features_in_: n_features,
                config: self.config,
            },
            missing_values: self.missing_values,
            config: Default::default(),
            stratification_features: Vec::new(),
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for StratifiedSamplingImputer<StratifiedSamplingImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        for i in 0..n_samples {
            // Determine stratum for this sample
            let stratum_key = self.determine_stratum(&X_imputed.row(i).to_owned())?;

            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    if let Some(stratum_dists) = self.state.strata_distributions_.get(&stratum_key)
                    {
                        if let Some(distribution) = stratum_dists.get(&j) {
                            let imputed_value = self.sample_from_distribution(distribution)?;
                            X_imputed[[i, j]] = imputed_value;
                        }
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl StratifiedSamplingImputer<Untrained> {
    /// Compute distributions for each stratum
    fn compute_stratified_distributions(
        &self,
        X: &Array2<f64>,
    ) -> Result<
        (
            HashMap<Vec<usize>, HashMap<usize, SampleDistribution>>,
            Vec<Array1<f64>>,
        ),
        SklearsError,
    > {
        let (n_samples, n_features) = X.dim();

        // Compute strata boundaries for stratification features
        let mut feature_strata = Vec::new();
        for &feature_idx in &self.stratification_features {
            let column = X.column(feature_idx);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if valid_values.is_empty() {
                feature_strata.push(Array1::from_vec(vec![0.0, 1.0]));
                continue;
            }

            // Create quantile-based strata boundaries
            let mut sorted_values = valid_values;
            sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut boundaries = Vec::new();
            for i in 0..=self.config.n_strata {
                let quantile = i as f64 / self.config.n_strata as f64;
                let idx = ((sorted_values.len() - 1) as f64 * quantile) as usize;
                let idx = idx.min(sorted_values.len() - 1);
                boundaries.push(sorted_values[idx]);
            }

            feature_strata.push(Array1::from_vec(boundaries));
        }

        // Assign samples to strata and compute distributions
        let mut strata_samples: HashMap<Vec<usize>, Vec<Array1<f64>>> = HashMap::new();

        for i in 0..n_samples {
            let row = X.row(i).to_owned();
            let stratum_key = self.assign_to_stratum(&row, &feature_strata)?;

            strata_samples
                .entry(stratum_key)
                .or_insert_with(Vec::new)
                .push(row);
        }

        // Compute distributions for each stratum and feature
        let mut strata_distributions = HashMap::new();

        for (stratum_key, samples) in strata_samples {
            let mut feature_distributions = HashMap::new();

            for j in 0..n_features {
                let feature_values: Vec<f64> = samples
                    .iter()
                    .map(|row| row[j])
                    .filter(|&x| !self.is_missing(x))
                    .collect();

                if !feature_values.is_empty() {
                    let distribution = self.create_empirical_distribution(&feature_values)?;
                    feature_distributions.insert(j, distribution);
                }
            }

            strata_distributions.insert(stratum_key, feature_distributions);
        }

        Ok((strata_distributions, feature_strata))
    }

    /// Assign a sample to a stratum
    fn assign_to_stratum(
        &self,
        row: &Array1<f64>,
        feature_strata: &[Array1<f64>],
    ) -> Result<Vec<usize>, SklearsError> {
        let mut stratum_key = Vec::new();

        for (i, &feature_idx) in self.stratification_features.iter().enumerate() {
            let value = row[feature_idx];
            if self.is_missing(value) {
                stratum_key.push(0); // Default stratum for missing values
                continue;
            }

            let boundaries = &feature_strata[i];
            let mut stratum = 0;

            for k in 1..boundaries.len() {
                if value <= boundaries[k] {
                    stratum = k - 1;
                    break;
                }
            }

            stratum_key.push(stratum);
        }

        Ok(stratum_key)
    }

    /// Create empirical distribution from values
    fn create_empirical_distribution(
        &self,
        values: &[f64],
    ) -> Result<SampleDistribution, SklearsError> {
        let uniform_weight = 1.0 / values.len() as f64;
        let weights = vec![uniform_weight; values.len()];

        // Compute cumulative weights
        let mut cumulative_weights = Vec::new();
        let mut cumsum = 0.0;
        for &weight in &weights {
            cumsum += weight;
            cumulative_weights.push(cumsum);
        }

        Ok(SampleDistribution {
            values: values.to_vec(),
            weights,
            cumulative_weights,
            distribution_type: DistributionType::Empirical,
        })
    }
}

impl StratifiedSamplingImputer<StratifiedSamplingImputerTrained> {
    /// Determine stratum for a sample
    fn determine_stratum(&self, row: &Array1<f64>) -> Result<Vec<usize>, SklearsError> {
        if let Some(ref stratify_features) = self.state.config.stratify_by {
            let mut stratum_key = Vec::new();

            for (i, &feature_idx) in stratify_features.iter().enumerate() {
                let value = row[feature_idx];
                if self.is_missing(value) {
                    stratum_key.push(0); // Default stratum for missing values
                    continue;
                }

                let boundaries = &self.state.feature_strata_[i];
                let mut stratum = 0;

                for k in 1..boundaries.len() {
                    if value <= boundaries[k] {
                        stratum = k - 1;
                        break;
                    }
                }

                stratum_key.push(stratum);
            }

            Ok(stratum_key)
        } else {
            Ok(vec![0]) // Default stratum
        }
    }

    /// Sample from a distribution
    fn sample_from_distribution(
        &self,
        distribution: &SampleDistribution,
    ) -> Result<f64, SklearsError> {
        let mut rng = Random::default();
        let random_value = rng.gen::<f64>();

        // Find the corresponding value using cumulative weights
        for (i, &cum_weight) in distribution.cumulative_weights.iter().enumerate() {
            if random_value <= cum_weight {
                return Ok(distribution.values[i]);
            }
        }

        // Fallback to last value
        Ok(distribution.values.last().copied().unwrap_or(0.0))
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_sampling_simple_imputer() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = SamplingSimpleImputer::new()
            .strategy("mean".to_string())
            .n_samples(100)
            .sampling_strategy(SamplingStrategy::Simple);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        // Check that NaN was replaced
        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(X_imputed[[2, 2]], 9.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bootstrap_sampling() {
        let X = array![[1.0, 2.0], [3.0, f64::NAN], [5.0, 6.0], [7.0, 8.0]];

        let imputer = SamplingSimpleImputer::new()
            .strategy("mean".to_string())
            .sampling_strategy(SamplingStrategy::Bootstrap);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stratified_sampling_imputer() {
        let X = array![
            [1.0, 2.0, 0.0],      // stratum 0
            [2.0, f64::NAN, 0.0], // stratum 0
            [8.0, 9.0, 1.0],      // stratum 1
            [9.0, 10.0, 1.0]      // stratum 1
        ];

        let imputer = StratifiedSamplingImputer::new()
            .stratify_by(vec![2]) // Stratify by the third column
            .n_strata(2);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_latin_hypercube_sampling() {
        let X = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];

        let imputer = SamplingSimpleImputer::new()
            .strategy("mean".to_string())
            .sampling_strategy(SamplingStrategy::LatinHypercube)
            .n_samples(3);

        let fitted = imputer.fit(&X.view(), &()).unwrap();
        let X_imputed = fitted.transform(&X.view()).unwrap();

        assert!(!X_imputed[[1, 1]].is_nan());
        assert_abs_diff_eq!(X_imputed[[0, 0]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_sample_distribution() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let weights = vec![0.1, 0.2, 0.4, 0.2, 0.1];
        let cumulative_weights = vec![0.1, 0.3, 0.7, 0.9, 1.0];

        let distribution = SampleDistribution {
            values,
            weights,
            cumulative_weights,
            distribution_type: DistributionType::Empirical,
        };

        assert_eq!(distribution.values.len(), 5);
        assert_eq!(distribution.weights.len(), 5);
        assert_eq!(distribution.cumulative_weights.len(), 5);
        assert!((distribution.cumulative_weights.last().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sampling_config() {
        let config = SamplingConfig {
            n_samples: 500,
            strategy: SamplingStrategy::Importance,
            importance_sampling: true,
            ..Default::default()
        };

        let imputer = SamplingSimpleImputer::new().sampling_config(config.clone());

        assert_eq!(imputer.config.n_samples, 500);
        assert!(matches!(
            imputer.config.strategy,
            SamplingStrategy::Importance
        ));
        assert!(imputer.config.importance_sampling);
    }
}

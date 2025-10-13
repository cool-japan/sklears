//! Multi-scale RBF kernel approximation methods
//!
//! This module implements multi-scale RBF kernel approximation that uses multiple bandwidth
//! parameters to capture patterns at different scales. This provides better approximation
//! quality for data with features at multiple scales.

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::Distribution;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Multi-scale bandwidth selection strategy
#[derive(Debug, Clone, Copy)]
/// BandwidthStrategy
pub enum BandwidthStrategy {
    /// Manual specification of gamma values
    Manual,
    /// Logarithmic spacing: gamma_i = gamma_min * (gamma_max/gamma_min)^(i/(n_scales-1))
    LogarithmicSpacing,
    /// Linear spacing: gamma_i = gamma_min + i * (gamma_max - gamma_min) / (n_scales - 1)
    LinearSpacing,
    /// Geometric progression: gamma_i = gamma_min * ratio^i
    GeometricProgression,
    /// Adaptive spacing based on data characteristics
    Adaptive,
}

/// Feature combination strategy for multi-scale features
#[derive(Debug, Clone, Copy)]
/// CombinationStrategy
pub enum CombinationStrategy {
    /// Concatenate features from all scales
    Concatenation,
    /// Weighted average of features across scales
    WeightedAverage,
    /// Maximum response across scales
    MaxPooling,
    /// Average pooling across scales
    AveragePooling,
    /// Attention-based combination
    Attention,
}

/// Multi-scale RBF sampler that generates random Fourier features at multiple scales
///
/// This sampler generates RBF kernel approximations using multiple bandwidth parameters
/// (gamma values) to capture patterns at different scales. Each scale captures different
/// frequency characteristics of the data, providing a more comprehensive representation.
///
/// # Mathematical Background
///
/// For multiple scales with bandwidths γ₁, γ₂, ..., γₖ, the multi-scale RBF kernel is:
/// K(x,y) = Σᵢ wᵢ * exp(-γᵢ||x-y||²)
///
/// The random Fourier features for each scale i are:
/// zᵢ(x) = √(2/nᵢ) * [cos(ωᵢⱼᵀx + bᵢⱼ), sin(ωᵢⱼᵀx + bᵢⱼ)]
/// where ωᵢⱼ ~ N(0, 2γᵢI) and bᵢⱼ ~ Uniform[0, 2π]
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::{MultiScaleRBFSampler, BandwidthStrategy, CombinationStrategy};
/// use sklears_core::traits::{Transform, Fit, Untrained}
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
///
/// let sampler = MultiScaleRBFSampler::new(100)
///     .n_scales(3)
///     .gamma_range(0.1, 10.0)
///     .bandwidth_strategy(BandwidthStrategy::LogarithmicSpacing)
///     .combination_strategy(CombinationStrategy::Concatenation);
///
/// let fitted = sampler.fit(&x, &()).unwrap();
/// let features = fitted.transform(&x).unwrap();
/// ```
#[derive(Debug, Clone)]
/// MultiScaleRBFSampler
pub struct MultiScaleRBFSampler<State = Untrained> {
    /// Number of components per scale
    pub n_components_per_scale: usize,
    /// Number of scales
    pub n_scales: usize,
    /// Minimum gamma value
    pub gamma_min: Float,
    /// Maximum gamma value
    pub gamma_max: Float,
    /// Manual gamma values (used when strategy is Manual)
    pub manual_gammas: Vec<Float>,
    /// Bandwidth selection strategy
    pub bandwidth_strategy: BandwidthStrategy,
    /// Feature combination strategy
    pub combination_strategy: CombinationStrategy,
    /// Scale weights for weighted combination
    pub scale_weights: Vec<Float>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,

    // Fitted attributes
    gammas_: Option<Vec<Float>>,
    random_weights_: Option<Vec<Array2<Float>>>,
    random_offsets_: Option<Vec<Array1<Float>>>,
    attention_weights_: Option<Array1<Float>>, // For attention-based combination

    // State marker
    _state: PhantomData<State>,
}

impl MultiScaleRBFSampler<Untrained> {
    /// Create a new multi-scale RBF sampler
    ///
    /// # Arguments
    /// * `n_components_per_scale` - Number of random features per scale
    pub fn new(n_components_per_scale: usize) -> Self {
        Self {
            n_components_per_scale,
            n_scales: 3,
            gamma_min: 0.1,
            gamma_max: 10.0,
            manual_gammas: vec![],
            bandwidth_strategy: BandwidthStrategy::LogarithmicSpacing,
            combination_strategy: CombinationStrategy::Concatenation,
            scale_weights: vec![],
            random_state: None,
            gammas_: None,
            random_weights_: None,
            random_offsets_: None,
            attention_weights_: None,
            _state: PhantomData,
        }
    }

    /// Set the number of scales
    pub fn n_scales(mut self, n_scales: usize) -> Self {
        self.n_scales = n_scales;
        self
    }

    /// Set the gamma range for automatic bandwidth selection
    pub fn gamma_range(mut self, gamma_min: Float, gamma_max: Float) -> Self {
        self.gamma_min = gamma_min;
        self.gamma_max = gamma_max;
        self
    }

    /// Set manual gamma values
    pub fn manual_gammas(mut self, gammas: Vec<Float>) -> Self {
        self.n_scales = gammas.len();
        self.manual_gammas = gammas;
        self.bandwidth_strategy = BandwidthStrategy::Manual;
        self
    }

    /// Set the bandwidth selection strategy
    pub fn bandwidth_strategy(mut self, strategy: BandwidthStrategy) -> Self {
        self.bandwidth_strategy = strategy;
        self
    }

    /// Set the feature combination strategy
    pub fn combination_strategy(mut self, strategy: CombinationStrategy) -> Self {
        self.combination_strategy = strategy;
        self
    }

    /// Set weights for different scales (used in weighted combination)
    pub fn scale_weights(mut self, weights: Vec<Float>) -> Self {
        self.scale_weights = weights;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Compute gamma values based on the selected strategy
    fn compute_gammas(&self, x: &Array2<Float>) -> Result<Vec<Float>> {
        match self.bandwidth_strategy {
            BandwidthStrategy::Manual => {
                if self.manual_gammas.is_empty() {
                    return Err(SklearsError::InvalidParameter {
                        name: "manual_gammas".to_string(),
                        reason: "manual gammas not provided".to_string(),
                    });
                }
                Ok(self.manual_gammas.clone())
            }
            BandwidthStrategy::LogarithmicSpacing => {
                let mut gammas = Vec::with_capacity(self.n_scales);
                if self.n_scales == 1 {
                    gammas.push((self.gamma_min * self.gamma_max).sqrt());
                } else {
                    let log_min = self.gamma_min.ln();
                    let log_max = self.gamma_max.ln();
                    for i in 0..self.n_scales {
                        let t = i as Float / (self.n_scales - 1) as Float;
                        let log_gamma = log_min + t * (log_max - log_min);
                        gammas.push(log_gamma.exp());
                    }
                }
                Ok(gammas)
            }
            BandwidthStrategy::LinearSpacing => {
                let mut gammas = Vec::with_capacity(self.n_scales);
                if self.n_scales == 1 {
                    gammas.push((self.gamma_min + self.gamma_max) / 2.0);
                } else {
                    for i in 0..self.n_scales {
                        let t = i as Float / (self.n_scales - 1) as Float;
                        let gamma = self.gamma_min + t * (self.gamma_max - self.gamma_min);
                        gammas.push(gamma);
                    }
                }
                Ok(gammas)
            }
            BandwidthStrategy::GeometricProgression => {
                let mut gammas = Vec::with_capacity(self.n_scales);
                let ratio = if self.n_scales == 1 {
                    1.0
                } else {
                    (self.gamma_max / self.gamma_min).powf(1.0 / (self.n_scales - 1) as Float)
                };
                for i in 0..self.n_scales {
                    let gamma = self.gamma_min * ratio.powi(i as i32);
                    gammas.push(gamma);
                }
                Ok(gammas)
            }
            BandwidthStrategy::Adaptive => {
                // Adaptive bandwidth selection based on data characteristics
                self.compute_adaptive_gammas(x)
            }
        }
    }

    /// Compute adaptive gamma values based on data characteristics
    fn compute_adaptive_gammas(&self, x: &Array2<Float>) -> Result<Vec<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for adaptive bandwidth selection".to_string(),
            ));
        }

        // Compute pairwise distances for a subset of points
        let n_subset = (n_samples.min(100)) as usize;
        let mut distances = Vec::new();

        for i in 0..n_subset {
            for j in (i + 1)..n_subset {
                let diff = &x.row(i) - &x.row(j);
                let dist_sq = diff.mapv(|x| x * x).sum();
                distances.push(dist_sq.sqrt());
            }
        }

        if distances.is_empty() {
            return Ok(vec![1.0; self.n_scales]);
        }

        // Sort distances
        distances.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Use percentiles to determine scales
        let mut gammas = Vec::with_capacity(self.n_scales);
        for i in 0..self.n_scales {
            let percentile = if self.n_scales == 1 {
                0.5
            } else {
                i as Float / (self.n_scales - 1) as Float
            };

            let idx = ((distances.len() - 1) as Float * percentile) as usize;
            let characteristic_distance = distances[idx];

            // gamma = 1 / (2 * sigma^2), where sigma is related to characteristic distance
            let gamma = if characteristic_distance > 0.0 {
                1.0 / (2.0 * characteristic_distance * characteristic_distance)
            } else {
                1.0
            };

            gammas.push(gamma);
        }

        Ok(gammas)
    }
}

impl Fit<Array2<Float>, ()> for MultiScaleRBFSampler<Untrained> {
    type Fitted = MultiScaleRBFSampler<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input array is empty".to_string(),
            ));
        }

        if self.n_scales == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_scales".to_string(),
                reason: "must be positive".to_string(),
            });
        }

        let mut rng = match self.random_state {
            Some(seed) => RealStdRng::seed_from_u64(seed),
            None => RealStdRng::from_seed(thread_rng().gen()),
        };

        // Compute gamma values for each scale
        let gammas = self.compute_gammas(x)?;

        // Generate random weights and offsets for each scale
        let mut random_weights = Vec::with_capacity(self.n_scales);
        let mut random_offsets = Vec::with_capacity(self.n_scales);

        for &gamma in &gammas {
            // Generate random weights ~ N(0, 2*gamma*I)
            let std_dev = (2.0 * gamma).sqrt();
            let mut weights = Array2::zeros((self.n_components_per_scale, n_features));
            for i in 0..self.n_components_per_scale {
                for j in 0..n_features {
                    weights[[i, j]] =
                        rng.sample::<Float, _>(RandNormal::new(0.0, std_dev).map_err(|e| {
                            SklearsError::NumericalError(format!(
                                "Error creating normal distribution: {}",
                                e
                            ))
                        })?);
                }
            }

            // Generate random offsets ~ Uniform[0, 2π]
            let mut offsets = Array1::zeros(self.n_components_per_scale);
            for i in 0..self.n_components_per_scale {
                offsets[i] = rng
                    .sample::<Float, _>(RandUniform::new(0.0, 2.0 * std::f64::consts::PI).unwrap());
            }

            random_weights.push(weights);
            random_offsets.push(offsets);
        }

        // Compute attention weights if using attention-based combination
        let attention_weights =
            if matches!(self.combination_strategy, CombinationStrategy::Attention) {
                Some(compute_attention_weights(&gammas)?)
            } else {
                None
            };

        Ok(MultiScaleRBFSampler {
            n_components_per_scale: self.n_components_per_scale,
            n_scales: self.n_scales,
            gamma_min: self.gamma_min,
            gamma_max: self.gamma_max,
            manual_gammas: self.manual_gammas,
            bandwidth_strategy: self.bandwidth_strategy,
            combination_strategy: self.combination_strategy,
            scale_weights: self.scale_weights,
            random_state: self.random_state,
            gammas_: Some(gammas),
            random_weights_: Some(random_weights),
            random_offsets_: Some(random_offsets),
            attention_weights_: attention_weights,
            _state: PhantomData,
        })
    }
}

/// Compute attention weights based on gamma values
fn compute_attention_weights(gammas: &[Float]) -> Result<Array1<Float>> {
    // Simple attention mechanism: higher gamma (smaller scale) gets more weight
    let weights: Vec<Float> = gammas.iter().map(|&g| g.ln()).collect();
    let weights_array = Array1::from(weights);

    // Softmax normalization
    let max_weight = weights_array
        .iter()
        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    let exp_weights = weights_array.mapv(|w| (w - max_weight).exp());
    let sum_exp = exp_weights.sum();

    Ok(exp_weights.mapv(|w| w / sum_exp))
}

impl Transform<Array2<Float>> for MultiScaleRBFSampler<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let gammas = self
            .gammas_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform".to_string(),
            })?;

        let random_weights =
            self.random_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let random_offsets =
            self.random_offsets_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "transform".to_string(),
                })?;

        let (n_samples, n_features) = x.dim();

        // Generate features for each scale
        let mut scale_features = Vec::with_capacity(self.n_scales);

        for i in 0..self.n_scales {
            let weights = &random_weights[i];
            let offsets = &random_offsets[i];

            if n_features != weights.ncols() {
                return Err(SklearsError::InvalidInput(format!(
                    "Input has {} features, expected {}",
                    n_features,
                    weights.ncols()
                )));
            }

            // Compute X @ W.T + b
            let projection = x.dot(&weights.t()) + offsets;

            // Apply cosine transformation and normalize
            let normalization = (2.0 / weights.nrows() as Float).sqrt();
            let features = projection.mapv(|x| x.cos() * normalization);

            scale_features.push(features);
        }

        // Combine features across scales
        match self.combination_strategy {
            CombinationStrategy::Concatenation => self.concatenate_features(scale_features),
            CombinationStrategy::WeightedAverage => self.weighted_average_features(scale_features),
            CombinationStrategy::MaxPooling => self.max_pooling_features(scale_features),
            CombinationStrategy::AveragePooling => self.average_pooling_features(scale_features),
            CombinationStrategy::Attention => self.attention_combine_features(scale_features),
        }
    }
}

impl MultiScaleRBFSampler<Trained> {
    /// Concatenate features from all scales
    fn concatenate_features(&self, scale_features: Vec<Array2<Float>>) -> Result<Array2<Float>> {
        if scale_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale features to concatenate".to_string(),
            ));
        }

        let n_samples = scale_features[0].nrows();
        let total_features: usize = scale_features.iter().map(|f| f.ncols()).sum();

        let mut result = Array2::zeros((n_samples, total_features));
        let mut col_offset = 0;

        for features in scale_features {
            let n_cols = features.ncols();
            result
                .slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(&features);
            col_offset += n_cols;
        }

        Ok(result)
    }

    /// Compute weighted average of features across scales
    fn weighted_average_features(
        &self,
        scale_features: Vec<Array2<Float>>,
    ) -> Result<Array2<Float>> {
        if scale_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale features to average".to_string(),
            ));
        }

        let weights = if self.scale_weights.is_empty() {
            // Equal weights
            vec![1.0 / self.n_scales as Float; self.n_scales]
        } else {
            // Normalize provided weights
            let sum: Float = self.scale_weights.iter().sum();
            self.scale_weights.iter().map(|&w| w / sum).collect()
        };

        let mut result = scale_features[0].clone() * weights[0];
        for (i, features) in scale_features.iter().enumerate().skip(1) {
            result = result + features * weights[i];
        }

        Ok(result)
    }

    /// Apply max pooling across scales
    fn max_pooling_features(&self, scale_features: Vec<Array2<Float>>) -> Result<Array2<Float>> {
        if scale_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale features for max pooling".to_string(),
            ));
        }

        let mut result = scale_features[0].clone();
        for features in scale_features.iter().skip(1) {
            for ((i, j), val) in features.indexed_iter() {
                if *val > result[[i, j]] {
                    result[[i, j]] = *val;
                }
            }
        }

        Ok(result)
    }

    /// Apply average pooling across scales
    fn average_pooling_features(
        &self,
        scale_features: Vec<Array2<Float>>,
    ) -> Result<Array2<Float>> {
        if scale_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale features for average pooling".to_string(),
            ));
        }

        let mut result = scale_features[0].clone();
        for features in scale_features.iter().skip(1) {
            result = result + features;
        }

        result.mapv_inplace(|x| x / self.n_scales as Float);
        Ok(result)
    }

    /// Apply attention-based combination of features
    fn attention_combine_features(
        &self,
        scale_features: Vec<Array2<Float>>,
    ) -> Result<Array2<Float>> {
        if scale_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No scale features for attention combination".to_string(),
            ));
        }

        let attention_weights =
            self.attention_weights_
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "attention combination".to_string(),
                })?;

        let mut result = scale_features[0].clone() * attention_weights[0];
        for (i, features) in scale_features.iter().enumerate().skip(1) {
            result = result + features * attention_weights[i];
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_scale_rbf_sampler_basic() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler = MultiScaleRBFSampler::new(10)
            .n_scales(3)
            .gamma_range(0.1, 10.0)
            .bandwidth_strategy(BandwidthStrategy::LogarithmicSpacing)
            .combination_strategy(CombinationStrategy::Concatenation)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        // Concatenation should give 3 scales * 10 components = 30 features
        assert_eq!(features.shape(), &[3, 30]);

        // Check that features are bounded (cosine function)
        for &val in features.iter() {
            assert!(val >= -2.0 && val <= 2.0);
        }
    }

    #[test]
    fn test_different_bandwidth_strategies() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let strategies = [
            BandwidthStrategy::LogarithmicSpacing,
            BandwidthStrategy::LinearSpacing,
            BandwidthStrategy::GeometricProgression,
            BandwidthStrategy::Adaptive,
        ];

        for strategy in &strategies {
            let sampler = MultiScaleRBFSampler::new(5)
                .n_scales(3)
                .gamma_range(0.1, 10.0)
                .bandwidth_strategy(*strategy)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.shape(), &[2, 15]); // 3 scales * 5 components
        }
    }

    #[test]
    fn test_different_combination_strategies() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let strategies = [
            (CombinationStrategy::Concatenation, 30), // 3 scales * 10 components
            (CombinationStrategy::WeightedAverage, 10), // Same as single scale
            (CombinationStrategy::MaxPooling, 10),
            (CombinationStrategy::AveragePooling, 10),
            (CombinationStrategy::Attention, 10),
        ];

        for (strategy, expected_features) in &strategies {
            let sampler = MultiScaleRBFSampler::new(10)
                .n_scales(3)
                .combination_strategy(*strategy)
                .random_state(42);

            let fitted = sampler.fit(&x, &()).unwrap();
            let features = fitted.transform(&x).unwrap();

            assert_eq!(features.shape(), &[3, *expected_features]);
        }
    }

    #[test]
    fn test_manual_gammas() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let manual_gammas = vec![0.1, 1.0, 10.0];

        let sampler = MultiScaleRBFSampler::new(8)
            .manual_gammas(manual_gammas.clone())
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 24]); // 3 scales * 8 components
        assert_eq!(fitted.gammas_.as_ref().unwrap(), &manual_gammas);
    }

    #[test]
    fn test_scale_weights() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let weights = vec![1.0, 2.0, 0.5];

        let sampler = MultiScaleRBFSampler::new(10)
            .n_scales(3)
            .combination_strategy(CombinationStrategy::WeightedAverage)
            .scale_weights(weights.clone())
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 10]);
    }

    #[test]
    fn test_reproducibility() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let sampler1 = MultiScaleRBFSampler::new(20)
            .n_scales(4)
            .bandwidth_strategy(BandwidthStrategy::LogarithmicSpacing)
            .combination_strategy(CombinationStrategy::Concatenation)
            .random_state(123);

        let sampler2 = MultiScaleRBFSampler::new(20)
            .n_scales(4)
            .bandwidth_strategy(BandwidthStrategy::LogarithmicSpacing)
            .combination_strategy(CombinationStrategy::Concatenation)
            .random_state(123);

        let fitted1 = sampler1.fit(&x, &()).unwrap();
        let fitted2 = sampler2.fit(&x, &()).unwrap();

        let features1 = fitted1.transform(&x).unwrap();
        let features2 = fitted2.transform(&x).unwrap();

        for (f1, f2) in features1.iter().zip(features2.iter()) {
            assert_abs_diff_eq!(f1, f2, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_adaptive_bandwidth() {
        let x = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [5.0, 5.0],
            [5.1, 5.1],
            [10.0, 10.0],
            [10.1, 10.1]
        ];

        let sampler = MultiScaleRBFSampler::new(15)
            .n_scales(3)
            .bandwidth_strategy(BandwidthStrategy::Adaptive)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[6, 45]); // 3 scales * 15 components

        // Check that adaptive gammas were computed
        let gammas = fitted.gammas_.as_ref().unwrap();
        assert_eq!(gammas.len(), 3);
        assert!(gammas.iter().all(|&g| g > 0.0));
    }

    #[test]
    fn test_error_handling() {
        // Empty input
        let empty = Array2::<Float>::zeros((0, 0));
        let sampler = MultiScaleRBFSampler::new(10);
        assert!(sampler.clone().fit(&empty, &()).is_err());

        // Zero scales
        let x = array![[1.0, 2.0]];
        let invalid_sampler = MultiScaleRBFSampler::new(10).n_scales(0);
        assert!(invalid_sampler.fit(&x, &()).is_err());

        // Dimension mismatch in transform
        let x_train = array![[1.0, 2.0], [3.0, 4.0]];
        let x_test = array![[1.0, 2.0, 3.0]]; // Wrong number of features

        let fitted = sampler.fit(&x_train, &()).unwrap();
        assert!(fitted.transform(&x_test).is_err());
    }

    #[test]
    fn test_single_scale() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];

        let sampler = MultiScaleRBFSampler::new(15)
            .n_scales(1)
            .gamma_range(1.0, 1.0)
            .random_state(42);

        let fitted = sampler.fit(&x, &()).unwrap();
        let features = fitted.transform(&x).unwrap();

        assert_eq!(features.shape(), &[2, 15]);

        let gammas = fitted.gammas_.as_ref().unwrap();
        assert_eq!(gammas.len(), 1);
    }

    #[test]
    fn test_gamma_computation_strategies() {
        let sampler = MultiScaleRBFSampler::new(10)
            .n_scales(4)
            .gamma_range(0.1, 10.0);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        // Test logarithmic spacing
        let log_sampler = sampler
            .clone()
            .bandwidth_strategy(BandwidthStrategy::LogarithmicSpacing);
        let log_gammas = log_sampler.compute_gammas(&x).unwrap();
        assert_eq!(log_gammas.len(), 4);
        assert_abs_diff_eq!(log_gammas[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(log_gammas[3], 10.0, epsilon = 1e-10);

        // Test linear spacing
        let lin_sampler = sampler
            .clone()
            .bandwidth_strategy(BandwidthStrategy::LinearSpacing);
        let lin_gammas = lin_sampler.compute_gammas(&x).unwrap();
        assert_eq!(lin_gammas.len(), 4);
        assert_abs_diff_eq!(lin_gammas[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(lin_gammas[3], 10.0, epsilon = 1e-10);

        // Test geometric progression
        let geo_sampler = sampler
            .clone()
            .bandwidth_strategy(BandwidthStrategy::GeometricProgression);
        let geo_gammas = geo_sampler.compute_gammas(&x).unwrap();
        assert_eq!(geo_gammas.len(), 4);
        assert_abs_diff_eq!(geo_gammas[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(geo_gammas[3], 10.0, epsilon = 1e-10);
    }
}

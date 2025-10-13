//! Sampling Methods Module
//!
//! This module provides comprehensive sampling algorithms for efficient
//! feature extraction and data processing with various sampling strategies.

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, rng};
use sklears_core::prelude::SklearsError;
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet};

/// Reservoir Sampling for Large Dataset Processing
///
/// Implements reservoir sampling algorithm for selecting a fixed-size sample
/// from a large or streaming dataset with uniform probability.
///
/// This algorithm maintains a reservoir of fixed size and ensures that each
/// element in the stream has an equal probability of being included in the
/// final sample, regardless of the total size of the stream.
///
/// # Parameters
///
/// * `reservoir_size` - Size of the sample to maintain
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::ReservoirSampler;
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((1000, 5), (0..5000).map(|x| x as f64).collect())?;
///
/// let mut sampler = ReservoirSampler::new()
///     .reservoir_size(100)
///     .random_state(42);
///
/// let sampled_data = sampler.sample_data(&data.view())?;
/// assert_eq!(sampled_data.nrows(), 100);
/// assert_eq!(sampled_data.ncols(), 5);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ReservoirSampler {
    reservoir_size: usize,
    random_state: Option<u64>,
    reservoir: Vec<usize>,
    seen_count: usize,
}

impl ReservoirSampler {
    /// Create a new ReservoirSampler
    pub fn new() -> Self {
        Self {
            reservoir_size: 1000,
            random_state: None,
            reservoir: Vec::new(),
            seen_count: 0,
        }
    }

    /// Set the reservoir size
    pub fn reservoir_size(mut self, size: usize) -> Self {
        self.reservoir_size = size;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices from a dataset using reservoir sampling
    pub fn sample_indices(&mut self, dataset_size: usize) -> SklResult<Vec<usize>> {
        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        let actual_size = self.reservoir_size.min(dataset_size);
        let mut reservoir = Vec::with_capacity(actual_size);

        // Fill reservoir with first k elements
        for i in 0..actual_size {
            reservoir.push(i);
        }

        // Replace elements with gradually decreasing probability
        for i in actual_size..dataset_size {
            let j = rng.gen_range(0..=i);
            if j < actual_size {
                reservoir[j] = i;
            }
        }

        self.reservoir = reservoir.clone();
        self.seen_count = dataset_size;
        Ok(reservoir)
    }

    /// Sample data using reservoir sampling
    pub fn sample_data(&mut self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let indices = self.sample_indices(data.nrows())?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }

    /// Get current reservoir indices
    pub fn get_reservoir_indices(&self) -> &[usize] {
        &self.reservoir
    }

    /// Get count of seen samples
    pub fn get_seen_count(&self) -> usize {
        self.seen_count
    }
}

impl Default for ReservoirSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Importance Sampling for Weighted Data Selection
///
/// Implements importance sampling for selecting samples based on weights,
/// useful for handling imbalanced datasets or focusing on important regions.
///
/// This sampling method allows for biased sampling where certain data points
/// are more likely to be selected based on their importance weights.
///
/// # Parameters
///
/// * `sample_size` - Number of samples to select
/// * `replacement` - Whether to sample with replacement
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::ImportanceSampler;
/// # use scirs2_core::ndarray::{Array1, Array2};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())?;
/// let weights = Array1::from_vec((0..100).map(|x| (x + 1) as f64).collect());
///
/// let sampler = ImportanceSampler::new()
///     .sample_size(20)
///     .replacement(true)
///     .random_state(42);
///
/// let indices = sampler.sample_indices(&weights.view())?;
/// assert_eq!(indices.len(), 20);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ImportanceSampler {
    sample_size: usize,
    replacement: bool,
    random_state: Option<u64>,
}

impl ImportanceSampler {
    /// Create a new ImportanceSampler
    pub fn new() -> Self {
        Self {
            sample_size: 1000,
            replacement: false,
            random_state: None,
        }
    }

    /// Set the sample size
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set whether to sample with replacement
    pub fn replacement(mut self, replacement: bool) -> Self {
        self.replacement = replacement;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices based on importance weights
    pub fn sample_indices(&self, weights: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        if weights.len() == 0 {
            return Err(SklearsError::InvalidInput(
                "Weights cannot be empty".to_string(),
            ));
        }

        // Normalize weights to probabilities
        let weight_sum: Float = weights.sum();
        if weight_sum <= 0.0 {
            return Err(SklearsError::InvalidInput(
                "Weights must be positive".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        // Create cumulative distribution
        let mut cumulative = Vec::with_capacity(weights.len());
        let mut sum = 0.0;
        for &weight in weights.iter() {
            sum += weight / weight_sum;
            cumulative.push(sum);
        }

        let mut selected_indices = Vec::with_capacity(self.sample_size);
        let mut used_indices = HashSet::new();

        for _ in 0..self.sample_size {
            let random_value: Float = rng.gen_range(0.0..1.0);

            // Find index using binary search on cumulative distribution
            let mut idx = 0;
            for (i, &cum_prob) in cumulative.iter().enumerate() {
                if random_value <= cum_prob {
                    idx = i;
                    break;
                }
            }

            if self.replacement || !used_indices.contains(&idx) {
                selected_indices.push(idx);
                if !self.replacement {
                    used_indices.insert(idx);
                }
            } else if !self.replacement {
                // Find next available index
                for i in 0..weights.len() {
                    if !used_indices.contains(&i) {
                        selected_indices.push(i);
                        used_indices.insert(i);
                        break;
                    }
                }
            }

            if !self.replacement && used_indices.len() == weights.len() {
                break;
            }
        }

        Ok(selected_indices)
    }

    /// Sample data using importance sampling
    pub fn sample_data(
        &self,
        data: &ArrayView2<Float>,
        weights: &ArrayView1<Float>,
    ) -> SklResult<Array2<Float>> {
        if data.nrows() != weights.len() {
            return Err(SklearsError::InvalidInput(
                "Data rows and weights length must match".to_string(),
            ));
        }

        let indices = self.sample_indices(weights)?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }
}

impl Default for ImportanceSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Stratified Sampling for Balanced Data Selection
///
/// Implements stratified sampling to maintain the same proportion of samples
/// in each stratum as in the original dataset, useful for maintaining class balance.
///
/// This method ensures that each subgroup (stratum) is represented proportionally
/// in the sample, which is particularly important for classification tasks.
///
/// # Parameters
///
/// * `sample_size` - Total number of samples to select
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::StratifiedSampler;
/// # use scirs2_core::ndarray::{Array1, Array2};
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())?;
/// let strata = Array1::from_vec((0..100).map(|x| (x % 3) as f64).collect());
///
/// let sampler = StratifiedSampler::new()
///     .sample_size(30)
///     .random_state(42);
///
/// let sampled_data = sampler.sample_data(&data.view(), &strata.view())?;
/// assert_eq!(sampled_data.nrows(), 30);
/// assert_eq!(sampled_data.ncols(), 3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct StratifiedSampler {
    sample_size: usize,
    random_state: Option<u64>,
}

impl StratifiedSampler {
    /// Create a new StratifiedSampler
    pub fn new() -> Self {
        Self {
            sample_size: 1000,
            random_state: None,
        }
    }

    /// Set the sample size
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices using stratified sampling
    pub fn sample_indices(&self, strata: &ArrayView1<Float>) -> SklResult<Vec<usize>> {
        if strata.len() == 0 {
            return Err(SklearsError::InvalidInput(
                "Strata cannot be empty".to_string(),
            ));
        }

        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        // Group indices by stratum
        let mut stratum_indices: HashMap<String, Vec<usize>> = HashMap::new();
        for (idx, &stratum) in strata.iter().enumerate() {
            let key = format!("{:.6}", stratum); // Use string representation for grouping
            stratum_indices
                .entry(key)
                .or_insert_with(Vec::new)
                .push(idx);
        }

        let total_samples = strata.len();
        let mut selected_indices = Vec::new();

        // Sample from each stratum proportionally
        for (_, indices) in stratum_indices.iter() {
            let stratum_proportion = indices.len() as f64 / total_samples as f64;
            let stratum_sample_size =
                (self.sample_size as f64 * stratum_proportion).round() as usize;
            let stratum_sample_size = stratum_sample_size.min(indices.len());

            if stratum_sample_size > 0 {
                let mut stratum_indices = indices.clone();
                Self::shuffle(&mut stratum_indices, &mut rng);
                selected_indices.extend(stratum_indices.into_iter().take(stratum_sample_size));
            }
        }

        // If we haven't reached the target sample size, add more samples
        while selected_indices.len() < self.sample_size && selected_indices.len() < total_samples {
            let all_indices: Vec<usize> = (0..total_samples).collect();
            let mut remaining: Vec<usize> = all_indices
                .into_iter()
                .filter(|&idx| !selected_indices.contains(&idx))
                .collect();

            if remaining.is_empty() {
                break;
            }

            Self::shuffle(&mut remaining, &mut rng);
            let needed = self.sample_size - selected_indices.len();
            selected_indices.extend(remaining.into_iter().take(needed));
        }

        selected_indices.truncate(self.sample_size);
        Ok(selected_indices)
    }

    /// Sample data using stratified sampling
    pub fn sample_data(
        &self,
        data: &ArrayView2<Float>,
        strata: &ArrayView1<Float>,
    ) -> SklResult<Array2<Float>> {
        if data.nrows() != strata.len() {
            return Err(SklearsError::InvalidInput(
                "Data rows and strata length must match".to_string(),
            ));
        }

        let indices = self.sample_indices(strata)?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }

    /// Helper function to shuffle a vector using Fisher-Yates algorithm
    fn shuffle<T>(slice: &mut [T], rng: &mut Random) {
        for i in (1..slice.len()).rev() {
            let j = rng.gen_range(0..=i);
            slice.swap(i, j);
        }
    }
}

impl Default for StratifiedSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Bootstrap Sampling for Statistical Inference
///
/// Implements bootstrap sampling for creating multiple resampled datasets
/// with replacement, useful for statistical inference and uncertainty estimation.
///
/// # Parameters
///
/// * `n_bootstrap_samples` - Number of bootstrap samples to generate
/// * `sample_size` - Size of each bootstrap sample (default: same as original)
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::BootstrapSampler;
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((100, 3), (0..300).map(|x| x as f64).collect())?;
///
/// let sampler = BootstrapSampler::new()
///     .n_bootstrap_samples(10)
///     .random_state(42);
///
/// let bootstrap_samples = sampler.sample_multiple(&data.view())?;
/// assert_eq!(bootstrap_samples.len(), 10);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct BootstrapSampler {
    n_bootstrap_samples: usize,
    sample_size: Option<usize>,
    random_state: Option<u64>,
}

impl BootstrapSampler {
    /// Create a new BootstrapSampler
    pub fn new() -> Self {
        Self {
            n_bootstrap_samples: 100,
            sample_size: None,
            random_state: None,
        }
    }

    /// Set the number of bootstrap samples to generate
    pub fn n_bootstrap_samples(mut self, n_samples: usize) -> Self {
        self.n_bootstrap_samples = n_samples;
        self
    }

    /// Set the size of each bootstrap sample
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = Some(size);
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Generate a single bootstrap sample
    pub fn sample_single(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        let sample_size = self.sample_size.unwrap_or(data.nrows());
        let mut indices = Vec::with_capacity(sample_size);

        // Sample with replacement
        for _ in 0..sample_size {
            let idx = rng.gen_range(0..data.nrows());
            indices.push(idx);
        }

        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }

    /// Generate multiple bootstrap samples
    pub fn sample_multiple(&self, data: &ArrayView2<Float>) -> SklResult<Vec<Array2<Float>>> {
        let mut samples = Vec::with_capacity(self.n_bootstrap_samples);

        for i in 0..self.n_bootstrap_samples {
            // Use different seed for each sample if random_state is set
            let sampler = if let Some(base_seed) = self.random_state {
                self.clone().random_state(base_seed + i as u64)
            } else {
                self.clone()
            };

            let sample = sampler.sample_single(data)?;
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Generate bootstrap indices for external use
    pub fn sample_indices(&self, n_samples: usize) -> SklResult<Vec<Vec<usize>>> {
        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        let sample_size = self.sample_size.unwrap_or(n_samples);
        let mut all_indices = Vec::with_capacity(self.n_bootstrap_samples);

        for _ in 0..self.n_bootstrap_samples {
            let mut indices = Vec::with_capacity(sample_size);
            for _ in 0..sample_size {
                let idx = rng.gen_range(0..n_samples);
                indices.push(idx);
            }
            all_indices.push(indices);
        }

        Ok(all_indices)
    }
}

impl Default for BootstrapSampler {
    fn default() -> Self {
        Self::new()
    }
}

/// Systematic Sampling for Regular Interval Selection
///
/// Implements systematic sampling where every k-th element is selected
/// after a random starting point, providing good coverage of the dataset.
///
/// # Parameters
///
/// * `sample_size` - Number of samples to select
/// * `random_state` - Random state for reproducible sampling
///
/// # Examples
///
/// ```
/// # use sklears_feature_extraction::engineering::SystematicSampler;
/// # use scirs2_core::ndarray::Array2;
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let data = Array2::from_shape_vec((1000, 3), (0..3000).map(|x| x as f64).collect())?;
///
/// let sampler = SystematicSampler::new()
///     .sample_size(100)
///     .random_state(42);
///
/// let sampled_data = sampler.sample_data(&data.view())?;
/// assert_eq!(sampled_data.nrows(), 100);
/// assert_eq!(sampled_data.ncols(), 3);
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct SystematicSampler {
    sample_size: usize,
    random_state: Option<u64>,
}

impl SystematicSampler {
    /// Create a new SystematicSampler
    pub fn new() -> Self {
        Self {
            sample_size: 100,
            random_state: None,
        }
    }

    /// Set the sample size
    pub fn sample_size(mut self, size: usize) -> Self {
        self.sample_size = size;
        self
    }

    /// Set the random state for reproducible sampling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Sample indices using systematic sampling
    pub fn sample_indices(&self, population_size: usize) -> SklResult<Vec<usize>> {
        if population_size == 0 {
            return Err(SklearsError::InvalidInput(
                "Population size cannot be zero".to_string(),
            ));
        }

        let sample_size = self.sample_size.min(population_size);
        if sample_size == 0 {
            return Ok(Vec::new());
        }

        let mut rng = if let Some(seed) = self.random_state {
            Random::with_seed(seed)
        } else {
            Random::new()
        };

        let interval = population_size as f64 / sample_size as f64;
        let start = rng.gen_range(0.0..interval);

        let mut indices = Vec::with_capacity(sample_size);
        for i in 0..sample_size {
            let idx = (start + i as f64 * interval) as usize;
            if idx < population_size {
                indices.push(idx);
            }
        }

        Ok(indices)
    }

    /// Sample data using systematic sampling
    pub fn sample_data(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let indices = self.sample_indices(data.nrows())?;
        let sampled_data = data.select(Axis(0), &indices);
        Ok(sampled_data)
    }
}

impl Default for SystematicSampler {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_reservoir_sampler_basic() {
        let mut sampler = ReservoirSampler::new()
            .reservoir_size(10)
            .random_state(42);

        let indices = sampler.sample_indices(20).unwrap();
        assert_eq!(indices.len(), 10);
        assert_eq!(sampler.get_seen_count(), 20);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 20);
        }
    }

    #[test]
    fn test_importance_sampler_basic() {
        let weights = Array1::from_vec(vec![0.1, 0.3, 0.2, 0.4]);
        let sampler = ImportanceSampler::new()
            .sample_size(3)
            .random_state(42);

        let indices = sampler.sample_indices(&weights.view()).unwrap();
        assert_eq!(indices.len(), 3);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 4);
        }
    }

    #[test]
    fn test_stratified_sampler_basic() {
        let strata = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let sampler = StratifiedSampler::new()
            .sample_size(4)
            .random_state(42);

        let indices = sampler.sample_indices(&strata.view()).unwrap();
        assert_eq!(indices.len(), 4);

        // All indices should be valid
        for &idx in &indices {
            assert!(idx < 6);
        }
    }

    #[test]
    fn test_bootstrap_sampler_basic() {
        let data = Array2::from_shape_vec((10, 3), (0..30).map(|x| x as f64).collect()).unwrap();
        let sampler = BootstrapSampler::new()
            .n_bootstrap_samples(5)
            .random_state(42);

        let samples = sampler.sample_multiple(&data.view()).unwrap();
        assert_eq!(samples.len(), 5);

        for sample in &samples {
            assert_eq!(sample.nrows(), 10);
            assert_eq!(sample.ncols(), 3);
        }
    }

    #[test]
    fn test_systematic_sampler_basic() {
        let data = Array2::from_shape_vec((100, 2), (0..200).map(|x| x as f64).collect()).unwrap();
        let sampler = SystematicSampler::new()
            .sample_size(10)
            .random_state(42);

        let sampled_data = sampler.sample_data(&data.view()).unwrap();
        assert_eq!(sampled_data.nrows(), 10);
        assert_eq!(sampled_data.ncols(), 2);
    }

    #[test]
    fn test_sampling_with_empty_data() {
        let empty_data = Array2::zeros((0, 3));

        let mut reservoir_sampler = ReservoirSampler::new().reservoir_size(5);
        let result = reservoir_sampler.sample_data(&empty_data.view());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().nrows(), 0);

        let systematic_sampler = SystematicSampler::new().sample_size(5);
        let result = systematic_sampler.sample_data(&empty_data.view());
        assert!(result.is_ok());
        assert_eq!(result.unwrap().nrows(), 0);
    }

    #[test]
    fn test_sampling_consistency_with_seed() {
        let data = Array2::from_shape_vec((50, 2), (0..100).map(|x| x as f64).collect()).unwrap();

        let mut sampler1 = ReservoirSampler::new()
            .reservoir_size(10)
            .random_state(42);
        let mut sampler2 = ReservoirSampler::new()
            .reservoir_size(10)
            .random_state(42);

        let sample1 = sampler1.sample_data(&data.view()).unwrap();
        let sample2 = sampler2.sample_data(&data.view()).unwrap();

        // Results should be identical with same seed
        assert_eq!(sample1.shape(), sample2.shape());
        for (a, b) in sample1.iter().zip(sample2.iter()) {
            assert_eq!(a, b);
        }
    }
}
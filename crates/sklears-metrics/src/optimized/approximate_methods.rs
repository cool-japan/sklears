//! Approximate Methods for Efficient Metric Computation
//!
//! This module provides approximate algorithms for metric computation that trade accuracy
//! for significant performance improvements. These methods are particularly useful for
//! very large datasets where exact computation would be prohibitively expensive.
//!
//! ## Key Features
//!
//! - **Sampling-based Methods**: Reservoir sampling with confidence intervals
//! - **Sketching Algorithms**: Count-Min sketch for frequency estimation
//! - **Approximate Confusion Matrix**: Hybrid exact/approximate counting
//! - **Quantile Estimation**: Memory-efficient percentile computation
//! - **Configurable Accuracy**: Trade-off between speed and precision

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive, ToPrimitive};
use std::collections::{BTreeSet, HashMap};
use std::hash::Hash;

/// Configuration for approximate metrics
#[derive(Debug, Clone)]
pub struct ApproximateConfig {
    /// Sample rate for sampling-based approximation (0.0 to 1.0)
    pub sample_rate: f64,
    /// Maximum number of samples to use
    pub max_samples: usize,
    /// Confidence level for approximate results
    pub confidence_level: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for ApproximateConfig {
    fn default() -> Self {
        Self {
            sample_rate: 0.1, // Use 10% of data by default
            max_samples: 10_000,
            confidence_level: 0.95,
            seed: None,
        }
    }
}

/// Reservoir sampling for unbiased sampling
pub struct ReservoirSampler<T> {
    reservoir: Vec<T>,
    size: usize,
    count: usize,
    rng_state: u64,
}

impl<T: Clone> ReservoirSampler<T> {
    pub fn new(size: usize, seed: Option<u64>) -> Self {
        Self {
            reservoir: Vec::with_capacity(size),
            size,
            count: 0,
            rng_state: seed.unwrap_or(42),
        }
    }

    /// Simple LCG for reproducible randomness
    fn next_random(&mut self) -> u64 {
        self.rng_state = self.rng_state.wrapping_mul(1103515245).wrapping_add(12345);
        self.rng_state
    }

    pub fn add(&mut self, item: T) {
        self.count += 1;

        if self.reservoir.len() < self.size {
            self.reservoir.push(item);
        } else {
            let j = (self.next_random() as usize) % self.count;
            if j < self.size {
                self.reservoir[j] = item;
            }
        }
    }

    pub fn samples(&self) -> &[T] {
        &self.reservoir
    }

    pub fn count(&self) -> usize {
        self.count
    }

    pub fn reset(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }
}

/// Sampling-based approximate metrics
pub struct SamplingMetrics<F> {
    _config: ApproximateConfig,
    true_sampler: ReservoirSampler<F>,
    pred_sampler: ReservoirSampler<F>,
}

impl<F: FloatTrait + FromPrimitive + Clone> SamplingMetrics<F> {
    pub fn new(config: ApproximateConfig) -> Self {
        let max_samples = config.max_samples;
        let seed = config.seed;

        Self {
            _config: config,
            true_sampler: ReservoirSampler::new(max_samples, seed),
            pred_sampler: ReservoirSampler::new(max_samples, seed.map(|s| s + 1)),
        }
    }

    /// Add samples for approximation
    pub fn add_samples(&mut self, y_true: &Array1<F>, y_pred: &Array1<F>) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
            self.true_sampler.add(true_val);
            self.pred_sampler.add(pred_val);
        }

        Ok(())
    }

    /// Approximate mean absolute error based on samples
    pub fn approximate_mean_absolute_error(&self) -> MetricsResult<F> {
        let true_samples = self.true_sampler.samples();
        let pred_samples = self.pred_sampler.samples();

        if true_samples.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let sum = true_samples
            .iter()
            .zip(pred_samples.iter())
            .map(|(t, p)| (*t - *p).abs())
            .fold(F::zero(), |acc, x| acc + x);

        Ok(sum / F::from(true_samples.len()).unwrap())
    }

    /// Approximate mean squared error based on samples
    pub fn approximate_mean_squared_error(&self) -> MetricsResult<F> {
        let true_samples = self.true_sampler.samples();
        let pred_samples = self.pred_sampler.samples();

        if true_samples.is_empty() {
            return Err(MetricsError::EmptyInput);
        }

        let sum = true_samples
            .iter()
            .zip(pred_samples.iter())
            .map(|(t, p)| {
                let diff = *t - *p;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x);

        Ok(sum / F::from(true_samples.len()).unwrap())
    }

    /// Get confidence interval for the approximation
    pub fn confidence_interval_mae(&self) -> MetricsResult<(F, F)> {
        let true_samples = self.true_sampler.samples();
        let pred_samples = self.pred_sampler.samples();

        if true_samples.len() < 2 {
            return Err(MetricsError::InvalidInput(
                "Need at least 2 samples for confidence interval".to_string(),
            ));
        }

        let errors: Vec<F> = true_samples
            .iter()
            .zip(pred_samples.iter())
            .map(|(t, p)| (*t - *p).abs())
            .collect();

        let mean =
            errors.iter().fold(F::zero(), |acc, &x| acc + x) / F::from(errors.len()).unwrap();

        // Calculate standard error
        let variance = errors
            .iter()
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .fold(F::zero(), |acc, x| acc + x)
            / F::from(errors.len() - 1).unwrap();

        let std_error = (variance / F::from(errors.len()).unwrap()).sqrt();

        // Use t-distribution approximation (assuming normal for large samples)
        let t_value = F::from(1.96).unwrap(); // 95% confidence for large samples
        let margin = std_error * t_value;

        Ok((mean - margin, mean + margin))
    }

    /// Number of samples collected
    pub fn sample_count(&self) -> usize {
        self.true_sampler.count()
    }

    /// Effective sample size used for computation
    pub fn effective_sample_size(&self) -> usize {
        self.true_sampler.samples().len()
    }

    /// Reset all samples
    pub fn reset(&mut self) {
        self.true_sampler.reset();
        self.pred_sampler.reset();
    }
}

/// Count-Min Sketch for approximate frequency counting
pub struct CountMinSketch {
    width: usize,
    _depth: usize,
    tables: Vec<Vec<u64>>,
    hash_seeds: Vec<u64>,
}

impl CountMinSketch {
    pub fn new(width: usize, depth: usize, seed: Option<u64>) -> Self {
        let base_seed = seed.unwrap_or(42);
        let mut hash_seeds = Vec::with_capacity(depth);
        for i in 0..depth {
            hash_seeds.push(base_seed.wrapping_add(i as u64 * 982451653));
        }

        Self {
            width,
            _depth: depth,
            tables: vec![vec![0; width]; depth],
            hash_seeds,
        }
    }

    fn hash(&self, value: u64, seed: u64) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        hasher.write_u64(seed);
        hasher.write_u64(value);
        (hasher.finish() as usize) % self.width
    }

    pub fn add(&mut self, value: u64) {
        for (i, &seed) in self.hash_seeds.iter().enumerate() {
            let hash_val = self.hash(value, seed);
            self.tables[i][hash_val] += 1;
        }
    }

    pub fn estimate(&self, value: u64) -> u64 {
        let mut min_count = u64::MAX;
        for (i, &seed) in self.hash_seeds.iter().enumerate() {
            let hash_val = self.hash(value, seed);
            min_count = min_count.min(self.tables[i][hash_val]);
        }
        min_count
    }

    pub fn reset(&mut self) {
        for table in &mut self.tables {
            for count in table {
                *count = 0;
            }
        }
    }
}

/// Approximate histogram using Count-Min Sketch
pub struct ApproximateHistogram<F> {
    sketch: CountMinSketch,
    bins: Vec<F>,
    bin_width: F,
    min_value: F,
    total_count: u64,
}

impl<F: FloatTrait + FromPrimitive + ToPrimitive + Copy> ApproximateHistogram<F> {
    pub fn new(
        min_val: F,
        max_val: F,
        num_bins: usize,
        sketch_width: usize,
        sketch_depth: usize,
    ) -> Self {
        let bin_width = (max_val - min_val) / F::from(num_bins).unwrap();
        let mut bins = Vec::with_capacity(num_bins);
        for i in 0..num_bins {
            bins.push(min_val + F::from(i).unwrap() * bin_width);
        }

        Self {
            sketch: CountMinSketch::new(sketch_width, sketch_depth, None),
            bins,
            bin_width,
            min_value: min_val,
            total_count: 0,
        }
    }

    pub fn add_value(&mut self, value: F) {
        let bin_index = if value < self.min_value {
            0
        } else {
            let idx = ((value - self.min_value) / self.bin_width)
                .to_usize()
                .unwrap_or(0);
            idx.min(self.bins.len() - 1)
        };

        self.sketch.add(bin_index as u64);
        self.total_count += 1;
    }

    pub fn estimate_frequency(&self, value: F) -> u64 {
        let bin_index = if value < self.min_value {
            0
        } else {
            let idx = ((value - self.min_value) / self.bin_width)
                .to_usize()
                .unwrap_or(0);
            idx.min(self.bins.len() - 1)
        };

        self.sketch.estimate(bin_index as u64)
    }

    pub fn total_count(&self) -> u64 {
        self.total_count
    }

    pub fn reset(&mut self) {
        self.sketch.reset();
        self.total_count = 0;
    }
}

/// Approximate percentiles using quantile sketches
pub struct QuantileSketch<F> {
    values: Vec<F>,
    max_size: usize,
    compressed: bool,
}

impl<F: FloatTrait + FromPrimitive + Copy + PartialOrd> QuantileSketch<F> {
    pub fn new(max_size: usize) -> Self {
        Self {
            values: Vec::new(),
            max_size,
            compressed: false,
        }
    }

    pub fn add(&mut self, value: F) {
        self.values.push(value);

        if self.values.len() > self.max_size {
            self.compress();
        }
    }

    fn compress(&mut self) {
        self.values
            .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Keep every other element to maintain approximate quantiles
        let mut compressed = Vec::with_capacity(self.max_size / 2);
        for (i, &val) in self.values.iter().enumerate() {
            if i % 2 == 0 {
                compressed.push(val);
            }
        }

        self.values = compressed;
        self.compressed = true;
    }

    pub fn quantile(&mut self, q: f64) -> Option<F> {
        if self.values.is_empty() {
            return None;
        }

        if !self.compressed {
            self.values
                .sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        }

        let index = (q * (self.values.len() - 1) as f64) as usize;
        Some(self.values[index.min(self.values.len() - 1)])
    }

    pub fn approximate_median(&mut self) -> Option<F> {
        self.quantile(0.5)
    }

    pub fn reset(&mut self) {
        self.values.clear();
        self.compressed = false;
    }
}

/// Approximate confusion matrix using sketching techniques
pub struct ApproximateConfusionMatrix<T> {
    /// Count-Min sketch for approximate counting
    sketch_width: usize,
    _sketch_depth: usize,
    sketches: Vec<Vec<u64>>,
    /// Hash functions for the sketch
    hash_seeds: Vec<u64>,
    /// Labels seen so far
    labels: BTreeSet<T>,
    /// Exact counts for small matrices
    exact_counts: Option<HashMap<(T, T), usize>>,
    /// Threshold for switching to approximation
    approximation_threshold: usize,
    n_samples: usize,
}

impl<T: PartialEq + Copy + Ord + Hash> ApproximateConfusionMatrix<T> {
    pub fn new(sketch_width: usize, sketch_depth: usize, approximation_threshold: usize) -> Self {
        let mut hash_seeds = Vec::with_capacity(sketch_depth);
        for i in 0..sketch_depth {
            hash_seeds.push(i as u64 * 982451653); // Prime number for hashing
        }

        Self {
            sketch_width,
            _sketch_depth: sketch_depth,
            sketches: vec![vec![0; sketch_width]; sketch_depth],
            hash_seeds,
            labels: BTreeSet::new(),
            exact_counts: Some(HashMap::new()),
            approximation_threshold,
            n_samples: 0,
        }
    }

    /// Hash function for Count-Min sketch
    fn hash(&self, key: (T, T), seed: u64) -> usize {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::Hasher;

        let mut hasher = DefaultHasher::new();
        hasher.write_u64(seed);
        key.hash(&mut hasher);
        (hasher.finish() as usize) % self.sketch_width
    }

    /// Update with new predictions
    pub fn update(&mut self, y_true: &Array1<T>, y_pred: &Array1<T>) -> MetricsResult<()> {
        if y_true.len() != y_pred.len() {
            return Err(MetricsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![y_pred.len()],
            });
        }

        for (&true_label, &pred_label) in y_true.iter().zip(y_pred.iter()) {
            self.update_single(true_label, pred_label);
        }

        Ok(())
    }

    /// Update with a single prediction
    pub fn update_single(&mut self, y_true: T, y_pred: T) {
        self.labels.insert(y_true);
        self.labels.insert(y_pred);
        self.n_samples += 1;

        let key = (y_true, y_pred);

        // Use exact counting if below threshold
        if let Some(ref mut exact_counts) = self.exact_counts {
            if self.n_samples < self.approximation_threshold {
                *exact_counts.entry(key).or_insert(0) += 1;
                return;
            } else {
                // Switch to approximation - transfer exact counts to sketch
                let counts_to_transfer: Vec<((T, T), usize)> = exact_counts.drain().collect();
                self.exact_counts = None;
                for ((true_label, pred_label), count) in counts_to_transfer {
                    for _ in 0..count {
                        self.update_sketch((true_label, pred_label));
                    }
                }
            }
        }

        self.update_sketch(key);
    }

    fn update_sketch(&mut self, key: (T, T)) {
        for (i, &seed) in self.hash_seeds.iter().enumerate() {
            let hash_val = self.hash(key, seed);
            self.sketches[i][hash_val] += 1;
        }
    }

    /// Get approximate count for a label pair
    pub fn get(&self, true_label: T, pred_label: T) -> usize {
        let key = (true_label, pred_label);

        // Use exact counts if available
        if let Some(ref exact_counts) = self.exact_counts {
            return exact_counts.get(&key).copied().unwrap_or(0);
        }

        // Use sketch approximation
        let mut min_count = u64::MAX;
        for (i, &seed) in self.hash_seeds.iter().enumerate() {
            let hash_val = self.hash(key, seed);
            min_count = min_count.min(self.sketches[i][hash_val]);
        }

        min_count as usize
    }

    pub fn labels(&self) -> Vec<T> {
        self.labels.iter().copied().collect()
    }

    pub fn n_samples(&self) -> usize {
        self.n_samples
    }

    /// Get approximate accuracy
    pub fn approximate_accuracy(&self) -> f64 {
        if self.n_samples == 0 {
            return 0.0;
        }

        let correct: usize = self
            .labels
            .iter()
            .map(|&label| self.get(label, label))
            .sum();

        correct as f64 / self.n_samples as f64
    }

    /// Reset the matrix
    pub fn reset(&mut self) {
        for sketch in &mut self.sketches {
            for count in sketch {
                *count = 0;
            }
        }
        self.labels.clear();
        self.exact_counts = Some(HashMap::new());
        self.n_samples = 0;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    use scirs2_core::ndarray::array;

    #[test]
    fn test_reservoir_sampler() {
        let mut sampler = ReservoirSampler::new(3, Some(42));

        // Add more items than the reservoir size
        for i in 0..10 {
            sampler.add(i);
        }

        assert_eq!(sampler.samples().len(), 3);
        assert_eq!(sampler.count(), 10);
    }

    #[test]
    fn test_sampling_metrics() {
        let config = ApproximateConfig {
            max_samples: 100,
            ..ApproximateConfig::default()
        };
        let mut metrics = SamplingMetrics::new(config);

        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 2.1, 2.9, 4.1, 4.9];

        metrics.add_samples(&y_true, &y_pred).unwrap();

        let mae = metrics.approximate_mean_absolute_error().unwrap();
        assert!(mae > 0.0);
        assert!(mae < 1.0); // Should be reasonable approximation

        let (lower, upper) = metrics.confidence_interval_mae().unwrap();
        assert!(lower <= mae);
        assert!(mae <= upper);
    }

    #[test]
    fn test_count_min_sketch() {
        let mut sketch = CountMinSketch::new(100, 5, Some(42));

        // Add some values
        sketch.add(1);
        sketch.add(1);
        sketch.add(2);

        assert_eq!(sketch.estimate(1), 2);
        assert_eq!(sketch.estimate(2), 1);
        assert_eq!(sketch.estimate(3), 0);
    }

    #[test]
    fn test_approximate_histogram() {
        let mut hist = ApproximateHistogram::new(0.0, 10.0, 10, 100, 5);

        // Add values
        hist.add_value(1.5);
        hist.add_value(2.5);
        hist.add_value(1.5);

        assert_eq!(hist.total_count(), 3);
        assert!(hist.estimate_frequency(1.5) >= 2);
    }

    #[test]
    fn test_quantile_sketch() {
        let mut sketch = QuantileSketch::new(100);

        for i in 0..20 {
            sketch.add(i as f64);
        }

        let median = sketch.approximate_median().unwrap();
        assert!(median >= 8.0 && median <= 12.0); // Should be around 9.5
    }

    #[test]
    fn test_approximate_confusion_matrix() {
        let mut matrix = ApproximateConfusionMatrix::new(100, 5, 1000);

        let y_true = array![0, 1, 2, 0, 1, 2];
        let y_pred = array![0, 1, 1, 0, 0, 1];

        matrix.update(&y_true, &y_pred).unwrap();

        assert_eq!(matrix.n_samples(), 6);
        assert_eq!(matrix.get(0, 0), 2); // Should be exact for small data
        assert!(matrix.approximate_accuracy() > 0.0);
    }
}

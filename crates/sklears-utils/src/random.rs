//! Random number generation utilities

use crate::{UtilsError, UtilsResult};
use lazy_static::lazy_static;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use std::sync::Mutex;

lazy_static! {
    static ref GLOBAL_RNG: Mutex<StdRng> = Mutex::new(StdRng::seed_from_u64(42));
}

/// Set the global random seed for reproducible results
pub fn set_random_state(seed: u64) {
    let mut rng = GLOBAL_RNG.lock().unwrap();
    *rng = StdRng::seed_from_u64(seed);
}

/// Get a random number generator with the specified seed
pub fn get_rng(seed: Option<u64>) -> StdRng {
    match seed {
        Some(s) => StdRng::seed_from_u64(s),
        None => {
            let mut rng = GLOBAL_RNG.lock().unwrap();
            StdRng::seed_from_u64(rng.random())
        }
    }
}

/// Generate random indices for sampling
pub fn random_indices(
    n_samples: usize,
    size: usize,
    replace: bool,
    seed: Option<u64>,
) -> UtilsResult<Vec<usize>> {
    if !replace && size > n_samples {
        return Err(UtilsError::InvalidParameter(format!(
            "Cannot sample {size} items from {n_samples} without replacement"
        )));
    }

    let mut rng = get_rng(seed);
    let mut indices = Vec::with_capacity(size);

    if replace {
        // Sampling with replacement
        for _ in 0..size {
            indices.push(rng.random_range(0..n_samples));
        }
    } else {
        // Sampling without replacement
        let mut available: Vec<usize> = (0..n_samples).collect();
        for _ in 0..size {
            let idx = rng.random_range(0..available.len());
            indices.push(available.swap_remove(idx));
        }
    }

    Ok(indices)
}

/// Shuffle an array of indices in place
pub fn shuffle_indices(indices: &mut [usize], seed: Option<u64>) {
    let mut rng = get_rng(seed);
    for i in (1..indices.len()).rev() {
        let j = rng.random_range(0..=i);
        indices.swap(i, j);
    }
}

/// Generate a random permutation of indices
pub fn random_permutation(n: usize, seed: Option<u64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n).collect();
    shuffle_indices(&mut indices, seed);
    indices
}

/// Split indices into train/test sets
pub fn train_test_split_indices(
    n_samples: usize,
    test_size: f64,
    shuffle: bool,
    seed: Option<u64>,
) -> UtilsResult<(Vec<usize>, Vec<usize>)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(UtilsError::InvalidParameter(format!(
            "test_size must be in (0, 1), got {test_size}"
        )));
    }

    let test_samples = (n_samples as f64 * test_size).round() as usize;
    let train_samples = n_samples - test_samples;

    let indices = if shuffle {
        random_permutation(n_samples, seed)
    } else {
        (0..n_samples).collect()
    };

    let train_indices = indices[..train_samples].to_vec();
    let test_indices = indices[train_samples..].to_vec();

    Ok((train_indices, test_indices))
}

/// Generate random weights that sum to 1
pub fn random_weights(n: usize, seed: Option<u64>) -> Vec<f64> {
    let mut rng = get_rng(seed);
    let mut weights: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
    let sum: f64 = weights.iter().sum();

    if sum > 0.0 {
        for w in &mut weights {
            *w /= sum;
        }
    } else {
        // Fallback to uniform weights
        let uniform_weight = 1.0 / n as f64;
        weights.fill(uniform_weight);
    }

    weights
}

/// Bootstrap sampling: sample n_samples with replacement
pub fn bootstrap_indices(n_samples: usize, seed: Option<u64>) -> Vec<usize> {
    random_indices(n_samples, n_samples, true, seed).unwrap()
}

/// Generate k-fold cross-validation indices
pub fn k_fold_indices(
    n_samples: usize,
    n_splits: usize,
    shuffle: bool,
    seed: Option<u64>,
) -> UtilsResult<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_splits < 2 {
        return Err(UtilsError::InvalidParameter(format!(
            "n_splits must be at least 2, got {n_splits}"
        )));
    }

    if n_splits > n_samples {
        return Err(UtilsError::InvalidParameter(format!(
            "n_splits {n_splits} cannot be greater than the number of samples {n_samples}"
        )));
    }

    let indices = if shuffle {
        random_permutation(n_samples, seed)
    } else {
        (0..n_samples).collect()
    };

    let mut folds = Vec::with_capacity(n_splits);
    let fold_sizes: Vec<usize> = (0..n_splits)
        .map(|i| (n_samples + n_splits - i - 1) / n_splits)
        .collect();

    let mut start = 0;
    for fold_size in fold_sizes {
        let end = start + fold_size;
        let test_indices = indices[start..end].to_vec();
        let mut train_indices = Vec::with_capacity(n_samples - fold_size);
        train_indices.extend(&indices[..start]);
        train_indices.extend(&indices[end..]);

        folds.push((train_indices, test_indices));
        start = end;
    }

    Ok(folds)
}

/// Generate stratified train/test split indices
pub fn stratified_split_indices(
    labels: &[i32],
    test_size: f64,
    seed: Option<u64>,
) -> UtilsResult<(Vec<usize>, Vec<usize>)> {
    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(UtilsError::InvalidParameter(format!(
            "test_size must be in (0, 1), got {test_size}"
        )));
    }

    // Group indices by class
    let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        class_indices.entry(label).or_default().push(idx);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Split each class separately
    for indices in class_indices.values() {
        let n_class = indices.len();
        let n_test = (n_class as f64 * test_size).round() as usize;
        let n_test = n_test.max(1).min(n_class - 1); // Ensure at least 1 in each split

        let (class_train, class_test) =
            train_test_split_indices(n_class, n_test as f64 / n_class as f64, true, seed)?;

        train_indices.extend(class_train.iter().map(|&i| indices[i]));
        test_indices.extend(class_test.iter().map(|&i| indices[i]));
    }

    Ok((train_indices, test_indices))
}

/// Reservoir sampling: efficiently sample k items from a stream of unknown size
pub fn reservoir_sampling<T: Clone>(
    items: impl Iterator<Item = T>,
    k: usize,
    seed: Option<u64>,
) -> Vec<T> {
    if k == 0 {
        return Vec::new();
    }

    let mut rng = get_rng(seed);
    let mut reservoir = Vec::with_capacity(k);

    for (i, item) in items.enumerate() {
        if i < k {
            // Fill the reservoir for the first k items
            reservoir.push(item);
        } else {
            // Randomly replace items in the reservoir
            let j = rng.random_range(0..=i);
            if j < k {
                reservoir[j] = item;
            }
        }
    }

    reservoir
}

/// Weighted sampling without replacement using systematic sampling
pub fn weighted_sampling_without_replacement(
    weights: &[f64],
    k: usize,
    seed: Option<u64>,
) -> UtilsResult<Vec<usize>> {
    if weights.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    if k > weights.len() {
        return Err(UtilsError::InvalidParameter(format!(
            "Cannot sample {} items from {} weights without replacement",
            k,
            weights.len()
        )));
    }

    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return Err(UtilsError::InvalidParameter(
            "Sum of weights must be positive".to_string(),
        ));
    }

    let mut rng = get_rng(seed);
    let mut cumsum = Vec::with_capacity(weights.len());
    let mut running_sum = 0.0;

    for &w in weights {
        running_sum += w;
        cumsum.push(running_sum / sum);
    }

    let mut selected = Vec::new();
    let mut used = vec![false; weights.len()];

    for _ in 0..k {
        loop {
            let r: f64 = rng.random();
            let idx = cumsum
                .binary_search_by(|&x| x.partial_cmp(&r).unwrap())
                .unwrap_or_else(|i| i);

            if idx < weights.len() && !used[idx] {
                used[idx] = true;
                selected.push(idx);
                break;
            }
        }
    }

    Ok(selected)
}

/// Importance sampling: sample indices according to importance weights
pub fn importance_sampling(
    weights: &[f64],
    n_samples: usize,
    seed: Option<u64>,
) -> UtilsResult<Vec<usize>> {
    if weights.is_empty() {
        return Err(UtilsError::EmptyInput);
    }

    let sum: f64 = weights.iter().sum();
    if sum <= 0.0 {
        return Err(UtilsError::InvalidParameter(
            "Sum of weights must be positive".to_string(),
        ));
    }

    let mut rng = get_rng(seed);
    let mut cumsum = Vec::with_capacity(weights.len());
    let mut running_sum = 0.0;

    for &w in weights {
        running_sum += w;
        cumsum.push(running_sum / sum);
    }

    let mut samples = Vec::with_capacity(n_samples);

    for _ in 0..n_samples {
        let r: f64 = rng.random();
        let idx = cumsum
            .binary_search_by(|&x| x.partial_cmp(&r).unwrap())
            .unwrap_or_else(|i| i);
        samples.push(idx.min(weights.len() - 1));
    }

    Ok(samples)
}

/// Advanced distribution sampling utilities
pub struct DistributionSampler {
    rng: StdRng,
}

impl DistributionSampler {
    /// Create a new distribution sampler with optional seed
    pub fn new(seed: Option<u64>) -> Self {
        Self { rng: get_rng(seed) }
    }

    /// Sample from normal distribution
    pub fn normal(&mut self, mean: f64, std: f64, n: usize) -> UtilsResult<Vec<f64>> {
        if std <= 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Standard deviation must be positive".to_string(),
            ));
        }

        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            // Box-Muller transform
            let u1 = self.rng.random::<f64>();
            let u2 = self.rng.random::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            samples.push(mean + std * z);
        }
        Ok(samples)
    }

    /// Sample from uniform distribution
    pub fn uniform(&mut self, low: f64, high: f64, n: usize) -> UtilsResult<Vec<f64>> {
        if low >= high {
            return Err(UtilsError::InvalidParameter(
                "Low bound must be less than high bound".to_string(),
            ));
        }

        let samples = (0..n)
            .map(|_| {
                let u = self.rng.random::<f64>();
                low + (high - low) * u
            })
            .collect();
        Ok(samples)
    }

    /// Sample from beta distribution
    pub fn beta(&mut self, alpha: f64, beta: f64, n: usize) -> UtilsResult<Vec<f64>> {
        if alpha <= 0.0 || beta <= 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Beta parameters must be positive".to_string(),
            ));
        }

        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            // Simple gamma ratio method
            let x = self.gamma_sample(alpha);
            let y = self.gamma_sample(beta);
            samples.push(x / (x + y));
        }
        Ok(samples)
    }

    /// Helper function to sample from gamma distribution
    fn gamma_sample(&mut self, shape: f64) -> f64 {
        // Simple approximation for gamma sampling
        if shape < 1.0 {
            let u = self.rng.random::<f64>();
            u.powf(1.0 / shape)
        } else {
            // Marsaglia and Tsang's method (simplified)
            let d = shape - 1.0 / 3.0;
            let c = 1.0 / (9.0 * d).sqrt();
            loop {
                let x = self.normal_sample();
                let v = (1.0 + c * x).powi(3);
                if v > 0.0 {
                    let u = self.rng.random::<f64>();
                    if u < 1.0 - 0.0331 * x.powi(4)
                        || u.ln() < 0.5 * x.powi(2) + d * (1.0 - v + v.ln())
                    {
                        return d * v;
                    }
                }
            }
        }
    }

    /// Helper function to sample from standard normal
    fn normal_sample(&mut self) -> f64 {
        let u1 = self.rng.random::<f64>();
        let u2 = self.rng.random::<f64>();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Sample from gamma distribution
    pub fn gamma(&mut self, shape: f64, scale: f64, n: usize) -> UtilsResult<Vec<f64>> {
        if shape <= 0.0 || scale <= 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Gamma parameters must be positive".to_string(),
            ));
        }

        let samples = (0..n).map(|_| self.gamma_sample(shape) * scale).collect();
        Ok(samples)
    }

    /// Sample from multivariate normal distribution (diagonal covariance)
    pub fn multivariate_normal_diag(
        &mut self,
        mean: &[f64],
        variances: &[f64],
        n: usize,
    ) -> UtilsResult<Vec<Vec<f64>>> {
        if mean.len() != variances.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![mean.len()],
                actual: vec![variances.len()],
            });
        }

        for &var in variances {
            if var <= 0.0 {
                return Err(UtilsError::InvalidParameter(
                    "All variances must be positive".to_string(),
                ));
            }
        }

        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            let mut sample = Vec::with_capacity(mean.len());
            for (&m, &v) in mean.iter().zip(variances.iter()) {
                let z = self.normal_sample();
                sample.push(m + z * v.sqrt());
            }
            samples.push(sample);
        }

        Ok(samples)
    }

    /// Sample from truncated normal distribution
    pub fn truncated_normal(
        &mut self,
        mean: f64,
        std: f64,
        low: f64,
        high: f64,
        n: usize,
    ) -> UtilsResult<Vec<f64>> {
        if std <= 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Standard deviation must be positive".to_string(),
            ));
        }

        if low >= high {
            return Err(UtilsError::InvalidParameter(
                "Low bound must be less than high bound".to_string(),
            ));
        }

        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            loop {
                let sample = mean + std * self.normal_sample();
                if sample >= low && sample <= high {
                    samples.push(sample);
                    break;
                }
            }
        }

        Ok(samples)
    }

    /// Sample from mixture distribution
    pub fn mixture_normal(
        &mut self,
        components: &[(f64, f64, f64)], // (weight, mean, std)
        n: usize,
    ) -> UtilsResult<Vec<f64>> {
        if components.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        // Validate and normalize weights
        let total_weight: f64 = components.iter().map(|(w, _, _)| w).sum();
        if total_weight <= 0.0 {
            return Err(UtilsError::InvalidParameter(
                "Total mixture weight must be positive".to_string(),
            ));
        }

        for &(_, _, std) in components {
            if std <= 0.0 {
                return Err(UtilsError::InvalidParameter(
                    "All standard deviations must be positive".to_string(),
                ));
            }
        }

        let mut cumulative_weights = Vec::with_capacity(components.len());
        let mut sum = 0.0;
        for &(weight, _, _) in components {
            sum += weight / total_weight;
            cumulative_weights.push(sum);
        }

        let mut samples = Vec::with_capacity(n);

        for _ in 0..n {
            // Choose component
            let r: f64 = self.rng.random();
            let component_idx = cumulative_weights
                .binary_search_by(|&x| x.partial_cmp(&r).unwrap())
                .unwrap_or_else(|i| i);

            let (_, mean, std) = components[component_idx];
            samples.push(mean + std * self.normal_sample());
        }

        Ok(samples)
    }
}

/// Thread-safe random state management
pub struct ThreadSafeRng {
    rng: Mutex<StdRng>,
}

impl ThreadSafeRng {
    /// Create a new thread-safe RNG with optional seed
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            rng: Mutex::new(get_rng(seed)),
        }
    }

    /// Generate a random number in range [0, 1)
    pub fn gen(&self) -> f64 {
        let mut rng = self.rng.lock().unwrap();
        rng.random()
    }

    /// Generate a random integer in range [0, n)
    pub fn gen_range(&self, n: usize) -> usize {
        let mut rng = self.rng.lock().unwrap();
        rng.random_range(0..n)
    }

    /// Generate random indices for sampling
    pub fn sample_indices(
        &self,
        n_samples: usize,
        size: usize,
        replace: bool,
    ) -> UtilsResult<Vec<usize>> {
        random_indices(n_samples, size, replace, None)
    }

    /// Serialize the current random state
    pub fn get_state(&self) -> [u8; 32] {
        let _rng = self.rng.lock().unwrap();
        // This is a simplified version - in practice you'd want to properly serialize the RNG state
        [0u8; 32] // Placeholder
    }

    /// Deserialize and set random state
    pub fn set_state(&self, _state: [u8; 32]) {
        // Placeholder - in practice you'd restore the RNG state
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_random_state() {
        set_random_state(42);
        let indices1 = random_indices(100, 10, false, None).unwrap();

        set_random_state(42);
        let indices2 = random_indices(100, 10, false, None).unwrap();

        assert_eq!(indices1, indices2);
    }

    #[test]
    fn test_random_indices_without_replacement() {
        let indices = random_indices(10, 5, false, Some(42)).unwrap();
        assert_eq!(indices.len(), 5);

        // Check uniqueness
        let mut sorted = indices.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 5);

        // Check bounds
        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_random_indices_with_replacement() {
        let indices = random_indices(5, 10, true, Some(42)).unwrap();
        assert_eq!(indices.len(), 10);

        // Check bounds
        for &idx in &indices {
            assert!(idx < 5);
        }
    }

    #[test]
    fn test_train_test_split_indices() {
        let (train, test) = train_test_split_indices(100, 0.2, true, Some(42)).unwrap();

        assert_eq!(train.len() + test.len(), 100);
        assert!((test.len() as f64 / 100.0 - 0.2).abs() < 0.1);

        // Check no overlap
        let mut all_indices = train.clone();
        all_indices.extend(&test);
        all_indices.sort();
        all_indices.dedup();
        assert_eq!(all_indices.len(), 100);
    }

    #[test]
    fn test_random_weights() {
        let weights = random_weights(5, Some(42));
        assert_eq!(weights.len(), 5);

        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        for &w in &weights {
            assert!(w >= 0.0);
        }
    }

    #[test]
    fn test_bootstrap_indices() {
        let indices = bootstrap_indices(10, Some(42));
        assert_eq!(indices.len(), 10);

        for &idx in &indices {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_stratified_split() {
        let labels = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
        let (train, test) = stratified_split_indices(&labels, 0.3, Some(42)).unwrap();

        assert_eq!(train.len() + test.len(), 10);

        // Check that each class appears in both train and test
        let train_labels: Vec<i32> = train.iter().map(|&i| labels[i]).collect();
        let test_labels: Vec<i32> = test.iter().map(|&i| labels[i]).collect();

        for &class in &[0, 1, 2] {
            assert!(train_labels.contains(&class));
            assert!(test_labels.contains(&class));
        }
    }

    #[test]
    fn test_reservoir_sampling() {
        let items: Vec<i32> = (0..100).collect();
        let sample = reservoir_sampling(items.into_iter(), 10, Some(42));

        assert_eq!(sample.len(), 10);

        // Check that all sampled items are in valid range
        for &item in &sample {
            assert!(item < 100);
        }
    }

    #[test]
    fn test_importance_sampling() {
        let weights = vec![0.1, 0.3, 0.6]; // Biased towards last item
        let samples = importance_sampling(&weights, 1000, Some(42)).unwrap();

        assert_eq!(samples.len(), 1000);

        // Count occurrences
        let mut counts = [0; 3];
        for &idx in &samples {
            counts[idx] += 1;
        }

        // Last item (index 2) should be sampled most frequently
        assert!(counts[2] > counts[1]);
        assert!(counts[1] > counts[0]);
    }

    #[test]
    fn test_weighted_sampling_without_replacement() {
        let weights = vec![1.0, 2.0, 3.0, 4.0];
        let sample = weighted_sampling_without_replacement(&weights, 3, Some(42)).unwrap();

        assert_eq!(sample.len(), 3);

        // Check uniqueness
        let mut sorted = sample.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
    }

    #[test]
    fn test_distribution_sampler() {
        let mut sampler = DistributionSampler::new(Some(42));

        // Test normal distribution
        let normal_samples = sampler.normal(0.0, 1.0, 100).unwrap();
        assert_eq!(normal_samples.len(), 100);

        // Test uniform distribution
        let uniform_samples = sampler.uniform(0.0, 1.0, 100).unwrap();
        assert_eq!(uniform_samples.len(), 100);
        for &sample in &uniform_samples {
            assert!(sample >= 0.0 && sample < 1.0);
        }

        // Test beta distribution
        let beta_samples = sampler.beta(2.0, 3.0, 100).unwrap();
        assert_eq!(beta_samples.len(), 100);
        for &sample in &beta_samples {
            assert!(sample >= 0.0 && sample <= 1.0);
        }

        // Test gamma distribution
        let gamma_samples = sampler.gamma(2.0, 1.0, 100).unwrap();
        assert_eq!(gamma_samples.len(), 100);
        for &sample in &gamma_samples {
            assert!(sample >= 0.0);
        }

        // Test multivariate normal
        let mean = vec![0.0, 1.0];
        let variances = vec![1.0, 2.0];
        let mv_samples = sampler
            .multivariate_normal_diag(&mean, &variances, 50)
            .unwrap();
        assert_eq!(mv_samples.len(), 50);
        for sample in &mv_samples {
            assert_eq!(sample.len(), 2);
        }

        // Test truncated normal
        let truncated_samples = sampler.truncated_normal(0.0, 1.0, -1.0, 1.0, 50).unwrap();
        assert_eq!(truncated_samples.len(), 50);
        for &sample in &truncated_samples {
            assert!(sample >= -1.0 && sample <= 1.0);
        }

        // Test mixture normal
        let components = vec![(0.3, -1.0, 0.5), (0.7, 1.0, 0.5)];
        let mixture_samples = sampler.mixture_normal(&components, 100).unwrap();
        assert_eq!(mixture_samples.len(), 100);
    }

    #[test]
    fn test_thread_safe_rng() {
        let rng = ThreadSafeRng::new(Some(42));

        let val1 = rng.gen();
        let val2 = rng.gen();
        assert_ne!(val1, val2);

        let idx = rng.gen_range(10);
        assert!(idx < 10);
    }
}

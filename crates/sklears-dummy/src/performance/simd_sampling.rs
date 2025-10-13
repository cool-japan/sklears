//! SIMD-accelerated sampling operations for dummy estimator performance
//!
//! This module provides optimized sampling implementations. Full SIMD functionality
//! requires nightly Rust features - scalar fallbacks are provided for stable compilation.

use scirs2_core::random::{thread_rng, Rng};

/// SIMD-accelerated random sampling (scalar fallback)
pub fn simd_uniform_samples(n: usize) -> Vec<f64> {
    let mut samples = Vec::with_capacity(n);
    let mut rng = thread_rng();

    for _ in 0..n {
        samples.push(rng.gen_range(0.0..1.0));
    }

    samples
}

/// SIMD-accelerated normal sampling using Box-Muller transform (scalar fallback)
pub fn simd_normal_samples(n: usize, mean: f64, std_dev: f64) -> Vec<f64> {
    let uniform_samples = simd_uniform_samples(n * 2); // Need pairs for Box-Muller
    let mut normal_samples = Vec::with_capacity(n);

    let mut i = 0;
    while i + 1 < uniform_samples.len() && normal_samples.len() < n {
        let u1 = uniform_samples[i];
        let u2 = uniform_samples[i + 1];

        // Box-Muller transformation
        if u1 > 0.0 {
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            normal_samples.push(mean + std_dev * z);
        }
        i += 2;
    }

    normal_samples.truncate(n);
    normal_samples
}

/// SIMD-accelerated weighted sampling (scalar fallback)
pub fn simd_weighted_sampling(weights: &[f64], n_samples: usize) -> Vec<usize> {
    if weights.is_empty() {
        return Vec::new();
    }

    // Compute cumulative weights
    let mut cumulative_weights = Vec::with_capacity(weights.len());
    let mut sum = 0.0;

    for &weight in weights {
        sum += weight;
        cumulative_weights.push(sum);
    }

    // Normalize
    let total_weight = sum;
    if total_weight == 0.0 {
        return Vec::new();
    }

    // Generate uniform samples
    let uniform_samples = simd_uniform_samples(n_samples);
    let mut indices = Vec::with_capacity(n_samples);

    for sample in uniform_samples {
        let target = sample * total_weight;
        // Binary search for the index
        let index = cumulative_weights
            .binary_search_by(|&x| x.partial_cmp(&target).unwrap())
            .unwrap_or_else(|x| x);
        indices.push(index.min(weights.len() - 1));
    }

    indices
}

/// SIMD-accelerated bootstrap sampling (scalar fallback)
pub fn simd_bootstrap_indices(n_samples: usize, n_bootstrap: usize) -> Vec<usize> {
    let uniform_samples = simd_uniform_samples(n_bootstrap);

    uniform_samples
        .iter()
        .map(|&u| (u * n_samples as f64).floor() as usize)
        .map(|idx| idx.min(n_samples - 1))
        .collect()
}

/// SIMD-accelerated stratified sampling (scalar fallback)
pub fn simd_stratified_sampling(
    strata_sizes: &[usize],
    samples_per_stratum: &[usize],
) -> Vec<(usize, usize)> {
    let mut results = Vec::new();

    for (stratum_idx, (&stratum_size, &n_samples)) in strata_sizes
        .iter()
        .zip(samples_per_stratum.iter())
        .enumerate()
    {
        if stratum_size == 0 || n_samples == 0 {
            continue;
        }

        let indices = simd_bootstrap_indices(stratum_size, n_samples);
        for idx in indices {
            results.push((stratum_idx, idx));
        }
    }

    results
}

/// Reservoir sampling for large datasets (scalar implementation)
pub fn reservoir_sampling<T: Clone>(data: &[T], k: usize) -> Vec<T> {
    if k >= data.len() {
        return data.to_vec();
    }

    let mut reservoir = Vec::with_capacity(k);
    let mut rng = thread_rng();

    // Fill reservoir with first k elements
    for i in 0..k {
        reservoir.push(data[i].clone());
    }

    // Replace elements with gradually decreasing probability
    for i in k..data.len() {
        let j = rng.gen_range(0..=i);
        if j < k {
            reservoir[j] = data[i].clone();
        }
    }

    reservoir
}

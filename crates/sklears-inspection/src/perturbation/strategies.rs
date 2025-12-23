//! Individual perturbation strategy implementations
//!
//! This module contains the specific implementations of different perturbation strategies
//! including Gaussian, uniform, adversarial, synthetic data generation, and others.

use super::core::{NoiseDistribution, PerturbationConfig, PerturbationStrategy};
use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};

/// Generate perturbations based on the specified strategy
pub fn generate_perturbations(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    match config.strategy {
        PerturbationStrategy::Gaussian => {
            #[cfg(feature = "simd")]
            {
                gaussian_perturbation_simd(X, config)
            }
            #[cfg(not(feature = "simd"))]
            {
                gaussian_perturbation(X, config)
            }
        }
        PerturbationStrategy::Uniform => {
            #[cfg(feature = "simd")]
            {
                uniform_perturbation_simd(X, config)
            }
            #[cfg(not(feature = "simd"))]
            {
                uniform_perturbation(X, config)
            }
        }
        PerturbationStrategy::Adversarial => adversarial_perturbation(X, config),
        PerturbationStrategy::Synthetic => synthetic_perturbation(X, config),
        PerturbationStrategy::DistributionPreserving => {
            distribution_preserving_perturbation(X, config)
        }
        PerturbationStrategy::Structured => structured_perturbation(X, config),
        PerturbationStrategy::SaltPepper => salt_pepper_perturbation(X, config),
        PerturbationStrategy::Dropout => dropout_perturbation(X, config),
    }
}

/// Gaussian noise perturbation
fn gaussian_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let std_dev = match &config.noise_distribution {
        NoiseDistribution::Gaussian { std } => *std,
        _ => config.magnitude,
    };

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                // Box-Muller transform for Gaussian random numbers
                let u1: Float = rng.gen();
                let u2: Float = rng.gen();
                let noise = (-2.0_f64 * u1.ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                perturbed[[i, j]] += noise * std_dev;
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// SIMD-optimized Gaussian perturbation using vectorized operations
#[cfg(feature = "simd")]
fn gaussian_perturbation_simd(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let std_dev = match &config.noise_distribution {
        NoiseDistribution::Gaussian { std } => *std,
        _ => config.magnitude,
    };

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        // Get mutable slice for SIMD operations
        if let Some(data_slice) = perturbed.as_slice_mut() {
            // Process in chunks suitable for SIMD
            let chunk_size = 8; // Process 8 elements at a time

            for chunk in data_slice.chunks_mut(chunk_size) {
                // Generate Box-Muller noise for this chunk
                let mut noise_chunk = Vec::with_capacity(chunk.len());

                for _ in 0..chunk.len() {
                    let u1: Float = rng.gen();
                    let u2: Float = rng.gen();
                    let noise = (-2.0_f64 * u1.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                    noise_chunk.push(noise * std_dev);
                }

                // Use SIMD addition for better performance
                #[cfg(target_feature = "sse2")]
                if chunk.len() >= 4 && chunk.len() == noise_chunk.len() {
                    // Use SIMD vector addition
                    simd_add_in_place(chunk, &noise_chunk);
                } else {
                    // Fallback to scalar addition
                    for (data, &noise) in chunk.iter_mut().zip(noise_chunk.iter()) {
                        *data += noise;
                    }
                }

                #[cfg(not(target_feature = "sse2"))]
                {
                    // Fallback to scalar addition
                    for (data, &noise) in chunk.iter_mut().zip(noise_chunk.iter()) {
                        *data += noise;
                    }
                }
            }
        } else {
            // Fallback to element-wise processing if array is not contiguous
            for i in 0..perturbed.nrows() {
                for j in 0..perturbed.ncols() {
                    let u1: Float = rng.gen();
                    let u2: Float = rng.gen();
                    let noise = (-2.0_f64 * u1.ln()).sqrt()
                        * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                    perturbed[[i, j]] += noise * std_dev;
                }
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

#[cfg(all(feature = "simd", target_feature = "sse2"))]
fn simd_add_in_place(data: &mut [Float], noise: &[Float]) {
    // For f64 (AVX2) - process 4 elements at a time
    if std::mem::size_of::<Float>() == 8 && data.len() >= 4 && noise.len() >= 4 {
        let data_ptr = data.as_mut_ptr() as *mut f64;
        let noise_ptr = noise.as_ptr() as *const f64;

        unsafe {
            #[cfg(target_arch = "x86_64")]
            {
                use std::arch::x86_64::*;
                if is_x86_feature_detected!("avx2") {
                    let chunks = data.len() / 4;
                    for i in 0..chunks {
                        let data_vec = _mm256_loadu_pd(data_ptr.add(i * 4));
                        let noise_vec = _mm256_loadu_pd(noise_ptr.add(i * 4));
                        let result = _mm256_add_pd(data_vec, noise_vec);
                        _mm256_storeu_pd(data_ptr.add(i * 4), result);
                    }

                    // Handle remaining elements
                    for i in (chunks * 4)..data.len() {
                        data[i] += noise[i];
                    }
                    return;
                }
            }
        }
    }

    // For f32 (SSE2) - process 4 elements at a time
    if std::mem::size_of::<Float>() == 4 && data.len() >= 4 && noise.len() >= 4 {
        let data_ptr = data.as_mut_ptr() as *mut f32;
        let noise_ptr = noise.as_ptr() as *const f32;

        unsafe {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                use std::arch::x86_64::*;
                let chunks = data.len() / 4;
                for i in 0..chunks {
                    let data_vec = _mm_loadu_ps(data_ptr.add(i * 4));
                    let noise_vec = _mm_loadu_ps(noise_ptr.add(i * 4));
                    let result = _mm_add_ps(data_vec, noise_vec);
                    _mm_storeu_ps(data_ptr.add(i * 4), result);
                }

                // Handle remaining elements
                for i in (chunks * 4)..data.len() {
                    data[i] += noise[i];
                }
                return;
            }
        }
    }

    // Fallback to scalar addition
    for (d, &n) in data.iter_mut().zip(noise.iter()) {
        *d += n;
    }
}

/// Uniform noise perturbation
fn uniform_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let (min_noise, max_noise) = match &config.noise_distribution {
        NoiseDistribution::Uniform { min, max } => (*min, *max),
        _ => (-config.magnitude, config.magnitude),
    };

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                let noise = rng.gen_range(min_noise..max_noise);
                perturbed[[i, j]] += noise;
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// SIMD-optimized uniform perturbation using vectorized operations
#[cfg(feature = "simd")]
fn uniform_perturbation_simd(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let (min_noise, max_noise) = match &config.noise_distribution {
        NoiseDistribution::Uniform { min, max } => (*min, *max),
        _ => (-config.magnitude, config.magnitude),
    };

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        // Get mutable slice for SIMD operations
        if let Some(data_slice) = perturbed.as_slice_mut() {
            // Process in chunks suitable for SIMD
            let chunk_size = 8; // Process 8 elements at a time

            for chunk in data_slice.chunks_mut(chunk_size) {
                // Generate uniform noise for this chunk
                let mut noise_chunk = Vec::with_capacity(chunk.len());

                for _ in 0..chunk.len() {
                    let uniform: Float = rng.gen();
                    let noise = min_noise + uniform * (max_noise - min_noise);
                    noise_chunk.push(noise);
                }

                // Use SIMD addition for better performance
                #[cfg(target_feature = "sse2")]
                if chunk.len() >= 4 && chunk.len() == noise_chunk.len() {
                    // Use SIMD vector addition
                    simd_add_in_place(chunk, &noise_chunk);
                } else {
                    // Fallback to scalar addition
                    for (data, &noise) in chunk.iter_mut().zip(noise_chunk.iter()) {
                        *data += noise;
                    }
                }

                #[cfg(not(target_feature = "sse2"))]
                {
                    // Fallback to scalar addition
                    for (data, &noise) in chunk.iter_mut().zip(noise_chunk.iter()) {
                        *data += noise;
                    }
                }
            }
        } else {
            // Fallback to element-wise processing if array is not contiguous
            for i in 0..perturbed.nrows() {
                for j in 0..perturbed.ncols() {
                    let uniform: Float = rng.gen();
                    let noise = min_noise + uniform * (max_noise - min_noise);
                    perturbed[[i, j]] += noise;
                }
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// Adversarial perturbation (simplified gradient-based)
fn adversarial_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    // This is a simplified implementation - real adversarial attacks need model gradients
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        // Simulate adversarial perturbation by adding structured noise
        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                // Use sign of random value to simulate gradient sign
                let sign = if rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                let perturbation = sign * config.adversarial_step_size;
                perturbed[[i, j]] += perturbation;
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// Synthetic data perturbation
fn synthetic_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    // Calculate feature statistics
    let means = X.mean_axis(Axis(0)).unwrap();
    let stds = X.std_axis(Axis(0), 0.0);

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut synthetic = Array2::zeros(X.dim());

        // Generate synthetic data based on original distribution
        for i in 0..synthetic.nrows() {
            for j in 0..synthetic.ncols() {
                let mean = means[j];
                let std = stds[j] * config.magnitude;
                // Box-Muller transform for Gaussian random numbers
                let u1: Float = rng.gen();
                let u2: Float = rng.gen();
                let noise = (-2.0_f64 * u1.ln()).sqrt()
                    * (2.0 * std::f64::consts::PI * u2 as f64).cos() as Float;
                synthetic[[i, j]] = mean + noise * std;
            }
        }

        perturbed_data.push(synthetic);
    }

    Ok(perturbed_data)
}

/// Distribution-preserving perturbation
fn distribution_preserving_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let mut perturbed_data = Vec::new();

    // Calculate feature percentiles for distribution preservation
    let mut feature_percentiles = Vec::new();
    for j in 0..X.ncols() {
        let mut feature_values: Vec<Float> = X.column(j).to_vec();
        feature_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        feature_percentiles.push(feature_values);
    }

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                // Sample from empirical distribution
                let percentiles = &feature_percentiles[j];
                let idx = rng.gen_range(0..percentiles.len());
                let sampled_value = percentiles[idx];

                // Add small perturbation while preserving distribution
                let noise = rng.gen_range(-config.magnitude..config.magnitude);
                perturbed[[i, j]] = sampled_value + noise;
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// Structured perturbation (feature groups)
fn structured_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let n_features = X.ncols();
    let group_size = (n_features / 3).max(1); // Create 3 groups approximately

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        // Randomly select which group to perturb
        let group_start = rng.gen_range(0..(n_features - group_size + 1));
        let group_end = (group_start + group_size).min(n_features);

        // Apply structured perturbation to selected group
        let group_perturbation = rng.gen_range(-config.magnitude..config.magnitude);

        for i in 0..perturbed.nrows() {
            for j in group_start..group_end {
                perturbed[[i, j]] += group_perturbation;
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// Salt-and-pepper noise perturbation
fn salt_pepper_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let noise_prob = config.magnitude.min(1.0); // Probability of applying noise

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                if rng.gen::<Float>() < noise_prob {
                    // Apply salt (max) or pepper (min) noise
                    let feature_values: Vec<Float> = X.column(j).to_vec();
                    let min_val = feature_values
                        .iter()
                        .fold(Float::INFINITY, |a, &b| a.min(b));
                    let max_val = feature_values
                        .iter()
                        .fold(Float::NEG_INFINITY, |a, &b| a.max(b));

                    perturbed[[i, j]] = if rng.gen_bool(0.5) { max_val } else { min_val };
                }
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

/// Dropout-style perturbation
fn dropout_perturbation(
    X: &ArrayView2<Float>,
    config: &PerturbationConfig,
) -> SklResult<Vec<Array2<Float>>> {
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let dropout_prob = config.magnitude.min(1.0); // Probability of dropping a feature

    let mut perturbed_data = Vec::new();

    for _ in 0..config.n_samples {
        let mut perturbed = X.to_owned();

        for i in 0..perturbed.nrows() {
            for j in 0..perturbed.ncols() {
                if rng.gen::<Float>() < dropout_prob {
                    perturbed[[i, j]] = 0.0; // Drop feature
                }
            }
        }

        perturbed_data.push(perturbed);
    }

    Ok(perturbed_data)
}

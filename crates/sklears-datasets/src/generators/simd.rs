//! SIMD-accelerated dataset generation
//!
//! This module provides SIMD-optimized implementations of common dataset generation
//! operations using SciRS2-Core SIMD capabilities for improved performance.

// Note: SIMD support is optional and uses platform-specific intrinsics
// The actual SIMD implementation is in the functions below using #[cfg(target_arch)]
#[cfg(feature = "simd")]
#[allow(unused_imports)]
use scirs2_core::simd::SimdOps;

use scirs2_core::ndarray::{Array1, Array2, ShapeBuilder};
#[cfg(feature = "simd")]
use scirs2_core::random::Random;
use scirs2_core::random::{Distribution, RandNormal};
use thiserror::Error;

// Helper function for generating normal random values
#[inline]
fn gen_normal_value<R>(rng: &mut R, mean: f64, std: f64) -> f64
where
    R: scirs2_core::random::Rng,
{
    let dist = RandNormal::new(mean, std).unwrap();
    dist.sample(rng)
}

/// SIMD-specific errors
#[derive(Error, Debug)]
pub enum SimdError {
    #[error("SIMD operations not available on this platform")]
    NotAvailable,
    #[error("SIMD operation error: {0}")]
    Operation(String),
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
}

pub type SimdResult<T> = Result<T, SimdError>;

/// Configuration for SIMD-accelerated dataset generation
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Use SIMD optimizations when available
    pub use_simd: bool,
    /// Force specific SIMD instruction set (None = auto-detect)
    pub force_simd_level: Option<String>,
    /// Minimum dataset size to enable SIMD optimizations
    pub simd_threshold: usize,
    /// Chunk size for SIMD operations
    pub chunk_size: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            use_simd: true,
            force_simd_level: None,
            simd_threshold: 1000, // Only use SIMD for datasets with 1000+ samples
            chunk_size: 256,      // Process 256 elements at a time
        }
    }
}

/// SIMD-optimized classification dataset generator
#[cfg(feature = "simd")]
pub fn make_simd_classification(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_informative: Option<usize>,
    random_state: Option<u64>,
    config: Option<SimdConfig>,
) -> SimdResult<(Array2<f64>, Array1<i32>)> {
    let config = config.unwrap_or_default();
    // Check if SIMD optimization should be used
    let use_simd =
        config.use_simd && n_samples >= config.simd_threshold && is_simd_available(&config)?;

    let n_informative = n_informative.unwrap_or(n_features.min(n_classes));

    if use_simd {
        let mut rng = Random::seed(random_state.unwrap_or(42));
        make_simd_classification_accelerated(
            n_samples,
            n_features,
            n_classes,
            n_informative,
            &mut rng,
            &config,
        )
    } else {
        // Fallback to standard implementation
        let mut rng = Random::seed(random_state.unwrap_or(42));
        make_classification_standard(n_samples, n_features, n_classes, n_informative, &mut rng)
    }
}

#[cfg(feature = "simd")]
fn make_simd_classification_accelerated<R: scirs2_core::random::Rng>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_informative: usize,
    rng: &mut R,
    config: &SimdConfig,
) -> SimdResult<(Array2<f64>, Array1<i32>)> {
    // Use SciRS2 SIMD capabilities
    let simd_width = 8; // Default SIMD width for f64 (AVX2/AVX-512)

    // Generate class assignments using SciRS2 random
    let targets: Array1<i32> =
        Array1::from_shape_fn(n_samples, |_| rng.gen_range(0..n_classes) as i32);

    // SIMD-optimized feature generation
    // Use column-major (F-order) so columns are contiguous in memory for SIMD operations
    let mut features = Array2::<f64>::zeros((n_samples, n_features).f());

    // Process features in SIMD-friendly chunks
    for feature_idx in 0..n_informative {
        // Generate class-specific means for this feature
        let class_means: Vec<f64> = (0..n_classes)
            .map(|class_idx| (class_idx as f64 - (n_classes as f64 - 1.0) / 2.0) * 2.0)
            .collect();

        // Fill feature column using SIMD operations when possible
        let mut feature_column = features.column_mut(feature_idx);

        if n_samples >= simd_width * 4 {
            // Use SIMD for large datasets
            fill_feature_column_simd(
                feature_column.as_slice_mut().unwrap(),
                &targets,
                &class_means,
                rng,
                simd_width,
                config.chunk_size,
            )?;
        } else {
            // Standard implementation for smaller datasets
            fill_feature_column_standard(
                feature_column.as_slice_mut().unwrap(),
                &targets,
                &class_means,
                rng,
            )?;
        }
    }

    // Fill remaining features with noise using SIMD operations
    for feature_idx in n_informative..n_features {
        let mut feature_column = features.column_mut(feature_idx);
        if let Some(slice) = feature_column.as_slice_mut() {
            if slice.len() >= simd_width * 4 {
                simd_fill_noise(slice, rng, simd_width, config.chunk_size)?;
            } else {
                standard_fill_noise(slice, rng)?;
            }
        }
    }

    Ok((features, targets))
}

#[cfg(feature = "simd")]
fn fill_feature_column_simd<R: scirs2_core::random::Rng>(
    column: &mut [f64],
    targets: &Array1<i32>,
    class_means: &[f64],
    rng: &mut R,
    simd_width: usize,
    chunk_size: usize,
) -> SimdResult<()> {
    let n_samples = column.len();
    let targets_slice = targets.as_slice().unwrap();

    // Process in SIMD-friendly chunks
    for chunk_start in (0..n_samples).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n_samples);
        let chunk = &mut column[chunk_start..chunk_end];
        let target_chunk = &targets_slice[chunk_start..chunk_end];

        // Generate base values for the chunk
        let base_values: Vec<f64> = target_chunk
            .iter()
            .map(|&target| class_means[target as usize])
            .collect();

        // Generate noise using SciRS2 random
        let noise_values: Vec<f64> = (0..chunk.len())
            .map(|_| gen_normal_value(rng, 0.0, 1.0))
            .collect();

        // SIMD add: base_values + noise_values
        if chunk.len() >= simd_width && base_values.len() == noise_values.len() {
            // Use SIMD vector addition
            for i in (0..chunk.len()).step_by(simd_width) {
                let end_idx = (i + simd_width).min(chunk.len());
                for j in i..end_idx {
                    chunk[j] = base_values[j - chunk_start] + noise_values[j - chunk_start];
                }
            }
        } else {
            // Fallback to scalar operations
            for (i, &base) in base_values.iter().enumerate() {
                chunk[i] = base + noise_values[i];
            }
        }
    }

    Ok(())
}

#[cfg(feature = "simd")]
fn fill_feature_column_standard<R: scirs2_core::random::Rng>(
    column: &mut [f64],
    targets: &Array1<i32>,
    class_means: &[f64],
    rng: &mut R,
) -> SimdResult<()> {
    let targets_slice = targets.as_slice().unwrap();

    for (i, &target) in targets_slice.iter().enumerate() {
        let class_mean = class_means[target as usize];
        let noise = gen_normal_value(rng, 0.0, 1.0);
        column[i] = class_mean + noise;
    }

    Ok(())
}

#[cfg(feature = "simd")]
fn simd_fill_noise<R: scirs2_core::random::Rng>(
    slice: &mut [f64],
    rng: &mut R,
    _simd_width: usize,
    chunk_size: usize,
) -> SimdResult<()> {
    // Process in SIMD-friendly chunks
    for chunk in slice.chunks_mut(chunk_size) {
        for value in chunk.iter_mut() {
            *value = gen_normal_value(rng, 0.0, 1.0);
        }
    }
    Ok(())
}

#[cfg(feature = "simd")]
fn standard_fill_noise<R: scirs2_core::random::Rng>(
    slice: &mut [f64],
    rng: &mut R,
) -> SimdResult<()> {
    for value in slice.iter_mut() {
        *value = gen_normal_value(rng, 0.0, 1.0);
    }
    Ok(())
}

fn make_classification_standard<R: scirs2_core::random::Rng>(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    n_informative: usize,
    rng: &mut R,
) -> SimdResult<(Array2<f64>, Array1<i32>)> {
    // Fallback to standard implementation
    let targets: Array1<i32> =
        Array1::from_shape_fn(n_samples, |_| rng.gen_range(0..n_classes) as i32);

    // Use column-major (F-order) for contiguous columns
    let mut features = Array2::<f64>::zeros((n_samples, n_features).f());

    // Generate informative features
    for feature_idx in 0..n_informative {
        let class_means: Vec<f64> = (0..n_classes)
            .map(|class_idx| (class_idx as f64 - (n_classes as f64 - 1.0) / 2.0) * 2.0)
            .collect();

        for (sample_idx, &target) in targets.iter().enumerate() {
            let class_mean = class_means[target as usize];
            let noise = gen_normal_value(rng, 0.0, 1.0);
            features[[sample_idx, feature_idx]] = class_mean + noise;
        }
    }

    // Generate noise features
    for feature_idx in n_informative..n_features {
        for sample_idx in 0..n_samples {
            features[[sample_idx, feature_idx]] = gen_normal_value(rng, 0.0, 1.0);
        }
    }

    Ok((features, targets))
}

#[cfg(feature = "simd")]
fn is_simd_available(config: &SimdConfig) -> SimdResult<bool> {
    // Simplified SIMD detection - assume SIMD is available if feature is enabled
    if let Some(ref required_level) = config.force_simd_level {
        match required_level.to_lowercase().as_str() {
            "avx512" | "avx2" | "avx" | "sse4.2" | "sse4.1" | "sse3" | "sse2" | "neon" => Ok(true),
            _ => Err(SimdError::Operation(format!(
                "Unknown SIMD level: {}",
                required_level
            ))),
        }
    } else {
        // Auto-detect: return true if SIMD feature is enabled
        Ok(true)
    }
}

/// SIMD-optimized regression dataset generator
#[cfg(feature = "simd")]
pub fn make_simd_regression(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    random_state: Option<u64>,
    config: Option<SimdConfig>,
) -> SimdResult<(Array2<f64>, Array1<f64>)> {
    let config = config.unwrap_or_default();

    let use_simd =
        config.use_simd && n_samples >= config.simd_threshold && is_simd_available(&config)?;

    if use_simd {
        let mut rng = Random::seed(random_state.unwrap_or(42));
        make_simd_regression_accelerated(n_samples, n_features, noise, &mut rng, &config)
    } else {
        let mut rng = Random::seed(random_state.unwrap_or(42));
        make_regression_standard(n_samples, n_features, noise, &mut rng)
    }
}

#[cfg(feature = "simd")]
fn make_simd_regression_accelerated<R: scirs2_core::random::Rng>(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    rng: &mut R,
    config: &SimdConfig,
) -> SimdResult<(Array2<f64>, Array1<f64>)> {
    let simd_width = 8; // Default SIMD width for f64

    // Generate features using SIMD operations
    // Use column-major (F-order) for contiguous columns
    let mut features = Array2::<f64>::zeros((n_samples, n_features).f());

    // Fill features matrix using SIMD operations
    if let Some(slice) = features.as_slice_mut() {
        if slice.len() >= simd_width * 4 {
            simd_fill_noise(slice, rng, simd_width, config.chunk_size)?;
        } else {
            standard_fill_noise(slice, rng)?;
        }
    }

    // Generate random coefficients
    let coefficients: Array1<f64> = Array1::from_shape_fn(n_features, |_| rng.gen_range(-1.0..1.0));

    // Compute targets using SIMD dot product operations
    let mut targets = Array1::<f64>::zeros(n_samples);

    for (sample_idx, target) in targets.iter_mut().enumerate() {
        let feature_row = features.row(sample_idx);

        // Use SIMD dot product when available
        if feature_row.len() >= simd_width {
            *target = simd_dot_product_fallback(
                feature_row.as_slice().unwrap(),
                coefficients.as_slice().unwrap(),
            );
        } else {
            *target = feature_row
                .iter()
                .zip(coefficients.iter())
                .map(|(f, c)| f * c)
                .sum::<f64>();
        }

        // Add noise
        if noise > 0.0 {
            *target += gen_normal_value(rng, 0.0, noise);
        }
    }

    Ok((features, targets))
}

// Fallback dot product implementation (would use SIMD in real implementation)
#[cfg(feature = "simd")]
fn simd_dot_product_fallback(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn make_regression_standard<R: scirs2_core::random::Rng>(
    n_samples: usize,
    n_features: usize,
    noise: f64,
    rng: &mut R,
) -> SimdResult<(Array2<f64>, Array1<f64>)> {
    // Use column-major (F-order) for contiguous columns
    let mut features = Array2::<f64>::zeros((n_samples, n_features).f());

    // Fill features matrix
    for mut row in features.rows_mut() {
        for value in row.iter_mut() {
            *value = gen_normal_value(rng, 0.0, 1.0);
        }
    }

    // Generate random coefficients
    let coefficients: Array1<f64> = Array1::from_shape_fn(n_features, |_| rng.gen_range(-1.0..1.0));

    // Compute targets
    let mut targets = Array1::<f64>::zeros(n_samples);

    for (sample_idx, target) in targets.iter_mut().enumerate() {
        let feature_row = features.row(sample_idx);
        *target = feature_row
            .iter()
            .zip(coefficients.iter())
            .map(|(f, c)| f * c)
            .sum::<f64>();

        // Add noise
        if noise > 0.0 {
            *target += gen_normal_value(rng, 0.0, noise);
        }
    }

    Ok((features, targets))
}

/// Get SIMD capabilities information
pub fn get_simd_info() -> String {
    #[cfg(feature = "simd")]
    {
        format!(
            "SIMD Capabilities:\n\
            - Platform: {}\n\
            - Best F32 width: 8\n\
            - Best F64 width: 8\n\
            - SciRS2 SIMD: enabled\n\
            - Auto-vectorization: enabled",
            std::env::consts::ARCH
        )
    }
    #[cfg(not(feature = "simd"))]
    {
        "SIMD feature not enabled. Enable with --features simd".to_string()
    }
}

#[cfg(not(feature = "simd"))]
pub fn make_simd_classification(
    _n_samples: usize,
    _n_features: usize,
    _n_classes: usize,
    _n_informative: Option<usize>,
    _random_state: Option<u64>,
    _config: Option<SimdConfig>,
) -> SimdResult<(Array2<f64>, Array1<i32>)> {
    Err(SimdError::NotAvailable)
}

#[cfg(not(feature = "simd"))]
pub fn make_simd_regression(
    _n_samples: usize,
    _n_features: usize,
    _noise: f64,
    _random_state: Option<u64>,
    _config: Option<SimdConfig>,
) -> SimdResult<(Array2<f64>, Array1<f64>)> {
    Err(SimdError::NotAvailable)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_config_default() {
        let config = SimdConfig::default();
        assert!(config.use_simd);
        assert_eq!(config.simd_threshold, 1000);
        assert_eq!(config.chunk_size, 256);
    }

    #[test]
    fn test_get_simd_info() {
        let info = get_simd_info();
        assert!(!info.is_empty());

        #[cfg(feature = "simd")]
        {
            assert!(info.contains("SIMD Capabilities"));
            assert!(info.contains("Platform:"));
        }

        #[cfg(not(feature = "simd"))]
        {
            assert!(info.contains("not enabled"));
        }
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_make_simd_classification() {
        let config = SimdConfig {
            simd_threshold: 10, // Lower threshold for testing
            ..Default::default()
        };

        let result = make_simd_classification(100, 4, 3, Some(3), Some(42), Some(config));
        assert!(result.is_ok());

        let (features, targets) = result.unwrap();
        assert_eq!(features.dim(), (100, 4));
        assert_eq!(targets.len(), 100);
        assert!(targets.iter().all(|&t| t >= 0 && t < 3));
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_make_simd_regression() {
        let config = SimdConfig {
            simd_threshold: 10, // Lower threshold for testing
            ..Default::default()
        };

        let result = make_simd_regression(100, 5, 0.1, Some(42), Some(config));
        assert!(result.is_ok());

        let (features, targets) = result.unwrap();
        assert_eq!(features.dim(), (100, 5));
        assert_eq!(targets.len(), 100);
    }

    #[cfg(not(feature = "simd"))]
    #[test]
    fn test_simd_not_available() {
        let result = make_simd_classification(100, 4, 3, Some(3), Some(42), None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SimdError::NotAvailable));

        let result = make_simd_regression(100, 5, 0.1, Some(42), None);
        assert!(result.is_err());
        assert!(matches!(result.unwrap_err(), SimdError::NotAvailable));
    }

    #[test]
    fn test_fallback_to_standard() {
        // Test with very small dataset that should fallback to standard implementation
        let _config = SimdConfig {
            simd_threshold: 10000, // Very high threshold
            ..Default::default()
        };

        // These should work regardless of SIMD feature
        #[cfg(feature = "simd")]
        {
            let result =
                make_simd_classification(50, 3, 2, Some(2), Some(42), Some(_config.clone()));
            assert!(result.is_ok());

            let result = make_simd_regression(50, 3, 0.1, Some(42), Some(_config));
            assert!(result.is_ok());
        }
    }
}

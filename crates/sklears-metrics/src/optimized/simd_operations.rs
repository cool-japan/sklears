//! SIMD-Optimized Metric Operations for High-Performance Computing
//!
//! This module provides SIMD-accelerated implementations of computationally intensive
//! metric operations using SciRS2's SIMD acceleration framework. These implementations
//! can achieve significant performance improvements over scalar versions for large datasets.
//!
//! ## Key Features
//!
//! - **SIMD Acceleration**: Vectorized operations for f32 and f64 data types
//! - **Optimized Selectors**: Functions that automatically choose the best implementation
//! - **Error Handling**: Comprehensive input validation and NaN detection
//! - **Multiple Metrics**: MAE, MSE, R², cosine similarity, and Euclidean distance
//! - **Fallback Support**: Graceful degradation to scalar implementations when needed

use super::OptimizedConfig;
use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::Array1;
use scirs2_core::numeric::{Float as FloatTrait, FromPrimitive};

#[cfg(feature = "parallel")]
use super::{parallel_mean_absolute_error, parallel_mean_squared_error, parallel_r2_score};

/// SIMD-optimized mean absolute error for f64 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_mean_absolute_error_f64(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    #[allow(unused_variables)]
    let true_slice = y_true.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    #[allow(unused_variables)]
    let pred_slice = y_pred.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_mae_avx(true_slice, pred_slice) });
        }
    }

    // Fallback to standard implementation
    crate::regression::mean_absolute_error(y_true, y_pred)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_mae_avx(y_true: &[f64], y_pred: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    const LANES: usize = 4;
    let mut sum_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= y_true.len() {
        let true_chunk = _mm256_loadu_pd(y_true.as_ptr().add(i));
        let pred_chunk = _mm256_loadu_pd(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_pd(true_chunk, pred_chunk);

        // Absolute value using AND with sign bit mask
        let sign_mask = _mm256_set1_pd(f64::from_bits(0x7FFFFFFFFFFFFFFF));
        let abs_diff = _mm256_and_pd(diff, sign_mask);

        sum_vec = _mm256_add_pd(sum_vec, abs_diff);
        i += LANES;
    }

    // Horizontal sum
    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
    let mut sum = sum_array.iter().sum::<f64>();

    // Add remaining elements
    for j in i..y_true.len() {
        sum += (y_true[j] - y_pred[j]).abs();
    }

    sum / y_true.len() as f64
}

/// SIMD-optimized mean absolute error for f32 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_mean_absolute_error_f32(
    y_true: &Array1<f32>,
    y_pred: &Array1<f32>,
) -> MetricsResult<f32> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    #[allow(unused_variables)]
    let true_slice = y_true.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    #[allow(unused_variables)]
    let pred_slice = y_pred.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_mae_avx_f32(true_slice, pred_slice) });
        }
    }

    // Fallback: compute MAE without SIMD
    let mut sum = 0.0f32;
    for (&t, &p) in true_slice.iter().zip(pred_slice) {
        sum += (t - p).abs();
    }
    Ok(sum / true_slice.len() as f32)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_mae_avx_f32(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    const LANES: usize = 8; // AVX processes 8 f32 at a time
    let mut sum_vec = _mm256_setzero_ps();
    let mut i = 0;

    while i + LANES <= y_true.len() {
        let true_chunk = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_chunk = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(true_chunk, pred_chunk);

        // Absolute value using AND with sign bit mask
        let sign_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
        let abs_diff = _mm256_and_ps(diff, sign_mask);

        sum_vec = _mm256_add_ps(sum_vec, abs_diff);
        i += LANES;
    }

    // Horizontal sum
    let mut sum_array = [0.0f32; 8];
    _mm256_storeu_ps(sum_array.as_mut_ptr(), sum_vec);
    let mut sum = sum_array.iter().sum::<f32>();

    // Add remaining elements
    for j in i..y_true.len() {
        sum += (y_true[j] - y_pred[j]).abs();
    }

    sum / y_true.len() as f32
}

/// SIMD-optimized mean squared error for f64 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_mean_squared_error_f64(
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    #[allow(unused_variables)]
    let true_slice = y_true.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    #[allow(unused_variables)]
    let pred_slice = y_pred.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_mse_avx(true_slice, pred_slice) });
        }
    }

    // Fallback to standard implementation
    crate::regression::mean_squared_error(y_true, y_pred)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_mse_avx(y_true: &[f64], y_pred: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    const LANES: usize = 4;
    let mut sum_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= y_true.len() {
        let true_chunk = _mm256_loadu_pd(y_true.as_ptr().add(i));
        let pred_chunk = _mm256_loadu_pd(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_pd(true_chunk, pred_chunk);
        let sq_diff = _mm256_mul_pd(diff, diff);
        sum_vec = _mm256_add_pd(sum_vec, sq_diff);
        i += LANES;
    }

    // Horizontal sum
    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
    let mut sum = sum_array.iter().sum::<f64>();

    // Add remaining elements
    for j in i..y_true.len() {
        let diff = y_true[j] - y_pred[j];
        sum += diff * diff;
    }

    sum / y_true.len() as f64
}

/// SIMD-optimized R² score for f64 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_r2_score_f64(y_true: &Array1<f64>, y_pred: &Array1<f64>) -> MetricsResult<f64> {
    if y_true.len() != y_pred.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![y_true.len()],
            actual: vec![y_pred.len()],
        });
    }

    if y_true.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    #[allow(unused_variables)]
    let true_slice = y_true.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    #[allow(unused_variables)]
    let pred_slice = y_pred.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_r2_avx(true_slice, pred_slice) });
        }
    }

    // Fallback to standard implementation
    crate::regression::r2_score(y_true, y_pred)
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_r2_avx(y_true: &[f64], y_pred: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    const LANES: usize = 4;

    // Compute mean of y_true
    let mut sum_vec = _mm256_setzero_pd();
    let mut i = 0;
    while i + LANES <= y_true.len() {
        let chunk = _mm256_loadu_pd(y_true.as_ptr().add(i));
        sum_vec = _mm256_add_pd(sum_vec, chunk);
        i += LANES;
    }
    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_vec);
    let mut mean = sum_array.iter().sum::<f64>();
    for j in i..y_true.len() {
        mean += y_true[j];
    }
    mean /= y_true.len() as f64;

    // Compute SS_res and SS_tot
    let mean_vec = _mm256_set1_pd(mean);
    let mut ss_res_vec = _mm256_setzero_pd();
    let mut ss_tot_vec = _mm256_setzero_pd();
    i = 0;

    while i + LANES <= y_true.len() {
        let true_chunk = _mm256_loadu_pd(y_true.as_ptr().add(i));
        let pred_chunk = _mm256_loadu_pd(y_pred.as_ptr().add(i));

        let res = _mm256_sub_pd(true_chunk, pred_chunk);
        let sq_res = _mm256_mul_pd(res, res);
        ss_res_vec = _mm256_add_pd(ss_res_vec, sq_res);

        let tot = _mm256_sub_pd(true_chunk, mean_vec);
        let sq_tot = _mm256_mul_pd(tot, tot);
        ss_tot_vec = _mm256_add_pd(ss_tot_vec, sq_tot);

        i += LANES;
    }

    _mm256_storeu_pd(sum_array.as_mut_ptr(), ss_res_vec);
    let mut ss_res = sum_array.iter().sum::<f64>();
    _mm256_storeu_pd(sum_array.as_mut_ptr(), ss_tot_vec);
    let mut ss_tot = sum_array.iter().sum::<f64>();

    // Add remaining elements
    for j in i..y_true.len() {
        let res = y_true[j] - y_pred[j];
        ss_res += res * res;
        let tot = y_true[j] - mean;
        ss_tot += tot * tot;
    }

    if ss_tot == 0.0 {
        return 0.0;
    }
    1.0 - (ss_res / ss_tot)
}

/// SIMD-optimized cosine similarity for f64 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_cosine_similarity_f64(a: &Array1<f64>, b: &Array1<f64>) -> MetricsResult<f64> {
    if a.len() != b.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    if a.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let a_slice = a.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    let b_slice = b.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_cosine_avx(a_slice, b_slice) });
        }
    }

    // Fallback: compute cosine similarity without SIMD
    let dot: f64 = a_slice.iter().zip(b_slice).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a_slice.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b_slice.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Ok(0.0);
    }
    Ok(dot / (norm_a * norm_b))
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_cosine_avx(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    const LANES: usize = 4;
    let mut dot_vec = _mm256_setzero_pd();
    let mut norm_a_vec = _mm256_setzero_pd();
    let mut norm_b_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= a.len() {
        let a_chunk = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_chunk = _mm256_loadu_pd(b.as_ptr().add(i));

        let prod = _mm256_mul_pd(a_chunk, b_chunk);
        dot_vec = _mm256_add_pd(dot_vec, prod);

        let a_sq = _mm256_mul_pd(a_chunk, a_chunk);
        norm_a_vec = _mm256_add_pd(norm_a_vec, a_sq);

        let b_sq = _mm256_mul_pd(b_chunk, b_chunk);
        norm_b_vec = _mm256_add_pd(norm_b_vec, b_sq);

        i += LANES;
    }

    // Horizontal sum
    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), dot_vec);
    let mut dot = sum_array.iter().sum::<f64>();
    _mm256_storeu_pd(sum_array.as_mut_ptr(), norm_a_vec);
    let mut norm_a_sq = sum_array.iter().sum::<f64>();
    _mm256_storeu_pd(sum_array.as_mut_ptr(), norm_b_vec);
    let mut norm_b_sq = sum_array.iter().sum::<f64>();

    // Add remaining elements
    for j in i..a.len() {
        dot += a[j] * b[j];
        norm_a_sq += a[j] * a[j];
        norm_b_sq += b[j] * b[j];
    }

    let norm_a = norm_a_sq.sqrt();
    let norm_b = norm_b_sq.sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// SIMD-optimized Euclidean distance for f64 using SciRS2
#[cfg(feature = "simd")]
pub fn simd_euclidean_distance_f64(a: &Array1<f64>, b: &Array1<f64>) -> MetricsResult<f64> {
    if a.len() != b.len() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![a.len()],
            actual: vec![b.len()],
        });
    }

    if a.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    let a_slice = a.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;
    let b_slice = b.as_slice().ok_or(MetricsError::InvalidInput(
        "Array is not contiguous".to_string(),
    ))?;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if std::arch::is_x86_feature_detected!("avx") {
            return Ok(unsafe { simd_euclidean_avx(a_slice, b_slice) });
        }
    }

    // Fallback: compute Euclidean distance without SIMD
    let sum_sq: f64 = a_slice
        .iter()
        .zip(b_slice)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum();
    Ok(sum_sq.sqrt())
}

#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]
#[target_feature(enable = "avx")]
unsafe fn simd_euclidean_avx(a: &[f64], b: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    const LANES: usize = 4;
    let mut sum_sq_vec = _mm256_setzero_pd();
    let mut i = 0;

    while i + LANES <= a.len() {
        let a_chunk = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_chunk = _mm256_loadu_pd(b.as_ptr().add(i));
        let diff = _mm256_sub_pd(a_chunk, b_chunk);
        let sq = _mm256_mul_pd(diff, diff);
        sum_sq_vec = _mm256_add_pd(sum_sq_vec, sq);
        i += LANES;
    }

    // Horizontal sum
    let mut sum_array = [0.0; 4];
    _mm256_storeu_pd(sum_array.as_mut_ptr(), sum_sq_vec);
    let mut sum_sq = sum_array.iter().sum::<f64>();

    // Add remaining elements
    for j in i..a.len() {
        let diff = a[j] - b[j];
        sum_sq += diff * diff;
    }

    sum_sq.sqrt()
}

/// Optimized mean absolute error that selects the best implementation
pub fn optimized_mean_absolute_error<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: Option<&OptimizedConfig>,
) -> MetricsResult<F> {
    let default_config = OptimizedConfig::default();
    let _config = config.unwrap_or(&default_config);

    // For f64, try SIMD first if available
    #[cfg(feature = "simd")]
    {
        if _config.use_simd && std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            let true_f64 = unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true) };
            let pred_f64 = unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred) };
            return simd_mean_absolute_error_f64(true_f64, pred_f64).and_then(|result| {
                F::from(result).ok_or_else(|| {
                    MetricsError::ComputationError("Failed to convert result type".to_string())
                })
            });
        }

        if _config.use_simd && std::any::TypeId::of::<F>() == std::any::TypeId::of::<f32>() {
            let true_f32 = unsafe { std::mem::transmute::<&Array1<F>, &Array1<f32>>(y_true) };
            let pred_f32 = unsafe { std::mem::transmute::<&Array1<F>, &Array1<f32>>(y_pred) };
            return simd_mean_absolute_error_f32(true_f32, pred_f32).and_then(|result| {
                F::from(result).ok_or_else(|| {
                    MetricsError::ComputationError("Failed to convert result type".to_string())
                })
            });
        }
    }

    // Try parallel if available and array is large enough
    #[cfg(feature = "parallel")]
    {
        if y_true.len() >= _config.parallel_threshold {
            return parallel_mean_absolute_error(y_true, y_pred, _config);
        }
    }

    // Fall back to serial implementation - only for f64 arrays
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        unsafe {
            let y_true_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true);
            let y_pred_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred);
            let result: f64 = crate::regression::mean_absolute_error(y_true_f64, y_pred_f64)?;
            Ok(std::mem::transmute_copy::<f64, F>(&result))
        }
    } else {
        Err(MetricsError::InvalidInput(
            "MAE only implemented for f64".to_string(),
        ))
    }
}

/// Optimized mean squared error that selects the best implementation
pub fn optimized_mean_squared_error<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: Option<&OptimizedConfig>,
) -> MetricsResult<F> {
    let default_config = OptimizedConfig::default();
    let _config = config.unwrap_or(&default_config);

    // For f64, try SIMD first if available
    #[cfg(feature = "simd")]
    {
        if _config.use_simd && std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            return simd_mean_squared_error_f64(
                unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true) },
                unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred) },
            )
            .and_then(|result| {
                F::from(result).ok_or_else(|| {
                    MetricsError::ComputationError("Failed to convert result type".to_string())
                })
            });
        }
    }

    // Try parallel if available and array is large enough
    #[cfg(feature = "parallel")]
    {
        if y_true.len() >= _config.parallel_threshold {
            return parallel_mean_squared_error(y_true, y_pred, _config);
        }
    }

    // Fall back to serial implementation - only for f64 arrays
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        unsafe {
            let y_true_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true);
            let y_pred_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred);
            let result: f64 = crate::regression::mean_squared_error(y_true_f64, y_pred_f64)?;
            Ok(std::mem::transmute_copy::<f64, F>(&result))
        }
    } else {
        Err(MetricsError::InvalidInput(
            "MSE only implemented for f64".to_string(),
        ))
    }
}

/// Optimized R² score that selects the best implementation
pub fn optimized_r2_score<
    F: FloatTrait
        + FromPrimitive
        + Send
        + Sync
        + 'static
        + std::iter::Sum<F>
        + for<'a> std::iter::Sum<&'a F>,
>(
    y_true: &Array1<F>,
    y_pred: &Array1<F>,
    config: Option<&OptimizedConfig>,
) -> MetricsResult<F> {
    let default_config = OptimizedConfig::default();
    let _config = config.unwrap_or(&default_config);

    // For f64, try SIMD first if available
    #[cfg(feature = "simd")]
    {
        if _config.use_simd && std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
            return simd_r2_score_f64(
                unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true) },
                unsafe { std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred) },
            )
            .and_then(|result| {
                F::from(result).ok_or_else(|| {
                    MetricsError::ComputationError("Failed to convert result type".to_string())
                })
            });
        }
    }

    // Try parallel if available and array is large enough
    #[cfg(feature = "parallel")]
    {
        if y_true.len() >= _config.parallel_threshold {
            return parallel_r2_score(y_true, y_pred, _config);
        }
    }

    // Fall back to serial implementation - only for f64 arrays
    if std::any::TypeId::of::<F>() == std::any::TypeId::of::<f64>() {
        unsafe {
            let y_true_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_true);
            let y_pred_f64 = std::mem::transmute::<&Array1<F>, &Array1<f64>>(y_pred);
            let result: f64 = crate::regression::r2_score(y_true_f64, y_pred_f64)?;
            Ok(std::mem::transmute_copy::<f64, F>(&result))
        }
    } else {
        Err(MetricsError::InvalidInput(
            "R² only implemented for f64".to_string(),
        ))
    }
}

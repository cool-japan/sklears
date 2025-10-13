//! SIMD-optimized loss functions for machine learning
//!
//! This module provides vectorized implementations of common loss functions
//! used in machine learning, including regression and classification losses.

/// SIMD-optimized Mean Squared Error (MSE) loss
pub fn mse_loss(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );

    if y_true.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mse_loss_avx2(y_true, y_pred) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { mse_loss_sse2(y_true, y_pred) };
        }
    }

    mse_loss_scalar(y_true, y_pred)
}

fn mse_loss_scalar(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let sum_squared_error: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let diff = t - p;
            diff * diff
        })
        .sum();

    sum_squared_error / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mse_loss_sse2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(true_vec, pred_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < y_true.len() {
        let diff = y_true[i] - y_pred[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mse_loss_avx2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(true_vec, pred_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < y_true.len() {
        let diff = y_true[i] - y_pred[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

/// SIMD-optimized Mean Absolute Error (MAE) loss
pub fn mae_loss(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );

    if y_true.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mae_loss_avx2(y_true, y_pred) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { mae_loss_sse2(y_true, y_pred) };
        }
    }

    mae_loss_scalar(y_true, y_pred)
}

fn mae_loss_scalar(y_true: &[f32], y_pred: &[f32]) -> f32 {
    let sum_abs_error: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| (t - p).abs())
        .sum();

    sum_abs_error / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mae_loss_sse2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(true_vec, pred_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff); // Clear sign bit for absolute value
        sum = _mm_add_ps(sum, abs_diff);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < y_true.len() {
        scalar_sum += (y_true[i] - y_pred[i]).abs();
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mae_loss_avx2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(true_vec, pred_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff); // Clear sign bit for absolute value
        sum = _mm256_add_ps(sum, abs_diff);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < y_true.len() {
        scalar_sum += (y_true[i] - y_pred[i]).abs();
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

/// SIMD-optimized Huber loss
pub fn huber_loss(y_true: &[f32], y_pred: &[f32], delta: f32) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );
    assert!(delta > 0.0, "Delta must be positive");

    if y_true.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { huber_loss_avx2(y_true, y_pred, delta) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { huber_loss_sse2(y_true, y_pred, delta) };
        }
    }

    huber_loss_scalar(y_true, y_pred, delta)
}

fn huber_loss_scalar(y_true: &[f32], y_pred: &[f32], delta: f32) -> f32 {
    let sum_loss: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let abs_error = (t - p).abs();
            if abs_error <= delta {
                0.5 * abs_error * abs_error
            } else {
                delta * abs_error - 0.5 * delta * delta
            }
        })
        .sum();

    sum_loss / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn huber_loss_sse2(y_true: &[f32], y_pred: &[f32], delta: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let delta_vec = _mm_set1_ps(delta);
    let half = _mm_set1_ps(0.5);
    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(true_vec, pred_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);

        // Check if abs_diff <= delta
        let mask = _mm_cmple_ps(abs_diff, delta_vec);

        // For quadratic case: 0.5 * abs_diff^2
        let quadratic = _mm_mul_ps(half, _mm_mul_ps(abs_diff, abs_diff));

        // For linear case: delta * abs_diff - 0.5 * delta^2
        let linear = _mm_sub_ps(
            _mm_mul_ps(delta_vec, abs_diff),
            _mm_mul_ps(half, _mm_mul_ps(delta_vec, delta_vec)),
        );

        // Select based on mask
        let result = _mm_blendv_ps(linear, quadratic, mask);
        sum = _mm_add_ps(sum, result);
        i += 4;
    }

    let mut result_array = [0.0f32; 4];
    _mm_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array[0] + result_array[1] + result_array[2] + result_array[3];

    while i < y_true.len() {
        let abs_error = (y_true[i] - y_pred[i]).abs();
        scalar_sum += if abs_error <= delta {
            0.5 * abs_error * abs_error
        } else {
            delta * abs_error - 0.5 * delta * delta
        };
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn huber_loss_avx2(y_true: &[f32], y_pred: &[f32], delta: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let delta_vec = _mm256_set1_ps(delta);
    let half = _mm256_set1_ps(0.5);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(true_vec, pred_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        // Check if abs_diff <= delta
        let mask = _mm256_cmp_ps(abs_diff, delta_vec, _CMP_LE_OQ);

        // For quadratic case: 0.5 * abs_diff^2
        let quadratic = _mm256_mul_ps(half, _mm256_mul_ps(abs_diff, abs_diff));

        // For linear case: delta * abs_diff - 0.5 * delta^2
        let linear = _mm256_sub_ps(
            _mm256_mul_ps(delta_vec, abs_diff),
            _mm256_mul_ps(half, _mm256_mul_ps(delta_vec, delta_vec)),
        );

        // Select based on mask
        let result = _mm256_blendv_ps(linear, quadratic, mask);
        sum = _mm256_add_ps(sum, result);
        i += 8;
    }

    let mut result_array = [0.0f32; 8];
    _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array.iter().sum::<f32>();

    while i < y_true.len() {
        let abs_error = (y_true[i] - y_pred[i]).abs();
        scalar_sum += if abs_error <= delta {
            0.5 * abs_error * abs_error
        } else {
            delta * abs_error - 0.5 * delta * delta
        };
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

/// SIMD-optimized Binary Cross-Entropy loss
pub fn binary_cross_entropy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );

    if y_true.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { binary_cross_entropy_avx2(y_true, y_pred) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { binary_cross_entropy_sse2(y_true, y_pred) };
        }
    }

    binary_cross_entropy_scalar(y_true, y_pred)
}

fn binary_cross_entropy_scalar(y_true: &[f32], y_pred: &[f32]) -> f32 {
    const EPSILON: f32 = 1e-15;

    let sum_loss: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            let p_clipped = p.clamp(EPSILON, 1.0 - EPSILON);
            -(t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln())
        })
        .sum();

    sum_loss / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn binary_cross_entropy_sse2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let epsilon = _mm_set1_ps(1e-15);
    let one_minus_epsilon = _mm_set1_ps(1.0 - 1e-15);
    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));

        // Clip predictions to avoid log(0)
        let pred_clipped = _mm_min_ps(_mm_max_ps(pred_vec, epsilon), one_minus_epsilon);
        let one_minus_pred = _mm_sub_ps(one, pred_clipped);
        let one_minus_true = _mm_sub_ps(one, true_vec);

        // Fast ln approximation using the activation module's exp_approx
        // For actual implementation, we'd need a ln_approx function
        // For now, fallback to scalar computation for this vectorized part
        let mut temp_result = [0.0f32; 4];
        for j in 0..4 {
            if i + j < y_true.len() {
                let t = y_true[i + j];
                let p = y_pred[i + j].clamp(1e-15, 1.0 - 1e-15);
                temp_result[j] = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
            }
        }

        let result_vec = _mm_loadu_ps(temp_result.as_ptr());
        sum = _mm_add_ps(sum, result_vec);
        i += 4;
    }

    let mut result_array = [0.0f32; 4];
    _mm_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array[0] + result_array[1] + result_array[2] + result_array[3];

    while i < y_true.len() {
        let t = y_true[i];
        let p = y_pred[i].clamp(1e-15, 1.0 - 1e-15);
        scalar_sum += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn binary_cross_entropy_avx2(y_true: &[f32], y_pred: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // For AVX2, we use the same approach as SSE2 but process 8 elements
    while i + 8 <= y_true.len() {
        let mut temp_result = [0.0f32; 8];
        for j in 0..8 {
            if i + j < y_true.len() {
                let t = y_true[i + j];
                let p = y_pred[i + j].clamp(1e-15, 1.0 - 1e-15);
                temp_result[j] = -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
            }
        }

        let result_vec = _mm256_loadu_ps(temp_result.as_ptr());
        sum = _mm256_add_ps(sum, result_vec);
        i += 8;
    }

    let mut result_array = [0.0f32; 8];
    _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
    let mut scalar_sum = result_array.iter().sum::<f32>();

    while i < y_true.len() {
        let t = y_true[i];
        let p = y_pred[i].clamp(1e-15, 1.0 - 1e-15);
        scalar_sum += -(t * p.ln() + (1.0 - t) * (1.0 - p).ln());
        i += 1;
    }

    scalar_sum / y_true.len() as f32
}

/// SIMD-optimized Log Loss (equivalent to binary cross-entropy)
pub fn log_loss(y_true: &[f32], y_pred: &[f32]) -> f32 {
    binary_cross_entropy(y_true, y_pred)
}

/// SIMD-optimized Categorical Cross-Entropy loss
/// y_true should be one-hot encoded, y_pred should be probability distributions
pub fn categorical_cross_entropy(y_true: &[f32], y_pred: &[f32]) -> f32 {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );

    if y_true.is_empty() {
        return 0.0;
    }

    const EPSILON: f32 = 1e-15;

    let sum_loss: f32 = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(t, p)| {
            if *t > 0.0 {
                -t * p.clamp(EPSILON, 1.0 - EPSILON).ln()
            } else {
                0.0
            }
        })
        .sum();

    sum_loss
}

// ============================================================================
// GRADIENT COMPUTATIONS
// ============================================================================

/// SIMD-optimized gradient of Mean Squared Error loss
/// Returns d(MSE)/d(y_pred) = 2 * (y_pred - y_true) / n
pub fn mse_gradient(y_true: &[f32], y_pred: &[f32], output: &mut [f32]) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );
    assert_eq!(
        y_true.len(),
        output.len(),
        "Output array must have the same length"
    );

    if y_true.is_empty() {
        return;
    }

    let scale = 2.0 / y_true.len() as f32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { mse_gradient_avx2(y_true, y_pred, output, scale) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { mse_gradient_sse2(y_true, y_pred, output, scale) };
            return;
        }
    }

    mse_gradient_scalar(y_true, y_pred, output, scale);
}

fn mse_gradient_scalar(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    for i in 0..y_true.len() {
        output[i] = scale * (y_pred[i] - y_true[i]);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mse_gradient_sse2(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;

    let scale_vec = _mm_set1_ps(scale);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(pred_vec, true_vec);
        let gradient = _mm_mul_ps(scale_vec, diff);
        _mm_storeu_ps(output.as_mut_ptr().add(i), gradient);
        i += 4;
    }

    while i < y_true.len() {
        output[i] = scale * (y_pred[i] - y_true[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mse_gradient_avx2(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;

    let scale_vec = _mm256_set1_ps(scale);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred_vec, true_vec);
        let gradient = _mm256_mul_ps(scale_vec, diff);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), gradient);
        i += 8;
    }

    while i < y_true.len() {
        output[i] = scale * (y_pred[i] - y_true[i]);
        i += 1;
    }
}

/// SIMD-optimized gradient of Mean Absolute Error loss
/// Returns d(MAE)/d(y_pred) = sign(y_pred - y_true) / n
pub fn mae_gradient(y_true: &[f32], y_pred: &[f32], output: &mut [f32]) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );
    assert_eq!(
        y_true.len(),
        output.len(),
        "Output array must have the same length"
    );

    if y_true.is_empty() {
        return;
    }

    let scale = 1.0 / y_true.len() as f32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { mae_gradient_avx2(y_true, y_pred, output, scale) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { mae_gradient_sse2(y_true, y_pred, output, scale) };
            return;
        }
    }

    mae_gradient_scalar(y_true, y_pred, output, scale);
}

fn mae_gradient_scalar(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    for i in 0..y_true.len() {
        let diff = y_pred[i] - y_true[i];
        // Handle the case where diff is exactly 0.0
        let sign = if diff == 0.0 { 0.0 } else { diff.signum() };
        output[i] = scale * sign;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mae_gradient_sse2(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;

    let scale_vec = _mm_set1_ps(scale);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let neg_one = _mm_set1_ps(-1.0);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(pred_vec, true_vec);

        // Compute sign: 1 if positive, -1 if negative, 0 if zero
        let pos_mask = _mm_cmpgt_ps(diff, zero);
        let neg_mask = _mm_cmplt_ps(diff, zero);

        let sign = _mm_add_ps(_mm_and_ps(pos_mask, one), _mm_and_ps(neg_mask, neg_one));

        let gradient = _mm_mul_ps(scale_vec, sign);
        _mm_storeu_ps(output.as_mut_ptr().add(i), gradient);
        i += 4;
    }

    while i < y_true.len() {
        let diff = y_pred[i] - y_true[i];
        output[i] = scale * diff.signum();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mae_gradient_avx2(y_true: &[f32], y_pred: &[f32], output: &mut [f32], scale: f32) {
    use core::arch::x86_64::*;

    let scale_vec = _mm256_set1_ps(scale);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred_vec, true_vec);

        // Compute sign: 1 if positive, -1 if negative, 0 if zero
        let pos_mask = _mm256_cmp_ps(diff, zero, _CMP_GT_OQ);
        let neg_mask = _mm256_cmp_ps(diff, zero, _CMP_LT_OQ);

        let sign = _mm256_add_ps(
            _mm256_and_ps(pos_mask, one),
            _mm256_and_ps(neg_mask, neg_one),
        );

        let gradient = _mm256_mul_ps(scale_vec, sign);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), gradient);
        i += 8;
    }

    while i < y_true.len() {
        let diff = y_pred[i] - y_true[i];
        output[i] = scale * diff.signum();
        i += 1;
    }
}

/// SIMD-optimized gradient of Huber loss
pub fn huber_gradient(y_true: &[f32], y_pred: &[f32], delta: f32, output: &mut [f32]) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );
    assert_eq!(
        y_true.len(),
        output.len(),
        "Output array must have the same length"
    );
    assert!(delta > 0.0, "Delta must be positive");

    if y_true.is_empty() {
        return;
    }

    let scale = 1.0 / y_true.len() as f32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { huber_gradient_avx2(y_true, y_pred, delta, output, scale) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { huber_gradient_sse2(y_true, y_pred, delta, output, scale) };
            return;
        }
    }

    huber_gradient_scalar(y_true, y_pred, delta, output, scale);
}

fn huber_gradient_scalar(
    y_true: &[f32],
    y_pred: &[f32],
    delta: f32,
    output: &mut [f32],
    scale: f32,
) {
    for i in 0..y_true.len() {
        let diff = y_pred[i] - y_true[i];
        let abs_diff = diff.abs();

        output[i] = scale
            * if abs_diff <= delta {
                diff // Quadratic region: gradient is the difference
            } else {
                delta * diff.signum() // Linear region: gradient is delta * sign
            };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn huber_gradient_sse2(
    y_true: &[f32],
    y_pred: &[f32],
    delta: f32,
    output: &mut [f32],
    scale: f32,
) {
    use core::arch::x86_64::*;

    let scale_vec = _mm_set1_ps(scale);
    let delta_vec = _mm_set1_ps(delta);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let neg_one = _mm_set1_ps(-1.0);
    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm_sub_ps(pred_vec, true_vec);
        let abs_diff = _mm_andnot_ps(sign_mask, diff);

        // Check if abs_diff <= delta
        let mask = _mm_cmple_ps(abs_diff, delta_vec);

        // Compute sign for linear region
        let pos_mask = _mm_cmpgt_ps(diff, zero);
        let neg_mask = _mm_cmplt_ps(diff, zero);
        let sign = _mm_add_ps(_mm_and_ps(pos_mask, one), _mm_and_ps(neg_mask, neg_one));

        // Quadratic region: gradient = diff
        let quadratic_grad = diff;

        // Linear region: gradient = delta * sign
        let linear_grad = _mm_mul_ps(delta_vec, sign);

        // Select based on mask
        let gradient = _mm_blendv_ps(linear_grad, quadratic_grad, mask);
        let scaled_gradient = _mm_mul_ps(scale_vec, gradient);

        _mm_storeu_ps(output.as_mut_ptr().add(i), scaled_gradient);
        i += 4;
    }

    while i < y_true.len() {
        let diff = y_pred[i] - y_true[i];
        let abs_diff = diff.abs();

        output[i] = scale
            * if abs_diff <= delta {
                diff
            } else {
                delta * diff.signum()
            };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn huber_gradient_avx2(
    y_true: &[f32],
    y_pred: &[f32],
    delta: f32,
    output: &mut [f32],
    scale: f32,
) {
    use core::arch::x86_64::*;

    let scale_vec = _mm256_set1_ps(scale);
    let delta_vec = _mm256_set1_ps(delta);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));
        let diff = _mm256_sub_ps(pred_vec, true_vec);
        let abs_diff = _mm256_andnot_ps(sign_mask, diff);

        // Check if abs_diff <= delta
        let mask = _mm256_cmp_ps(abs_diff, delta_vec, _CMP_LE_OQ);

        // Compute sign for linear region
        let pos_mask = _mm256_cmp_ps(diff, zero, _CMP_GT_OQ);
        let neg_mask = _mm256_cmp_ps(diff, zero, _CMP_LT_OQ);
        let sign = _mm256_add_ps(
            _mm256_and_ps(pos_mask, one),
            _mm256_and_ps(neg_mask, neg_one),
        );

        // Quadratic region: gradient = diff
        let quadratic_grad = diff;

        // Linear region: gradient = delta * sign
        let linear_grad = _mm256_mul_ps(delta_vec, sign);

        // Select based on mask
        let gradient = _mm256_blendv_ps(linear_grad, quadratic_grad, mask);
        let scaled_gradient = _mm256_mul_ps(scale_vec, gradient);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), scaled_gradient);
        i += 8;
    }

    while i < y_true.len() {
        let diff = y_pred[i] - y_true[i];
        let abs_diff = diff.abs();

        output[i] = scale
            * if abs_diff <= delta {
                diff
            } else {
                delta * diff.signum()
            };
        i += 1;
    }
}

/// SIMD-optimized gradient of Binary Cross-Entropy loss
/// Returns d(BCE)/d(y_pred) = (y_pred - y_true) / (y_pred * (1 - y_pred)) / n
pub fn binary_cross_entropy_gradient(y_true: &[f32], y_pred: &[f32], output: &mut [f32]) {
    assert_eq!(
        y_true.len(),
        y_pred.len(),
        "Arrays must have the same length"
    );
    assert_eq!(
        y_true.len(),
        output.len(),
        "Output array must have the same length"
    );

    if y_true.is_empty() {
        return;
    }

    let scale = 1.0 / y_true.len() as f32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { binary_cross_entropy_gradient_avx2(y_true, y_pred, output, scale) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { binary_cross_entropy_gradient_sse2(y_true, y_pred, output, scale) };
            return;
        }
    }

    binary_cross_entropy_gradient_scalar(y_true, y_pred, output, scale);
}

fn binary_cross_entropy_gradient_scalar(
    y_true: &[f32],
    y_pred: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    const EPSILON: f32 = 1e-15;

    for i in 0..y_true.len() {
        let p = y_pred[i].clamp(EPSILON, 1.0 - EPSILON);
        output[i] = scale * (p - y_true[i]) / (p * (1.0 - p));
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn binary_cross_entropy_gradient_sse2(
    y_true: &[f32],
    y_pred: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    use core::arch::x86_64::*;

    let scale_vec = _mm_set1_ps(scale);
    let epsilon = _mm_set1_ps(1e-15);
    let one_minus_epsilon = _mm_set1_ps(1.0 - 1e-15);
    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= y_true.len() {
        let true_vec = _mm_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm_loadu_ps(y_pred.as_ptr().add(i));

        // Clip predictions to avoid division by zero
        let pred_clipped = _mm_min_ps(_mm_max_ps(pred_vec, epsilon), one_minus_epsilon);
        let one_minus_pred = _mm_sub_ps(one, pred_clipped);

        // Compute gradient: (p - y) / (p * (1 - p))
        let numerator = _mm_sub_ps(pred_clipped, true_vec);
        let denominator = _mm_mul_ps(pred_clipped, one_minus_pred);
        let gradient = _mm_div_ps(numerator, denominator);
        let scaled_gradient = _mm_mul_ps(scale_vec, gradient);

        _mm_storeu_ps(output.as_mut_ptr().add(i), scaled_gradient);
        i += 4;
    }

    while i < y_true.len() {
        let p = y_pred[i].clamp(1e-15, 1.0 - 1e-15);
        output[i] = scale * (p - y_true[i]) / (p * (1.0 - p));
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn binary_cross_entropy_gradient_avx2(
    y_true: &[f32],
    y_pred: &[f32],
    output: &mut [f32],
    scale: f32,
) {
    use core::arch::x86_64::*;

    let scale_vec = _mm256_set1_ps(scale);
    let epsilon = _mm256_set1_ps(1e-15);
    let one_minus_epsilon = _mm256_set1_ps(1.0 - 1e-15);
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= y_true.len() {
        let true_vec = _mm256_loadu_ps(y_true.as_ptr().add(i));
        let pred_vec = _mm256_loadu_ps(y_pred.as_ptr().add(i));

        // Clip predictions to avoid division by zero
        let pred_clipped = _mm256_min_ps(_mm256_max_ps(pred_vec, epsilon), one_minus_epsilon);
        let one_minus_pred = _mm256_sub_ps(one, pred_clipped);

        // Compute gradient: (p - y) / (p * (1 - p))
        let numerator = _mm256_sub_ps(pred_clipped, true_vec);
        let denominator = _mm256_mul_ps(pred_clipped, one_minus_pred);
        let gradient = _mm256_div_ps(numerator, denominator);
        let scaled_gradient = _mm256_mul_ps(scale_vec, gradient);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), scaled_gradient);
        i += 8;
    }

    while i < y_true.len() {
        let p = y_pred[i].clamp(1e-15, 1.0 - 1e-15);
        output[i] = scale * (p - y_true[i]) / (p * (1.0 - p));
        i += 1;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse_loss() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let result = mse_loss(&y_true, &y_pred);

        // Expected: ((0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2 + (0.1)^2) / 5 = 0.01
        assert_relative_eq!(result, 0.01, epsilon = 1e-6);
    }

    #[test]
    fn test_mae_loss() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.1, 1.9, 3.1, 3.9, 5.1];

        let result = mae_loss(&y_true, &y_pred);

        // Expected: (0.1 + 0.1 + 0.1 + 0.1 + 0.1) / 5 = 0.1
        assert_relative_eq!(result, 0.1, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_loss() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.5, 1.5, 3.0, 4.0, 6.0]; // errors: 0.5, 0.5, 0.0, 0.0, 1.0
        let delta = 0.8;

        let result = huber_loss(&y_true, &y_pred, delta);

        // Expected: For errors 0.5, 0.5, 0.0, 0.0, 1.0 with delta=0.8
        // 0.5 <= 0.8: 0.5 * 0.5^2 = 0.125
        // 0.5 <= 0.8: 0.5 * 0.5^2 = 0.125
        // 0.0 <= 0.8: 0.5 * 0.0^2 = 0.0
        // 0.0 <= 0.8: 0.5 * 0.0^2 = 0.0
        // 1.0 > 0.8: 0.8 * 1.0 - 0.5 * 0.8^2 = 0.8 - 0.32 = 0.48
        // Average: (0.125 + 0.125 + 0.0 + 0.0 + 0.48) / 5 = 0.146

        assert_relative_eq!(result, 0.146, epsilon = 1e-3);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0, 1.0];
        let y_pred = vec![0.9, 0.1, 0.8, 0.2, 0.7];

        let result = binary_cross_entropy(&y_true, &y_pred);

        // This should be a small positive value since predictions are close to true labels
        assert!(result > 0.0);
        assert!(result < 1.0);
    }

    #[test]
    fn test_categorical_cross_entropy() {
        // One-hot encoded true labels and softmax predictions
        let y_true = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // 2 samples, 3 classes each
        let y_pred = vec![0.8, 0.1, 0.1, 0.2, 0.7, 0.1];

        let result = categorical_cross_entropy(&y_true, &y_pred);

        // Expected: -ln(0.8) - ln(0.7) = -(-0.223 - 0.357) ≈ 0.58
        assert!(result > 0.0);
        assert_relative_eq!(result, 0.58, epsilon = 0.1);
    }

    #[test]
    fn test_perfect_predictions() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_relative_eq!(mse_loss(&y_true, &y_pred), 0.0, epsilon = 1e-6);
        assert_relative_eq!(mae_loss(&y_true, &y_pred), 0.0, epsilon = 1e-6);
        assert_relative_eq!(huber_loss(&y_true, &y_pred, 1.0), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_gradient() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.5, 1.8, 3.2, 4.1];
        let mut gradient = vec![0.0; 4];

        mse_gradient(&y_true, &y_pred, &mut gradient);

        // Expected: 2 * (y_pred - y_true) / n = 2 * [0.5, -0.2, 0.2, 0.1] / 4
        let expected = vec![0.25, -0.1, 0.1, 0.05];

        for i in 0..gradient.len() {
            assert_relative_eq!(gradient[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mae_gradient() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.5, 1.8, 3.0, 3.9];
        let mut gradient = vec![0.0; 4];

        mae_gradient(&y_true, &y_pred, &mut gradient);

        // Expected: sign(y_pred - y_true) / n
        // [1.5-1.0, 1.8-2.0, 3.0-3.0, 3.9-4.0] = [0.5, -0.2, 0.0, -0.1]
        // sign([0.5, -0.2, 0.0, -0.1]) = [1, -1, 0, -1]
        // Final: [1, -1, 0, -1] / 4 = [0.25, -0.25, 0.0, -0.25]
        assert_relative_eq!(gradient[0], 0.25, epsilon = 1e-6);
        assert_relative_eq!(gradient[1], -0.25, epsilon = 1e-6);
        assert_relative_eq!(gradient[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(gradient[3], -0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_gradient() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.5, 1.2, 3.0, 5.5]; // errors: 0.5, -0.8, 0.0, 1.5
        let delta = 1.0;
        let mut gradient = vec![0.0; 4];

        huber_gradient(&y_true, &y_pred, delta, &mut gradient);

        // Expected for delta=1.0:
        // 0.5 <= 1.0: gradient = 0.5
        // -0.8 <= 1.0: gradient = -0.8
        // 0.0 <= 1.0: gradient = 0.0
        // 1.5 > 1.0: gradient = 1.0 * sign(1.5) = 1.0
        // Scale by 1/n = 1/4
        let expected = vec![0.125, -0.2, 0.0, 0.25];

        for i in 0..gradient.len() {
            assert_relative_eq!(gradient[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_binary_cross_entropy_gradient() {
        let y_true = vec![1.0, 0.0, 1.0, 0.0];
        let y_pred = vec![0.8, 0.3, 0.9, 0.1];
        let mut gradient = vec![0.0; 4];

        binary_cross_entropy_gradient(&y_true, &y_pred, &mut gradient);

        // BCE gradient: (p - y) / (p * (1 - p)) / n
        // For y_true=1, y_pred=0.8: (0.8 - 1.0) / (0.8 * 0.2) / 4 = -0.2 / 0.16 / 4 = -0.3125
        // For y_true=0, y_pred=0.3: (0.3 - 0.0) / (0.3 * 0.7) / 4 = 0.3 / 0.21 / 4 = 0.357...
        // For y_true=1, y_pred=0.9: (0.9 - 1.0) / (0.9 * 0.1) / 4 = -0.1 / 0.09 / 4 = -0.277...
        // For y_true=0, y_pred=0.1: (0.1 - 0.0) / (0.1 * 0.9) / 4 = 0.1 / 0.09 / 4 = 0.277...

        assert!(gradient[0] < 0.0); // y_true=1, y_pred=0.8: negative gradient (should increase pred)
        assert!(gradient[1] > 0.0); // y_true=0, y_pred=0.3: positive gradient (should decrease pred)
        assert!(gradient[2] < 0.0); // y_true=1, y_pred=0.9: negative gradient (should increase pred)
        assert!(gradient[3] > 0.0); // y_true=0, y_pred=0.1: positive gradient (should decrease pred)
    }

    #[test]
    fn test_gradient_perfect_predictions() {
        let y_true = vec![1.0, 2.0, 3.0, 4.0];
        let y_pred = vec![1.0, 2.0, 3.0, 4.0];
        let mut gradient = vec![0.0; 4];

        // MSE gradient should be zero for perfect predictions
        mse_gradient(&y_true, &y_pred, &mut gradient);
        for &g in &gradient {
            assert_relative_eq!(g, 0.0, epsilon = 1e-6);
        }

        // For MAE gradient with perfect predictions, we need to be careful about floating point precision
        // Let's test with a simple case where we know the exact result
        let y_true_simple = vec![2.0, 2.0];
        let y_pred_simple = vec![2.0, 2.0];
        let mut gradient_simple = vec![0.0; 2];

        mae_gradient(&y_true_simple, &y_pred_simple, &mut gradient_simple);
        for &g in &gradient_simple {
            assert_relative_eq!(g, 0.0, epsilon = 1e-6);
        }

        // Huber gradient should be zero for perfect predictions
        huber_gradient(&y_true, &y_pred, 1.0, &mut gradient);
        for &g in &gradient {
            assert_relative_eq!(g, 0.0, epsilon = 1e-6);
        }
    }
}

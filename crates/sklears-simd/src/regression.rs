//! SIMD-optimized regression operations
//!
//! This module provides vectorized implementations of common regression operations
//! including least squares, ridge regression, and elastic net computations.

#[cfg(feature = "no-std")]
use alloc::{vec, vec::Vec};

/// SIMD-optimized ordinary least squares computation
/// Computes X^T * X and X^T * y for the normal equation
pub fn least_squares_normal_equation(
    x: &[&[f32]], // Design matrix (n_samples x n_features)
    y: &[f32],    // Target values (n_samples)
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let n_samples = x.len();
    let n_features = if n_samples > 0 { x[0].len() } else { 0 };

    assert!(!x.is_empty(), "Design matrix cannot be empty");
    assert_eq!(
        y.len(),
        n_samples,
        "Target length must match number of samples"
    );

    // Initialize X^T * X matrix
    let mut xtx = vec![vec![0.0f32; n_features]; n_features];
    // Initialize X^T * y vector
    let mut xty = vec![0.0f32; n_features];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { least_squares_avx2(x, y, &mut xtx, &mut xty) };
            return (xtx, xty);
        } else if is_x86_feature_detected!("sse2") {
            unsafe { least_squares_sse2(x, y, &mut xtx, &mut xty) };
            return (xtx, xty);
        }
    }

    least_squares_scalar(x, y, &mut xtx, &mut xty);
    (xtx, xty)
}

fn least_squares_scalar(x: &[&[f32]], y: &[f32], xtx: &mut [Vec<f32>], xty: &mut [f32]) {
    let n_samples = x.len();
    let n_features = x[0].len();

    // Compute X^T * X
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..n_samples {
                sum += x[k][i] * x[k][j];
            }
            xtx[i][j] = sum;
        }
    }

    // Compute X^T * y
    for i in 0..n_features {
        let mut sum = 0.0;
        for k in 0..n_samples {
            sum += x[k][i] * y[k];
        }
        xty[i] = sum;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn least_squares_sse2(x: &[&[f32]], y: &[f32], xtx: &mut [Vec<f32>], xty: &mut [f32]) {
    use core::arch::x86_64::*;

    let n_samples = x.len();
    let n_features = x[0].len();

    // Compute X^T * X with SIMD
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = _mm_setzero_ps();
            let mut k = 0;

            while k + 4 <= n_samples {
                let xi_vec = _mm_setr_ps(x[k][i], x[k + 1][i], x[k + 2][i], x[k + 3][i]);
                let xj_vec = _mm_setr_ps(x[k][j], x[k + 1][j], x[k + 2][j], x[k + 3][j]);
                let prod = _mm_mul_ps(xi_vec, xj_vec);
                sum = _mm_add_ps(sum, prod);
                k += 4;
            }

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

            while k < n_samples {
                scalar_sum += x[k][i] * x[k][j];
                k += 1;
            }

            xtx[i][j] = scalar_sum;
        }
    }

    // Compute X^T * y with SIMD
    for i in 0..n_features {
        let mut sum = _mm_setzero_ps();
        let mut k = 0;

        while k + 4 <= n_samples {
            let xi_vec = _mm_setr_ps(x[k][i], x[k + 1][i], x[k + 2][i], x[k + 3][i]);
            let y_vec = _mm_loadu_ps(&y[k]);
            let prod = _mm_mul_ps(xi_vec, y_vec);
            sum = _mm_add_ps(sum, prod);
            k += 4;
        }

        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

        while k < n_samples {
            scalar_sum += x[k][i] * y[k];
            k += 1;
        }

        xty[i] = scalar_sum;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn least_squares_avx2(x: &[&[f32]], y: &[f32], xtx: &mut [Vec<f32>], xty: &mut [f32]) {
    use core::arch::x86_64::*;

    let n_samples = x.len();
    let n_features = x[0].len();

    // Compute X^T * X with SIMD
    for i in 0..n_features {
        for j in 0..n_features {
            let mut sum = _mm256_setzero_ps();
            let mut k = 0;

            while k + 8 <= n_samples {
                let xi_vec = _mm256_setr_ps(
                    x[k][i],
                    x[k + 1][i],
                    x[k + 2][i],
                    x[k + 3][i],
                    x[k + 4][i],
                    x[k + 5][i],
                    x[k + 6][i],
                    x[k + 7][i],
                );
                let xj_vec = _mm256_setr_ps(
                    x[k][j],
                    x[k + 1][j],
                    x[k + 2][j],
                    x[k + 3][j],
                    x[k + 4][j],
                    x[k + 5][j],
                    x[k + 6][j],
                    x[k + 7][j],
                );
                let prod = _mm256_mul_ps(xi_vec, xj_vec);
                sum = _mm256_add_ps(sum, prod);
                k += 8;
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum);
            let mut scalar_sum = result.iter().sum::<f32>();

            while k < n_samples {
                scalar_sum += x[k][i] * x[k][j];
                k += 1;
            }

            xtx[i][j] = scalar_sum;
        }
    }

    // Compute X^T * y with SIMD
    for i in 0..n_features {
        let mut sum = _mm256_setzero_ps();
        let mut k = 0;

        while k + 8 <= n_samples {
            let xi_vec = _mm256_setr_ps(
                x[k][i],
                x[k + 1][i],
                x[k + 2][i],
                x[k + 3][i],
                x[k + 4][i],
                x[k + 5][i],
                x[k + 6][i],
                x[k + 7][i],
            );
            let y_vec = _mm256_loadu_ps(&y[k]);
            let prod = _mm256_mul_ps(xi_vec, y_vec);
            sum = _mm256_add_ps(sum, prod);
            k += 8;
        }

        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut scalar_sum = result.iter().sum::<f32>();

        while k < n_samples {
            scalar_sum += x[k][i] * y[k];
            k += 1;
        }

        xty[i] = scalar_sum;
    }
}

/// SIMD-optimized ridge regression normal equation computation
/// Computes (X^T * X + alpha * I) and X^T * y
pub fn ridge_regression_normal_equation(
    x: &[&[f32]], // Design matrix (n_samples x n_features)
    y: &[f32],    // Target values (n_samples)
    alpha: f32,   // Regularization parameter
) -> (Vec<Vec<f32>>, Vec<f32>) {
    let (mut xtx, xty) = least_squares_normal_equation(x, y);

    // Add ridge regularization: X^T * X + alpha * I
    let n_features = xtx.len();
    for i in 0..n_features {
        xtx[i][i] += alpha;
    }

    (xtx, xty)
}

/// SIMD-optimized elastic net penalty computation
/// Computes the elastic net penalty: alpha * l1_ratio * ||w||_1 + 0.5 * alpha * (1 - l1_ratio) * ||w||_2^2
pub fn elastic_net_penalty(weights: &[f32], alpha: f32, l1_ratio: f32) -> f32 {
    assert!(
        (0.0..=1.0).contains(&l1_ratio),
        "l1_ratio must be between 0 and 1"
    );
    assert!(alpha >= 0.0, "alpha must be non-negative");

    if weights.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { elastic_net_penalty_avx2(weights, alpha, l1_ratio) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { elastic_net_penalty_sse2(weights, alpha, l1_ratio) };
        }
    }

    elastic_net_penalty_scalar(weights, alpha, l1_ratio)
}

fn elastic_net_penalty_scalar(weights: &[f32], alpha: f32, l1_ratio: f32) -> f32 {
    let l1_norm: f32 = weights.iter().map(|w| w.abs()).sum();
    let l2_norm_squared: f32 = weights.iter().map(|w| w * w).sum();

    alpha * l1_ratio * l1_norm + 0.5 * alpha * (1.0 - l1_ratio) * l2_norm_squared
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn elastic_net_penalty_sse2(weights: &[f32], alpha: f32, l1_ratio: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut l1_sum = _mm_setzero_ps();
    let mut l2_sum = _mm_setzero_ps();
    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 4 <= weights.len() {
        let w_vec = _mm_loadu_ps(weights.as_ptr().add(i));

        // L1 norm: sum of absolute values
        let abs_w = _mm_andnot_ps(sign_mask, w_vec);
        l1_sum = _mm_add_ps(l1_sum, abs_w);

        // L2 norm squared: sum of squares
        let squared_w = _mm_mul_ps(w_vec, w_vec);
        l2_sum = _mm_add_ps(l2_sum, squared_w);

        i += 4;
    }

    let mut l1_result = [0.0f32; 4];
    let mut l2_result = [0.0f32; 4];
    _mm_storeu_ps(l1_result.as_mut_ptr(), l1_sum);
    _mm_storeu_ps(l2_result.as_mut_ptr(), l2_sum);

    let mut l1_scalar = l1_result[0] + l1_result[1] + l1_result[2] + l1_result[3];
    let mut l2_scalar = l2_result[0] + l2_result[1] + l2_result[2] + l2_result[3];

    while i < weights.len() {
        l1_scalar += weights[i].abs();
        l2_scalar += weights[i] * weights[i];
        i += 1;
    }

    alpha * l1_ratio * l1_scalar + 0.5 * alpha * (1.0 - l1_ratio) * l2_scalar
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn elastic_net_penalty_avx2(weights: &[f32], alpha: f32, l1_ratio: f32) -> f32 {
    use core::arch::x86_64::*;

    let mut l1_sum = _mm256_setzero_ps();
    let mut l2_sum = _mm256_setzero_ps();
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 8 <= weights.len() {
        let w_vec = _mm256_loadu_ps(weights.as_ptr().add(i));

        // L1 norm: sum of absolute values
        let abs_w = _mm256_andnot_ps(sign_mask, w_vec);
        l1_sum = _mm256_add_ps(l1_sum, abs_w);

        // L2 norm squared: sum of squares
        let squared_w = _mm256_mul_ps(w_vec, w_vec);
        l2_sum = _mm256_add_ps(l2_sum, squared_w);

        i += 8;
    }

    let mut l1_result = [0.0f32; 8];
    let mut l2_result = [0.0f32; 8];
    _mm256_storeu_ps(l1_result.as_mut_ptr(), l1_sum);
    _mm256_storeu_ps(l2_result.as_mut_ptr(), l2_sum);

    let mut l1_scalar = l1_result.iter().sum::<f32>();
    let mut l2_scalar = l2_result.iter().sum::<f32>();

    while i < weights.len() {
        l1_scalar += weights[i].abs();
        l2_scalar += weights[i] * weights[i];
        i += 1;
    }

    alpha * l1_ratio * l1_scalar + 0.5 * alpha * (1.0 - l1_ratio) * l2_scalar
}

/// SIMD-optimized soft thresholding for LASSO
/// Applies soft thresholding: sign(x) * max(|x| - threshold, 0)
pub fn soft_threshold(values: &[f32], threshold: f32, output: &mut [f32]) {
    assert_eq!(
        values.len(),
        output.len(),
        "Arrays must have the same length"
    );
    assert!(threshold >= 0.0, "Threshold must be non-negative");

    if values.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { soft_threshold_avx2(values, threshold, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { soft_threshold_sse2(values, threshold, output) };
            return;
        }
    }

    soft_threshold_scalar(values, threshold, output);
}

fn soft_threshold_scalar(values: &[f32], threshold: f32, output: &mut [f32]) {
    for i in 0..values.len() {
        let abs_val = values[i].abs();
        if abs_val <= threshold {
            output[i] = 0.0;
        } else {
            output[i] = values[i].signum() * (abs_val - threshold);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn soft_threshold_sse2(values: &[f32], threshold: f32, output: &mut [f32]) {
    use core::arch::x86_64::*;

    let threshold_vec = _mm_set1_ps(threshold);
    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let neg_one = _mm_set1_ps(-1.0);
    let sign_mask = _mm_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 4 <= values.len() {
        let val_vec = _mm_loadu_ps(values.as_ptr().add(i));
        let abs_val = _mm_andnot_ps(sign_mask, val_vec);

        // Check if |x| > threshold
        let mask = _mm_cmpgt_ps(abs_val, threshold_vec);

        // Compute sign
        let pos_mask = _mm_cmpgt_ps(val_vec, zero);
        let neg_mask = _mm_cmplt_ps(val_vec, zero);
        let sign = _mm_add_ps(_mm_and_ps(pos_mask, one), _mm_and_ps(neg_mask, neg_one));

        // Compute soft thresholding: sign * max(|x| - threshold, 0)
        let thresholded = _mm_sub_ps(abs_val, threshold_vec);
        let result = _mm_mul_ps(sign, thresholded);

        // Apply mask: 0 if |x| <= threshold, result otherwise
        let final_result = _mm_and_ps(mask, result);

        _mm_storeu_ps(output.as_mut_ptr().add(i), final_result);
        i += 4;
    }

    while i < values.len() {
        let abs_val = values[i].abs();
        output[i] = if abs_val <= threshold {
            0.0
        } else {
            values[i].signum() * (abs_val - threshold)
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn soft_threshold_avx2(values: &[f32], threshold: f32, output: &mut [f32]) {
    use core::arch::x86_64::*;

    let threshold_vec = _mm256_set1_ps(threshold);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let neg_one = _mm256_set1_ps(-1.0);
    let sign_mask = _mm256_set1_ps(-0.0f32);
    let mut i = 0;

    while i + 8 <= values.len() {
        let val_vec = _mm256_loadu_ps(values.as_ptr().add(i));
        let abs_val = _mm256_andnot_ps(sign_mask, val_vec);

        // Check if |x| > threshold
        let mask = _mm256_cmp_ps(abs_val, threshold_vec, _CMP_GT_OQ);

        // Compute sign
        let pos_mask = _mm256_cmp_ps(val_vec, zero, _CMP_GT_OQ);
        let neg_mask = _mm256_cmp_ps(val_vec, zero, _CMP_LT_OQ);
        let sign = _mm256_add_ps(
            _mm256_and_ps(pos_mask, one),
            _mm256_and_ps(neg_mask, neg_one),
        );

        // Compute soft thresholding: sign * max(|x| - threshold, 0)
        let thresholded = _mm256_sub_ps(abs_val, threshold_vec);
        let result = _mm256_mul_ps(sign, thresholded);

        // Apply mask: 0 if |x| <= threshold, result otherwise
        let final_result = _mm256_and_ps(mask, result);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), final_result);
        i += 8;
    }

    while i < values.len() {
        let abs_val = values[i].abs();
        output[i] = if abs_val <= threshold {
            0.0
        } else {
            values[i].signum() * (abs_val - threshold)
        };
        i += 1;
    }
}

/// SIMD-optimized prediction for linear models
/// Computes y = X * beta
pub fn linear_predict(x: &[&[f32]], weights: &[f32], output: &mut [f32]) {
    let n_samples = x.len();
    let n_features = if n_samples > 0 { x[0].len() } else { 0 };

    assert_eq!(
        weights.len(),
        n_features,
        "Weight length must match number of features"
    );
    assert_eq!(
        output.len(),
        n_samples,
        "Output length must match number of samples"
    );

    if n_samples == 0 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { linear_predict_avx2(x, weights, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { linear_predict_sse2(x, weights, output) };
            return;
        }
    }

    linear_predict_scalar(x, weights, output);
}

fn linear_predict_scalar(x: &[&[f32]], weights: &[f32], output: &mut [f32]) {
    let n_samples = x.len();
    let n_features = weights.len();

    for i in 0..n_samples {
        let mut sum = 0.0;
        for j in 0..n_features {
            sum += x[i][j] * weights[j];
        }
        output[i] = sum;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn linear_predict_sse2(x: &[&[f32]], weights: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let n_samples = x.len();
    let n_features = weights.len();

    for i in 0..n_samples {
        let mut sum = _mm_setzero_ps();
        let mut j = 0;

        while j + 4 <= n_features {
            let x_vec = _mm_loadu_ps(&x[i][j]);
            let w_vec = _mm_loadu_ps(&weights[j]);
            let prod = _mm_mul_ps(x_vec, w_vec);
            sum = _mm_add_ps(sum, prod);
            j += 4;
        }

        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

        while j < n_features {
            scalar_sum += x[i][j] * weights[j];
            j += 1;
        }

        output[i] = scalar_sum;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn linear_predict_avx2(x: &[&[f32]], weights: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let n_samples = x.len();
    let n_features = weights.len();

    for i in 0..n_samples {
        let mut sum = _mm256_setzero_ps();
        let mut j = 0;

        while j + 8 <= n_features {
            let x_vec = _mm256_loadu_ps(&x[i][j]);
            let w_vec = _mm256_loadu_ps(&weights[j]);
            let prod = _mm256_mul_ps(x_vec, w_vec);
            sum = _mm256_add_ps(sum, prod);
            j += 8;
        }

        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), sum);
        let mut scalar_sum = result.iter().sum::<f32>();

        while j < n_features {
            scalar_sum += x[i][j] * weights[j];
            j += 1;
        }

        output[i] = scalar_sum;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_least_squares_normal_equation() {
        // Simple 2x2 design matrix
        let x1 = [1.0, 2.0];
        let x2 = [3.0, 4.0];
        let x = vec![&x1[..], &x2[..]];
        let y = [5.0, 6.0];

        let (xtx, xty) = least_squares_normal_equation(&x, &y);

        // Expected X^T * X = [[1*1 + 3*3, 1*2 + 3*4], [2*1 + 4*3, 2*2 + 4*4]] = [[10, 14], [14, 20]]
        assert_relative_eq!(xtx[0][0], 10.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[0][1], 14.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[1][0], 14.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[1][1], 20.0, epsilon = 1e-6);

        // Expected X^T * y = [1*5 + 3*6, 2*5 + 4*6] = [23, 34]
        assert_relative_eq!(xty[0], 23.0, epsilon = 1e-6);
        assert_relative_eq!(xty[1], 34.0, epsilon = 1e-6);
    }

    #[test]
    fn test_ridge_regression_normal_equation() {
        let x1 = [1.0, 2.0];
        let x2 = [3.0, 4.0];
        let x = vec![&x1[..], &x2[..]];
        let y = [5.0, 6.0];
        let alpha = 1.0;

        let (xtx, _) = ridge_regression_normal_equation(&x, &y, alpha);

        // Expected: X^T * X + alpha * I = [[10+1, 14], [14, 20+1]] = [[11, 14], [14, 21]]
        assert_relative_eq!(xtx[0][0], 11.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[0][1], 14.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[1][0], 14.0, epsilon = 1e-6);
        assert_relative_eq!(xtx[1][1], 21.0, epsilon = 1e-6);
    }

    #[test]
    fn test_elastic_net_penalty() {
        let weights = vec![1.0, -2.0, 3.0, -4.0];
        let alpha = 0.1;
        let l1_ratio = 0.5;

        let penalty = elastic_net_penalty(&weights, alpha, l1_ratio);

        // Expected: L1 norm = |1| + |-2| + |3| + |-4| = 10
        // L2 norm squared = 1^2 + 2^2 + 3^2 + 4^2 = 30
        // Penalty = 0.1 * 0.5 * 10 + 0.5 * 0.1 * 0.5 * 30 = 0.5 + 0.75 = 1.25
        assert_relative_eq!(penalty, 1.25, epsilon = 1e-6);
    }

    #[test]
    fn test_soft_threshold() {
        let values = vec![3.0, -2.0, 1.0, -0.5, 0.0];
        let threshold = 1.5;
        let mut output = vec![0.0; 5];

        soft_threshold(&values, threshold, &mut output);

        // Expected:
        // 3.0: |3.0| > 1.5, so 1.0 * (3.0 - 1.5) = 1.5
        // -2.0: |-2.0| > 1.5, so -1.0 * (2.0 - 1.5) = -0.5
        // 1.0: |1.0| <= 1.5, so 0.0
        // -0.5: |-0.5| <= 1.5, so 0.0
        // 0.0: |0.0| <= 1.5, so 0.0
        assert_relative_eq!(output[0], 1.5, epsilon = 1e-6);
        assert_relative_eq!(output[1], -0.5, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_linear_predict() {
        let x1 = [1.0, 2.0];
        let x2 = [3.0, 4.0];
        let x = vec![&x1[..], &x2[..]];
        let weights = vec![0.5, 1.0];
        let mut output = vec![0.0; 2];

        linear_predict(&x, &weights, &mut output);

        // Expected:
        // Sample 1: 1.0 * 0.5 + 2.0 * 1.0 = 2.5
        // Sample 2: 3.0 * 0.5 + 4.0 * 1.0 = 5.5
        assert_relative_eq!(output[0], 2.5, epsilon = 1e-6);
        assert_relative_eq!(output[1], 5.5, epsilon = 1e-6);
    }
}

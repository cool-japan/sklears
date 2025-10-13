//! SIMD-optimized reduction and scan operations
//!
//! This module provides vectorized implementations of reduction operations
//! including parallel reductions, prefix sums (scan), and segment-based operations.

#[cfg(feature = "no-std")]
use alloc::{vec, vec::Vec};

/// SIMD-optimized parallel reduction sum
/// Computes the sum of all elements in the array using parallel reduction
pub fn parallel_sum_f32_simd(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && arr.len() >= 8 {
            return unsafe { parallel_sum_avx2(arr) };
        } else if is_x86_feature_detected!("sse2") && arr.len() >= 4 {
            return unsafe { parallel_sum_sse2(arr) };
        }
    }

    parallel_sum_scalar(arr)
}

fn parallel_sum_scalar(arr: &[f32]) -> f32 {
    arr.iter().sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn parallel_sum_sse2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        sum = _mm_add_ps(sum, vec);
        i += 4;
    }

    // Horizontal sum of the SIMD register
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < arr.len() {
        total += arr[i];
        i += 1;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn parallel_sum_avx2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        sum = _mm256_add_ps(sum, vec);
        i += 8;
    }

    // Horizontal sum of the SIMD register
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < arr.len() {
        total += arr[i];
        i += 1;
    }

    total
}

/// SIMD-optimized parallel reduction product
/// Computes the product of all elements in the array using parallel reduction
pub fn parallel_product_f32_simd(arr: &[f32]) -> f32 {
    if arr.is_empty() {
        return 1.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && arr.len() >= 8 {
            return unsafe { parallel_product_avx2(arr) };
        } else if is_x86_feature_detected!("sse2") && arr.len() >= 4 {
            return unsafe { parallel_product_sse2(arr) };
        }
    }

    parallel_product_scalar(arr)
}

fn parallel_product_scalar(arr: &[f32]) -> f32 {
    arr.iter().product()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn parallel_product_sse2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut product = _mm_set1_ps(1.0);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        product = _mm_mul_ps(product, vec);
        i += 4;
    }

    // Horizontal product of the SIMD register
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), product);
    let mut total = result[0] * result[1] * result[2] * result[3];

    // Handle remaining elements
    while i < arr.len() {
        total *= arr[i];
        i += 1;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn parallel_product_avx2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut product = _mm256_set1_ps(1.0);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        product = _mm256_mul_ps(product, vec);
        i += 8;
    }

    // Horizontal product of the SIMD register
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), product);
    let mut total = result.iter().product::<f32>();

    // Handle remaining elements
    while i < arr.len() {
        total *= arr[i];
        i += 1;
    }

    total
}

/// SIMD-optimized parallel reduction maximum
/// Finds the maximum element in the array using parallel reduction
pub fn parallel_max_f32_simd(arr: &[f32]) -> Option<f32> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && arr.len() >= 8 {
            return Some(unsafe { parallel_max_avx2(arr) });
        } else if is_x86_feature_detected!("sse2") && arr.len() >= 4 {
            return Some(unsafe { parallel_max_sse2(arr) });
        }
    }

    parallel_max_scalar(arr)
}

fn parallel_max_scalar(arr: &[f32]) -> Option<f32> {
    arr.iter()
        .fold(None, |acc, &x| Some(acc.map_or(x, |max| x.max(max))))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn parallel_max_sse2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut max_vec = _mm_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        max_vec = _mm_max_ps(max_vec, vec);
        i += 4;
    }

    // Horizontal max of the SIMD register
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0].max(result[1]).max(result[2]).max(result[3]);

    // Handle remaining elements
    while i < arr.len() {
        max_val = max_val.max(arr[i]);
        i += 1;
    }

    max_val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn parallel_max_avx2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        max_vec = _mm256_max_ps(max_vec, vec);
        i += 8;
    }

    // Horizontal max of the SIMD register
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0];
    for j in 1..8 {
        max_val = max_val.max(result[j]);
    }

    // Handle remaining elements
    while i < arr.len() {
        max_val = max_val.max(arr[i]);
        i += 1;
    }

    max_val
}

/// SIMD-optimized parallel reduction minimum
/// Finds the minimum element in the array using parallel reduction
pub fn parallel_min_f32_simd(arr: &[f32]) -> Option<f32> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && arr.len() >= 8 {
            return Some(unsafe { parallel_min_avx2(arr) });
        } else if is_x86_feature_detected!("sse2") && arr.len() >= 4 {
            return Some(unsafe { parallel_min_sse2(arr) });
        }
    }

    parallel_min_scalar(arr)
}

fn parallel_min_scalar(arr: &[f32]) -> Option<f32> {
    arr.iter()
        .fold(None, |acc, &x| Some(acc.map_or(x, |min| x.min(min))))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn parallel_min_sse2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut min_vec = _mm_set1_ps(f32::INFINITY);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        min_vec = _mm_min_ps(min_vec, vec);
        i += 4;
    }

    // Horizontal min of the SIMD register
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), min_vec);
    let mut min_val = result[0].min(result[1]).min(result[2]).min(result[3]);

    // Handle remaining elements
    while i < arr.len() {
        min_val = min_val.min(arr[i]);
        i += 1;
    }

    min_val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn parallel_min_avx2(arr: &[f32]) -> f32 {
    use core::arch::x86_64::*;

    let mut min_vec = _mm256_set1_ps(f32::INFINITY);
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        min_vec = _mm256_min_ps(min_vec, vec);
        i += 8;
    }

    // Horizontal min of the SIMD register
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), min_vec);
    let mut min_val = result[0];
    for j in 1..8 {
        min_val = min_val.min(result[j]);
    }

    // Handle remaining elements
    while i < arr.len() {
        min_val = min_val.min(arr[i]);
        i += 1;
    }

    min_val
}

/// SIMD-optimized prefix sum (inclusive scan)
/// Computes cumulative sum where output[i] = sum of input[0] through input[i]
pub fn prefix_sum_f32_simd(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output must have same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && input.len() >= 8 {
            unsafe { prefix_sum_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") && input.len() >= 4 {
            unsafe { prefix_sum_sse2(input, output) };
            return;
        }
    }

    prefix_sum_scalar(input, output);
}

fn prefix_sum_scalar(input: &[f32], output: &mut [f32]) {
    if input.is_empty() {
        return;
    }

    output[0] = input[0];
    for i in 1..input.len() {
        output[i] = output[i - 1] + input[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn prefix_sum_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let len = input.len();
    let mut running_sum = 0.0;

    let mut i = 0;
    while i + 4 <= len {
        // Load 4 elements
        let vec = _mm_loadu_ps(&input[i]);

        // Extract elements for prefix sum computation
        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vec);

        // Compute prefix sum for this block
        output[i] = running_sum + temp[0];
        output[i + 1] = running_sum + temp[0] + temp[1];
        output[i + 2] = running_sum + temp[0] + temp[1] + temp[2];
        output[i + 3] = running_sum + temp[0] + temp[1] + temp[2] + temp[3];

        // Update running sum
        running_sum = output[i + 3];

        i += 4;
    }

    // Handle remaining elements
    while i < len {
        output[i] = running_sum + input[i];
        running_sum = output[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn prefix_sum_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let len = input.len();
    let mut running_sum = 0.0;

    let mut i = 0;
    while i + 8 <= len {
        // Load 8 elements
        let vec = _mm256_loadu_ps(&input[i]);

        // Extract elements for prefix sum computation
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), vec);

        // Compute prefix sum for this block
        output[i] = running_sum + temp[0];
        output[i + 1] = running_sum + temp[0] + temp[1];
        output[i + 2] = running_sum + temp[0] + temp[1] + temp[2];
        output[i + 3] = running_sum + temp[0] + temp[1] + temp[2] + temp[3];
        output[i + 4] = running_sum + temp[0] + temp[1] + temp[2] + temp[3] + temp[4];
        output[i + 5] = running_sum + temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5];
        output[i + 6] =
            running_sum + temp[0] + temp[1] + temp[2] + temp[3] + temp[4] + temp[5] + temp[6];
        output[i + 7] = running_sum
            + temp[0]
            + temp[1]
            + temp[2]
            + temp[3]
            + temp[4]
            + temp[5]
            + temp[6]
            + temp[7];

        // Update running sum
        running_sum = output[i + 7];

        i += 8;
    }

    // Handle remaining elements
    while i < len {
        output[i] = running_sum + input[i];
        running_sum = output[i];
        i += 1;
    }
}

/// SIMD-optimized exclusive scan (prefix sum where output[i] = sum of input[0] through input[i-1])
/// output[0] = 0, output[i] = sum of input[0] through input[i-1] for i > 0
pub fn exclusive_scan_f32_simd(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output must have same length"
    );

    if input.is_empty() {
        return;
    }

    // Compute exclusive scan by shifting the inclusive scan
    let mut temp = vec![0.0; input.len()];
    prefix_sum_f32_simd(input, &mut temp);

    output[0] = 0.0;
    for i in 1..input.len() {
        output[i] = temp[i - 1];
    }
}

/// SIMD-optimized segmented reduction
/// Performs reduction within segments defined by segment flags
/// When segment_flags[i] is true, a new segment starts at position i
pub fn segmented_sum_f32_simd(input: &[f32], segment_flags: &[bool], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        segment_flags.len(),
        "Input and flags must have same length"
    );
    assert_eq!(
        input.len(),
        output.len(),
        "Input and output must have same length"
    );

    if input.is_empty() {
        return;
    }

    let mut running_sum = 0.0;

    for i in 0..input.len() {
        if segment_flags[i] {
            running_sum = input[i];
        } else {
            running_sum += input[i];
        }
        output[i] = running_sum;
    }
}

/// SIMD-optimized conditional reduction
/// Performs reduction only on elements where condition[i] is true
pub fn conditional_sum_f32_simd(input: &[f32], condition: &[bool]) -> f32 {
    assert_eq!(
        input.len(),
        condition.len(),
        "Input and condition must have same length"
    );

    if input.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && input.len() >= 8 {
            return unsafe { conditional_sum_avx2(input, condition) };
        } else if is_x86_feature_detected!("sse2") && input.len() >= 4 {
            return unsafe { conditional_sum_sse2(input, condition) };
        }
    }

    conditional_sum_scalar(input, condition)
}

fn conditional_sum_scalar(input: &[f32], condition: &[bool]) -> f32 {
    input
        .iter()
        .zip(condition.iter())
        .map(|(&val, &cond)| if cond { val } else { 0.0 })
        .sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn conditional_sum_sse2(input: &[f32], condition: &[bool]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= input.len() {
        let vec = _mm_loadu_ps(&input[i]);

        // Create mask from boolean conditions
        let mask = _mm_set_ps(
            if condition[i + 3] { 1.0 } else { 0.0 },
            if condition[i + 2] { 1.0 } else { 0.0 },
            if condition[i + 1] { 1.0 } else { 0.0 },
            if condition[i] { 1.0 } else { 0.0 },
        );

        let masked_vec = _mm_mul_ps(vec, mask);
        sum = _mm_add_ps(sum, masked_vec);

        i += 4;
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < input.len() {
        if condition[i] {
            total += input[i];
        }
        i += 1;
    }

    total
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn conditional_sum_avx2(input: &[f32], condition: &[bool]) -> f32 {
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= input.len() {
        let vec = _mm256_loadu_ps(&input[i]);

        // Create mask from boolean conditions
        let mask = _mm256_set_ps(
            if condition[i + 7] { 1.0 } else { 0.0 },
            if condition[i + 6] { 1.0 } else { 0.0 },
            if condition[i + 5] { 1.0 } else { 0.0 },
            if condition[i + 4] { 1.0 } else { 0.0 },
            if condition[i + 3] { 1.0 } else { 0.0 },
            if condition[i + 2] { 1.0 } else { 0.0 },
            if condition[i + 1] { 1.0 } else { 0.0 },
            if condition[i] { 1.0 } else { 0.0 },
        );

        let masked_vec = _mm256_mul_ps(vec, mask);
        sum = _mm256_add_ps(sum, masked_vec);

        i += 8;
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < input.len() {
        if condition[i] {
            total += input[i];
        }
        i += 1;
    }

    total
}

/// SIMD-optimized reduce by key operation
/// Groups consecutive elements with the same key and reduces each group
#[derive(Debug, Clone, PartialEq)]
pub struct KeyValue<K, V> {
    pub key: K,
    pub value: V,
}

pub fn reduce_by_key_f32_simd(
    input: &[KeyValue<i32, f32>],
    reduction_op: fn(f32, f32) -> f32,
) -> Vec<KeyValue<i32, f32>> {
    if input.is_empty() {
        return Vec::new();
    }

    let mut result = Vec::new();
    let mut current_key = input[0].key;
    let mut current_value = input[0].value;

    for item in input.iter().skip(1) {
        if item.key == current_key {
            current_value = reduction_op(current_value, item.value);
        } else {
            result.push(KeyValue {
                key: current_key,
                value: current_value,
            });
            current_key = item.key;
            current_value = item.value;
        }
    }

    // Don't forget the last group
    result.push(KeyValue {
        key: current_key,
        value: current_value,
    });

    result
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_parallel_sum() {
        let arr = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let sum = parallel_sum_f32_simd(&arr);
        let expected: f32 = arr.iter().sum();
        assert_relative_eq!(sum, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_product() {
        let arr = vec![1.0, 2.0, 3.0, 4.0];
        let product = parallel_product_f32_simd(&arr);
        let expected: f32 = arr.iter().product();
        assert_relative_eq!(product, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_parallel_max() {
        let arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let max_val = parallel_max_f32_simd(&arr);
        assert_eq!(max_val, Some(9.0));
    }

    #[test]
    fn test_parallel_min() {
        let arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let min_val = parallel_min_f32_simd(&arr);
        assert_eq!(min_val, Some(1.0));
    }

    #[test]
    fn test_prefix_sum() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; input.len()];
        prefix_sum_f32_simd(&input, &mut output);

        let expected = vec![1.0, 3.0, 6.0, 10.0, 15.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_exclusive_scan() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mut output = vec![0.0; input.len()];
        exclusive_scan_f32_simd(&input, &mut output);

        let expected = vec![0.0, 1.0, 3.0, 6.0, 10.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_segmented_sum() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let flags = vec![true, false, false, true, false, true];
        let mut output = vec![0.0; input.len()];

        segmented_sum_f32_simd(&input, &flags, &mut output);

        // Segments: [1,2,3], [4,5], [6]
        // Expected cumulative sums: [1,3,6], [4,9], [6]
        let expected = vec![1.0, 3.0, 6.0, 4.0, 9.0, 6.0];
        for (i, &expected_val) in expected.iter().enumerate() {
            assert_relative_eq!(output[i], expected_val, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_conditional_sum() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let condition = vec![true, false, true, false, true];

        let sum = conditional_sum_f32_simd(&input, &condition);
        assert_relative_eq!(sum, 9.0, epsilon = 1e-6); // 1 + 3 + 5
    }

    #[test]
    fn test_reduce_by_key() {
        let input = vec![
            KeyValue {
                key: 1,
                value: 10.0,
            },
            KeyValue {
                key: 1,
                value: 20.0,
            },
            KeyValue {
                key: 2,
                value: 30.0,
            },
            KeyValue {
                key: 2,
                value: 40.0,
            },
            KeyValue {
                key: 3,
                value: 50.0,
            },
        ];

        let result = reduce_by_key_f32_simd(&input, |a, b| a + b);

        assert_eq!(result.len(), 3);
        assert_eq!(
            result[0],
            KeyValue {
                key: 1,
                value: 30.0
            }
        );
        assert_eq!(
            result[1],
            KeyValue {
                key: 2,
                value: 70.0
            }
        );
        assert_eq!(
            result[2],
            KeyValue {
                key: 3,
                value: 50.0
            }
        );
    }

    #[test]
    fn test_empty_arrays() {
        let empty: Vec<f32> = vec![];
        assert_eq!(parallel_sum_f32_simd(&empty), 0.0);
        assert_eq!(parallel_product_f32_simd(&empty), 1.0);
        assert_eq!(parallel_max_f32_simd(&empty), None);
        assert_eq!(parallel_min_f32_simd(&empty), None);
    }

    #[test]
    fn test_single_element() {
        let arr = vec![42.0];
        assert_eq!(parallel_sum_f32_simd(&arr), 42.0);
        assert_eq!(parallel_product_f32_simd(&arr), 42.0);
        assert_eq!(parallel_max_f32_simd(&arr), Some(42.0));
        assert_eq!(parallel_min_f32_simd(&arr), Some(42.0));
    }
}

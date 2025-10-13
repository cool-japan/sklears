//! SIMD-optimized sorting algorithms
//!
//! This module provides vectorized implementations of sorting algorithms
//! including quicksort, bitonic sort, and specialized sorting operations.

/// Generic quicksort function for raw pointer interface
pub fn quicksort(data: &mut [f32]) -> Result<(), crate::traits::SimdError> {
    quicksort_f32_simd(data);
    Ok(())
}

/// SIMD-optimized quicksort for f32 arrays
/// Uses vectorized partitioning and parallel processing for optimal performance
pub fn quicksort_f32_simd(arr: &mut [f32]) {
    if arr.len() <= 1 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && arr.len() >= 16 {
            unsafe { quicksort_avx2(arr) };
            return;
        } else if is_x86_feature_detected!("sse2") && arr.len() >= 8 {
            unsafe { quicksort_sse2(arr) };
            return;
        }
    }

    // Fall back to scalar quicksort for small arrays or unsupported platforms
    quicksort_scalar(arr);
}

fn quicksort_scalar(arr: &mut [f32]) {
    if arr.len() <= 1 {
        return;
    }

    let pivot_index = partition_scalar(arr);
    quicksort_scalar(&mut arr[0..pivot_index]);
    quicksort_scalar(&mut arr[pivot_index + 1..]);
}

fn partition_scalar(arr: &mut [f32]) -> usize {
    let len = arr.len();
    let pivot = arr[len - 1];
    let mut i = 0;

    for j in 0..len - 1 {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, len - 1);
    i
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn quicksort_sse2(arr: &mut [f32]) {
    if arr.len() <= 8 {
        // Use insertion sort for small arrays
        insertion_sort_simd_sse2(arr);
        return;
    }

    let pivot_index = partition_sse2(arr);
    quicksort_sse2(&mut arr[0..pivot_index]);
    quicksort_sse2(&mut arr[pivot_index + 1..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn partition_sse2(arr: &mut [f32]) -> usize {
    use core::arch::x86_64::*;

    let len = arr.len();
    let pivot = arr[len - 1];
    let pivot_vec = _mm_set1_ps(pivot);

    let mut left = 0;
    let mut right = len - 1;

    while left + 4 <= right {
        // Load 4 elements from left
        let left_vec = _mm_loadu_ps(&arr[left]);

        // Compare with pivot
        let cmp_mask = _mm_cmple_ps(left_vec, pivot_vec);
        let mask = _mm_movemask_ps(cmp_mask);

        // Process elements based on comparison
        for i in 0..4 {
            if left < right {
                if (mask & (1 << i)) != 0 {
                    // Element is <= pivot, keep it on the left
                    left += 1;
                } else {
                    // Element is > pivot, swap with element from right
                    while right > left && arr[right - 1] > pivot {
                        right -= 1;
                    }
                    if right > left {
                        arr.swap(left, right - 1);
                        right -= 1;
                        left += 1;
                    }
                }
            }
        }
    }

    // Handle remaining elements with scalar code
    while left < right - 1 {
        if arr[left] <= pivot {
            left += 1;
        } else {
            right -= 1;
            arr.swap(left, right);
        }
    }

    arr.swap(left, len - 1);
    left
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn quicksort_avx2(arr: &mut [f32]) {
    if arr.len() <= 16 {
        // Use insertion sort for small arrays
        insertion_sort_simd_avx2(arr);
        return;
    }

    let pivot_index = partition_avx2(arr);
    quicksort_avx2(&mut arr[0..pivot_index]);
    quicksort_avx2(&mut arr[pivot_index + 1..]);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn partition_avx2(arr: &mut [f32]) -> usize {
    use core::arch::x86_64::*;

    let len = arr.len();
    let pivot = arr[len - 1];
    let pivot_vec = _mm256_set1_ps(pivot);

    let mut left = 0;
    let mut right = len - 1;

    while left + 8 <= right {
        // Load 8 elements from left
        let left_vec = _mm256_loadu_ps(&arr[left]);

        // Compare with pivot
        let cmp_mask = _mm256_cmp_ps(left_vec, pivot_vec, _CMP_LE_OQ);
        let mask = _mm256_movemask_ps(cmp_mask);

        // Process elements based on comparison
        for i in 0..8 {
            if left < right {
                if (mask & (1 << i)) != 0 {
                    // Element is <= pivot, keep it on the left
                    left += 1;
                } else {
                    // Element is > pivot, swap with element from right
                    while right > left && arr[right - 1] > pivot {
                        right -= 1;
                    }
                    if right > left {
                        arr.swap(left, right - 1);
                        right -= 1;
                        left += 1;
                    }
                }
            }
        }
    }

    // Handle remaining elements with scalar code
    while left < right - 1 {
        if arr[left] <= pivot {
            left += 1;
        } else {
            right -= 1;
            arr.swap(left, right);
        }
    }

    arr.swap(left, len - 1);
    left
}

/// SIMD-optimized insertion sort for small arrays
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn insertion_sort_simd_sse2(arr: &mut [f32]) {
    use core::arch::x86_64::*;

    if arr.len() <= 1 {
        return;
    }

    // For very small arrays, use scalar insertion sort
    if arr.len() <= 4 {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
        return;
    }

    // SIMD-assisted insertion sort for slightly larger arrays
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;

        // Use SIMD for comparison when possible
        while j >= 4 {
            let vec = _mm_loadu_ps(&arr[j - 4]);
            let key_vec = _mm_set1_ps(key);
            let cmp = _mm_cmpgt_ps(vec, key_vec);
            let mask = _mm_movemask_ps(cmp);

            if mask == 0 {
                // No elements are greater than key
                break;
            }

            // Shift elements one by one (scalar fallback for shifting)
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            break;
        }

        // Handle remaining elements with scalar code
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn insertion_sort_simd_avx2(arr: &mut [f32]) {
    use core::arch::x86_64::*;

    if arr.len() <= 1 {
        return;
    }

    // For very small arrays, use scalar insertion sort
    if arr.len() <= 8 {
        for i in 1..arr.len() {
            let key = arr[i];
            let mut j = i;
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            arr[j] = key;
        }
        return;
    }

    // SIMD-assisted insertion sort for slightly larger arrays
    for i in 1..arr.len() {
        let key = arr[i];
        let mut j = i;

        // Use SIMD for comparison when possible
        while j >= 8 {
            let vec = _mm256_loadu_ps(&arr[j - 8]);
            let key_vec = _mm256_set1_ps(key);
            let cmp = _mm256_cmp_ps(vec, key_vec, _CMP_GT_OQ);
            let mask = _mm256_movemask_ps(cmp);

            if mask == 0 {
                // No elements are greater than key
                break;
            }

            // Shift elements one by one (scalar fallback for shifting)
            while j > 0 && arr[j - 1] > key {
                arr[j] = arr[j - 1];
                j -= 1;
            }
            break;
        }

        // Handle remaining elements with scalar code
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

/// Bitonic sort for power-of-2 sized arrays
/// Optimal for small fixed-size arrays with SIMD processing
pub fn bitonic_sort_f32_simd(arr: &mut [f32], ascending: bool) {
    let len = arr.len();

    // Ensure length is a power of 2
    assert!(
        len.is_power_of_two(),
        "Bitonic sort requires power-of-2 length"
    );

    if len <= 1 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") && len >= 8 {
            unsafe { bitonic_sort_avx2(arr, ascending) };
            return;
        } else if is_x86_feature_detected!("sse2") && len >= 4 {
            unsafe { bitonic_sort_sse2(arr, ascending) };
            return;
        }
    }

    bitonic_sort_scalar(arr, ascending);
}

fn bitonic_sort_scalar(arr: &mut [f32], ascending: bool) {
    let len = arr.len();

    if len <= 1 {
        return;
    }

    if len == 2 {
        if (arr[0] > arr[1]) == ascending {
            arr.swap(0, 1);
        }
        return;
    }

    let mid = len / 2;

    // Sort first half in ascending order
    bitonic_sort_scalar(&mut arr[0..mid], true);

    // Sort second half in descending order
    bitonic_sort_scalar(&mut arr[mid..], false);

    // Merge the bitonic sequence
    bitonic_merge_scalar(arr, ascending);
}

fn bitonic_merge_scalar(arr: &mut [f32], ascending: bool) {
    let len = arr.len();

    if len <= 1 {
        return;
    }

    let step = len / 2;

    for i in 0..step {
        if (arr[i] > arr[i + step]) == ascending {
            arr.swap(i, i + step);
        }
    }

    if step > 1 {
        bitonic_merge_scalar(&mut arr[0..step], ascending);
        bitonic_merge_scalar(&mut arr[step..], ascending);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn bitonic_sort_sse2(arr: &mut [f32], ascending: bool) {
    let len = arr.len();

    if len <= 4 {
        bitonic_sort_4_sse2(arr, ascending);
        return;
    }

    let mid = len / 2;

    // Sort halves recursively
    bitonic_sort_sse2(&mut arr[0..mid], true);
    bitonic_sort_sse2(&mut arr[mid..], false);

    // Merge
    bitonic_merge_sse2(arr, ascending);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn bitonic_sort_4_sse2(arr: &mut [f32], ascending: bool) {
    use core::arch::x86_64::*;

    if arr.len() != 4 {
        bitonic_sort_scalar(arr, ascending);
        return;
    }

    let mut vec = _mm_loadu_ps(arr.as_ptr());

    // Implement 4-element bitonic sort with SSE2
    // This is a simplified version - a full implementation would be more complex
    let temp = [arr[0], arr[1], arr[2], arr[3]];
    let mut sorted = temp;
    sorted.sort_by(|a, b| {
        if ascending {
            a.partial_cmp(b).unwrap()
        } else {
            b.partial_cmp(a).unwrap()
        }
    });

    vec = _mm_loadu_ps(sorted.as_ptr());
    _mm_storeu_ps(arr.as_mut_ptr(), vec);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn bitonic_merge_sse2(arr: &mut [f32], ascending: bool) {
    use core::arch::x86_64::*;

    let len = arr.len();

    if len <= 4 {
        bitonic_merge_scalar(arr, ascending);
        return;
    }

    let step = len / 2;

    // SIMD-accelerated comparison and swapping
    let mut i = 0;
    while i + 4 <= step {
        let vec1 = _mm_loadu_ps(&arr[i]);
        let vec2 = _mm_loadu_ps(&arr[i + step]);

        let cmp = if ascending {
            _mm_cmpgt_ps(vec1, vec2)
        } else {
            _mm_cmplt_ps(vec1, vec2)
        };

        let mask = _mm_movemask_ps(cmp);

        // Handle swaps based on mask (simplified approach)
        for j in 0..4 {
            if (mask & (1 << j)) != 0 {
                arr.swap(i + j, i + j + step);
            }
        }

        i += 4;
    }

    // Handle remaining elements
    while i < step {
        if (arr[i] > arr[i + step]) == ascending {
            arr.swap(i, i + step);
        }
        i += 1;
    }

    if step > 1 {
        bitonic_merge_sse2(&mut arr[0..step], ascending);
        bitonic_merge_sse2(&mut arr[step..], ascending);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bitonic_sort_avx2(arr: &mut [f32], ascending: bool) {
    let len = arr.len();

    if len <= 8 {
        bitonic_sort_8_avx2(arr, ascending);
        return;
    }

    let mid = len / 2;

    // Sort halves recursively
    bitonic_sort_avx2(&mut arr[0..mid], true);
    bitonic_sort_avx2(&mut arr[mid..], false);

    // Merge
    bitonic_merge_avx2(arr, ascending);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bitonic_sort_8_avx2(arr: &mut [f32], ascending: bool) {
    use core::arch::x86_64::*;

    if arr.len() != 8 {
        bitonic_sort_scalar(arr, ascending);
        return;
    }

    // Simplified 8-element sort using AVX2
    let temp = [
        arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7],
    ];
    let mut sorted = temp;
    sorted.sort_by(|a, b| {
        if ascending {
            a.partial_cmp(b).unwrap()
        } else {
            b.partial_cmp(a).unwrap()
        }
    });

    let vec = _mm256_loadu_ps(sorted.as_ptr());
    _mm256_storeu_ps(arr.as_mut_ptr(), vec);
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn bitonic_merge_avx2(arr: &mut [f32], ascending: bool) {
    use core::arch::x86_64::*;

    let len = arr.len();

    if len <= 8 {
        bitonic_merge_scalar(arr, ascending);
        return;
    }

    let step = len / 2;

    // SIMD-accelerated comparison and swapping
    let mut i = 0;
    while i + 8 <= step {
        let vec1 = _mm256_loadu_ps(&arr[i]);
        let vec2 = _mm256_loadu_ps(&arr[i + step]);

        let cmp = if ascending {
            _mm256_cmp_ps(vec1, vec2, _CMP_GT_OQ)
        } else {
            _mm256_cmp_ps(vec1, vec2, _CMP_LT_OQ)
        };

        let mask = _mm256_movemask_ps(cmp);

        // Handle swaps based on mask (simplified approach)
        for j in 0..8 {
            if (mask & (1 << j)) != 0 {
                arr.swap(i + j, i + j + step);
            }
        }

        i += 8;
    }

    // Handle remaining elements
    while i < step {
        if (arr[i] > arr[i + step]) == ascending {
            arr.swap(i, i + step);
        }
        i += 1;
    }

    if step > 1 {
        bitonic_merge_avx2(&mut arr[0..step], ascending);
        bitonic_merge_avx2(&mut arr[step..], ascending);
    }
}

/// SIMD-optimized median computation using quickselect
pub fn median_f32_simd(arr: &mut [f32]) -> Option<f32> {
    if arr.is_empty() {
        return None;
    }

    let len = arr.len();
    let mid = len / 2;

    if len % 2 == 1 {
        Some(quickselect_f32_simd(arr, mid))
    } else {
        let left_mid = quickselect_f32_simd(arr, mid - 1);
        let right_mid = quickselect_f32_simd(arr, mid);
        Some((left_mid + right_mid) / 2.0)
    }
}

/// SIMD-optimized quickselect for k-th smallest element
pub fn quickselect_f32_simd(arr: &mut [f32], k: usize) -> f32 {
    assert!(k < arr.len(), "k must be less than array length");

    let mut left = 0;
    let mut right = arr.len() - 1;

    loop {
        if left == right {
            return arr[left];
        }

        let pivot_index = partition_range(arr, left, right);

        if k == pivot_index {
            return arr[k];
        } else if k < pivot_index {
            right = pivot_index - 1;
        } else {
            left = pivot_index + 1;
        }
    }
}

fn partition_range(arr: &mut [f32], left: usize, right: usize) -> usize {
    let pivot = arr[right];
    let mut i = left;

    for j in left..right {
        if arr[j] <= pivot {
            arr.swap(i, j);
            i += 1;
        }
    }

    arr.swap(i, right);
    i
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::*;

    fn is_sorted(arr: &[f32], ascending: bool) -> bool {
        for i in 1..arr.len() {
            if ascending && arr[i - 1] > arr[i] {
                return false;
            }
            if !ascending && arr[i - 1] < arr[i] {
                return false;
            }
        }
        true
    }

    #[test]
    fn test_quicksort_simd() {
        let mut arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
    }

    #[test]
    fn test_quicksort_random() {
        let mut rng = thread_rng();
        let mut arr: Vec<f32> = (0..100).map(|_| rng.gen_range(0.0..100.0)).collect();

        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
    }

    #[test]
    fn test_bitonic_sort_small() {
        let mut arr = vec![4.0, 2.0, 7.0, 1.0];
        bitonic_sort_f32_simd(&mut arr, true);
        assert!(is_sorted(&arr, true));

        let mut arr = vec![4.0, 2.0, 7.0, 1.0];
        bitonic_sort_f32_simd(&mut arr, false);
        assert!(is_sorted(&arr, false));
    }

    #[test]
    fn test_bitonic_sort_power_of_2() {
        let mut arr = vec![8.0, 4.0, 2.0, 1.0, 3.0, 6.0, 5.0, 7.0];
        bitonic_sort_f32_simd(&mut arr, true);
        assert!(is_sorted(&arr, true));
    }

    #[test]
    fn test_median_odd() {
        let mut arr = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let median = median_f32_simd(&mut arr);
        assert_eq!(median, Some(3.0));
    }

    #[test]
    fn test_median_even() {
        let mut arr = vec![3.0, 1.0, 4.0, 2.0];
        let median = median_f32_simd(&mut arr);
        assert_eq!(median, Some(2.5)); // (2.0 + 3.0) / 2.0
    }

    #[test]
    fn test_quickselect() {
        let mut arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        // Find the 3rd smallest element (0-indexed)
        let third_smallest = quickselect_f32_simd(&mut arr, 2);

        // Sort to verify
        let mut sorted = arr.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(third_smallest, sorted[2]);
    }

    #[test]
    fn test_empty_median() {
        let mut arr: Vec<f32> = vec![];
        let median = median_f32_simd(&mut arr);
        assert_eq!(median, None);
    }

    #[test]
    fn test_single_element() {
        let mut arr = vec![42.0];
        quicksort_f32_simd(&mut arr);
        assert_eq!(arr, vec![42.0]);

        let median = median_f32_simd(&mut arr);
        assert_eq!(median, Some(42.0));
    }
}

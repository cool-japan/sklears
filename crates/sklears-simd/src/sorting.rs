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
        #[cfg(not(feature = "no-std"))]
        if crate::simd_feature_detected!("avx2") && arr.len() >= 16 {
            unsafe { quicksort_avx2(arr) };
            return;
        }
        if crate::simd_feature_detected!("sse2") && arr.len() >= 8 {
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
    // SSE2 lacks variable-index vector permute (_mm_permutevar_ps requires AVX),
    // so we use the proven scalar Lomuto partition while keeping the SIMD
    // insertion-sort base case for small sub-arrays.
    if arr.len() <= 8 {
        insertion_sort_simd_sse2(arr);
        return;
    }
    let pivot_index = partition_scalar(arr);
    quicksort_sse2(&mut arr[0..pivot_index]);
    quicksort_sse2(&mut arr[pivot_index + 1..]);
}

// Compile-time permutation LUT for AVX2 compress.
// COMPRESS_LUT[mask] is a permutation of [0..8] where the set-bit (≤pivot) lanes
// come first (positions 0..popcount(mask)), followed by the clear-bit (>pivot) lanes.
// Used with _mm256_permutevar8x32_ps to gather matching elements contiguously.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
const fn build_compress_lut() -> [[u32; 8]; 256] {
    let mut lut = [[0u32; 8]; 256];
    let mut m: usize = 0;
    while m < 256 {
        let mut count_set: usize = 0;
        let mut b: usize = 0;
        while b < 8 {
            if (m >> b) & 1 == 1 {
                count_set += 1;
            }
            b += 1;
        }
        let mut cur_lo: usize = 0;
        let mut cur_hi: usize = count_set;
        let mut b2: usize = 0;
        while b2 < 8 {
            if (m >> b2) & 1 == 1 {
                lut[m][cur_lo] = b2 as u32;
                cur_lo += 1;
            } else {
                lut[m][cur_hi] = b2 as u32;
                cur_hi += 1;
            }
            b2 += 1;
        }
        m += 1;
    }
    lut
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
static COMPRESS_LUT: [[u32; 8]; 256] = build_compress_lut();

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
#[target_feature(enable = "avx2")]
unsafe fn quicksort_avx2(arr: &mut [f32]) {
    if arr.len() <= 1 {
        return;
    }
    let len = arr.len();
    let mut le_buf: Vec<f32> = Vec::with_capacity(len);
    let mut gt_buf: Vec<f32> = Vec::with_capacity(len);
    quicksort_avx2_impl(arr, &mut le_buf, &mut gt_buf);
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
#[target_feature(enable = "avx2")]
unsafe fn quicksort_avx2_impl(arr: &mut [f32], le_buf: &mut Vec<f32>, gt_buf: &mut Vec<f32>) {
    let len = arr.len();
    if len <= 1 {
        return;
    }
    if len <= 16 {
        insertion_sort_simd_avx2(arr);
        return;
    }
    let pivot_pos = partition_avx2_buffered(arr, le_buf, gt_buf);
    let (left, rest) = arr.split_at_mut(pivot_pos);
    let right = &mut rest[1..];
    quicksort_avx2_impl(left, le_buf, gt_buf);
    quicksort_avx2_impl(right, le_buf, gt_buf);
}

// Lomuto-style partition using AVX2 compress: in each 8-lane pass, elements ≤ pivot
// are gathered contiguously via _mm256_permutevar8x32_ps + LUT, then buffered.
// The array is reassembled as [≤pivot elements | pivot | >pivot elements].
// Buffers are allocated once by the entry point and reused across all recursive calls.
#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
#[target_feature(enable = "avx2")]
unsafe fn partition_avx2_buffered(
    arr: &mut [f32],
    le_buf: &mut Vec<f32>,
    gt_buf: &mut Vec<f32>,
) -> usize {
    use core::arch::x86_64::*;

    let len = arr.len();
    let pivot = arr[len - 1];
    let pivot_vec = _mm256_set1_ps(pivot);

    le_buf.clear();
    gt_buf.clear();

    let mut i = 0;

    // Process 8 elements per iteration (all elements before the pivot at arr[len-1])
    while i + 8 < len {
        let data_vec = _mm256_loadu_ps(arr.as_ptr().add(i));
        let cmp = _mm256_cmp_ps(data_vec, pivot_vec, _CMP_LE_OQ);
        let mask = _mm256_movemask_ps(cmp) as usize;
        let count_le = mask.count_ones() as usize;
        let count_gt = 8 - count_le;

        // Permute so the ≤pivot lanes land in the low prefix
        let le_perm = _mm256_loadu_si256(COMPRESS_LUT[mask].as_ptr() as *const __m256i);
        let le_result = _mm256_permutevar8x32_ps(data_vec, le_perm);
        let mut tmp = [0.0f32; 8];
        _mm256_storeu_ps(tmp.as_mut_ptr(), le_result);
        le_buf.extend_from_slice(&tmp[..count_le]);

        // Permute so the >pivot lanes land in the low prefix
        let gt_mask = (!mask) & 0xFF;
        let gt_perm = _mm256_loadu_si256(COMPRESS_LUT[gt_mask].as_ptr() as *const __m256i);
        let gt_result = _mm256_permutevar8x32_ps(data_vec, gt_perm);
        _mm256_storeu_ps(tmp.as_mut_ptr(), gt_result);
        gt_buf.extend_from_slice(&tmp[..count_gt]);

        i += 8;
    }

    // Scalar tail for any remaining elements before the pivot
    while i < len - 1 {
        if arr[i] <= pivot {
            le_buf.push(arr[i]);
        } else {
            gt_buf.push(arr[i]);
        }
        i += 1;
    }

    // Reassemble: [ ≤pivot elements | pivot | >pivot elements ]
    let pivot_pos = le_buf.len();
    arr[..pivot_pos].copy_from_slice(le_buf.as_slice());
    arr[pivot_pos] = pivot;
    let gt_start = pivot_pos + 1;
    arr[gt_start..gt_start + gt_buf.len()].copy_from_slice(gt_buf.as_slice());

    pivot_pos
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
        if j >= 4 {
            let vec = _mm_loadu_ps(&arr[j - 4]);
            let key_vec = _mm_set1_ps(key);
            let cmp = _mm_cmpgt_ps(vec, key_vec);
            let mask = _mm_movemask_ps(cmp);

            if mask != 0 {
                // Shift elements one by one (scalar fallback for shifting)
                while j > 0 && arr[j - 1] > key {
                    arr[j] = arr[j - 1];
                    j -= 1;
                }
            }
        }

        // Handle remaining elements with scalar code
        while j > 0 && arr[j - 1] > key {
            arr[j] = arr[j - 1];
            j -= 1;
        }
        arr[j] = key;
    }
}

#[cfg(all(
    any(target_arch = "x86", target_arch = "x86_64"),
    not(feature = "no-std")
))]
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
        if j >= 8 {
            let vec = _mm256_loadu_ps(&arr[j - 8]);
            let key_vec = _mm256_set1_ps(key);
            let cmp = _mm256_cmp_ps(vec, key_vec, _CMP_GT_OQ);
            let mask = _mm256_movemask_ps(cmp);

            if mask != 0 {
                // Shift elements one by one (scalar fallback for shifting)
                while j > 0 && arr[j - 1] > key {
                    arr[j] = arr[j - 1];
                    j -= 1;
                }
            }
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
        if crate::simd_feature_detected!("avx2") && len >= 8 {
            unsafe { bitonic_sort_avx2(arr, ascending) };
            return;
        } else if crate::simd_feature_detected!("sse2") && len >= 4 {
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

    // Implement 4-element bitonic sort with SSE2
    // This is a simplified version - a full implementation would be more complex
    let temp = [arr[0], arr[1], arr[2], arr[3]];
    let mut sorted = temp;
    sorted.sort_by(|a, b| {
        if ascending {
            a.partial_cmp(b).expect("operation should succeed")
        } else {
            b.partial_cmp(a).expect("operation should succeed")
        }
    });

    let vec = _mm_loadu_ps(sorted.as_ptr());
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
            a.partial_cmp(b).expect("operation should succeed")
        } else {
            b.partial_cmp(a).expect("operation should succeed")
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
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;
    use scirs2_core::random::prelude::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

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
        let mut arr: Vec<f32> = (0..100).map(|_| rng.random_range(0.0..100.0)).collect();

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
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
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

    fn multiset_eq(a: &[f32], b: &[f32]) -> bool {
        let mut va: Vec<u32> = a.iter().map(|x| x.to_bits()).collect();
        let mut vb: Vec<u32> = b.iter().map(|x| x.to_bits()).collect();
        va.sort_unstable();
        vb.sort_unstable();
        va == vb
    }

    #[test]
    fn test_quicksort_already_sorted() {
        let mut arr: Vec<f32> = (0..50).map(|i| i as f32).collect();
        let original = arr.clone();
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
        assert!(multiset_eq(&arr, &original));
    }

    #[test]
    fn test_quicksort_reverse_sorted() {
        let mut arr: Vec<f32> = (0..50).rev().map(|i| i as f32).collect();
        let original = arr.clone();
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
        assert!(multiset_eq(&arr, &original));
    }

    #[test]
    fn test_quicksort_all_equal() {
        let mut arr = vec![7.0f32; 100];
        let original = arr.clone();
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
        assert!(multiset_eq(&arr, &original));
    }

    #[test]
    fn test_quicksort_heavy_duplicates() {
        let mut rng = thread_rng();
        // Only 3 distinct values among 200 elements: lots of ties in the partition
        let mut arr: Vec<f32> = (0..200)
            .map(|_| [1.0f32, 2.0, 3.0][rng.random_range(0usize..3)])
            .collect();
        let original = arr.clone();
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
        assert!(multiset_eq(&arr, &original));
    }

    #[test]
    fn test_quicksort_non_multiple_of_8() {
        let mut rng = thread_rng();
        // Sizes that don't align to the 8-lane AVX2 width
        for size in [17usize, 23, 31, 41, 97, 103] {
            let mut arr: Vec<f32> = (0..size)
                .map(|_| rng.random_range(0.0f32..1000.0))
                .collect();
            let original = arr.clone();
            quicksort_f32_simd(&mut arr);
            assert!(is_sorted(&arr, true), "size {size} not sorted");
            assert!(multiset_eq(&arr, &original), "size {size} multiset changed");
        }
    }

    #[test]
    fn test_quicksort_large() {
        let mut rng = thread_rng();
        let mut arr: Vec<f32> = (0..1000)
            .map(|_| rng.random_range(0.0f32..10000.0))
            .collect();
        let original = arr.clone();
        quicksort_f32_simd(&mut arr);
        assert!(is_sorted(&arr, true));
        assert!(multiset_eq(&arr, &original));
    }
}

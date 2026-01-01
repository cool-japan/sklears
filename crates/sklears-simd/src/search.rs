//! SIMD-optimized search algorithms
//!
//! This module provides vectorized implementations of search algorithms
//! including binary search, linear search, and approximate nearest neighbor search.

use crate::distance::euclidean_distance;

#[cfg(feature = "no-std")]
use alloc::collections::{BTreeMap as HashMap, BTreeSet as HashSet};
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::collections::{HashMap, HashSet};

#[cfg(feature = "no-std")]
use core::cmp::Ordering;
#[cfg(not(feature = "no-std"))]
use std::cmp::Ordering;

/// SIMD-optimized binary search for sorted f32 arrays
/// Returns the index where target is found, or None if not found
pub fn binary_search_f32_simd(arr: &[f32], target: f32) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && arr.len() >= 16 {
            return unsafe { binary_search_avx2(arr, target) };
        } else if crate::simd_feature_detected!("sse2") && arr.len() >= 8 {
            return unsafe { binary_search_sse2(arr, target) };
        }
    }

    binary_search_scalar(arr, target)
}

fn binary_search_scalar(arr: &[f32], target: f32) -> Option<usize> {
    let mut left = 0;
    let mut right = arr.len();

    while left < right {
        let mid = left + (right - left) / 2;

        match arr[mid].partial_cmp(&target) {
            Some(Ordering::Equal) => return Some(mid),
            Some(Ordering::Less) => left = mid + 1,
            Some(Ordering::Greater) => right = mid,
            None => return None, // NaN handling
        }
    }

    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn binary_search_sse2(arr: &[f32], target: f32) -> Option<usize> {
    use core::arch::x86_64::*;

    let mut left = 0;
    let mut right = arr.len();
    let target_vec = _mm_set1_ps(target);

    while left < right {
        let mid = left + (right - left) / 2;

        // Try SIMD comparison for small ranges
        if right - left <= 4 && left + 4 <= arr.len() {
            let vec = _mm_loadu_ps(&arr[left]);
            let eq_mask = _mm_cmpeq_ps(vec, target_vec);
            let mask = _mm_movemask_ps(eq_mask);

            if mask != 0 {
                // Found target, determine exact position
                for i in 0..4 {
                    if (mask & (1 << i)) != 0 {
                        return Some(left + i);
                    }
                }
            }

            // If not found in SIMD range, fall back to scalar
            return binary_search_scalar(&arr[left..right], target).map(|idx| left + idx);
        }

        // Regular binary search step
        match arr[mid].partial_cmp(&target) {
            Some(Ordering::Equal) => return Some(mid),
            Some(Ordering::Less) => left = mid + 1,
            Some(Ordering::Greater) => right = mid,
            None => return None,
        }
    }

    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn binary_search_avx2(arr: &[f32], target: f32) -> Option<usize> {
    use core::arch::x86_64::*;

    let mut left = 0;
    let mut right = arr.len();
    let target_vec = _mm256_set1_ps(target);

    while left < right {
        let mid = left + (right - left) / 2;

        // Try SIMD comparison for small ranges
        if right - left <= 8 && left + 8 <= arr.len() {
            let vec = _mm256_loadu_ps(&arr[left]);
            let eq_mask = _mm256_cmp_ps(vec, target_vec, _CMP_EQ_OQ);
            let mask = _mm256_movemask_ps(eq_mask);

            if mask != 0 {
                // Found target, determine exact position
                for i in 0..8 {
                    if (mask & (1 << i)) != 0 {
                        return Some(left + i);
                    }
                }
            }

            // If not found in SIMD range, fall back to scalar
            return binary_search_scalar(&arr[left..right], target).map(|idx| left + idx);
        }

        // Regular binary search step
        match arr[mid].partial_cmp(&target) {
            Some(Ordering::Equal) => return Some(mid),
            Some(Ordering::Less) => left = mid + 1,
            Some(Ordering::Greater) => right = mid,
            None => return None,
        }
    }

    None
}

/// SIMD-optimized linear search for unsorted arrays
/// Returns the first index where target is found, or None if not found
pub fn linear_search_f32_simd(arr: &[f32], target: f32) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { linear_search_avx2(arr, target) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { linear_search_sse2(arr, target) };
        }
    }

    linear_search_scalar(arr, target)
}

fn linear_search_scalar(arr: &[f32], target: f32) -> Option<usize> {
    for (i, &value) in arr.iter().enumerate() {
        if value == target {
            return Some(i);
        }
    }
    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn linear_search_sse2(arr: &[f32], target: f32) -> Option<usize> {
    use core::arch::x86_64::*;

    let target_vec = _mm_set1_ps(target);
    let mut i = 0;

    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        let eq_mask = _mm_cmpeq_ps(vec, target_vec);
        let mask = _mm_movemask_ps(eq_mask);

        if mask != 0 {
            // Found target, determine exact position
            for j in 0..4 {
                if (mask & (1 << j)) != 0 {
                    return Some(i + j);
                }
            }
        }

        i += 4;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] == target {
            return Some(i);
        }
        i += 1;
    }

    None
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn linear_search_avx2(arr: &[f32], target: f32) -> Option<usize> {
    use core::arch::x86_64::*;

    let target_vec = _mm256_set1_ps(target);
    let mut i = 0;

    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        let eq_mask = _mm256_cmp_ps(vec, target_vec, _CMP_EQ_OQ);
        let mask = _mm256_movemask_ps(eq_mask);

        if mask != 0 {
            // Found target, determine exact position
            for j in 0..8 {
                if (mask & (1 << j)) != 0 {
                    return Some(i + j);
                }
            }
        }

        i += 8;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] == target {
            return Some(i);
        }
        i += 1;
    }

    None
}

/// Result for nearest neighbor search
#[derive(Debug, Clone, PartialEq)]
pub struct NearestNeighborResult {
    pub index: usize,
    pub distance: f32,
}

/// SIMD-optimized k-nearest neighbors search
/// Returns the k nearest neighbors to the query point
pub fn k_nearest_neighbors_simd(
    points: &[&[f32]],
    query: &[f32],
    k: usize,
) -> Vec<NearestNeighborResult> {
    if points.is_empty() || k == 0 {
        return Vec::new();
    }

    let k = k.min(points.len());
    let mut distances: Vec<(usize, f32)> = Vec::with_capacity(points.len());

    // Compute distances to all points
    for (i, point) in points.iter().enumerate() {
        let distance = euclidean_distance(query, point);
        distances.push((i, distance));
    }

    // Sort by distance and take k smallest
    distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

    distances
        .into_iter()
        .take(k)
        .map(|(index, distance)| NearestNeighborResult { index, distance })
        .collect()
}

/// SIMD-optimized approximate nearest neighbor search using LSH (Locality Sensitive Hashing)
/// This is a simplified implementation for demonstration
pub struct LSHTable {
    tables: Vec<LSHHashTable>,
    num_tables: usize,
    hash_size: usize,
}

struct LSHHashTable {
    buckets: HashMap<u64, Vec<usize>>,
    random_vectors: Vec<Vec<f32>>,
}

impl LSHTable {
    /// Create a new LSH table for approximate nearest neighbor search
    pub fn new(dimensions: usize, num_tables: usize, hash_size: usize) -> Self {
        let mut tables = Vec::with_capacity(num_tables);

        for _ in 0..num_tables {
            let mut random_vectors = Vec::with_capacity(hash_size);

            // Generate random unit vectors for hashing
            for _ in 0..hash_size {
                let mut vec = Vec::with_capacity(dimensions);
                let mut sum_squares = 0.0;

                // Generate random vector
                use scirs2_core::random::thread_rng;
                let mut rng = thread_rng();
                for _ in 0..dimensions {
                    let val: f32 = rng.random::<f32>() - 0.5;
                    vec.push(val);
                    sum_squares += val * val;
                }

                // Normalize to unit vector
                let norm = sum_squares.sqrt();
                if norm > 0.0 {
                    for val in &mut vec {
                        *val /= norm;
                    }
                }

                random_vectors.push(vec);
            }

            tables.push(LSHHashTable {
                buckets: HashMap::new(),
                random_vectors,
            });
        }

        LSHTable {
            tables,
            num_tables,
            hash_size,
        }
    }

    /// Add a point to the LSH table
    pub fn add_point(&mut self, point: &[f32], index: usize) {
        for i in 0..self.tables.len() {
            let hash = self.hash_point(&self.tables[i], point);
            self.tables[i].buckets.entry(hash).or_default().push(index);
        }
    }

    /// Query for approximate nearest neighbors
    pub fn query(&self, point: &[f32], max_candidates: usize) -> Vec<usize> {
        let mut candidates = HashSet::new();

        for table in &self.tables {
            let hash = self.hash_point(table, point);

            if let Some(bucket) = table.buckets.get(&hash) {
                for &index in bucket {
                    candidates.insert(index);
                    if candidates.len() >= max_candidates {
                        break;
                    }
                }
            }

            if candidates.len() >= max_candidates {
                break;
            }
        }

        candidates.into_iter().collect()
    }

    fn hash_point(&self, table: &LSHHashTable, point: &[f32]) -> u64 {
        let mut hash = 0u64;

        for (i, random_vec) in table.random_vectors.iter().enumerate() {
            // Compute dot product
            let dot_product = crate::vector::dot_product(point, random_vec);

            // Use sign of dot product as hash bit
            if dot_product >= 0.0 {
                hash |= 1u64 << i;
            }
        }

        hash
    }
}

/// SIMD-optimized range search
/// Returns all points within a specified distance of the query point
pub fn range_search_simd(
    points: &[&[f32]],
    query: &[f32],
    radius: f32,
) -> Vec<NearestNeighborResult> {
    let mut results = Vec::new();
    let _radius_squared = radius * radius;

    for (i, point) in points.iter().enumerate() {
        let distance = euclidean_distance(query, point);
        if distance <= radius {
            results.push(NearestNeighborResult { index: i, distance });
        }
    }

    // Sort by distance
    results.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(Ordering::Equal)
    });

    results
}

/// SIMD-optimized argmax - find index of maximum element
pub fn argmax_f32_simd(arr: &[f32]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && arr.len() >= 8 {
            return Some(unsafe { argmax_avx2(arr) });
        } else if crate::simd_feature_detected!("sse2") && arr.len() >= 4 {
            return Some(unsafe { argmax_sse2(arr) });
        }
    }

    argmax_scalar(arr)
}

fn argmax_scalar(arr: &[f32]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    let mut max_idx = 0;
    let mut max_val = arr[0];

    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val > max_val {
            max_val = val;
            max_idx = i;
        }
    }

    Some(max_idx)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn argmax_sse2(arr: &[f32]) -> usize {
    use core::arch::x86_64::*;

    let mut max_val = arr[0];
    let mut max_idx = 0;
    let mut i = 0;

    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vec);

        for j in 0..4 {
            if temp[j] > max_val {
                max_val = temp[j];
                max_idx = i + j;
            }
        }

        i += 4;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] > max_val {
            max_val = arr[i];
            max_idx = i;
        }
        i += 1;
    }

    max_idx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn argmax_avx2(arr: &[f32]) -> usize {
    use core::arch::x86_64::*;

    let mut max_val = arr[0];
    let mut max_idx = 0;
    let mut i = 0;

    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), vec);

        for j in 0..8 {
            if temp[j] > max_val {
                max_val = temp[j];
                max_idx = i + j;
            }
        }

        i += 8;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] > max_val {
            max_val = arr[i];
            max_idx = i;
        }
        i += 1;
    }

    max_idx
}

/// SIMD-optimized argmin - find index of minimum element
pub fn argmin_f32_simd(arr: &[f32]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") && arr.len() >= 8 {
            return Some(unsafe { argmin_avx2(arr) });
        } else if crate::simd_feature_detected!("sse2") && arr.len() >= 4 {
            return Some(unsafe { argmin_sse2(arr) });
        }
    }

    argmin_scalar(arr)
}

fn argmin_scalar(arr: &[f32]) -> Option<usize> {
    if arr.is_empty() {
        return None;
    }

    let mut min_idx = 0;
    let mut min_val = arr[0];

    for (i, &val) in arr.iter().enumerate().skip(1) {
        if val < min_val {
            min_val = val;
            min_idx = i;
        }
    }

    Some(min_idx)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn argmin_sse2(arr: &[f32]) -> usize {
    use core::arch::x86_64::*;

    let mut min_val = arr[0];
    let mut min_idx = 0;
    let mut i = 0;

    while i + 4 <= arr.len() {
        let vec = _mm_loadu_ps(&arr[i]);
        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), vec);

        for j in 0..4 {
            if temp[j] < min_val {
                min_val = temp[j];
                min_idx = i + j;
            }
        }

        i += 4;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] < min_val {
            min_val = arr[i];
            min_idx = i;
        }
        i += 1;
    }

    min_idx
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn argmin_avx2(arr: &[f32]) -> usize {
    use core::arch::x86_64::*;

    let mut min_val = arr[0];
    let mut min_idx = 0;
    let mut i = 0;

    while i + 8 <= arr.len() {
        let vec = _mm256_loadu_ps(&arr[i]);
        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), vec);

        for j in 0..8 {
            if temp[j] < min_val {
                min_val = temp[j];
                min_idx = i + j;
            }
        }

        i += 8;
    }

    // Handle remaining elements
    while i < arr.len() {
        if arr[i] < min_val {
            min_val = arr[i];
            min_idx = i;
        }
        i += 1;
    }

    min_idx
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::vec;

    #[test]
    fn test_binary_search_found() {
        let arr = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        assert_eq!(binary_search_f32_simd(&arr, 7.0), Some(3));
        assert_eq!(binary_search_f32_simd(&arr, 1.0), Some(0));
        assert_eq!(binary_search_f32_simd(&arr, 15.0), Some(7));
    }

    #[test]
    fn test_binary_search_not_found() {
        let arr = vec![1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0];
        assert_eq!(binary_search_f32_simd(&arr, 6.0), None);
        assert_eq!(binary_search_f32_simd(&arr, 0.0), None);
        assert_eq!(binary_search_f32_simd(&arr, 16.0), None);
    }

    #[test]
    fn test_linear_search() {
        let arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0];
        assert_eq!(linear_search_f32_simd(&arr, 4.0), Some(2));
        assert_eq!(linear_search_f32_simd(&arr, 1.0), Some(1)); // First occurrence
        assert_eq!(linear_search_f32_simd(&arr, 8.0), None);
    }

    #[test]
    fn test_k_nearest_neighbors() {
        let p1 = [1.0, 1.0];
        let p2 = [2.0, 2.0];
        let p3 = [5.0, 5.0];
        let p4 = [6.0, 6.0];
        let points = vec![&p1[..], &p2[..], &p3[..], &p4[..]];

        let query = [1.5, 1.5];
        let neighbors = k_nearest_neighbors_simd(&points, &query, 2);

        assert_eq!(neighbors.len(), 2);
        // Should return the two closest points (p1 and p2)
        assert!(neighbors[0].index < 2);
        assert!(neighbors[1].index < 2);
    }

    #[test]
    fn test_range_search() {
        let p1 = [1.0, 1.0];
        let p2 = [2.0, 2.0];
        let p3 = [5.0, 5.0];
        let points = vec![&p1[..], &p2[..], &p3[..]];

        let query = [1.5, 1.5];
        let results = range_search_simd(&points, &query, 1.0);

        // Should find p1 and p2 within distance 1.0
        assert!(results.len() >= 1);
        assert!(results.iter().all(|r| r.distance <= 1.0));
    }

    #[test]
    fn test_argmax() {
        let arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert_eq!(argmax_f32_simd(&arr), Some(5)); // Index of 9.0
    }

    #[test]
    fn test_argmin() {
        let arr = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        assert_eq!(argmin_f32_simd(&arr), Some(1)); // Index of first 1.0
    }

    #[test]
    fn test_empty_arrays() {
        let empty: Vec<f32> = vec![];
        assert_eq!(binary_search_f32_simd(&empty, 1.0), None);
        assert_eq!(linear_search_f32_simd(&empty, 1.0), None);
        assert_eq!(argmax_f32_simd(&empty), None);
        assert_eq!(argmin_f32_simd(&empty), None);
    }

    #[test]
    fn test_single_element() {
        let arr = vec![42.0];
        assert_eq!(binary_search_f32_simd(&arr, 42.0), Some(0));
        assert_eq!(linear_search_f32_simd(&arr, 42.0), Some(0));
        assert_eq!(argmax_f32_simd(&arr), Some(0));
        assert_eq!(argmin_f32_simd(&arr), Some(0));
    }

    #[test]
    fn test_lsh_table() {
        let mut lsh = LSHTable::new(2, 3, 4);

        // Add some points
        let p1 = vec![1.0, 1.0];
        let p2 = vec![2.0, 2.0];
        let p3 = vec![10.0, 10.0];

        lsh.add_point(&p1, 0);
        lsh.add_point(&p2, 1);
        lsh.add_point(&p3, 2);

        // Query for similar points
        let query = vec![1.1, 1.1];
        let candidates = lsh.query(&query, 5);

        // Should return some candidates
        assert!(!candidates.is_empty());
    }
}

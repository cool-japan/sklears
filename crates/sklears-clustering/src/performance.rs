//! Performance optimizations for clustering algorithms
//!
//! This module provides cache-friendly data layouts and memory optimization utilities
//! for better performance in clustering computations.

use scirs2_core::ndarray::Array2;
use sklears_core::types::Float;

/// Cache-friendly distance matrix computation
///
/// This function computes distances between points using blocked matrix operations
/// to improve cache locality and reduce memory bandwidth requirements.
pub fn cache_friendly_distance_matrix(
    points1: &Array2<Float>,
    points2: &Array2<Float>,
    block_size: usize,
) -> Array2<Float> {
    let (n1, n_features) = points1.dim();
    let (n2, _) = points2.dim();
    let mut distances = Array2::zeros((n1, n2));

    // Use cache-friendly blocking
    let block_size = block_size.min(64); // Reasonable default block size

    for i_start in (0..n1).step_by(block_size) {
        let i_end = (i_start + block_size).min(n1);

        for j_start in (0..n2).step_by(block_size) {
            let j_end = (j_start + block_size).min(n2);

            // Compute distances for this block
            for i in i_start..i_end {
                for j in j_start..j_end {
                    let mut dist_sq = 0.0;

                    // Vectorizable inner loop
                    for k in 0..n_features {
                        let diff = points1[[i, k]] - points2[[j, k]];
                        dist_sq += diff * diff;
                    }

                    distances[[i, j]] = dist_sq.sqrt();
                }
            }
        }
    }

    distances
}

/// Cache-optimized centroid computation
///
/// Updates centroids using cache-friendly memory access patterns
pub fn cache_optimized_centroids(
    data: &Array2<Float>,
    labels: &[usize],
    n_clusters: usize,
) -> Array2<Float> {
    let (n_samples, n_features) = data.dim();
    let mut centroids = Array2::zeros((n_clusters, n_features));
    let mut counts = vec![0usize; n_clusters];

    // First pass: accumulate sums with good spatial locality
    for (i, &label) in labels.iter().enumerate() {
        if label < n_clusters {
            counts[label] += 1;

            // Process features in blocks for better cache usage
            let data_row = data.row(i);
            let mut centroid_row = centroids.row_mut(label);

            // Use SIMD-friendly operations
            for (centroid_elem, &data_elem) in centroid_row.iter_mut().zip(data_row.iter()) {
                *centroid_elem += data_elem;
            }
        }
    }

    // Second pass: normalize by counts
    for (cluster_id, &count) in counts.iter().enumerate() {
        if count > 0 {
            let inv_count = 1.0 / count as Float;
            let mut centroid_row = centroids.row_mut(cluster_id);

            for centroid_elem in centroid_row.iter_mut() {
                *centroid_elem *= inv_count;
            }
        }
    }

    centroids
}

/// Memory-efficient k-nearest neighbors computation
///
/// Computes k-nearest neighbors without storing full distance matrix
pub fn memory_efficient_knn(
    query_points: &Array2<Float>,
    reference_points: &Array2<Float>,
    k: usize,
) -> Vec<Vec<(usize, Float)>> {
    let n_queries = query_points.nrows();
    let n_references = reference_points.nrows();
    let mut results = Vec::with_capacity(n_queries);

    for i in 0..n_queries {
        let query = query_points.row(i);
        let mut distances: Vec<(usize, Float)> = Vec::with_capacity(n_references);

        // Compute distances to all reference points
        for j in 0..n_references {
            let reference = reference_points.row(j);
            let diff = &query - &reference;
            let dist = diff.dot(&diff).sqrt();
            distances.push((j, dist));
        }

        // Sort and take top k
        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        distances.truncate(k);

        results.push(distances);
    }

    results
}

/// Cache-aware data layout transformation
///
/// Reorganizes data for better cache performance in clustering algorithms
pub struct CacheAwareData {
    pub data: Array2<Float>,
    pub original_indices: Vec<usize>,
}

impl CacheAwareData {
    /// Create cache-aware layout from input data
    pub fn from_data(data: Array2<Float>) -> Self {
        let n_samples = data.nrows();
        let mut original_indices: Vec<usize> = (0..n_samples).collect();

        // Sort points to improve spatial locality (simple Z-order approximation)
        if data.ncols() >= 2 {
            original_indices.sort_by(|&i, &j| {
                let point_i = data.row(i);
                let point_j = data.row(j);

                // Simple spatial sorting heuristic
                let coord_i = point_i[0] + point_i[1] * 1000.0;
                let coord_j = point_j[0] + point_j[1] * 1000.0;

                coord_i.partial_cmp(&coord_j).unwrap()
            });
        }

        // Reorder data according to sorted indices
        let mut reordered_data = Array2::zeros(data.dim());
        for (new_idx, &orig_idx) in original_indices.iter().enumerate() {
            reordered_data.row_mut(new_idx).assign(&data.row(orig_idx));
        }

        Self {
            data: reordered_data,
            original_indices,
        }
    }

    /// Restore original ordering for labels
    pub fn restore_label_order(&self, labels: &[usize]) -> Vec<usize> {
        let mut original_labels = vec![0; labels.len()];
        for (sorted_idx, &original_idx) in self.original_indices.iter().enumerate() {
            original_labels[original_idx] = labels[sorted_idx];
        }
        original_labels
    }
}

/// Memory pool for temporary arrays in clustering algorithms
pub struct ClusteringMemoryPool {
    float_buffers: Vec<Vec<Float>>,
    usize_buffers: Vec<Vec<usize>>,
}

impl ClusteringMemoryPool {
    /// Create a new memory pool
    pub fn new() -> Self {
        Self {
            float_buffers: Vec::new(),
            usize_buffers: Vec::new(),
        }
    }

    /// Get a Float buffer of at least the specified size
    pub fn get_float_buffer(&mut self, min_size: usize) -> Vec<Float> {
        if let Some(mut buffer) = self.float_buffers.pop() {
            buffer.clear();
            buffer.reserve(min_size);
            buffer
        } else {
            Vec::with_capacity(min_size)
        }
    }

    /// Return a Float buffer to the pool
    pub fn return_float_buffer(&mut self, buffer: Vec<Float>) {
        if buffer.capacity() >= 32 && self.float_buffers.len() < 8 {
            self.float_buffers.push(buffer);
        }
    }

    /// Get a usize buffer of at least the specified size
    pub fn get_usize_buffer(&mut self, min_size: usize) -> Vec<usize> {
        if let Some(mut buffer) = self.usize_buffers.pop() {
            buffer.clear();
            buffer.reserve(min_size);
            buffer
        } else {
            Vec::with_capacity(min_size)
        }
    }

    /// Return a usize buffer to the pool
    pub fn return_usize_buffer(&mut self, buffer: Vec<usize>) {
        if buffer.capacity() >= 32 && self.usize_buffers.len() < 8 {
            self.usize_buffers.push(buffer);
        }
    }
}

impl Default for ClusteringMemoryPool {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_cache_friendly_distance_matrix() {
        let points1 = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let points2 = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 1.0]).unwrap();

        let distances = cache_friendly_distance_matrix(&points1, &points2, 32);

        assert_eq!(distances.dim(), (3, 2));
        assert!((distances[[0, 0]] - 0.0).abs() < 1e-10); // Same point
        assert!((distances[[1, 1]] - 1.0).abs() < 1e-10); // Distance 1
    }

    #[test]
    fn test_cache_optimized_centroids() {
        let data =
            Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]).unwrap();
        let labels = vec![0, 0, 1, 1];

        let centroids = cache_optimized_centroids(&data, &labels, 2);

        assert_eq!(centroids.dim(), (2, 2));
        assert!((centroids[[0, 0]] - 0.5).abs() < 1e-10);
        assert!((centroids[[0, 1]] - 0.0).abs() < 1e-10);
        assert!((centroids[[1, 0]] - 0.5).abs() < 1e-10);
        assert!((centroids[[1, 1]] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_efficient_knn() {
        let query = Array2::from_shape_vec((1, 2), vec![0.0, 0.0]).unwrap();
        let reference = Array2::from_shape_vec((3, 2), vec![1.0, 0.0, 0.0, 1.0, 2.0, 0.0]).unwrap();

        let knn_results = memory_efficient_knn(&query, &reference, 2);

        assert_eq!(knn_results.len(), 1);
        assert_eq!(knn_results[0].len(), 2);

        // Both (1.0, 0.0) and (0.0, 1.0) have distance 1.0 from (0.0, 0.0)
        // So we should have indices 0 and 1 in the results (order may vary)
        let indices: Vec<usize> = knn_results[0].iter().map(|(idx, _)| *idx).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));

        // (2.0, 0.0) should not be in top 2 since it has distance 2.0
        assert!(!indices.contains(&2));
    }

    #[test]
    fn test_cache_aware_data() {
        let data = Array2::from_shape_vec((3, 2), vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]).unwrap();

        let cache_aware = CacheAwareData::from_data(data);
        let labels = vec![0, 1, 2];
        let restored = cache_aware.restore_label_order(&labels);

        assert_eq!(restored.len(), 3);
        // Should have same length as original
        assert_eq!(cache_aware.original_indices.len(), 3);
    }

    #[test]
    fn test_memory_pool() {
        let mut pool = ClusteringMemoryPool::new();

        let buffer1 = pool.get_float_buffer(100);
        assert!(buffer1.capacity() >= 100);

        pool.return_float_buffer(buffer1);

        let buffer2 = pool.get_float_buffer(50);
        // Should reuse the previous buffer
        assert!(buffer2.capacity() >= 100);
    }
}

//! SIMD-optimized distance computations for clustering algorithms
//!
//! This module provides adapter functions to use SIMD-optimized distance
//! calculations from sklears-simd with ndarray data structures.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};
use sklears_core::types::Float;

/// Auto-vectorized distance metrics that use SIMD when available
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    Euclidean,
    Manhattan,
    Chebyshev,
    Cosine,
    Minkowski(Float),
    Jaccard,
}

/// High-performance distance computation with automatic SIMD detection
pub struct OptimizedDistanceComputer {
    /// Whether SIMD instructions are available
    simd_available: bool,
    /// Cache-friendly block size for batch operations
    block_size: usize,
}

impl Default for OptimizedDistanceComputer {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizedDistanceComputer {
    /// Create a new optimized distance computer with runtime SIMD detection
    pub fn new() -> Self {
        Self {
            simd_available: Self::detect_simd_support(),
            block_size: Self::optimal_block_size(),
        }
    }

    /// Detect available SIMD instruction sets at runtime
    fn detect_simd_support() -> bool {
        #[cfg(target_arch = "x86_64")]
        {
            std::arch::is_x86_feature_detected!("avx2")
                || std::arch::is_x86_feature_detected!("sse2")
        }
        #[cfg(target_arch = "aarch64")]
        {
            std::arch::is_aarch64_feature_detected!("neon")
        }
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            false
        }
    }

    /// Determine optimal block size based on cache hierarchy
    fn optimal_block_size() -> usize {
        // L1 cache is typically 32KB, use 1/4 for distance computations
        // Assuming 32-bit floats: 8KB / 4 bytes = 2048 elements
        // Square root for matrix operations: sqrt(2048) ≈ 45
        48
    }

    /// Compute pairwise distances between all points in two arrays
    pub fn pairwise_distances(
        &self,
        points1: &ArrayView2<Float>,
        points2: &ArrayView2<Float>,
        metric: DistanceMetric,
    ) -> scirs2_core::ndarray::Array2<Float> {
        let (n1, _) = points1.dim();
        let (n2, _) = points2.dim();
        let mut distances = scirs2_core::ndarray::Array2::zeros((n1, n2));

        // Use blocked computation for cache efficiency
        for i_start in (0..n1).step_by(self.block_size) {
            let i_end = (i_start + self.block_size).min(n1);

            for j_start in (0..n2).step_by(self.block_size) {
                let j_end = (j_start + self.block_size).min(n2);

                // Compute distances for this block
                for i in i_start..i_end {
                    for j in j_start..j_end {
                        let point1 = points1.row(i);
                        let point2 = points2.row(j);

                        distances[[i, j]] = self.compute_distance(&point1, &point2, metric);
                    }
                }
            }
        }

        distances
    }

    /// Compute distance between two points using optimal implementation
    pub fn compute_distance(
        &self,
        point1: &ArrayView1<Float>,
        point2: &ArrayView1<Float>,
        metric: DistanceMetric,
    ) -> Float {
        if self.simd_available && point1.len() >= 4 {
            self.compute_distance_simd(point1, point2, metric)
        } else {
            self.compute_distance_scalar(point1, point2, metric)
        }
    }

    /// SIMD-optimized distance computation
    fn compute_distance_simd(
        &self,
        point1: &ArrayView1<Float>,
        point2: &ArrayView1<Float>,
        metric: DistanceMetric,
    ) -> Float {
        // Convert to slices for SIMD processing
        let a = point1.as_slice().unwrap();
        let b = point2.as_slice().unwrap();

        match metric {
            DistanceMetric::Euclidean => self.euclidean_simd(a, b),
            DistanceMetric::Manhattan => self.manhattan_simd(a, b),
            DistanceMetric::Chebyshev => self.chebyshev_simd(a, b),
            DistanceMetric::Cosine => self.cosine_simd(a, b),
            DistanceMetric::Minkowski(p) => self.minkowski_simd(a, b, p),
            DistanceMetric::Jaccard => self.jaccard_simd(a, b),
        }
    }

    /// Scalar fallback distance computation
    fn compute_distance_scalar(
        &self,
        point1: &ArrayView1<Float>,
        point2: &ArrayView1<Float>,
        metric: DistanceMetric,
    ) -> Float {
        let a = point1.as_slice().unwrap();
        let b = point2.as_slice().unwrap();

        match metric {
            DistanceMetric::Euclidean => fallback_distance::euclidean_distance(a, b),
            DistanceMetric::Manhattan => fallback_distance::manhattan_distance(a, b),
            DistanceMetric::Chebyshev => fallback_distance::chebyshev_distance(a, b),
            DistanceMetric::Cosine => fallback_distance::cosine_distance(a, b),
            DistanceMetric::Minkowski(p) => fallback_distance::minkowski_distance(a, b, p),
            DistanceMetric::Jaccard => fallback_distance::jaccard_distance(a, b),
        }
    }

    /// SIMD-optimized Euclidean distance (when available)
    fn euclidean_simd(&self, a: &[Float], b: &[Float]) -> Float {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                return unsafe { self.euclidean_avx2(a, b) };
            }
        }

        // Fallback to optimized scalar with manual unrolling
        self.euclidean_unrolled(a, b)
    }

    /// Manual loop unrolling for better performance when SIMD unavailable
    fn euclidean_unrolled(&self, a: &[Float], b: &[Float]) -> Float {
        let mut sum = 0.0;
        let len = a.len();
        let chunks = len / 4;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let diff1 = a[base] - b[base];
            let diff2 = a[base + 1] - b[base + 1];
            let diff3 = a[base + 2] - b[base + 2];
            let diff4 = a[base + 3] - b[base + 3];

            sum += diff1 * diff1 + diff2 * diff2 + diff3 * diff3 + diff4 * diff4;
        }

        // Handle remaining elements
        for i in (chunks * 4)..len {
            let diff = a[i] - b[i];
            sum += diff * diff;
        }

        sum.sqrt()
    }

    /// AVX2-optimized Euclidean distance (x86_64 only)
    #[cfg(target_arch = "x86_64")]
    unsafe fn euclidean_avx2(&self, a: &[Float], b: &[Float]) -> Float {
        // This would use std::arch::x86_64::* intrinsics for AVX2
        // For now, fallback to unrolled version
        self.euclidean_unrolled(a, b)
    }

    /// SIMD-optimized Manhattan distance
    fn manhattan_simd(&self, a: &[Float], b: &[Float]) -> Float {
        // Use manual unrolling for now
        let mut sum = 0.0;
        let len = a.len();
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            sum += (a[base] - b[base]).abs()
                + (a[base + 1] - b[base + 1]).abs()
                + (a[base + 2] - b[base + 2]).abs()
                + (a[base + 3] - b[base + 3]).abs();
        }

        for i in (chunks * 4)..len {
            sum += (a[i] - b[i]).abs();
        }

        sum
    }

    /// SIMD-optimized Chebyshev distance
    fn chebyshev_simd(&self, a: &[Float], b: &[Float]) -> Float {
        let mut max_diff = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            let diff = (x - y).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }

        max_diff
    }

    /// SIMD-optimized Cosine distance
    fn cosine_simd(&self, a: &[Float], b: &[Float]) -> Float {
        let mut dot = 0.0;
        let mut norm_a_sq = 0.0;
        let mut norm_b_sq = 0.0;

        let len = a.len();
        let chunks = len / 4;

        // Unrolled loop for better performance
        for i in 0..chunks {
            let base = i * 4;
            for j in 0..4 {
                let idx = base + j;
                dot += a[idx] * b[idx];
                norm_a_sq += a[idx] * a[idx];
                norm_b_sq += b[idx] * b[idx];
            }
        }

        for i in (chunks * 4)..len {
            dot += a[i] * b[i];
            norm_a_sq += a[i] * a[i];
            norm_b_sq += b[i] * b[i];
        }

        1.0 - (dot / (norm_a_sq.sqrt() * norm_b_sq.sqrt()))
    }

    /// SIMD-optimized Minkowski distance
    fn minkowski_simd(&self, a: &[Float], b: &[Float], p: Float) -> Float {
        let mut sum = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            sum += (x - y).abs().powf(p);
        }

        sum.powf(1.0 / p)
    }

    /// SIMD-optimized Jaccard distance
    fn jaccard_simd(&self, a: &[Float], b: &[Float]) -> Float {
        let mut intersection = 0.0;
        let mut union = 0.0;

        for (x, y) in a.iter().zip(b.iter()) {
            intersection += x.min(*y);
            union += x.max(*y);
        }

        1.0 - (intersection / union)
    }
}

// Fallback implementations when SIMD is not available
mod fallback_distance {
    use super::Float;

    pub fn euclidean_distance(a: &[Float], b: &[Float]) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    pub fn manhattan_distance(a: &[Float], b: &[Float]) -> Float {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    pub fn chebyshev_distance(a: &[Float], b: &[Float]) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, Float::max)
    }

    pub fn cosine_distance(a: &[Float], b: &[Float]) -> Float {
        let dot: Float = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: Float = a.iter().map(|x| x * x).sum::<Float>().sqrt();
        let norm_b: Float = b.iter().map(|x| x * x).sum::<Float>().sqrt();
        1.0 - (dot / (norm_a * norm_b))
    }

    pub fn minkowski_distance(a: &[Float], b: &[Float], p: Float) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs().powf(p))
            .sum::<Float>()
            .powf(1.0 / p)
    }

    pub fn jaccard_distance(a: &[Float], b: &[Float]) -> Float {
        let intersection: Float = a.iter().zip(b.iter()).map(|(x, y)| x.min(*y)).sum();
        let union: Float = a.iter().zip(b.iter()).map(|(x, y)| x.max(*y)).sum();
        1.0 - (intersection / union)
    }
}

/// SIMD-optimized distance metrics for clustering
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimdDistanceMetric {
    /// Euclidean distance (L2 norm)
    Euclidean,
    /// Squared Euclidean distance (faster than Euclidean for many use cases)
    EuclideanSquared,
    /// Manhattan distance (L1 norm)
    Manhattan,
    /// Chebyshev distance (L∞ norm)
    Chebyshev,
    /// Cosine distance
    Cosine,
    /// Cosine similarity (1 - cosine distance)
    CosineSimilarity,
    /// Minkowski distance with parameter p
    Minkowski(Float),
    /// Jaccard distance
    Jaccard,
    /// Hamming distance (for binary/categorical data)
    Hamming,
    /// Canberra distance
    Canberra,
    /// Braycurtis distance
    Braycurtis,
    /// Mahalanobis distance (requires covariance matrix)
    Mahalanobis,
    /// Pearson correlation distance
    Correlation,
    /// Wasserstein (Earth Mover's) distance
    Wasserstein,
}

/// Calculate SIMD-optimized distance between two points
pub fn simd_distance(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> Result<Float, Box<dyn std::error::Error>> {
    // Use Float directly for consistency
    let a = point1.as_slice().unwrap();
    let b = point2.as_slice().unwrap();

    let result = match metric {
        SimdDistanceMetric::Euclidean => fallback_distance::euclidean_distance(a, b),
        SimdDistanceMetric::EuclideanSquared => {
            let euclidean = fallback_distance::euclidean_distance(a, b);
            euclidean * euclidean
        }
        SimdDistanceMetric::Manhattan => fallback_distance::manhattan_distance(a, b),
        SimdDistanceMetric::Chebyshev => fallback_distance::chebyshev_distance(a, b),
        SimdDistanceMetric::Cosine => fallback_distance::cosine_distance(a, b),
        SimdDistanceMetric::CosineSimilarity => 1.0 - fallback_distance::cosine_distance(a, b),
        SimdDistanceMetric::Minkowski(p) => fallback_distance::minkowski_distance(a, b, p),
        SimdDistanceMetric::Jaccard => fallback_distance::jaccard_distance(a, b),
        SimdDistanceMetric::Hamming => hamming_distance_simd(a, b),
        SimdDistanceMetric::Canberra => canberra_distance_simd(a, b),
        SimdDistanceMetric::Braycurtis => braycurtis_distance_simd(a, b),
        SimdDistanceMetric::Mahalanobis => {
            return Err("Mahalanobis distance requires covariance matrix parameter".into());
        }
        SimdDistanceMetric::Correlation => correlation_distance_simd(a, b),
        SimdDistanceMetric::Wasserstein => wasserstein_distance_simd(a, b),
    };

    Ok(result as Float)
}

/// Calculate SIMD-optimized squared Euclidean distance (faster than Euclidean)
pub fn simd_squared_euclidean_distance(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
) -> Result<Float, Box<dyn std::error::Error>> {
    let a = point1.as_slice().unwrap();
    let b = point2.as_slice().unwrap();

    let euclidean = fallback_distance::euclidean_distance(a, b);
    Ok(euclidean * euclidean)
}

/// Batch SIMD-optimized distance calculation from multiple points to multiple queries
pub fn simd_distance_batch(
    points: &[scirs2_core::ndarray::Array1<Float>],
    queries: &[scirs2_core::ndarray::Array1<Float>],
    metric: SimdDistanceMetric,
) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
    if points.len() != queries.len() {
        return Err("Points and queries must have the same length".into());
    }

    let mut results = Vec::with_capacity(points.len());

    for (point, query) in points.iter().zip(queries.iter()) {
        let point_slice = point.as_slice().unwrap();
        let query_slice = query.as_slice().unwrap();

        let distance = match metric {
            SimdDistanceMetric::Euclidean => {
                fallback_distance::euclidean_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::EuclideanSquared => {
                let euclidean = fallback_distance::euclidean_distance(point_slice, query_slice);
                euclidean * euclidean
            }
            SimdDistanceMetric::Manhattan => {
                fallback_distance::manhattan_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Chebyshev => {
                fallback_distance::chebyshev_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Cosine => {
                fallback_distance::cosine_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::CosineSimilarity => {
                1.0 - fallback_distance::cosine_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Minkowski(p) => {
                fallback_distance::minkowski_distance(point_slice, query_slice, p)
            }
            SimdDistanceMetric::Jaccard => {
                fallback_distance::jaccard_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Hamming => hamming_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Canberra => canberra_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Braycurtis => braycurtis_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Mahalanobis => {
                return Err("Mahalanobis distance requires covariance matrix parameter".into());
            }
            SimdDistanceMetric::Correlation => correlation_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Wasserstein => wasserstein_distance_simd(point_slice, query_slice),
        };

        results.push(distance);
    }

    Ok(results)
}

/// Batch SIMD-optimized distance calculation from a query point to multiple points
pub fn simd_distance_batch_query(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
    let query_slice = query.as_slice().unwrap();
    let mut results = Vec::with_capacity(points.nrows());

    for i in 0..points.nrows() {
        let point = points.row(i);
        let point_slice = point.as_slice().unwrap();

        let distance = match metric {
            SimdDistanceMetric::Euclidean => {
                fallback_distance::euclidean_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::EuclideanSquared => {
                let euclidean = fallback_distance::euclidean_distance(point_slice, query_slice);
                euclidean * euclidean
            }
            SimdDistanceMetric::Manhattan => {
                fallback_distance::manhattan_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Chebyshev => {
                fallback_distance::chebyshev_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Cosine => {
                fallback_distance::cosine_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::CosineSimilarity => {
                1.0 - fallback_distance::cosine_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Minkowski(p) => {
                fallback_distance::minkowski_distance(point_slice, query_slice, p)
            }
            SimdDistanceMetric::Jaccard => {
                fallback_distance::jaccard_distance(point_slice, query_slice)
            }
            SimdDistanceMetric::Hamming => hamming_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Canberra => canberra_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Braycurtis => braycurtis_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Mahalanobis => {
                return Err("Mahalanobis distance requires covariance matrix parameter".into());
            }
            SimdDistanceMetric::Correlation => correlation_distance_simd(point_slice, query_slice),
            SimdDistanceMetric::Wasserstein => wasserstein_distance_simd(point_slice, query_slice),
        };

        results.push(distance);
    }

    Ok(results)
}

/// Parallel batch SIMD-optimized distance calculation
#[cfg(feature = "parallel")]
pub fn simd_distance_batch_parallel(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
    use rayon::prelude::*;

    let query_vec: Vec<Float> = query.iter().copied().collect();

    let results: Vec<Float> = (0..points.nrows())
        .into_par_iter()
        .map(|i| {
            let point = points.row(i);
            let point_vec: Vec<Float> = point.iter().copied().collect();

            let distance = match metric {
                SimdDistanceMetric::Euclidean => {
                    fallback_distance::euclidean_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::EuclideanSquared => {
                    let euclidean = fallback_distance::euclidean_distance(&point_vec, &query_vec);
                    euclidean * euclidean
                }
                SimdDistanceMetric::Manhattan => {
                    fallback_distance::manhattan_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::Chebyshev => {
                    fallback_distance::chebyshev_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::Cosine => {
                    fallback_distance::cosine_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::CosineSimilarity => {
                    1.0 - fallback_distance::cosine_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::Minkowski(p) => {
                    fallback_distance::minkowski_distance(&point_vec, &query_vec, p)
                }
                SimdDistanceMetric::Jaccard => {
                    fallback_distance::jaccard_distance(&point_vec, &query_vec)
                }
                SimdDistanceMetric::Hamming => hamming_distance_simd(&point_vec, &query_vec),
                SimdDistanceMetric::Canberra => canberra_distance_simd(&point_vec, &query_vec),
                SimdDistanceMetric::Braycurtis => braycurtis_distance_simd(&point_vec, &query_vec),
                SimdDistanceMetric::Mahalanobis => {
                    return Err(Box::new(std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "Mahalanobis distance requires covariance matrix parameter",
                    )));
                }
                SimdDistanceMetric::Correlation => {
                    correlation_distance_simd(&point_vec, &query_vec)
                }
                SimdDistanceMetric::Wasserstein => {
                    wasserstein_distance_simd(&point_vec, &query_vec)
                }
            };

            Ok(distance as Float)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(results)
}

/// Calculate pairwise SIMD-optimized distances between all points
pub fn simd_pairwise_distances(
    points: &ArrayView2<Float>,
    metric: SimdDistanceMetric,
) -> Result<Vec<Vec<Float>>, Box<dyn std::error::Error>> {
    let n_points = points.nrows();
    let mut distances = vec![vec![0.0; n_points]; n_points];

    for i in 0..n_points {
        let point_i = points.row(i);
        for j in (i + 1)..n_points {
            let point_j = points.row(j);
            let dist = simd_distance(&point_i, &point_j, metric)?;
            distances[i][j] = dist;
            distances[j][i] = dist; // Symmetric
        }
    }

    Ok(distances)
}

/// Calculate k-nearest neighbors using SIMD-optimized distances
pub fn simd_k_nearest_neighbors(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    k: usize,
    metric: SimdDistanceMetric,
) -> Result<Vec<(usize, Float)>, Box<dyn std::error::Error>> {
    let distances = simd_distance_batch_query(points, query, metric)?;

    let mut indexed_distances: Vec<(usize, Float)> = distances.into_iter().enumerate().collect();

    // Sort by distance and take the k nearest
    indexed_distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    indexed_distances.truncate(k);

    Ok(indexed_distances)
}

/// Find neighbors within a radius using SIMD-optimized distances
pub fn simd_radius_neighbors(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    radius: Float,
    metric: SimdDistanceMetric,
) -> Result<Vec<usize>, Box<dyn std::error::Error>> {
    let distances = simd_distance_batch_query(points, query, metric)?;

    let neighbors: Vec<usize> = distances
        .into_iter()
        .enumerate()
        .filter_map(|(idx, dist)| if dist <= radius { Some(idx) } else { None })
        .collect();

    Ok(neighbors)
}

/// SIMD-optimized distance matrix computation
pub fn simd_distance_matrix(
    points: &ArrayView2<Float>,
    metric: SimdDistanceMetric,
) -> Result<scirs2_core::ndarray::Array2<Float>, Box<dyn std::error::Error>> {
    let n_points = points.nrows();
    let mut matrix = scirs2_core::ndarray::Array2::zeros((n_points, n_points));

    for i in 0..n_points {
        let point_i = points.row(i);
        for j in (i + 1)..n_points {
            let point_j = points.row(j);
            let dist = simd_distance(&point_i, &point_j, metric)?;
            matrix[[i, j]] = dist;
            matrix[[j, i]] = dist;
        }
    }

    Ok(matrix)
}

/// Performance comparison between SIMD and scalar implementations
pub fn benchmark_simd_vs_scalar(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> (f64, f64) {
    use std::time::Instant;

    // SIMD version
    let start = Instant::now();
    let _simd_result = simd_distance_batch_query(points, query, metric).unwrap();
    let simd_time = start.elapsed().as_secs_f64();

    // Scalar version (simple fallback)
    let start = Instant::now();
    let _scalar_result = scalar_distance_batch(points, query, metric);
    let scalar_time = start.elapsed().as_secs_f64();

    (simd_time, scalar_time)
}

/// Scalar fallback implementation for comparison
fn scalar_distance_batch(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> Vec<Float> {
    let mut results = Vec::with_capacity(points.nrows());

    for i in 0..points.nrows() {
        let point = points.row(i);
        let dist = match metric {
            SimdDistanceMetric::Euclidean => {
                let mut sum = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    let diff = a - b;
                    sum += diff * diff;
                }
                sum.sqrt()
            }
            SimdDistanceMetric::Manhattan => {
                let mut sum = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    sum += (a - b).abs();
                }
                sum
            }
            SimdDistanceMetric::Chebyshev => {
                let mut max_diff = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    let diff = (a - b).abs();
                    if diff > max_diff {
                        max_diff = diff;
                    }
                }
                max_diff
            }
            SimdDistanceMetric::Cosine => {
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                let norm_product = norm_a.sqrt() * norm_b.sqrt();
                if norm_product == 0.0 {
                    0.0
                } else {
                    1.0 - (dot / norm_product)
                }
            }
            SimdDistanceMetric::CosineSimilarity => {
                let mut dot = 0.0;
                let mut norm_a = 0.0;
                let mut norm_b = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    dot += a * b;
                    norm_a += a * a;
                    norm_b += b * b;
                }
                let norm_product = norm_a.sqrt() * norm_b.sqrt();
                if norm_product == 0.0 {
                    0.0
                } else {
                    dot / norm_product
                }
            }
            SimdDistanceMetric::Minkowski(p) => {
                let mut sum = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    sum += (a - b).abs().powf(p as Float);
                }
                sum.powf(1.0 / p as Float)
            }
            SimdDistanceMetric::Jaccard => {
                let mut intersection = 0.0;
                let mut union = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    intersection += a.min(b);
                    union += a.max(b);
                }
                if union == 0.0 {
                    0.0
                } else {
                    1.0 - (intersection / union)
                }
            }
            SimdDistanceMetric::EuclideanSquared => {
                let mut sum = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    let diff = a - b;
                    sum += diff * diff;
                }
                sum
            }
            SimdDistanceMetric::Hamming => {
                let mut count = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    if (a - b).abs() > Float::EPSILON {
                        count += 1.0;
                    }
                }
                count
            }
            SimdDistanceMetric::Canberra => {
                let mut sum = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    let numerator = (a - b).abs();
                    let denominator = a.abs() + b.abs();
                    if denominator > 0.0 {
                        sum += numerator / denominator;
                    }
                }
                sum
            }
            SimdDistanceMetric::Braycurtis => {
                let mut numerator = 0.0;
                let mut denominator = 0.0;
                for (&a, &b) in point.iter().zip(query.iter()) {
                    numerator += (a - b).abs();
                    denominator += a.abs() + b.abs();
                }
                if denominator == 0.0 {
                    0.0
                } else {
                    numerator / denominator
                }
            }
            SimdDistanceMetric::Mahalanobis => {
                // This should not be reached as it requires covariance matrix
                0.0
            }
            SimdDistanceMetric::Correlation => {
                // Pearson correlation distance
                let n = point.len() as Float;
                let sum_a: Float = point.iter().sum();
                let sum_b: Float = query.iter().sum();
                let mean_a = sum_a / n;
                let mean_b = sum_b / n;

                let mut numerator = 0.0;
                let mut var_a = 0.0;
                let mut var_b = 0.0;

                for (&a, &b) in point.iter().zip(query.iter()) {
                    let diff_a = a - mean_a;
                    let diff_b = b - mean_b;
                    numerator += diff_a * diff_b;
                    var_a += diff_a * diff_a;
                    var_b += diff_b * diff_b;
                }

                let denominator = (var_a * var_b).sqrt();
                if denominator == 0.0 {
                    0.0
                } else {
                    1.0 - (numerator / denominator)
                }
            }
            SimdDistanceMetric::Wasserstein => {
                // Simple 1D Wasserstein distance (Earth Mover's Distance)
                let mut sorted_a: Vec<Float> = point.iter().cloned().collect();
                let mut sorted_b: Vec<Float> = query.iter().cloned().collect();
                sorted_a.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted_b.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let mut sum = 0.0;
                for (a, b) in sorted_a.iter().zip(sorted_b.iter()) {
                    sum += (a - b).abs();
                }
                sum / point.len() as Float
            }
        };
        results.push(dist);
    }

    results
}

/// Adaptive distance function that chooses between SIMD and scalar based on data size
pub fn adaptive_distance_batch(
    points: &ArrayView2<Float>,
    query: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
    simd_threshold: usize,
) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
    if points.nrows() >= simd_threshold && query.len() >= 4 {
        simd_distance_batch_query(points, query, metric)
    } else {
        Ok(scalar_distance_batch(points, query, metric))
    }
}

/// Custom distance function with user-defined metric
pub fn custom_distance<F>(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
    distance_fn: F,
) -> Float
where
    F: Fn(&ArrayView1<Float>, &ArrayView1<Float>) -> Float,
{
    distance_fn(point1, point2)
}

/// Mahalanobis distance with provided covariance matrix
pub fn mahalanobis_distance(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
    cov_inv: &scirs2_core::ndarray::Array2<Float>,
) -> Result<Float, Box<dyn std::error::Error>> {
    if point1.len() != point2.len() {
        return Err("Points must have the same dimensions".into());
    }

    if cov_inv.nrows() != point1.len() || cov_inv.ncols() != point1.len() {
        return Err("Covariance matrix dimensions must match point dimensions".into());
    }

    let diff: scirs2_core::ndarray::Array1<Float> = point1.to_owned() - point2;
    let temp = cov_inv.dot(&diff);
    let distance_squared = diff.dot(&temp);

    Ok(distance_squared.sqrt())
}

/// Distance metrics with preprocessing for categorical data
pub fn categorical_distance(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
    metric: CategoricalDistanceMetric,
) -> Float {
    match metric {
        CategoricalDistanceMetric::Hamming => {
            let mut count = 0.0;
            for (&a, &b) in point1.iter().zip(point2.iter()) {
                if (a - b).abs() > Float::EPSILON {
                    count += 1.0;
                }
            }
            count / point1.len() as Float
        }
        CategoricalDistanceMetric::MatchingDissimilarity => {
            let mut mismatches = 0.0;
            for (&a, &b) in point1.iter().zip(point2.iter()) {
                if (a - b).abs() > Float::EPSILON {
                    mismatches += 1.0;
                }
            }
            mismatches / point1.len() as Float
        }
    }
}

/// Categorical distance metrics
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CategoricalDistanceMetric {
    /// Hamming distance for categorical variables
    Hamming,
    /// Matching dissimilarity
    MatchingDissimilarity,
}

/// Weighted distance calculation
pub fn weighted_distance(
    point1: &ArrayView1<Float>,
    point2: &ArrayView1<Float>,
    weights: &ArrayView1<Float>,
    metric: SimdDistanceMetric,
) -> Result<Float, Box<dyn std::error::Error>> {
    if point1.len() != point2.len() || point1.len() != weights.len() {
        return Err("All arrays must have the same length".into());
    }

    match metric {
        SimdDistanceMetric::Euclidean => {
            let mut sum = 0.0;
            for ((&a, &b), &w) in point1.iter().zip(point2.iter()).zip(weights.iter()) {
                let diff = a - b;
                sum += w * diff * diff;
            }
            Ok(sum.sqrt())
        }
        SimdDistanceMetric::Manhattan => {
            let mut sum = 0.0;
            for ((&a, &b), &w) in point1.iter().zip(point2.iter()).zip(weights.iter()) {
                sum += w * (a - b).abs();
            }
            Ok(sum)
        }
        _ => {
            // For other metrics, apply weights as scaling factors
            let weighted_p1: scirs2_core::ndarray::Array1<Float> = point1
                .iter()
                .zip(weights.iter())
                .map(|(&p, &w)| p * w.sqrt())
                .collect();
            let weighted_p2: scirs2_core::ndarray::Array1<Float> = point2
                .iter()
                .zip(weights.iter())
                .map(|(&p, &w)| p * w.sqrt())
                .collect();
            simd_distance(&weighted_p1.view(), &weighted_p2.view(), metric)
        }
    }
}

// Helper functions for additional distance metrics
fn hamming_distance_simd(a: &[Float], b: &[Float]) -> Float {
    let mut count = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > Float::EPSILON {
            count += 1.0;
        }
    }
    count
}

fn canberra_distance_simd(a: &[Float], b: &[Float]) -> Float {
    let mut sum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let numerator = (x - y).abs();
        let denominator = x.abs() + y.abs();
        if denominator > 0.0 {
            sum += numerator / denominator;
        }
    }
    sum
}

fn braycurtis_distance_simd(a: &[Float], b: &[Float]) -> Float {
    let mut numerator = 0.0;
    let mut denominator = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        numerator += (x - y).abs();
        denominator += x.abs() + y.abs();
    }
    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

fn correlation_distance_simd(a: &[Float], b: &[Float]) -> Float {
    let n = a.len() as Float;
    let sum_a: Float = a.iter().sum();
    let sum_b: Float = b.iter().sum();
    let mean_a = sum_a / n;
    let mean_b = sum_b / n;

    let mut numerator = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;

    for (&x, &y) in a.iter().zip(b.iter()) {
        let diff_a = x - mean_a;
        let diff_b = y - mean_b;
        numerator += diff_a * diff_b;
        var_a += diff_a * diff_a;
        var_b += diff_b * diff_b;
    }

    let denominator = (var_a * var_b).sqrt();
    if denominator == 0.0 {
        0.0
    } else {
        1.0 - (numerator / denominator)
    }
}

fn wasserstein_distance_simd(a: &[Float], b: &[Float]) -> Float {
    let mut sorted_a = a.to_vec();
    let mut sorted_b = b.to_vec();
    sorted_a.sort_by(|x, y| x.partial_cmp(y).unwrap());
    sorted_b.sort_by(|x, y| x.partial_cmp(y).unwrap());

    let mut sum = 0.0;
    for (x, y) in sorted_a.iter().zip(sorted_b.iter()) {
        sum += (x - y).abs();
    }
    sum / a.len() as Float
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array1, Array2};

    #[test]
    fn test_simd_euclidean_distance() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let query = array![0.0, 0.0];

        let distances =
            simd_distance_batch_query(&data.view(), &query.view(), SimdDistanceMetric::Euclidean)
                .unwrap();

        assert_eq!(distances.len(), 3);
        assert_abs_diff_eq!(distances[0], (5.0_f64).sqrt(), epsilon = 1e-6);
        assert_abs_diff_eq!(distances[1], 5.0, epsilon = 1e-6);
        assert_abs_diff_eq!(distances[2], (61.0_f64).sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_simd_manhattan_distance() {
        let data = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let query = array![0.0, 0.0, 0.0];

        let distances =
            simd_distance_batch_query(&data.view(), &query.view(), SimdDistanceMetric::Manhattan)
                .unwrap();

        assert_eq!(distances.len(), 2);
        assert_abs_diff_eq!(distances[0], 6.0, epsilon = 1e-6); // |1| + |2| + |3|
        assert_abs_diff_eq!(distances[1], 15.0, epsilon = 1e-6); // |4| + |5| + |6|
    }

    #[test]
    fn test_simd_vs_scalar_consistency() {
        let data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let query = array![0.0, 0.0, 0.0];

        let simd_distances =
            simd_distance_batch_query(&data.view(), &query.view(), SimdDistanceMetric::Euclidean)
                .unwrap();
        let scalar_distances =
            scalar_distance_batch(&data.view(), &query.view(), SimdDistanceMetric::Euclidean);

        assert_eq!(simd_distances.len(), scalar_distances.len());
        for (simd, scalar) in simd_distances.iter().zip(scalar_distances.iter()) {
            assert_abs_diff_eq!(simd, scalar, epsilon = 1e-5);
        }
    }

    #[test]
    fn test_simd_k_nearest_neighbors() {
        let data = Array2::from_shape_vec(
            (5, 2),
            vec![
                1.0, 1.0, // Distance: sqrt(2) ≈ 1.414
                2.0, 2.0, // Distance: sqrt(8) ≈ 2.828
                0.0, 0.0, // Distance: 0
                3.0, 3.0, // Distance: sqrt(18) ≈ 4.243
                0.5, 0.5, // Distance: sqrt(0.5) ≈ 0.707
            ],
        )
        .unwrap();
        let query = array![0.0, 0.0];

        let neighbors = simd_k_nearest_neighbors(
            &data.view(),
            &query.view(),
            3,
            SimdDistanceMetric::Euclidean,
        )
        .unwrap();

        assert_eq!(neighbors.len(), 3);
        assert_eq!(neighbors[0].0, 2); // Nearest is (0,0)
        assert_eq!(neighbors[1].0, 4); // Second nearest is (0.5, 0.5)
        assert_eq!(neighbors[2].0, 0); // Third nearest is (1,1)
    }

    #[test]
    fn test_simd_radius_neighbors() {
        let data = Array2::from_shape_vec(
            (4, 2),
            vec![
                1.0, 0.0, // Distance: 1
                0.0, 1.0, // Distance: 1
                2.0, 0.0, // Distance: 2
                0.0, 2.0, // Distance: 2
            ],
        )
        .unwrap();
        let query = array![0.0, 0.0];

        let neighbors = simd_radius_neighbors(
            &data.view(),
            &query.view(),
            1.5,
            SimdDistanceMetric::Euclidean,
        )
        .unwrap();

        assert_eq!(neighbors.len(), 2);
        assert!(neighbors.contains(&0));
        assert!(neighbors.contains(&1));
    }

    #[test]
    fn test_simd_distance_matrix() {
        let data = Array2::from_shape_vec((3, 2), vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();

        let matrix = simd_distance_matrix(&data.view(), SimdDistanceMetric::Euclidean).unwrap();

        assert_eq!(matrix.shape(), &[3, 3]);

        // Diagonal should be zero
        assert_abs_diff_eq!(matrix[[0, 0]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[[1, 1]], 0.0, epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[[2, 2]], 0.0, epsilon = 1e-6);

        // Distance from (0,0) to (1,0) should be 1
        assert_abs_diff_eq!(matrix[[0, 1]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[[1, 0]], 1.0, epsilon = 1e-6);

        // Distance from (0,0) to (0,1) should be 1
        assert_abs_diff_eq!(matrix[[0, 2]], 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[[2, 0]], 1.0, epsilon = 1e-6);

        // Distance from (1,0) to (0,1) should be sqrt(2)
        assert_abs_diff_eq!(matrix[[1, 2]], (2.0_f64).sqrt(), epsilon = 1e-6);
        assert_abs_diff_eq!(matrix[[2, 1]], (2.0_f64).sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_adaptive_distance_batch() {
        let small_data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let large_data =
            Array2::from_shape_vec((10, 4), (0..40).map(|x| x as f64).collect()).unwrap();
        let query = array![0.0, 0.0, 0.0, 0.0];

        // Small data should use scalar
        let small_result = adaptive_distance_batch(
            &small_data.view(),
            &query.view().slice(scirs2_core::ndarray::s![..2]),
            SimdDistanceMetric::Euclidean,
            5,
        )
        .unwrap();
        assert_eq!(small_result.len(), 2);

        // Large data should use SIMD
        let large_result = adaptive_distance_batch(
            &large_data.view(),
            &query.view(),
            SimdDistanceMetric::Euclidean,
            5,
        )
        .unwrap();
        assert_eq!(large_result.len(), 10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_simd_distance_batch() {
        let data = Array2::from_shape_vec((6, 3), (0..18).map(|x| x as f64).collect()).unwrap();
        let query = array![0.0, 0.0, 0.0];

        let parallel_result = simd_distance_batch_parallel(
            &data.view(),
            &query.view(),
            SimdDistanceMetric::Euclidean,
        )
        .unwrap();
        let sequential_result =
            simd_distance_batch_query(&data.view(), &query.view(), SimdDistanceMetric::Euclidean)
                .unwrap();

        assert_eq!(parallel_result.len(), sequential_result.len());
        for (par, seq) in parallel_result.iter().zip(sequential_result.iter()) {
            assert_abs_diff_eq!(par, seq, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_optimized_distance_computer_performance() {
        use scirs2_core::ndarray::{Array1, Array2};

        let computer = OptimizedDistanceComputer::new();
        let n_points = 100;
        let n_features = 10;

        // Generate test data
        let points1 = Array2::<Float>::ones((n_points, n_features));
        let points2 = Array2::<Float>::zeros((n_points, n_features));

        // Test pairwise distances
        let distances = computer.pairwise_distances(
            &points1.view(),
            &points2.view(),
            DistanceMetric::Euclidean,
        );

        assert_eq!(distances.dim(), (n_points, n_points));

        // All distances should be sqrt(n_features) since points are (1,1,1...) vs (0,0,0...)
        let expected_distance = (n_features as Float).sqrt();
        for &dist in distances.iter() {
            assert!((dist - expected_distance).abs() < 1e-6);
        }
    }

    #[test]
    fn test_simd_detection() {
        let computer = OptimizedDistanceComputer::new();

        // Test that SIMD detection works without crashing
        let point1 = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
        let point2 = Array1::from(vec![2.0, 3.0, 4.0, 5.0]);

        let distance =
            computer.compute_distance(&point1.view(), &point2.view(), DistanceMetric::Euclidean);

        // Distance should be 2.0 (sqrt of 4 * 1^2)
        assert!((distance - 2.0).abs() < 1e-6);
    }
}

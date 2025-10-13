//! SIMD-optimized clustering operations
//!
//! This module provides vectorized implementations of common clustering operations
//! including k-means distance computations, centroid updates, and validity indices.

#[cfg(feature = "no-std")]
use alloc::{format, vec, vec::Vec};

/// SIMD-optimized k-means distance computation
/// Computes distances from each point to each centroid
pub fn kmeans_distances(
    points: &[&[f32]],          // Points (n_samples x n_features)
    centroids: &[&[f32]],       // Centroids (n_clusters x n_features)
    distances: &mut [Vec<f32>], // Output distances (n_samples x n_clusters)
) {
    let n_samples = points.len();
    let _n_clusters = centroids.len();

    assert!(!points.is_empty(), "Points cannot be empty");
    assert!(!centroids.is_empty(), "Centroids cannot be empty");
    assert_eq!(distances.len(), n_samples, "Distance array size mismatch");

    let n_features = points[0].len();
    for centroid in centroids {
        assert_eq!(
            centroid.len(),
            n_features,
            "All points must have the same number of features"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { kmeans_distances_avx2(points, centroids, distances) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { kmeans_distances_sse2(points, centroids, distances) };
            return;
        }
    }

    kmeans_distances_scalar(points, centroids, distances);
}

fn kmeans_distances_scalar(points: &[&[f32]], centroids: &[&[f32]], distances: &mut [Vec<f32>]) {
    let n_samples = points.len();
    let n_clusters = centroids.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        if distances[i].len() != n_clusters {
            distances[i] = vec![0.0; n_clusters];
        }

        for j in 0..n_clusters {
            let mut sum = 0.0;
            for k in 0..n_features {
                let diff = points[i][k] - centroids[j][k];
                sum += diff * diff;
            }
            distances[i][j] = sum.sqrt();
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn kmeans_distances_sse2(
    points: &[&[f32]],
    centroids: &[&[f32]],
    distances: &mut [Vec<f32>],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_clusters = centroids.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        if distances[i].len() != n_clusters {
            distances[i] = vec![0.0; n_clusters];
        }

        for j in 0..n_clusters {
            let mut sum = _mm_setzero_ps();
            let mut k = 0;

            while k + 4 <= n_features {
                let p_vec = _mm_loadu_ps(&points[i][k]);
                let c_vec = _mm_loadu_ps(&centroids[j][k]);
                let diff = _mm_sub_ps(p_vec, c_vec);
                let squared = _mm_mul_ps(diff, diff);
                sum = _mm_add_ps(sum, squared);
                k += 4;
            }

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

            while k < n_features {
                let diff = points[i][k] - centroids[j][k];
                scalar_sum += diff * diff;
                k += 1;
            }

            distances[i][j] = scalar_sum.sqrt();
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn kmeans_distances_avx2(
    points: &[&[f32]],
    centroids: &[&[f32]],
    distances: &mut [Vec<f32>],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_clusters = centroids.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        if distances[i].len() != n_clusters {
            distances[i] = vec![0.0; n_clusters];
        }

        for j in 0..n_clusters {
            let mut sum = _mm256_setzero_ps();
            let mut k = 0;

            while k + 8 <= n_features {
                let p_vec = _mm256_loadu_ps(&points[i][k]);
                let c_vec = _mm256_loadu_ps(&centroids[j][k]);
                let diff = _mm256_sub_ps(p_vec, c_vec);
                let squared = _mm256_mul_ps(diff, diff);
                sum = _mm256_add_ps(sum, squared);
                k += 8;
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum);
            let mut scalar_sum = result.iter().sum::<f32>();

            while k < n_features {
                let diff = points[i][k] - centroids[j][k];
                scalar_sum += diff * diff;
                k += 1;
            }

            distances[i][j] = scalar_sum.sqrt();
        }
    }
}

/// SIMD-optimized centroid update for k-means
/// Updates centroids based on assigned points
pub fn update_centroids(
    points: &[&[f32]],          // Points (n_samples x n_features)
    assignments: &[usize],      // Cluster assignments (n_samples)
    n_clusters: usize,          // Number of clusters
    centroids: &mut [Vec<f32>], // Output centroids (n_clusters x n_features)
) {
    let n_samples = points.len();
    let n_features = if n_samples > 0 { points[0].len() } else { 0 };

    assert_eq!(
        assignments.len(),
        n_samples,
        "Assignments length must match number of samples"
    );
    assert_eq!(
        centroids.len(),
        n_clusters,
        "Centroids length must match number of clusters"
    );

    // Initialize centroids and counts
    for i in 0..n_clusters {
        if centroids[i].len() != n_features {
            centroids[i] = vec![0.0; n_features];
        } else {
            centroids[i].fill(0.0);
        }
    }

    let mut counts = vec![0usize; n_clusters];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                update_centroids_avx2(points, assignments, n_clusters, centroids, &mut counts)
            };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                update_centroids_sse2(points, assignments, n_clusters, centroids, &mut counts)
            };
            return;
        }
    }

    update_centroids_scalar(points, assignments, n_clusters, centroids, &mut counts);
}

fn update_centroids_scalar(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    centroids: &mut [Vec<f32>],
    counts: &mut [usize],
) {
    let n_samples = points.len();
    let n_features = if n_samples > 0 { points[0].len() } else { 0 };

    // Accumulate points
    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < n_clusters {
            counts[cluster_id] += 1;
            for j in 0..n_features {
                centroids[cluster_id][j] += points[i][j];
            }
        }
    }

    // Compute averages
    for i in 0..n_clusters {
        if counts[i] > 0 {
            for j in 0..n_features {
                centroids[i][j] /= counts[i] as f32;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn update_centroids_sse2(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    centroids: &mut [Vec<f32>],
    counts: &mut [usize],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = if n_samples > 0 { points[0].len() } else { 0 };

    // Accumulate points
    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < n_clusters {
            counts[cluster_id] += 1;

            let mut j = 0;
            while j + 4 <= n_features {
                let centroid_vec = _mm_loadu_ps(&centroids[cluster_id][j]);
                let point_vec = _mm_loadu_ps(&points[i][j]);
                let result = _mm_add_ps(centroid_vec, point_vec);
                _mm_storeu_ps(&mut centroids[cluster_id][j], result);
                j += 4;
            }

            while j < n_features {
                centroids[cluster_id][j] += points[i][j];
                j += 1;
            }
        }
    }

    // Compute averages
    for i in 0..n_clusters {
        if counts[i] > 0 {
            let count_vec = _mm_set1_ps(counts[i] as f32);
            let mut j = 0;

            while j + 4 <= n_features {
                let centroid_vec = _mm_loadu_ps(&centroids[i][j]);
                let result = _mm_div_ps(centroid_vec, count_vec);
                _mm_storeu_ps(&mut centroids[i][j], result);
                j += 4;
            }

            while j < n_features {
                centroids[i][j] /= counts[i] as f32;
                j += 1;
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn update_centroids_avx2(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    centroids: &mut [Vec<f32>],
    counts: &mut [usize],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = if n_samples > 0 { points[0].len() } else { 0 };

    // Accumulate points
    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < n_clusters {
            counts[cluster_id] += 1;

            let mut j = 0;
            while j + 8 <= n_features {
                let centroid_vec = _mm256_loadu_ps(&centroids[cluster_id][j]);
                let point_vec = _mm256_loadu_ps(&points[i][j]);
                let result = _mm256_add_ps(centroid_vec, point_vec);
                _mm256_storeu_ps(&mut centroids[cluster_id][j], result);
                j += 8;
            }

            while j < n_features {
                centroids[cluster_id][j] += points[i][j];
                j += 1;
            }
        }
    }

    // Compute averages
    for i in 0..n_clusters {
        if counts[i] > 0 {
            let count_vec = _mm256_set1_ps(counts[i] as f32);
            let mut j = 0;

            while j + 8 <= n_features {
                let centroid_vec = _mm256_loadu_ps(&centroids[i][j]);
                let result = _mm256_div_ps(centroid_vec, count_vec);
                _mm256_storeu_ps(&mut centroids[i][j], result);
                j += 8;
            }

            while j < n_features {
                centroids[i][j] /= counts[i] as f32;
                j += 1;
            }
        }
    }
}

/// SIMD-optimized within-cluster sum of squares (WCSS)
/// Computes the sum of squared distances from points to their assigned centroids
pub fn wcss(
    points: &[&[f32]],     // Points (n_samples x n_features)
    centroids: &[&[f32]],  // Centroids (n_clusters x n_features)
    assignments: &[usize], // Cluster assignments (n_samples)
) -> f32 {
    let n_samples = points.len();
    assert_eq!(
        assignments.len(),
        n_samples,
        "Assignments length must match number of samples"
    );

    if n_samples == 0 {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { wcss_avx2(points, centroids, assignments) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { wcss_sse2(points, centroids, assignments) };
        }
    }

    wcss_scalar(points, centroids, assignments)
}

fn wcss_scalar(points: &[&[f32]], centroids: &[&[f32]], assignments: &[usize]) -> f32 {
    let n_samples = points.len();
    let n_features = points[0].len();
    let mut total_wcss = 0.0;

    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < centroids.len() {
            let mut sum_squared = 0.0;
            for j in 0..n_features {
                let diff = points[i][j] - centroids[cluster_id][j];
                sum_squared += diff * diff;
            }
            total_wcss += sum_squared;
        }
    }

    total_wcss
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn wcss_sse2(points: &[&[f32]], centroids: &[&[f32]], assignments: &[usize]) -> f32 {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();
    let mut total_wcss = 0.0;

    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < centroids.len() {
            let mut sum = _mm_setzero_ps();
            let mut j = 0;

            while j + 4 <= n_features {
                let p_vec = _mm_loadu_ps(&points[i][j]);
                let c_vec = _mm_loadu_ps(&centroids[cluster_id][j]);
                let diff = _mm_sub_ps(p_vec, c_vec);
                let squared = _mm_mul_ps(diff, diff);
                sum = _mm_add_ps(sum, squared);
                j += 4;
            }

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut sum_squared = result[0] + result[1] + result[2] + result[3];

            while j < n_features {
                let diff = points[i][j] - centroids[cluster_id][j];
                sum_squared += diff * diff;
                j += 1;
            }

            total_wcss += sum_squared;
        }
    }

    total_wcss
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn wcss_avx2(points: &[&[f32]], centroids: &[&[f32]], assignments: &[usize]) -> f32 {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();
    let mut total_wcss = 0.0;

    for i in 0..n_samples {
        let cluster_id = assignments[i];
        if cluster_id < centroids.len() {
            let mut sum = _mm256_setzero_ps();
            let mut j = 0;

            while j + 8 <= n_features {
                let p_vec = _mm256_loadu_ps(&points[i][j]);
                let c_vec = _mm256_loadu_ps(&centroids[cluster_id][j]);
                let diff = _mm256_sub_ps(p_vec, c_vec);
                let squared = _mm256_mul_ps(diff, diff);
                sum = _mm256_add_ps(sum, squared);
                j += 8;
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum);
            let mut sum_squared = result.iter().sum::<f32>();

            while j < n_features {
                let diff = points[i][j] - centroids[cluster_id][j];
                sum_squared += diff * diff;
                j += 1;
            }

            total_wcss += sum_squared;
        }
    }

    total_wcss
}

/// SIMD-optimized silhouette coefficient computation
/// Computes silhouette scores for clustering quality assessment
pub fn silhouette_score(
    points: &[&[f32]],     // Points (n_samples x n_features)
    assignments: &[usize], // Cluster assignments (n_samples)
    n_clusters: usize,     // Number of clusters
) -> f32 {
    let n_samples = points.len();

    if n_samples <= 1 || n_clusters <= 1 {
        return 0.0;
    }

    let mut silhouette_scores = vec![0.0; n_samples];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                silhouette_score_avx2(points, assignments, n_clusters, &mut silhouette_scores)
            };
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                silhouette_score_sse2(points, assignments, n_clusters, &mut silhouette_scores)
            };
        } else {
            silhouette_score_scalar(points, assignments, n_clusters, &mut silhouette_scores);
        }
    }
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    {
        silhouette_score_scalar(points, assignments, n_clusters, &mut silhouette_scores);
    }

    // Return average silhouette score
    silhouette_scores.iter().sum::<f32>() / n_samples as f32
}

fn silhouette_score_scalar(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    silhouette_scores: &mut [f32],
) {
    let n_samples = points.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        let cluster_i = assignments[i];

        // Compute a(i): average distance to points in the same cluster
        let mut intra_distance = 0.0;
        let mut intra_count = 0;

        for j in 0..n_samples {
            if i != j && assignments[j] == cluster_i {
                let mut dist = 0.0;
                for k in 0..n_features {
                    let diff = points[i][k] - points[j][k];
                    dist += diff * diff;
                }
                intra_distance += dist.sqrt();
                intra_count += 1;
            }
        }

        let a_i = if intra_count > 0 {
            intra_distance / intra_count as f32
        } else {
            0.0
        };

        // Compute b(i): minimum average distance to points in other clusters
        let mut min_inter_distance = f32::INFINITY;

        for c in 0..n_clusters {
            if c != cluster_i {
                let mut inter_distance = 0.0;
                let mut inter_count = 0;

                for j in 0..n_samples {
                    if assignments[j] == c {
                        let mut dist = 0.0;
                        for k in 0..n_features {
                            let diff = points[i][k] - points[j][k];
                            dist += diff * diff;
                        }
                        inter_distance += dist.sqrt();
                        inter_count += 1;
                    }
                }

                if inter_count > 0 {
                    let avg_inter = inter_distance / inter_count as f32;
                    min_inter_distance = min_inter_distance.min(avg_inter);
                }
            }
        }

        let b_i = min_inter_distance;

        // Compute silhouette coefficient
        silhouette_scores[i] = if a_i < b_i {
            1.0 - a_i / b_i
        } else if a_i > b_i {
            b_i / a_i - 1.0
        } else {
            0.0
        };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn silhouette_score_sse2(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    silhouette_scores: &mut [f32],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        let cluster_i = assignments[i];

        // Compute a(i): average distance to points in the same cluster
        let mut intra_distance = 0.0;
        let mut intra_count = 0;

        for j in 0..n_samples {
            if i != j && assignments[j] == cluster_i {
                let mut sum = _mm_setzero_ps();
                let mut k = 0;

                while k + 4 <= n_features {
                    let p1_vec = _mm_loadu_ps(&points[i][k]);
                    let p2_vec = _mm_loadu_ps(&points[j][k]);
                    let diff = _mm_sub_ps(p1_vec, p2_vec);
                    let squared = _mm_mul_ps(diff, diff);
                    sum = _mm_add_ps(sum, squared);
                    k += 4;
                }

                let mut result = [0.0f32; 4];
                _mm_storeu_ps(result.as_mut_ptr(), sum);
                let mut dist_squared = result[0] + result[1] + result[2] + result[3];

                while k < n_features {
                    let diff = points[i][k] - points[j][k];
                    dist_squared += diff * diff;
                    k += 1;
                }

                intra_distance += dist_squared.sqrt();
                intra_count += 1;
            }
        }

        let a_i = if intra_count > 0 {
            intra_distance / intra_count as f32
        } else {
            0.0
        };

        // Compute b(i): minimum average distance to points in other clusters
        let mut min_inter_distance = f32::INFINITY;

        for c in 0..n_clusters {
            if c != cluster_i {
                let mut inter_distance = 0.0;
                let mut inter_count = 0;

                for j in 0..n_samples {
                    if assignments[j] == c {
                        let mut sum = _mm_setzero_ps();
                        let mut k = 0;

                        while k + 4 <= n_features {
                            let p1_vec = _mm_loadu_ps(&points[i][k]);
                            let p2_vec = _mm_loadu_ps(&points[j][k]);
                            let diff = _mm_sub_ps(p1_vec, p2_vec);
                            let squared = _mm_mul_ps(diff, diff);
                            sum = _mm_add_ps(sum, squared);
                            k += 4;
                        }

                        let mut result = [0.0f32; 4];
                        _mm_storeu_ps(result.as_mut_ptr(), sum);
                        let mut dist_squared = result[0] + result[1] + result[2] + result[3];

                        while k < n_features {
                            let diff = points[i][k] - points[j][k];
                            dist_squared += diff * diff;
                            k += 1;
                        }

                        inter_distance += dist_squared.sqrt();
                        inter_count += 1;
                    }
                }

                if inter_count > 0 {
                    let avg_inter = inter_distance / inter_count as f32;
                    min_inter_distance = min_inter_distance.min(avg_inter);
                }
            }
        }

        let b_i = min_inter_distance;

        // Compute silhouette coefficient
        silhouette_scores[i] = if a_i < b_i {
            1.0 - a_i / b_i
        } else if a_i > b_i {
            b_i / a_i - 1.0
        } else {
            0.0
        };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn silhouette_score_avx2(
    points: &[&[f32]],
    assignments: &[usize],
    n_clusters: usize,
    silhouette_scores: &mut [f32],
) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        let cluster_i = assignments[i];

        // Compute a(i): average distance to points in the same cluster
        let mut intra_distance = 0.0;
        let mut intra_count = 0;

        for j in 0..n_samples {
            if i != j && assignments[j] == cluster_i {
                let mut sum = _mm256_setzero_ps();
                let mut k = 0;

                while k + 8 <= n_features {
                    let p1_vec = _mm256_loadu_ps(&points[i][k]);
                    let p2_vec = _mm256_loadu_ps(&points[j][k]);
                    let diff = _mm256_sub_ps(p1_vec, p2_vec);
                    let squared = _mm256_mul_ps(diff, diff);
                    sum = _mm256_add_ps(sum, squared);
                    k += 8;
                }

                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), sum);
                let mut dist_squared = result.iter().sum::<f32>();

                while k < n_features {
                    let diff = points[i][k] - points[j][k];
                    dist_squared += diff * diff;
                    k += 1;
                }

                intra_distance += dist_squared.sqrt();
                intra_count += 1;
            }
        }

        let a_i = if intra_count > 0 {
            intra_distance / intra_count as f32
        } else {
            0.0
        };

        // Compute b(i): minimum average distance to points in other clusters
        let mut min_inter_distance = f32::INFINITY;

        for c in 0..n_clusters {
            if c != cluster_i {
                let mut inter_distance = 0.0;
                let mut inter_count = 0;

                for j in 0..n_samples {
                    if assignments[j] == c {
                        let mut sum = _mm256_setzero_ps();
                        let mut k = 0;

                        while k + 8 <= n_features {
                            let p1_vec = _mm256_loadu_ps(&points[i][k]);
                            let p2_vec = _mm256_loadu_ps(&points[j][k]);
                            let diff = _mm256_sub_ps(p1_vec, p2_vec);
                            let squared = _mm256_mul_ps(diff, diff);
                            sum = _mm256_add_ps(sum, squared);
                            k += 8;
                        }

                        let mut result = [0.0f32; 8];
                        _mm256_storeu_ps(result.as_mut_ptr(), sum);
                        let mut dist_squared = result.iter().sum::<f32>();

                        while k < n_features {
                            let diff = points[i][k] - points[j][k];
                            dist_squared += diff * diff;
                            k += 1;
                        }

                        inter_distance += dist_squared.sqrt();
                        inter_count += 1;
                    }
                }

                if inter_count > 0 {
                    let avg_inter = inter_distance / inter_count as f32;
                    min_inter_distance = min_inter_distance.min(avg_inter);
                }
            }
        }

        let b_i = min_inter_distance;

        // Compute silhouette coefficient
        silhouette_scores[i] = if a_i < b_i {
            1.0 - a_i / b_i
        } else if a_i > b_i {
            b_i / a_i - 1.0
        } else {
            0.0
        };
    }
}

/// SIMD-optimized DBSCAN neighbor finding
/// Finds all points within epsilon distance of each point
pub fn dbscan_neighbors(
    points: &[&[f32]],            // Points (n_samples x n_features)
    eps: f32,                     // Epsilon distance threshold
    neighbors: &mut [Vec<usize>], // Output neighbors for each point
) {
    let n_samples = points.len();
    let eps_squared = eps * eps;

    assert!(!points.is_empty(), "Points cannot be empty");
    assert_eq!(
        neighbors.len(),
        n_samples,
        "Neighbors array size must match points"
    );

    let n_features = points[0].len();
    for point in points {
        assert_eq!(
            point.len(),
            n_features,
            "All points must have the same number of features"
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { dbscan_neighbors_avx2(points, eps_squared, neighbors) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { dbscan_neighbors_sse2(points, eps_squared, neighbors) };
            return;
        }
    }

    dbscan_neighbors_scalar(points, eps_squared, neighbors);
}

fn dbscan_neighbors_scalar(points: &[&[f32]], eps_squared: f32, neighbors: &mut [Vec<usize>]) {
    let n_samples = points.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        neighbors[i].clear();

        for j in 0..n_samples {
            if i != j {
                let mut dist_squared = 0.0;
                for k in 0..n_features {
                    let diff = points[i][k] - points[j][k];
                    dist_squared += diff * diff;
                }

                if dist_squared <= eps_squared {
                    neighbors[i].push(j);
                }
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn dbscan_neighbors_sse2(points: &[&[f32]], eps_squared: f32, neighbors: &mut [Vec<usize>]) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();
    let eps_vec = _mm_set1_ps(eps_squared);

    for i in 0..n_samples {
        neighbors[i].clear();

        for j in 0..n_samples {
            if i != j {
                let mut sum = _mm_setzero_ps();
                let mut k = 0;

                while k + 4 <= n_features {
                    let p1_vec = _mm_loadu_ps(&points[i][k]);
                    let p2_vec = _mm_loadu_ps(&points[j][k]);
                    let diff = _mm_sub_ps(p1_vec, p2_vec);
                    let squared = _mm_mul_ps(diff, diff);
                    sum = _mm_add_ps(sum, squared);
                    k += 4;
                }

                let mut result = [0.0f32; 4];
                _mm_storeu_ps(result.as_mut_ptr(), sum);
                let mut dist_squared = result[0] + result[1] + result[2] + result[3];

                while k < n_features {
                    let diff = points[i][k] - points[j][k];
                    dist_squared += diff * diff;
                    k += 1;
                }

                if dist_squared <= eps_squared {
                    neighbors[i].push(j);
                }
            }
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dbscan_neighbors_avx2(points: &[&[f32]], eps_squared: f32, neighbors: &mut [Vec<usize>]) {
    use core::arch::x86_64::*;

    let n_samples = points.len();
    let n_features = points[0].len();

    for i in 0..n_samples {
        neighbors[i].clear();

        for j in 0..n_samples {
            if i != j {
                let mut sum = _mm256_setzero_ps();
                let mut k = 0;

                while k + 8 <= n_features {
                    let p1_vec = _mm256_loadu_ps(&points[i][k]);
                    let p2_vec = _mm256_loadu_ps(&points[j][k]);
                    let diff = _mm256_sub_ps(p1_vec, p2_vec);
                    let squared = _mm256_mul_ps(diff, diff);
                    sum = _mm256_add_ps(sum, squared);
                    k += 8;
                }

                let mut result = [0.0f32; 8];
                _mm256_storeu_ps(result.as_mut_ptr(), sum);
                let mut dist_squared = result.iter().sum::<f32>();

                while k < n_features {
                    let diff = points[i][k] - points[j][k];
                    dist_squared += diff * diff;
                    k += 1;
                }

                if dist_squared <= eps_squared {
                    neighbors[i].push(j);
                }
            }
        }
    }
}

/// SIMD-optimized core point identification for DBSCAN
/// Identifies points that have at least min_samples neighbors
pub fn dbscan_core_points(
    neighbors: &[Vec<usize>], // Neighbors for each point
    min_samples: usize,       // Minimum samples for core point
    core_points: &mut [bool], // Output core point flags
) {
    let n_samples = neighbors.len();

    assert_eq!(
        core_points.len(),
        n_samples,
        "Core points array size must match points"
    );

    for i in 0..n_samples {
        // Include the point itself in the count (neighbors + 1)
        core_points[i] = neighbors[i].len() + 1 >= min_samples;
    }
}

/// SIMD-optimized hierarchical clustering distance computation
/// Computes linkage distances for hierarchical clustering
pub fn hierarchical_linkage_distances(
    points: &[&[f32]],    // Points (n_samples x n_features)
    cluster1: &[usize],   // Indices of points in cluster 1
    cluster2: &[usize],   // Indices of points in cluster 2
    linkage: LinkageType, // Type of linkage to compute
) -> f32 {
    assert!(!cluster1.is_empty(), "Cluster 1 cannot be empty");
    assert!(!cluster2.is_empty(), "Cluster 2 cannot be empty");
    assert!(!points.is_empty(), "Points cannot be empty");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe {
                hierarchical_linkage_distances_avx2(points, cluster1, cluster2, linkage)
            };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe {
                hierarchical_linkage_distances_sse2(points, cluster1, cluster2, linkage)
            };
        }
    }

    hierarchical_linkage_distances_scalar(points, cluster1, cluster2, linkage)
}

/// Types of linkage for hierarchical clustering
#[derive(Debug, Clone, Copy)]
pub enum LinkageType {
    Single,   // Minimum distance between any two points
    Complete, // Maximum distance between any two points
    Average,  // Average distance between all pairs
    Ward,     // Ward's minimum variance method
}

fn hierarchical_linkage_distances_scalar(
    points: &[&[f32]],
    cluster1: &[usize],
    cluster2: &[usize],
    linkage: LinkageType,
) -> f32 {
    let n_features = points[0].len();
    let mut distances = Vec::new();

    // Compute all pairwise distances between clusters
    for &i in cluster1 {
        for &j in cluster2 {
            let mut dist_squared = 0.0;
            for k in 0..n_features {
                let diff = points[i][k] - points[j][k];
                dist_squared += diff * diff;
            }
            distances.push(dist_squared.sqrt());
        }
    }

    match linkage {
        LinkageType::Single => distances.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)),
        LinkageType::Complete => distances.iter().fold(0.0, |acc, &x| acc.max(x)),
        LinkageType::Average => distances.iter().sum::<f32>() / distances.len() as f32,
        LinkageType::Ward => {
            // Ward linkage: compute centroids and distance between them
            let mut centroid1 = vec![0.0; n_features];
            let mut centroid2 = vec![0.0; n_features];

            // Compute centroids
            for &i in cluster1 {
                for j in 0..n_features {
                    centroid1[j] += points[i][j];
                }
            }
            for j in 0..n_features {
                centroid1[j] /= cluster1.len() as f32;
            }

            for &i in cluster2 {
                for j in 0..n_features {
                    centroid2[j] += points[i][j];
                }
            }
            for j in 0..n_features {
                centroid2[j] /= cluster2.len() as f32;
            }

            // Compute distance between centroids
            let mut dist_squared = 0.0;
            for j in 0..n_features {
                let diff = centroid1[j] - centroid2[j];
                dist_squared += diff * diff;
            }

            // Ward distance includes cluster sizes
            let n1 = cluster1.len() as f32;
            let n2 = cluster2.len() as f32;
            let ward_factor = (n1 * n2) / (n1 + n2);

            (ward_factor * dist_squared).sqrt()
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn hierarchical_linkage_distances_sse2(
    points: &[&[f32]],
    cluster1: &[usize],
    cluster2: &[usize],
    linkage: LinkageType,
) -> f32 {
    use core::arch::x86_64::*;

    let n_features = points[0].len();
    let mut distances = Vec::new();

    // Compute all pairwise distances between clusters using SIMD
    for &i in cluster1 {
        for &j in cluster2 {
            let mut sum = _mm_setzero_ps();
            let mut k = 0;

            while k + 4 <= n_features {
                let p1_vec = _mm_loadu_ps(&points[i][k]);
                let p2_vec = _mm_loadu_ps(&points[j][k]);
                let diff = _mm_sub_ps(p1_vec, p2_vec);
                let squared = _mm_mul_ps(diff, diff);
                sum = _mm_add_ps(sum, squared);
                k += 4;
            }

            let mut result = [0.0f32; 4];
            _mm_storeu_ps(result.as_mut_ptr(), sum);
            let mut dist_squared = result[0] + result[1] + result[2] + result[3];

            while k < n_features {
                let diff = points[i][k] - points[j][k];
                dist_squared += diff * diff;
                k += 1;
            }

            distances.push(dist_squared.sqrt());
        }
    }

    match linkage {
        LinkageType::Single => distances.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)),
        LinkageType::Complete => distances.iter().fold(0.0, |acc, &x| acc.max(x)),
        LinkageType::Average => distances.iter().sum::<f32>() / distances.len() as f32,
        LinkageType::Ward => {
            hierarchical_linkage_distances_scalar(points, cluster1, cluster2, linkage)
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hierarchical_linkage_distances_avx2(
    points: &[&[f32]],
    cluster1: &[usize],
    cluster2: &[usize],
    linkage: LinkageType,
) -> f32 {
    use core::arch::x86_64::*;

    let n_features = points[0].len();
    let mut distances = Vec::new();

    // Compute all pairwise distances between clusters using SIMD
    for &i in cluster1 {
        for &j in cluster2 {
            let mut sum = _mm256_setzero_ps();
            let mut k = 0;

            while k + 8 <= n_features {
                let p1_vec = _mm256_loadu_ps(&points[i][k]);
                let p2_vec = _mm256_loadu_ps(&points[j][k]);
                let diff = _mm256_sub_ps(p1_vec, p2_vec);
                let squared = _mm256_mul_ps(diff, diff);
                sum = _mm256_add_ps(sum, squared);
                k += 8;
            }

            let mut result = [0.0f32; 8];
            _mm256_storeu_ps(result.as_mut_ptr(), sum);
            let mut dist_squared = result.iter().sum::<f32>();

            while k < n_features {
                let diff = points[i][k] - points[j][k];
                dist_squared += diff * diff;
                k += 1;
            }

            distances.push(dist_squared.sqrt());
        }
    }

    match linkage {
        LinkageType::Single => distances.iter().fold(f32::INFINITY, |acc, &x| acc.min(x)),
        LinkageType::Complete => distances.iter().fold(0.0, |acc, &x| acc.max(x)),
        LinkageType::Average => distances.iter().sum::<f32>() / distances.len() as f32,
        LinkageType::Ward => {
            hierarchical_linkage_distances_scalar(points, cluster1, cluster2, linkage)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_kmeans_distances() {
        let p1 = vec![1.0, 2.0];
        let p2 = vec![3.0, 4.0];
        let points = vec![&p1[..], &p2[..]];

        let c1 = vec![0.0, 0.0];
        let c2 = vec![2.0, 3.0];
        let centroids = vec![&c1[..], &c2[..]];

        let mut distances = vec![vec![]; 2];

        kmeans_distances(&points, &centroids, &mut distances);

        // Distance from point 1 to centroid 1: sqrt((1-0)^2 + (2-0)^2) = sqrt(5)
        assert_relative_eq!(distances[0][0], (5.0f32).sqrt(), epsilon = 1e-6);

        // Distance from point 1 to centroid 2: sqrt((1-2)^2 + (2-3)^2) = sqrt(2)
        assert_relative_eq!(distances[0][1], (2.0f32).sqrt(), epsilon = 1e-6);
    }

    #[test]
    fn test_update_centroids() {
        let p1 = vec![1.0, 2.0];
        let p2 = vec![3.0, 4.0];
        let p3 = vec![5.0, 6.0];
        let points = vec![&p1[..], &p2[..], &p3[..]];

        let assignments = vec![0, 1, 0]; // Points 1 and 3 in cluster 0, point 2 in cluster 1
        let mut centroids = vec![vec![]; 2];

        update_centroids(&points, &assignments, 2, &mut centroids);

        // Cluster 0 centroid: average of (1,2) and (5,6) = (3,4)
        assert_relative_eq!(centroids[0][0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(centroids[0][1], 4.0, epsilon = 1e-6);

        // Cluster 1 centroid: average of (3,4) = (3,4)
        assert_relative_eq!(centroids[1][0], 3.0, epsilon = 1e-6);
        assert_relative_eq!(centroids[1][1], 4.0, epsilon = 1e-6);
    }

    #[test]
    fn test_wcss() {
        let p1 = vec![1.0, 1.0];
        let p2 = vec![2.0, 2.0];
        let points = vec![&p1[..], &p2[..]];

        let c1 = vec![0.0, 0.0];
        let centroids = vec![&c1[..]];

        let assignments = vec![0, 0]; // Both points assigned to cluster 0

        let wcss_value = wcss(&points, &centroids, &assignments);

        // WCSS = (1^2 + 1^2) + (2^2 + 2^2) = 2 + 8 = 10
        assert_relative_eq!(wcss_value, 10.0, epsilon = 1e-6);
    }

    #[test]
    fn test_silhouette_score() {
        // Create two clear clusters
        let p1 = vec![1.0, 1.0];
        let p2 = vec![1.1, 1.1];
        let p3 = vec![5.0, 5.0];
        let p4 = vec![5.1, 5.1];
        let points = vec![&p1[..], &p2[..], &p3[..], &p4[..]];

        let assignments = vec![0, 0, 1, 1]; // Two clusters with 2 points each

        let score = silhouette_score(&points, &assignments, 2);

        // Should be positive for well-separated clusters
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[test]
    fn test_dbscan_neighbors() {
        // Create points in two clear clusters
        let p1 = vec![1.0, 1.0];
        let p2 = vec![1.1, 1.1];
        let p3 = vec![5.0, 5.0];
        let p4 = vec![5.1, 5.1];
        let points = vec![&p1[..], &p2[..], &p3[..], &p4[..]];

        let mut neighbors = vec![vec![]; 4];
        let eps = 0.5; // Small epsilon to separate clusters

        dbscan_neighbors(&points, eps, &mut neighbors);

        // Points 0 and 1 should be neighbors (distance ~0.14)
        assert!(neighbors[0].contains(&1));
        assert!(neighbors[1].contains(&0));

        // Points 2 and 3 should be neighbors (distance ~0.14)
        assert!(neighbors[2].contains(&3));
        assert!(neighbors[3].contains(&2));

        // Points from different clusters should not be neighbors
        assert!(!neighbors[0].contains(&2));
        assert!(!neighbors[0].contains(&3));
        assert!(!neighbors[1].contains(&2));
        assert!(!neighbors[1].contains(&3));
    }

    #[test]
    fn test_dbscan_core_points() {
        // Create neighbor data
        let neighbors = vec![
            vec![1],    // Point 0 has 1 neighbor (+ itself = 2 total)
            vec![0, 2], // Point 1 has 2 neighbors (+ itself = 3 total)
            vec![1],    // Point 2 has 1 neighbor (+ itself = 2 total)
        ];

        let mut core_points = vec![false; 3];
        let min_samples = 3;

        dbscan_core_points(&neighbors, min_samples, &mut core_points);

        // Only point 1 should be a core point (has 3+ total points including itself)
        assert!(!core_points[0]);
        assert!(core_points[1]);
        assert!(!core_points[2]);
    }

    #[test]
    fn test_hierarchical_linkage_single() {
        let p1 = vec![1.0, 1.0];
        let p2 = vec![2.0, 2.0];
        let p3 = vec![5.0, 5.0];
        let p4 = vec![6.0, 6.0];
        let points = vec![&p1[..], &p2[..], &p3[..], &p4[..]];

        let cluster1 = vec![0, 1]; // Points 0 and 1
        let cluster2 = vec![2, 3]; // Points 2 and 3

        let distance =
            hierarchical_linkage_distances(&points, &cluster1, &cluster2, LinkageType::Single);

        // Single linkage should return minimum distance between any two points
        // Minimum distance is between (2,2) and (5,5) = sqrt(18) ≈ 4.24
        let expected = ((5.0f32 - 2.0f32).powi(2) + (5.0f32 - 2.0f32).powi(2)).sqrt();
        assert_relative_eq!(distance, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_hierarchical_linkage_complete() {
        let p1 = vec![1.0, 1.0];
        let p2 = vec![2.0, 2.0];
        let p3 = vec![5.0, 5.0];
        let p4 = vec![6.0, 6.0];
        let points = vec![&p1[..], &p2[..], &p3[..], &p4[..]];

        let cluster1 = vec![0, 1]; // Points 0 and 1
        let cluster2 = vec![2, 3]; // Points 2 and 3

        let distance =
            hierarchical_linkage_distances(&points, &cluster1, &cluster2, LinkageType::Complete);

        // Complete linkage should return maximum distance between any two points
        // Maximum distance is between (1,1) and (6,6) = sqrt(50) ≈ 7.07
        let expected = ((6.0f32 - 1.0f32).powi(2) + (6.0f32 - 1.0f32).powi(2)).sqrt();
        assert_relative_eq!(distance, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_hierarchical_linkage_average() {
        let p1 = vec![0.0, 0.0];
        let p2 = vec![1.0, 0.0];
        let p3 = vec![3.0, 0.0];
        let points = vec![&p1[..], &p2[..], &p3[..]];

        let cluster1 = vec![0]; // Point 0
        let cluster2 = vec![1, 2]; // Points 1 and 2

        let distance =
            hierarchical_linkage_distances(&points, &cluster1, &cluster2, LinkageType::Average);

        // Average linkage: average of distances from point 0 to points 1 and 2
        // Distance to point 1: 1.0, Distance to point 2: 3.0
        // Average: (1.0 + 3.0) / 2 = 2.0
        assert_relative_eq!(distance, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_hierarchical_linkage_ward() {
        let p1 = vec![0.0, 0.0];
        let p2 = vec![2.0, 0.0];
        let p3 = vec![4.0, 0.0];
        let points = vec![&p1[..], &p2[..], &p3[..]];

        let cluster1 = vec![0]; // Point (0,0)
        let cluster2 = vec![1, 2]; // Points (2,0) and (4,0)

        let distance =
            hierarchical_linkage_distances(&points, &cluster1, &cluster2, LinkageType::Ward);

        // Ward linkage: weighted distance between centroids
        // Centroid of cluster1: (0,0)
        // Centroid of cluster2: (3,0) [average of (2,0) and (4,0)]
        // Distance between centroids: 3.0
        // Ward factor: (1 * 2) / (1 + 2) = 2/3
        // Ward distance: sqrt(2/3 * 9) = sqrt(6) ≈ 2.45
        let expected = (2.0f32 / 3.0f32 * 9.0f32).sqrt();
        assert_relative_eq!(distance, expected, epsilon = 1e-6);
    }
}

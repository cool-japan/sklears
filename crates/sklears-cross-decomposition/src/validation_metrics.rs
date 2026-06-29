//! Self-contained validation metrics for cross-decomposition evaluation.
//!
//! This module provides the *real* numerical routines used by the validation
//! framework: clustering-stability indices (Jaccard / Rand / Adjusted Rand),
//! the silhouette coefficient, principal angles between subspaces, and an
//! empirical p-value helper for resampling tests. None of these functions
//! fabricate values; each computes its result from its inputs.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::types::Float;

/// Errors that can arise while computing validation metrics.
#[derive(Debug, Clone)]
pub enum MetricError {
    /// Inputs had incompatible shapes.
    ShapeMismatch(String),
    /// A required linear-algebra routine failed.
    Linalg(String),
}

impl std::fmt::Display for MetricError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricError::ShapeMismatch(msg) => write!(f, "metric shape mismatch: {msg}"),
            MetricError::Linalg(msg) => write!(f, "metric linalg failure: {msg}"),
        }
    }
}

impl std::error::Error for MetricError {}

/// Counts of agreeing/disagreeing pairs between two label vectors.
///
/// Returns `(a, b, c, d)` where, over all unordered pairs of points,
/// `a` = together in both clusterings, `b` = together in A only,
/// `c` = together in B only, `d` = separate in both.
fn pair_counts(labels_a: &[usize], labels_b: &[usize]) -> (u64, u64, u64, u64) {
    let n = labels_a.len();
    let (mut a, mut b, mut c, mut d) = (0u64, 0u64, 0u64, 0u64);
    for i in 0..n {
        for j in (i + 1)..n {
            let same_a = labels_a[i] == labels_a[j];
            let same_b = labels_b[i] == labels_b[j];
            match (same_a, same_b) {
                (true, true) => a += 1,
                (true, false) => b += 1,
                (false, true) => c += 1,
                (false, false) => d += 1,
            }
        }
    }
    (a, b, c, d)
}

/// Jaccard index between two clusterings.
///
/// `J = a / (a + b + c)` over all point pairs. Returns `1.0` for identical
/// clusterings, `< 1.0` otherwise. If both inputs put every point in its own
/// cluster (no positive pairs at all), the index is defined as `1.0`.
pub fn jaccard_index(labels_a: &[usize], labels_b: &[usize]) -> Result<Float, MetricError> {
    if labels_a.len() != labels_b.len() {
        return Err(MetricError::ShapeMismatch(format!(
            "label lengths differ: {} vs {}",
            labels_a.len(),
            labels_b.len()
        )));
    }
    let (a, b, c, _d) = pair_counts(labels_a, labels_b);
    let denom = a + b + c;
    if denom == 0 {
        return Ok(1.0);
    }
    Ok(a as Float / denom as Float)
}

/// Rand index between two clusterings: `(a + d) / (n choose 2)`.
///
/// Returns `1.0` for identical clusterings. With fewer than two points the
/// index is defined as `1.0` (no pairs to disagree on).
pub fn rand_index(labels_a: &[usize], labels_b: &[usize]) -> Result<Float, MetricError> {
    if labels_a.len() != labels_b.len() {
        return Err(MetricError::ShapeMismatch(format!(
            "label lengths differ: {} vs {}",
            labels_a.len(),
            labels_b.len()
        )));
    }
    let (a, b, c, d) = pair_counts(labels_a, labels_b);
    let total = a + b + c + d;
    if total == 0 {
        return Ok(1.0);
    }
    Ok((a + d) as Float / total as Float)
}

/// Adjusted Rand index between two clusterings (chance-corrected).
///
/// `ARI = (RI - E[RI]) / (max(RI) - E[RI])`, computed from the contingency
/// table. Equals `1.0` for identical clusterings and ~`0.0` for random label
/// agreement. Returns `1.0` in the degenerate all-singletons / single-point
/// case where the denominator vanishes.
pub fn adjusted_rand_index(labels_a: &[usize], labels_b: &[usize]) -> Result<Float, MetricError> {
    if labels_a.len() != labels_b.len() {
        return Err(MetricError::ShapeMismatch(format!(
            "label lengths differ: {} vs {}",
            labels_a.len(),
            labels_b.len()
        )));
    }
    let n = labels_a.len();
    if n < 2 {
        return Ok(1.0);
    }

    let max_a = labels_a.iter().copied().max().unwrap_or(0) + 1;
    let max_b = labels_b.iter().copied().max().unwrap_or(0) + 1;
    let mut contingency = vec![0u64; max_a * max_b];
    let mut row_sums = vec![0u64; max_a];
    let mut col_sums = vec![0u64; max_b];
    for (&la, &lb) in labels_a.iter().zip(labels_b.iter()) {
        contingency[la * max_b + lb] += 1;
        row_sums[la] += 1;
        col_sums[lb] += 1;
    }

    let comb2 = |x: u64| -> Float { (x as Float) * (x as Float - 1.0) / 2.0 };

    let sum_comb_c: Float = contingency.iter().map(|&v| comb2(v)).sum();
    let sum_comb_a: Float = row_sums.iter().map(|&v| comb2(v)).sum();
    let sum_comb_b: Float = col_sums.iter().map(|&v| comb2(v)).sum();
    let total_pairs = comb2(n as u64);

    let expected = sum_comb_a * sum_comb_b / total_pairs;
    let max_index = 0.5 * (sum_comb_a + sum_comb_b);
    let denom = max_index - expected;
    if denom.abs() < Float::EPSILON {
        return Ok(1.0);
    }
    Ok((sum_comb_c - expected) / denom)
}

/// Euclidean distance between two rows of a data matrix.
fn row_distance(data: &ArrayView2<Float>, i: usize, j: usize) -> Float {
    data.row(i)
        .iter()
        .zip(data.row(j).iter())
        .map(|(&u, &v)| (u - v) * (u - v))
        .sum::<Float>()
        .sqrt()
}

/// Silhouette coefficient averaged over all points.
///
/// For each point, `s = (b - a) / max(a, b)` with `a` = mean intra-cluster
/// distance and `b` = the smallest mean distance to any other cluster.
/// Points whose cluster is a singleton contribute `0` (their `a` is undefined).
/// Requires the data matrix and labels. Well-separated clusters yield values
/// near `+1`; overlapping clusters yield values near `0` or negative.
pub fn silhouette_coefficient(
    data: &ArrayView2<Float>,
    labels: &[usize],
) -> Result<Float, MetricError> {
    let n = data.nrows();
    if labels.len() != n {
        return Err(MetricError::ShapeMismatch(format!(
            "labels length {} does not match {} samples",
            labels.len(),
            n
        )));
    }
    if n < 2 {
        return Ok(0.0);
    }

    let n_clusters = labels.iter().copied().max().unwrap_or(0) + 1;
    if n_clusters < 2 {
        return Ok(0.0);
    }

    let mut cluster_sizes = vec![0usize; n_clusters];
    for &l in labels {
        cluster_sizes[l] += 1;
    }

    let mut silhouette_sum = 0.0;
    let mut counted = 0usize;

    for i in 0..n {
        // Sum of distances from point i to every cluster.
        let mut sum_dist = vec![0.0 as Float; n_clusters];
        for j in 0..n {
            if i == j {
                continue;
            }
            sum_dist[labels[j]] += row_distance(data, i, j);
        }

        let own = labels[i];
        if cluster_sizes[own] <= 1 {
            // Singleton cluster: silhouette undefined, contributes 0.
            counted += 1;
            continue;
        }

        let a_i = sum_dist[own] / (cluster_sizes[own] - 1) as Float;

        let mut b_i = Float::INFINITY;
        for (cluster, &size) in cluster_sizes.iter().enumerate() {
            if cluster == own || size == 0 {
                continue;
            }
            let mean_other = sum_dist[cluster] / size as Float;
            if mean_other < b_i {
                b_i = mean_other;
            }
        }

        if b_i.is_finite() {
            let denom = a_i.max(b_i);
            if denom > 0.0 {
                silhouette_sum += (b_i - a_i) / denom;
            }
            counted += 1;
        }
    }

    if counted == 0 {
        return Ok(0.0);
    }
    Ok(silhouette_sum / counted as Float)
}

/// Assign each row of `data` to its nearest centroid (1-NN to centroids),
/// returning a label per row. Used to build a stable, deterministic clustering
/// of canonical scores given a fixed set of reference centroids.
pub fn assign_to_centroids(data: &ArrayView2<Float>, centroids: &Array2<Float>) -> Vec<usize> {
    (0..data.nrows())
        .map(|i| {
            let row = data.row(i);
            let mut best = 0usize;
            let mut best_dist = Float::INFINITY;
            for (c, centroid) in centroids.outer_iter().enumerate() {
                let dist = row
                    .iter()
                    .zip(centroid.iter())
                    .map(|(&u, &v)| (u - v) * (u - v))
                    .sum::<Float>();
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Deterministic k-means (Lloyd's algorithm) with farthest-point seeding.
///
/// Returns `(labels, centroids)`. Seeding is deterministic (first point, then
/// repeatedly the point farthest from the current centroid set) so the result
/// is reproducible without an RNG. Empty clusters are re-seeded to the
/// currently farthest point. This is a real clustering routine, not a stub.
pub fn kmeans(
    data: &ArrayView2<Float>,
    k: usize,
    max_iter: usize,
) -> Result<(Vec<usize>, Array2<Float>), MetricError> {
    let n = data.nrows();
    let d = data.ncols();
    let k = k.min(n).max(1);

    // Farthest-point (k-means++ style, deterministic) initialization.
    let mut centroids = Array2::<Float>::zeros((k, d));
    centroids.row_mut(0).assign(&data.row(0));
    for c in 1..k {
        let mut best_point = 0usize;
        let mut best_min_dist = -1.0 as Float;
        for i in 0..n {
            let mut min_dist = Float::INFINITY;
            for prev in 0..c {
                let dist = data
                    .row(i)
                    .iter()
                    .zip(centroids.row(prev).iter())
                    .map(|(&u, &v)| (u - v) * (u - v))
                    .sum::<Float>();
                if dist < min_dist {
                    min_dist = dist;
                }
            }
            if min_dist > best_min_dist {
                best_min_dist = min_dist;
                best_point = i;
            }
        }
        centroids.row_mut(c).assign(&data.row(best_point));
    }

    let mut labels = vec![0usize; n];
    for _ in 0..max_iter {
        let new_labels = assign_to_centroids(data, &centroids);
        if new_labels == labels {
            labels = new_labels;
            break;
        }
        labels = new_labels;

        // Recompute centroids.
        let mut sums = Array2::<Float>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for (i, &lab) in labels.iter().enumerate() {
            let mut target = sums.row_mut(lab);
            target += &data.row(i);
            counts[lab] += 1;
        }
        for (c, &count) in counts.iter().enumerate() {
            if count > 0 {
                let inv = 1.0 / count as Float;
                centroids.row_mut(c).assign(&(&sums.row(c) * inv));
            } else {
                // Empty cluster: re-seed to the point farthest from its centroid.
                let mut worst_point = 0usize;
                let mut worst_dist = -1.0 as Float;
                for (i, &lab) in labels.iter().enumerate() {
                    let dist = data
                        .row(i)
                        .iter()
                        .zip(centroids.row(lab).iter())
                        .map(|(&u, &v)| (u - v) * (u - v))
                        .sum::<Float>();
                    if dist > worst_dist {
                        worst_dist = dist;
                        worst_point = i;
                    }
                }
                centroids.row_mut(c).assign(&data.row(worst_point));
            }
        }
    }

    Ok((labels, centroids))
}

/// Orthonormalize the columns of a matrix via modified Gram-Schmidt.
///
/// Returns a matrix whose columns form an orthonormal basis of the column
/// space of the input (dropping numerically dependent columns). The result
/// has the same number of rows and at most `min(rows, cols)` columns.
pub fn orthonormal_basis(matrix: &ArrayView2<Float>) -> Array2<Float> {
    let n_rows = matrix.nrows();
    let n_cols = matrix.ncols();
    let mut basis: Vec<Array1<Float>> = Vec::with_capacity(n_cols);

    for col in matrix.columns() {
        let mut v = col.to_owned();
        for q in &basis {
            let proj = q.dot(&v);
            v = &v - &(q * proj);
        }
        let norm = v.dot(&v).sqrt();
        if norm > 1e-10 {
            v.mapv_inplace(|x| x / norm);
            basis.push(v);
        }
    }

    let mut result = Array2::<Float>::zeros((n_rows, basis.len()));
    for (j, q) in basis.iter().enumerate() {
        result.column_mut(j).assign(q);
    }
    result
}

/// Principal angles (in radians) between two subspaces.
///
/// `u` and `v` are arbitrary spanning matrices (columns span each subspace).
/// The angles are `arccos` of the singular values of `Q_uᵀ Q_v`, where
/// `Q_u`, `Q_v` are orthonormal bases of the two column spaces. Identical
/// subspaces give angles of `0`; orthogonal subspaces give `π/2`. The number
/// of returned angles equals `min(rank(u), rank(v))`.
pub fn principal_angles(
    u: &ArrayView2<Float>,
    v: &ArrayView2<Float>,
) -> Result<Array1<Float>, MetricError> {
    if u.nrows() != v.nrows() {
        return Err(MetricError::ShapeMismatch(format!(
            "subspaces live in different ambient dimensions: {} vs {}",
            u.nrows(),
            v.nrows()
        )));
    }

    let qu = orthonormal_basis(u);
    let qv = orthonormal_basis(v);
    if qu.ncols() == 0 || qv.ncols() == 0 {
        return Ok(Array1::zeros(0));
    }

    let cross = qu.t().dot(&qv);
    let (_, singular_values, _) = scirs2_linalg::compat::svd(&cross, false)
        .map_err(|e| MetricError::Linalg(e.to_string()))?;

    let angles = singular_values.mapv(|s| s.clamp(-1.0, 1.0).acos());
    Ok(angles)
}

/// Subspace recovery score in `[0, 1]` derived from principal angles.
///
/// Computed as the mean of `cos(theta_i)` over the principal angles, i.e. the
/// average alignment between the two subspaces. `1.0` means perfect recovery
/// (all angles zero); `0.0` means fully orthogonal subspaces. Returns `0.0`
/// when there are no comparable directions.
pub fn subspace_recovery(angles: &Array1<Float>) -> Float {
    if angles.is_empty() {
        return 0.0;
    }
    let sum_cos: Float = angles.iter().map(|&a| a.cos()).sum();
    (sum_cos / angles.len() as Float).clamp(0.0, 1.0)
}

/// Empirical p-value for a one-sided "greater-or-equal" resampling test.
///
/// Uses the standard `(#{null >= observed} + 1) / (n + 1)` estimator, which is
/// never zero and is unbiased under the permutation null. `null_stats` is the
/// resampled null distribution of the statistic.
pub fn empirical_p_value(observed: Float, null_stats: &[Float]) -> Float {
    let n = null_stats.len();
    if n == 0 {
        return 1.0;
    }
    let count = null_stats
        .iter()
        .filter(|&&s| s >= observed - 1e-12)
        .count();
    (count as Float + 1.0) / (n as Float + 1.0)
}

/// Sample quantile of a slice using linear interpolation between order
/// statistics (type-7, the default used by NumPy / R).
pub fn quantile(values: &[Float], q: Float) -> Float {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<Float> = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let q = q.clamp(0.0, 1.0);
    let pos = q * (sorted.len() as Float - 1.0);
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = pos - lo as Float;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn jaccard_identical_is_one() {
        let a = vec![0, 0, 1, 1, 2, 2];
        assert!((jaccard_index(&a, &a).expect("jaccard") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn jaccard_different_is_less_than_one() {
        let a = vec![0, 0, 1, 1];
        let b = vec![0, 1, 0, 1];
        let j = jaccard_index(&a, &b).expect("jaccard");
        assert!(j < 1.0, "expected j < 1.0, got {j}");
        assert!(j >= 0.0);
    }

    #[test]
    fn rand_identical_is_one() {
        let a = vec![0, 1, 1, 2, 2, 0];
        assert!((rand_index(&a, &a).expect("rand") - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rand_different_is_less_than_one() {
        let a = vec![0, 0, 1, 1];
        let b = vec![0, 1, 0, 1];
        let r = rand_index(&a, &b).expect("rand");
        assert!((0.0..1.0).contains(&r), "got {r}");
    }

    #[test]
    fn adjusted_rand_identical_is_one() {
        let a = vec![0, 0, 1, 1, 2, 2];
        assert!((adjusted_rand_index(&a, &a).expect("ari") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn adjusted_rand_relabel_invariant() {
        let a = vec![0, 0, 1, 1, 2, 2];
        let b = vec![2, 2, 0, 0, 1, 1]; // same partition, relabeled
        assert!((adjusted_rand_index(&a, &b).expect("ari") - 1.0).abs() < 1e-9);
    }

    #[test]
    fn silhouette_well_separated_near_one() {
        // Two tight, far-apart clusters.
        let data = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [10.0, 10.0],
            [10.1, 10.0],
            [10.0, 10.1],
        ];
        let labels = vec![0, 0, 0, 1, 1, 1];
        let s = silhouette_coefficient(&data.view(), &labels).expect("silhouette");
        assert!(s > 0.9, "expected silhouette near 1, got {s}");
    }

    #[test]
    fn silhouette_bad_clustering_is_low() {
        // Same tight clusters but labels split them wrongly (interleaved).
        let data = array![[0.0, 0.0], [0.1, 0.0], [10.0, 10.0], [10.1, 10.0],];
        let good = vec![0, 0, 1, 1];
        let bad = vec![0, 1, 0, 1];
        let s_good = silhouette_coefficient(&data.view(), &good).expect("silhouette");
        let s_bad = silhouette_coefficient(&data.view(), &bad).expect("silhouette");
        assert!(s_good > s_bad, "good {s_good} should beat bad {s_bad}");
        assert!(s_bad < 0.5);
    }

    #[test]
    fn kmeans_recovers_two_clusters() {
        let data = array![
            [0.0, 0.0],
            [0.2, 0.1],
            [0.1, 0.2],
            [9.0, 9.0],
            [9.1, 8.9],
            [8.9, 9.1],
        ];
        let (labels, centroids) = kmeans(&data.view(), 2, 50).expect("kmeans");
        assert_eq!(centroids.nrows(), 2);
        // First three and last three should each share a label.
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
        assert_ne!(labels[0], labels[3]);
    }

    #[test]
    fn principal_angles_identical_subspace_is_zero() {
        let u = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let angles = principal_angles(&u.view(), &u.view()).expect("angles");
        assert_eq!(angles.len(), 2);
        for &a in angles.iter() {
            assert!(a.abs() < 1e-6, "expected 0, got {a}");
        }
        assert!((subspace_recovery(&angles) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn principal_angles_orthogonal_subspace_is_half_pi() {
        let u = array![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]];
        let v = array![[0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let angles = principal_angles(&u.view(), &v.view()).expect("angles");
        let half_pi = std::f64::consts::FRAC_PI_2 as Float;
        for &a in angles.iter() {
            assert!((a - half_pi).abs() < 1e-6, "expected pi/2, got {a}");
        }
        assert!(subspace_recovery(&angles) < 1e-6);
    }

    #[test]
    fn empirical_p_value_extremes() {
        // Observed far above the null -> small p-value.
        let null: Vec<Float> = (0..100).map(|i| i as Float / 100.0).collect();
        let p_small = empirical_p_value(5.0, &null);
        assert!(p_small < 0.02, "got {p_small}");
        // Observed at the bottom of the null -> p-value near 1.
        let p_large = empirical_p_value(-5.0, &null);
        assert!(p_large > 0.98, "got {p_large}");
    }

    #[test]
    fn quantile_matches_known_values() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        assert!((quantile(&v, 0.0) - 1.0).abs() < 1e-12);
        assert!((quantile(&v, 1.0) - 4.0).abs() < 1e-12);
        assert!((quantile(&v, 0.5) - 2.5).abs() < 1e-12);
    }
}

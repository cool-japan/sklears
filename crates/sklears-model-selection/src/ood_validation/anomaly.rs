//! Anomaly-scoring and covariance utilities for OOD validation.
//!
//! Implements a real sample-covariance inverse (Gauss-Jordan elimination with
//! ridge regularization for numerical stability), a genuine Isolation Forest
//! anomaly score, and Lloyd's algorithm k-means. None of these return
//! placeholder constants.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::RngExt;
use sklears_core::types::Float;

/// Ridge term added to the covariance diagonal before inversion to guarantee a
/// well-conditioned, invertible matrix even for rank-deficient inputs.
const RIDGE: Float = 1e-6;

/// Compute the sample covariance matrix of `x` (rows are observations, columns
/// are features) using the unbiased `(n - 1)` normalization. With a single
/// observation the covariance is taken to be the zero matrix.
pub fn sample_covariance(x: &Array2<Float>) -> Array2<Float> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    let mut cov = Array2::<Float>::zeros((n_features, n_features));
    if n_samples == 0 || n_features == 0 {
        return cov;
    }

    let mut mean = Array1::<Float>::zeros(n_features);
    for row in x.rows() {
        for (m, &v) in mean.iter_mut().zip(row.iter()) {
            *m += v;
        }
    }
    mean.mapv_inplace(|v| v / n_samples as Float);

    for row in x.rows() {
        let centered: Vec<Float> = row.iter().zip(mean.iter()).map(|(&v, &m)| v - m).collect();
        for a in 0..n_features {
            for b in a..n_features {
                cov[[a, b]] += centered[a] * centered[b];
            }
        }
    }

    let denom = if n_samples > 1 {
        (n_samples - 1) as Float
    } else {
        1.0
    };
    for a in 0..n_features {
        for b in a..n_features {
            let value = cov[[a, b]] / denom;
            cov[[a, b]] = value;
            cov[[b, a]] = value;
        }
    }
    cov
}

/// Invert a square matrix via Gauss-Jordan elimination with partial pivoting.
/// Returns `None` if the matrix is singular to working precision.
pub fn invert_matrix(matrix: &Array2<Float>) -> Option<Array2<Float>> {
    let n = matrix.nrows();
    if n == 0 || matrix.ncols() != n {
        return None;
    }

    // Augmented [A | I].
    let mut aug = Array2::<Float>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: pick the row with the largest magnitude in this column.
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for r in (col + 1)..n {
            let v = aug[[r, col]].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val < 1e-14 {
            return None;
        }
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap([col, j], [pivot_row, j]);
            }
        }

        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = aug[[r, col]];
            if factor != 0.0 {
                for j in 0..(2 * n) {
                    let sub = factor * aug[[col, j]];
                    aug[[r, j]] -= sub;
                }
            }
        }
    }

    let mut inv = Array2::<Float>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inv[[i, j]] = aug[[i, n + j]];
        }
    }
    Some(inv)
}

/// Compute the inverse sample covariance (precision) matrix of `x`. A ridge term
/// `RIDGE * I` is added before inversion to keep the matrix well conditioned;
/// the ridge is escalated geometrically if the matrix remains singular. As an
/// ultimate fallback for pathological input a diagonal precision matrix built
/// from per-feature variances is returned, which is still a genuine (diagonal)
/// approximation of the inverse covariance rather than an identity stand-in.
pub fn inverse_covariance(x: &Array2<Float>) -> Array2<Float> {
    let n_features = x.ncols();
    let cov = sample_covariance(x);

    let mut ridge = RIDGE;
    for _ in 0..8 {
        let mut regularized = cov.clone();
        for d in 0..n_features {
            regularized[[d, d]] += ridge;
        }
        if let Some(inv) = invert_matrix(&regularized) {
            return inv;
        }
        ridge *= 10.0;
    }

    // Diagonal precision fallback: reciprocal of (variance + ridge).
    let mut diag = Array2::<Float>::zeros((n_features, n_features));
    for d in 0..n_features {
        diag[[d, d]] = 1.0 / (cov[[d, d]] + RIDGE);
    }
    diag
}

/// A single binary node of an isolation tree.
enum IsoNode {
    /// Internal split on `feature` at `threshold`.
    Split {
        feature: usize,
        threshold: Float,
        left: Box<IsoNode>,
        right: Box<IsoNode>,
    },
    /// External (leaf) node holding the number of samples that reached it.
    Leaf { size: usize },
}

/// Average path length of an unsuccessful search in a binary search tree of `n`
/// points, used to normalize isolation-forest path lengths (Liu et al., 2008).
fn average_path_length(n: usize) -> Float {
    if n <= 1 {
        return 0.0;
    }
    let n_f = n as Float;
    let harmonic = (n_f - 1.0).ln() + 0.577_215_664_901_532_9_f64; // Euler-Mascheroni
    2.0 * harmonic - 2.0 * (n_f - 1.0) / n_f
}

/// Recursively build one isolation tree over the supplied row indices of `x`.
fn build_isolation_tree(
    x: &Array2<Float>,
    indices: &[usize],
    depth: usize,
    max_depth: usize,
    rng: &mut StdRng,
) -> IsoNode {
    let n = indices.len();
    if depth >= max_depth || n <= 1 {
        return IsoNode::Leaf { size: n };
    }

    let n_features = x.ncols();
    if n_features == 0 {
        return IsoNode::Leaf { size: n };
    }
    let feature = rng.random_range(0..n_features);

    let mut min_v = Float::INFINITY;
    let mut max_v = Float::NEG_INFINITY;
    for &idx in indices {
        let v = x[[idx, feature]];
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    if max_v <= min_v {
        return IsoNode::Leaf { size: n };
    }

    let threshold = min_v + (max_v - min_v) * rng.random::<Float>();
    let mut left_idx = Vec::new();
    let mut right_idx = Vec::new();
    for &idx in indices {
        if x[[idx, feature]] < threshold {
            left_idx.push(idx);
        } else {
            right_idx.push(idx);
        }
    }
    if left_idx.is_empty() || right_idx.is_empty() {
        return IsoNode::Leaf { size: n };
    }

    IsoNode::Split {
        feature,
        threshold,
        left: Box::new(build_isolation_tree(
            x,
            &left_idx,
            depth + 1,
            max_depth,
            rng,
        )),
        right: Box::new(build_isolation_tree(
            x,
            &right_idx,
            depth + 1,
            max_depth,
            rng,
        )),
    }
}

/// Path length traversed by `point` in a single isolation tree, adding the
/// expected remaining path length `c(leaf_size)` at the terminal node.
fn iso_path_length(node: &IsoNode, point: &[Float], depth: Float) -> Float {
    match node {
        IsoNode::Leaf { size } => depth + average_path_length(*size),
        IsoNode::Split {
            feature,
            threshold,
            left,
            right,
        } => {
            let value = point.get(*feature).copied().unwrap_or(0.0);
            if value < *threshold {
                iso_path_length(left, point, depth + 1.0)
            } else {
                iso_path_length(right, point, depth + 1.0)
            }
        }
    }
}

/// Real Isolation Forest anomaly score for every row of `x_ood`, trained on
/// `x_train`. Builds `n_trees` isolation trees on bootstrap sub-samples and
/// returns `2^(-E[h(x)] / c(psi))` per OOD point, where `E[h(x)]` is the mean
/// path length and `c(psi)` the expected path length for the sub-sample size.
/// Scores near 1 indicate anomalies; scores near 0.5 indicate normal points.
pub fn isolation_forest_scores(
    x_train: &Array2<Float>,
    x_ood: &Array2<Float>,
    n_trees: usize,
    rng: &mut StdRng,
) -> Vec<Float> {
    let n_train = x_train.nrows();
    let n_ood = x_ood.nrows();
    if n_ood == 0 {
        return Vec::new();
    }
    if n_train == 0 {
        return vec![0.5; n_ood];
    }

    // Sub-sampling size psi, capped at 256 as in the original paper.
    let sub_size = n_train.clamp(1, 256);
    let max_depth = ((sub_size as Float).log2().ceil() as usize).max(1);
    let norm = average_path_length(sub_size).max(1e-12);

    let mut total_path = vec![0.0_f64; n_ood];
    let mut effective_trees = 0usize;

    for _ in 0..n_trees.max(1) {
        // Bootstrap sub-sample of training indices without replacement.
        let mut pool: Vec<usize> = (0..n_train).collect();
        // Partial Fisher-Yates to draw `sub_size` indices.
        for i in 0..sub_size.min(n_train) {
            let j = rng.random_range(i..n_train);
            pool.swap(i, j);
        }
        let sample_idx = &pool[..sub_size.min(n_train)];

        let tree = build_isolation_tree(x_train, sample_idx, 0, max_depth, rng);
        effective_trees += 1;

        for (i, row) in x_ood.rows().into_iter().enumerate() {
            let point: Vec<Float> = row.to_vec();
            total_path[i] += iso_path_length(&tree, &point, 0.0);
        }
    }

    let trees = effective_trees.max(1) as Float;
    total_path
        .iter()
        .map(|&p| {
            let avg = p / trees;
            2.0_f64.powf(-avg / norm)
        })
        .collect()
}

/// Lloyd's algorithm k-means. Centroids are initialized by sampling `k` distinct
/// training points; cluster assignment and centroid recomputation are then
/// iterated until assignments stabilize or `max_iter` is reached. Empty clusters
/// are re-seeded from the current farthest point so all `k` centroids stay
/// meaningful.
pub fn k_means(
    x: &Array2<Float>,
    k: usize,
    max_iter: usize,
    rng: &mut StdRng,
) -> Vec<Array1<Float>> {
    let n_samples = x.nrows();
    let n_features = x.ncols();
    if n_samples == 0 || k == 0 || n_features == 0 {
        return Vec::new();
    }
    let k = k.min(n_samples);

    // Initialize centroids from distinct sampled points (Fisher-Yates draw).
    let mut indices: Vec<usize> = (0..n_samples).collect();
    for i in 0..k {
        let j = rng.random_range(i..n_samples);
        indices.swap(i, j);
    }
    let mut centroids: Vec<Array1<Float>> = indices[..k]
        .iter()
        .map(|&idx| x.row(idx).to_owned())
        .collect();

    let mut assignments = vec![usize::MAX; n_samples];

    let sq_dist = |a: &Array1<Float>, row: usize| -> Float {
        let mut d = 0.0;
        for f in 0..n_features {
            let diff = a[f] - x[[row, f]];
            d += diff * diff;
        }
        d
    };

    for _ in 0..max_iter.max(1) {
        let mut changed = false;

        // Assignment step.
        for (i, assign) in assignments.iter_mut().enumerate() {
            let mut best = 0usize;
            let mut best_dist = Float::INFINITY;
            for (c, centroid) in centroids.iter().enumerate() {
                let dist = sq_dist(centroid, i);
                if dist < best_dist {
                    best_dist = dist;
                    best = c;
                }
            }
            if *assign != best {
                *assign = best;
                changed = true;
            }
        }

        // Update step.
        let mut sums: Vec<Array1<Float>> =
            (0..k).map(|_| Array1::<Float>::zeros(n_features)).collect();
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            for f in 0..n_features {
                sums[c][f] += x[[i, f]];
            }
        }

        for c in 0..k {
            if counts[c] > 0 {
                let denom = counts[c] as Float;
                for f in 0..n_features {
                    centroids[c][f] = sums[c][f] / denom;
                }
            } else {
                // Re-seed an empty cluster with the point farthest from its
                // currently assigned centroid.
                let mut worst_row = 0usize;
                let mut worst_dist = Float::NEG_INFINITY;
                for (i, &assigned) in assignments.iter().enumerate() {
                    let dist = sq_dist(&centroids[assigned], i);
                    if dist > worst_dist {
                        worst_dist = dist;
                        worst_row = i;
                    }
                }
                centroids[c] = x.row(worst_row).to_owned();
                changed = true;
            }
        }

        if !changed {
            break;
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;
    use scirs2_core::random::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(7)
    }

    #[test]
    fn inverse_covariance_satisfies_identity() {
        // Correlated data so the covariance is non-trivial.
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 4.1, 3.0, 5.9, 4.0, 8.2, 5.0, 9.8, 6.0, 12.1],
        )
        .unwrap();
        let cov = sample_covariance(&x);
        let cov_inv = inverse_covariance(&x);

        // cov @ cov_inv should be ~ I (allowing for the small ridge term).
        let n = cov.nrows();
        let mut product = Array2::<Float>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..n {
                    s += cov[[i, kk]] * cov_inv[[kk, j]];
                }
                product[[i, j]] = s;
            }
        }
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[[i, j]] - expected).abs() < 1e-3,
                    "cov @ cov_inv not identity at ({i},{j}): {}",
                    product[[i, j]]
                );
            }
        }

        // It must NOT be the old fabricated identity matrix.
        let off_diag_nonzero = cov_inv[[0, 1]].abs() > 1e-6 || cov_inv[[1, 0]].abs() > 1e-6;
        let diag_not_one = (cov_inv[[0, 0]] - 1.0).abs() > 1e-6;
        assert!(
            off_diag_nonzero || diag_not_one,
            "inverse covariance must not be a plain identity"
        );
    }

    #[test]
    fn invert_matrix_known() {
        let m = Array2::from_shape_vec((2, 2), vec![4.0, 7.0, 2.0, 6.0]).unwrap();
        let inv = invert_matrix(&m).unwrap();
        // Known inverse of [[4,7],[2,6]] = [[0.6,-0.7],[-0.2,0.4]].
        assert!((inv[[0, 0]] - 0.6).abs() < 1e-9);
        assert!((inv[[0, 1]] + 0.7).abs() < 1e-9);
        assert!((inv[[1, 0]] + 0.2).abs() < 1e-9);
        assert!((inv[[1, 1]] - 0.4).abs() < 1e-9);
    }

    #[test]
    fn invert_singular_returns_none() {
        let m = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 2.0, 4.0]).unwrap();
        assert!(invert_matrix(&m).is_none());
    }

    #[test]
    fn isolation_forest_flags_outliers_higher() {
        // Tight training cluster around the origin.
        let train = Array2::from_shape_fn((100, 2), |(i, _)| (i as Float % 5.0) * 0.01);
        let ood = Array2::from_shape_vec((2, 2), vec![0.02, 0.02, 50.0, 50.0]).unwrap();
        let mut r = rng();
        let scores = isolation_forest_scores(&train, &ood, 100, &mut r);
        assert_eq!(scores.len(), 2);
        // The far outlier should score higher (more anomalous) than the inlier.
        assert!(
            scores[1] > scores[0],
            "outlier score {} should exceed inlier score {}",
            scores[1],
            scores[0]
        );
        // Prove these are not the old fabricated 0.5 constant.
        assert!((scores[1] - 0.5).abs() > 1e-3 || (scores[0] - 0.5).abs() > 1e-3);
    }

    #[test]
    fn k_means_recovers_two_clusters() {
        // Two well-separated clusters.
        let mut data = Vec::new();
        for _ in 0..50 {
            data.push(0.0);
            data.push(0.0);
        }
        for _ in 0..50 {
            data.push(10.0);
            data.push(10.0);
        }
        let x = Array2::from_shape_vec((100, 2), data).unwrap();
        let mut r = rng();
        let centroids = k_means(&x, 2, 50, &mut r);
        assert_eq!(centroids.len(), 2);
        // One centroid near (0,0), one near (10,10).
        let near_zero = centroids
            .iter()
            .any(|c| c[0].abs() < 0.5 && c[1].abs() < 0.5);
        let near_ten = centroids
            .iter()
            .any(|c| (c[0] - 10.0).abs() < 0.5 && (c[1] - 10.0).abs() < 0.5);
        assert!(
            near_zero && near_ten,
            "k-means should recover both clusters"
        );
    }
}

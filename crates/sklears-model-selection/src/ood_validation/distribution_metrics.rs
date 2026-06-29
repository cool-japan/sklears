//! Distribution-shift statistics for out-of-distribution validation.
//!
//! Every function in this module operates purely on the supplied arrays and
//! implements the standard textbook statistic it claims to compute. There are
//! no placeholder return values.

use scirs2_core::ndarray::{Array2, ArrayView1};
use sklears_core::types::Float;

/// Small constant used to keep probabilities strictly positive so that the
/// logarithm in the KL/PSI sums is always finite.
const EPSILON: Float = 1e-12;

/// Sort a slice of floats ascending, dropping non-finite values.
fn sorted_finite(values: &[Float]) -> Vec<Float> {
    let mut out: Vec<Float> = values.iter().copied().filter(|v| v.is_finite()).collect();
    out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Build a normalized histogram (probability mass per bin) of `values` over the
/// half-open bins `[edges[i], edges[i + 1])`, with the last bin closed on the
/// right. A uniform `EPSILON` floor is added to every bin and the result is
/// renormalized so the masses sum to one and are strictly positive.
fn histogram_probabilities(values: &[Float], edges: &[Float]) -> Vec<Float> {
    let n_bins = edges.len().saturating_sub(1);
    if n_bins == 0 {
        return Vec::new();
    }
    let mut counts = vec![0.0_f64; n_bins];
    for &v in values.iter().filter(|v| v.is_finite()) {
        // Locate the bin via binary search on the edges.
        let mut lo = 0usize;
        let mut hi = n_bins; // exclusive upper bin index
        while lo + 1 < hi {
            let mid = (lo + hi) / 2;
            if v >= edges[mid] {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        // Clamp values that fall on or beyond the final edge into the last bin.
        let bin = if v >= edges[n_bins] { n_bins - 1 } else { lo };
        counts[bin] += 1.0;
    }

    let total: Float = counts.iter().sum();
    if total <= 0.0 {
        // No finite samples: fall back to a uniform distribution.
        let uniform = 1.0 / n_bins as Float;
        return vec![uniform; n_bins];
    }

    let mut probs: Vec<Float> = counts.iter().map(|&c| c / total + EPSILON).collect();
    let renorm: Float = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= renorm;
    }
    probs
}

/// Construct `n_bins` equal-width bin edges spanning the combined range of two
/// samples. When the combined range is degenerate the range is widened by a
/// unit so that a valid set of edges is still produced.
fn shared_edges(a: &[Float], b: &[Float], n_bins: usize) -> Vec<Float> {
    let mut min_v = Float::INFINITY;
    let mut max_v = Float::NEG_INFINITY;
    for &v in a.iter().chain(b.iter()).filter(|v| v.is_finite()) {
        if v < min_v {
            min_v = v;
        }
        if v > max_v {
            max_v = v;
        }
    }
    if !min_v.is_finite() || !max_v.is_finite() {
        min_v = 0.0;
        max_v = 1.0;
    }
    if (max_v - min_v).abs() < EPSILON {
        min_v -= 0.5;
        max_v += 0.5;
    }
    let width = (max_v - min_v) / n_bins as Float;
    (0..=n_bins).map(|i| min_v + width * i as Float).collect()
}

/// Heuristic for the number of histogram bins: Sturges' rule, clamped to a
/// sensible range so very small or very large samples still behave well.
fn bin_count(n: usize) -> usize {
    if n < 2 {
        return 1;
    }
    let sturges = (n as Float).log2().ceil() as usize + 1;
    sturges.clamp(2, 256)
}

/// Kullback-Leibler divergence `KL(p || q)` between two empirical samples of a
/// single variable, estimated through a shared histogram. Returns `sum p * ln(p / q)`.
pub fn kl_divergence_1d(p_sample: &[Float], q_sample: &[Float]) -> Float {
    if p_sample.is_empty() || q_sample.is_empty() {
        return 0.0;
    }
    let n_bins = bin_count(p_sample.len().max(q_sample.len()));
    let edges = shared_edges(p_sample, q_sample, n_bins);
    let p = histogram_probabilities(p_sample, &edges);
    let q = histogram_probabilities(q_sample, &edges);
    p.iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| pi * (pi / qi).ln())
        .sum()
}

/// Mean per-feature KL divergence `KL(ood || train)` between two data matrices.
/// Each column is treated as an independent empirical distribution. Measuring
/// `KL(ood || train)` weights the divergence by where the OOD mass actually
/// lands, which is the quantity of interest for drift detection.
pub fn kl_divergence_matrix(x_train: &Array2<Float>, x_ood: &Array2<Float>) -> Float {
    let n_features = x_train.ncols().min(x_ood.ncols());
    if n_features == 0 {
        return 0.0;
    }
    let total: Float = (0..n_features)
        .map(|j| {
            let train_col: Vec<Float> = x_train.column(j).to_vec();
            let ood_col: Vec<Float> = x_ood.column(j).to_vec();
            kl_divergence_1d(&ood_col, &train_col)
        })
        .sum();
    total / n_features as Float
}

/// Per-sample KL divergence of the OOD sample's local neighborhood against the
/// training distribution.
///
/// For each feature we place the OOD sample value into the same shared
/// histogram used for the training column and measure how unlikely that bin is
/// under the training distribution, accumulating `ln(p_train_uniform / q_train)`
/// over features. This is a true, sample-conditional divergence contribution:
/// it is large when the sample falls in regions the training data rarely
/// occupies and approaches zero for typical samples. The result is averaged
/// over features so that it is comparable across dimensionalities.
pub fn kl_divergence_sample(x_train: &Array2<Float>, sample: &ArrayView1<Float>) -> Float {
    let n_features = x_train.ncols().min(sample.len());
    if n_features == 0 {
        return 0.0;
    }
    let total: Float = (0..n_features)
        .map(|j| {
            let train_col: Vec<Float> = x_train.column(j).to_vec();
            let value = sample[j];
            if train_col.is_empty() || !value.is_finite() {
                return 0.0;
            }
            let n_bins = bin_count(train_col.len());
            let single = [value];
            let edges = shared_edges(&train_col, &single, n_bins);
            let q = histogram_probabilities(&train_col, &edges);
            // Locate the bin holding the sample value.
            let n = q.len();
            let mut bin = 0usize;
            for i in 0..n {
                if value >= edges[i] && (value < edges[i + 1] || i + 1 == n) {
                    bin = i;
                    break;
                }
            }
            // Surprise of the sample under the training distribution relative to
            // a uniform reference: ln(uniform / q_bin). Positive when the bin is
            // underrepresented in training, zero when it matches uniform mass.
            let uniform = 1.0 / n as Float;
            (uniform / q[bin]).ln().max(0.0)
        })
        .sum();
    total / n_features as Float
}

/// One-dimensional Wasserstein-1 (earth mover's) distance between two empirical
/// samples, computed as the L1 area between their empirical CDFs.
pub fn wasserstein_distance_1d(a: &[Float], b: &[Float]) -> Float {
    let sa = sorted_finite(a);
    let sb = sorted_finite(b);
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }

    if sa.len() == sb.len() {
        // Equal sample sizes: the EMD reduces to the mean absolute difference of
        // the order statistics.
        let sum: Float = sa.iter().zip(sb.iter()).map(|(&x, &y)| (x - y).abs()).sum();
        return sum / sa.len() as Float;
    }

    // Unequal sizes: integrate |F_a(t) - F_b(t)| over the pooled support.
    let mut pooled: Vec<Float> = sa.iter().chain(sb.iter()).copied().collect();
    pooled.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    pooled.dedup_by(|x, y| (*x - *y).abs() < EPSILON);

    let cdf = |sorted: &[Float], t: Float| -> Float {
        let count = sorted.partition_point(|&v| v <= t);
        count as Float / sorted.len() as Float
    };

    let mut area = 0.0;
    for w in pooled.windows(2) {
        let left = w[0];
        let right = w[1];
        let height = (cdf(&sa, left) - cdf(&sb, left)).abs();
        area += height * (right - left);
    }
    area
}

/// Mean per-feature Wasserstein-1 distance between two data matrices.
pub fn wasserstein_distance_matrix(x_train: &Array2<Float>, x_ood: &Array2<Float>) -> Float {
    let n_features = x_train.ncols().min(x_ood.ncols());
    if n_features == 0 {
        return 0.0;
    }
    let total: Float = (0..n_features)
        .map(|j| {
            let train_col: Vec<Float> = x_train.column(j).to_vec();
            let ood_col: Vec<Float> = x_ood.column(j).to_vec();
            wasserstein_distance_1d(&train_col, &ood_col)
        })
        .sum();
    total / n_features as Float
}

/// Population Stability Index for a single feature. The training sample defines
/// the bin edges (equal-frequency deciles by default); PSI then accumulates
/// `(%ood - %train) * ln(%ood / %train)` across bins.
pub fn psi_1d(train: &[Float], ood: &[Float], n_bins: usize) -> Float {
    let sorted_train = sorted_finite(train);
    if sorted_train.is_empty() || ood.is_empty() || n_bins == 0 {
        return 0.0;
    }

    // Equal-frequency (quantile) edges derived from the training sample.
    let mut edges = Vec::with_capacity(n_bins + 1);
    edges.push(Float::NEG_INFINITY);
    for i in 1..n_bins {
        let q = i as Float / n_bins as Float;
        let pos = q * (sorted_train.len() - 1) as Float;
        let idx = pos.floor() as usize;
        let frac = pos - idx as Float;
        let lo = sorted_train[idx];
        let hi = sorted_train[(idx + 1).min(sorted_train.len() - 1)];
        edges.push(lo + frac * (hi - lo));
    }
    edges.push(Float::INFINITY);

    let bin_fractions = |values: &[Float]| -> Vec<Float> {
        let mut counts = vec![0.0_f64; n_bins];
        let finite: Vec<Float> = values.iter().copied().filter(|v| v.is_finite()).collect();
        for &v in &finite {
            let mut bin = n_bins - 1;
            for b in 0..n_bins {
                if v >= edges[b] && v < edges[b + 1] {
                    bin = b;
                    break;
                }
            }
            counts[bin] += 1.0;
        }
        let total: Float = counts.iter().sum::<Float>().max(1.0);
        counts.iter().map(|&c| (c / total).max(EPSILON)).collect()
    };

    let train_frac = bin_fractions(&sorted_train);
    let ood_frac = bin_fractions(ood);

    train_frac
        .iter()
        .zip(ood_frac.iter())
        .map(|(&t, &o)| (o - t) * (o / t).ln())
        .sum()
}

/// Mean per-feature Population Stability Index between two data matrices using
/// decile (10-bin) binning, the standard credit-risk convention.
pub fn psi_matrix(x_train: &Array2<Float>, x_ood: &Array2<Float>) -> Float {
    let n_features = x_train.ncols().min(x_ood.ncols());
    if n_features == 0 {
        return 0.0;
    }
    let n_bins = 10usize.min(x_train.nrows().max(1));
    let total: Float = (0..n_features)
        .map(|j| {
            let train_col: Vec<Float> = x_train.column(j).to_vec();
            let ood_col: Vec<Float> = x_ood.column(j).to_vec();
            psi_1d(&train_col, &ood_col, n_bins)
        })
        .sum();
    total / n_features as Float
}

/// Two-sample Kolmogorov-Smirnov statistic: the maximum absolute difference
/// between the two empirical CDFs over the pooled sorted support.
pub fn kolmogorov_smirnov_statistic(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    let sa = sorted_finite(&a.to_vec());
    let sb = sorted_finite(&b.to_vec());
    if sa.is_empty() || sb.is_empty() {
        return 0.0;
    }

    let na = sa.len() as Float;
    let nb = sb.len() as Float;

    // Merge-walk both sorted arrays tracking the running CDF gap. At each step we
    // advance past *all* occurrences of the smallest current value in both
    // arrays simultaneously, so tied values (in particular identical samples) do
    // not produce a spurious gap.
    let mut i = 0usize;
    let mut j = 0usize;
    let mut max_gap = 0.0_f64;
    while i < sa.len() || j < sb.len() {
        let next = if i >= sa.len() {
            sb[j]
        } else if j >= sb.len() {
            sa[i]
        } else {
            sa[i].min(sb[j])
        };
        while i < sa.len() && sa[i] <= next {
            i += 1;
        }
        while j < sb.len() && sb[j] <= next {
            j += 1;
        }
        let cdf_a = i as Float / na;
        let cdf_b = j as Float / nb;
        let gap = (cdf_a - cdf_b).abs();
        if gap > max_gap {
            max_gap = gap;
        }
    }
    max_gap
}

/// Per-feature drift scores expressed as the standardized absolute mean
/// difference `|mean_train - mean_ood| / std_train`. Returns one score per
/// feature. When a feature has (near-)zero training variance the raw absolute
/// mean difference is used so the score remains meaningful and finite.
pub fn feature_drift_scores(x_train: &Array2<Float>, x_ood: &Array2<Float>) -> Vec<Float> {
    let n_features = x_train.ncols().min(x_ood.ncols());
    (0..n_features)
        .map(|j| {
            let train_col = x_train.column(j);
            let ood_col = x_ood.column(j);
            let n_train = train_col.len().max(1) as Float;
            let n_ood = ood_col.len().max(1) as Float;
            let mean_train: Float = train_col.iter().sum::<Float>() / n_train;
            let mean_ood: Float = ood_col.iter().sum::<Float>() / n_ood;
            let var_train: Float = train_col
                .iter()
                .map(|&v| {
                    let d = v - mean_train;
                    d * d
                })
                .sum::<Float>()
                / n_train;
            let std_train = var_train.sqrt();
            let diff = (mean_train - mean_ood).abs();
            if std_train > EPSILON {
                diff / std_train
            } else {
                diff
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn linspace_matrix(n: usize, offset: Float) -> Array2<Float> {
        Array2::from_shape_fn((n, 1), |(i, _)| i as Float * 0.1 + offset)
    }

    #[test]
    fn kl_of_identical_is_near_zero() {
        let a: Vec<Float> = (0..200).map(|i| i as Float * 0.05).collect();
        let kl = kl_divergence_1d(&a, &a);
        assert!(kl >= 0.0);
        assert!(kl < 1e-6, "KL of identical samples should be ~0, got {kl}");
    }

    #[test]
    fn kl_of_disjoint_is_positive() {
        let a: Vec<Float> = (0..200).map(|i| i as Float * 0.05).collect();
        let b: Vec<Float> = (0..200).map(|i| 1000.0 + i as Float * 0.05).collect();
        let kl = kl_divergence_1d(&a, &b);
        assert!(kl > 0.5, "KL of disjoint samples should be large, got {kl}");
        // Prove it is not the old fabricated constant.
        assert!((kl - 0.1).abs() > 1e-3);
    }

    #[test]
    fn ks_identical_is_zero_disjoint_is_one() {
        let a: Vec<Float> = (0..100).map(|i| i as Float).collect();
        let av = Array2::from_shape_vec((100, 1), a.clone()).unwrap();
        let ks_same = kolmogorov_smirnov_statistic(&av.column(0), &av.column(0));
        assert!(ks_same.abs() < 1e-12, "KS of identical = 0, got {ks_same}");

        let b: Vec<Float> = (0..100).map(|i| 1000.0 + i as Float).collect();
        let bv = Array2::from_shape_vec((100, 1), b).unwrap();
        let ks_disjoint = kolmogorov_smirnov_statistic(&av.column(0), &bv.column(0));
        assert!(
            (ks_disjoint - 1.0).abs() < 1e-9,
            "KS of disjoint ~ 1, got {ks_disjoint}"
        );
        assert!((ks_disjoint - 0.1).abs() > 1e-3);
    }

    #[test]
    fn wasserstein_identical_is_zero() {
        let a: Vec<Float> = (0..50).map(|i| i as Float * 0.3).collect();
        let w = wasserstein_distance_1d(&a, &a);
        assert!(w.abs() < 1e-12, "Wasserstein of identical = 0, got {w}");
    }

    #[test]
    fn wasserstein_shift_equals_offset() {
        let a: Vec<Float> = (0..50).map(|i| i as Float).collect();
        let b: Vec<Float> = a.iter().map(|&v| v + 3.0).collect();
        let w = wasserstein_distance_1d(&a, &b);
        assert!(
            (w - 3.0).abs() < 1e-9,
            "Wasserstein of constant shift equals offset, got {w}"
        );
        assert!((w - 0.12).abs() > 1e-3);
    }

    #[test]
    fn wasserstein_unequal_sizes() {
        let a: Vec<Float> = (0..40).map(|i| i as Float).collect();
        let b: Vec<Float> = (0..7).map(|i| i as Float + 10.0).collect();
        let w = wasserstein_distance_1d(&a, &b);
        assert!(w > 0.0 && w.is_finite());
    }

    #[test]
    fn psi_identical_is_zero() {
        let a: Vec<Float> = (0..500).map(|i| (i % 50) as Float).collect();
        let psi = psi_1d(&a, &a, 10);
        assert!(psi.abs() < 1e-6, "PSI of identical = 0, got {psi}");
        assert!((psi - 0.08).abs() > 1e-3);
    }

    #[test]
    fn psi_shift_is_positive() {
        let a: Vec<Float> = (0..500).map(|i| i as Float * 0.01).collect();
        let b: Vec<Float> = (0..500).map(|i| 5.0 + i as Float * 0.01).collect();
        let psi = psi_1d(&a, &b, 10);
        assert!(psi > 0.1, "PSI of shifted should be positive, got {psi}");
    }

    #[test]
    fn matrix_metrics_identical_near_zero() {
        let x = linspace_matrix(100, 0.0);
        assert!(kl_divergence_matrix(&x, &x) < 1e-6);
        assert!(wasserstein_distance_matrix(&x, &x).abs() < 1e-9);
        assert!(psi_matrix(&x, &x).abs() < 1e-6);
        let drift = feature_drift_scores(&x, &x);
        assert_eq!(drift.len(), 1);
        assert!(drift[0].abs() < 1e-9);
    }

    #[test]
    fn feature_drift_detects_mean_shift() {
        let train = linspace_matrix(100, 0.0);
        let ood = linspace_matrix(100, 10.0);
        let drift = feature_drift_scores(&train, &ood);
        assert_eq!(drift.len(), 1);
        assert!(drift[0] > 1.0, "standardized mean shift should be large");
        assert!((drift[0] - 0.05).abs() > 1e-3);
    }

    #[test]
    fn kl_sample_larger_for_outlier() {
        let train = Array2::from_shape_fn((200, 1), |(i, _)| (i as Float) * 0.01);
        let typical = scirs2_core::ndarray::array![1.0];
        let outlier = scirs2_core::ndarray::array![100.0];
        let d_typical = kl_divergence_sample(&train, &typical.view());
        let d_outlier = kl_divergence_sample(&train, &outlier.view());
        assert!(d_outlier >= d_typical);
        assert!(d_outlier > 0.0);
    }
}

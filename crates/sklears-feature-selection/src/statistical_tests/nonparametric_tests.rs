//! Non-parametric statistical tests for feature selection

use scirs2_core::error::{CoreError, ErrorContext};
use scirs2_core::ndarray::ArrayView1;
type CoreResult<T> = std::result::Result<T, CoreError>;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Mann–Whitney U test (Wilcoxon rank-sum test) for two independent samples.
///
/// Returns `(U_statistic, p_value)` using the normal approximation (no
/// continuity correction).  The reported U is `min(U1, U2)`.
pub fn mann_whitney_u(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    let n1 = x.len();
    let n2 = y.len();
    if n1 == 0 || n2 == 0 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "mann_whitney_u: both samples must be non-empty",
        )));
    }

    // Build pooled vector with group tags (0 = x, 1 = y)
    let mut combined: Vec<(f64, u8)> = Vec::with_capacity(n1 + n2);
    for &v in x.iter() {
        combined.push((v, 0u8));
    }
    for &v in y.iter() {
        combined.push((v, 1u8));
    }
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    let values: Vec<f64> = combined.iter().map(|(v, _)| *v).collect();
    let ranks = average_ranks(&values);

    let r1: f64 = ranks
        .iter()
        .zip(combined.iter())
        .filter(|(_, (_, g))| *g == 0)
        .map(|(r, _)| r)
        .sum();

    let u1 = r1 - (n1 * (n1 + 1)) as f64 / 2.0;
    let u2 = (n1 * n2) as f64 - u1;
    let u = u1.min(u2);

    // Normal approximation
    let mean_u = (n1 * n2) as f64 / 2.0;
    let var_u = (n1 * n2 * (n1 + n2 + 1)) as f64 / 12.0;
    let z = (u - mean_u) / var_u.sqrt();
    let p = 2.0 * normal_cdf(-z.abs());

    Ok((u, p.clamp(0.0, 1.0)))
}

/// Kruskal–Wallis H test for k independent groups.
///
/// `samples` is a slice of `ArrayView1<f64>`, one entry per group.
/// Returns `(H_statistic, p_value)` where the p-value is from the
/// chi-squared distribution with `k − 1` degrees of freedom.
pub fn kruskal_wallis(samples: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    let k = samples.len();
    if k < 2 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "kruskal_wallis: need at least 2 groups",
        )));
    }
    let n_total: usize = samples.iter().map(|s| s.len()).sum();
    if n_total == 0 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "kruskal_wallis: all samples are empty",
        )));
    }
    for (i, s) in samples.iter().enumerate() {
        if s.is_empty() {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "kruskal_wallis: group {i} is empty"
            ))));
        }
    }

    // Pool values with group index
    let mut combined: Vec<(f64, usize)> = Vec::with_capacity(n_total);
    for (g, s) in samples.iter().enumerate() {
        for &v in s.iter() {
            combined.push((v, g));
        }
    }
    combined.sort_by(|a, b| a.0.total_cmp(&b.0));

    let values: Vec<f64> = combined.iter().map(|(v, _)| *v).collect();
    let ranks = average_ranks(&values);

    // Accumulate rank sums per group
    let mut group_rank_sums = vec![0.0f64; k];
    for (rank, (_, g)) in ranks.iter().zip(combined.iter()) {
        group_rank_sums[*g] += rank;
    }

    let n = n_total as f64;
    // H = (12 / (N*(N+1))) * Σ (Ri² / ni) − 3*(N+1)
    let h: f64 = (12.0 / (n * (n + 1.0)))
        * samples
            .iter()
            .zip(group_rank_sums.iter())
            .map(|(s, &ri)| ri * ri / s.len() as f64)
            .sum::<f64>()
        - 3.0 * (n + 1.0);

    let df = (k - 1) as f64;
    let p = chi2_sf(h.max(0.0), df);
    Ok((h, p))
}

/// Wilcoxon signed-rank test for paired samples.
///
/// Returns `(W_statistic, p_value)` where W = min(W⁺, W⁻) and the
/// p-value uses the normal approximation (no continuity correction).
/// Pairs with zero difference are excluded (standard convention).
pub fn wilcoxon_signed_rank(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    if x.len() != y.len() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "wilcoxon_signed_rank: x and y must have the same length",
        )));
    }
    if x.is_empty() {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "wilcoxon_signed_rank: input is empty",
        )));
    }

    // Compute signed differences, drop zeros
    let diffs: Vec<(f64, f64)> = x
        .iter()
        .zip(y.iter())
        .map(|(&a, &b)| a - b)
        .filter(|&d| d.abs() > 1e-14)
        .map(|d| (d.abs(), d.signum()))
        .collect();

    let n = diffs.len();
    if n == 0 {
        // All differences are zero → cannot reject H0
        return Ok((0.0, 1.0));
    }

    let abs_vals: Vec<f64> = diffs.iter().map(|(a, _)| *a).collect();
    let ranks = average_ranks(&abs_vals);

    let w_plus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, (_, s))| *s > 0.0)
        .map(|(r, _)| r)
        .sum();
    let w_minus: f64 = ranks
        .iter()
        .zip(diffs.iter())
        .filter(|(_, (_, s))| *s < 0.0)
        .map(|(r, _)| r)
        .sum();
    let w = w_plus.min(w_minus);

    let mean_w = n as f64 * (n as f64 + 1.0) / 4.0;
    let var_w = n as f64 * (n as f64 + 1.0) * (2.0 * n as f64 + 1.0) / 24.0;
    let z = (w - mean_w) / var_w.sqrt();
    let p = 2.0 * normal_cdf(-z.abs());
    Ok((w, p.clamp(0.0, 1.0)))
}

/// Friedman test for repeated-measures data.
///
/// `blocks` is a slice of `ArrayView1<f64>` where each element is one
/// complete block (row) of length k (treatments).  Returns
/// `(chi2_statistic, p_value)` with `k − 1` degrees of freedom.
pub fn friedman_test(blocks: &[ArrayView1<f64>]) -> CoreResult<(f64, f64)> {
    let n = blocks.len(); // number of blocks (subjects / rows)
    if n < 2 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "friedman_test: need at least 2 blocks",
        )));
    }
    let k = blocks[0].len(); // number of treatments (columns)
    if k < 2 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "friedman_test: need at least 2 treatments per block",
        )));
    }
    for (i, b) in blocks.iter().enumerate() {
        if b.len() != k {
            return Err(CoreError::InvalidInput(ErrorContext::new(format!(
                "friedman_test: block {i} has length {} but expected {k}",
                b.len()
            ))));
        }
    }

    // Rank within each block, accumulate per-treatment rank sums
    let mut r_j = vec![0.0f64; k];
    for block in blocks {
        let vals: Vec<f64> = block.iter().copied().collect();
        let ranks = average_ranks(&vals);
        for (j, &r) in ranks.iter().enumerate() {
            r_j[j] += r;
        }
    }

    // chi2 = (12 / (n*k*(k+1))) * Σ Rj² − 3*n*(k+1)
    let chi2 = (12.0 / (n as f64 * k as f64 * (k as f64 + 1.0)))
        * r_j.iter().map(|&r| r * r).sum::<f64>()
        - 3.0 * n as f64 * (k as f64 + 1.0);

    let p = chi2_sf(chi2.max(0.0), (k - 1) as f64);
    Ok((chi2, p))
}

/// Two-sample Kolmogorov–Smirnov test.
///
/// Returns `(D_statistic, p_value)` where D is the supremum of the
/// absolute difference between the two empirical CDFs and the p-value
/// uses the Kolmogorov asymptotic distribution.
pub fn kolmogorov_smirnov(x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> CoreResult<(f64, f64)> {
    let n1 = x.len();
    let n2 = y.len();
    if n1 == 0 || n2 == 0 {
        return Err(CoreError::InvalidInput(ErrorContext::new(
            "kolmogorov_smirnov: both samples must be non-empty",
        )));
    }

    let mut xs: Vec<f64> = x.iter().copied().collect();
    let mut ys: Vec<f64> = y.iter().copied().collect();
    xs.sort_by(f64::total_cmp);
    ys.sort_by(f64::total_cmp);

    // Two-pointer ECDF merge: walk through all unique breakpoints
    let mut d = 0.0f64;
    let mut i = 0usize;
    let mut j = 0usize;

    while i < n1 || j < n2 {
        // Determine the current breakpoint value
        let t = match (i < n1, j < n2) {
            (true, true) => {
                if xs[i].total_cmp(&ys[j]).is_le() {
                    xs[i]
                } else {
                    ys[j]
                }
            }
            (true, false) => xs[i],
            (false, true) => ys[j],
            (false, false) => break,
        };

        // Advance both pointers past all values ≤ t
        while i < n1 && xs[i].total_cmp(&t).is_le() {
            i += 1;
        }
        while j < n2 && ys[j].total_cmp(&t).is_le() {
            j += 1;
        }

        let diff = (i as f64 / n1 as f64 - j as f64 / n2 as f64).abs();
        if diff > d {
            d = diff;
        }
    }

    // Effective sample size correction
    let z = d * ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let p = kolmogorov_pvalue(z);
    Ok((d, p))
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Compute average (mid) ranks for `values`.
///
/// The input does **not** need to be pre-sorted; the function uses an
/// auxiliary permutation so the returned ranks align with the original
/// `values` slice.  Ties receive the mean of the tied ranks (1-indexed).
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }

    // Build sorted index permutation
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| a.1.total_cmp(&b.1));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        // Find the extent of the current tie group
        let mut j = i + 1;
        while j < n && (indexed[j].1 - indexed[i].1).abs() < 1e-14 {
            j += 1;
        }
        // 1-indexed average rank for the group [i, j)
        let avg = (i + j + 1) as f64 / 2.0;
        for entry in &indexed[i..j] {
            ranks[entry.0] = avg;
        }
        i = j;
    }
    ranks
}

/// Standard normal CDF Φ(x).
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf_approx(x / std::f64::consts::SQRT_2))
}

/// Error function approximation (Abramowitz & Stegun 7.1.26).
/// Maximum error < 1.5 × 10⁻⁷.
fn erf_approx(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0f64 } else { 1.0f64 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1_f64 * x);
    let poly = t
        * (0.254_829_592_f64
            + t * (-0.284_496_736_f64
                + t * (1.421_413_741_f64 + t * (-1.453_152_027_f64 + t * 1.061_405_429_f64))));
    sign * (1.0 - poly * (-x * x).exp())
}

/// Chi-squared survival function P(χ² > x | df).
fn chi2_sf(x: f64, df: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    upper_regularized_gamma(df / 2.0, x / 2.0)
}

/// Upper regularized incomplete gamma function Q(a, x) = 1 − P(a, x).
fn upper_regularized_gamma(a: f64, x: f64) -> f64 {
    if x < 0.0 {
        return 1.0;
    }
    if x < a + 1.0 {
        1.0 - lower_gamma_series(a, x)
    } else {
        upper_gamma_cf(a, x)
    }
}

/// Lower regularized incomplete gamma via series expansion.
fn lower_gamma_series(a: f64, x: f64) -> f64 {
    let lg = log_gamma(a);
    let mut term = 1.0 / a;
    let mut sum = term;
    for i in 1..300usize {
        term *= x / (a + i as f64);
        sum += term;
        if term.abs() < sum.abs() * 1e-12 {
            break;
        }
    }
    (sum * (-x).exp() * x.powf(a) / lg.exp()).clamp(0.0, 1.0)
}

/// Upper regularized incomplete gamma via Lentz continued fraction.
fn upper_gamma_cf(a: f64, x: f64) -> f64 {
    let lg = log_gamma(a);
    let tiny = 1e-30f64;
    let mut f = tiny;
    let mut c = f;
    let mut d = 1.0 - (a - 1.0) / x;
    if d.abs() < tiny {
        d = tiny;
    }
    d = 1.0 / d;
    f = d;

    for i in 1..300usize {
        let b_i = 2.0 * i as f64 + 1.0 - a + x;
        let a_i = -(i as f64) * (i as f64 - a);

        d = b_i + a_i * d;
        if d.abs() < tiny {
            d = tiny;
        }
        c = b_i + a_i / c;
        if c.abs() < tiny {
            c = tiny;
        }
        d = 1.0 / d;
        let delta = d * c;
        f *= delta;
        if (delta - 1.0).abs() < 1e-12 {
            break;
        }
    }
    ((-x).exp() * x.powf(a) * f / lg.exp()).clamp(0.0, 1.0)
}

/// Natural log of the gamma function via Lanczos approximation (g = 7).
fn log_gamma(z: f64) -> f64 {
    // Lanczos coefficients for g = 7, n = 9
    const G: f64 = 7.0;
    const C: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_403,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_908,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_312_5e-7,
    ];
    let z = z - 1.0;
    let mut x = C[0];
    for (i, &ci) in C[1..].iter().enumerate() {
        x += ci / (z + i as f64 + 1.0);
    }
    let t = z + G + 0.5;
    0.5 * std::f64::consts::TAU.ln() + (z + 0.5) * t.ln() - t + x.ln()
}

/// Asymptotic p-value for the Kolmogorov distribution.
///
/// Uses the alternating series P(D_n > z) ≈ 2 Σ_{k=1}^∞ (−1)^{k+1}
/// exp(−2 k² z²), which converges rapidly for moderate z.
fn kolmogorov_pvalue(z: f64) -> f64 {
    if z < 0.27 {
        return 1.0;
    }
    if z > 3.1 {
        return 0.0;
    }
    let mut p = 0.0f64;
    for k in 1i32..=40 {
        let term = (-2.0 * (k as f64).powi(2) * z * z).exp();
        if k % 2 == 0 {
            p -= 2.0 * term;
        } else {
            p += 2.0 * term;
        }
        if term < 1e-14 {
            break;
        }
    }
    p.clamp(0.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // ── Mann–Whitney U ────────────────────────────────────────────────────────

    #[test]
    fn mann_whitney_clearly_separated_gives_small_p() {
        // x ≪ y → strong evidence against H0
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![20.0, 21.0, 22.0, 23.0, 24.0];
        let (u, p) = mann_whitney_u(&x.view(), &y.view()).unwrap();
        // U should be 0 (no x exceeds any y)
        assert!((u - 0.0).abs() < 1e-9, "Expected U=0, got {u}");
        assert!(
            p < 0.05,
            "Expected p < 0.05 for well-separated groups, got {p}"
        );
    }

    #[test]
    fn mann_whitney_identical_gives_large_p() {
        let x = array![5.0, 6.0, 7.0, 8.0, 9.0];
        let y = array![5.0, 6.0, 7.0, 8.0, 9.0];
        let (_u, p) = mann_whitney_u(&x.view(), &y.view()).unwrap();
        assert!(p > 0.5, "Expected large p for identical groups, got {p}");
    }

    #[test]
    fn mann_whitney_empty_returns_error() {
        let x = array![1.0, 2.0];
        let y: scirs2_core::ndarray::Array1<f64> = scirs2_core::ndarray::Array1::zeros(0);
        assert!(mann_whitney_u(&x.view(), &y.view()).is_err());
    }

    // ── Kruskal–Wallis ────────────────────────────────────────────────────────

    #[test]
    fn kruskal_wallis_identical_groups_gives_large_p() {
        let g1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let g3 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let samples = [g1.view(), g2.view(), g3.view()];
        let (h, p) = kruskal_wallis(&samples).unwrap();
        assert!(h.abs() < 1e-6, "Expected H≈0 for identical groups, got {h}");
        assert!(p > 0.9, "Expected p near 1 for identical groups, got {p}");
    }

    #[test]
    fn kruskal_wallis_separated_groups_gives_small_p() {
        // Larger groups with large separation → strong signal
        let g1 = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let g2 = array![50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0];
        let g3 = array![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0];
        let samples = [g1.view(), g2.view(), g3.view()];
        let (_h, p) = kruskal_wallis(&samples).unwrap();
        assert!(
            p < 0.01,
            "Expected p < 0.01 for well-separated groups, got {p}"
        );
    }

    #[test]
    fn kruskal_wallis_fewer_than_2_groups_returns_error() {
        let g1 = array![1.0, 2.0, 3.0];
        let samples = [g1.view()];
        assert!(kruskal_wallis(&samples).is_err());
    }

    // ── Wilcoxon signed-rank ──────────────────────────────────────────────────

    #[test]
    fn wilcoxon_zero_differences_returns_one() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (w, p) = wilcoxon_signed_rank(&x.view(), &y.view()).unwrap();
        assert!((w - 0.0).abs() < 1e-9, "Expected W=0, got {w}");
        assert!((p - 1.0).abs() < 1e-9, "Expected p=1, got {p}");
    }

    #[test]
    fn wilcoxon_large_consistent_shift_gives_small_p() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0];
        let (_w, p) = wilcoxon_signed_rank(&x.view(), &y.view()).unwrap();
        assert!(
            p < 0.05,
            "Expected p < 0.05 for large consistent shift, got {p}"
        );
    }

    #[test]
    fn wilcoxon_length_mismatch_returns_error() {
        let x = array![1.0, 2.0, 3.0];
        let y = array![1.0, 2.0];
        assert!(wilcoxon_signed_rank(&x.view(), &y.view()).is_err());
    }

    // ── Friedman ──────────────────────────────────────────────────────────────

    #[test]
    fn friedman_identical_treatments_gives_zero_stat() {
        // Each block has the same values across all treatments
        let b1 = array![5.0, 5.0, 5.0];
        let b2 = array![3.0, 3.0, 3.0];
        let b3 = array![8.0, 8.0, 8.0];
        let blocks = [b1.view(), b2.view(), b3.view()];
        let (chi2, p) = friedman_test(&blocks).unwrap();
        assert!(
            chi2.abs() < 1e-6,
            "Expected chi2≈0 for tied treatments, got {chi2}"
        );
        assert!(p > 0.9, "Expected p near 1, got {p}");
    }

    #[test]
    fn friedman_consistent_ordering_gives_max_stat() {
        // Use many blocks with perfect rank agreement → strong signal
        // Treatment A always ranks 1st, B always 2nd, C always 3rd
        let b1 = array![1.0, 2.0, 3.0];
        let b2 = array![4.0, 5.0, 6.0];
        let b3 = array![7.0, 8.0, 9.0];
        let b4 = array![10.0, 11.0, 12.0];
        let b5 = array![13.0, 14.0, 15.0];
        let b6 = array![16.0, 17.0, 18.0];
        let b7 = array![19.0, 20.0, 21.0];
        let b8 = array![22.0, 23.0, 24.0];
        let b9 = array![25.0, 26.0, 27.0];
        let b10 = array![28.0, 29.0, 30.0];
        let blocks = [
            b1.view(),
            b2.view(),
            b3.view(),
            b4.view(),
            b5.view(),
            b6.view(),
            b7.view(),
            b8.view(),
            b9.view(),
            b10.view(),
        ];
        let (_chi2, p) = friedman_test(&blocks).unwrap();
        assert!(
            p < 0.05,
            "Expected p < 0.05 for consistent treatment ordering, got {p}"
        );
    }

    #[test]
    fn friedman_unequal_block_sizes_returns_error() {
        let b1 = array![1.0, 2.0, 3.0];
        let b2 = array![1.0, 2.0];
        let blocks = [b1.view(), b2.view()];
        assert!(friedman_test(&blocks).is_err());
    }

    // ── Kolmogorov–Smirnov ────────────────────────────────────────────────────

    #[test]
    fn ks_same_sample_gives_zero_d() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (d, _p) = kolmogorov_smirnov(&x.view(), &x.view()).unwrap();
        assert!(
            d.abs() < 1e-12,
            "Expected D=0 when both samples are equal, got {d}"
        );
    }

    #[test]
    fn ks_clearly_separated_gives_d_one_and_small_p() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![10.0, 11.0, 12.0, 13.0, 14.0];
        let (d, p) = kolmogorov_smirnov(&x.view(), &y.view()).unwrap();
        assert!(
            (d - 1.0).abs() < 1e-12,
            "Expected D=1 for fully separated samples, got {d}"
        );
        assert!(p < 0.05, "Expected p < 0.05 for separated samples, got {p}");
    }

    #[test]
    fn ks_empty_input_returns_error() {
        let x = array![1.0, 2.0];
        let y: scirs2_core::ndarray::Array1<f64> = scirs2_core::ndarray::Array1::zeros(0);
        assert!(kolmogorov_smirnov(&x.view(), &y.view()).is_err());
    }

    // ── Helper: average_ranks ─────────────────────────────────────────────────

    #[test]
    fn average_ranks_no_ties() {
        let v = vec![3.0, 1.0, 2.0];
        let r = average_ranks(&v);
        // 1→rank1, 2→rank2, 3→rank3 (1-indexed)
        assert!((r[0] - 3.0).abs() < 1e-12);
        assert!((r[1] - 1.0).abs() < 1e-12);
        assert!((r[2] - 2.0).abs() < 1e-12);
    }

    #[test]
    fn average_ranks_with_ties() {
        // [1, 1, 3] → tied pair gets rank (1+2)/2 = 1.5; 3 → rank 3
        let v = vec![1.0, 1.0, 3.0];
        let r = average_ranks(&v);
        assert!((r[0] - 1.5).abs() < 1e-12, "Expected 1.5, got {}", r[0]);
        assert!((r[1] - 1.5).abs() < 1e-12, "Expected 1.5, got {}", r[1]);
        assert!((r[2] - 3.0).abs() < 1e-12, "Expected 3.0, got {}", r[2]);
    }
}

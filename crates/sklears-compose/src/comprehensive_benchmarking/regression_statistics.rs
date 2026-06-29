//! Classical regression-detection statistics
//!
//! All routines operate on plain `&[f64]` samples so that they can be
//! wired against `BenchmarkResult.metrics` without taking on extra
//! dependencies. They are intentionally allocation-light and never call
//! `.unwrap()`. Fallbacks use a neutral, well-defined value (typically
//! 0.0) when computations are otherwise undefined (e.g., variance of a
//! single sample).

use std::time::SystemTime;

// ------------------------------------------------------------------------------------------------
// Basic descriptive statistics
// ------------------------------------------------------------------------------------------------

/// Sample mean of `xs`. Returns 0.0 when the slice is empty.
pub fn series_mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let n = xs.len() as f64;
    xs.iter().sum::<f64>() / n
}

/// Bessel-corrected sample variance. Returns 0.0 when n < 2.
pub fn series_var(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 2 {
        return 0.0;
    }
    let mean = series_mean(xs);
    let sum_sq: f64 = xs.iter().map(|x| (x - mean).powi(2)).sum();
    sum_sq / (n as f64 - 1.0)
}

/// Bessel-corrected sample standard deviation.
pub fn series_std(xs: &[f64]) -> f64 {
    series_var(xs).sqrt()
}

// ------------------------------------------------------------------------------------------------
// Welch's two-sample t-test
// ------------------------------------------------------------------------------------------------

/// Outcome of Welch's two-sample t-test.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WelchTTestResult {
    pub mean_a: f64,
    pub mean_b: f64,
    pub mean_diff: f64,
    pub t_statistic: f64,
    pub df: f64,
    pub p_value: f64,
    pub ci95_low: f64,
    pub ci95_high: f64,
}

/// Welch's two-sample t-test (no equal-variance assumption).
///
/// Returns `None` when either sample has fewer than two observations or
/// when the pooled standard error is zero, since the test is undefined
/// in those cases.
pub fn welch_t_test(a: &[f64], b: &[f64]) -> Option<WelchTTestResult> {
    let n_a = a.len();
    let n_b = b.len();
    if n_a < 2 || n_b < 2 {
        return None;
    }
    let mean_a = series_mean(a);
    let mean_b = series_mean(b);
    let var_a = series_var(a);
    let var_b = series_var(b);
    let se_sq = var_a / (n_a as f64) + var_b / (n_b as f64);
    if se_sq <= 0.0 {
        return None;
    }
    let se = se_sq.sqrt();
    let t = (mean_a - mean_b) / se;
    let df = welch_satterthwaite_df(var_a, n_a, var_b, n_b);
    let p = student_t_two_sided_p(t, df);
    let t_crit = student_t_critical_two_sided(0.05, df);
    let halfwidth = t_crit * se;
    Some(WelchTTestResult {
        mean_a,
        mean_b,
        mean_diff: mean_a - mean_b,
        t_statistic: t,
        df,
        p_value: p,
        ci95_low: (mean_a - mean_b) - halfwidth,
        ci95_high: (mean_a - mean_b) + halfwidth,
    })
}

/// Welch–Satterthwaite degrees-of-freedom approximation.
pub fn welch_satterthwaite_df(var_a: f64, n_a: usize, var_b: f64, n_b: usize) -> f64 {
    if n_a < 2 || n_b < 2 {
        return 0.0;
    }
    let na = n_a as f64;
    let nb = n_b as f64;
    let s_a = var_a / na;
    let s_b = var_b / nb;
    let num = (s_a + s_b).powi(2);
    let den = s_a.powi(2) / (na - 1.0) + s_b.powi(2) / (nb - 1.0);
    if den <= 0.0 {
        return 0.0;
    }
    num / den
}

// ------------------------------------------------------------------------------------------------
// Mann–Whitney U test
// ------------------------------------------------------------------------------------------------

/// Mann–Whitney U test (rank-sum, non-parametric).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MannWhitneyResult {
    pub u_a: f64,
    pub u_b: f64,
    pub z_statistic: f64,
    pub p_value: f64,
}

/// Compute the Mann–Whitney U statistic together with its normal
/// approximation z and two-sided p-value, with tie correction.
pub fn mann_whitney_u(a: &[f64], b: &[f64]) -> Option<MannWhitneyResult> {
    let n_a = a.len();
    let n_b = b.len();
    if n_a == 0 || n_b == 0 {
        return None;
    }
    // Build (value, group) pairs and rank them with average ranks for
    // ties. group: 0 = a, 1 = b.
    let mut tagged: Vec<(f64, u8)> = Vec::with_capacity(n_a + n_b);
    for &x in a {
        tagged.push((x, 0));
    }
    for &x in b {
        tagged.push((x, 1));
    }
    tagged.sort_by(|p, q| p.0.partial_cmp(&q.0).unwrap_or(std::cmp::Ordering::Equal));
    let n = tagged.len();
    let mut ranks = vec![0.0_f64; n];
    let mut i = 0;
    let mut tie_correction = 0.0_f64;
    while i < n {
        let mut j = i + 1;
        while j < n && (tagged[j].0 - tagged[i].0).abs() < 1e-12 {
            j += 1;
        }
        let avg_rank = ((i + 1) as f64 + j as f64) / 2.0;
        for rank in ranks.iter_mut().take(j).skip(i) {
            *rank = avg_rank;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_correction += t.powi(3) - t;
        }
        i = j;
    }
    let r_a: f64 = tagged
        .iter()
        .zip(ranks.iter())
        .filter(|((_, g), _)| *g == 0)
        .map(|(_, r)| *r)
        .sum();
    let n_a_f = n_a as f64;
    let n_b_f = n_b as f64;
    let u_a = r_a - n_a_f * (n_a_f + 1.0) / 2.0;
    let u_b = n_a_f * n_b_f - u_a;
    let mean_u = n_a_f * n_b_f / 2.0;
    let n_total = n_a_f + n_b_f;
    let var_u =
        (n_a_f * n_b_f / 12.0) * ((n_total + 1.0) - tie_correction / (n_total * (n_total - 1.0)));
    let z = if var_u > 0.0 {
        (u_a - mean_u) / var_u.sqrt()
    } else {
        0.0
    };
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
    Some(MannWhitneyResult {
        u_a,
        u_b,
        z_statistic: z,
        p_value: p_value.clamp(0.0, 1.0),
    })
}

// ------------------------------------------------------------------------------------------------
// CUSUM change-point detection
// ------------------------------------------------------------------------------------------------

/// CUSUM change-point detection result.
#[derive(Debug, Clone)]
pub struct CusumResult {
    /// Indices in `xs` at which an alarm fired.
    pub alarms: Vec<usize>,
    /// Reference mean used as μ₀.
    pub reference_mean: f64,
    /// Reference standard deviation used to derive `k` and `h`.
    pub reference_sigma: f64,
    /// Slack `k` (typically 0.5σ).
    pub slack: f64,
    /// Alarm threshold `h` (typically 5σ).
    pub threshold: f64,
}

/// Two-sided CUSUM (Page 1954). `k` is the slack (default 0.5σ if `None`)
/// and `h` is the alarm threshold (default 5σ if `None`).
///
/// The reference mean μ₀ and σ are estimated from the leading
/// `min(len/2, 8)` samples (clamped to ≥ 2). Estimating from the whole
/// series would inflate σ when a step change is present and prevent the
/// detector from firing — the textbook recipe is to learn the reference
/// from a stable warmup window.
pub fn cusum_changepoint(xs: &[f64], k: Option<f64>, h: Option<f64>) -> Option<CusumResult> {
    if xs.len() < 2 {
        return None;
    }
    let warmup = (xs.len() / 2).min(8).max(2);
    let warm = &xs[..warmup];
    let mu0 = series_mean(warm);
    let sigma_estimate = series_std(warm);
    // When the warmup window is constant (zero variance) we still need a
    // non-degenerate σ. Fall back to a small fraction of |μ₀| or 1.0.
    let sigma = if sigma_estimate > 1e-12 {
        sigma_estimate
    } else {
        (mu0.abs() * 0.05).max(1e-3)
    };
    let k_val = k.unwrap_or(0.5 * sigma);
    let h_val = h.unwrap_or(5.0 * sigma);
    let mut s_pos = 0.0_f64;
    let mut s_neg = 0.0_f64;
    let mut alarms = Vec::new();
    for (i, &x) in xs.iter().enumerate() {
        s_pos = (s_pos + x - mu0 - k_val).max(0.0);
        s_neg = (s_neg - (x - mu0) - k_val).max(0.0);
        if s_pos > h_val || s_neg > h_val {
            alarms.push(i);
            s_pos = 0.0;
            s_neg = 0.0;
        }
    }
    Some(CusumResult {
        alarms,
        reference_mean: mu0,
        reference_sigma: sigma,
        slack: k_val,
        threshold: h_val,
    })
}

// ------------------------------------------------------------------------------------------------
// EWMA drift detection
// ------------------------------------------------------------------------------------------------

/// EWMA drift detection result.
#[derive(Debug, Clone)]
pub struct EwmaResult {
    /// Smoothed series μ_t = α·x_t + (1−α)·μ_{t−1}.
    pub smoothed: Vec<f64>,
    /// Indices where |x_t − μ_t| exceeds 3σ.
    pub alarms: Vec<usize>,
}

/// Exponentially weighted moving average drift detector.
/// `alpha` ∈ (0, 1] is the smoothing factor (default 0.3).
pub fn ewma_drift(xs: &[f64], alpha: Option<f64>) -> EwmaResult {
    let alpha = alpha.unwrap_or(0.3).clamp(1e-6, 1.0);
    if xs.is_empty() {
        return EwmaResult {
            smoothed: Vec::new(),
            alarms: Vec::new(),
        };
    }
    let mut smoothed = Vec::with_capacity(xs.len());
    let mut mu = xs[0];
    smoothed.push(mu);
    for &x in &xs[1..] {
        mu = alpha * x + (1.0 - alpha) * mu;
        smoothed.push(mu);
    }
    let sigma = series_std(xs).max(1e-12);
    let mut alarms = Vec::new();
    for (i, (&x, &m)) in xs.iter().zip(smoothed.iter()).enumerate() {
        if (x - m).abs() > 3.0 * sigma {
            alarms.push(i);
        }
    }
    EwmaResult { smoothed, alarms }
}

// ------------------------------------------------------------------------------------------------
// Effect size and linear regression
// ------------------------------------------------------------------------------------------------

/// Cohen's d effect size with the pooled (equal-weighted) standard
/// deviation. Returns 0.0 when the pooled SD is degenerate.
pub fn cohens_d(a: &[f64], b: &[f64]) -> f64 {
    let n_a = a.len();
    let n_b = b.len();
    if n_a < 2 || n_b < 2 {
        return 0.0;
    }
    let mean_a = series_mean(a);
    let mean_b = series_mean(b);
    let var_a = series_var(a);
    let var_b = series_var(b);
    let pooled =
        ((n_a as f64 - 1.0) * var_a + (n_b as f64 - 1.0) * var_b) / (n_a as f64 + n_b as f64 - 2.0);
    if pooled <= 0.0 {
        return 0.0;
    }
    (mean_a - mean_b) / pooled.sqrt()
}

/// Ordinary least squares fit `y ≈ a + b·x`. Returns `(slope, intercept,
/// r_squared)`. R² is clamped into [0, 1]; degenerate cases yield zeros.
pub fn linear_regression(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return (0.0, ys.first().copied().unwrap_or(0.0), 0.0);
    }
    let nf = n as f64;
    let mean_x = xs.iter().take(n).sum::<f64>() / nf;
    let mean_y = ys.iter().take(n).sum::<f64>() / nf;
    let mut sxx = 0.0_f64;
    let mut sxy = 0.0_f64;
    let mut syy = 0.0_f64;
    for i in 0..n {
        let dx = xs[i] - mean_x;
        let dy = ys[i] - mean_y;
        sxx += dx * dx;
        sxy += dx * dy;
        syy += dy * dy;
    }
    if sxx <= 0.0 {
        return (0.0, mean_y, 0.0);
    }
    let slope = sxy / sxx;
    let intercept = mean_y - slope * mean_x;
    let r2 = if syy > 0.0 {
        ((sxy * sxy) / (sxx * syy)).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (slope, intercept, r2)
}

// ------------------------------------------------------------------------------------------------
// Distribution-function approximations
// ------------------------------------------------------------------------------------------------

/// Two-sided p-value of Student's t with `df` degrees of freedom.
///
/// Uses the standard regularized incomplete beta relation
/// `P(|T| ≥ |t|) = I_{df/(df+t²)}(df/2, 1/2)` (Press et al., NR §6.4).
pub fn student_t_two_sided_p(t: f64, df: f64) -> f64 {
    if df <= 0.0 || !t.is_finite() {
        return 1.0;
    }
    let x = df / (df + t * t);
    let p = regularized_incomplete_beta(x, df / 2.0, 0.5);
    p.clamp(0.0, 1.0)
}

/// Approximate two-sided Student-t critical value via Cornish–Fisher
/// adjustment of the normal critical value.
pub fn student_t_critical_two_sided(alpha: f64, df: f64) -> f64 {
    if df <= 0.0 {
        return f64::INFINITY;
    }
    let z = inverse_standard_normal_cdf(1.0 - alpha / 2.0);
    if df > 200.0 {
        return z;
    }
    // Fisher–Cornish expansion (Abramowitz & Stegun 26.7.5).
    let g1 = (z.powi(3) + z) / 4.0;
    let g2 = (5.0 * z.powi(5) + 16.0 * z.powi(3) + 3.0 * z) / 96.0;
    z + g1 / df + g2 / (df * df)
}

/// Standard normal CDF Φ(x) using the rational approximation
/// `0.5 * (1 + erf(x / √2))`.
pub fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Inverse standard normal CDF using Beasley–Springer–Moro (1995).
pub fn inverse_standard_normal_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-12, 1.0 - 1e-12);
    let a = [
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239,
    ];
    let b = [
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    ];
    let c = [
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838,
        -2.549732539343734,
        4.374664141464968,
        2.938163982698783,
    ];
    let d = [
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996,
        3.754408661907416,
    ];
    let p_low = 0.02425;
    let p_high = 1.0 - p_low;
    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Error function via Chebyshev approximation (Abramowitz & Stegun 7.1.26).
/// Maximum error ≈ 1.5e-7 over real x.
pub fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let abs_x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * abs_x);
    let exp_term = (-abs_x * abs_x).exp();
    let poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
        + 0.254829592)
        * t;
    sign * (1.0 - poly * exp_term).clamp(-1.0, 1.0)
}

/// Regularized incomplete beta `I_x(a, b)` via the continued-fraction
/// expansion in NR §6.4. Stable for the parameter range used by Student's t.
pub fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    let bt = (ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b) + a * x.ln() + b * (1.0 - x).ln()).exp();
    if x < (a + 1.0) / (a + b + 2.0) {
        bt * beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - bt * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

/// Continued fraction used by `regularized_incomplete_beta`.
fn beta_continued_fraction(x: f64, a: f64, b: f64) -> f64 {
    const FPMIN: f64 = 1e-30;
    const EPS: f64 = 1e-12;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < FPMIN {
        d = FPMIN;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..200 {
        let m_f = m as f64;
        let m2 = 2.0 * m_f;
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        h *= d * c;
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < FPMIN {
            d = FPMIN;
        }
        c = 1.0 + aa / c;
        if c.abs() < FPMIN {
            c = FPMIN;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;
        if (del - 1.0).abs() < EPS {
            break;
        }
    }
    h
}

/// Lanczos approximation for ln Γ(x), accurate to ~1e-12 for x > 0.
pub fn ln_gamma(x: f64) -> f64 {
    let coeffs = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5,
    ];
    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000000000190015;
    for &c in &coeffs {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.5066282746310005 * ser / x).ln()
}

// ------------------------------------------------------------------------------------------------
// Common types shared with the regression detector
// ------------------------------------------------------------------------------------------------

/// A change-point in a benchmark time series (the `RegressionDetector`-
/// facing form of `CusumResult`).
#[derive(Debug, Clone)]
pub struct Changepoint {
    pub benchmark_id: String,
    pub detection_time: SystemTime,
    pub pre_change_mean: f64,
    pub post_change_mean: f64,
    pub magnitude_change: f64,
    pub confidence: f64,
}

// ================================================================================================
// TESTS
// ================================================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_series_mean_var_std_constant_zero_var() {
        let xs = [3.0, 3.0, 3.0];
        assert!((series_mean(&xs) - 3.0).abs() < 1e-12);
        assert!(series_var(&xs).abs() < 1e-12);
        assert!(series_std(&xs).abs() < 1e-12);
    }

    #[test]
    fn test_welch_t_test_rejects_at_p_05_for_clear_diff() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 11.0, 12.0];
        let result = welch_t_test(&a, &b).expect("welch defined for n>=2");
        // sample means 2 vs 11 → t very negative, p ≪ 0.05.
        assert!(result.t_statistic < 0.0);
        assert!(result.p_value < 0.05);
        // 95% CI of mean diff (a − b) does NOT cover 0.
        assert!(result.ci95_high < 0.0);
    }

    #[test]
    fn test_welch_t_test_no_reject_for_identical_samples() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let result = welch_t_test(&a, &b).expect("welch defined for n>=2");
        assert!(result.t_statistic.abs() < 1e-9);
        assert!(result.p_value > 0.99);
    }

    #[test]
    fn test_welch_t_test_returns_none_for_too_few_samples() {
        assert!(welch_t_test(&[1.0], &[2.0, 3.0]).is_none());
    }

    #[test]
    fn test_welch_satterthwaite_df_basic() {
        let df = welch_satterthwaite_df(1.0, 10, 1.0, 10);
        // equal variance and equal n → df = 2(n − 1) = 18.
        assert!((df - 18.0).abs() < 1e-9);
    }

    #[test]
    fn test_mann_whitney_no_diff_for_identical_samples() {
        // When a == b (4 samples each, no gap), U should equal the
        // null-expectation n_a * n_b / 2 = 8.
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let result = mann_whitney_u(&a, &b).expect("U defined");
        assert!((result.u_a - 8.0).abs() < 1e-9);
        assert!(result.z_statistic.abs() < 1e-9);
        assert!(result.p_value > 0.99);
    }

    #[test]
    fn test_mann_whitney_extreme_separation() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [10.0, 11.0, 12.0, 13.0, 14.0];
        let result = mann_whitney_u(&a, &b).expect("U defined");
        // All A-ranks are 1..5, sum = 15, U_a = 15 - 5*6/2 = 0.
        assert!(result.u_a.abs() < 1e-9);
        assert!(result.p_value < 0.05);
    }

    #[test]
    fn test_cusum_detects_step_change() {
        let xs = [1.0, 1.0, 1.0, 1.0, 5.0, 5.0, 5.0, 5.0];
        let result = cusum_changepoint(&xs, None, None).expect("cusum defined");
        assert!(!result.alarms.is_empty(), "should fire on step change");
        // Earliest alarm should land in the post-change region.
        let first = result.alarms[0];
        assert!(first >= 4, "alarm at index {first} should be at/after step");
    }

    #[test]
    fn test_cusum_no_alarm_on_stationary_series() {
        let xs = [1.0, 1.01, 0.99, 1.0, 1.02, 0.98, 1.0, 1.01];
        let result = cusum_changepoint(&xs, None, None).expect("cusum defined");
        assert!(
            result.alarms.is_empty(),
            "stationary series should not alarm: {:?}",
            result.alarms
        );
    }

    #[test]
    fn test_ewma_drifts_toward_new_mean() {
        // x stays at 0 for 6 steps, then jumps to 10 for 14.
        let mut xs = vec![0.0_f64; 6];
        xs.extend(std::iter::repeat_n(10.0_f64, 14));
        let result = ewma_drift(&xs, Some(0.3));
        // Last smoothed value should have drifted significantly toward 10.
        let last = *result.smoothed.last().expect("smoothed non-empty");
        assert!(last > 5.0, "expected EWMA to track up, got {last}");
        assert!(last < 10.0, "expected EWMA to lag the input, got {last}");
        // First smoothed value is x[0].
        assert!((result.smoothed[0] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_cohens_d_zero_for_identical_distributions() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(cohens_d(&a, &b).abs() < 1e-12);
    }

    #[test]
    fn test_cohens_d_large_for_separated_means() {
        let a = [0.0, 1.0, 2.0];
        let b = [10.0, 11.0, 12.0];
        let d = cohens_d(&a, &b);
        // (0+1+2)/3 − (10+11+12)/3 = −10; pooled sd = 1.
        assert!(d < -5.0, "expected large negative d, got {d}");
    }

    #[test]
    fn test_linear_regression_perfect_fit() {
        let xs = [0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = [1.0, 3.0, 5.0, 7.0, 9.0];
        let (slope, intercept, r2) = linear_regression(&xs, &ys);
        assert!((slope - 2.0).abs() < 1e-9);
        assert!((intercept - 1.0).abs() < 1e-9);
        assert!((r2 - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_standard_normal_cdf_known_values() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((standard_normal_cdf(1.0) - 0.8413447).abs() < 1e-4);
        assert!((standard_normal_cdf(-1.0) - 0.1586553).abs() < 1e-4);
    }

    #[test]
    fn test_inverse_standard_normal_cdf_round_trip() {
        for &p in &[0.025_f64, 0.1, 0.5, 0.9, 0.975] {
            let z = inverse_standard_normal_cdf(p);
            let back = standard_normal_cdf(z);
            assert!((back - p).abs() < 1e-3, "p={p} round-trip back={back}");
        }
    }

    #[test]
    fn test_student_t_two_sided_p_at_zero_is_one() {
        assert!((student_t_two_sided_p(0.0, 10.0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_student_t_critical_two_sided_large_df_matches_normal() {
        let crit_inf = student_t_critical_two_sided(0.05, 1.0e6);
        // z_{0.975} ≈ 1.959964
        assert!((crit_inf - 1.959964).abs() < 1e-3);
    }
}

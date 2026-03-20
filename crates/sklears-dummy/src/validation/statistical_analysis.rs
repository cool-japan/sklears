use scirs2_core::ndarray::Array1;
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::Float;

/// Perform Kolmogorov-Smirnov test for distribution comparison
pub fn kolmogorov_smirnov_test(
    sample1: &Array1<Float>,
    sample2: &Array1<Float>,
) -> Result<KSTestResult> {
    if sample1.is_empty() || sample2.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Samples cannot be empty for KS test".to_string(),
        ));
    }

    let mut sorted1 = sample1.to_vec();
    let mut sorted2 = sample2.to_vec();
    sorted1.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted2.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = sorted1.len() as Float;
    let n2 = sorted2.len() as Float;

    let mut max_diff: Float = 0.0;
    let mut i1 = 0;
    let mut i2 = 0;

    while i1 < sorted1.len() && i2 < sorted2.len() {
        let cdf1 = i1 as Float / n1;
        let cdf2 = i2 as Float / n2;
        max_diff = max_diff.max((cdf1 - cdf2).abs());

        if sorted1[i1] <= sorted2[i2] {
            i1 += 1;
        } else {
            i2 += 1;
        }
    }

    let ks_statistic = max_diff;
    let critical_value = 1.36 * ((n1 + n2) / (n1 * n2)).sqrt();
    let p_value = if ks_statistic > critical_value {
        0.01
    } else {
        0.10
    };

    Ok(KSTestResult {
        statistic: ks_statistic,
        p_value,
        critical_value,
        is_significant: ks_statistic > critical_value,
    })
}

/// Perform Shapiro-Wilk test for normality
pub fn shapiro_wilk_test(sample: &Array1<Float>) -> Result<ShapiroWilkResult> {
    let n = sample.len();
    if !(3..=5000).contains(&n) {
        return Err(SklearsError::InvalidInput(
            "Sample size must be between 3 and 5000 for Shapiro-Wilk test".to_string(),
        ));
    }

    let mut sorted_sample = sample.to_vec();
    sorted_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = sorted_sample.iter().sum::<Float>() / n as Float;
    let ss = sorted_sample
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<Float>();

    // Simplified W statistic calculation
    let mut numerator = 0.0;
    for i in 0..n / 2 {
        let coeff = shapiro_wilk_coefficient(n, i);
        numerator += coeff * (sorted_sample[n - 1 - i] - sorted_sample[i]);
    }

    let w_statistic = if ss > 0.0 {
        (numerator.powi(2) / ss).clamp(0.0, 1.0)
    } else {
        1.0 // Perfect normality if no variance
    };
    let p_value = if w_statistic < 0.9 { 0.01 } else { 0.10 };

    Ok(ShapiroWilkResult {
        statistic: w_statistic,
        p_value,
        is_normal: w_statistic > 0.95,
    })
}

/// Get Shapiro-Wilk coefficient (simplified approximation)
fn shapiro_wilk_coefficient(n: usize, i: usize) -> Float {
    // Simplified coefficient calculation
    let m = n as Float;
    let rank = (i + 1) as Float;
    let c = (2.0 * rank - 1.0) / (2.0 * m);
    inverse_normal_cdf(c)
}

/// Approximate inverse normal CDF
fn inverse_normal_cdf(p: Float) -> Float {
    if p <= 0.0 {
        return Float::NEG_INFINITY;
    }
    if p >= 1.0 {
        return Float::INFINITY;
    }

    // Rational approximation (Beasley-Springer-Moro algorithm)
    let a = [
        0.0,
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383_577_518_672_69e2,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];

    let b = [
        0.0,
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];

    let c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];

    let d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        // Rational approximation for lower region
        let q = (-2.0 * p.ln()).sqrt();
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
            + c[5] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p <= p_high {
        // Rational approximation for central region
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    } else {
        // Rational approximation for upper region
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q
            + c[5] / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    }
}

/// Perform Anderson-Darling test for normality
pub fn anderson_darling_test(sample: &Array1<Float>) -> Result<AndersonDarlingResult> {
    let n = sample.len();
    if n < 3 {
        return Err(SklearsError::InvalidInput(
            "Sample size must be at least 3 for Anderson-Darling test".to_string(),
        ));
    }

    let mut sorted_sample = sample.to_vec();
    sorted_sample.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mean = sorted_sample.iter().sum::<Float>() / n as Float;
    let variance = sorted_sample
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<Float>()
        / (n - 1) as Float;
    let std_dev = variance.sqrt();

    let mut ad_statistic = 0.0;
    for i in 0..n {
        let zi = (sorted_sample[i] - mean) / std_dev;
        let phi_zi = normal_cdf(zi);
        let phi_zi_rev = normal_cdf(-zi);

        if phi_zi > 0.0 && phi_zi < 1.0 && phi_zi_rev > 0.0 && phi_zi_rev < 1.0 {
            ad_statistic += (2 * i + 1) as Float * (phi_zi.ln() + phi_zi_rev.ln());
        }
    }

    ad_statistic = -(n as Float) - ad_statistic / n as Float;
    ad_statistic *= 1.0 + 0.75 / n as Float + 2.25 / (n as Float).powi(2);

    let critical_value = 0.752; // 5% significance level
    let p_value = if ad_statistic > critical_value {
        0.01
    } else {
        0.10
    };

    Ok(AndersonDarlingResult {
        statistic: ad_statistic,
        p_value,
        critical_value,
        is_normal: ad_statistic <= critical_value,
    })
}

/// Normal CDF approximation
fn normal_cdf(x: Float) -> Float {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Error function approximation
fn erf(x: Float) -> Float {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Result of Kolmogorov-Smirnov test
#[derive(Debug, Clone)]
pub struct KSTestResult {
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// critical_value
    pub critical_value: Float,
    /// is_significant
    pub is_significant: bool,
}

/// Result of Shapiro-Wilk test
#[derive(Debug, Clone)]
pub struct ShapiroWilkResult {
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// is_normal
    pub is_normal: bool,
}

/// Result of Anderson-Darling test
#[derive(Debug, Clone)]
pub struct AndersonDarlingResult {
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// critical_value
    pub critical_value: Float,
    /// is_normal
    pub is_normal: bool,
}

/// Comprehensive statistical analysis of validation results
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// normality_tests
    pub normality_tests: Vec<NormalityTest>,
    /// distribution_tests
    pub distribution_tests: Vec<DistributionTest>,
    /// summary_statistics
    pub summary_statistics: SummaryStatistics,
}

/// Normality test result
#[derive(Debug, Clone)]
pub struct NormalityTest {
    /// test_name
    pub test_name: String,
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// is_normal
    pub is_normal: bool,
}

/// Distribution comparison test
#[derive(Debug, Clone)]
pub struct DistributionTest {
    /// test_name
    pub test_name: String,
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// is_significant
    pub is_significant: bool,
}

/// Summary statistics for validation results
#[derive(Debug, Clone)]
pub struct SummaryStatistics {
    /// mean
    pub mean: Float,
    /// median
    pub median: Float,
    /// std
    pub std: Float,
    /// variance
    pub variance: Float,
    /// skewness
    pub skewness: Float,
    /// kurtosis
    pub kurtosis: Float,
    /// min
    pub min: Float,
    /// max
    pub max: Float,
    /// range
    pub range: Float,
    /// q25
    pub q25: Float,
    /// q75
    pub q75: Float,
    /// iqr
    pub iqr: Float,
}

impl SummaryStatistics {
    pub fn from_sample(sample: &Array1<Float>) -> Self {
        if sample.is_empty() {
            return Self::default();
        }

        let mut sorted = sample.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sample.len() as Float;
        let mean = sample.iter().sum::<Float>() / n;
        let variance = sample.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n;
        let std = variance.sqrt();

        let median = if sorted.len() % 2 == 0 {
            let mid = sorted.len() / 2;
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let min = sorted[0];
        let max = sorted[sorted.len() - 1];
        let range = max - min;

        let q25_idx = (sorted.len() as Float * 0.25) as usize;
        let q75_idx = (sorted.len() as Float * 0.75) as usize;
        let q25 = sorted[q25_idx.min(sorted.len() - 1)];
        let q75 = sorted[q75_idx.min(sorted.len() - 1)];
        let iqr = q75 - q25;

        let m3 = sample.iter().map(|&x| (x - mean).powi(3)).sum::<Float>() / n;
        let m4 = sample.iter().map(|&x| (x - mean).powi(4)).sum::<Float>() / n;
        let skewness = if std > 0.0 { m3 / std.powi(3) } else { 0.0 };
        let kurtosis = if std > 0.0 {
            m4 / std.powi(4) - 3.0
        } else {
            0.0
        };

        Self {
            mean,
            median,
            std,
            variance,
            skewness,
            kurtosis,
            min,
            max,
            range,
            q25,
            q75,
            iqr,
        }
    }
}

impl Default for SummaryStatistics {
    fn default() -> Self {
        Self {
            mean: 0.0,
            median: 0.0,
            std: 0.0,
            variance: 0.0,
            skewness: 0.0,
            kurtosis: 0.0,
            min: 0.0,
            max: 0.0,
            range: 0.0,
            q25: 0.0,
            q75: 0.0,
            iqr: 0.0,
        }
    }
}

//! Statistical Utilities
//!
//! This module provides comprehensive statistical analysis utilities for machine learning,
//! including statistical tests, confidence intervals, correlation analysis, hypothesis testing,
//! and distribution fitting utilities.

use crate::{math_utils::SpecialFunctions, UtilsError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

/// Statistical test results
#[derive(Debug, Clone)]
pub struct TestResult {
    pub statistic: f64,
    pub p_value: f64,
    pub critical_value: Option<f64>,
    pub test_name: String,
    pub significant: bool,
}

impl TestResult {
    pub fn new(statistic: f64, p_value: f64, test_name: String, alpha: f64) -> Self {
        Self {
            statistic,
            p_value,
            critical_value: None,
            test_name,
            significant: p_value < alpha,
        }
    }

    pub fn with_critical_value(mut self, critical_value: f64) -> Self {
        self.critical_value = Some(critical_value);
        self
    }
}

/// Confidence interval
#[derive(Debug, Clone)]
pub struct ConfidenceInterval {
    pub lower: f64,
    pub upper: f64,
    pub confidence_level: f64,
    pub parameter: String,
}

impl ConfidenceInterval {
    pub fn new(lower: f64, upper: f64, confidence_level: f64, parameter: String) -> Self {
        Self {
            lower,
            upper,
            confidence_level,
            parameter,
        }
    }

    pub fn width(&self) -> f64 {
        self.upper - self.lower
    }

    pub fn contains(&self, value: f64) -> bool {
        value >= self.lower && value <= self.upper
    }
}

/// Statistical tests implementation
pub struct StatisticalTests;

impl StatisticalTests {
    /// One-sample t-test
    pub fn one_sample_ttest(
        data: &Array1<f64>,
        population_mean: f64,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n = data.len() as f64;
        let sample_mean = data.mean().unwrap();
        let sample_std = Self::standard_deviation(data);

        if sample_std < f64::EPSILON {
            return Err(UtilsError::InvalidParameter(
                "Standard deviation is zero".to_string(),
            ));
        }

        let t_statistic = (sample_mean - population_mean) / (sample_std / n.sqrt());
        let degrees_of_freedom = n - 1.0;

        // Approximate p-value using t-distribution (simplified)
        let p_value = Self::t_distribution_cdf(-t_statistic.abs(), degrees_of_freedom) * 2.0;

        Ok(TestResult::new(
            t_statistic,
            p_value,
            "One-sample t-test".to_string(),
            alpha,
        ))
    }

    /// Two-sample t-test (assuming equal variances)
    pub fn two_sample_ttest(
        data1: &Array1<f64>,
        data2: &Array1<f64>,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if data1.is_empty() || data2.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;
        let mean1 = data1.mean().unwrap();
        let mean2 = data2.mean().unwrap();
        let std1 = Self::standard_deviation(data1);
        let std2 = Self::standard_deviation(data2);

        // Pooled standard deviation
        let pooled_std =
            ((std1.powi(2) * (n1 - 1.0) + std2.powi(2) * (n2 - 1.0)) / (n1 + n2 - 2.0)).sqrt();

        if pooled_std < f64::EPSILON {
            return Err(UtilsError::InvalidParameter(
                "Pooled standard deviation is zero".to_string(),
            ));
        }

        let t_statistic = (mean1 - mean2) / (pooled_std * (1.0 / n1 + 1.0 / n2).sqrt());
        let degrees_of_freedom = n1 + n2 - 2.0;

        let p_value = Self::t_distribution_cdf(-t_statistic.abs(), degrees_of_freedom) * 2.0;

        Ok(TestResult::new(
            t_statistic,
            p_value,
            "Two-sample t-test".to_string(),
            alpha,
        ))
    }

    /// Welch's t-test (unequal variances)
    pub fn welch_ttest(
        data1: &Array1<f64>,
        data2: &Array1<f64>,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if data1.is_empty() || data2.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n1 = data1.len() as f64;
        let n2 = data2.len() as f64;
        let mean1 = data1.mean().unwrap();
        let mean2 = data2.mean().unwrap();
        let var1 = Self::variance(data1);
        let var2 = Self::variance(data2);

        let se = (var1 / n1 + var2 / n2).sqrt();
        if se < f64::EPSILON {
            return Err(UtilsError::InvalidParameter(
                "Standard error is zero".to_string(),
            ));
        }

        let t_statistic = (mean1 - mean2) / se;

        // Welch-Satterthwaite degrees of freedom
        let degrees_of_freedom = (var1 / n1 + var2 / n2).powi(2)
            / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));

        let p_value = Self::t_distribution_cdf(-t_statistic.abs(), degrees_of_freedom) * 2.0;

        Ok(TestResult::new(
            t_statistic,
            p_value,
            "Welch's t-test".to_string(),
            alpha,
        ))
    }

    /// Chi-square goodness of fit test
    pub fn chi_square_goodness_of_fit(
        observed: &Array1<f64>,
        expected: &Array1<f64>,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if observed.len() != expected.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![expected.len()],
                actual: vec![observed.len()],
            });
        }

        if observed.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let mut chi_square = 0.0;
        for (obs, exp) in observed.iter().zip(expected.iter()) {
            if *exp <= 0.0 {
                return Err(UtilsError::InvalidParameter(
                    "Expected frequencies must be positive".to_string(),
                ));
            }
            chi_square += (obs - exp).powi(2) / exp;
        }

        let degrees_of_freedom = (observed.len() - 1) as f64;
        let p_value = 1.0 - Self::chi_square_cdf(chi_square, degrees_of_freedom);

        Ok(TestResult::new(
            chi_square,
            p_value,
            "Chi-square goodness of fit".to_string(),
            alpha,
        ))
    }

    /// Kolmogorov-Smirnov test for normality
    pub fn ks_test_normality(data: &Array1<f64>, alpha: f64) -> Result<TestResult, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n = data.len() as f64;
        let mean = data.mean().unwrap();
        let std = Self::standard_deviation(data);

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut d_plus = 0.0;
        let mut d_minus = 0.0;

        for (i, &value) in sorted_data.iter().enumerate() {
            let empirical_cdf = (i + 1) as f64 / n;
            let theoretical_cdf = Self::normal_cdf((value - mean) / std);

            d_plus = d_plus.max(empirical_cdf - theoretical_cdf);
            d_minus = d_minus.max(theoretical_cdf - empirical_cdf);
        }

        let ks_statistic = d_plus.max(d_minus);

        // Approximate p-value using Kolmogorov distribution
        let p_value = Self::kolmogorov_smirnov_p_value(ks_statistic, n);

        Ok(TestResult::new(
            ks_statistic,
            p_value,
            "Kolmogorov-Smirnov normality test".to_string(),
            alpha,
        ))
    }

    /// Anderson-Darling test for normality
    pub fn anderson_darling_test(data: &Array1<f64>, alpha: f64) -> Result<TestResult, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n = data.len() as f64;
        let mean = data.mean().unwrap();
        let std = Self::standard_deviation(data);

        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mut ad_statistic = 0.0;

        for (i, &value) in sorted_data.iter().enumerate() {
            let z = (value - mean) / std;
            let phi = Self::normal_cdf(z);
            let phi_complement = 1.0 - phi;

            if phi > 0.0 && phi < 1.0 && phi_complement > 0.0 {
                let j = i + 1;
                ad_statistic +=
                    ((2 * j - 1) as f64) * (phi.ln() + sorted_data[n as usize - j].ln());
            }
        }

        ad_statistic = -n - ad_statistic / n;

        // Adjust for finite sample size
        ad_statistic *= 1.0 + 0.75 / n + 2.25 / n.powi(2);

        // Approximate p-value
        let p_value = if ad_statistic >= 0.6 {
            (-1.2337141 / ad_statistic).exp()
                * (2.00012
                    + (ad_statistic
                        * (-3.00021
                            + ad_statistic
                                * (12.24425
                                    + ad_statistic
                                        * (-17.2385
                                            + ad_statistic * (12.79 - ad_statistic * 5.27)))))
                        .exp())
        } else if ad_statistic >= 0.34 {
            (-0.9177 - 2.0637 * ad_statistic).exp()
        } else if ad_statistic >= 0.2 {
            1.0 - (-8.318 + 42.796 * ad_statistic - 59.938 * ad_statistic.powi(2)).exp()
        } else {
            1.0 - (-13.436 + 101.14 * ad_statistic - 223.73 * ad_statistic.powi(2)).exp()
        };

        Ok(TestResult::new(
            ad_statistic,
            p_value,
            "Anderson-Darling normality test".to_string(),
            alpha,
        ))
    }

    // Helper functions for statistical distributions

    fn standard_deviation(data: &Array1<f64>) -> f64 {
        Self::variance(data).sqrt()
    }

    fn variance(data: &Array1<f64>) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mean = data.mean().unwrap();
        let sum_squares = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>();
        sum_squares / (data.len() - 1) as f64
    }

    /// Approximate normal CDF using error function
    fn normal_cdf(x: f64) -> f64 {
        0.5 * (1.0 + SpecialFunctions::erf(x / 2.0_f64.sqrt()))
    }

    /// Approximate t-distribution CDF (simplified)
    fn t_distribution_cdf(t: f64, df: f64) -> f64 {
        if df >= 30.0 {
            // For large df, t-distribution approaches normal
            return Self::normal_cdf(t);
        }

        // Simplified approximation for t-distribution
        let x = t / (t.powi(2) + df).sqrt();
        0.5 + 0.5 * x * SpecialFunctions::gamma((df + 1.0) / 2.0)
            / ((df * std::f64::consts::PI).sqrt() * SpecialFunctions::gamma(df / 2.0))
    }

    /// Approximate chi-square CDF
    fn chi_square_cdf(x: f64, df: f64) -> f64 {
        if x <= 0.0 {
            return 0.0;
        }

        // Use incomplete gamma function
        SpecialFunctions::gamma_inc(df / 2.0, x / 2.0) / SpecialFunctions::gamma(df / 2.0)
    }

    /// Approximate Kolmogorov-Smirnov p-value
    fn kolmogorov_smirnov_p_value(d: f64, n: f64) -> f64 {
        let lambda = d * n.sqrt();
        let mut p_value = 0.0;

        for i in 1..=10 {
            let term = (-2.0 * (i as f64).powi(2) * lambda.powi(2)).exp();
            if i % 2 == 1 {
                p_value += term;
            } else {
                p_value -= term;
            }
        }

        2.0 * p_value
    }
}

/// Confidence interval computation utilities
pub struct ConfidenceIntervals;

impl ConfidenceIntervals {
    /// Confidence interval for mean (t-distribution)
    pub fn mean_ci(
        data: &Array1<f64>,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        if !(0.0..1.0).contains(&confidence_level) {
            return Err(UtilsError::InvalidParameter(
                "Confidence level must be between 0 and 1".to_string(),
            ));
        }

        let n = data.len() as f64;
        let mean = data.mean().unwrap();
        let std = StatisticalTests::standard_deviation(data);
        let se = std / n.sqrt();

        let alpha = 1.0 - confidence_level;
        let df = n - 1.0;

        // Approximate t-critical value (simplified)
        let t_critical = Self::t_critical_value(alpha / 2.0, df);
        let margin_of_error = t_critical * se;

        Ok(ConfidenceInterval::new(
            mean - margin_of_error,
            mean + margin_of_error,
            confidence_level,
            "Mean".to_string(),
        ))
    }

    /// Confidence interval for proportion
    pub fn proportion_ci(
        successes: usize,
        trials: usize,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval, UtilsError> {
        if trials == 0 {
            return Err(UtilsError::InvalidParameter(
                "Number of trials must be positive".to_string(),
            ));
        }

        if successes > trials {
            return Err(UtilsError::InvalidParameter(
                "Successes cannot exceed trials".to_string(),
            ));
        }

        let p = successes as f64 / trials as f64;
        let n = trials as f64;
        let alpha = 1.0 - confidence_level;

        // Use normal approximation for large samples
        if n * p >= 5.0 && n * (1.0 - p) >= 5.0 {
            let z_critical = Self::normal_critical_value(alpha / 2.0);
            let se = (p * (1.0 - p) / n).sqrt();
            let margin_of_error = z_critical * se;

            Ok(ConfidenceInterval::new(
                (p - margin_of_error).max(0.0),
                (p + margin_of_error).min(1.0),
                confidence_level,
                "Proportion".to_string(),
            ))
        } else {
            // Use Wilson score interval for small samples
            let z = Self::normal_critical_value(alpha / 2.0);
            let z2 = z * z;
            let center = (p + z2 / (2.0 * n)) / (1.0 + z2 / n);
            let width = z / (1.0 + z2 / n) * (p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt();

            Ok(ConfidenceInterval::new(
                (center - width).max(0.0),
                (center + width).min(1.0),
                confidence_level,
                "Proportion (Wilson)".to_string(),
            ))
        }
    }

    /// Confidence interval for variance
    pub fn variance_ci(
        data: &Array1<f64>,
        confidence_level: f64,
    ) -> Result<ConfidenceInterval, UtilsError> {
        if data.len() <= 1 {
            return Err(UtilsError::InsufficientData {
                min: 2,
                actual: data.len(),
            });
        }

        let n = data.len() as f64;
        let variance = StatisticalTests::variance(data);
        let df = n - 1.0;
        let alpha = 1.0 - confidence_level;

        // Chi-square critical values (approximated)
        let chi2_lower = Self::chi_square_critical_value(1.0 - alpha / 2.0, df);
        let chi2_upper = Self::chi_square_critical_value(alpha / 2.0, df);

        let lower = df * variance / chi2_upper;
        let upper = df * variance / chi2_lower;

        Ok(ConfidenceInterval::new(
            lower,
            upper,
            confidence_level,
            "Variance".to_string(),
        ))
    }

    // Helper functions for critical values

    fn normal_critical_value(alpha: f64) -> f64 {
        // Approximation for normal critical values
        if alpha <= 0.001 {
            3.291
        } else if alpha <= 0.005 {
            2.807
        } else if alpha <= 0.01 {
            2.576
        } else if alpha <= 0.025 {
            1.960
        } else if alpha <= 0.05 {
            1.645
        } else if alpha <= 0.1 {
            1.282
        } else {
            0.674
        }
    }

    fn t_critical_value(alpha: f64, df: f64) -> f64 {
        if df >= 30.0 {
            return Self::normal_critical_value(alpha);
        }

        // Simplified t-critical value approximation
        let normal_val = Self::normal_critical_value(alpha);
        let correction = (1.0 + (normal_val.powi(2) + 1.0) / (4.0 * df))
            * (1.0
                + (5.0 * normal_val.powi(4) + 16.0 * normal_val.powi(2) + 3.0)
                    / (96.0 * df.powi(2)));
        normal_val * correction
    }

    fn chi_square_critical_value(alpha: f64, df: f64) -> f64 {
        // Simplified chi-square critical value approximation
        let h = 2.0 / (9.0 * df);
        let normal_val = Self::normal_critical_value(alpha);
        df * (1.0 - h + normal_val * (h * 2.0).sqrt()).powi(3)
    }
}

/// Correlation analysis utilities
pub struct CorrelationAnalysis;

impl CorrelationAnalysis {
    /// Pearson correlation coefficient
    pub fn pearson_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, UtilsError> {
        if x.len() != y.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        if x.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let _n = x.len() as f64;
        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator < f64::EPSILON {
            return Ok(0.0);
        }

        Ok(numerator / denominator)
    }

    /// Spearman rank correlation coefficient
    pub fn spearman_correlation(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, UtilsError> {
        if x.len() != y.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        let ranks_x = Self::compute_ranks(x);
        let ranks_y = Self::compute_ranks(y);

        Self::pearson_correlation(&ranks_x, &ranks_y)
    }

    /// Kendall's tau correlation coefficient
    pub fn kendall_tau(x: &Array1<f64>, y: &Array1<f64>) -> Result<f64, UtilsError> {
        if x.len() != y.len() {
            return Err(UtilsError::ShapeMismatch {
                expected: vec![x.len()],
                actual: vec![y.len()],
            });
        }

        if x.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let n = x.len();
        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let sign_x = (x[j] - x[i]).signum();
                let sign_y = (y[j] - y[i]).signum();

                if sign_x * sign_y > 0.0 {
                    concordant += 1;
                } else if sign_x * sign_y < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        Ok((concordant - discordant) as f64 / total_pairs as f64)
    }

    /// Correlation matrix for multiple variables
    pub fn correlation_matrix(data: &Array2<f64>) -> Result<Array2<f64>, UtilsError> {
        let (n_samples, n_features) = data.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(UtilsError::EmptyInput);
        }

        let mut corr_matrix = Array2::zeros((n_features, n_features));

        for i in 0..n_features {
            for j in 0..n_features {
                if i == j {
                    corr_matrix[(i, j)] = 1.0;
                } else {
                    let col_i = data.column(i).to_owned();
                    let col_j = data.column(j).to_owned();
                    corr_matrix[(i, j)] = Self::pearson_correlation(&col_i, &col_j)?;
                }
            }
        }

        Ok(corr_matrix)
    }

    /// Test correlation significance
    pub fn correlation_test(
        correlation: f64,
        n: usize,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if n < 3 {
            return Err(UtilsError::InsufficientData { min: 3, actual: n });
        }

        let df = (n - 2) as f64;
        let t_statistic = correlation * (df / (1.0 - correlation.powi(2))).sqrt();

        let p_value = 2.0 * StatisticalTests::t_distribution_cdf(-t_statistic.abs(), df);

        Ok(TestResult::new(
            t_statistic,
            p_value,
            "Correlation significance test".to_string(),
            alpha,
        ))
    }

    fn compute_ranks(data: &Array1<f64>) -> Array1<f64> {
        let mut indexed_data: Vec<(usize, f64)> =
            data.iter().enumerate().map(|(i, &x)| (i, x)).collect();
        indexed_data.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let mut ranks = Array1::zeros(data.len());

        for (rank, (original_index, _)) in indexed_data.iter().enumerate() {
            ranks[*original_index] = (rank + 1) as f64;
        }

        // Handle ties by averaging ranks
        let mut i = 0;
        while i < indexed_data.len() {
            let current_value = indexed_data[i].1;
            let mut j = i + 1;

            while j < indexed_data.len() && (indexed_data[j].1 - current_value).abs() < f64::EPSILON
            {
                j += 1;
            }

            if j > i + 1 {
                // There are ties
                let average_rank = ((i + 1) + j) as f64 / 2.0;
                for k in i..j {
                    ranks[indexed_data[k].0] = average_rank;
                }
            }

            i = j;
        }

        ranks
    }
}

/// Distribution fitting utilities
pub struct DistributionFitting;

impl DistributionFitting {
    /// Fit normal distribution parameters
    pub fn fit_normal(data: &Array1<f64>) -> Result<(f64, f64), UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let mean = data.mean().unwrap();
        let std = StatisticalTests::standard_deviation(data);

        Ok((mean, std))
    }

    /// Fit exponential distribution parameter
    pub fn fit_exponential(data: &Array1<f64>) -> Result<f64, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        // Check for positive values
        if data.iter().any(|&x| x <= 0.0) {
            return Err(UtilsError::InvalidParameter(
                "Exponential distribution requires positive values".to_string(),
            ));
        }

        let mean = data.mean().unwrap();
        Ok(1.0 / mean) // Lambda parameter
    }

    /// Fit uniform distribution parameters
    pub fn fit_uniform(data: &Array1<f64>) -> Result<(f64, f64), UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let min = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        Ok((min, max))
    }

    /// Goodness of fit test using chi-square
    pub fn goodness_of_fit_test(
        data: &Array1<f64>,
        expected_cdf: fn(f64, &[f64]) -> f64,
        parameters: &[f64],
        bins: usize,
        alpha: f64,
    ) -> Result<TestResult, UtilsError> {
        if data.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        if bins < 2 {
            return Err(UtilsError::InvalidParameter(
                "Number of bins must be at least 2".to_string(),
            ));
        }

        let n = data.len() as f64;
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let bin_width = (max_val - min_val) / bins as f64;
        let mut observed = Array1::zeros(bins);
        let mut expected = Array1::zeros(bins);

        // Count observed frequencies
        for &value in data.iter() {
            let bin_index = ((value - min_val) / bin_width).floor() as usize;
            let bin_index = bin_index.min(bins - 1);
            observed[bin_index] += 1.0;
        }

        // Calculate expected frequencies
        for i in 0..bins {
            let lower = min_val + i as f64 * bin_width;
            let upper = min_val + (i + 1) as f64 * bin_width;
            let prob = expected_cdf(upper, parameters) - expected_cdf(lower, parameters);
            expected[i] = n * prob;
        }

        StatisticalTests::chi_square_goodness_of_fit(&observed, &expected, alpha)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_one_sample_ttest() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = StatisticalTests::one_sample_ttest(&data, 3.0, 0.05).unwrap();

        assert_eq!(result.test_name, "One-sample t-test");
        assert!(!result.statistic.is_nan());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_two_sample_ttest() {
        let data1 = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let data2 = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = StatisticalTests::two_sample_ttest(&data1, &data2, 0.05).unwrap();

        assert_eq!(result.test_name, "Two-sample t-test");
        assert!(!result.statistic.is_nan());
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];
        let correlation = CorrelationAnalysis::pearson_correlation(&x, &y).unwrap();

        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // y = x^2
        let correlation = CorrelationAnalysis::spearman_correlation(&x, &y).unwrap();

        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_confidence_interval_mean() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let ci = ConfidenceIntervals::mean_ci(&data, 0.95).unwrap();

        assert_eq!(ci.parameter, "Mean");
        assert_eq!(ci.confidence_level, 0.95);
        assert!(ci.lower < ci.upper);
        assert!(ci.contains(3.0)); // Should contain the sample mean
    }

    #[test]
    fn test_confidence_interval_proportion() {
        let ci = ConfidenceIntervals::proportion_ci(30, 100, 0.95).unwrap();

        assert_eq!(ci.parameter, "Proportion");
        assert!(ci.lower >= 0.0 && ci.upper <= 1.0);
        assert!(ci.contains(0.3)); // Should contain the sample proportion
    }

    #[test]
    fn test_chi_square_goodness_of_fit() {
        let observed = array![10.0, 15.0, 8.0, 12.0];
        let expected = array![11.25, 11.25, 11.25, 11.25];
        let result =
            StatisticalTests::chi_square_goodness_of_fit(&observed, &expected, 0.05).unwrap();

        assert_eq!(result.test_name, "Chi-square goodness of fit");
        assert!(result.statistic >= 0.0);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_correlation_matrix() {
        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let corr_matrix = CorrelationAnalysis::correlation_matrix(&data).unwrap();

        assert_eq!(corr_matrix.shape(), &[3, 3]);

        // Diagonal should be 1.0
        for i in 0..3 {
            assert_abs_diff_eq!(corr_matrix[(i, i)], 1.0, epsilon = 1e-10);
        }

        // Matrix should be symmetric
        for i in 0..3 {
            for j in 0..3 {
                assert_abs_diff_eq!(corr_matrix[(i, j)], corr_matrix[(j, i)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_distribution_fitting() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let (mean, std) = DistributionFitting::fit_normal(&data).unwrap();
        assert_abs_diff_eq!(mean, 3.0, epsilon = 1e-10);
        assert!(std > 0.0);

        let (min, max) = DistributionFitting::fit_uniform(&data).unwrap();
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
    }

    #[test]
    fn test_kendall_tau() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let tau = CorrelationAnalysis::kendall_tau(&x, &y).unwrap();

        assert_abs_diff_eq!(tau, 1.0, epsilon = 1e-10);
    }
}

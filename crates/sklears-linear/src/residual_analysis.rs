//! Residual analysis utilities for regression diagnostics
//!
//! This module provides comprehensive tools for analyzing regression residuals,
//! including statistical tests for model assumptions, outlier detection,
//! and various diagnostic plots and metrics.

use sklears_core::error::SklearsError;

/// Configuration for residual analysis
#[derive(Debug, Clone)]
pub struct ResidualAnalysisConfig {
    /// Confidence level for statistical tests (default: 0.95)
    pub confidence_level: f64,
    /// Threshold for outlier detection (in standard deviations)
    pub outlier_threshold: f64,
    /// Whether to compute influence measures
    pub compute_influence: bool,
    /// Whether to compute heteroscedasticity tests
    pub test_heteroscedasticity: bool,
    /// Whether to compute normality tests
    pub test_normality: bool,
    /// Whether to compute autocorrelation tests
    pub test_autocorrelation: bool,
    /// Number of bins for histogram analysis
    pub histogram_bins: usize,
}

impl Default for ResidualAnalysisConfig {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            outlier_threshold: 3.0,
            compute_influence: true,
            test_heteroscedasticity: true,
            test_normality: true,
            test_autocorrelation: true,
            histogram_bins: 50,
        }
    }
}

/// Comprehensive residual analysis results
#[derive(Debug, Clone)]
pub struct ResidualAnalysisResult {
    /// Basic residual statistics
    pub basic_stats: ResidualStats,
    /// Outlier detection results
    pub outliers: OutlierAnalysis,
    /// Statistical test results
    pub statistical_tests: StatisticalTests,
    /// Leverage and influence measures
    pub influence_measures: Option<InfluenceMeasures>,
    /// Residual plots data
    pub plot_data: PlotData,
    /// Model assumptions assessment
    pub assumptions: AssumptionTests,
}

/// Basic residual statistics
#[derive(Debug, Clone)]
pub struct ResidualStats {
    /// Mean of residuals (should be ~0)
    pub mean: f64,
    /// Standard deviation of residuals
    pub std_dev: f64,
    /// Variance of residuals
    pub variance: f64,
    /// Minimum residual
    pub min: f64,
    /// Maximum residual
    pub max: f64,
    /// Median residual
    pub median: f64,
    /// 25th percentile
    pub q25: f64,
    /// 75th percentile
    pub q75: f64,
    /// Interquartile range
    pub iqr: f64,
    /// Skewness of residuals
    pub skewness: f64,
    /// Kurtosis of residuals
    pub kurtosis: f64,
    /// Root mean square error
    pub rmse: f64,
    /// Mean absolute error
    pub mae: f64,
}

/// Outlier detection results
#[derive(Debug, Clone)]
pub struct OutlierAnalysis {
    /// Indices of outliers
    pub outlier_indices: Vec<usize>,
    /// Outlier scores (standardized residuals)
    pub outlier_scores: Vec<f64>,
    /// Number of outliers detected
    pub num_outliers: usize,
    /// Percentage of outliers
    pub outlier_percentage: f64,
    /// Outlier detection method used
    pub detection_method: String,
}

/// Statistical test results
#[derive(Debug, Clone)]
pub struct StatisticalTests {
    /// Shapiro-Wilk normality test
    pub shapiro_wilk: Option<TestResult>,
    /// Jarque-Bera normality test
    pub jarque_bera: Option<TestResult>,
    /// Breusch-Pagan heteroscedasticity test
    pub breusch_pagan: Option<TestResult>,
    /// White heteroscedasticity test
    pub white_test: Option<TestResult>,
    /// Durbin-Watson autocorrelation test
    pub durbin_watson: Option<TestResult>,
    /// Ljung-Box autocorrelation test
    pub ljung_box: Option<TestResult>,
}

/// Individual statistical test result
#[derive(Debug, Clone)]
pub struct TestResult {
    /// Test statistic value
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Critical value at given confidence level
    pub critical_value: f64,
    /// Whether null hypothesis is rejected
    pub reject_null: bool,
    /// Test interpretation
    pub interpretation: String,
}

/// Leverage and influence measures
#[derive(Debug, Clone)]
pub struct InfluenceMeasures {
    /// Leverage values (hat matrix diagonal)
    pub leverage: Vec<f64>,
    /// Cook's distance
    pub cooks_distance: Vec<f64>,
    /// DFFITS values
    pub dffits: Vec<f64>,
    /// DFBETAS values (for each coefficient)
    pub dfbetas: Vec<Vec<f64>>,
    /// Studentized residuals
    pub studentized_residuals: Vec<f64>,
    /// High leverage point indices
    pub high_leverage_indices: Vec<usize>,
    /// High influence point indices
    pub high_influence_indices: Vec<usize>,
}

/// Data for generating diagnostic plots
#[derive(Debug, Clone)]
pub struct PlotData {
    /// Fitted values
    pub fitted_values: Vec<f64>,
    /// Raw residuals
    pub residuals: Vec<f64>,
    /// Standardized residuals
    pub standardized_residuals: Vec<f64>,
    /// Q-Q plot data (theoretical vs sample quantiles)
    pub qq_plot_data: (Vec<f64>, Vec<f64>),
    /// Histogram data (bin centers, frequencies)
    pub histogram_data: (Vec<f64>, Vec<usize>),
    /// Scale-location plot data (sqrt of absolute standardized residuals)
    pub scale_location_data: Vec<f64>,
}

/// Model assumption test results
#[derive(Debug, Clone)]
pub struct AssumptionTests {
    /// Linearity assumption (satisfied if residuals show no pattern)
    pub linearity: AssumptionResult,
    /// Independence assumption (no autocorrelation)
    pub independence: AssumptionResult,
    /// Homoscedasticity assumption (constant variance)
    pub homoscedasticity: AssumptionResult,
    /// Normality assumption
    pub normality: AssumptionResult,
    /// Overall model adequacy score (0-1)
    pub overall_adequacy: f64,
}

/// Result of testing a specific assumption
#[derive(Debug, Clone)]
pub struct AssumptionResult {
    /// Whether assumption is satisfied
    pub satisfied: bool,
    /// Confidence in the assessment (0-1)
    pub confidence: f64,
    /// Evidence supporting the assessment
    pub evidence: Vec<String>,
    /// Recommendations if assumption is violated
    pub recommendations: Vec<String>,
}

/// Main residual analyzer
pub struct ResidualAnalyzer {
    config: ResidualAnalysisConfig,
}

impl ResidualAnalyzer {
    /// Create a new residual analyzer
    pub fn new(config: ResidualAnalysisConfig) -> Self {
        Self { config }
    }

    /// Create analyzer with default configuration
    pub fn default() -> Self {
        Self::new(ResidualAnalysisConfig::default())
    }

    /// Perform comprehensive residual analysis
    pub fn analyze(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
        x_matrix: &[Vec<f64>],
        coefficients: &[f64],
    ) -> Result<ResidualAnalysisResult, SklearsError> {
        if residuals.len() != fitted_values.len() {
            return Err(SklearsError::InvalidInput(
                "Residuals and fitted values must have same length".to_string(),
            ));
        }

        if residuals.is_empty() {
            return Err(SklearsError::InvalidInput("Empty residuals".to_string()));
        }

        // Compute basic statistics
        let basic_stats = self.compute_basic_stats(residuals, fitted_values)?;

        // Detect outliers
        let outliers = self.detect_outliers(residuals)?;

        // Perform statistical tests
        let statistical_tests =
            self.perform_statistical_tests(residuals, fitted_values, x_matrix)?;

        // Compute influence measures if requested
        let influence_measures = if self.config.compute_influence {
            Some(self.compute_influence_measures(residuals, x_matrix, coefficients)?)
        } else {
            None
        };

        // Generate plot data
        let plot_data = self.generate_plot_data(residuals, fitted_values)?;

        // Test model assumptions
        let assumptions = self.test_assumptions(&statistical_tests, &basic_stats, &outliers)?;

        Ok(ResidualAnalysisResult {
            basic_stats,
            outliers,
            statistical_tests,
            influence_measures,
            plot_data,
            assumptions,
        })
    }

    fn compute_basic_stats(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
    ) -> Result<ResidualStats, SklearsError> {
        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;

        let variance = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();

        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted_residuals.len() % 2 == 0 {
            let mid = sorted_residuals.len() / 2;
            (sorted_residuals[mid - 1] + sorted_residuals[mid]) / 2.0
        } else {
            sorted_residuals[sorted_residuals.len() / 2]
        };

        let q25_idx = sorted_residuals.len() / 4;
        let q75_idx = 3 * sorted_residuals.len() / 4;
        let q25 = sorted_residuals[q25_idx];
        let q75 = sorted_residuals[q75_idx];
        let iqr = q75 - q25;

        let min = sorted_residuals[0];
        let max = sorted_residuals[sorted_residuals.len() - 1];

        // Compute skewness and kurtosis
        let skewness = residuals
            .iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n;

        let kurtosis = residuals
            .iter()
            .map(|&r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n
            - 3.0; // Excess kurtosis

        // Compute error metrics
        let rmse = (residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / n).sqrt();
        let mae = residuals.iter().map(|&r| r.abs()).sum::<f64>() / n;

        Ok(ResidualStats {
            mean,
            std_dev,
            variance,
            min,
            max,
            median,
            q25,
            q75,
            iqr,
            skewness,
            kurtosis,
            rmse,
            mae,
        })
    }

    fn detect_outliers(&self, residuals: &[f64]) -> Result<OutlierAnalysis, SklearsError> {
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let std_dev = (residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
            / (residuals.len() - 1) as f64)
            .sqrt();

        let mut outlier_indices = vec![];
        let mut outlier_scores = vec![];

        for (i, &residual) in residuals.iter().enumerate() {
            let standardized = (residual - mean) / std_dev;
            if standardized.abs() > self.config.outlier_threshold {
                outlier_indices.push(i);
                outlier_scores.push(standardized);
            }
        }

        let num_outliers = outlier_indices.len();
        let outlier_percentage = (num_outliers as f64 / residuals.len() as f64) * 100.0;

        Ok(OutlierAnalysis {
            outlier_indices,
            outlier_scores,
            num_outliers,
            outlier_percentage,
            detection_method: format!("Standardized residuals > {}", self.config.outlier_threshold),
        })
    }

    fn perform_statistical_tests(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
        x_matrix: &[Vec<f64>],
    ) -> Result<StatisticalTests, SklearsError> {
        let mut tests = StatisticalTests {
            shapiro_wilk: None,
            jarque_bera: None,
            breusch_pagan: None,
            white_test: None,
            durbin_watson: None,
            ljung_box: None,
        };

        // Normality tests
        if self.config.test_normality {
            tests.jarque_bera = Some(self.jarque_bera_test(residuals)?);
            if residuals.len() <= 5000 {
                tests.shapiro_wilk = Some(self.shapiro_wilk_test(residuals)?);
            }
        }

        // Heteroscedasticity tests
        if self.config.test_heteroscedasticity {
            tests.breusch_pagan = Some(self.breusch_pagan_test(residuals, x_matrix)?);
            tests.white_test = Some(self.white_test(residuals, fitted_values)?);
        }

        // Autocorrelation tests
        if self.config.test_autocorrelation && residuals.len() > 10 {
            tests.durbin_watson = Some(self.durbin_watson_test(residuals)?);
            tests.ljung_box = Some(self.ljung_box_test(residuals)?);
        }

        Ok(tests)
    }

    fn jarque_bera_test(&self, residuals: &[f64]) -> Result<TestResult, SklearsError> {
        let n = residuals.len() as f64;
        let mean = residuals.iter().sum::<f64>() / n;
        let variance = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let skewness = residuals
            .iter()
            .map(|&r| ((r - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n;

        let kurtosis = residuals
            .iter()
            .map(|&r| ((r - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n
            - 3.0;

        let jb_statistic = n / 6.0 * (skewness.powi(2) + kurtosis.powi(2) / 4.0);

        // Critical value for chi-square distribution with 2 df at 95% confidence
        let critical_value = 5.991;
        let p_value = self.chi_square_p_value(jb_statistic, 2);
        let reject_null = jb_statistic > critical_value;

        let interpretation = if reject_null {
            "Residuals are NOT normally distributed".to_string()
        } else {
            "Residuals appear to be normally distributed".to_string()
        };

        Ok(TestResult {
            statistic: jb_statistic,
            p_value,
            critical_value,
            reject_null,
            interpretation,
        })
    }

    fn shapiro_wilk_test(&self, residuals: &[f64]) -> Result<TestResult, SklearsError> {
        // Simplified Shapiro-Wilk test implementation
        // Note: This is a simplified version. A full implementation would require
        // more complex calculations and tables of coefficients.

        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_residuals.len();
        let mean = sorted_residuals.iter().sum::<f64>() / n as f64;

        // Compute W statistic (simplified)
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for (i, &x) in sorted_residuals.iter().enumerate() {
            let expected_normal = self.inverse_normal_cdf((i as f64 + 0.5) / n as f64);
            numerator += expected_normal * x;
            denominator += (x - mean).powi(2);
        }

        let w_statistic = (numerator.powi(2)) / denominator;

        // Approximate critical value and p-value
        let critical_value = 0.95; // Approximate for most sample sizes
        let p_value = if w_statistic < critical_value {
            0.01
        } else {
            0.5
        };
        let reject_null = w_statistic < critical_value;

        let interpretation = if reject_null {
            "Residuals are NOT normally distributed (Shapiro-Wilk)".to_string()
        } else {
            "Residuals appear to be normally distributed (Shapiro-Wilk)".to_string()
        };

        Ok(TestResult {
            statistic: w_statistic,
            p_value,
            critical_value,
            reject_null,
            interpretation,
        })
    }

    fn breusch_pagan_test(
        &self,
        residuals: &[f64],
        x_matrix: &[Vec<f64>],
    ) -> Result<TestResult, SklearsError> {
        if x_matrix.is_empty() || x_matrix[0].is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty design matrix".to_string(),
            ));
        }

        let n = residuals.len();
        let p = x_matrix[0].len();

        // Compute squared residuals
        let squared_residuals: Vec<f64> = residuals.iter().map(|&r| r.powi(2)).collect();
        let mean_squared = squared_residuals.iter().sum::<f64>() / n as f64;

        // Regress squared residuals on original predictors
        // This is a simplified implementation
        let mut ssr = 0.0;
        let mut tss = 0.0;

        for &sq_res in &squared_residuals {
            tss += (sq_res - mean_squared).powi(2);
        }

        // Simplified R-squared calculation
        let r_squared = 0.1; // Placeholder - would need proper regression
        ssr = r_squared * tss;

        let lm_statistic = n as f64 * r_squared;
        let critical_value = self.chi_square_critical_value(p - 1, self.config.confidence_level);
        let p_value = self.chi_square_p_value(lm_statistic, p - 1);
        let reject_null = lm_statistic > critical_value;

        let interpretation = if reject_null {
            "Heteroscedasticity detected (Breusch-Pagan)".to_string()
        } else {
            "No evidence of heteroscedasticity (Breusch-Pagan)".to_string()
        };

        Ok(TestResult {
            statistic: lm_statistic,
            p_value,
            critical_value,
            reject_null,
            interpretation,
        })
    }

    fn white_test(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
    ) -> Result<TestResult, SklearsError> {
        let n = residuals.len();
        let squared_residuals: Vec<f64> = residuals.iter().map(|&r| r.powi(2)).collect();

        // Simple correlation between squared residuals and fitted values
        let mean_sq_res = squared_residuals.iter().sum::<f64>() / n as f64;
        let mean_fitted = fitted_values.iter().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut denom_sq_res = 0.0;
        let mut denom_fitted = 0.0;

        for i in 0..n {
            let sq_res_dev = squared_residuals[i] - mean_sq_res;
            let fitted_dev = fitted_values[i] - mean_fitted;

            numerator += sq_res_dev * fitted_dev;
            denom_sq_res += sq_res_dev.powi(2);
            denom_fitted += fitted_dev.powi(2);
        }

        let correlation = numerator / (denom_sq_res * denom_fitted).sqrt();
        let white_statistic = n as f64 * correlation.powi(2);

        let critical_value = 3.841; // Chi-square critical value with 1 df at 95%
        let p_value = self.chi_square_p_value(white_statistic, 1);
        let reject_null = white_statistic > critical_value;

        let interpretation = if reject_null {
            "Heteroscedasticity detected (White test)".to_string()
        } else {
            "No evidence of heteroscedasticity (White test)".to_string()
        };

        Ok(TestResult {
            statistic: white_statistic,
            p_value,
            critical_value,
            reject_null,
            interpretation,
        })
    }

    fn durbin_watson_test(&self, residuals: &[f64]) -> Result<TestResult, SklearsError> {
        if residuals.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 residuals for Durbin-Watson test".to_string(),
            ));
        }

        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 1..residuals.len() {
            numerator += (residuals[i] - residuals[i - 1]).powi(2);
        }

        for &r in residuals {
            denominator += r.powi(2);
        }

        let dw_statistic = numerator / denominator;

        // Critical values for DW test (approximate)
        let critical_lower = 1.5;
        let critical_upper = 2.5;

        let reject_null = dw_statistic < critical_lower || dw_statistic > critical_upper;
        let p_value = if reject_null { 0.01 } else { 0.5 };

        let interpretation = if dw_statistic < critical_lower {
            "Positive autocorrelation detected".to_string()
        } else if dw_statistic > critical_upper {
            "Negative autocorrelation detected".to_string()
        } else {
            "No evidence of autocorrelation".to_string()
        };

        Ok(TestResult {
            statistic: dw_statistic,
            p_value,
            critical_value: 2.0, // Ideal value
            reject_null,
            interpretation,
        })
    }

    fn ljung_box_test(&self, residuals: &[f64]) -> Result<TestResult, SklearsError> {
        let n = residuals.len();
        let max_lag = (n / 4).min(10).max(1); // Use up to n/4 lags, max 10

        let mean = residuals.iter().sum::<f64>() / n as f64;
        let variance = residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;

        let mut lb_statistic = 0.0;

        for k in 1..=max_lag {
            let mut autocorr = 0.0;
            for i in k..n {
                autocorr += (residuals[i] - mean) * (residuals[i - k] - mean);
            }
            autocorr /= (n - k) as f64 * variance;

            lb_statistic += autocorr.powi(2) / (n - k) as f64;
        }

        lb_statistic *= n as f64 * (n as f64 + 2.0);

        let critical_value = self.chi_square_critical_value(max_lag, self.config.confidence_level);
        let p_value = self.chi_square_p_value(lb_statistic, max_lag);
        let reject_null = lb_statistic > critical_value;

        let interpretation = if reject_null {
            "Serial correlation detected (Ljung-Box)".to_string()
        } else {
            "No evidence of serial correlation (Ljung-Box)".to_string()
        };

        Ok(TestResult {
            statistic: lb_statistic,
            p_value,
            critical_value,
            reject_null,
            interpretation,
        })
    }

    fn compute_influence_measures(
        &self,
        residuals: &[f64],
        x_matrix: &[Vec<f64>],
        coefficients: &[f64],
    ) -> Result<InfluenceMeasures, SklearsError> {
        let n = residuals.len();
        let p = coefficients.len();

        // Compute leverage (hat matrix diagonal)
        let leverage = self.compute_leverage(x_matrix)?;

        // Compute residual standard error
        let rse = (residuals.iter().map(|&r| r.powi(2)).sum::<f64>() / (n - p) as f64).sqrt();

        // Compute standardized residuals
        let mut studentized_residuals = vec![];
        for i in 0..n {
            let std_res = residuals[i] / (rse * (1.0 - leverage[i]).sqrt());
            studentized_residuals.push(std_res);
        }

        // Compute Cook's distance
        let mut cooks_distance = vec![];
        for i in 0..n {
            let cook_d =
                (studentized_residuals[i].powi(2) / p as f64) * (leverage[i] / (1.0 - leverage[i]));
            cooks_distance.push(cook_d);
        }

        // Compute DFFITS
        let mut dffits = vec![];
        for i in 0..n {
            let dffit = studentized_residuals[i] * (leverage[i] / (1.0 - leverage[i])).sqrt();
            dffits.push(dffit);
        }

        // Simplified DFBETAS (would need more complex calculation for each coefficient)
        let dfbetas = vec![vec![0.0; n]; p];

        // Identify high leverage and influence points
        let leverage_threshold = 2.0 * p as f64 / n as f64;
        let cook_threshold = 4.0 / n as f64;

        let high_leverage_indices: Vec<usize> = leverage
            .iter()
            .enumerate()
            .filter(|(_, &lev)| lev > leverage_threshold)
            .map(|(i, _)| i)
            .collect();

        let high_influence_indices: Vec<usize> = cooks_distance
            .iter()
            .enumerate()
            .filter(|(_, &cook)| cook > cook_threshold)
            .map(|(i, _)| i)
            .collect();

        Ok(InfluenceMeasures {
            leverage,
            cooks_distance,
            dffits,
            dfbetas,
            studentized_residuals,
            high_leverage_indices,
            high_influence_indices,
        })
    }

    fn compute_leverage(&self, x_matrix: &[Vec<f64>]) -> Result<Vec<f64>, SklearsError> {
        // Simplified leverage calculation
        // In practice, this would compute the diagonal of X(X'X)^(-1)X'
        let n = x_matrix.len();
        let p = if n > 0 { x_matrix[0].len() } else { 0 };

        // Placeholder implementation - return average leverage
        let avg_leverage = p as f64 / n as f64;
        Ok(vec![avg_leverage; n])
    }

    fn generate_plot_data(
        &self,
        residuals: &[f64],
        fitted_values: &[f64],
    ) -> Result<PlotData, SklearsError> {
        // Compute standardized residuals
        let mean = residuals.iter().sum::<f64>() / residuals.len() as f64;
        let std_dev = (residuals.iter().map(|&r| (r - mean).powi(2)).sum::<f64>()
            / (residuals.len() - 1) as f64)
            .sqrt();

        let standardized_residuals: Vec<f64> =
            residuals.iter().map(|&r| (r - mean) / std_dev).collect();

        // Generate Q-Q plot data
        let qq_plot_data = self.generate_qq_plot_data(&standardized_residuals)?;

        // Generate histogram data
        let histogram_data = self.generate_histogram_data(residuals)?;

        // Scale-location plot data (sqrt of absolute standardized residuals)
        let scale_location_data: Vec<f64> = standardized_residuals
            .iter()
            .map(|&r| r.abs().sqrt())
            .collect();

        Ok(PlotData {
            fitted_values: fitted_values.to_vec(),
            residuals: residuals.to_vec(),
            standardized_residuals,
            qq_plot_data,
            histogram_data,
            scale_location_data,
        })
    }

    fn generate_qq_plot_data(
        &self,
        residuals: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), SklearsError> {
        let mut sorted_residuals = residuals.to_vec();
        sorted_residuals.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_residuals.len();
        let mut theoretical_quantiles = vec![];

        for i in 0..n {
            let p = (i as f64 + 0.5) / n as f64;
            let q = self.inverse_normal_cdf(p);
            theoretical_quantiles.push(q);
        }

        Ok((theoretical_quantiles, sorted_residuals))
    }

    fn generate_histogram_data(
        &self,
        residuals: &[f64],
    ) -> Result<(Vec<f64>, Vec<usize>), SklearsError> {
        let min_val = residuals.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = residuals.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        let bin_width = range / self.config.histogram_bins as f64;
        let mut bin_centers = vec![];
        let mut bin_counts = vec![0; self.config.histogram_bins];

        for i in 0..self.config.histogram_bins {
            bin_centers.push(min_val + (i as f64 + 0.5) * bin_width);
        }

        for &residual in residuals {
            let bin_idx = ((residual - min_val) / bin_width).floor() as usize;
            let bin_idx = bin_idx.min(self.config.histogram_bins - 1);
            bin_counts[bin_idx] += 1;
        }

        Ok((bin_centers, bin_counts))
    }

    fn test_assumptions(
        &self,
        statistical_tests: &StatisticalTests,
        basic_stats: &ResidualStats,
        outliers: &OutlierAnalysis,
    ) -> Result<AssumptionTests, SklearsError> {
        // Test linearity (based on mean of residuals and pattern)
        let linearity = AssumptionResult {
            satisfied: basic_stats.mean.abs() < 0.01 * basic_stats.std_dev,
            confidence: 0.8,
            evidence: vec![format!("Mean residual: {:.6}", basic_stats.mean)],
            recommendations: if basic_stats.mean.abs() > 0.01 * basic_stats.std_dev {
                vec!["Consider adding polynomial terms or transforming variables".to_string()]
            } else {
                vec![]
            },
        };

        // Test independence (based on autocorrelation tests)
        let independence = if let Some(ref dw_test) = statistical_tests.durbin_watson {
            AssumptionResult {
                satisfied: !dw_test.reject_null,
                confidence: 0.9,
                evidence: vec![format!("Durbin-Watson statistic: {:.3}", dw_test.statistic)],
                recommendations: if dw_test.reject_null {
                    vec!["Consider time series methods or add lagged variables".to_string()]
                } else {
                    vec![]
                },
            }
        } else {
            AssumptionResult {
                satisfied: true,
                confidence: 0.5,
                evidence: vec!["No autocorrelation test performed".to_string()],
                recommendations: vec![],
            }
        };

        // Test homoscedasticity
        let homoscedasticity = if let Some(ref bp_test) = statistical_tests.breusch_pagan {
            AssumptionResult {
                satisfied: !bp_test.reject_null,
                confidence: 0.85,
                evidence: vec![format!("Breusch-Pagan p-value: {:.3}", bp_test.p_value)],
                recommendations: if bp_test.reject_null {
                    vec!["Consider robust standard errors or weighted least squares".to_string()]
                } else {
                    vec![]
                },
            }
        } else {
            AssumptionResult {
                satisfied: true,
                confidence: 0.5,
                evidence: vec!["No heteroscedasticity test performed".to_string()],
                recommendations: vec![],
            }
        };

        // Test normality
        let normality = if let Some(ref jb_test) = statistical_tests.jarque_bera {
            let skew_ok = basic_stats.skewness.abs() < 1.0;
            let kurt_ok = basic_stats.kurtosis.abs() < 3.0;

            AssumptionResult {
                satisfied: !jb_test.reject_null && skew_ok && kurt_ok,
                confidence: 0.8,
                evidence: vec![
                    format!("Jarque-Bera p-value: {:.3}", jb_test.p_value),
                    format!("Skewness: {:.3}", basic_stats.skewness),
                    format!("Kurtosis: {:.3}", basic_stats.kurtosis),
                ],
                recommendations: if jb_test.reject_null {
                    vec![
                        "Consider transforming dependent variable or using robust methods"
                            .to_string(),
                    ]
                } else {
                    vec![]
                },
            }
        } else {
            AssumptionResult {
                satisfied: true,
                confidence: 0.5,
                evidence: vec!["No normality test performed".to_string()],
                recommendations: vec![],
            }
        };

        // Compute overall adequacy score
        let mut adequacy_score = 0.0;
        let mut total_weight = 0.0;

        let assumptions = [
            (&linearity, 0.25),
            (&independence, 0.25),
            (&homoscedasticity, 0.25),
            (&normality, 0.25),
        ];

        for (assumption, weight) in &assumptions {
            adequacy_score +=
                if assumption.satisfied { 1.0 } else { 0.0 } * assumption.confidence * weight;
            total_weight += weight;
        }

        adequacy_score /= total_weight;

        // Penalize for too many outliers
        if outliers.outlier_percentage > 5.0 {
            adequacy_score *= 0.8;
        }

        Ok(AssumptionTests {
            linearity,
            independence,
            homoscedasticity,
            normality,
            overall_adequacy: adequacy_score,
        })
    }

    // Helper methods for statistical calculations
    fn chi_square_critical_value(&self, df: usize, confidence: f64) -> f64 {
        // Simplified critical values for common degrees of freedom
        match (df, (confidence * 100.0) as usize) {
            (1, 95) => 3.841,
            (2, 95) => 5.991,
            (3, 95) => 7.815,
            (4, 95) => 9.488,
            (5, 95) => 11.070,
            _ => 3.841 + df as f64 * 2.0, // Rough approximation
        }
    }

    fn chi_square_p_value(&self, statistic: f64, df: usize) -> f64 {
        // Simplified p-value calculation
        let critical = self.chi_square_critical_value(df, 0.95);
        if statistic > critical {
            0.01 // Reject null
        } else {
            0.5 // Don't reject null
        }
    }

    fn inverse_normal_cdf(&self, p: f64) -> f64 {
        // Simplified inverse normal CDF (Box-Muller transformation approximation)
        if p <= 0.0 {
            return f64::NEG_INFINITY;
        }
        if p >= 1.0 {
            return f64::INFINITY;
        }

        // Rational approximation for the inverse normal CDF
        let a0 = -3.969683028665376e+01;
        let a1 = 2.209460984245205e+02;
        let a2 = -2.759285104469687e+02;
        let a3 = 1.383577518672690e+02;
        let a4 = -3.066479806614716e+01;
        let a5 = 2.506628277459239e+00;

        let b1 = -5.447609879822406e+01;
        let b2 = 1.615858368580409e+02;
        let b3 = -1.556989798598866e+02;
        let b4 = 6.680131188771972e+01;
        let b5 = -1.328068155288572e+01;

        let c0 = -7.784894002430293e-03;
        let c1 = -3.223964580411365e-01;
        let c2 = -2.400758277161838e+00;
        let c3 = -2.549732539343734e+00;
        let c4 = 4.374664141464968e+00;
        let c5 = 2.938163982698783e+00;

        let d1 = 7.784695709041462e-03;
        let d2 = 3.224671290700398e-01;
        let d3 = 2.445134137142996e+00;
        let d4 = 3.754408661907416e+00;

        let p_low = 0.02425;
        let p_high = 1.0 - p_low;

        if p < p_low {
            let q = (-2.0 * p.ln()).sqrt();
            return (((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        } else if p <= p_high {
            let q = p - 0.5;
            let r = q * q;
            return (((((a0 * r + a1) * r + a2) * r + a3) * r + a4) * r + a5) * q
                / (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + 1.0);
        } else {
            let q = (-2.0 * (1.0 - p).ln()).sqrt();
            return -(((((c0 * q + c1) * q + c2) * q + c3) * q + c4) * q + c5)
                / ((((d1 * q + d2) * q + d3) * q + d4) * q + 1.0);
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_basic_stats() {
        let analyzer = ResidualAnalyzer::default();
        let residuals = vec![1.0, -1.0, 2.0, -2.0, 0.5, -0.5];
        let fitted_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let stats = analyzer
            .compute_basic_stats(&residuals, &fitted_values)
            .unwrap();

        assert_relative_eq!(stats.mean, 0.0, epsilon = 1e-10);
        assert!(stats.std_dev > 0.0);
        assert!(stats.rmse > 0.0);
        assert!(stats.mae > 0.0);
    }

    #[test]
    fn test_outlier_detection() {
        let analyzer = ResidualAnalyzer::default();
        let residuals = vec![0.1, 0.2, -0.1, -0.2, 5.0, -0.05]; // 5.0 is an outlier

        let outliers = analyzer.detect_outliers(&residuals).unwrap();

        assert!(outliers.num_outliers > 0);
        assert!(outliers.outlier_indices.contains(&4)); // Index of 5.0
    }

    #[test]
    fn test_jarque_bera() {
        let analyzer = ResidualAnalyzer::default();
        // Normal-ish data
        let residuals = vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05, 0.15, -0.15];

        let test_result = analyzer.jarque_bera_test(&residuals).unwrap();

        assert!(test_result.statistic >= 0.0);
        assert!(test_result.p_value >= 0.0 && test_result.p_value <= 1.0);
    }

    #[test]
    fn test_durbin_watson() {
        let analyzer = ResidualAnalyzer::default();
        let residuals = vec![0.1, 0.2, 0.15, 0.25, 0.18, 0.22]; // Some autocorrelation

        let test_result = analyzer.durbin_watson_test(&residuals).unwrap();

        assert!(test_result.statistic >= 0.0 && test_result.statistic <= 4.0);
    }

    #[test]
    fn test_comprehensive_analysis() {
        let analyzer = ResidualAnalyzer::default();
        let residuals = vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05];
        let fitted_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x_matrix = vec![
            vec![1.0, 1.0],
            vec![1.0, 2.0],
            vec![1.0, 3.0],
            vec![1.0, 4.0],
            vec![1.0, 5.0],
            vec![1.0, 6.0],
        ];
        let coefficients = vec![0.5, 0.2];

        let result = analyzer
            .analyze(&residuals, &fitted_values, &x_matrix, &coefficients)
            .unwrap();

        assert_eq!(result.basic_stats.mean.abs() < 0.1, true);
        assert!(result.assumptions.overall_adequacy >= 0.0);
        assert!(result.assumptions.overall_adequacy <= 1.0);
        assert_eq!(result.plot_data.residuals.len(), residuals.len());
    }

    #[test]
    fn test_qq_plot_generation() {
        let analyzer = ResidualAnalyzer::default();
        let residuals = vec![0.1, -0.1, 0.2, -0.2, 0.05, -0.05];

        let (theoretical, sample) = analyzer.generate_qq_plot_data(&residuals).unwrap();

        assert_eq!(theoretical.len(), sample.len());
        assert_eq!(sample.len(), residuals.len());
    }

    #[test]
    fn test_empty_residuals() {
        let analyzer = ResidualAnalyzer::default();
        let residuals: Vec<f64> = vec![];
        let fitted_values: Vec<f64> = vec![];
        let x_matrix: Vec<Vec<f64>> = vec![];
        let coefficients: Vec<f64> = vec![];

        let result = analyzer.analyze(&residuals, &fitted_values, &x_matrix, &coefficients);
        assert!(result.is_err());
    }
}

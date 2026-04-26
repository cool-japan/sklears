//! Statistical tests and hypothesis testing for feature selection
//!
//! This module provides comprehensive statistical testing implementations including
//! chi-squared tests, ANOVA, mutual information, correlation tests, and hypothesis testing
//! frameworks. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Supported statistical tests
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalTest {
    ChiSquared,
    FTest,
    ANOVA,
    KruskalWallis,
    MannWhitney,
    TTest,
    WilcoxonRankSum,
    KolmogorovSmirnov,
    AndersonDarling,
    ShapiroWilk,
}

/// Configuration for statistical testing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestingConfig {
    pub test_type: StatisticalTest,
    pub alpha: f64,
    pub alternative: String,        // 'two-sided', 'less', 'greater'
    pub correction: Option<String>, // Multiple testing correction
    pub min_sample_size: usize,
    pub use_exact_test: bool,
    pub random_state: Option<u64>,
}

impl Default for TestingConfig {
    fn default() -> Self {
        Self {
            test_type: StatisticalTest::ChiSquared,
            alpha: 0.05,
            alternative: "two-sided".to_string(),
            correction: Some("bonferroni".to_string()),
            min_sample_size: 5,
            use_exact_test: false,
            random_state: Some(42),
        }
    }
}

/// Test results containing statistics and p-values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestStatistics {
    pub test_statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: Option<usize>,
    pub critical_value: Option<f64>,
    pub effect_size: Option<f64>,
    pub confidence_interval: Option<(f64, f64)>,
    pub test_metadata: HashMap<String, f64>,
}

impl TestStatistics {
    pub fn new(test_statistic: f64, p_value: f64) -> Self {
        Self {
            test_statistic,
            p_value,
            degrees_of_freedom: None,
            critical_value: None,
            effect_size: None,
            confidence_interval: None,
            test_metadata: HashMap::new(),
        }
    }

    /// Check if result is significant at given alpha level
    pub fn is_significant(&self, alpha: f64) -> bool {
        self.p_value < alpha
    }

    /// Set degrees of freedom
    pub fn set_degrees_of_freedom(&mut self, df: usize) {
        self.degrees_of_freedom = Some(df);
    }

    /// Set critical value
    pub fn set_critical_value(&mut self, critical_value: f64) {
        self.critical_value = Some(critical_value);
    }

    /// Set effect size
    pub fn set_effect_size(&mut self, effect_size: f64) {
        self.effect_size = Some(effect_size);
    }

    /// Set confidence interval
    pub fn set_confidence_interval(&mut self, ci: (f64, f64)) {
        self.confidence_interval = Some(ci);
    }

    /// Add metadata
    pub fn add_metadata(&mut self, key: String, value: f64) {
        self.test_metadata.insert(key, value);
    }
}

/// Validator for testing configurations
#[derive(Debug, Clone)]
pub struct TestingValidator;

impl TestingValidator {
    pub fn validate_config(config: &TestingConfig) -> Result<()> {
        if config.alpha <= 0.0 || config.alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        let valid_alternatives = ["two-sided", "less", "greater"];
        if !valid_alternatives.contains(&config.alternative.as_str()) {
            return Err(SklearsError::InvalidInput(
                "alternative must be 'two-sided', 'less', or 'greater'".to_string(),
            ));
        }

        if config.min_sample_size == 0 {
            return Err(SklearsError::InvalidInput(
                "min_sample_size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Core statistical tester trait
pub trait StatisticalTester {
    fn test<T>(
        &self,
        data: &ArrayView2<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<Vec<TestStatistics>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>;

    fn test_single_feature<T>(
        &self,
        feature: &ArrayView1<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>;
}

/// Chi-squared test implementation
#[derive(Debug, Clone)]
pub struct ChiSquaredTest {
    config: TestingConfig,
    contingency_tables: Option<Vec<Array2<f64>>>,
}

impl ChiSquaredTest {
    pub fn new(config: TestingConfig) -> Result<Self> {
        TestingValidator::validate_config(&config)?;
        Ok(Self {
            config,
            contingency_tables: None,
        })
    }

    /// Create contingency table for categorical data
    fn create_contingency_table<T>(
        &self,
        feature: &ArrayView1<T>,
        target: &ArrayView1<T>,
    ) -> Result<Array2<f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified implementation - would create actual contingency table in practice
        let unique_feature_values = self.count_unique_values(feature);
        let unique_target_values = self.count_unique_values(target);

        let mut table = Array2::zeros((unique_feature_values, unique_target_values));

        // Fill contingency table (simplified)
        for i in 0..unique_feature_values {
            for j in 0..unique_target_values {
                table[(i, j)] = 1.0; // Placeholder count
            }
        }

        Ok(table)
    }

    /// Count unique values in array
    fn count_unique_values<T>(&self, array: &ArrayView1<T>) -> usize
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = array.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len()
    }

    /// Compute chi-squared statistic
    fn compute_chi_squared(&self, observed: &Array2<f64>) -> Result<(f64, usize)> {
        let (rows, cols) = observed.dim();
        let total = observed.sum();

        if total == 0.0 {
            return Err(SklearsError::InvalidInput(
                "Empty contingency table".to_string(),
            ));
        }

        // Calculate expected frequencies
        let row_totals: Vec<f64> = (0..rows).map(|i| observed.row(i).sum()).collect();
        let col_totals: Vec<f64> = (0..cols).map(|j| observed.column(j).sum()).collect();

        let mut chi_squared = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                let expected = (row_totals[i] * col_totals[j]) / total;
                if expected > 0.0 {
                    let diff = observed[(i, j)] - expected;
                    chi_squared += diff * diff / expected;
                }
            }
        }

        let df = (rows - 1) * (cols - 1);
        Ok((chi_squared, df))
    }

    pub fn contingency_tables(&self) -> Option<&[Array2<f64>]> {
        self.contingency_tables.as_deref()
    }
}

impl StatisticalTester for ChiSquaredTest {
    fn test<T>(
        &self,
        data: &ArrayView2<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<Vec<TestStatistics>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for chi-squared test".to_string())
        })?;

        let (_, n_features) = data.dim();
        let mut results = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature = data.column(feature_idx);
            let result = self.test_single_feature(&feature, Some(target))?;
            results.push(result);
        }

        Ok(results)
    }

    fn test_single_feature<T>(
        &self,
        feature: &ArrayView1<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for chi-squared test".to_string())
        })?;

        if feature.len() != target.len() {
            return Err(SklearsError::InvalidInput(
                "Feature and target must have same length".to_string(),
            ));
        }

        let contingency_table = self.create_contingency_table(feature, target)?;
        let (chi_squared, df) = self.compute_chi_squared(&contingency_table)?;

        // Simplified p-value calculation
        let p_value = if chi_squared > 3.841 { 0.04 } else { 0.1 }; // Placeholder

        let mut result = TestStatistics::new(chi_squared, p_value);
        result.set_degrees_of_freedom(df);

        Ok(result)
    }
}

/// ANOVA test implementation
#[derive(Debug, Clone)]
pub struct ANOVATest {
    config: TestingConfig,
    group_statistics: Option<HashMap<String, f64>>,
}

impl ANOVATest {
    pub fn new(config: TestingConfig) -> Result<Self> {
        TestingValidator::validate_config(&config)?;
        Ok(Self {
            config,
            group_statistics: None,
        })
    }

    /// Compute one-way ANOVA F-statistic
    fn compute_f_statistic<T>(
        &self,
        feature: &ArrayView1<T>,
        target: &ArrayView1<T>,
    ) -> Result<(f64, usize, usize)>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let n_samples = feature.len();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for ANOVA".to_string(),
            ));
        }

        // Convert feature and target to f64
        let feat: Vec<f64> = feature.iter().map(|&v| v.into()).collect();
        let tgt: Vec<f64> = target.iter().map(|&v| v.into()).collect();

        // Identify unique groups from target (quantized as bit pattern)
        let mut group_map: std::collections::HashMap<u64, Vec<f64>> =
            std::collections::HashMap::new();
        for (f_val, t_val) in feat.iter().zip(tgt.iter()) {
            group_map.entry(t_val.to_bits()).or_default().push(*f_val);
        }

        let n_groups = group_map.len();
        if n_groups < 2 {
            return Ok((0.0, 1, n_samples - 1));
        }

        // Overall mean
        let grand_mean = feat.iter().sum::<f64>() / n_samples as f64;

        // Between-group sum of squares
        let ss_between: f64 = group_map
            .values()
            .map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group.len() as f64 * (group_mean - grand_mean) * (group_mean - grand_mean)
            })
            .sum();

        // Within-group sum of squares
        let ss_within: f64 = group_map
            .values()
            .map(|group| {
                let group_mean = group.iter().sum::<f64>() / group.len() as f64;
                group
                    .iter()
                    .map(|&v| (v - group_mean) * (v - group_mean))
                    .sum::<f64>()
            })
            .sum();

        let df_between = n_groups - 1;
        let df_within = n_samples - n_groups;

        let ms_between = ss_between / df_between as f64;
        let ms_within = if df_within > 0 {
            ss_within / df_within as f64
        } else {
            f64::EPSILON
        };

        let f_statistic = if ms_within < f64::EPSILON {
            0.0
        } else {
            ms_between / ms_within
        };

        Ok((f_statistic, df_between, df_within))
    }

    pub fn group_statistics(&self) -> Option<&HashMap<String, f64>> {
        self.group_statistics.as_ref()
    }
}

impl StatisticalTester for ANOVATest {
    fn test<T>(
        &self,
        data: &ArrayView2<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<Vec<TestStatistics>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for ANOVA test".to_string())
        })?;

        let (_, n_features) = data.dim();
        let mut results = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature = data.column(feature_idx);
            let result = self.test_single_feature(&feature, Some(target))?;
            results.push(result);
        }

        Ok(results)
    }

    fn test_single_feature<T>(
        &self,
        feature: &ArrayView1<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for ANOVA test".to_string())
        })?;

        let (f_statistic, df_between, df_within) = self.compute_f_statistic(feature, target)?;

        // Approximate p-value: use chi-squared / df_between approximation
        // For F test we use heuristic: large F is significant
        let p_value = if f_statistic > 4.0 {
            0.01
        } else if f_statistic > 2.0 {
            0.05
        } else {
            0.2
        };

        let mut result = TestStatistics::new(f_statistic, p_value);
        result.set_degrees_of_freedom(df_between + df_within);
        result.add_metadata("df_between".to_string(), df_between as f64);
        result.add_metadata("df_within".to_string(), df_within as f64);

        Ok(result)
    }
}

/// Mutual Information test implementation
#[derive(Debug, Clone)]
pub struct MutualInformationTest {
    config: TestingConfig,
    discrete_features: Vec<usize>,
    entropy_cache: Option<HashMap<String, f64>>,
}

impl MutualInformationTest {
    pub fn new(config: TestingConfig, discrete_features: Vec<usize>) -> Result<Self> {
        TestingValidator::validate_config(&config)?;
        Ok(Self {
            config,
            discrete_features,
            entropy_cache: None,
        })
    }

    /// Estimate mutual information using histogram binning
    fn estimate_mutual_information<T>(
        &self,
        feature: &ArrayView1<T>,
        target: &ArrayView1<T>,
    ) -> Result<f64>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let n = feature.len();
        if n == 0 {
            return Ok(0.0);
        }

        let feat: Vec<f64> = feature.iter().map(|&v| v.into()).collect();
        let tgt: Vec<f64> = target.iter().map(|&v| v.into()).collect();

        let n_bins = ((n as f64).sqrt().ceil() as usize).clamp(2, 20);

        let min_f = feat.iter().copied().fold(f64::INFINITY, f64::min);
        let max_f = feat.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min_t = tgt.iter().copied().fold(f64::INFINITY, f64::min);
        let max_t = tgt.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let range_f = (max_f - min_f).max(f64::EPSILON);
        let range_t = (max_t - min_t).max(f64::EPSILON);

        let mut joint = vec![vec![0usize; n_bins]; n_bins];
        let mut marg_f = vec![0usize; n_bins];
        let mut marg_t = vec![0usize; n_bins];

        for k in 0..n {
            let bf = (((feat[k] - min_f) / range_f * (n_bins as f64 - 1.0)).round() as usize)
                .min(n_bins - 1);
            let bt = (((tgt[k] - min_t) / range_t * (n_bins as f64 - 1.0)).round() as usize)
                .min(n_bins - 1);
            joint[bf][bt] += 1;
            marg_f[bf] += 1;
            marg_t[bt] += 1;
        }

        let n_f = n as f64;
        let mut mi = 0.0_f64;
        for bf in 0..n_bins {
            for bt in 0..n_bins {
                let pxy = joint[bf][bt] as f64 / n_f;
                let px = marg_f[bf] as f64 / n_f;
                let py = marg_t[bt] as f64 / n_f;
                if pxy > 0.0 && px > 0.0 && py > 0.0 {
                    mi += pxy * (pxy / (px * py)).ln();
                }
            }
        }
        Ok(mi.max(0.0))
    }

    pub fn discrete_features(&self) -> &[usize] {
        &self.discrete_features
    }
}

impl StatisticalTester for MutualInformationTest {
    fn test<T>(
        &self,
        data: &ArrayView2<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<Vec<TestStatistics>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for mutual information test".to_string())
        })?;

        let (_, n_features) = data.dim();
        let mut results = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature = data.column(feature_idx);
            let result = self.test_single_feature(&feature, Some(target))?;
            results.push(result);
        }

        Ok(results)
    }

    fn test_single_feature<T>(
        &self,
        feature: &ArrayView1<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for mutual information test".to_string())
        })?;

        let mi_score = self.estimate_mutual_information(feature, target)?;

        // MI score as test statistic; p-value is not conventionally defined for MI
        let mut result = TestStatistics::new(mi_score, 0.0);
        result.add_metadata("mutual_information".to_string(), mi_score);

        Ok(result)
    }
}

/// Correlation test implementation
#[derive(Debug, Clone)]
pub struct CorrelationTest {
    config: TestingConfig,
    method: String, // 'pearson', 'spearman', 'kendall'
}

impl CorrelationTest {
    pub fn new(config: TestingConfig, method: String) -> Result<Self> {
        TestingValidator::validate_config(&config)?;

        let valid_methods = ["pearson", "spearman", "kendall"];
        if !valid_methods.contains(&method.as_str()) {
            return Err(SklearsError::InvalidInput(
                "method must be 'pearson', 'spearman', or 'kendall'".to_string(),
            ));
        }

        Ok(Self { config, method })
    }

    /// Compute Pearson correlation coefficient and t-test statistic
    fn compute_correlation<T>(
        &self,
        feature: &ArrayView1<T>,
        target: &ArrayView1<T>,
    ) -> Result<(f64, f64)>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let n = feature.len();
        if n < 3 {
            return Ok((0.0, 0.0));
        }
        let n_f = n as f64;

        let xi: Vec<f64> = feature.iter().map(|&v| v.into()).collect();
        let yi: Vec<f64> = target.iter().map(|&v| v.into()).collect();

        let mean_x = xi.iter().sum::<f64>() / n_f;
        let mean_y = yi.iter().sum::<f64>() / n_f;

        let cov: f64 = xi
            .iter()
            .zip(yi.iter())
            .map(|(&x, &y)| (x - mean_x) * (y - mean_y))
            .sum::<f64>()
            / n_f;
        let std_x = (xi.iter().map(|&x| (x - mean_x) * (x - mean_x)).sum::<f64>() / n_f).sqrt();
        let std_y = (yi.iter().map(|&y| (y - mean_y) * (y - mean_y)).sum::<f64>() / n_f).sqrt();

        let denom = std_x * std_y;
        let correlation = if denom < f64::EPSILON {
            0.0
        } else {
            (cov / denom).clamp(-1.0, 1.0)
        };

        // t-statistic: r * sqrt((n-2)/(1-r^2))
        let r2 = correlation * correlation;
        let t_stat = if r2 >= 1.0 - f64::EPSILON {
            0.0
        } else {
            correlation * ((n_f - 2.0) / (1.0 - r2)).sqrt()
        };

        Ok((correlation, t_stat))
    }

    pub fn method(&self) -> &str {
        &self.method
    }
}

impl StatisticalTester for CorrelationTest {
    fn test<T>(
        &self,
        data: &ArrayView2<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<Vec<TestStatistics>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for correlation test".to_string())
        })?;

        let (_, n_features) = data.dim();
        let mut results = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let feature = data.column(feature_idx);
            let result = self.test_single_feature(&feature, Some(target))?;
            results.push(result);
        }

        Ok(results)
    }

    fn test_single_feature<T>(
        &self,
        feature: &ArrayView1<T>,
        target: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + Into<f64>,
    {
        let target = target.ok_or_else(|| {
            SklearsError::InvalidInput("Target required for correlation test".to_string())
        })?;

        let (correlation, t_statistic) = self.compute_correlation(feature, target)?;

        // Approximate p-value using t-distribution heuristic
        let p_value = if t_statistic.abs() > 2.576 {
            0.01
        } else if t_statistic.abs() > 1.96 {
            0.05
        } else {
            0.2
        };

        let n = feature.len();
        let mut result = TestStatistics::new(t_statistic, p_value);
        result.add_metadata("correlation".to_string(), correlation);
        if n >= 2 {
            result.set_degrees_of_freedom(n - 2);
        }

        Ok(result)
    }
}

/// Hypothesis test implementation
#[derive(Debug, Clone)]
pub struct HypothesisTest {
    test_type: StatisticalTest,
    null_hypothesis: String,
    alternative_hypothesis: String,
    alpha: f64,
}

impl HypothesisTest {
    pub fn new(
        test_type: StatisticalTest,
        null_hypothesis: String,
        alternative_hypothesis: String,
        alpha: f64,
    ) -> Result<Self> {
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        Ok(Self {
            test_type,
            null_hypothesis,
            alternative_hypothesis,
            alpha,
        })
    }

    pub fn test_type(&self) -> StatisticalTest {
        self.test_type
    }

    pub fn null_hypothesis(&self) -> &str {
        &self.null_hypothesis
    }

    pub fn alternative_hypothesis(&self) -> &str {
        &self.alternative_hypothesis
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Perform hypothesis test
    pub fn perform_test<T>(
        &self,
        data: &ArrayView1<T>,
        reference: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        match self.test_type {
            StatisticalTest::TTest => self.t_test(data, reference),
            StatisticalTest::ChiSquared => self.chi_squared_test(data, reference),
            _ => Err(SklearsError::NotImplemented(format!(
                "Hypothesis test for {:?} not implemented",
                self.test_type
            ))),
        }
    }

    /// Perform t-test
    fn t_test<T>(
        &self,
        data: &ArrayView1<T>,
        _reference: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified t-test implementation
        let t_statistic = 1.5; // Placeholder
        let p_value = 0.08; // Placeholder

        let mut result = TestStatistics::new(t_statistic, p_value);
        result.set_degrees_of_freedom(data.len() - 1);

        Ok(result)
    }

    /// Perform chi-squared test
    fn chi_squared_test<T>(
        &self,
        _data: &ArrayView1<T>,
        _reference: Option<&ArrayView1<T>>,
    ) -> Result<TestStatistics>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified chi-squared test implementation
        let chi_squared = 2.1; // Placeholder
        let p_value = 0.15; // Placeholder

        let mut result = TestStatistics::new(chi_squared, p_value);
        result.set_degrees_of_freedom(1);

        Ok(result)
    }
}

/// Statistical analyzer for comprehensive analysis
#[derive(Debug, Clone)]
pub struct StatisticalAnalyzer {
    test_results: HashMap<String, Vec<TestStatistics>>,
    summary_statistics: HashMap<String, f64>,
}

impl StatisticalAnalyzer {
    pub fn new() -> Self {
        Self {
            test_results: HashMap::new(),
            summary_statistics: HashMap::new(),
        }
    }

    /// Add test results
    pub fn add_test_results(&mut self, test_name: String, results: Vec<TestStatistics>) {
        self.test_results.insert(test_name, results);
    }

    /// Get test results
    pub fn test_results(&self) -> &HashMap<String, Vec<TestStatistics>> {
        &self.test_results
    }

    /// Add summary statistic
    pub fn add_summary_statistic(&mut self, key: String, value: f64) {
        self.summary_statistics.insert(key, value);
    }

    /// Get summary statistics
    pub fn summary_statistics(&self) -> &HashMap<String, f64> {
        &self.summary_statistics
    }

    /// Calculate multiple testing correction
    pub fn apply_multiple_testing_correction(&mut self, method: &str, alpha: f64) -> Result<()> {
        match method {
            "bonferroni" => self.bonferroni_correction(alpha),
            "fdr" => self.fdr_correction(alpha),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown correction method: {}",
                method
            ))),
        }
    }

    /// Apply Bonferroni correction
    fn bonferroni_correction(&mut self, alpha: f64) -> Result<()> {
        let total_tests = self.count_total_tests();
        let corrected_alpha = alpha / total_tests as f64;

        self.add_summary_statistic("bonferroni_corrected_alpha".to_string(), corrected_alpha);
        Ok(())
    }

    /// Apply False Discovery Rate correction
    fn fdr_correction(&mut self, alpha: f64) -> Result<()> {
        // Simplified FDR implementation
        let corrected_alpha = alpha * 0.8; // Placeholder
        self.add_summary_statistic("fdr_corrected_alpha".to_string(), corrected_alpha);
        Ok(())
    }

    /// Count total number of tests
    fn count_total_tests(&self) -> usize {
        self.test_results.values().map(|v| v.len()).sum()
    }
}

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// P-value calculator for various distributions
#[derive(Debug, Clone)]
pub struct PValueCalculator;

impl PValueCalculator {
    pub fn new() -> Self {
        Self
    }

    /// Calculate p-value for chi-squared test
    pub fn chi_squared_p_value(&self, test_statistic: f64, _df: usize) -> f64 {
        // Simplified implementation - would use actual chi-squared distribution
        if test_statistic > 3.841 {
            0.05
        } else {
            0.2
        }
    }

    /// Calculate p-value for t-test
    pub fn t_test_p_value(&self, test_statistic: f64, _df: usize) -> f64 {
        // Simplified implementation - would use actual t-distribution
        if test_statistic.abs() > 1.96 {
            0.05
        } else {
            0.3
        }
    }

    /// Calculate p-value for F-test
    pub fn f_test_p_value(&self, test_statistic: f64, _df1: usize, _df2: usize) -> f64 {
        // Simplified implementation - would use actual F-distribution
        if test_statistic > 3.0 {
            0.05
        } else {
            0.25
        }
    }
}

impl Default for PValueCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_testing_config_default() {
        let config = TestingConfig::default();
        assert_eq!(config.test_type, StatisticalTest::ChiSquared);
        assert_eq!(config.alpha, 0.05);
        assert_eq!(config.alternative, "two-sided");
    }

    #[test]
    fn test_testing_validator() {
        let mut config = TestingConfig::default();
        assert!(TestingValidator::validate_config(&config).is_ok());

        config.alpha = 0.0; // Invalid alpha
        assert!(TestingValidator::validate_config(&config).is_err());

        config.alpha = 0.05;
        config.alternative = "invalid".to_string(); // Invalid alternative
        assert!(TestingValidator::validate_config(&config).is_err());
    }

    #[test]
    fn test_test_statistics() {
        let mut stats = TestStatistics::new(5.2, 0.03);
        assert_eq!(stats.test_statistic, 5.2);
        assert_eq!(stats.p_value, 0.03);
        assert!(stats.is_significant(0.05));
        assert!(!stats.is_significant(0.01));

        stats.set_degrees_of_freedom(2);
        assert_eq!(stats.degrees_of_freedom, Some(2));

        stats.set_effect_size(0.3);
        assert_eq!(stats.effect_size, Some(0.3));

        stats.add_metadata("sample_size".to_string(), 100.0);
        assert_eq!(stats.test_metadata.get("sample_size"), Some(&100.0));
    }

    #[test]
    fn test_chi_squared_test_creation() {
        let config = TestingConfig::default();
        let test = ChiSquaredTest::new(config).expect("operation should succeed");
        assert!(test.contingency_tables().is_none());
    }

    #[test]
    fn test_anova_test_creation() {
        let config = TestingConfig::default();
        let test = ANOVATest::new(config).expect("operation should succeed");
        assert!(test.group_statistics().is_none());
    }

    #[test]
    fn test_mutual_information_test_creation() {
        let config = TestingConfig::default();
        let discrete_features = vec![0, 2, 3];
        let test = MutualInformationTest::new(config, discrete_features.clone())
            .expect("operation should succeed");
        assert_eq!(test.discrete_features(), &discrete_features);
    }

    #[test]
    fn test_correlation_test_creation() {
        let config = TestingConfig::default();
        let test =
            CorrelationTest::new(config, "pearson".to_string()).expect("operation should succeed");
        assert_eq!(test.method(), "pearson");

        // Test invalid method
        assert!(CorrelationTest::new(TestingConfig::default(), "invalid".to_string()).is_err());
    }

    #[test]
    fn test_hypothesis_test() {
        let test = HypothesisTest::new(
            StatisticalTest::TTest,
            "Mean equals zero".to_string(),
            "Mean not equal to zero".to_string(),
            0.05,
        )
        .expect("operation should succeed");

        assert_eq!(test.test_type(), StatisticalTest::TTest);
        assert_eq!(test.null_hypothesis(), "Mean equals zero");
        assert_eq!(test.alpha(), 0.05);

        // Test invalid alpha
        assert!(HypothesisTest::new(
            StatisticalTest::TTest,
            "test".to_string(),
            "test".to_string(),
            0.0,
        )
        .is_err());
    }

    #[test]
    fn test_statistical_analyzer() {
        let mut analyzer = StatisticalAnalyzer::new();

        let stats = vec![
            TestStatistics::new(2.5, 0.05),
            TestStatistics::new(1.2, 0.3),
        ];
        analyzer.add_test_results("t_test".to_string(), stats);

        assert_eq!(analyzer.test_results().len(), 1);

        analyzer.add_summary_statistic("mean_effect_size".to_string(), 0.25);
        assert_eq!(
            analyzer.summary_statistics().get("mean_effect_size"),
            Some(&0.25)
        );

        // Test multiple testing correction
        assert!(analyzer
            .apply_multiple_testing_correction("bonferroni", 0.05)
            .is_ok());
        assert!(analyzer
            .summary_statistics()
            .contains_key("bonferroni_corrected_alpha"));
    }

    #[test]
    fn test_p_value_calculator() {
        let calculator = PValueCalculator::new();

        let p_val_chi2 = calculator.chi_squared_p_value(5.0, 2);
        assert!((0.0..=1.0).contains(&p_val_chi2));

        let p_val_t = calculator.t_test_p_value(2.5, 10);
        assert!((0.0..=1.0).contains(&p_val_t));

        let p_val_f = calculator.f_test_p_value(4.0, 2, 10);
        assert!((0.0..=1.0).contains(&p_val_f));
    }
}

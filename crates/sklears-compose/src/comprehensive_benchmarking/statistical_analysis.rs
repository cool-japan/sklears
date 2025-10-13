use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};

use super::config_types::*;
use super::core_data_processing::{TransformationData, DataRecord, DataValue, ProcessingError};

/// Comprehensive statistical processing engine providing advanced statistical analysis,
/// hypothesis testing, distribution analysis, and statistical modeling capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProcessingEngine {
    /// Statistical functions registry
    statistical_functions: HashMap<String, StatisticalFunction>,
    /// Analysis algorithms
    analysis_algorithms: HashMap<String, AnalysisAlgorithm>,
    /// Model fitters for statistical modeling
    model_fitters: HashMap<String, ModelFitter>,
    /// Statistical configuration
    statistical_config: StatisticalProcessingConfiguration,
    /// Performance monitoring
    performance_monitor: Arc<RwLock<StatisticalPerformanceMonitor>>,
}

impl StatisticalProcessingEngine {
    /// Create a new statistical processing engine
    pub fn new() -> Self {
        Self {
            statistical_functions: HashMap::new(),
            analysis_algorithms: HashMap::new(),
            model_fitters: HashMap::new(),
            statistical_config: StatisticalProcessingConfiguration::default(),
            performance_monitor: Arc::new(RwLock::new(StatisticalPerformanceMonitor::new())),
        }
    }

    /// Perform statistical analysis on transformation data
    pub async fn analyze_data(&self, data: &TransformationData, analysis_config: &StatisticalAnalysisConfiguration) -> Result<StatisticalAnalysisResult, ProcessingError> {
        let start_time = Utc::now();
        let mut result = StatisticalAnalysisResult::new();

        // Descriptive statistics
        if analysis_config.include_descriptive {
            let descriptive_stats = self.compute_descriptive_statistics(data)?;
            result.set_descriptive_statistics(descriptive_stats);
        }

        // Inferential statistics
        if analysis_config.include_inferential {
            let inferential_stats = self.compute_inferential_statistics(data, &analysis_config.inferential_config)?;
            result.set_inferential_statistics(inferential_stats);
        }

        // Hypothesis testing
        if let Some(ref hypothesis_config) = analysis_config.hypothesis_testing {
            let hypothesis_results = self.perform_hypothesis_tests(data, hypothesis_config)?;
            result.set_hypothesis_test_results(hypothesis_results);
        }

        // Distribution analysis
        if analysis_config.include_distribution_analysis {
            let distribution_analysis = self.analyze_distributions(data)?;
            result.set_distribution_analysis(distribution_analysis);
        }

        // Correlation analysis
        if analysis_config.include_correlation {
            let correlation_analysis = self.compute_correlations(data)?;
            result.set_correlation_analysis(correlation_analysis);
        }

        // Time series analysis
        if let Some(ref ts_config) = analysis_config.time_series_config {
            let time_series_analysis = self.analyze_time_series(data, ts_config)?;
            result.set_time_series_analysis(time_series_analysis);
        }

        // Record performance metrics
        {
            let mut monitor = self.performance_monitor.write().unwrap();
            monitor.record_analysis_operation(
                analysis_config.analysis_name.clone(),
                Utc::now().signed_duration_since(start_time),
                data.records.len(),
            );
        }

        Ok(result)
    }

    /// Compute descriptive statistics
    fn compute_descriptive_statistics(&self, data: &TransformationData) -> Result<DescriptiveStatistics, ProcessingError> {
        let mut field_stats = HashMap::new();

        for field_name in data.schema.fields.keys() {
            let values = self.extract_numeric_values(data, field_name)?;
            if !values.is_empty() {
                let stats = self.calculate_field_statistics(&values)?;
                field_stats.insert(field_name.clone(), stats);
            }
        }

        Ok(DescriptiveStatistics {
            field_statistics: field_stats,
            overall_statistics: self.calculate_overall_statistics(data)?,
            data_summary: self.create_data_summary(data)?,
        })
    }

    /// Extract numeric values from a field
    fn extract_numeric_values(&self, data: &TransformationData, field_name: &str) -> Result<Vec<f64>, ProcessingError> {
        let mut values = Vec::new();

        for record in &data.records {
            if let Some(value) = record.fields.get(field_name) {
                match value {
                    DataValue::Float(f) => values.push(*f),
                    DataValue::Integer(i) => values.push(*i as f64),
                    _ => continue,
                }
            }
        }

        Ok(values)
    }

    /// Calculate comprehensive field statistics
    fn calculate_field_statistics(&self, values: &[f64]) -> Result<FieldStatistics, ProcessingError> {
        if values.is_empty() {
            return Err(ProcessingError::ConfigurationError("No values provided".to_string()));
        }

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
        let std_dev = variance.sqrt();

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };

        let q1_idx = sorted.len() / 4;
        let q3_idx = 3 * sorted.len() / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx.min(sorted.len() - 1)];
        let iqr = q3 - q1;

        // Calculate skewness and kurtosis
        let skewness = self.calculate_skewness(values, mean, std_dev)?;
        let kurtosis = self.calculate_kurtosis(values, mean, std_dev)?;

        Ok(FieldStatistics {
            count: values.len(),
            mean,
            median,
            mode: self.calculate_mode(values)?,
            std_dev,
            variance,
            min: sorted[0],
            max: sorted[sorted.len() - 1],
            range: sorted[sorted.len() - 1] - sorted[0],
            q1,
            q3,
            iqr,
            skewness,
            kurtosis,
            percentiles: self.calculate_percentiles(&sorted)?,
        })
    }

    /// Calculate skewness
    fn calculate_skewness(&self, values: &[f64], mean: f64, std_dev: f64) -> Result<f64, ProcessingError> {
        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let skewness = values.iter()
            .map(|x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>() / n;

        Ok(skewness)
    }

    /// Calculate kurtosis
    fn calculate_kurtosis(&self, values: &[f64], mean: f64, std_dev: f64) -> Result<f64, ProcessingError> {
        if std_dev == 0.0 {
            return Ok(0.0);
        }

        let n = values.len() as f64;
        let kurtosis = values.iter()
            .map(|x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>() / n - 3.0; // Excess kurtosis

        Ok(kurtosis)
    }

    /// Calculate mode (most frequent value)
    fn calculate_mode(&self, values: &[f64]) -> Result<Option<f64>, ProcessingError> {
        let mut frequency_map = HashMap::new();

        for &value in values {
            *frequency_map.entry(format!("{:.6}", value)).or_insert(0) += 1;
        }

        let max_count = frequency_map.values().max().unwrap_or(&0);
        if *max_count <= 1 {
            return Ok(None);
        }

        let mode_str = frequency_map.iter()
            .find(|(_, &count)| count == *max_count)
            .map(|(value, _)| value)
            .unwrap();

        Ok(Some(mode_str.parse::<f64>().unwrap_or(0.0)))
    }

    /// Calculate percentiles
    fn calculate_percentiles(&self, sorted_values: &[f64]) -> Result<HashMap<u8, f64>, ProcessingError> {
        let percentiles = vec![5, 10, 25, 50, 75, 90, 95, 99];
        let mut result = HashMap::new();

        for p in percentiles {
            let index = (p as f64 / 100.0 * (sorted_values.len() - 1) as f64) as usize;
            result.insert(p, sorted_values[index.min(sorted_values.len() - 1)]);
        }

        Ok(result)
    }

    /// Calculate overall dataset statistics
    fn calculate_overall_statistics(&self, data: &TransformationData) -> Result<OverallStatistics, ProcessingError> {
        Ok(OverallStatistics {
            total_records: data.records.len(),
            total_fields: data.schema.fields.len(),
            completeness_ratio: self.calculate_completeness_ratio(data)?,
            missing_value_count: self.count_missing_values(data)?,
        })
    }

    /// Calculate data completeness ratio
    fn calculate_completeness_ratio(&self, data: &TransformationData) -> Result<f64, ProcessingError> {
        let total_cells = data.records.len() * data.schema.fields.len();
        if total_cells == 0 {
            return Ok(1.0);
        }

        let mut filled_cells = 0;
        for record in &data.records {
            for field_name in data.schema.fields.keys() {
                if let Some(value) = record.fields.get(field_name) {
                    if !matches!(value, DataValue::Null) {
                        filled_cells += 1;
                    }
                }
            }
        }

        Ok(filled_cells as f64 / total_cells as f64)
    }

    /// Count missing values
    fn count_missing_values(&self, data: &TransformationData) -> Result<usize, ProcessingError> {
        let mut missing_count = 0;
        for record in &data.records {
            for field_name in data.schema.fields.keys() {
                if let Some(value) = record.fields.get(field_name) {
                    if matches!(value, DataValue::Null) {
                        missing_count += 1;
                    }
                } else {
                    missing_count += 1;
                }
            }
        }
        Ok(missing_count)
    }

    /// Create data summary
    fn create_data_summary(&self, data: &TransformationData) -> Result<DataSummary, ProcessingError> {
        let mut field_types = HashMap::new();
        let mut unique_counts = HashMap::new();

        for field_name in data.schema.fields.keys() {
            let mut unique_values = std::collections::HashSet::new();
            let mut field_type_counts = HashMap::new();

            for record in &data.records {
                if let Some(value) = record.fields.get(field_name) {
                    let value_type = self.get_value_type_string(value);
                    *field_type_counts.entry(value_type).or_insert(0) += 1;
                    unique_values.insert(self.value_to_string(value));
                }
            }

            // Determine predominant type
            let predominant_type = field_type_counts
                .iter()
                .max_by_key(|(_, count)| *count)
                .map(|(type_name, _)| type_name.clone())
                .unwrap_or_else(|| "unknown".to_string());

            field_types.insert(field_name.clone(), predominant_type);
            unique_counts.insert(field_name.clone(), unique_values.len());
        }

        Ok(DataSummary {
            field_types,
            unique_value_counts: unique_counts,
        })
    }

    /// Get data value type as string
    fn get_value_type_string(&self, value: &DataValue) -> String {
        match value {
            DataValue::String(_) => "string".to_string(),
            DataValue::Integer(_) => "integer".to_string(),
            DataValue::Float(_) => "float".to_string(),
            DataValue::Boolean(_) => "boolean".to_string(),
            DataValue::Date(_) => "date".to_string(),
            DataValue::Array(_) => "array".to_string(),
            DataValue::Object(_) => "object".to_string(),
            DataValue::Null => "null".to_string(),
        }
    }

    /// Convert data value to string representation
    fn value_to_string(&self, value: &DataValue) -> String {
        match value {
            DataValue::String(s) => s.clone(),
            DataValue::Integer(i) => i.to_string(),
            DataValue::Float(f) => f.to_string(),
            DataValue::Boolean(b) => b.to_string(),
            DataValue::Date(d) => d.to_string(),
            DataValue::Null => "null".to_string(),
            _ => "complex".to_string(),
        }
    }

    /// Compute inferential statistics
    fn compute_inferential_statistics(&self, data: &TransformationData, config: &InferentialStatisticsConfiguration) -> Result<InferentialStatistics, ProcessingError> {
        let mut confidence_intervals = HashMap::new();
        let mut sample_statistics = HashMap::new();

        for field_name in &config.fields_to_analyze {
            let values = self.extract_numeric_values(data, field_name)?;
            if values.len() >= config.min_sample_size {
                // Calculate confidence interval
                let ci = self.calculate_confidence_interval(&values, config.confidence_level)?;
                confidence_intervals.insert(field_name.clone(), ci);

                // Calculate sample statistics
                let sample_stats = self.calculate_sample_statistics(&values)?;
                sample_statistics.insert(field_name.clone(), sample_stats);
            }
        }

        Ok(InferentialStatistics {
            confidence_intervals,
            sample_statistics,
            effect_sizes: self.calculate_effect_sizes(data, config)?,
        })
    }

    /// Calculate confidence interval
    fn calculate_confidence_interval(&self, values: &[f64], confidence_level: f64) -> Result<ConfidenceInterval, ProcessingError> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = {
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        };

        // Use t-distribution for small samples
        let t_value = if n >= 30.0 {
            // Use normal approximation for large samples
            match confidence_level {
                0.90 => 1.645,
                0.95 => 1.96,
                0.99 => 2.576,
                _ => 1.96, // Default to 95%
            }
        } else {
            // Simplified t-value calculation (would use proper t-table in practice)
            match confidence_level {
                0.90 => 1.8,
                0.95 => 2.1,
                0.99 => 2.8,
                _ => 2.1,
            }
        };

        let margin_of_error = t_value * (std_dev / n.sqrt());

        Ok(ConfidenceInterval {
            mean,
            lower_bound: mean - margin_of_error,
            upper_bound: mean + margin_of_error,
            confidence_level,
            margin_of_error,
        })
    }

    /// Calculate sample statistics
    fn calculate_sample_statistics(&self, values: &[f64]) -> Result<SampleStatistics, ProcessingError> {
        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
        let std_dev = variance.sqrt();
        let std_error = std_dev / n.sqrt();

        Ok(SampleStatistics {
            sample_size: values.len(),
            sample_mean: mean,
            sample_variance: variance,
            sample_std_dev: std_dev,
            standard_error: std_error,
        })
    }

    /// Calculate effect sizes
    fn calculate_effect_sizes(&self, data: &TransformationData, config: &InferentialStatisticsConfiguration) -> Result<HashMap<String, f64>, ProcessingError> {
        let mut effect_sizes = HashMap::new();

        // Calculate Cohen's d for specified field pairs
        for pair in &config.effect_size_pairs {
            let values1 = self.extract_numeric_values(data, &pair.field1)?;
            let values2 = self.extract_numeric_values(data, &pair.field2)?;

            if !values1.is_empty() && !values2.is_empty() {
                let cohens_d = self.calculate_cohens_d(&values1, &values2)?;
                effect_sizes.insert(format!("{}_{}", pair.field1, pair.field2), cohens_d);
            }
        }

        Ok(effect_sizes)
    }

    /// Calculate Cohen's d effect size
    fn calculate_cohens_d(&self, group1: &[f64], group2: &[f64]) -> Result<f64, ProcessingError> {
        let mean1 = group1.iter().sum::<f64>() / group1.len() as f64;
        let mean2 = group2.iter().sum::<f64>() / group2.len() as f64;

        let var1 = group1.iter().map(|x| (x - mean1).powi(2)).sum::<f64>() / (group1.len() - 1) as f64;
        let var2 = group2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / (group2.len() - 1) as f64;

        // Pooled standard deviation
        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        if pooled_std == 0.0 {
            Ok(0.0)
        } else {
            Ok((mean1 - mean2) / pooled_std)
        }
    }

    /// Perform hypothesis tests
    fn perform_hypothesis_tests(&self, data: &TransformationData, config: &HypothesisTestConfiguration) -> Result<Vec<HypothesisTestResult>, ProcessingError> {
        let mut results = Vec::new();

        for test in &config.tests {
            let result = match &test.test_type {
                HypothesisTestType::TTest(t_config) => {
                    self.perform_t_test(data, t_config)?
                },
                HypothesisTestType::ChiSquare(chi_config) => {
                    self.perform_chi_square_test(data, chi_config)?
                },
                HypothesisTestType::ANOVA(anova_config) => {
                    self.perform_anova_test(data, anova_config)?
                },
                HypothesisTestType::KolmogorovSmirnov(ks_config) => {
                    self.perform_ks_test(data, ks_config)?
                },
            };

            results.push(result);
        }

        Ok(results)
    }

    /// Perform t-test
    fn perform_t_test(&self, data: &TransformationData, config: &TTestConfiguration) -> Result<HypothesisTestResult, ProcessingError> {
        let values = self.extract_numeric_values(data, &config.field_name)?;

        let n = values.len() as f64;
        let mean = values.iter().sum::<f64>() / n;
        let std_dev = {
            let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
            variance.sqrt()
        };

        let t_statistic = (mean - config.population_mean) / (std_dev / n.sqrt());
        let degrees_of_freedom = n - 1.0;

        // Simplified p-value calculation (would use proper statistical tables in practice)
        let p_value = self.calculate_t_test_p_value(t_statistic, degrees_of_freedom)?;

        Ok(HypothesisTestResult {
            test_name: format!("t-test_{}", config.field_name),
            test_statistic: t_statistic,
            p_value,
            degrees_of_freedom: Some(degrees_of_freedom),
            critical_value: self.get_t_critical_value(config.alpha, degrees_of_freedom)?,
            reject_null: p_value < config.alpha,
            effect_size: Some((mean - config.population_mean) / std_dev), // Cohen's d
        })
    }

    /// Calculate t-test p-value (simplified)
    fn calculate_t_test_p_value(&self, t_statistic: f64, df: f64) -> Result<f64, ProcessingError> {
        // Simplified p-value calculation using normal approximation for large df
        if df >= 30.0 {
            let z_score = t_statistic;
            Ok(2.0 * (1.0 - self.standard_normal_cdf(z_score.abs())))
        } else {
            // Very simplified approximation for small samples
            Ok(0.05) // Placeholder
        }
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + (x / 2.0_f64.sqrt()).tanh())
    }

    /// Get t-critical value
    fn get_t_critical_value(&self, alpha: f64, df: f64) -> Result<f64, ProcessingError> {
        // Simplified critical value lookup
        if df >= 30.0 {
            match alpha {
                a if a <= 0.01 => Ok(2.576),
                a if a <= 0.05 => Ok(1.96),
                a if a <= 0.10 => Ok(1.645),
                _ => Ok(1.96),
            }
        } else {
            match alpha {
                a if a <= 0.01 => Ok(2.8),
                a if a <= 0.05 => Ok(2.1),
                a if a <= 0.10 => Ok(1.8),
                _ => Ok(2.1),
            }
        }
    }

    /// Perform chi-square test
    fn perform_chi_square_test(&self, data: &TransformationData, config: &ChiSquareConfiguration) -> Result<HypothesisTestResult, ProcessingError> {
        // Simplified chi-square test implementation
        Ok(HypothesisTestResult {
            test_name: format!("chi_square_{}", config.field_name),
            test_statistic: 0.0, // Placeholder
            p_value: 0.05, // Placeholder
            degrees_of_freedom: Some(1.0),
            critical_value: 3.841, // Chi-square critical value for α=0.05, df=1
            reject_null: false,
            effect_size: None,
        })
    }

    /// Perform ANOVA test
    fn perform_anova_test(&self, data: &TransformationData, config: &ANOVAConfiguration) -> Result<HypothesisTestResult, ProcessingError> {
        // Simplified ANOVA implementation
        Ok(HypothesisTestResult {
            test_name: format!("anova_{}", config.dependent_variable),
            test_statistic: 0.0, // F-statistic placeholder
            p_value: 0.05, // Placeholder
            degrees_of_freedom: Some(2.0),
            critical_value: 3.0, // F critical value placeholder
            reject_null: false,
            effect_size: None,
        })
    }

    /// Perform Kolmogorov-Smirnov test
    fn perform_ks_test(&self, data: &TransformationData, config: &KolmogorovSmirnovConfiguration) -> Result<HypothesisTestResult, ProcessingError> {
        // Simplified KS test implementation
        Ok(HypothesisTestResult {
            test_name: format!("ks_test_{}", config.field_name),
            test_statistic: 0.0, // KS statistic placeholder
            p_value: 0.05, // Placeholder
            degrees_of_freedom: None,
            critical_value: 0.136, // KS critical value placeholder
            reject_null: false,
            effect_size: None,
        })
    }

    /// Analyze distributions
    fn analyze_distributions(&self, data: &TransformationData) -> Result<DistributionAnalysis, ProcessingError> {
        let mut field_distributions = HashMap::new();

        for field_name in data.schema.fields.keys() {
            let values = self.extract_numeric_values(data, field_name)?;
            if !values.is_empty() {
                let dist_info = self.analyze_field_distribution(&values)?;
                field_distributions.insert(field_name.clone(), dist_info);
            }
        }

        Ok(DistributionAnalysis {
            field_distributions,
            distribution_tests: self.perform_distribution_tests(data)?,
        })
    }

    /// Analyze field distribution
    fn analyze_field_distribution(&self, values: &[f64]) -> Result<DistributionInfo, ProcessingError> {
        let stats = self.calculate_field_statistics(values)?;

        // Determine likely distribution based on characteristics
        let likely_distribution = if stats.skewness.abs() < 0.5 && stats.kurtosis.abs() < 1.0 {
            DistributionType::Normal
        } else if stats.skewness > 1.0 {
            DistributionType::LogNormal
        } else if stats.min >= 0.0 && stats.skewness > 0.5 {
            DistributionType::Exponential
        } else {
            DistributionType::Unknown
        };

        Ok(DistributionInfo {
            distribution_type: likely_distribution,
            parameters: self.estimate_distribution_parameters(values, &likely_distribution)?,
            goodness_of_fit: self.calculate_goodness_of_fit(values, &likely_distribution)?,
        })
    }

    /// Estimate distribution parameters
    fn estimate_distribution_parameters(&self, values: &[f64], dist_type: &DistributionType) -> Result<HashMap<String, f64>, ProcessingError> {
        let mut parameters = HashMap::new();

        match dist_type {
            DistributionType::Normal => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
                parameters.insert("mean".to_string(), mean);
                parameters.insert("std_dev".to_string(), variance.sqrt());
            },
            DistributionType::Exponential => {
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                parameters.insert("lambda".to_string(), 1.0 / mean);
            },
            DistributionType::LogNormal => {
                let log_values: Vec<f64> = values.iter().map(|x| x.ln()).collect();
                let log_mean = log_values.iter().sum::<f64>() / log_values.len() as f64;
                let log_var = log_values.iter().map(|x| (x - log_mean).powi(2)).sum::<f64>() / log_values.len() as f64;
                parameters.insert("mu".to_string(), log_mean);
                parameters.insert("sigma".to_string(), log_var.sqrt());
            },
            _ => {
                // Unknown distribution - return basic statistics
                let mean = values.iter().sum::<f64>() / values.len() as f64;
                parameters.insert("mean".to_string(), mean);
            }
        }

        Ok(parameters)
    }

    /// Calculate goodness of fit
    fn calculate_goodness_of_fit(&self, values: &[f64], dist_type: &DistributionType) -> Result<f64, ProcessingError> {
        // Simplified goodness of fit measure (R-squared equivalent)
        // In practice, would use KS test or Anderson-Darling test
        match dist_type {
            DistributionType::Normal => Ok(0.85), // Placeholder
            DistributionType::Exponential => Ok(0.75), // Placeholder
            DistributionType::LogNormal => Ok(0.80), // Placeholder
            _ => Ok(0.50), // Unknown distributions have poor fit
        }
    }

    /// Perform distribution tests
    fn perform_distribution_tests(&self, data: &TransformationData) -> Result<HashMap<String, DistributionTestResult>, ProcessingError> {
        let mut test_results = HashMap::new();

        for field_name in data.schema.fields.keys() {
            let values = self.extract_numeric_values(data, field_name)?;
            if values.len() >= 30 { // Minimum sample size for distribution tests
                let normality_test = self.test_normality(&values)?;
                test_results.insert(field_name.clone(), normality_test);
            }
        }

        Ok(test_results)
    }

    /// Test for normality
    fn test_normality(&self, values: &[f64]) -> Result<DistributionTestResult, ProcessingError> {
        let stats = self.calculate_field_statistics(values)?;

        // Simplified normality test based on skewness and kurtosis
        let skewness_test = stats.skewness.abs() < 0.5;
        let kurtosis_test = stats.kurtosis.abs() < 1.0;
        let is_normal = skewness_test && kurtosis_test;

        Ok(DistributionTestResult {
            test_name: "Normality Test".to_string(),
            test_statistic: stats.skewness.abs() + stats.kurtosis.abs(),
            p_value: if is_normal { 0.1 } else { 0.01 },
            is_significant: !is_normal,
        })
    }

    /// Compute correlations
    fn compute_correlations(&self, data: &TransformationData) -> Result<CorrelationAnalysis, ProcessingError> {
        let numeric_fields: Vec<String> = data.schema.fields.keys()
            .filter(|field_name| {
                let values = self.extract_numeric_values(data, field_name).unwrap_or_default();
                !values.is_empty()
            })
            .cloned()
            .collect();

        let mut correlation_matrix = HashMap::new();
        let mut correlation_tests = HashMap::new();

        for i in 0..numeric_fields.len() {
            for j in (i+1)..numeric_fields.len() {
                let field1 = &numeric_fields[i];
                let field2 = &numeric_fields[j];

                let values1 = self.extract_numeric_values(data, field1)?;
                let values2 = self.extract_numeric_values(data, field2)?;

                if values1.len() == values2.len() && !values1.is_empty() {
                    let correlation = self.calculate_correlation(&values1, &values2)?;
                    let correlation_test = self.test_correlation_significance(&values1, &values2, correlation)?;

                    correlation_matrix.insert(format!("{}_{}", field1, field2), correlation);
                    correlation_tests.insert(format!("{}_{}", field1, field2), correlation_test);
                }
            }
        }

        Ok(CorrelationAnalysis {
            correlation_matrix,
            correlation_tests,
            partial_correlations: HashMap::new(), // Would implement partial correlations
        })
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &[f64], y: &[f64]) -> Result<f64, ProcessingError> {
        if x.len() != y.len() || x.is_empty() {
            return Err(ProcessingError::ConfigurationError("Invalid data for correlation".to_string()));
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let sum_sq_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        let denominator = (sum_sq_x * sum_sq_y).sqrt();

        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Test correlation significance
    fn test_correlation_significance(&self, x: &[f64], y: &[f64], correlation: f64) -> Result<CorrelationTestResult, ProcessingError> {
        let n = x.len() as f64;
        let degrees_of_freedom = n - 2.0;

        // Calculate t-statistic for correlation
        let t_statistic = correlation * ((n - 2.0) / (1.0 - correlation.powi(2))).sqrt();

        // Simplified p-value calculation
        let p_value = self.calculate_t_test_p_value(t_statistic, degrees_of_freedom)?;

        Ok(CorrelationTestResult {
            correlation,
            t_statistic,
            p_value,
            degrees_of_freedom,
            is_significant: p_value < 0.05,
        })
    }

    /// Analyze time series
    fn analyze_time_series(&self, data: &TransformationData, config: &TimeSeriesConfiguration) -> Result<TimeSeriesAnalysis, ProcessingError> {
        // Extract time series data
        let time_series = self.extract_time_series_data(data, config)?;

        Ok(TimeSeriesAnalysis {
            trend_analysis: self.analyze_trend(&time_series)?,
            seasonality_analysis: self.analyze_seasonality(&time_series)?,
            stationarity_test: self.test_stationarity(&time_series)?,
            autocorrelation: self.calculate_autocorrelation(&time_series)?,
        })
    }

    /// Extract time series data
    fn extract_time_series_data(&self, data: &TransformationData, config: &TimeSeriesConfiguration) -> Result<Vec<TimeSeriesPoint>, ProcessingError> {
        let mut time_series = Vec::new();

        for record in &data.records {
            if let (Some(timestamp), Some(value)) = (
                record.fields.get(&config.timestamp_field),
                record.fields.get(&config.value_field)
            ) {
                if let (DataValue::Date(ts), DataValue::Float(val)) = (timestamp, value) {
                    time_series.push(TimeSeriesPoint {
                        timestamp: *ts,
                        value: *val,
                    });
                }
            }
        }

        // Sort by timestamp
        time_series.sort_by_key(|point| point.timestamp);
        Ok(time_series)
    }

    /// Analyze trend
    fn analyze_trend(&self, time_series: &[TimeSeriesPoint]) -> Result<TrendAnalysis, ProcessingError> {
        if time_series.len() < 2 {
            return Ok(TrendAnalysis {
                trend_direction: TrendDirection::None,
                trend_strength: 0.0,
                linear_regression: None,
            });
        }

        // Simple linear regression for trend
        let n = time_series.len() as f64;
        let x_values: Vec<f64> = (0..time_series.len()).map(|i| i as f64).collect();
        let y_values: Vec<f64> = time_series.iter().map(|p| p.value).collect();

        let slope = self.calculate_correlation(&x_values, &y_values)? *
                   (self.calculate_std_dev(&y_values)? / self.calculate_std_dev(&x_values)?);

        let trend_direction = if slope > 0.1 {
            TrendDirection::Increasing
        } else if slope < -0.1 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        };

        Ok(TrendAnalysis {
            trend_direction,
            trend_strength: slope.abs(),
            linear_regression: Some(LinearRegressionResult {
                slope,
                intercept: y_values.iter().sum::<f64>() / n - slope * (x_values.iter().sum::<f64>() / n),
                r_squared: self.calculate_correlation(&x_values, &y_values)?.powi(2),
            }),
        })
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, values: &[f64]) -> Result<f64, ProcessingError> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        Ok(variance.sqrt())
    }

    /// Analyze seasonality
    fn analyze_seasonality(&self, time_series: &[TimeSeriesPoint]) -> Result<SeasonalityAnalysis, ProcessingError> {
        // Simplified seasonality detection
        Ok(SeasonalityAnalysis {
            has_seasonality: false, // Placeholder
            seasonal_period: None,
            seasonal_strength: 0.0,
        })
    }

    /// Test stationarity
    fn test_stationarity(&self, time_series: &[TimeSeriesPoint]) -> Result<StationarityTest, ProcessingError> {
        // Simplified stationarity test (Augmented Dickey-Fuller would be used in practice)
        let values: Vec<f64> = time_series.iter().map(|p| p.value).collect();
        let stats = self.calculate_field_statistics(&values)?;

        Ok(StationarityTest {
            test_name: "Simplified Stationarity Test".to_string(),
            test_statistic: stats.variance, // Placeholder
            p_value: 0.05, // Placeholder
            is_stationary: stats.variance < 1.0, // Simplified criterion
        })
    }

    /// Calculate autocorrelation
    fn calculate_autocorrelation(&self, time_series: &[TimeSeriesPoint]) -> Result<Vec<AutocorrelationLag>, ProcessingError> {
        let values: Vec<f64> = time_series.iter().map(|p| p.value).collect();
        let mut autocorrelations = Vec::new();

        let max_lags = (values.len() / 4).min(20); // Limit to reasonable number of lags

        for lag in 1..=max_lags {
            if lag < values.len() {
                let x: Vec<f64> = values[..values.len()-lag].to_vec();
                let y: Vec<f64> = values[lag..].to_vec();
                let correlation = self.calculate_correlation(&x, &y)?;

                autocorrelations.push(AutocorrelationLag {
                    lag,
                    correlation,
                    is_significant: correlation.abs() > 0.2, // Simplified significance test
                });
            }
        }

        Ok(autocorrelations)
    }

    /// Register statistical function
    pub fn register_statistical_function(&mut self, function_id: String, function: StatisticalFunction) {
        self.statistical_functions.insert(function_id, function);
    }

    /// Register analysis algorithm
    pub fn register_analysis_algorithm(&mut self, algorithm_id: String, algorithm: AnalysisAlgorithm) {
        self.analysis_algorithms.insert(algorithm_id, algorithm);
    }

    /// Register model fitter
    pub fn register_model_fitter(&mut self, fitter_id: String, fitter: ModelFitter) {
        self.model_fitters.insert(fitter_id, fitter);
    }
}

// Supporting data structures and configurations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalProcessingConfiguration {
    pub default_confidence_level: f64,
    pub default_alpha: f64,
    pub min_sample_size: usize,
    pub enable_parallel_processing: bool,
}

impl Default for StatisticalProcessingConfiguration {
    fn default() -> Self {
        Self {
            default_confidence_level: 0.95,
            default_alpha: 0.05,
            min_sample_size: 30,
            enable_parallel_processing: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisConfiguration {
    pub analysis_name: String,
    pub include_descriptive: bool,
    pub include_inferential: bool,
    pub include_distribution_analysis: bool,
    pub include_correlation: bool,
    pub inferential_config: InferentialStatisticsConfiguration,
    pub hypothesis_testing: Option<HypothesisTestConfiguration>,
    pub time_series_config: Option<TimeSeriesConfiguration>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferentialStatisticsConfiguration {
    pub fields_to_analyze: Vec<String>,
    pub confidence_level: f64,
    pub min_sample_size: usize,
    pub effect_size_pairs: Vec<FieldPair>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldPair {
    pub field1: String,
    pub field2: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestConfiguration {
    pub tests: Vec<HypothesisTest>,
    pub multiple_comparison_correction: Option<MultipleComparisonCorrection>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTest {
    pub test_name: String,
    pub test_type: HypothesisTestType,
    pub alpha: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HypothesisTestType {
    TTest(TTestConfiguration),
    ChiSquare(ChiSquareConfiguration),
    ANOVA(ANOVAConfiguration),
    KolmogorovSmirnov(KolmogorovSmirnovConfiguration),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestConfiguration {
    pub field_name: String,
    pub population_mean: f64,
    pub alpha: f64,
    pub test_type: TTestType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TTestType {
    OneSample,
    TwoSample,
    Paired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareConfiguration {
    pub field_name: String,
    pub expected_frequencies: Vec<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ANOVAConfiguration {
    pub dependent_variable: String,
    pub independent_variables: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KolmogorovSmirnovConfiguration {
    pub field_name: String,
    pub reference_distribution: DistributionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MultipleComparisonCorrection {
    Bonferroni,
    BenjaminiHochberg,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesConfiguration {
    pub timestamp_field: String,
    pub value_field: String,
    pub analysis_window: Option<Duration>,
}

// Result structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalAnalysisResult {
    pub analysis_timestamp: DateTime<Utc>,
    pub descriptive_statistics: Option<DescriptiveStatistics>,
    pub inferential_statistics: Option<InferentialStatistics>,
    pub hypothesis_test_results: Option<Vec<HypothesisTestResult>>,
    pub distribution_analysis: Option<DistributionAnalysis>,
    pub correlation_analysis: Option<CorrelationAnalysis>,
    pub time_series_analysis: Option<TimeSeriesAnalysis>,
}

impl StatisticalAnalysisResult {
    pub fn new() -> Self {
        Self {
            analysis_timestamp: Utc::now(),
            descriptive_statistics: None,
            inferential_statistics: None,
            hypothesis_test_results: None,
            distribution_analysis: None,
            correlation_analysis: None,
            time_series_analysis: None,
        }
    }

    pub fn set_descriptive_statistics(&mut self, stats: DescriptiveStatistics) {
        self.descriptive_statistics = Some(stats);
    }

    pub fn set_inferential_statistics(&mut self, stats: InferentialStatistics) {
        self.inferential_statistics = Some(stats);
    }

    pub fn set_hypothesis_test_results(&mut self, results: Vec<HypothesisTestResult>) {
        self.hypothesis_test_results = Some(results);
    }

    pub fn set_distribution_analysis(&mut self, analysis: DistributionAnalysis) {
        self.distribution_analysis = Some(analysis);
    }

    pub fn set_correlation_analysis(&mut self, analysis: CorrelationAnalysis) {
        self.correlation_analysis = Some(analysis);
    }

    pub fn set_time_series_analysis(&mut self, analysis: TimeSeriesAnalysis) {
        self.time_series_analysis = Some(analysis);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescriptiveStatistics {
    pub field_statistics: HashMap<String, FieldStatistics>,
    pub overall_statistics: OverallStatistics,
    pub data_summary: DataSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldStatistics {
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub mode: Option<f64>,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: HashMap<u8, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverallStatistics {
    pub total_records: usize,
    pub total_fields: usize,
    pub completeness_ratio: f64,
    pub missing_value_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSummary {
    pub field_types: HashMap<String, String>,
    pub unique_value_counts: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferentialStatistics {
    pub confidence_intervals: HashMap<String, ConfidenceInterval>,
    pub sample_statistics: HashMap<String, SampleStatistics>,
    pub effect_sizes: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub mean: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub confidence_level: f64,
    pub margin_of_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleStatistics {
    pub sample_size: usize,
    pub sample_mean: f64,
    pub sample_variance: f64,
    pub sample_std_dev: f64,
    pub standard_error: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HypothesisTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: Option<f64>,
    pub critical_value: f64,
    pub reject_null: bool,
    pub effect_size: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    pub field_distributions: HashMap<String, DistributionInfo>,
    pub distribution_tests: HashMap<String, DistributionTestResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionInfo {
    pub distribution_type: DistributionType,
    pub parameters: HashMap<String, f64>,
    pub goodness_of_fit: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DistributionType {
    Normal,
    LogNormal,
    Exponential,
    Uniform,
    Poisson,
    Binomial,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionTestResult {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationAnalysis {
    pub correlation_matrix: HashMap<String, f64>,
    pub correlation_tests: HashMap<String, CorrelationTestResult>,
    pub partial_correlations: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationTestResult {
    pub correlation: f64,
    pub t_statistic: f64,
    pub p_value: f64,
    pub degrees_of_freedom: f64,
    pub is_significant: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesAnalysis {
    pub trend_analysis: TrendAnalysis,
    pub seasonality_analysis: SeasonalityAnalysis,
    pub stationarity_test: StationarityTest,
    pub autocorrelation: Vec<AutocorrelationLag>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub linear_regression: Option<LinearRegressionResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SeasonalityAnalysis {
    pub has_seasonality: bool,
    pub seasonal_period: Option<Duration>,
    pub seasonal_strength: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StationarityTest {
    pub test_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub is_stationary: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutocorrelationLag {
    pub lag: usize,
    pub correlation: f64,
    pub is_significant: bool,
}

// Abstract types for extension

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalFunction {
    pub function_id: String,
    pub function_name: String,
    pub parameters: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisAlgorithm {
    pub algorithm_id: String,
    pub algorithm_name: String,
    pub configuration: HashMap<String, DataValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFitter {
    pub fitter_id: String,
    pub model_type: String,
    pub fitting_configuration: ModelFittingConfiguration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFittingConfiguration {
    pub optimization_method: String,
    pub convergence_criteria: ConvergenceCriteria,
    pub parameter_constraints: HashMap<String, ParameterConstraint>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceCriteria {
    pub max_iterations: u32,
    pub tolerance: f64,
    pub relative_tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterConstraint {
    pub min_value: Option<f64>,
    pub max_value: Option<f64>,
    pub fixed_value: Option<f64>,
}

// Performance monitoring

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalPerformanceMonitor {
    operation_metrics: Vec<StatisticalOperationMetric>,
    performance_stats: StatisticalPerformanceStats,
}

impl StatisticalPerformanceMonitor {
    pub fn new() -> Self {
        Self {
            operation_metrics: Vec::new(),
            performance_stats: StatisticalPerformanceStats::default(),
        }
    }

    pub fn record_analysis_operation(&mut self, analysis_name: String, duration: Duration, records_processed: usize) {
        let metric = StatisticalOperationMetric {
            analysis_name,
            start_time: Utc::now() - duration,
            duration,
            records_processed,
        };

        self.operation_metrics.push(metric);
        self.update_performance_stats();

        if self.operation_metrics.len() > 1000 {
            self.operation_metrics.drain(0..500);
        }
    }

    fn update_performance_stats(&mut self) {
        if self.operation_metrics.is_empty() {
            return;
        }

        let total_operations = self.operation_metrics.len();
        let total_duration: Duration = self.operation_metrics.iter().map(|m| m.duration).sum();
        let total_records: usize = self.operation_metrics.iter().map(|m| m.records_processed).sum();

        self.performance_stats = StatisticalPerformanceStats {
            total_analyses: total_operations,
            average_duration: total_duration / total_operations as u32,
            total_records_analyzed: total_records,
            records_per_second: if total_duration.as_secs() > 0 {
                total_records as f64 / total_duration.as_secs() as f64
            } else {
                0.0
            },
        };
    }

    pub fn get_performance_stats(&self) -> &StatisticalPerformanceStats {
        &self.performance_stats
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalOperationMetric {
    pub analysis_name: String,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub records_processed: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalPerformanceStats {
    pub total_analyses: usize,
    pub average_duration: Duration,
    pub total_records_analyzed: usize,
    pub records_per_second: f64,
}

impl Default for StatisticalPerformanceStats {
    fn default() -> Self {
        Self {
            total_analyses: 0,
            average_duration: Duration::from_millis(0),
            total_records_analyzed: 0,
            records_per_second: 0.0,
        }
    }
}

// Error handling

#[derive(Debug, thiserror::Error)]
pub enum StatisticalError {
    #[error("Insufficient data: {0}")]
    InsufficientData(String),

    #[error("Statistical computation failed: {0}")]
    ComputationFailed(String),

    #[error("Invalid statistical parameters: {0}")]
    InvalidParameters(String),

    #[error("Distribution fitting failed: {0}")]
    DistributionFittingFailed(String),

    #[error("Hypothesis test failed: {0}")]
    HypothesisTestFailed(String),

    #[error("Time series analysis failed: {0}")]
    TimeSeriesAnalysisFailed(String),

    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

pub type StatisticalResult<T> = Result<T, StatisticalError>;
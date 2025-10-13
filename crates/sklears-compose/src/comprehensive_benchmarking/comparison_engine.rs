//! Comparison Engine for Benchmark Analysis
//!
//! This module provides comprehensive comparison capabilities including baseline comparisons,
//! statistical testing, effect size calculations, and comparison reporting.

use super::config_types::*;
use super::performance_analysis::{StatisticalSummary, DescriptiveStatistics};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use serde::{Serialize, Deserialize};

// ================================================================================================
// CORE COMPARISON ENGINE
// ================================================================================================

/// Comparison engine for benchmarking comparisons
pub struct ComparisonEngine {
    comparison_strategies: Vec<ComparisonStrategy>,
    baseline_manager: BaselineManager,
    statistical_comparisons: StatisticalComparisons,
    visualization_generator: VisualizationGenerator,
    engine_config: ComparisonEngineConfig,
    comparison_cache: ComparisonCache,
}

impl ComparisonEngine {
    /// Create a new comparison engine
    pub fn new(config: ComparisonEngineConfig) -> Self {
        Self {
            comparison_strategies: vec![
                ComparisonStrategy::BaselineComparison,
                ComparisonStrategy::PairwiseComparison,
                ComparisonStrategy::DistributionComparison,
            ],
            baseline_manager: BaselineManager::new(),
            statistical_comparisons: StatisticalComparisons::new(),
            visualization_generator: VisualizationGenerator::new(),
            engine_config: config,
            comparison_cache: ComparisonCache::new(),
        }
    }

    /// Compare benchmark results with baseline
    pub fn compare_with_baseline(&self, results: &[BenchmarkResult], baseline_name: &str) -> Result<ComparisonReport, ComparisonError> {
        let baseline = self.baseline_manager.get_baseline(baseline_name)
            .ok_or_else(|| ComparisonError::BaselineNotFound(baseline_name.to_string()))?;

        // Check cache first
        let cache_key = self.generate_cache_key(results, baseline_name);
        if let Some(cached_report) = self.comparison_cache.get(&cache_key) {
            return Ok(cached_report.clone());
        }

        let comparison_summary = self.generate_comparison_summary(results, &baseline)?;
        let detailed_comparisons = self.perform_detailed_comparisons(results, &baseline)?;
        let statistical_results = self.perform_statistical_comparisons(results, &baseline)?;
        let recommendations = self.generate_comparison_recommendations(results, &baseline)?;
        let visualizations = self.generate_comparison_visualizations(results, &baseline)?;

        let report = ComparisonReport {
            comparison_id: self.generate_comparison_id(),
            baseline_name: baseline_name.to_string(),
            comparison_timestamp: SystemTime::now(),
            comparison_summary,
            detailed_comparisons,
            statistical_results,
            recommendations,
            visualizations,
            confidence_score: self.calculate_comparison_confidence(results, &baseline)?,
            metadata: self.generate_comparison_metadata(results, &baseline),
        };

        // Cache the result
        self.comparison_cache.insert(cache_key, report.clone());

        Ok(report)
    }

    /// Perform pairwise comparison between two result sets
    pub fn compare_pairwise(&self, baseline_results: &[BenchmarkResult], comparison_results: &[BenchmarkResult]) -> Result<PairwiseComparisonReport, ComparisonError> {
        if baseline_results.is_empty() || comparison_results.is_empty() {
            return Err(ComparisonError::InsufficientData("Both result sets must contain data".to_string()));
        }

        let statistical_comparison = self.perform_pairwise_statistical_tests(baseline_results, comparison_results)?;
        let effect_size_analysis = self.calculate_pairwise_effect_sizes(baseline_results, comparison_results)?;
        let distribution_comparison = self.compare_distributions(baseline_results, comparison_results)?;
        let trend_comparison = self.compare_trends(baseline_results, comparison_results)?;

        Ok(PairwiseComparisonReport {
            comparison_id: self.generate_comparison_id(),
            baseline_count: baseline_results.len(),
            comparison_count: comparison_results.len(),
            statistical_comparison,
            effect_size_analysis,
            distribution_comparison,
            trend_comparison,
            overall_assessment: self.assess_pairwise_comparison(&statistical_comparison, &effect_size_analysis)?,
            comparison_timestamp: SystemTime::now(),
        })
    }

    /// Perform multiple comparison analysis
    pub fn compare_multiple(&self, result_groups: &HashMap<String, Vec<BenchmarkResult>>) -> Result<MultipleComparisonReport, ComparisonError> {
        if result_groups.len() < 2 {
            return Err(ComparisonError::InsufficientData("Need at least 2 groups for multiple comparison".to_string()));
        }

        let pairwise_comparisons = self.perform_all_pairwise_comparisons(result_groups)?;
        let anova_results = self.perform_anova_analysis(result_groups)?;
        let post_hoc_tests = self.perform_post_hoc_tests(result_groups)?;
        let multiple_testing_correction = self.apply_multiple_testing_correction(&pairwise_comparisons)?;

        Ok(MultipleComparisonReport {
            comparison_id: self.generate_comparison_id(),
            group_names: result_groups.keys().cloned().collect(),
            pairwise_comparisons,
            anova_results,
            post_hoc_tests,
            multiple_testing_correction,
            overall_ranking: self.rank_groups(result_groups)?,
            comparison_timestamp: SystemTime::now(),
        })
    }

    /// Compare distributions between result sets
    pub fn compare_distributions(&self, baseline_results: &[BenchmarkResult], comparison_results: &[BenchmarkResult]) -> Result<DistributionComparisonReport, ComparisonError> {
        let available_metrics = self.get_common_metrics(baseline_results, comparison_results);
        let mut metric_comparisons = HashMap::new();

        for metric_name in available_metrics {
            let baseline_values = self.extract_metric_values(baseline_results, &metric_name);
            let comparison_values = self.extract_metric_values(comparison_results, &metric_name);

            if !baseline_values.is_empty() && !comparison_values.is_empty() {
                let comparison = self.compare_metric_distributions(&baseline_values, &comparison_values)?;
                metric_comparisons.insert(metric_name, comparison);
            }
        }

        Ok(DistributionComparisonReport {
            comparison_id: self.generate_comparison_id(),
            metric_comparisons,
            overall_similarity: self.calculate_overall_distribution_similarity(&metric_comparisons),
            comparison_timestamp: SystemTime::now(),
        })
    }

    /// Add a new baseline
    pub fn add_baseline(&mut self, baseline: Baseline) -> Result<(), ComparisonError> {
        self.baseline_manager.add_baseline(baseline)
    }

    /// Update an existing baseline
    pub fn update_baseline(&mut self, baseline_name: &str, new_results: &[BenchmarkResult]) -> Result<(), ComparisonError> {
        self.baseline_manager.update_baseline(baseline_name, new_results)
    }

    /// List available baselines
    pub fn list_baselines(&self) -> Vec<String> {
        self.baseline_manager.list_baselines()
    }

    /// Get baseline information
    pub fn get_baseline_info(&self, baseline_name: &str) -> Option<BaselineInfo> {
        self.baseline_manager.get_baseline_info(baseline_name)
    }

    // Private helper methods
    fn generate_comparison_summary(&self, results: &[BenchmarkResult], baseline: &Baseline) -> Result<ComparisonSummary, ComparisonError> {
        let mut metric_comparisons = HashMap::new();
        let mut total_improvement = 0.0;
        let mut compared_metrics = 0;
        let mut significant_changes = 0;

        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if let Some(baseline_metric) = baseline.metrics.get(metric_name) {
                    let improvement = (baseline_metric.baseline_value - metric_value) / baseline_metric.baseline_value;
                    let is_significant = improvement.abs() > baseline_metric.acceptable_range.tolerance_percentage;

                    metric_comparisons.insert(
                        metric_name.clone(),
                        MetricComparison {
                            metric_name: metric_name.clone(),
                            current_value: *metric_value,
                            baseline_value: baseline_metric.baseline_value,
                            improvement_percentage: improvement * 100.0,
                            significance: if is_significant {
                                ComparisonSignificance::Significant
                            } else {
                                ComparisonSignificance::NotSignificant
                            },
                            relative_change: improvement,
                            absolute_change: baseline_metric.baseline_value - metric_value,
                        }
                    );

                    total_improvement += improvement;
                    compared_metrics += 1;

                    if is_significant {
                        significant_changes += 1;
                    }
                }
            }
        }

        let overall_improvement = if compared_metrics > 0 {
            total_improvement / compared_metrics as f64
        } else {
            0.0
        };

        Ok(ComparisonSummary {
            overall_improvement: overall_improvement * 100.0,
            metric_comparisons,
            total_metrics_compared: compared_metrics,
            significant_changes,
            improvement_distribution: self.calculate_improvement_distribution(&metric_comparisons),
            statistical_summary: self.calculate_comparison_statistics(&metric_comparisons)?,
        })
    }

    fn perform_detailed_comparisons(&self, results: &[BenchmarkResult], baseline: &Baseline) -> Result<HashMap<String, DetailedComparison>, ComparisonError> {
        let mut detailed_comparisons = HashMap::new();

        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if let Some(baseline_metric) = baseline.metrics.get(metric_name) {
                    let current_stats = StatisticalInfo {
                        sample_count: 1,
                        mean: *metric_value,
                        median: *metric_value,
                        standard_deviation: 0.0,
                        variance: 0.0,
                        min_value: *metric_value,
                        max_value: *metric_value,
                        confidence_interval: ConfidenceInterval {
                            confidence_level: 0.95,
                            lower_bound: *metric_value,
                            upper_bound: *metric_value,
                        },
                    };

                    let baseline_stats = StatisticalInfo {
                        sample_count: baseline_metric.sample_size,
                        mean: baseline_metric.baseline_value,
                        median: baseline_metric.baseline_value,
                        standard_deviation: 0.0, // Would be calculated from baseline data
                        variance: 0.0,
                        min_value: baseline_metric.baseline_value,
                        max_value: baseline_metric.baseline_value,
                        confidence_interval: ConfidenceInterval {
                            confidence_level: baseline_metric.confidence_level,
                            lower_bound: baseline_metric.acceptable_range.lower_bound.unwrap_or(0.0),
                            upper_bound: baseline_metric.acceptable_range.upper_bound.unwrap_or(f64::INFINITY),
                        },
                    };

                    let effect_size = self.calculate_effect_size(*metric_value, baseline_metric.baseline_value)?;
                    let practical_significance = self.assess_practical_significance(&effect_size, &baseline_metric.acceptable_range);

                    let detailed_comparison = DetailedComparison {
                        metric_name: metric_name.clone(),
                        current_statistics: current_stats,
                        baseline_statistics: baseline_stats,
                        effect_size,
                        confidence_interval: ConfidenceInterval {
                            confidence_level: 0.95,
                            lower_bound: effect_size - 0.1, // Simplified
                            upper_bound: effect_size + 0.1,
                        },
                        practical_significance,
                        quality_assessment: self.assess_comparison_quality(result, baseline_metric),
                    };

                    detailed_comparisons.insert(metric_name.clone(), detailed_comparison);
                }
            }
        }

        Ok(detailed_comparisons)
    }

    fn perform_statistical_comparisons(&self, results: &[BenchmarkResult], baseline: &Baseline) -> Result<Vec<StatisticalTestResult>, ComparisonError> {
        let mut test_results = Vec::new();

        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if let Some(baseline_metric) = baseline.metrics.get(metric_name) {
                    // Perform one-sample t-test (simplified)
                    let test_statistic = (*metric_value - baseline_metric.baseline_value) /
                        (baseline_metric.baseline_value * 0.1); // Simplified standard error

                    let p_value = self.calculate_p_value(test_statistic)?;
                    let is_significant = p_value < self.engine_config.significance_level;
                    let effect_size = self.calculate_effect_size(*metric_value, baseline_metric.baseline_value)?;

                    test_results.push(StatisticalTestResult {
                        test_name: "One-sample t-test".to_string(),
                        metric_name: metric_name.clone(),
                        test_statistic,
                        p_value,
                        significance_level: self.engine_config.significance_level,
                        is_significant,
                        effect_size,
                        confidence_interval: ConfidenceInterval {
                            confidence_level: 0.95,
                            lower_bound: *metric_value - 1.96 * 0.1, // Simplified
                            upper_bound: *metric_value + 1.96 * 0.1,
                        },
                        test_power: self.calculate_test_power(effect_size, baseline_metric.sample_size as f64)?,
                        assumptions_met: self.check_test_assumptions(result, baseline_metric)?,
                    });
                }
            }
        }

        Ok(test_results)
    }

    fn generate_comparison_recommendations(&self, results: &[BenchmarkResult], baseline: &Baseline) -> Result<Vec<ComparisonRecommendation>, ComparisonError> {
        let mut recommendations = Vec::new();

        for result in results {
            for (metric_name, metric_value) in &result.metrics {
                if let Some(baseline_metric) = baseline.metrics.get(metric_name) {
                    let degradation = (*metric_value - baseline_metric.baseline_value) / baseline_metric.baseline_value;

                    if degradation > baseline_metric.acceptable_range.warning_threshold {
                        let priority = if degradation > baseline_metric.acceptable_range.tolerance_percentage {
                            RecommendationPriority::High
                        } else {
                            RecommendationPriority::Medium
                        };

                        recommendations.push(ComparisonRecommendation {
                            recommendation_id: self.generate_recommendation_id(),
                            recommendation_type: ComparisonRecommendationType::PerformanceDegradation,
                            priority,
                            metric_name: metric_name.clone(),
                            current_value: *metric_value,
                            baseline_value: baseline_metric.baseline_value,
                            degradation_percentage: degradation * 100.0,
                            suggested_actions: vec![
                                format!("Investigate performance degradation in {}", metric_name),
                                "Profile the affected benchmark".to_string(),
                                "Check for environmental changes".to_string(),
                                "Consider updating baseline if improvement is expected".to_string(),
                            ],
                            expected_impact: ExpectedImpact {
                                performance_improvement: -degradation,
                                cost_impact: CostImpact::Medium,
                                implementation_effort: ImplementationEffort::Medium,
                                risk_level: RiskLevel::Medium,
                                time_to_implement: Duration::from_secs(86400 * 2),
                            },
                            confidence: 0.8,
                        });
                    } else if degradation < -0.1 { // Significant improvement
                        recommendations.push(ComparisonRecommendation {
                            recommendation_id: self.generate_recommendation_id(),
                            recommendation_type: ComparisonRecommendationType::PerformanceImprovement,
                            priority: RecommendationPriority::Low,
                            metric_name: metric_name.clone(),
                            current_value: *metric_value,
                            baseline_value: baseline_metric.baseline_value,
                            degradation_percentage: degradation * 100.0,
                            suggested_actions: vec![
                                "Consider updating baseline with improved performance".to_string(),
                                "Document the improvement for future reference".to_string(),
                                "Investigate what caused the improvement".to_string(),
                            ],
                            expected_impact: ExpectedImpact {
                                performance_improvement: -degradation,
                                cost_impact: CostImpact::Low,
                                implementation_effort: ImplementationEffort::Low,
                                risk_level: RiskLevel::VeryLow,
                                time_to_implement: Duration::from_secs(3600),
                            },
                            confidence: 0.9,
                        });
                    }
                }
            }
        }

        Ok(recommendations)
    }

    fn calculate_effect_size(&self, current_value: f64, baseline_value: f64) -> Result<f64, ComparisonError> {
        if baseline_value == 0.0 {
            return Err(ComparisonError::DivisionByZero("Cannot calculate effect size with zero baseline".to_string()));
        }
        Ok((current_value - baseline_value) / baseline_value)
    }

    fn calculate_p_value(&self, test_statistic: f64) -> Result<f64, ComparisonError> {
        // Simplified p-value calculation using normal approximation
        let abs_t = test_statistic.abs();
        if abs_t < 1.96 {
            Ok(0.05)
        } else if abs_t < 2.58 {
            Ok(0.01)
        } else {
            Ok(0.001)
        }
    }

    fn extract_metric_values(&self, results: &[BenchmarkResult], metric_name: &str) -> Vec<f64> {
        results.iter()
            .filter_map(|result| result.metrics.get(metric_name))
            .cloned()
            .collect()
    }

    fn get_common_metrics(&self, baseline_results: &[BenchmarkResult], comparison_results: &[BenchmarkResult]) -> Vec<String> {
        let baseline_metrics: std::collections::HashSet<String> = baseline_results.iter()
            .flat_map(|result| result.metrics.keys().cloned())
            .collect();

        let comparison_metrics: std::collections::HashSet<String> = comparison_results.iter()
            .flat_map(|result| result.metrics.keys().cloned())
            .collect();

        baseline_metrics.intersection(&comparison_metrics).cloned().collect()
    }

    fn generate_comparison_id(&self) -> String {
        format!("comp_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis())
    }

    fn generate_recommendation_id(&self) -> String {
        format!("rec_{}", SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis())
    }

    fn generate_cache_key(&self, results: &[BenchmarkResult], baseline_name: &str) -> String {
        format!("{}_{}_{}",
            baseline_name,
            results.len(),
            results.iter().map(|r| &r.result_id).collect::<Vec<_>>().join("_")
        )
    }
}

/// Configuration for the comparison engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonEngineConfig {
    pub significance_level: f64,
    pub confidence_level: f64,
    pub effect_size_thresholds: EffectSizeThresholds,
    pub enable_multiple_testing_correction: bool,
    pub cache_size: usize,
    pub visualization_enabled: bool,
}

impl Default for ComparisonEngineConfig {
    fn default() -> Self {
        Self {
            significance_level: 0.05,
            confidence_level: 0.95,
            effect_size_thresholds: EffectSizeThresholds::default(),
            enable_multiple_testing_correction: true,
            cache_size: 1000,
            visualization_enabled: true,
        }
    }
}

/// Effect size thresholds for interpretation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeThresholds {
    pub small: f64,
    pub medium: f64,
    pub large: f64,
}

impl Default for EffectSizeThresholds {
    fn default() -> Self {
        Self {
            small: 0.2,
            medium: 0.5,
            large: 0.8,
        }
    }
}

impl Default for ComparisonEngine {
    fn default() -> Self {
        Self::new(ComparisonEngineConfig::default())
    }
}

// ================================================================================================
// COMPARISON STRATEGIES
// ================================================================================================

/// Comparison strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonStrategy {
    BaselineComparison,
    PairwiseComparison,
    MultipleComparison,
    TrendComparison,
    DistributionComparison,
    Custom(String),
}

// ================================================================================================
// BASELINE MANAGEMENT
// ================================================================================================

/// Baseline manager for maintaining comparison baselines
pub struct BaselineManager {
    baselines: HashMap<String, Baseline>,
    update_policies: HashMap<String, BaselineUpdatePolicy>,
    validation_rules: Vec<BaselineValidationRule>,
}

impl BaselineManager {
    /// Create a new baseline manager
    pub fn new() -> Self {
        Self {
            baselines: HashMap::new(),
            update_policies: HashMap::new(),
            validation_rules: Vec::new(),
        }
    }

    /// Add a new baseline
    pub fn add_baseline(&mut self, baseline: Baseline) -> Result<(), ComparisonError> {
        // Validate baseline first
        self.validate_baseline(&baseline)?;

        self.baselines.insert(baseline.baseline_id.clone(), baseline);
        Ok(())
    }

    /// Get a baseline by name
    pub fn get_baseline(&self, baseline_name: &str) -> Option<&Baseline> {
        self.baselines.get(baseline_name)
    }

    /// Update an existing baseline
    pub fn update_baseline(&mut self, baseline_name: &str, new_results: &[BenchmarkResult]) -> Result<(), ComparisonError> {
        if let Some(baseline) = self.baselines.get_mut(baseline_name) {
            // Update baseline metrics with new results
            for result in new_results {
                for (metric_name, metric_value) in &result.metrics {
                    if let Some(baseline_metric) = baseline.metrics.get_mut(metric_name) {
                        // Simple update - would be more sophisticated in real implementation
                        baseline_metric.baseline_value = *metric_value;
                        baseline_metric.sample_size += 1;
                    } else {
                        // Add new metric to baseline
                        baseline.metrics.insert(metric_name.clone(), MetricBaseline {
                            metric_name: metric_name.clone(),
                            baseline_value: *metric_value,
                            acceptable_range: AcceptableRange::default(),
                            trend_info: None,
                            confidence_level: 0.95,
                            sample_size: 1,
                            tolerance: 0.1,
                            last_updated: SystemTime::now(),
                        });
                    }
                }
            }
            baseline.last_updated = SystemTime::now();
            Ok(())
        } else {
            Err(ComparisonError::BaselineNotFound(baseline_name.to_string()))
        }
    }

    /// List all available baselines
    pub fn list_baselines(&self) -> Vec<String> {
        self.baselines.keys().cloned().collect()
    }

    /// Get baseline information
    pub fn get_baseline_info(&self, baseline_name: &str) -> Option<BaselineInfo> {
        self.baselines.get(baseline_name).map(|baseline| {
            BaselineInfo {
                baseline_id: baseline.baseline_id.clone(),
                name: baseline.name.clone(),
                description: baseline.description.clone(),
                baseline_type: baseline.baseline_type.clone(),
                metric_count: baseline.metrics.len(),
                creation_time: baseline.creation_time,
                last_updated: baseline.last_updated,
                version: baseline.version.clone(),
            }
        })
    }

    /// Validate a baseline
    fn validate_baseline(&self, baseline: &Baseline) -> Result<(), ComparisonError> {
        if baseline.baseline_id.is_empty() {
            return Err(ComparisonError::ValidationError("Baseline ID cannot be empty".to_string()));
        }

        if baseline.metrics.is_empty() {
            return Err(ComparisonError::ValidationError("Baseline must contain at least one metric".to_string()));
        }

        for validation_rule in &self.validation_rules {
            self.apply_validation_rule(baseline, validation_rule)?;
        }

        Ok(())
    }

    fn apply_validation_rule(&self, baseline: &Baseline, rule: &BaselineValidationRule) -> Result<(), ComparisonError> {
        // Simplified validation rule application
        match rule.validation_type {
            BaselineValidationType::DataQuality => {
                // Check data quality
                for metric in baseline.metrics.values() {
                    if metric.baseline_value.is_nan() || metric.baseline_value.is_infinite() {
                        return Err(ComparisonError::ValidationError(
                            format!("Invalid baseline value for metric {}", metric.metric_name)
                        ));
                    }
                }
            },
            _ => {
                // Other validation types would be implemented here
            }
        }
        Ok(())
    }
}

impl Default for BaselineManager {
    fn default() -> Self {
        Self::new()
    }
}

/// Baseline for comparisons
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Baseline {
    pub baseline_id: String,
    pub name: String,
    pub description: String,
    pub baseline_type: BaselineType,
    pub metrics: HashMap<String, MetricBaseline>,
    pub creation_time: SystemTime,
    pub last_updated: SystemTime,
    pub version: String,
    pub metadata: HashMap<String, String>,
}

/// Metric baseline information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricBaseline {
    pub metric_name: String,
    pub baseline_value: f64,
    pub acceptable_range: AcceptableRange,
    pub trend_info: Option<TrendInfo>,
    pub confidence_level: f64,
    pub sample_size: u32,
    pub tolerance: f64,
    pub last_updated: SystemTime,
}

/// Acceptable range for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcceptableRange {
    pub lower_bound: Option<f64>,
    pub upper_bound: Option<f64>,
    pub tolerance_percentage: f64,
    pub warning_threshold: f64,
}

impl Default for AcceptableRange {
    fn default() -> Self {
        Self {
            lower_bound: None,
            upper_bound: None,
            tolerance_percentage: 0.1, // 10%
            warning_threshold: 0.05,   // 5%
        }
    }
}

/// Trend information for baseline metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendInfo {
    pub trend_direction: TrendDirection,
    pub trend_strength: f64,
    pub trend_confidence: f64,
    pub trend_start_time: SystemTime,
}

/// Baseline validation rules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineValidationRule {
    pub rule_name: String,
    pub validation_type: BaselineValidationType,
    pub condition: String,
    pub action: ValidationAction,
}

/// Baseline validation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BaselineValidationType {
    DataQuality,
    StatisticalSignificance,
    TrendConsistency,
    OutlierDetection,
    Custom(String),
}

/// Validation actions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ValidationAction {
    Accept,
    Reject,
    Flag,
    RequireReview,
    Custom(String),
}

/// Baseline information summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineInfo {
    pub baseline_id: String,
    pub name: String,
    pub description: String,
    pub baseline_type: BaselineType,
    pub metric_count: usize,
    pub creation_time: SystemTime,
    pub last_updated: SystemTime,
    pub version: String,
}

// ================================================================================================
// STATISTICAL COMPARISONS
// ================================================================================================

/// Statistical comparisons for benchmarks
pub struct StatisticalComparisons {
    comparison_tests: Vec<ComparisonTest>,
    effect_size_calculations: EffectSizeCalculations,
    power_analysis: PowerAnalysisMethods,
    multiple_testing_correction: MultipleTesting,
}

impl StatisticalComparisons {
    /// Create new statistical comparisons suite
    pub fn new() -> Self {
        Self {
            comparison_tests: vec![
                ComparisonTest {
                    test_name: "Two-sample t-test".to_string(),
                    test_type: ComparisonTestType::TwoSampleTTest,
                    assumptions: vec![
                        TestAssumption::Normality,
                        TestAssumption::EqualVariances,
                        TestAssumption::Independence,
                    ],
                    significance_level: 0.05,
                    power_requirement: 0.8,
                },
                ComparisonTest {
                    test_name: "Mann-Whitney U test".to_string(),
                    test_type: ComparisonTestType::MannWhitneyU,
                    assumptions: vec![
                        TestAssumption::Independence,
                        TestAssumption::RandomSampling,
                    ],
                    significance_level: 0.05,
                    power_requirement: 0.8,
                },
            ],
            effect_size_calculations: EffectSizeCalculations::new(),
            power_analysis: PowerAnalysisMethods::new(),
            multiple_testing_correction: MultipleTesting::new(),
        }
    }
}

impl Default for StatisticalComparisons {
    fn default() -> Self {
        Self::new()
    }
}

/// Comparison tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonTest {
    pub test_name: String,
    pub test_type: ComparisonTestType,
    pub assumptions: Vec<TestAssumption>,
    pub significance_level: f64,
    pub power_requirement: f64,
}

/// Comparison test types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonTestType {
    TwoSampleTTest,
    PairedTTest,
    MannWhitneyU,
    WilcoxonSignedRank,
    OneWayANOVA,
    KruskalWallis,
    Custom(String),
}

/// Test assumptions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestAssumption {
    Normality,
    EqualVariances,
    Independence,
    RandomSampling,
    Custom(String),
}

/// Supporting structures (simplified implementations)
pub struct EffectSizeCalculations;
pub struct PowerAnalysisMethods;
pub struct MultipleTesting;
pub struct VisualizationGenerator;

impl EffectSizeCalculations {
    pub fn new() -> Self { Self }
}

impl PowerAnalysisMethods {
    pub fn new() -> Self { Self }
}

impl MultipleTesting {
    pub fn new() -> Self { Self }
}

impl VisualizationGenerator {
    pub fn new() -> Self { Self }
}

// ================================================================================================
// COMPARISON REPORTS AND RESULTS
// ================================================================================================

/// Comprehensive comparison report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonReport {
    pub comparison_id: String,
    pub baseline_name: String,
    pub comparison_timestamp: SystemTime,
    pub comparison_summary: ComparisonSummary,
    pub detailed_comparisons: HashMap<String, DetailedComparison>,
    pub statistical_results: Vec<StatisticalTestResult>,
    pub recommendations: Vec<ComparisonRecommendation>,
    pub visualizations: Vec<VisualizationData>,
    pub confidence_score: f64,
    pub metadata: ComparisonMetadata,
}

/// Comparison summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonSummary {
    pub overall_improvement: f64,
    pub metric_comparisons: HashMap<String, MetricComparison>,
    pub total_metrics_compared: usize,
    pub significant_changes: usize,
    pub improvement_distribution: ImprovementDistribution,
    pub statistical_summary: ComparisonStatistics,
}

/// Metric comparison details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricComparison {
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub improvement_percentage: f64,
    pub significance: ComparisonSignificance,
    pub relative_change: f64,
    pub absolute_change: f64,
}

/// Comparison significance levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonSignificance {
    Significant,
    NotSignificant,
    Unknown,
}

/// Detailed comparison for individual metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedComparison {
    pub metric_name: String,
    pub current_statistics: StatisticalInfo,
    pub baseline_statistics: StatisticalInfo,
    pub effect_size: f64,
    pub confidence_interval: ConfidenceInterval,
    pub practical_significance: PracticalSignificance,
    pub quality_assessment: ComparisonQuality,
}

/// Statistical test result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTestResult {
    pub test_name: String,
    pub metric_name: String,
    pub test_statistic: f64,
    pub p_value: f64,
    pub significance_level: f64,
    pub is_significant: bool,
    pub effect_size: f64,
    pub confidence_interval: ConfidenceInterval,
    pub test_power: f64,
    pub assumptions_met: Vec<AssumptionCheck>,
}

/// Statistical information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalInfo {
    pub sample_count: u32,
    pub mean: f64,
    pub median: f64,
    pub standard_deviation: f64,
    pub variance: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub confidence_interval: ConfidenceInterval,
}

/// Confidence interval
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceInterval {
    pub confidence_level: f64,
    pub lower_bound: f64,
    pub upper_bound: f64,
}

/// Comparison recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonRecommendation {
    pub recommendation_id: String,
    pub recommendation_type: ComparisonRecommendationType,
    pub priority: RecommendationPriority,
    pub metric_name: String,
    pub current_value: f64,
    pub baseline_value: f64,
    pub degradation_percentage: f64,
    pub suggested_actions: Vec<String>,
    pub expected_impact: ExpectedImpact,
    pub confidence: f64,
}

/// Types of comparison recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonRecommendationType {
    PerformanceDegradation,
    PerformanceImprovement,
    InvestigateAnomaly,
    UpdateBaseline,
    Custom(String),
}

/// Supporting structures for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseComparisonReport {
    pub comparison_id: String,
    pub baseline_count: usize,
    pub comparison_count: usize,
    pub statistical_comparison: StatisticalComparison,
    pub effect_size_analysis: EffectSizeAnalysis,
    pub distribution_comparison: DistributionComparison,
    pub trend_comparison: TrendComparison,
    pub overall_assessment: OverallAssessment,
    pub comparison_timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleComparisonReport {
    pub comparison_id: String,
    pub group_names: Vec<String>,
    pub pairwise_comparisons: HashMap<String, PairwiseResult>,
    pub anova_results: ANOVAResult,
    pub post_hoc_tests: Vec<PostHocTestResult>,
    pub multiple_testing_correction: MultipleTestingCorrection,
    pub overall_ranking: Vec<GroupRanking>,
    pub comparison_timestamp: SystemTime,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparisonReport {
    pub comparison_id: String,
    pub metric_comparisons: HashMap<String, MetricDistributionComparison>,
    pub overall_similarity: f64,
    pub comparison_timestamp: SystemTime,
}

/// Supporting types (simplified)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImprovementDistribution {
    pub improvements: usize,
    pub degradations: usize,
    pub no_change: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonStatistics {
    pub mean_improvement: f64,
    pub median_improvement: f64,
    pub improvement_variance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PracticalSignificance {
    pub is_practically_significant: bool,
    pub magnitude: EffectMagnitude,
    pub business_impact: BusinessImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EffectMagnitude {
    Negligible,
    Small,
    Medium,
    Large,
    VeryLarge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BusinessImpact {
    pub impact_level: ImpactLevel,
    pub cost_implications: CostImplications,
    pub user_experience_impact: UserExperienceImpact,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostImplications {
    pub resource_cost_change: f64,
    pub operational_cost_change: f64,
    pub maintenance_cost_change: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserExperienceImpact {
    pub perceived_performance: f64,
    pub satisfaction_impact: f64,
    pub usability_impact: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonQuality {
    pub data_quality_score: f64,
    pub sample_size_adequacy: SampleSizeAdequacy,
    pub measurement_reliability: f64,
    pub external_validity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SampleSizeAdequacy {
    Adequate,
    Marginal,
    Inadequate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssumptionCheck {
    pub assumption: TestAssumption,
    pub is_met: bool,
    pub confidence: f64,
    pub test_statistic: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonMetadata {
    pub engine_version: String,
    pub comparison_parameters: HashMap<String, String>,
    pub data_sources: Vec<String>,
    pub analysis_duration: Duration,
}

// Placeholder implementations for complex structures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalComparison;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EffectSizeAnalysis;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionComparison;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendComparison;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairwiseResult;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ANOVAResult;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostHocTestResult;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultipleTestingCorrection;
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupRanking {
    pub group_name: String,
    pub rank: usize,
    pub score: f64,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDistributionComparison;

// ================================================================================================
// COMPARISON CACHE
// ================================================================================================

/// Cache for comparison results
pub struct ComparisonCache {
    cache: HashMap<String, ComparisonReport>,
    max_size: usize,
}

impl ComparisonCache {
    pub fn new() -> Self {
        Self {
            cache: HashMap::new(),
            max_size: 1000,
        }
    }

    pub fn get(&self, key: &str) -> Option<&ComparisonReport> {
        self.cache.get(key)
    }

    pub fn insert(&mut self, key: String, value: ComparisonReport) {
        if self.cache.len() >= self.max_size {
            // Simple LRU eviction
            if let Some(first_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&first_key);
            }
        }
        self.cache.insert(key, value);
    }
}

// ================================================================================================
// ERRORS
// ================================================================================================

/// Comparison errors
#[derive(Debug, thiserror::Error)]
pub enum ComparisonError {
    #[error("Baseline not found: {0}")]
    BaselineNotFound(String),
    #[error("Insufficient data: {0}")]
    InsufficientData(String),
    #[error("Division by zero: {0}")]
    DivisionByZero(String),
    #[error("Validation error: {0}")]
    ValidationError(String),
    #[error("Statistical error: {0}")]
    StatisticalError(String),
    #[error("Configuration error: {0}")]
    ConfigurationError(String),
    #[error("Cache error: {0}")]
    CacheError(String),
}

// ================================================================================================
// TRAIT IMPLEMENTATIONS AND UTILITIES
// ================================================================================================

// Helper implementations for the comparison engine
impl ComparisonEngine {
    fn generate_comparison_visualizations(&self, _results: &[BenchmarkResult], _baseline: &Baseline) -> Result<Vec<VisualizationData>, ComparisonError> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn calculate_comparison_confidence(&self, _results: &[BenchmarkResult], _baseline: &Baseline) -> Result<f64, ComparisonError> {
        // Simplified confidence calculation
        Ok(0.85)
    }

    fn generate_comparison_metadata(&self, results: &[BenchmarkResult], baseline: &Baseline) -> ComparisonMetadata {
        ComparisonMetadata {
            engine_version: "1.0.0".to_string(),
            comparison_parameters: HashMap::new(),
            data_sources: vec![
                format!("Results: {} benchmarks", results.len()),
                format!("Baseline: {}", baseline.name),
            ],
            analysis_duration: Duration::from_millis(100),
        }
    }

    fn calculate_improvement_distribution(&self, comparisons: &HashMap<String, MetricComparison>) -> ImprovementDistribution {
        let mut improvements = 0;
        let mut degradations = 0;
        let mut no_change = 0;

        for comparison in comparisons.values() {
            if comparison.improvement_percentage > 1.0 {
                improvements += 1;
            } else if comparison.improvement_percentage < -1.0 {
                degradations += 1;
            } else {
                no_change += 1;
            }
        }

        ImprovementDistribution {
            improvements,
            degradations,
            no_change,
        }
    }

    fn calculate_comparison_statistics(&self, comparisons: &HashMap<String, MetricComparison>) -> Result<ComparisonStatistics, ComparisonError> {
        if comparisons.is_empty() {
            return Ok(ComparisonStatistics {
                mean_improvement: 0.0,
                median_improvement: 0.0,
                improvement_variance: 0.0,
            });
        }

        let improvements: Vec<f64> = comparisons.values()
            .map(|c| c.improvement_percentage)
            .collect();

        let mean = improvements.iter().sum::<f64>() / improvements.len() as f64;

        let mut sorted_improvements = improvements.clone();
        sorted_improvements.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted_improvements[sorted_improvements.len() / 2];

        let variance = improvements.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / improvements.len() as f64;

        Ok(ComparisonStatistics {
            mean_improvement: mean,
            median_improvement: median,
            improvement_variance: variance,
        })
    }

    fn assess_practical_significance(&self, effect_size: &f64, _acceptable_range: &AcceptableRange) -> PracticalSignificance {
        let magnitude = if effect_size.abs() < 0.2 {
            EffectMagnitude::Negligible
        } else if effect_size.abs() < 0.5 {
            EffectMagnitude::Small
        } else if effect_size.abs() < 0.8 {
            EffectMagnitude::Medium
        } else {
            EffectMagnitude::Large
        };

        PracticalSignificance {
            is_practically_significant: effect_size.abs() > 0.2,
            magnitude,
            business_impact: BusinessImpact {
                impact_level: ImpactLevel::Medium,
                cost_implications: CostImplications {
                    resource_cost_change: effect_size * 0.1,
                    operational_cost_change: effect_size * 0.05,
                    maintenance_cost_change: effect_size * 0.02,
                },
                user_experience_impact: UserExperienceImpact {
                    perceived_performance: effect_size * 0.7,
                    satisfaction_impact: effect_size * 0.5,
                    usability_impact: effect_size * 0.3,
                },
            },
        }
    }

    fn assess_comparison_quality(&self, _result: &BenchmarkResult, _baseline_metric: &MetricBaseline) -> ComparisonQuality {
        ComparisonQuality {
            data_quality_score: 0.85,
            sample_size_adequacy: SampleSizeAdequacy::Adequate,
            measurement_reliability: 0.9,
            external_validity: 0.8,
        }
    }

    fn calculate_test_power(&self, _effect_size: f64, _sample_size: f64) -> Result<f64, ComparisonError> {
        // Simplified power calculation
        Ok(0.8)
    }

    fn check_test_assumptions(&self, _result: &BenchmarkResult, _baseline_metric: &MetricBaseline) -> Result<Vec<AssumptionCheck>, ComparisonError> {
        Ok(vec![
            AssumptionCheck {
                assumption: TestAssumption::Normality,
                is_met: true,
                confidence: 0.9,
                test_statistic: Some(0.5),
            },
            AssumptionCheck {
                assumption: TestAssumption::Independence,
                is_met: true,
                confidence: 0.95,
                test_statistic: None,
            },
        ])
    }

    // Placeholder implementations for complex methods
    fn perform_pairwise_statistical_tests(&self, _baseline: &[BenchmarkResult], _comparison: &[BenchmarkResult]) -> Result<StatisticalComparison, ComparisonError> {
        Ok(StatisticalComparison)
    }

    fn calculate_pairwise_effect_sizes(&self, _baseline: &[BenchmarkResult], _comparison: &[BenchmarkResult]) -> Result<EffectSizeAnalysis, ComparisonError> {
        Ok(EffectSizeAnalysis)
    }

    fn compare_trends(&self, _baseline: &[BenchmarkResult], _comparison: &[BenchmarkResult]) -> Result<TrendComparison, ComparisonError> {
        Ok(TrendComparison)
    }

    fn assess_pairwise_comparison(&self, _statistical: &StatisticalComparison, _effect_size: &EffectSizeAnalysis) -> Result<OverallAssessment, ComparisonError> {
        Ok(OverallAssessment::Good)
    }

    fn perform_all_pairwise_comparisons(&self, _groups: &HashMap<String, Vec<BenchmarkResult>>) -> Result<HashMap<String, PairwiseResult>, ComparisonError> {
        Ok(HashMap::new())
    }

    fn perform_anova_analysis(&self, _groups: &HashMap<String, Vec<BenchmarkResult>>) -> Result<ANOVAResult, ComparisonError> {
        Ok(ANOVAResult)
    }

    fn perform_post_hoc_tests(&self, _groups: &HashMap<String, Vec<BenchmarkResult>>) -> Result<Vec<PostHocTestResult>, ComparisonError> {
        Ok(Vec::new())
    }

    fn apply_multiple_testing_correction(&self, _comparisons: &HashMap<String, PairwiseResult>) -> Result<MultipleTestingCorrection, ComparisonError> {
        Ok(MultipleTestingCorrection)
    }

    fn rank_groups(&self, groups: &HashMap<String, Vec<BenchmarkResult>>) -> Result<Vec<GroupRanking>, ComparisonError> {
        let mut rankings = Vec::new();
        for (i, group_name) in groups.keys().enumerate() {
            rankings.push(GroupRanking {
                group_name: group_name.clone(),
                rank: i + 1,
                score: 1.0 / (i + 1) as f64,
            });
        }
        Ok(rankings)
    }

    fn compare_metric_distributions(&self, _baseline: &[f64], _comparison: &[f64]) -> Result<MetricDistributionComparison, ComparisonError> {
        Ok(MetricDistributionComparison)
    }

    fn calculate_overall_distribution_similarity(&self, _comparisons: &HashMap<String, MetricDistributionComparison>) -> f64 {
        0.8 // Placeholder
    }
}

// ================================================================================================
// TESTS
// ================================================================================================

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comparison_engine_creation() {
        let config = ComparisonEngineConfig::default();
        let engine = ComparisonEngine::new(config);
        assert_eq!(engine.comparison_strategies.len(), 3);
    }

    #[test]
    fn test_baseline_manager() {
        let mut manager = BaselineManager::new();

        let baseline = Baseline {
            baseline_id: "test_baseline".to_string(),
            name: "Test Baseline".to_string(),
            description: "A test baseline".to_string(),
            baseline_type: BaselineType::Historical,
            metrics: HashMap::new(),
            creation_time: SystemTime::now(),
            last_updated: SystemTime::now(),
            version: "1.0".to_string(),
            metadata: HashMap::new(),
        };

        assert!(manager.add_baseline(baseline).is_ok());
        assert!(manager.get_baseline("test_baseline").is_some());
        assert_eq!(manager.list_baselines().len(), 1);
    }

    #[test]
    fn test_effect_size_calculation() {
        let config = ComparisonEngineConfig::default();
        let engine = ComparisonEngine::new(config);

        let effect_size = engine.calculate_effect_size(110.0, 100.0).unwrap();
        assert_eq!(effect_size, 0.1);

        let negative_effect = engine.calculate_effect_size(90.0, 100.0).unwrap();
        assert_eq!(negative_effect, -0.1);
    }

    #[test]
    fn test_comparison_cache() {
        let mut cache = ComparisonCache::new();
        assert!(cache.get("test_key").is_none());
        assert_eq!(cache.max_size, 1000);
    }

    #[test]
    fn test_acceptable_range_default() {
        let range = AcceptableRange::default();
        assert_eq!(range.tolerance_percentage, 0.1);
        assert_eq!(range.warning_threshold, 0.05);
    }

    #[test]
    fn test_effect_size_thresholds() {
        let thresholds = EffectSizeThresholds::default();
        assert_eq!(thresholds.small, 0.2);
        assert_eq!(thresholds.medium, 0.5);
        assert_eq!(thresholds.large, 0.8);
    }

    #[test]
    fn test_baseline_validation() {
        let manager = BaselineManager::new();

        let invalid_baseline = Baseline {
            baseline_id: "".to_string(), // Invalid: empty ID
            name: "Test".to_string(),
            description: "Test".to_string(),
            baseline_type: BaselineType::Historical,
            metrics: HashMap::new(), // Invalid: no metrics
            creation_time: SystemTime::now(),
            last_updated: SystemTime::now(),
            version: "1.0".to_string(),
            metadata: HashMap::new(),
        };

        assert!(manager.validate_baseline(&invalid_baseline).is_err());
    }
}
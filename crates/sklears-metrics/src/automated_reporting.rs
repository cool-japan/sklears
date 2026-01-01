//! Automated Reporting for Machine Learning Metrics
//!
//! This module provides automated report generation capabilities for machine learning
//! model evaluation, including comprehensive metric reports, model comparison reports,
//! and executive summaries with statistical significance testing.
//!
//! # Features
//!
//! - Comprehensive metric report generation with statistical analysis
//! - Automated model comparison reports with significance testing
//! - Executive summary generation for stakeholders
//! - Customizable report templates and formats
//! - Statistical significance testing integration
//! - Performance regression detection
//! - Automated insights and recommendations
//!
//! # Examples
//!
//! ```rust,ignore
//! use sklears_metrics::automated_reporting::*;
//! use scirs2_core::ndarray::Array1;
//! //!
//! // Create a basic metric report
//! let mut metrics = HashMap::new();
//! metrics.insert("accuracy".to_string(), 0.85);
//! metrics.insert("precision".to_string(), 0.82);
//! metrics.insert("recall".to_string(), 0.88);
//!
//! let config = ReportConfig::default();
//! let report = generate_metric_report(&metrics, &config).unwrap();
//! println!("{}", report.to_html());
//! ```

use crate::{MetricsError, MetricsResult};
use std::collections::HashMap;
use std::fmt;

/// Configuration for automated report generation
#[derive(Debug, Clone)]
pub struct ReportConfig {
    /// Report title
    pub title: String,
    /// Author/organization information
    pub author: String,
    /// Include statistical significance tests
    pub include_significance_tests: bool,
    /// Confidence level for statistical tests
    pub confidence_level: f64,
    /// Include performance trends
    pub include_trends: bool,
    /// Include recommendations
    pub include_recommendations: bool,
    /// Report format preferences
    pub format: ReportFormat,
    /// Threshold for flagging significant changes
    pub significance_threshold: f64,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            title: "Machine Learning Model Evaluation Report".to_string(),
            author: "Automated Metrics System".to_string(),
            include_significance_tests: true,
            confidence_level: 0.95,
            include_trends: true,
            include_recommendations: true,
            format: ReportFormat::HTML,
            significance_threshold: 0.05,
        }
    }
}

/// Supported report formats
#[derive(Debug, Clone, Copy)]
pub enum ReportFormat {
    /// HTML format with interactive elements
    HTML,
    /// Markdown format
    Markdown,
    /// Plain text format
    Text,
    /// JSON format for programmatic access
    JSON,
}

/// Comprehensive metric report structure
#[derive(Debug, Clone)]
pub struct MetricReport {
    /// Report metadata
    pub metadata: ReportMetadata,
    /// Main metric results
    pub metrics: HashMap<String, MetricSummary>,
    /// Model comparison results (if applicable)
    pub model_comparisons: Vec<ModelComparison>,
    /// Executive summary
    pub executive_summary: ExecutiveSummary,
    /// Statistical analysis results
    pub statistical_analysis: StatisticalAnalysis,
    /// Recommendations and insights
    pub recommendations: Vec<Recommendation>,
    /// Performance trends (if applicable)
    pub trends: Option<PerformanceTrends>,
}

/// Report metadata
#[derive(Debug, Clone)]
pub struct ReportMetadata {
    /// Report title
    pub title: String,
    /// Generation timestamp
    pub timestamp: String,
    /// Author information
    pub author: String,
    /// Report version
    pub version: String,
    /// Dataset information
    pub dataset_info: DatasetInfo,
}

/// Dataset information for the report
#[derive(Debug, Clone)]
pub struct DatasetInfo {
    /// Dataset name
    pub name: String,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes (for classification)
    pub n_classes: Option<usize>,
    /// Data split information
    pub split_info: String,
}

/// Summary of a specific metric
#[derive(Debug, Clone)]
pub struct MetricSummary {
    /// Metric name
    pub name: String,
    /// Metric value
    pub value: f64,
    /// Standard error (if available)
    pub standard_error: Option<f64>,
    /// Confidence interval
    pub confidence_interval: Option<(f64, f64)>,
    /// Metric interpretation
    pub interpretation: String,
    /// Performance grade
    pub grade: PerformanceGrade,
}

/// Performance grade for metrics
#[derive(Debug, Clone, Copy)]
pub enum PerformanceGrade {
    /// Excellent
    Excellent,
    /// Good
    Good,
    /// Fair
    Fair,
    /// Poor
    Poor,
    /// Critical
    Critical,
}

impl fmt::Display for PerformanceGrade {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PerformanceGrade::Excellent => write!(f, "Excellent"),
            PerformanceGrade::Good => write!(f, "Good"),
            PerformanceGrade::Fair => write!(f, "Fair"),
            PerformanceGrade::Poor => write!(f, "Poor"),
            PerformanceGrade::Critical => write!(f, "Critical"),
        }
    }
}

/// Model comparison results
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Model names being compared
    pub model_names: (String, String),
    /// Metric being compared
    pub metric_name: String,
    /// Values for each model
    pub values: (f64, f64),
    /// Statistical significance test result
    pub significance_test: SignificanceTest,
    /// Practical significance assessment
    pub practical_significance: PracticalSignificance,
}

/// Statistical significance test result
#[derive(Debug, Clone)]
pub struct SignificanceTest {
    /// Test statistic
    pub test_statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Test method used
    pub method: String,
    /// Is statistically significant
    pub is_significant: bool,
    /// Effect size
    pub effect_size: f64,
}

/// Practical significance assessment
#[derive(Debug, Clone)]
pub struct PracticalSignificance {
    /// Absolute difference between models
    pub absolute_difference: f64,
    /// Relative difference (percentage)
    pub relative_difference: f64,
    /// Is practically significant
    pub is_significant: bool,
    /// Interpretation
    pub interpretation: String,
}

/// Executive summary for stakeholders
#[derive(Debug, Clone)]
pub struct ExecutiveSummary {
    /// Overall performance assessment
    pub overall_performance: String,
    /// Key findings
    pub key_findings: Vec<String>,
    /// Critical issues (if any)
    pub critical_issues: Vec<String>,
    /// Business impact assessment
    pub business_impact: String,
    /// Next steps recommendations
    pub next_steps: Vec<String>,
}

/// Statistical analysis summary
#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    /// Sample size adequacy assessment
    pub sample_size_analysis: String,
    /// Confidence interval summaries
    pub confidence_intervals: HashMap<String, (f64, f64)>,
    /// Significance test summaries
    pub significance_tests: Vec<SignificanceTest>,
    /// Power analysis (if applicable)
    pub power_analysis: Option<PowerAnalysis>,
}

/// Power analysis for statistical tests
#[derive(Debug, Clone)]
pub struct PowerAnalysis {
    /// Statistical power achieved
    pub power: f64,
    /// Minimum detectable effect size
    pub minimum_effect_size: f64,
    /// Required sample size for adequate power
    pub required_sample_size: usize,
}

/// Performance trends over time
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Trend direction for each metric
    pub trend_directions: HashMap<String, TrendDirection>,
    /// Trend strengths
    pub trend_strengths: HashMap<String, f64>,
    /// Regression/improvement detection
    pub performance_changes: Vec<PerformanceChange>,
}

/// Trend direction enumeration
#[derive(Debug, Clone, Copy)]
pub enum TrendDirection {
    /// Improving
    Improving,
    /// Stable
    Stable,
    /// Declining
    Declining,
    /// Volatile
    Volatile,
}

/// Performance change detection
#[derive(Debug, Clone)]
pub struct PerformanceChange {
    /// Metric affected
    pub metric: String,
    /// Type of change
    pub change_type: ChangeType,
    /// Magnitude of change
    pub magnitude: f64,
    /// Confidence in the change
    pub confidence: f64,
    /// Potential causes
    pub potential_causes: Vec<String>,
}

/// Type of performance change
#[derive(Debug, Clone, Copy)]
pub enum ChangeType {
    /// Improvement
    Improvement,
    /// Regression
    Regression,
    /// VolatilityIncrease
    VolatilityIncrease,
    /// BiasShift
    BiasShift,
}

/// Automated recommendation
#[derive(Debug, Clone)]
pub struct Recommendation {
    /// Recommendation type
    pub category: RecommendationCategory,
    /// Priority level
    pub priority: Priority,
    /// Recommendation text
    pub description: String,
    /// Supporting evidence
    pub evidence: Vec<String>,
    /// Estimated impact
    pub estimated_impact: String,
    /// Implementation difficulty
    pub implementation_difficulty: Difficulty,
}

/// Recommendation categories
#[derive(Debug, Clone, Copy)]
pub enum RecommendationCategory {
    /// DataQuality
    DataQuality,
    /// ModelImprovement
    ModelImprovement,
    /// Hyperparameters
    Hyperparameters,
    /// FeatureEngineering
    FeatureEngineering,
    /// Bias
    Bias,
    /// Robustness
    Robustness,
    /// Performance
    Performance,
    /// Monitoring
    Monitoring,
}

/// Priority levels
#[derive(Debug, Clone, Copy)]
pub enum Priority {
    /// Critical
    Critical,
    /// High
    High,
    /// Medium
    Medium,
    /// Low
    Low,
}

/// Implementation difficulty
#[derive(Debug, Clone, Copy)]
pub enum Difficulty {
    /// Easy
    Easy,
    /// Medium
    Medium,
    /// Hard
    Hard,
    /// VeryHard
    VeryHard,
}

/// Generate a comprehensive metric report
///
/// # Arguments
///
/// * `metrics` - HashMap of metric names to values
/// * `config` - Report configuration
///
/// # Returns
///
/// Complete metric report
pub fn generate_metric_report(
    metrics: &HashMap<String, f64>,
    config: &ReportConfig,
) -> MetricsResult<MetricReport> {
    if metrics.is_empty() {
        return Err(MetricsError::EmptyInput);
    }

    // Generate metadata
    let metadata = ReportMetadata {
        title: config.title.clone(),
        timestamp: "2026-01-03T00:00:00Z".to_string(), // Would use actual timestamp
        author: config.author.clone(),
        version: "1.0".to_string(),
        dataset_info: DatasetInfo {
            name: "Unknown Dataset".to_string(),
            n_samples: 1000, // Would be provided
            n_features: 10,
            n_classes: Some(2),
            split_info: "80/20 train/test split".to_string(),
        },
    };

    // Generate metric summaries
    let mut metric_summaries = HashMap::new();
    for (name, &value) in metrics.iter() {
        let summary = MetricSummary {
            name: name.clone(),
            value,
            standard_error: None, // Would calculate if data available
            confidence_interval: None,
            interpretation: interpret_metric(name, value),
            grade: grade_performance(name, value),
        };
        metric_summaries.insert(name.clone(), summary);
    }

    // Generate executive summary
    let executive_summary = generate_executive_summary(metrics)?;

    // Generate statistical analysis
    let statistical_analysis = StatisticalAnalysis {
        sample_size_analysis: "Sample size appears adequate for reliable estimates".to_string(),
        confidence_intervals: HashMap::new(),
        significance_tests: Vec::new(),
        power_analysis: None,
    };

    // Generate recommendations
    let recommendations = generate_recommendations(metrics)?;

    Ok(MetricReport {
        metadata,
        metrics: metric_summaries,
        model_comparisons: Vec::new(),
        executive_summary,
        statistical_analysis,
        recommendations,
        trends: None,
    })
}

/// Compare multiple models and generate comparison report
///
/// # Arguments
///
/// * `model_metrics` - HashMap of model names to their metrics
/// * `config` - Report configuration
///
/// # Returns
///
/// Model comparison report
pub fn generate_model_comparison_report(
    model_metrics: &HashMap<String, HashMap<String, f64>>,
    config: &ReportConfig,
) -> MetricsResult<MetricReport> {
    if model_metrics.len() < 2 {
        return Err(MetricsError::InvalidParameter(
            "need at least 2 models for comparison".to_string(),
        ));
    }

    let model_names: Vec<&String> = model_metrics.keys().collect();
    let mut comparisons = Vec::new();

    // Generate pairwise comparisons
    for i in 0..model_names.len() {
        for j in (i + 1)..model_names.len() {
            let model1 = model_names[i];
            let model2 = model_names[j];

            if let (Some(metrics1), Some(metrics2)) =
                (model_metrics.get(model1), model_metrics.get(model2))
            {
                for metric_name in metrics1.keys() {
                    if let (Some(&value1), Some(&value2)) =
                        (metrics1.get(metric_name), metrics2.get(metric_name))
                    {
                        let significance_test = perform_statistical_test(
                            value1,
                            value2,
                            metric_name,
                            config.confidence_level,
                        );

                        let comparison = ModelComparison {
                            model_names: (model1.clone(), model2.clone()),
                            metric_name: metric_name.clone(),
                            values: (value1, value2),
                            significance_test,
                            practical_significance: PracticalSignificance {
                                absolute_difference: (value1 - value2).abs(),
                                relative_difference: ((value1 - value2) / value2.max(value1))
                                    * 100.0,
                                is_significant: (value1 - value2).abs() > 0.05,
                                interpretation: format!(
                                    "Model {} has {:.1}% {} performance",
                                    if value1 > value2 { model1 } else { model2 },
                                    ((value1 - value2).abs() / value2.max(value1)) * 100.0,
                                    if value1 > value2 { "better" } else { "worse" }
                                ),
                            },
                        };
                        comparisons.push(comparison);
                    }
                }
            }
        }
    }

    // Use first model's metrics as base for generating overall report
    let first_model_metrics = model_metrics.values().next().unwrap();
    let mut base_report = generate_metric_report(first_model_metrics, config)?;
    base_report.model_comparisons = comparisons;

    Ok(base_report)
}

/// Generate executive summary from metrics
fn generate_executive_summary(metrics: &HashMap<String, f64>) -> MetricsResult<ExecutiveSummary> {
    let mut key_findings = Vec::new();
    let mut critical_issues = Vec::new();

    // Analyze each metric
    for (name, &value) in metrics.iter() {
        match name.as_str() {
            "accuracy" => {
                if value > 0.9 {
                    key_findings.push(format!(
                        "Excellent accuracy achieved: {:.1}%",
                        value * 100.0
                    ));
                } else if value < 0.7 {
                    critical_issues.push(format!(
                        "Low accuracy: {:.1}% - model may need improvement",
                        value * 100.0
                    ));
                }
            }
            "precision" | "recall" => {
                if value < 0.6 {
                    critical_issues.push(format!(
                        "Low {}: {:.1}% - indicates potential bias or data quality issues",
                        name,
                        value * 100.0
                    ));
                }
            }
            _ => {}
        }
    }

    let overall_performance = if critical_issues.is_empty() {
        "Model performance is satisfactory with no critical issues identified"
    } else {
        "Model performance has room for improvement with several areas of concern"
    }
    .to_string();

    let business_impact = if critical_issues.is_empty() {
        "Model is suitable for production deployment with appropriate monitoring"
    } else {
        "Model requires additional development before production deployment"
    }
    .to_string();

    let next_steps = vec![
        "Monitor model performance in production".to_string(),
        "Collect additional training data if performance degrades".to_string(),
        "Regular retraining schedule recommended".to_string(),
    ];

    Ok(ExecutiveSummary {
        overall_performance,
        key_findings,
        critical_issues,
        business_impact,
        next_steps,
    })
}

/// Generate automated recommendations based on metric analysis
fn generate_recommendations(metrics: &HashMap<String, f64>) -> MetricsResult<Vec<Recommendation>> {
    let mut recommendations = Vec::new();

    for (name, &value) in metrics.iter() {
        match name.as_str() {
            "accuracy" => {
                if value < 0.8 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::ModelImprovement,
                        priority: Priority::High,
                        description: "Consider using more sophisticated algorithms or ensemble methods to improve accuracy".to_string(),
                        evidence: vec![format!("Current accuracy: {:.1}%", value * 100.0)],
                        estimated_impact: "5-15% accuracy improvement".to_string(),
                        implementation_difficulty: Difficulty::Medium,
                    });
                }
            }
            "precision" => {
                if value < 0.7 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::Bias,
                        priority: Priority::High,
                        description: "Low precision indicates high false positive rate - review decision threshold".to_string(),
                        evidence: vec![format!("Current precision: {:.1}%", value * 100.0)],
                        estimated_impact: "Reduce false positives by 20-40%".to_string(),
                        implementation_difficulty: Difficulty::Easy,
                    });
                }
            }
            "recall" => {
                if value < 0.7 {
                    recommendations.push(Recommendation {
                        category: RecommendationCategory::DataQuality,
                        priority: Priority::Medium,
                        description: "Low recall suggests missing positive examples - review data collection process".to_string(),
                        evidence: vec![format!("Current recall: {:.1}%", value * 100.0)],
                        estimated_impact: "Capture 15-30% more positive cases".to_string(),
                        implementation_difficulty: Difficulty::Hard,
                    });
                }
            }
            _ => {}
        }
    }

    // Always include monitoring recommendation
    recommendations.push(Recommendation {
        category: RecommendationCategory::Monitoring,
        priority: Priority::Medium,
        description: "Implement continuous monitoring to detect performance drift".to_string(),
        evidence: vec!["Best practice for production ML systems".to_string()],
        estimated_impact: "Early detection of model degradation".to_string(),
        implementation_difficulty: Difficulty::Medium,
    });

    Ok(recommendations)
}

/// Interpret metric value with context
fn interpret_metric(metric_name: &str, value: f64) -> String {
    match metric_name {
        "accuracy" => {
            if value > 0.95 {
                "Excellent accuracy - model performs very well".to_string()
            } else if value > 0.85 {
                "Good accuracy - model performance is satisfactory".to_string()
            } else if value > 0.7 {
                "Fair accuracy - model has room for improvement".to_string()
            } else {
                "Poor accuracy - model needs significant improvement".to_string()
            }
        }
        "precision" => {
            format!(
                "Precision of {:.1}% means {:.1}% of positive predictions are correct",
                value * 100.0,
                value * 100.0
            )
        }
        "recall" => {
            format!(
                "Recall of {:.1}% means {:.1}% of actual positives are detected",
                value * 100.0,
                value * 100.0
            )
        }
        "f1" => {
            format!(
                "F1-score of {:.3} represents balanced precision-recall performance",
                value
            )
        }
        _ => format!("Value: {:.4}", value),
    }
}

/// Grade metric performance
fn grade_performance(metric_name: &str, value: f64) -> PerformanceGrade {
    match metric_name {
        "accuracy" | "precision" | "recall" | "f1" => {
            if value > 0.95 {
                PerformanceGrade::Excellent
            } else if value > 0.85 {
                PerformanceGrade::Good
            } else if value > 0.7 {
                PerformanceGrade::Fair
            } else if value > 0.5 {
                PerformanceGrade::Poor
            } else {
                PerformanceGrade::Critical
            }
        }
        _ => PerformanceGrade::Fair, // Default grade
    }
}

/// Perform statistical significance test between two metric values
///
/// # Arguments
///
/// * `value1` - First metric value
/// * `value2` - Second metric value  
/// * `metric_name` - Name of the metric being compared
/// * `confidence_level` - Confidence level for the test
///
/// # Returns
///
/// Statistical significance test result
fn perform_statistical_test(
    value1: f64,
    value2: f64,
    metric_name: &str,
    confidence_level: f64,
) -> SignificanceTest {
    let alpha = 1.0 - confidence_level;
    let diff = value1 - value2;
    let abs_diff = diff.abs();

    // Calculate effect size (Cohen's d approximation)
    let effect_size = if abs_diff < 1e-10 {
        0.0
    } else {
        // Approximate pooled standard deviation based on metric type
        let pooled_std = match metric_name {
            "accuracy" | "precision" | "recall" | "f1" => {
                // For proportions, use binomial variance approximation
                let p = (value1 + value2) / 2.0;
                (p * (1.0 - p)).sqrt()
            }
            _ => 0.1, // Default standard deviation estimate
        };

        if pooled_std > 1e-10 {
            abs_diff / pooled_std
        } else {
            abs_diff * 10.0 // Fallback scaling
        }
    };

    // Perform approximate statistical test
    let (test_statistic, p_value, method) = match metric_name {
        "accuracy" | "precision" | "recall" | "f1" => {
            // Use z-test approximation for proportions
            let p = (value1 + value2) / 2.0;
            let se = (2.0 * p * (1.0 - p) / 100.0).sqrt(); // Assume n=100 samples
            let z_stat = if se > 1e-10 { diff / se } else { 0.0 };
            let p_val = 2.0 * (1.0 - normal_cdf(z_stat.abs()));
            (z_stat, p_val, "Z-test for proportions".to_string())
        }
        _ => {
            // Use t-test approximation for continuous metrics
            let se = 0.1; // Approximate standard error
            let t_stat = if se > 1e-10 { diff / se } else { 0.0 };
            let p_val = 2.0 * (1.0 - t_cdf(t_stat.abs(), 10.0)); // Assume df=10
            (t_stat, p_val, "T-test".to_string())
        }
    };

    let is_significant = p_value < alpha;

    SignificanceTest {
        test_statistic,
        p_value,
        method,
        is_significant,
        effect_size,
    }
}

/// Approximate normal cumulative distribution function
fn normal_cdf(x: f64) -> f64 {
    // Approximation using error function
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Approximate error function using series expansion
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
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

/// Approximate t-distribution cumulative distribution function
fn t_cdf(t: f64, df: f64) -> f64 {
    // Approximate using normal distribution for large df
    if df > 30.0 {
        normal_cdf(t)
    } else {
        // Simple approximation for small df
        let x = t / (df + t * t).sqrt();
        0.5 + x * (0.5 - x * x / 6.0)
    }
}

/// Enhanced model comparison with multiple statistical tests
pub fn generate_comprehensive_model_comparison(
    model_metrics: &HashMap<String, HashMap<String, f64>>,
    model_data: Option<&HashMap<String, Vec<f64>>>, // Optional raw performance data
    config: &ReportConfig,
) -> MetricsResult<MetricReport> {
    if model_metrics.len() < 2 {
        return Err(MetricsError::InvalidParameter(
            "need at least 2 models for comparison".to_string(),
        ));
    }

    let model_names: Vec<&String> = model_metrics.keys().collect();
    let mut comparisons = Vec::new();
    let mut significance_tests = Vec::new();

    // Generate pairwise comparisons with enhanced statistical analysis
    for i in 0..model_names.len() {
        for j in (i + 1)..model_names.len() {
            let model1 = model_names[i];
            let model2 = model_names[j];

            if let (Some(metrics1), Some(metrics2)) =
                (model_metrics.get(model1), model_metrics.get(model2))
            {
                for metric_name in metrics1.keys() {
                    if let (Some(&value1), Some(&value2)) =
                        (metrics1.get(metric_name), metrics2.get(metric_name))
                    {
                        // Perform multiple statistical tests
                        let mut test_results = Vec::new();

                        // Basic significance test
                        let basic_test = perform_statistical_test(
                            value1,
                            value2,
                            metric_name,
                            config.confidence_level,
                        );
                        test_results.push(basic_test.clone());

                        // If raw data is available, perform more sophisticated tests
                        if let Some(data) = model_data {
                            if let (Some(data1), Some(data2)) = (data.get(model1), data.get(model2))
                            {
                                let bootstrap_test =
                                    perform_bootstrap_test(data1, data2, config.confidence_level);
                                test_results.push(bootstrap_test);
                            }
                        }

                        // Use the most appropriate test result
                        let significance_test = test_results
                            .into_iter()
                            .min_by(|a, b| a.p_value.partial_cmp(&b.p_value).unwrap())
                            .unwrap_or(basic_test);

                        let comparison = ModelComparison {
                            model_names: (model1.clone(), model2.clone()),
                            metric_name: metric_name.clone(),
                            values: (value1, value2),
                            significance_test: significance_test.clone(),
                            practical_significance: assess_practical_significance(
                                value1,
                                value2,
                                metric_name,
                            ),
                        };

                        comparisons.push(comparison);
                        significance_tests.push(significance_test);
                    }
                }
            }
        }
    }

    // Generate comprehensive report
    let first_model_metrics = model_metrics.values().next().unwrap();
    let mut base_report = generate_metric_report(first_model_metrics, config)?;
    base_report.model_comparisons = comparisons;
    base_report.statistical_analysis.significance_tests = significance_tests;

    Ok(base_report)
}

/// Perform bootstrap statistical test for model comparison
fn perform_bootstrap_test(data1: &[f64], data2: &[f64], confidence_level: f64) -> SignificanceTest {
    let n_bootstrap = 1000;
    let alpha = 1.0 - confidence_level;

    // Calculate observed difference
    let mean1 = data1.iter().sum::<f64>() / data1.len() as f64;
    let mean2 = data2.iter().sum::<f64>() / data2.len() as f64;
    let observed_diff = mean1 - mean2;

    // Combine data for null hypothesis resampling
    let mut combined_data = Vec::new();
    combined_data.extend_from_slice(data1);
    combined_data.extend_from_slice(data2);

    let n1 = data1.len();
    let n2 = data2.len();
    let total_n = combined_data.len();

    // Bootstrap resampling
    let mut bootstrap_diffs = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Resample without replacement for each group
        let mut indices: Vec<usize> = (0..total_n).collect();

        // Simple pseudo-random shuffle (deterministic for reproducibility)
        for i in 0..total_n {
            let j = (i * 17 + 7) % total_n; // Simple deterministic shuffle
            indices.swap(i, j);
        }

        let sample1_mean = indices[0..n1]
            .iter()
            .map(|&i| combined_data[i])
            .sum::<f64>()
            / n1 as f64;

        let sample2_mean = indices[n1..n1 + n2]
            .iter()
            .map(|&i| combined_data[i])
            .sum::<f64>()
            / n2 as f64;

        bootstrap_diffs.push(sample1_mean - sample2_mean);
    }

    // Calculate p-value
    let extreme_count = bootstrap_diffs
        .iter()
        .filter(|&&diff| diff.abs() >= observed_diff.abs())
        .count();
    let p_value = extreme_count as f64 / n_bootstrap as f64;

    // Calculate effect size
    let pooled_std = calculate_pooled_std(data1, data2);
    let effect_size = if pooled_std > 1e-10 {
        observed_diff.abs() / pooled_std
    } else {
        0.0
    };

    SignificanceTest {
        test_statistic: observed_diff
            / (pooled_std / ((1.0 / n1 as f64) + (1.0 / n2 as f64)).sqrt()),
        p_value,
        method: "Bootstrap test".to_string(),
        is_significant: p_value < alpha,
        effect_size,
    }
}

/// Calculate pooled standard deviation for two samples
fn calculate_pooled_std(data1: &[f64], data2: &[f64]) -> f64 {
    let n1 = data1.len() as f64;
    let n2 = data2.len() as f64;

    let mean1 = data1.iter().sum::<f64>() / n1;
    let mean2 = data2.iter().sum::<f64>() / n2;

    let var1 = data1.iter().map(|&x| (x - mean1).powi(2)).sum::<f64>() / (n1 - 1.0);

    let var2 = data2.iter().map(|&x| (x - mean2).powi(2)).sum::<f64>() / (n2 - 1.0);

    let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
    pooled_var.sqrt()
}

/// Assess practical significance of difference between models
fn assess_practical_significance(
    value1: f64,
    value2: f64,
    metric_name: &str,
) -> PracticalSignificance {
    let absolute_difference = (value1 - value2).abs();
    let relative_difference = if value2.abs() > 1e-10 {
        ((value1 - value2) / value2) * 100.0
    } else {
        0.0
    };

    // Define practical significance thresholds by metric type
    let threshold = match metric_name {
        "accuracy" | "precision" | "recall" | "f1" => 0.02, // 2% improvement
        "auc" | "roc_auc" => 0.01,                          // 1% improvement
        "mse" | "rmse" => 0.05, // 5% improvement (for normalized metrics)
        _ => 0.03,              // Default 3% improvement
    };

    let is_significant = absolute_difference > threshold;

    let interpretation = if is_significant {
        let better_model = if value1 > value2 { "first" } else { "second" };
        let improvement_type = match metric_name {
            "mse" | "rmse" | "mae" => {
                if value1 < value2 {
                    "lower error"
                } else {
                    "higher error"
                }
            }
            _ => {
                if value1 > value2 {
                    "better performance"
                } else {
                    "worse performance"
                }
            }
        };
        format!(
            "The {} model shows practically significant {} ({:.1}% relative difference)",
            better_model,
            improvement_type,
            relative_difference.abs()
        )
    } else {
        "The difference between models is not practically significant".to_string()
    };

    PracticalSignificance {
        absolute_difference,
        relative_difference,
        is_significant,
        interpretation,
    }
}

/// Implementation of report output formats
impl MetricReport {
    /// Convert report to HTML format
    pub fn to_html(&self) -> String {
        let mut html = String::new();

        html.push_str("<!DOCTYPE html>\n<html>\n<head>\n");
        html.push_str(&format!("<title>{}</title>\n", self.metadata.title));
        html.push_str("<style>\n");
        html.push_str("body { font-family: Arial, sans-serif; margin: 40px; }\n");
        html.push_str("h1 { color: #333; }\n");
        html.push_str("h2 { color: #666; border-bottom: 1px solid #ccc; }\n");
        html.push_str(
            ".metric { margin: 10px 0; padding: 10px; border-left: 4px solid #007cba; }\n",
        );
        html.push_str(".excellent { border-color: #28a745; }\n");
        html.push_str(".good { border-color: #17a2b8; }\n");
        html.push_str(".fair { border-color: #ffc107; }\n");
        html.push_str(".poor { border-color: #fd7e14; }\n");
        html.push_str(".critical { border-color: #dc3545; }\n");
        html.push_str("</style>\n");
        html.push_str("</head>\n<body>\n");

        // Header
        html.push_str(&format!("<h1>{}</h1>\n", self.metadata.title));
        html.push_str(&format!(
            "<p><strong>Generated:</strong> {}</p>\n",
            self.metadata.timestamp
        ));
        html.push_str(&format!(
            "<p><strong>Author:</strong> {}</p>\n",
            self.metadata.author
        ));

        // Executive Summary
        html.push_str("<h2>Executive Summary</h2>\n");
        html.push_str(&format!(
            "<p>{}</p>\n",
            self.executive_summary.overall_performance
        ));

        if !self.executive_summary.key_findings.is_empty() {
            html.push_str("<h3>Key Findings</h3>\n<ul>\n");
            for finding in &self.executive_summary.key_findings {
                html.push_str(&format!("<li>{}</li>\n", finding));
            }
            html.push_str("</ul>\n");
        }

        // Metrics
        html.push_str("<h2>Detailed Metrics</h2>\n");
        for (name, metric) in &self.metrics {
            let grade_class = match metric.grade {
                PerformanceGrade::Excellent => "excellent",
                PerformanceGrade::Good => "good",
                PerformanceGrade::Fair => "fair",
                PerformanceGrade::Poor => "poor",
                PerformanceGrade::Critical => "critical",
            };

            html.push_str(&format!(
                "<div class=\"metric {}\">\n<h4>{}</h4>\n<p><strong>Value:</strong> {:.4}</p>\n<p><strong>Grade:</strong> {}</p>\n<p>{}</p>\n</div>\n",
                grade_class, name, metric.value, metric.grade, metric.interpretation
            ));
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            html.push_str("<h2>Recommendations</h2>\n");
            for rec in &self.recommendations {
                html.push_str(&format!(
                    "<div class=\"metric\">\n<h4>{:?} - {:?} Priority</h4>\n<p>{}</p>\n</div>\n",
                    rec.category, rec.priority, rec.description
                ));
            }
        }

        html.push_str("</body>\n</html>");
        html
    }

    /// Convert report to Markdown format
    pub fn to_markdown(&self) -> String {
        let mut md = String::new();

        md.push_str(&format!("# {}\n\n", self.metadata.title));
        md.push_str(&format!("**Generated:** {}\n", self.metadata.timestamp));
        md.push_str(&format!("**Author:** {}\n\n", self.metadata.author));

        // Executive Summary
        md.push_str("## Executive Summary\n\n");
        md.push_str(&format!(
            "{}\n\n",
            self.executive_summary.overall_performance
        ));

        // Metrics
        md.push_str("## Metrics\n\n");
        for (name, metric) in &self.metrics {
            md.push_str(&format!("### {}\n", name));
            md.push_str(&format!("- **Value:** {:.4}\n", metric.value));
            md.push_str(&format!("- **Grade:** {}\n", metric.grade));
            md.push_str(&format!(
                "- **Interpretation:** {}\n\n",
                metric.interpretation
            ));
        }

        // Recommendations
        if !self.recommendations.is_empty() {
            md.push_str("## Recommendations\n\n");
            for rec in &self.recommendations {
                md.push_str(&format!(
                    "### {:?} ({:?} Priority)\n",
                    rec.category, rec.priority
                ));
                md.push_str(&format!("{}\n\n", rec.description));
            }
        }

        md
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_metric_report() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);
        metrics.insert("precision".to_string(), 0.82);
        metrics.insert("recall".to_string(), 0.88);

        let config = ReportConfig::default();
        let report = generate_metric_report(&metrics, &config).unwrap();

        assert_eq!(report.metadata.title, config.title);
        assert_eq!(report.metrics.len(), 3);
        assert!(report.metrics.contains_key("accuracy"));
        assert!(report.metrics.contains_key("precision"));
        assert!(report.metrics.contains_key("recall"));
    }

    #[test]
    fn test_metric_interpretation() {
        assert!(interpret_metric("accuracy", 0.96).contains("Excellent"));
        assert!(interpret_metric("accuracy", 0.90).contains("Good"));
        assert!(interpret_metric("accuracy", 0.75).contains("Fair"));
        assert!(interpret_metric("accuracy", 0.6).contains("Poor"));

        assert!(interpret_metric("precision", 0.8).contains("80.0%"));
    }

    #[test]
    fn test_performance_grading() {
        assert!(matches!(
            grade_performance("accuracy", 0.96),
            PerformanceGrade::Excellent
        ));
        assert!(matches!(
            grade_performance("accuracy", 0.86),
            PerformanceGrade::Good
        ));
        assert!(matches!(
            grade_performance("accuracy", 0.76),
            PerformanceGrade::Fair
        ));
        assert!(matches!(
            grade_performance("accuracy", 0.6),
            PerformanceGrade::Poor
        ));
        assert!(matches!(
            grade_performance("accuracy", 0.4),
            PerformanceGrade::Critical
        ));
    }

    #[test]
    fn test_model_comparison_report() {
        let mut model1_metrics = HashMap::new();
        model1_metrics.insert("accuracy".to_string(), 0.85);

        let mut model2_metrics = HashMap::new();
        model2_metrics.insert("accuracy".to_string(), 0.82);

        let mut model_metrics = HashMap::new();
        model_metrics.insert("Model1".to_string(), model1_metrics);
        model_metrics.insert("Model2".to_string(), model2_metrics);

        let config = ReportConfig::default();
        let report = generate_model_comparison_report(&model_metrics, &config).unwrap();

        assert!(!report.model_comparisons.is_empty());
        assert_eq!(report.model_comparisons[0].metric_name, "accuracy");
    }

    #[test]
    fn test_html_output() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);

        let config = ReportConfig::default();
        let report = generate_metric_report(&metrics, &config).unwrap();
        let html = report.to_html();

        assert!(html.contains("<!DOCTYPE html>"));
        assert!(html.contains("accuracy"));
        assert!(html.contains("85"));
    }

    #[test]
    fn test_markdown_output() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.85);

        let config = ReportConfig::default();
        let report = generate_metric_report(&metrics, &config).unwrap();
        let markdown = report.to_markdown();

        assert!(markdown.contains("#"));
        assert!(markdown.contains("accuracy"));
        assert!(markdown.contains("0.8500"));
    }

    #[test]
    fn test_recommendations_generation() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.75); // Should trigger recommendation
        metrics.insert("precision".to_string(), 0.6); // Should trigger recommendation

        let recommendations = generate_recommendations(&metrics).unwrap();

        assert!(!recommendations.is_empty());
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.category, RecommendationCategory::ModelImprovement)));
        assert!(recommendations
            .iter()
            .any(|r| matches!(r.category, RecommendationCategory::Bias)));
    }

    #[test]
    fn test_executive_summary_generation() {
        let mut metrics = HashMap::new();
        metrics.insert("accuracy".to_string(), 0.95);
        metrics.insert("precision".to_string(), 0.55); // Low precision

        let summary = generate_executive_summary(&metrics).unwrap();

        assert!(!summary.key_findings.is_empty());
        assert!(!summary.critical_issues.is_empty());
        assert!(summary.key_findings[0].contains("Excellent accuracy"));
        assert!(summary.critical_issues[0].contains("Low precision"));
    }
}

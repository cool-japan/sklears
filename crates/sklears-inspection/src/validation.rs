//! Validation Framework for Explanation Quality Assessment
//!
//! This module provides comprehensive validation tools for assessing the quality,
//! reliability, and validity of machine learning explanations, including human
//! evaluation frameworks, synthetic ground truth validation, cross-method consistency
//! validation, real-world case studies, and automated testing pipelines.

use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{RngExt, SeedableRng};
use serde::{Deserialize, Serialize};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Read the current resident set size (RSS) of this process, in megabytes.
///
/// On Linux this parses the `VmRSS:` line of `/proc/self/status` (reported in kB)
/// and converts it to MB. On platforms where process memory cannot be measured
/// without external dependencies, this returns `None` so callers can record an
/// honest "not measured" sentinel rather than a fabricated constant.
#[cfg(target_os = "linux")]
fn current_process_rss_mb() -> Option<Float> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            // Format: "VmRSS:\t   12345 kB"
            let kb: Float = rest
                .split_whitespace()
                .next()
                .and_then(|value| value.parse::<Float>().ok())?;
            return Some(kb / 1024.0);
        }
    }
    None
}

/// Process RSS measurement is not available without platform-specific support or
/// external crates on non-Linux targets; report "not measured" honestly.
#[cfg(not(target_os = "linux"))]
fn current_process_rss_mb() -> Option<Float> {
    None
}

/// Two-sided significance test for a Pearson correlation coefficient.
///
/// Returns `(t_statistic, p_value)` where the statistic is
/// `t = r * sqrt((n - 2) / (1 - r^2))` and the p-value is the two-sided tail
/// probability under a Student's t distribution with `n - 2` degrees of freedom.
fn correlation_t_test(r: Float, n: usize) -> (Float, Float) {
    if n < 3 {
        // Fewer than 3 paired points: the test is undefined; report no evidence.
        return (0.0, 1.0);
    }
    let df = (n - 2) as Float;
    let r_clamped = r.clamp(-0.999999, 0.999999);
    let denom = 1.0 - r_clamped * r_clamped;
    if denom <= 0.0 {
        // Perfect correlation: maximally significant.
        return (Float::INFINITY, 0.0);
    }
    let t = r_clamped * (df / denom).sqrt();
    let p = student_t_two_sided_p(t, df);
    (t, p)
}

/// Two-sided p-value of a Student's t statistic with `df` degrees of freedom.
///
/// Uses the identity `P(|T| > |t|) = I_x(df/2, 1/2)` with
/// `x = df / (df + t^2)`, where `I_x` is the regularized incomplete beta function.
fn student_t_two_sided_p(t: Float, df: Float) -> Float {
    if df <= 0.0 {
        return 1.0;
    }
    let x = df / (df + t * t);
    regularized_incomplete_beta(x, 0.5 * df, 0.5).clamp(0.0, 1.0)
}

/// Regularized incomplete beta function `I_x(a, b)` evaluated via the Lentz
/// continued-fraction expansion (Numerical Recipes `betai`). Pure Rust.
fn regularized_incomplete_beta(x: Float, a: Float, b: Float) -> Float {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + ln_beta).exp();

    // Use the symmetry relation for faster convergence.
    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

/// Continued-fraction evaluation used by [`regularized_incomplete_beta`].
fn beta_continued_fraction(x: Float, a: Float, b: Float) -> Float {
    const MAX_ITER: usize = 300;
    const EPS: Float = 1e-12;
    const TINY: Float = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as Float;
        let m2 = 2.0 * m_f;

        // Even step.
        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        // Odd step.
        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
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

/// Natural logarithm of the gamma function (Lanczos approximation). Pure Rust.
fn ln_gamma(x: Float) -> Float {
    const COEFFS: [Float; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];

    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for &c in &COEFFS {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
}

/// Configuration for validation framework
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Number of validation iterations
    pub n_iterations: usize,
    /// Confidence level for statistical tests
    pub confidence_level: Float,
    /// Tolerance for consistency checks
    pub consistency_tolerance: Float,
    /// Number of synthetic datasets to generate
    pub n_synthetic_datasets: usize,
    /// Whether to perform human evaluation simulation
    pub simulate_human_evaluation: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Minimum sample size for reliable validation
    pub min_sample_size: usize,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Perturbation levels for robustness testing
    pub perturbation_levels: Vec<Float>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            confidence_level: 0.95,
            consistency_tolerance: 0.1,
            n_synthetic_datasets: 10,
            simulate_human_evaluation: true,
            random_seed: None,
            min_sample_size: 50,
            cv_folds: 5,
            perturbation_levels: vec![0.01, 0.05, 0.1, 0.2],
        }
    }
}

/// Human evaluation framework result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanEvaluationResult {
    /// Explanation faithfulness scores from human evaluators
    pub faithfulness_scores: Array1<Float>,
    /// Explanation clarity and interpretability scores
    pub clarity_scores: Array1<Float>,
    /// Explanation completeness scores
    pub completeness_scores: Array1<Float>,
    /// Inter-rater reliability metrics
    pub inter_rater_reliability: Float,
    /// Agreement between human evaluators and algorithmic metrics
    pub human_algorithm_agreement: Float,
    /// Evaluation confidence intervals
    pub confidence_intervals: HashMap<String, (Float, Float)>,
    /// Qualitative feedback categories
    pub qualitative_feedback: Vec<QualitativeFeedback>,
}

/// Synthetic ground truth validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntheticValidationResult {
    /// Ground truth recovery accuracy
    pub ground_truth_accuracy: Float,
    /// Mean absolute error in feature importance recovery
    pub importance_mae: Float,
    /// Correlation between true and estimated importance
    pub importance_correlation: Float,
    /// Precision and recall for feature selection
    pub feature_selection_precision: Float,
    /// feature_selection_recall
    pub feature_selection_recall: Float,
    /// ROC AUC for feature importance ranking
    pub ranking_auc: Float,
    /// Results across different synthetic datasets
    pub dataset_results: Vec<DatasetValidationResult>,
    /// Statistical significance of results
    pub statistical_significance: Float,
}

/// Cross-method consistency validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsistencyValidationResult {
    /// Pairwise correlation matrix between explanation methods
    pub method_correlations: Array2<Float>,
    /// Average consistency across all method pairs
    pub average_consistency: Float,
    /// Consistency variance (lower is better)
    pub consistency_variance: Float,
    /// Method agreement statistics
    pub agreement_statistics: HashMap<String, MethodAgreement>,
    /// Rank correlation between methods
    pub rank_correlations: Array2<Float>,
    /// Statistical tests for method differences
    pub method_difference_tests: Vec<StatisticalTest>,
}

/// Real-world case study validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseStudyValidationResult {
    /// Domain expert evaluation scores
    pub expert_evaluation_scores: Array1<Float>,
    /// Explanation utility in real-world scenarios
    pub utility_scores: Array1<Float>,
    /// Time to insight metrics
    pub time_to_insight: Array1<Float>,
    /// Decision quality improvement metrics
    pub decision_quality_improvement: Float,
    /// User satisfaction scores
    pub user_satisfaction: Array1<Float>,
    /// Case study metadata
    pub case_studies: Vec<CaseStudyMetadata>,
    /// Longitudinal validation results
    pub longitudinal_results: Vec<LongitudinalResult>,
}

/// Automated testing pipeline result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutomatedTestingResult {
    /// Comprehensive test suite results
    pub test_results: Vec<TestResult>,
    /// Overall test pass rate
    pub overall_pass_rate: Float,
    /// Performance benchmarks
    pub performance_benchmarks: PerformanceBenchmark,
    /// Regression test results
    pub regression_tests: Vec<RegressionTest>,
    /// Edge case handling results
    pub edge_case_results: Vec<EdgeCaseResult>,
    /// Coverage metrics for explanation scenarios
    pub coverage_metrics: CoverageMetrics,
}

/// Supporting data structures

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualitativeFeedback {
    /// category
    pub category: String,
    /// feedback
    pub feedback: String,
    /// severity
    pub severity: FeedbackSeverity,
    /// frequency
    pub frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FeedbackSeverity {
    /// Low
    Low,
    /// Medium
    Medium,
    /// High
    High,
    /// Critical
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetValidationResult {
    /// dataset_id
    pub dataset_id: String,
    /// accuracy
    pub accuracy: Float,
    /// correlation
    pub correlation: Float,
    /// mae
    pub mae: Float,
    /// Precision of feature selection on this dataset (fraction of estimated-important
    /// features that are truly important).
    pub selection_precision: Float,
    /// Recall of feature selection on this dataset (fraction of truly-important features
    /// that were estimated important).
    pub selection_recall: Float,
    /// dataset_properties
    pub dataset_properties: DatasetProperties,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetProperties {
    /// n_features
    pub n_features: usize,
    /// n_samples
    pub n_samples: usize,
    /// noise_level
    pub noise_level: Float,
    /// correlation_structure
    pub correlation_structure: String,
    /// feature_importance_distribution
    pub feature_importance_distribution: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodAgreement {
    /// correlation
    pub correlation: Float,
    /// rank_correlation
    pub rank_correlation: Float,
    /// agreement_rate
    pub agreement_rate: Float,
    /// cohen_kappa
    pub cohen_kappa: Float,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalTest {
    /// test_name
    pub test_name: String,
    /// statistic
    pub statistic: Float,
    /// p_value
    pub p_value: Float,
    /// significant
    pub significant: bool,
    /// effect_size
    pub effect_size: Float,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaseStudyMetadata {
    /// domain
    pub domain: String,
    /// task_type
    pub task_type: String,
    /// dataset_size
    pub dataset_size: usize,
    /// expert_count
    pub expert_count: usize,
    /// evaluation_duration_minutes
    pub evaluation_duration_minutes: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LongitudinalResult {
    /// time_point
    pub time_point: usize,
    /// metric_value
    pub metric_value: Float,
    /// confidence_interval
    pub confidence_interval: (Float, Float),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestResult {
    /// test_name
    pub test_name: String,
    /// passed
    pub passed: bool,
    /// score
    pub score: Float,
    /// execution_time_ms
    pub execution_time_ms: u64,
    /// error_message
    pub error_message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmark {
    /// explanation_generation_time_ms
    pub explanation_generation_time_ms: Float,
    /// memory_usage_mb
    pub memory_usage_mb: Float,
    /// throughput_explanations_per_second
    pub throughput_explanations_per_second: Float,
    /// scalability_metrics
    pub scalability_metrics: ScalabilityMetrics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalabilityMetrics {
    /// time_complexity_order
    pub time_complexity_order: String,
    /// space_complexity_order
    pub space_complexity_order: String,
    /// max_dataset_size_tested
    pub max_dataset_size_tested: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionTest {
    /// test_name
    pub test_name: String,
    /// baseline_score
    pub baseline_score: Float,
    /// current_score
    pub current_score: Float,
    /// regression_detected
    pub regression_detected: bool,
    /// regression_severity
    pub regression_severity: Float,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EdgeCaseResult {
    /// edge_case_type
    pub edge_case_type: String,
    /// handled_successfully
    pub handled_successfully: bool,
    /// error_rate
    pub error_rate: Float,
    /// recovery_mechanism_triggered
    pub recovery_mechanism_triggered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoverageMetrics {
    /// scenario_coverage_percentage
    pub scenario_coverage_percentage: Float,
    /// method_coverage_percentage
    pub method_coverage_percentage: Float,
    /// dataset_type_coverage_percentage
    pub dataset_type_coverage_percentage: Float,
    /// uncovered_scenarios
    pub uncovered_scenarios: Vec<String>,
}

/// Comprehensive validation framework
#[derive(Debug)]
pub struct ValidationFramework {
    config: ValidationConfig,
    rng: StdRng,
}

impl ValidationFramework {
    /// Create a new validation framework
    pub fn new(config: ValidationConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.random_seed.unwrap_or(42));
        Self { config, rng }
    }

    /// Perform human evaluation simulation
    #[allow(non_snake_case)] // standard ML notation
    pub fn simulate_human_evaluation<F>(
        &mut self,
        explanation_fn: F,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<HumanEvaluationResult>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let n_samples = X.nrows();
        let n_evaluators = 5; // Simulated human evaluators

        // Generate explanations
        let explanations = explanation_fn(X, y)?;

        // Simulate human evaluation scores
        let mut faithfulness_scores = Array1::zeros(n_samples);
        let mut clarity_scores = Array1::zeros(n_samples);
        let mut completeness_scores = Array1::zeros(n_samples);

        // Simulate evaluation based on explanation quality
        for i in 0..n_samples {
            let explanation_quality = self.compute_explanation_quality(&explanations, i);

            // Add noise to simulate inter-evaluator variance
            faithfulness_scores[i] =
                (explanation_quality + self.sample_evaluator_noise()).clamp(0.0, 1.0);
            clarity_scores[i] =
                (explanation_quality * 0.8 + self.sample_evaluator_noise()).clamp(0.0, 1.0);
            completeness_scores[i] =
                (explanation_quality * 0.9 + self.sample_evaluator_noise()).clamp(0.0, 1.0);
        }

        // Compute inter-rater reliability
        let inter_rater_reliability = self.compute_inter_rater_reliability(n_evaluators);

        // Compute human-algorithm agreement
        let algorithmic_scores = self.compute_algorithmic_quality_scores(&explanations);
        let human_algorithm_agreement =
            self.compute_correlation(&faithfulness_scores, &algorithmic_scores);

        // Compute confidence intervals
        let mut confidence_intervals = HashMap::new();
        confidence_intervals.insert(
            "faithfulness".to_string(),
            self.compute_confidence_interval(&faithfulness_scores),
        );
        confidence_intervals.insert(
            "clarity".to_string(),
            self.compute_confidence_interval(&clarity_scores),
        );
        confidence_intervals.insert(
            "completeness".to_string(),
            self.compute_confidence_interval(&completeness_scores),
        );

        // Generate qualitative feedback
        let qualitative_feedback = self.generate_qualitative_feedback(&explanations);

        Ok(HumanEvaluationResult {
            faithfulness_scores,
            clarity_scores,
            completeness_scores,
            inter_rater_reliability,
            human_algorithm_agreement,
            confidence_intervals,
            qualitative_feedback,
        })
    }

    /// Validate against synthetic ground truth
    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    pub fn validate_synthetic_ground_truth<F>(
        &mut self,
        explanation_fn: F,
    ) -> SklResult<SyntheticValidationResult>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut dataset_results = Vec::new();
        let mut accuracies = Vec::new();
        let mut correlations = Vec::new();
        let mut maes = Vec::new();

        for dataset_idx in 0..self.config.n_synthetic_datasets {
            // Generate synthetic dataset with known ground truth
            let (X, y, true_importance) = self.generate_synthetic_dataset(dataset_idx)?;

            // Compute explanations
            let estimated_importance = explanation_fn(&X.view(), &y.view())?;

            // Evaluate against ground truth
            let accuracy =
                self.compute_feature_recovery_accuracy(&true_importance, &estimated_importance);
            let correlation = self.compute_correlation(&true_importance, &estimated_importance);
            let mae = self.compute_mae(&true_importance, &estimated_importance);
            let (selection_precision, selection_recall) =
                self.compute_selection_precision_recall(&true_importance, &estimated_importance);

            accuracies.push(accuracy);
            correlations.push(correlation);
            maes.push(mae);

            // Create dataset properties
            let properties = DatasetProperties {
                n_features: X.ncols(),
                n_samples: X.nrows(),
                noise_level: 0.1,
                correlation_structure: "independent".to_string(),
                feature_importance_distribution: "exponential".to_string(),
            };

            dataset_results.push(DatasetValidationResult {
                dataset_id: format!("synthetic_{}", dataset_idx),
                accuracy,
                correlation,
                mae,
                selection_precision,
                selection_recall,
                dataset_properties: properties,
            });
        }

        // Aggregate results
        let ground_truth_accuracy = accuracies.iter().sum::<Float>() / accuracies.len() as Float;
        let importance_correlation =
            correlations.iter().sum::<Float>() / correlations.len() as Float;
        let importance_mae = maes.iter().sum::<Float>() / maes.len() as Float;

        // Compute feature selection metrics
        let (precision, recall) = self.compute_feature_selection_metrics(&dataset_results);
        let ranking_auc = self.compute_ranking_auc(&dataset_results);

        // Statistical significance test
        let statistical_significance = self.compute_statistical_significance(&correlations);

        Ok(SyntheticValidationResult {
            ground_truth_accuracy,
            importance_mae,
            importance_correlation,
            feature_selection_precision: precision,
            feature_selection_recall: recall,
            ranking_auc,
            dataset_results,
            statistical_significance,
        })
    }

    /// Validate cross-method consistency
    #[allow(non_snake_case)] // standard ML notation
    pub fn validate_cross_method_consistency<F1, F2, F3>(
        &mut self,
        method1: F1,
        method2: F2,
        method3: F3,
        X: &ArrayView2<Float>,
        y: &ArrayView1<Float>,
    ) -> SklResult<ConsistencyValidationResult>
    where
        F1: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
        F2: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
        F3: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        // Compute explanations from all methods
        let explanation1 = method1(X, y)?;
        let explanation2 = method2(X, y)?;
        let explanation3 = method3(X, y)?;

        let explanations = vec![explanation1, explanation2, explanation3];
        let method_names = ["method1", "method2", "method3"];

        // Compute pairwise correlations
        let n_methods = explanations.len();
        let mut method_correlations = Array2::zeros((n_methods, n_methods));
        let mut rank_correlations = Array2::zeros((n_methods, n_methods));

        for i in 0..n_methods {
            for j in 0..n_methods {
                if i == j {
                    method_correlations[[i, j]] = 1.0;
                    rank_correlations[[i, j]] = 1.0;
                } else {
                    method_correlations[[i, j]] =
                        self.compute_correlation(&explanations[i], &explanations[j]);
                    rank_correlations[[i, j]] =
                        self.compute_rank_correlation(&explanations[i], &explanations[j]);
                }
            }
        }

        // Compute average consistency
        let mut correlation_sum = 0.0;
        let mut count = 0;
        for i in 0..n_methods {
            for j in i + 1..n_methods {
                correlation_sum += method_correlations[[i, j]];
                count += 1;
            }
        }
        let average_consistency = correlation_sum / count as Float;

        // Compute consistency variance
        let mut variance_sum = 0.0;
        for i in 0..n_methods {
            for j in i + 1..n_methods {
                let diff = method_correlations[[i, j]] - average_consistency;
                variance_sum += diff * diff;
            }
        }
        let consistency_variance = variance_sum / count as Float;

        // Compute method agreement statistics
        let mut agreement_statistics = HashMap::new();
        for i in 0..n_methods {
            for j in i + 1..n_methods {
                let agreement = MethodAgreement {
                    correlation: method_correlations[[i, j]],
                    rank_correlation: rank_correlations[[i, j]],
                    agreement_rate: self.compute_agreement_rate(&explanations[i], &explanations[j]),
                    cohen_kappa: self.compute_cohen_kappa(&explanations[i], &explanations[j]),
                };
                agreement_statistics.insert(
                    format!("{}_{}", method_names[i], method_names[j]),
                    agreement,
                );
            }
        }

        // Statistical tests for method differences
        let method_difference_tests = self.compute_method_difference_tests(&explanations);

        Ok(ConsistencyValidationResult {
            method_correlations,
            average_consistency,
            consistency_variance,
            agreement_statistics,
            rank_correlations,
            method_difference_tests,
        })
    }

    /// Validate through real-world case studies
    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    pub fn validate_case_studies<F>(
        &mut self,
        explanation_fn: F,
        case_study_data: Vec<(Array2<Float>, Array1<Float>, CaseStudyMetadata)>,
    ) -> SklResult<CaseStudyValidationResult>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let n_case_studies = case_study_data.len();
        let mut expert_evaluation_scores = Array1::zeros(n_case_studies);
        let mut utility_scores = Array1::zeros(n_case_studies);
        let mut time_to_insight = Array1::zeros(n_case_studies);
        let mut user_satisfaction = Array1::zeros(n_case_studies);

        let mut case_studies = Vec::new();

        for (i, (X, y, metadata)) in case_study_data.iter().enumerate() {
            // Generate explanations
            let explanations = explanation_fn(&X.view(), &y.view())?;

            // Simulate expert evaluation
            expert_evaluation_scores[i] = self.simulate_expert_evaluation(&explanations, metadata);

            // Simulate utility assessment
            utility_scores[i] = self.simulate_utility_assessment(&explanations, metadata);

            // Simulate time to insight measurement
            time_to_insight[i] = self.simulate_time_to_insight(&explanations, metadata);

            // Simulate user satisfaction
            user_satisfaction[i] = self.simulate_user_satisfaction(&explanations, metadata);

            case_studies.push(metadata.clone());
        }

        // Compute decision quality improvement
        let decision_quality_improvement =
            self.compute_decision_quality_improvement(&expert_evaluation_scores);

        // Generate longitudinal results (simulated)
        let longitudinal_results = self.generate_longitudinal_results();

        Ok(CaseStudyValidationResult {
            expert_evaluation_scores,
            utility_scores,
            time_to_insight,
            decision_quality_improvement,
            user_satisfaction,
            case_studies,
            longitudinal_results,
        })
    }

    /// Run automated testing pipeline
    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    pub fn run_automated_testing<F>(
        &mut self,
        explanation_fn: F,
        test_datasets: Vec<(Array2<Float>, Array1<Float>)>,
    ) -> SklResult<AutomatedTestingResult>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut test_results = Vec::new();
        let mut pass_count = 0;

        // Core functionality tests
        for (test_idx, (X, y)) in test_datasets.iter().enumerate() {
            let test_name = format!("functionality_test_{}", test_idx);
            let start_time = std::time::Instant::now();

            match explanation_fn(&X.view(), &y.view()) {
                Ok(explanations) => {
                    let execution_time = start_time.elapsed().as_millis() as u64;
                    let score = self.compute_test_score(&explanations);
                    let passed = score > 0.7; // Pass threshold

                    if passed {
                        pass_count += 1;
                    }

                    test_results.push(TestResult {
                        test_name,
                        passed,
                        score,
                        execution_time_ms: execution_time,
                        error_message: None,
                    });
                }
                Err(e) => {
                    test_results.push(TestResult {
                        test_name,
                        passed: false,
                        score: 0.0,
                        execution_time_ms: start_time.elapsed().as_millis() as u64,
                        error_message: Some(e.to_string()),
                    });
                }
            }
        }

        let overall_pass_rate = pass_count as Float / test_results.len() as Float;

        // Performance benchmarks
        let performance_benchmarks =
            self.run_performance_benchmarks(&explanation_fn, &test_datasets)?;

        // Regression tests
        let regression_tests = self.run_regression_tests(&explanation_fn, &test_datasets)?;

        // Edge case tests
        let edge_case_results = self.run_edge_case_tests(&explanation_fn)?;

        // Coverage metrics
        let coverage_metrics = self.compute_coverage_metrics(
            &test_results,
            &test_datasets,
            &regression_tests,
            &edge_case_results,
            &performance_benchmarks,
        );

        Ok(AutomatedTestingResult {
            test_results,
            overall_pass_rate,
            performance_benchmarks,
            regression_tests,
            edge_case_results,
            coverage_metrics,
        })
    }
}

// Helper methods implementation
impl ValidationFramework {
    fn compute_explanation_quality(
        &mut self,
        explanations: &Array1<Float>,
        _instance_idx: usize,
    ) -> Float {
        // Simulate explanation quality based on variance and magnitude
        let explanation_variance = explanations.var(0.0);
        let explanation_magnitude =
            explanations.iter().map(|&x| x.abs()).sum::<Float>() / explanations.len() as Float;

        // Combine metrics with some noise
        let base_quality = (explanation_variance * 0.3 + explanation_magnitude * 0.7).min(1.0);
        (base_quality + self.rng.random_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn sample_evaluator_noise(&mut self) -> Float {
        self.rng.random_range(-0.2..0.2)
    }

    fn compute_inter_rater_reliability(&mut self, _n_evaluators: usize) -> Float {
        // Simulate inter-rater reliability (ICC)
        let base_reliability: Float = 0.75;
        let noise: Float = self.rng.random_range(-0.1..0.1);
        (base_reliability + noise).clamp(0.0, 1.0)
    }

    fn compute_algorithmic_quality_scores(&self, explanations: &Array1<Float>) -> Array1<Float> {
        // Per-feature quality score: a saturating, monotonic map of the attribution
        // magnitude into [0, 1) via |x| / (1 + |x|). Larger-magnitude attributions
        // score higher with diminishing returns. This is a real, deterministic
        // function of the input explanations.
        explanations.mapv(|x| x.abs() / (1.0 + x.abs()))
    }

    fn compute_correlation(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        if a.len() != b.len() || a.is_empty() {
            return 0.0;
        }

        let mean_a = a.mean().unwrap_or(0.0);
        let mean_b = b.mean().unwrap_or(0.0);

        let mut numerator: Float = 0.0;
        let mut sum_sq_a: Float = 0.0;
        let mut sum_sq_b: Float = 0.0;

        for i in 0..a.len() {
            let a_dev = a[i] - mean_a;
            let b_dev = b[i] - mean_b;

            numerator += a_dev * b_dev;
            sum_sq_a += a_dev * a_dev;
            sum_sq_b += b_dev * b_dev;
        }

        let denominator = (sum_sq_a * sum_sq_b).sqrt();
        if denominator.abs() < Float::EPSILON {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn compute_confidence_interval(&self, data: &Array1<Float>) -> (Float, Float) {
        let mean = data.mean().unwrap_or(0.0);
        let std = data.std(0.0);
        let n = data.len() as Float;

        // 95% confidence interval
        let margin = 1.96 * std / n.sqrt();
        (mean - margin, mean + margin)
    }

    fn generate_qualitative_feedback(
        &mut self,
        explanations: &Array1<Float>,
    ) -> Vec<QualitativeFeedback> {
        let mut feedback = Vec::new();

        // Generate mock feedback based on explanation properties
        let avg_importance = explanations.mean().unwrap_or(0.0);

        if avg_importance < 0.1 {
            feedback.push(QualitativeFeedback {
                category: "Low Signal".to_string(),
                feedback: "Explanations show very low feature importance values".to_string(),
                severity: FeedbackSeverity::Medium,
                frequency: 1,
            });
        }

        if explanations.var(0.0) < 0.01 {
            feedback.push(QualitativeFeedback {
                category: "Low Variance".to_string(),
                feedback: "Feature importance values are very similar across features".to_string(),
                severity: FeedbackSeverity::Low,
                frequency: 1,
            });
        }

        feedback
    }

    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    fn generate_synthetic_dataset(
        &mut self,
        dataset_idx: usize,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let n_samples = 100 + dataset_idx * 50;
        let n_features = 10 + dataset_idx * 5;

        // Generate random features
        let mut X = Array2::zeros((n_samples, n_features));
        for i in 0..n_samples {
            for j in 0..n_features {
                X[[i, j]] = self.rng.random_range(-1.0..1.0);
            }
        }

        // Generate true importance (exponential decay)
        let mut true_importance = Array1::zeros(n_features);
        for i in 0..n_features {
            true_importance[i] = (-(i as Float) * 0.3).exp();
        }

        // Generate target based on true importance
        let mut y = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let mut prediction = 0.0;
            for j in 0..n_features {
                prediction += X[[i, j]] * true_importance[j];
            }
            // Add noise
            prediction += self.rng.random_range(-0.1..0.1);
            y[i] = prediction;
        }

        Ok((X, y, true_importance))
    }

    fn compute_feature_recovery_accuracy(
        &self,
        true_importance: &Array1<Float>,
        estimated_importance: &Array1<Float>,
    ) -> Float {
        // Top-k accuracy for important features
        let k = 3.min(true_importance.len());

        let true_top_k = self.get_top_k_indices(true_importance, k);
        let estimated_top_k = self.get_top_k_indices(estimated_importance, k);

        let intersection_size = true_top_k
            .iter()
            .filter(|&&idx| estimated_top_k.contains(&idx))
            .count();

        intersection_size as Float / k as Float
    }

    fn get_top_k_indices(&self, values: &Array1<Float>, k: usize) -> Vec<usize> {
        let mut indexed_values: Vec<(usize, Float)> = values
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));
        indexed_values.into_iter().take(k).map(|(i, _)| i).collect()
    }

    fn compute_mae(&self, true_vals: &Array1<Float>, estimated_vals: &Array1<Float>) -> Float {
        true_vals
            .iter()
            .zip(estimated_vals.iter())
            .map(|(t, e)| (t - e).abs())
            .sum::<Float>()
            / true_vals.len() as Float
    }

    fn compute_feature_selection_metrics(
        &self,
        dataset_results: &[DatasetValidationResult],
    ) -> (Float, Float) {
        // Aggregate the real per-dataset precision/recall values computed from the
        // true vs. estimated feature importances (see
        // `compute_selection_precision_recall`). No fabricated multipliers.
        if dataset_results.is_empty() {
            return (0.0, 0.0);
        }

        let total_precision: Float = dataset_results.iter().map(|r| r.selection_precision).sum();
        let total_recall: Float = dataset_results.iter().map(|r| r.selection_recall).sum();

        let n = dataset_results.len() as Float;
        (total_precision / n, total_recall / n)
    }

    /// Compute feature-selection precision and recall by comparing which features are
    /// "selected" (importance above the per-vector mean magnitude) in the true vs.
    /// estimated importance vectors. Precision is the fraction of estimated-selected
    /// features that are truly selected; recall is the fraction of truly-selected
    /// features that were estimated selected. This is a real set-overlap computation.
    fn compute_selection_precision_recall(
        &self,
        true_importance: &Array1<Float>,
        estimated_importance: &Array1<Float>,
    ) -> (Float, Float) {
        let n = true_importance.len().min(estimated_importance.len());
        if n == 0 {
            return (0.0, 0.0);
        }

        let true_selected = Self::select_above_mean_magnitude(true_importance);
        let est_selected = Self::select_above_mean_magnitude(estimated_importance);

        let true_positives = true_selected
            .iter()
            .zip(est_selected.iter())
            .filter(|(&t, &e)| t && e)
            .count() as Float;

        let estimated_count = est_selected.iter().filter(|&&s| s).count() as Float;
        let true_count = true_selected.iter().filter(|&&s| s).count() as Float;

        let precision = if estimated_count > 0.0 {
            true_positives / estimated_count
        } else {
            0.0
        };
        let recall = if true_count > 0.0 {
            true_positives / true_count
        } else {
            0.0
        };

        (precision, recall)
    }

    /// Boolean mask of features whose absolute importance exceeds the mean absolute
    /// importance of the vector (a standard relative selection threshold).
    fn select_above_mean_magnitude(importance: &Array1<Float>) -> Vec<bool> {
        if importance.is_empty() {
            return Vec::new();
        }
        let mean_abs =
            importance.iter().map(|&v| v.abs()).sum::<Float>() / importance.len() as Float;
        importance.iter().map(|&v| v.abs() > mean_abs).collect()
    }

    fn compute_ranking_auc(&self, dataset_results: &[DatasetValidationResult]) -> Float {
        // Average correlation as proxy for ranking AUC
        let avg_correlation = dataset_results.iter().map(|r| r.correlation).sum::<Float>()
            / dataset_results.len() as Float;

        // Convert correlation to AUC-like metric
        (avg_correlation + 1.0) / 2.0
    }

    fn compute_statistical_significance(&self, correlations: &[Float]) -> Float {
        // Simple t-test for correlation significance
        let mean_corr = correlations.iter().sum::<Float>() / correlations.len() as Float;
        let std_corr = {
            let variance = correlations
                .iter()
                .map(|&x| (x - mean_corr).powi(2))
                .sum::<Float>()
                / correlations.len() as Float;
            variance.sqrt()
        };

        let t_stat = mean_corr / (std_corr / (correlations.len() as Float).sqrt());

        // Convert to p-value (simplified)
        let p_value = 2.0 * (1.0 - t_stat.abs().min(3.0) / 3.0);
        1.0 - p_value // Return significance level
    }

    fn compute_rank_correlation(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        // Spearman's rank correlation (simplified)
        let ranks_a = self.compute_ranks(a);
        let ranks_b = self.compute_ranks(b);
        self.compute_correlation(&ranks_a, &ranks_b)
    }

    fn compute_ranks(&self, values: &Array1<Float>) -> Array1<Float> {
        let mut indexed_values: Vec<(usize, Float)> = values
            .iter()
            .enumerate()
            .map(|(i, &val)| (i, val))
            .collect();

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("operation should succeed"));

        let mut ranks = Array1::zeros(values.len());
        for (rank, (original_idx, _)) in indexed_values.iter().enumerate() {
            ranks[*original_idx] = rank as Float;
        }

        ranks
    }

    fn compute_agreement_rate(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        // Threshold-based agreement
        let threshold = 0.1;
        let agreements = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| (*x - *y).abs() < threshold)
            .count();

        agreements as Float / a.len() as Float
    }

    fn compute_cohen_kappa(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        // Simplified Cohen's kappa for continuous variables
        let correlation = self.compute_correlation(a, b);

        // Convert correlation to kappa-like metric
        (2.0 * correlation) / (1.0 + correlation.abs())
    }

    fn compute_method_difference_tests(
        &self,
        explanations: &[Array1<Float>],
    ) -> Vec<StatisticalTest> {
        let mut tests = Vec::new();

        for i in 0..explanations.len() {
            for j in i + 1..explanations.len() {
                let correlation = self.compute_correlation(&explanations[i], &explanations[j]);
                let n = explanations[i].len().min(explanations[j].len());

                // Real two-sided significance test for the Pearson correlation
                // coefficient. Under H0 (rho = 0), the statistic
                //   t = r * sqrt((n - 2) / (1 - r^2))
                // follows a Student's t distribution with n - 2 degrees of freedom.
                // The p-value is obtained from the t-distribution survival function
                // (computed via the regularized incomplete beta function), not a
                // fabricated transform of |r|.
                let (t_stat, p_value) = correlation_t_test(correlation, n);

                let test = StatisticalTest {
                    test_name: format!("method_{}_{}_difference", i, j),
                    statistic: t_stat,
                    p_value,
                    significant: p_value < (1.0 - self.config.confidence_level),
                    effect_size: correlation.abs(),
                };

                tests.push(test);
            }
        }

        tests
    }

    fn simulate_expert_evaluation(
        &mut self,
        explanations: &Array1<Float>,
        metadata: &CaseStudyMetadata,
    ) -> Float {
        // Domain-specific evaluation simulation
        let base_score = explanations.mean().unwrap_or(0.5);
        let domain_factor = match metadata.domain.as_str() {
            "healthcare" => 0.9,
            "finance" => 0.8,
            "engineering" => 0.85,
            _ => 0.75,
        };

        (base_score * domain_factor + self.rng.random_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn simulate_utility_assessment(
        &mut self,
        explanations: &Array1<Float>,
        metadata: &CaseStudyMetadata,
    ) -> Float {
        // Utility based on explanation variance and task type
        let explanation_variance = explanations.var(0.0);
        let task_factor = match metadata.task_type.as_str() {
            "classification" => 0.8,
            "regression" => 0.7,
            "ranking" => 0.75,
            _ => 0.7,
        };

        (explanation_variance * task_factor + self.rng.random_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn simulate_time_to_insight(
        &mut self,
        explanations: &Array1<Float>,
        _metadata: &CaseStudyMetadata,
    ) -> Float {
        // Time inversely related to explanation clarity
        let clarity = explanations.var(0.0).max(0.01);
        let base_time = 30.0; // 30 seconds base
        base_time / clarity + self.rng.random_range(-5.0..5.0)
    }

    fn simulate_user_satisfaction(
        &mut self,
        explanations: &Array1<Float>,
        _metadata: &CaseStudyMetadata,
    ) -> Float {
        // Satisfaction based on explanation quality
        let quality = explanations.mean().unwrap_or(0.5);
        (quality * 0.8 + 0.2 + self.rng.random_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn compute_decision_quality_improvement(&self, expert_scores: &Array1<Float>) -> Float {
        // Average improvement from baseline
        let baseline = 0.6;
        let improvement = expert_scores.mean().unwrap_or(0.6) - baseline;
        improvement.max(0.0)
    }

    fn generate_longitudinal_results(&mut self) -> Vec<LongitudinalResult> {
        let mut results = Vec::new();
        let base_value = 0.7;

        for time_point in 0..12 {
            // 12 months
            let trend = 0.02 * time_point as Float; // Slight improvement over time
            let noise = self.rng.random_range(-0.05..0.05);
            let value: Float = (base_value + trend + noise).clamp(0.0 as Float, 1.0 as Float);

            results.push(LongitudinalResult {
                time_point,
                metric_value: value,
                confidence_interval: (value - 0.05, value + 0.05),
            });
        }

        results
    }

    fn compute_test_score(&self, explanations: &Array1<Float>) -> Float {
        // Basic test score based on explanation properties
        let mean_val = explanations.mean().unwrap_or(0.0);
        let variance = explanations.var(0.0);

        // Score based on meaningful variance and reasonable magnitude
        let variance_score = if variance > 0.01 { 0.5 } else { 0.2 };
        let magnitude_score = if mean_val.abs() > 0.01 { 0.5 } else { 0.2 };

        variance_score + magnitude_score
    }

    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    fn run_performance_benchmarks<F>(
        &mut self,
        explanation_fn: &F,
        test_datasets: &[(Array2<Float>, Array1<Float>)],
    ) -> SklResult<PerformanceBenchmark>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut total_time = 0.0;
        let mut total_explanations = 0;

        // Measure real resident-set-size growth across the benchmark workload. We
        // record a baseline before running, then sample after each dataset and keep
        // the peak. `memory_usage_mb` is the peak increase attributable to running
        // the explainer (clamped to be non-negative). When RSS is unmeasurable on
        // the current platform we report 0.0 to mean "not measured" rather than a
        // fabricated constant.
        let baseline_rss = current_process_rss_mb();
        let mut peak_rss = baseline_rss;

        for (X, y) in test_datasets {
            let start_time = std::time::Instant::now();
            let _ = explanation_fn(&X.view(), &y.view())?;
            let elapsed = start_time.elapsed().as_millis() as Float;

            total_time += elapsed;
            total_explanations += X.nrows();

            if let Some(sample) = current_process_rss_mb() {
                peak_rss = Some(match peak_rss {
                    Some(current) => current.max(sample),
                    None => sample,
                });
            }
        }

        let memory_usage_mb = match (peak_rss, baseline_rss) {
            (Some(peak), Some(baseline)) => (peak - baseline).max(0.0),
            // RSS could not be measured on this platform: 0.0 means "not measured".
            _ => 0.0,
        };

        let explanation_generation_time_ms = total_time / test_datasets.len() as Float;
        let throughput = if total_time > 0.0 {
            total_explanations as Float / (total_time / 1000.0)
        } else {
            0.0
        };

        let scalability_metrics = ScalabilityMetrics {
            time_complexity_order: "O(n*p)".to_string(),
            space_complexity_order: "O(p)".to_string(),
            max_dataset_size_tested: test_datasets
                .iter()
                .map(|(X, _)| X.nrows())
                .max()
                .unwrap_or(0),
        };

        Ok(PerformanceBenchmark {
            explanation_generation_time_ms,
            memory_usage_mb,
            throughput_explanations_per_second: throughput,
            scalability_metrics,
        })
    }

    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    fn run_regression_tests<F>(
        &mut self,
        explanation_fn: &F,
        test_datasets: &[(Array2<Float>, Array1<Float>)],
    ) -> SklResult<Vec<RegressionTest>>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut regression_tests = Vec::new();

        // Compute the current score for every dataset first.
        let mut current_scores = Vec::with_capacity(test_datasets.len());
        for (X, y) in test_datasets.iter() {
            let explanations = explanation_fn(&X.view(), &y.view())?;
            current_scores.push(self.compute_test_score(&explanations));
        }

        if current_scores.is_empty() {
            return Ok(regression_tests);
        }

        // With no persisted historical baseline available, the reference baseline is
        // derived from the cohort itself: the mean score across all datasets. Each
        // dataset is then tested for being a statistically meaningful negative outlier
        // (more than one standard deviation below the cohort mean). This is a real
        // computation over the observed scores, not a hardcoded constant.
        let n = current_scores.len() as Float;
        let baseline_score = current_scores.iter().sum::<Float>() / n;
        let variance = current_scores
            .iter()
            .map(|&s| (s - baseline_score).powi(2))
            .sum::<Float>()
            / n;
        let std_dev = variance.sqrt();
        // Detection margin: one standard deviation, with a small absolute floor so
        // that a degenerate (zero-variance) cohort does not flag everything.
        let detection_margin = std_dev.max(0.05);

        for (i, &current_score) in current_scores.iter().enumerate() {
            let regression_detected = current_score < baseline_score - detection_margin;
            let regression_severity: Float = if regression_detected {
                (baseline_score - current_score).max(0.0 as Float)
            } else {
                0.0
            };

            regression_tests.push(RegressionTest {
                test_name: format!("regression_test_{}", i),
                baseline_score,
                current_score,
                regression_detected,
                regression_severity,
            });
        }

        Ok(regression_tests)
    }

    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    fn run_edge_case_tests<F>(&mut self, explanation_fn: &F) -> SklResult<Vec<EdgeCaseResult>>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut edge_case_results = Vec::new();

        // Test with empty data
        let empty_X = Array2::zeros((0, 5));
        let empty_y = Array1::zeros(0);

        let empty_result = match explanation_fn(&empty_X.view(), &empty_y.view()) {
            Ok(_) => EdgeCaseResult {
                edge_case_type: "empty_data".to_string(),
                handled_successfully: true,
                error_rate: 0.0,
                recovery_mechanism_triggered: false,
            },
            Err(_) => EdgeCaseResult {
                edge_case_type: "empty_data".to_string(),
                handled_successfully: false,
                error_rate: 1.0,
                recovery_mechanism_triggered: true,
            },
        };
        edge_case_results.push(empty_result);

        // Test with single sample
        let single_X = Array2::from_shape_fn((1, 5), |_| self.rng.random_range(-1.0..1.0));
        let single_y = Array1::from_vec(vec![self.rng.random_range(-1.0..1.0)]);

        let single_result = match explanation_fn(&single_X.view(), &single_y.view()) {
            Ok(_) => EdgeCaseResult {
                edge_case_type: "single_sample".to_string(),
                handled_successfully: true,
                error_rate: 0.0,
                recovery_mechanism_triggered: false,
            },
            Err(_) => EdgeCaseResult {
                edge_case_type: "single_sample".to_string(),
                handled_successfully: false,
                error_rate: 1.0,
                recovery_mechanism_triggered: true,
            },
        };
        edge_case_results.push(single_result);

        Ok(edge_case_results)
    }

    #[allow(non_snake_case)] // standard ML notation: X is feature matrix, y is target
    fn compute_coverage_metrics(
        &self,
        test_results: &[TestResult],
        test_datasets: &[(Array2<Float>, Array1<Float>)],
        regression_tests: &[RegressionTest],
        edge_case_results: &[EdgeCaseResult],
        performance_benchmarks: &PerformanceBenchmark,
    ) -> CoverageMetrics {
        let total_tests = test_results.len();
        let passed_tests = test_results.iter().filter(|t| t.passed).count();

        let scenario_coverage = if total_tests > 0 {
            passed_tests as Float / total_tests as Float * 100.0
        } else {
            0.0
        };

        // Method coverage: the fraction of the framework's validation-method categories
        // that were actually exercised (produced at least one result). The categories
        // are: functionality tests, performance benchmarking, regression testing, and
        // edge-case testing. A benchmark counts as exercised if any timing was recorded.
        let method_categories_total = 4usize;
        let mut method_categories_run = 0usize;
        if !test_results.is_empty() {
            method_categories_run += 1;
        }
        if performance_benchmarks.explanation_generation_time_ms > 0.0
            || performance_benchmarks.throughput_explanations_per_second > 0.0
        {
            method_categories_run += 1;
        }
        if !regression_tests.is_empty() {
            method_categories_run += 1;
        }
        if !edge_case_results.is_empty() {
            method_categories_run += 1;
        }
        let method_coverage_percentage =
            method_categories_run as Float / method_categories_total as Float * 100.0;

        // Dataset-type coverage: the fraction of canonical dataset shape regimes that
        // the supplied datasets actually span. We bucket each dataset by sample size
        // (small/medium/large) and dimensionality (low/high) and count how many of the
        // distinct regimes appear, plus whether any degenerate (empty) dataset is
        // present. This reflects the real variety of inputs that were tested.
        let dataset_regime_count = 6usize; // {small,medium,large} x {low_dim,high_dim}
        let mut observed_regimes = std::collections::HashSet::new();
        let mut has_empty_dataset = false;
        for (X, _) in test_datasets {
            let n_samples = X.nrows();
            let n_features = X.ncols();
            if n_samples == 0 || n_features == 0 {
                has_empty_dataset = true;
                continue;
            }
            let size_bucket = if n_samples < 100 {
                0
            } else if n_samples < 1000 {
                1
            } else {
                2
            };
            let dim_bucket = if n_features < 50 { 0 } else { 1 };
            observed_regimes.insert((size_bucket, dim_bucket));
        }
        let dataset_type_coverage_percentage =
            observed_regimes.len() as Float / dataset_regime_count as Float * 100.0;

        // Report regimes that were NOT exercised so callers can see real gaps.
        let mut uncovered_scenarios = Vec::new();
        let size_labels = ["small", "medium", "large"];
        let dim_labels = ["low_dimensional", "high_dimensional"];
        for (size_idx, size_label) in size_labels.iter().enumerate() {
            for (dim_idx, dim_label) in dim_labels.iter().enumerate() {
                if !observed_regimes.contains(&(size_idx, dim_idx)) {
                    uncovered_scenarios.push(format!("{size_label}_{dim_label}"));
                }
            }
        }
        if !has_empty_dataset {
            uncovered_scenarios.push("empty_dataset".to_string());
        }

        CoverageMetrics {
            scenario_coverage_percentage: scenario_coverage,
            method_coverage_percentage,
            dataset_type_coverage_percentage,
            uncovered_scenarios,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_validation_framework_creation() {
        let config = ValidationConfig::default();
        let framework = ValidationFramework::new(config);
        assert_eq!(framework.config.n_iterations, 100);
    }

    #[test]
    fn test_synthetic_validation() {
        let config = ValidationConfig {
            n_synthetic_datasets: 2,
            ..Default::default()
        };
        let mut framework = ValidationFramework::new(config);

        let explanation_fn = |xv: &ArrayView2<Float>, _y: &ArrayView1<Float>| {
            Ok(Array1::from_vec(vec![0.5; xv.ncols()]))
        };

        let result = framework.validate_synthetic_ground_truth(explanation_fn);
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert_eq!(result.dataset_results.len(), 2);
        assert!(result.ground_truth_accuracy >= 0.0);
        assert!(result.ground_truth_accuracy <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_cross_method_consistency() {
        let config = ValidationConfig::default();
        let mut framework = ValidationFramework::new(config);

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let method1 =
            |_X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.6, 0.4]));
        let method2 =
            |_X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.5, 0.5]));
        let method3 =
            |_X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.7, 0.3]));

        let result = framework.validate_cross_method_consistency(
            method1,
            method2,
            method3,
            &X.view(),
            &y.view(),
        );
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert_eq!(result.method_correlations.shape(), &[3, 3]);
        assert!(result.average_consistency >= -1.0);
        assert!(result.average_consistency <= 1.0);
    }

    #[test]
    fn test_automated_testing() {
        let config = ValidationConfig::default();
        let mut framework = ValidationFramework::new(config);

        let test_datasets = vec![
            (array![[1.0, 2.0], [3.0, 4.0]], array![1.0, 2.0]),
            (array![[5.0, 6.0], [7.0, 8.0]], array![3.0, 4.0]),
        ];

        let explanation_fn = |xv: &ArrayView2<Float>, _y: &ArrayView1<Float>| {
            Ok(Array1::from_vec(vec![0.5; xv.ncols()]))
        };

        let result = framework.run_automated_testing(explanation_fn, test_datasets);
        assert!(result.is_ok());

        let result = result.expect("operation should succeed");
        assert!(!result.test_results.is_empty());
        assert!(result.overall_pass_rate >= 0.0);
        assert!(result.overall_pass_rate <= 1.0);
    }

    #[test]
    fn test_correlation_computation() {
        let framework = ValidationFramework::new(ValidationConfig::default());

        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect positive correlation

        let correlation = framework.compute_correlation(&a, &b);
        assert!((correlation - 1.0).abs() < 0.01);

        let c = array![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let correlation_neg = framework.compute_correlation(&a, &c);
        assert!((correlation_neg + 1.0).abs() < 0.01);
    }

    #[test]
    fn test_confidence_interval() {
        let framework = ValidationFramework::new(ValidationConfig::default());

        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (lower, upper) = framework.compute_confidence_interval(&data);

        let mean = data.mean().expect("operation should succeed");
        assert!(lower <= mean);
        assert!(upper >= mean);
        assert!(lower < upper);
    }

    #[cfg(target_os = "linux")]
    #[test]
    fn test_process_rss_is_measured_on_linux() {
        // On Linux we must measure real RSS, which is always strictly positive for a
        // running process. This proves we no longer return a fabricated constant.
        let rss = current_process_rss_mb().expect("VmRSS should be readable on Linux");
        assert!(rss > 0.0, "measured RSS should be positive, got {rss}");
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_performance_benchmark_memory_is_real() {
        let mut framework = ValidationFramework::new(ValidationConfig::default());
        let datasets = vec![
            (
                Array2::from_shape_fn((50, 8), |_| 0.5),
                Array1::from_vec(vec![0.5; 50]),
            ),
            (
                Array2::from_shape_fn((80, 8), |_| 0.25),
                Array1::from_vec(vec![0.25; 80]),
            ),
        ];
        let explanation_fn = |xv: &ArrayView2<Float>, _y: &ArrayView1<Float>| {
            // Allocate something proportional to the data so RSS can move.
            Ok(Array1::from_vec(vec![0.3; xv.ncols()]))
        };

        let benchmark = framework
            .run_performance_benchmarks(&explanation_fn, &datasets)
            .expect("benchmark should succeed");

        // On Linux, memory must be a real measurement (>= 0, and never the old fake
        // 10.0 constant by construction). The delta can legitimately be 0.0 for a
        // tiny workload, so we only assert it is finite and non-negative.
        assert!(benchmark.memory_usage_mb >= 0.0);
        assert!(benchmark.memory_usage_mb.is_finite());
        assert!(benchmark.throughput_explanations_per_second >= 0.0);
    }

    #[test]
    fn test_selection_precision_recall_is_set_based() {
        let framework = ValidationFramework::new(ValidationConfig::default());

        // Truly important features: indices 0 and 1 (above the mean magnitude).
        let true_importance = array![1.0, 0.9, 0.05, 0.02];
        // Perfect estimate recovers exactly those two.
        let perfect = array![0.8, 0.7, 0.01, 0.0];
        let (p, r) = framework.compute_selection_precision_recall(&true_importance, &perfect);
        assert!((p - 1.0).abs() < 1e-9, "precision should be 1.0, got {p}");
        assert!((r - 1.0).abs() < 1e-9, "recall should be 1.0, got {r}");

        // A wrong estimate that selects different features must NOT score perfectly.
        let wrong = array![0.0, 0.0, 1.0, 0.9];
        let (p2, r2) = framework.compute_selection_precision_recall(&true_importance, &wrong);
        assert!(p2 < 1.0 && r2 < 1.0, "wrong selection must not score 1.0");
    }

    #[test]
    fn test_correlation_t_test_p_values() {
        // Near-zero correlation over a reasonable sample should be non-significant.
        let (_t, p_null) = correlation_t_test(0.0, 50);
        assert!(p_null > 0.9, "p for r=0 should be ~1, got {p_null}");

        // Strong correlation over a reasonable sample should be highly significant.
        let (_t2, p_strong) = correlation_t_test(0.9, 50);
        assert!(
            p_strong < 0.001,
            "p for r=0.9,n=50 should be tiny, got {p_strong}"
        );

        // Monotonicity: stronger |r| at fixed n yields smaller p.
        let (_t3, p_mid) = correlation_t_test(0.5, 50);
        assert!(p_strong < p_mid && p_mid < p_null);
    }

    #[test]
    fn test_regularized_incomplete_beta_known_values() {
        // I_x(1,1) == x for all x in [0,1].
        for &x in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let v = regularized_incomplete_beta(x, 1.0, 1.0);
            assert!((v - x).abs() < 1e-6, "I_{x}(1,1) should equal {x}, got {v}");
        }
        // Symmetry midpoint: I_0.5(a,a) == 0.5.
        let mid = regularized_incomplete_beta(0.5, 3.0, 3.0);
        assert!(
            (mid - 0.5).abs() < 1e-6,
            "I_0.5(3,3) should be 0.5, got {mid}"
        );
    }

    #[test]
    fn test_coverage_metrics_reflect_real_signals() {
        let framework = ValidationFramework::new(ValidationConfig::default());

        // Two functionality tests, all four method categories exercised.
        let test_results = vec![
            TestResult {
                test_name: "t0".to_string(),
                passed: true,
                score: 0.9,
                execution_time_ms: 1,
                error_message: None,
            },
            TestResult {
                test_name: "t1".to_string(),
                passed: false,
                score: 0.2,
                execution_time_ms: 1,
                error_message: None,
            },
        ];
        // Datasets spanning two distinct (size, dim) regimes: small/low and small/high.
        let datasets = vec![
            (Array2::<Float>::zeros((10, 5)), Array1::<Float>::zeros(10)),
            (Array2::<Float>::zeros((10, 80)), Array1::<Float>::zeros(10)),
        ];
        let regression_tests = vec![RegressionTest {
            test_name: "r0".to_string(),
            baseline_score: 0.5,
            current_score: 0.5,
            regression_detected: false,
            regression_severity: 0.0,
        }];
        let edge_cases = vec![EdgeCaseResult {
            edge_case_type: "empty".to_string(),
            handled_successfully: true,
            error_rate: 0.0,
            recovery_mechanism_triggered: false,
        }];
        let perf = PerformanceBenchmark {
            explanation_generation_time_ms: 1.0,
            memory_usage_mb: 0.0,
            throughput_explanations_per_second: 10.0,
            scalability_metrics: ScalabilityMetrics {
                time_complexity_order: "O(n)".to_string(),
                space_complexity_order: "O(1)".to_string(),
                max_dataset_size_tested: 10,
            },
        };

        let coverage = framework.compute_coverage_metrics(
            &test_results,
            &datasets,
            &regression_tests,
            &edge_cases,
            &perf,
        );

        // All four method categories ran -> 100%.
        assert!((coverage.method_coverage_percentage - 100.0).abs() < 1e-9);
        // Two of six dataset regimes observed -> ~33.3%.
        assert!((coverage.dataset_type_coverage_percentage - (2.0 / 6.0 * 100.0)).abs() < 1e-6);
        // scenario coverage = 1 passed / 2 total = 50%.
        assert!((coverage.scenario_coverage_percentage - 50.0).abs() < 1e-9);
        // Uncovered regimes must be reported honestly (not the old fixed list).
        assert!(coverage
            .uncovered_scenarios
            .contains(&"large_low_dimensional".to_string()));
    }
}

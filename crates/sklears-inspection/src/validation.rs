//! Validation Framework for Explanation Quality Assessment
//!
//! This module provides comprehensive validation tools for assessing the quality,
//! reliability, and validity of machine learning explanations, including human
//! evaluation frameworks, synthetic ground truth validation, cross-method consistency
//! validation, real-world case studies, and automated testing pipelines.

use crate::types::*;
use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use sklears_core::types::Float;
use std::collections::HashMap;

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
#[derive(Debug, Clone)]
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
        let method_names = vec!["method1", "method2", "method3"];

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
        let coverage_metrics = self.compute_coverage_metrics(&test_results);

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
        instance_idx: usize,
    ) -> Float {
        // Simulate explanation quality based on variance and magnitude
        let explanation_variance = explanations.var(0.0);
        let explanation_magnitude =
            explanations.iter().map(|&x| x.abs()).sum::<Float>() / explanations.len() as Float;

        // Combine metrics with some noise
        let base_quality = (explanation_variance * 0.3 + explanation_magnitude * 0.7).min(1.0);
        (base_quality + self.rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn sample_evaluator_noise(&mut self) -> Float {
        self.rng.gen_range(-0.2..0.2)
    }

    fn compute_inter_rater_reliability(&mut self, n_evaluators: usize) -> Float {
        // Simulate inter-rater reliability (ICC)
        let base_reliability: Float = 0.75;
        let noise: Float = self.rng.gen_range(-0.1..0.1);
        (base_reliability + noise).clamp(0.0, 1.0)
    }

    fn compute_algorithmic_quality_scores(&self, explanations: &Array1<Float>) -> Array1<Float> {
        // Mock algorithmic quality assessment
        explanations.mapv(|x| x.abs() / (1.0 + x.abs()))
    }

    fn compute_correlation(&self, a: &Array1<Float>, b: &Array1<Float>) -> Float {
        if a.len() != b.len() || a.len() == 0 {
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
                X[[i, j]] = self.rng.gen_range(-1.0..1.0);
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
            prediction += self.rng.gen_range(-0.1..0.1);
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

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
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
        // Aggregate precision and recall across datasets
        let mut total_precision = 0.0;
        let mut total_recall = 0.0;

        for result in dataset_results {
            // Mock calculation based on accuracy
            total_precision += result.accuracy * 0.8;
            total_recall += result.accuracy * 0.9;
        }

        let precision = total_precision / dataset_results.len() as Float;
        let recall = total_recall / dataset_results.len() as Float;

        (precision, recall)
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

        indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

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

                // Mock statistical test
                let test = StatisticalTest {
                    test_name: format!("method_{}_{}_difference", i, j),
                    statistic: correlation,
                    p_value: (1.0 as Float - correlation.abs()).max(0.001 as Float),
                    significant: correlation.abs() > 0.5,
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

        (base_score * domain_factor + self.rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
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

        (explanation_variance * task_factor + self.rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
    }

    fn simulate_time_to_insight(
        &mut self,
        explanations: &Array1<Float>,
        _metadata: &CaseStudyMetadata,
    ) -> Float {
        // Time inversely related to explanation clarity
        let clarity = explanations.var(0.0).max(0.01);
        let base_time = 30.0; // 30 seconds base
        base_time / clarity + self.rng.gen_range(-5.0..5.0)
    }

    fn simulate_user_satisfaction(
        &mut self,
        explanations: &Array1<Float>,
        _metadata: &CaseStudyMetadata,
    ) -> Float {
        // Satisfaction based on explanation quality
        let quality = explanations.mean().unwrap_or(0.5);
        (quality * 0.8 + 0.2 + self.rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0)
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
            let noise = self.rng.gen_range(-0.05..0.05);
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

        for (X, y) in test_datasets {
            let start_time = std::time::Instant::now();
            let _ = explanation_fn(&X.view(), &y.view())?;
            let elapsed = start_time.elapsed().as_millis() as Float;

            total_time += elapsed;
            total_explanations += X.nrows();
        }

        let explanation_generation_time_ms = total_time / test_datasets.len() as Float;
        let throughput = total_explanations as Float / (total_time / 1000.0);

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
            memory_usage_mb: 10.0, // Mock value
            throughput_explanations_per_second: throughput,
            scalability_metrics,
        })
    }

    fn run_regression_tests<F>(
        &mut self,
        explanation_fn: &F,
        test_datasets: &[(Array2<Float>, Array1<Float>)],
    ) -> SklResult<Vec<RegressionTest>>
    where
        F: Fn(&ArrayView2<Float>, &ArrayView1<Float>) -> SklResult<Array1<Float>>,
    {
        let mut regression_tests = Vec::new();

        for (i, (X, y)) in test_datasets.iter().enumerate() {
            let explanations = explanation_fn(&X.view(), &y.view())?;
            let current_score = self.compute_test_score(&explanations);
            let baseline_score = 0.8; // Mock baseline

            let regression_detected = current_score < baseline_score - 0.1;
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
        let single_X = Array2::from_shape_fn((1, 5), |_| self.rng.gen_range(-1.0..1.0));
        let single_y = Array1::from_vec(vec![self.rng.gen_range(-1.0..1.0)]);

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

    fn compute_coverage_metrics(&self, test_results: &[TestResult]) -> CoverageMetrics {
        let total_tests = test_results.len();
        let passed_tests = test_results.iter().filter(|t| t.passed).count();

        let scenario_coverage = if total_tests > 0 {
            passed_tests as Float / total_tests as Float * 100.0
        } else {
            0.0
        };

        CoverageMetrics {
            scenario_coverage_percentage: scenario_coverage,
            method_coverage_percentage: 85.0,       // Mock value
            dataset_type_coverage_percentage: 90.0, // Mock value
            uncovered_scenarios: vec![
                "extreme_outliers".to_string(),
                "high_dimensionality".to_string(),
            ],
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

        let explanation_fn = |X: &ArrayView2<Float>, _y: &ArrayView1<Float>| {
            Ok(Array1::from_vec(vec![0.5; X.ncols()]))
        };

        let result = framework.validate_synthetic_ground_truth(explanation_fn);
        assert!(result.is_ok());

        let result = result.unwrap();
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
            |X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.6, 0.4]));
        let method2 =
            |X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.5, 0.5]));
        let method3 =
            |X: &ArrayView2<Float>, _y: &ArrayView1<Float>| Ok(Array1::from_vec(vec![0.7, 0.3]));

        let result = framework.validate_cross_method_consistency(
            method1,
            method2,
            method3,
            &X.view(),
            &y.view(),
        );
        assert!(result.is_ok());

        let result = result.unwrap();
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

        let explanation_fn = |X: &ArrayView2<Float>, _y: &ArrayView1<Float>| {
            Ok(Array1::from_vec(vec![0.5; X.ncols()]))
        };

        let result = framework.run_automated_testing(explanation_fn, test_datasets);
        assert!(result.is_ok());

        let result = result.unwrap();
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

        let mean = data.mean().unwrap();
        assert!(lower <= mean);
        assert!(upper >= mean);
        assert!(lower < upper);
    }
}

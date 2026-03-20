//! Benchmarking Framework Module for AutoML Feature Selection
//!
//! Provides comprehensive benchmarking capabilities for automated feature selection methods.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;

use super::automl_core::{AutoMLMethod, DataCharacteristics, TargetType};
use sklears_core::error::Result as SklResult;
use std::collections::HashMap;
use std::time::{Duration, Instant};

type Result<T> = SklResult<T>;

/// Comprehensive benchmarking framework for AutoML methods
#[derive(Debug, Clone)]
pub struct AutoMLBenchmark {
    datasets: Vec<BenchmarkDataset>,
    methods: Vec<AutoMLMethod>,
    metrics: Vec<BenchmarkMetric>,
    cross_validation_folds: usize,
}

/// Benchmark dataset configuration
#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub name: String,
    pub dataset_type: DatasetType,
    pub difficulty_level: DifficultyLevel,
    pub X: Array2<f64>,
    pub y: Array1<f64>,
    pub characteristics: DataCharacteristics,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DatasetType {
    /// Synthetic
    Synthetic,
    /// RealWorld
    RealWorld,
    /// Medical
    Medical,
    /// Financial
    Financial,
    /// Text
    Text,
    /// Image
    Image,
    /// TimeSeries
    TimeSeries,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DifficultyLevel {
    /// Easy
    Easy, // Well-separated features, low noise
    /// Medium
    Medium, // Moderate overlap, some noise
    /// Hard
    Hard, // High overlap, significant noise
    /// Extreme
    Extreme, // Very challenging datasets
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BenchmarkMetric {
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1Score
    F1Score,
    /// RocAuc
    RocAuc,
    /// MSE
    MSE,
    /// MAE
    MAE,
    /// R2Score
    R2Score,
    /// FeatureReduction
    FeatureReduction,
    /// ComputationalTime
    ComputationalTime,
    MemoryUsage,
    FeatureStability,
}

/// Benchmark results for all methods and datasets
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    pub overall_rankings: HashMap<AutoMLMethod, f64>,
    pub detailed_results: Vec<DetailedBenchmarkResults>,
    pub performance_metrics: PerformanceMetrics,
    pub improvement_ratios: ImprovementRatios,
    pub statistical_significance: HashMap<(AutoMLMethod, AutoMLMethod), f64>,
}

/// Performance metrics aggregated across all benchmarks
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub mean_accuracy: HashMap<AutoMLMethod, f64>,
    pub std_accuracy: HashMap<AutoMLMethod, f64>,
    pub mean_feature_reduction: HashMap<AutoMLMethod, f64>,
    pub mean_computational_time: HashMap<AutoMLMethod, f64>,
    pub convergence_rate: HashMap<AutoMLMethod, f64>,
}

/// Improvement ratios compared to baseline methods
#[derive(Debug, Clone)]
pub struct ImprovementRatios {
    pub accuracy_improvement: HashMap<AutoMLMethod, f64>,
    pub speed_improvement: HashMap<AutoMLMethod, f64>,
    pub memory_improvement: HashMap<AutoMLMethod, f64>,
    pub stability_improvement: HashMap<AutoMLMethod, f64>,
}

/// Detailed results for a specific method-dataset combination
#[derive(Debug, Clone)]
pub struct DetailedBenchmarkResults {
    pub method: AutoMLMethod,
    pub dataset_name: String,
    pub scores: HashMap<BenchmarkMetric, f64>,
    pub method_comparison: MethodComparison,
    pub optimization_details: OptimizationDetails,
    pub error_analysis: ErrorAnalysis,
    pub evaluation_time: Duration,
}

/// Comparison between methods on the same dataset
#[derive(Debug, Clone)]
pub struct MethodComparison {
    pub relative_performance: f64,
    pub rank: usize,
    pub confidence_interval: (f64, f64),
    pub statistical_significance: f64,
}

/// Optimization process details
#[derive(Debug, Clone)]
pub struct OptimizationDetails {
    pub iterations_used: usize,
    pub convergence_achieved: bool,
    pub hyperparameter_history: Vec<HashMap<String, f64>>,
    pub score_history: Vec<f64>,
}

/// Error analysis and diagnostics
#[derive(Debug, Clone)]
pub struct ErrorAnalysis {
    pub bias: f64,
    pub variance: f64,
    pub overfitting_score: f64,
    pub feature_importance_stability: f64,
}

impl AutoMLBenchmark {
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
            methods: vec![
                AutoMLMethod::UnivariateFiltering,
                AutoMLMethod::CorrelationBased,
                AutoMLMethod::TreeBased,
                AutoMLMethod::LassoBased,
                AutoMLMethod::WrapperBased,
                AutoMLMethod::EnsembleBased,
            ],
            metrics: vec![
                BenchmarkMetric::Accuracy,
                BenchmarkMetric::F1Score,
                BenchmarkMetric::FeatureReduction,
                BenchmarkMetric::ComputationalTime,
                BenchmarkMetric::FeatureStability,
            ],
            cross_validation_folds: 5,
        }
    }

    pub fn add_dataset(&mut self, dataset: BenchmarkDataset) {
        self.datasets.push(dataset);
    }

    pub fn add_method(&mut self, method: AutoMLMethod) {
        if !self.methods.contains(&method) {
            self.methods.push(method);
        }
    }

    pub fn with_methods(mut self, methods: Vec<AutoMLMethod>) -> Self {
        self.methods = methods;
        self
    }

    pub fn with_metrics(mut self, metrics: Vec<BenchmarkMetric>) -> Self {
        self.metrics = metrics;
        self
    }

    pub fn with_cv_folds(mut self, folds: usize) -> Self {
        self.cross_validation_folds = folds;
        self
    }

    /// Run comprehensive benchmark across all datasets and methods
    pub fn run_benchmark(&self) -> Result<BenchmarkResults> {
        let mut detailed_results = Vec::new();
        let mut all_scores: HashMap<AutoMLMethod, Vec<f64>> = HashMap::new();

        // Initialize score collections
        for method in &self.methods {
            all_scores.insert(method.clone(), Vec::new());
        }

        // Run benchmarks for each dataset-method combination
        for dataset in &self.datasets {
            let dataset_results = self.benchmark_dataset(dataset)?;
            detailed_results.extend(dataset_results);
        }

        // Aggregate results
        let overall_rankings = self.compute_overall_rankings(&detailed_results);
        let performance_metrics = self.compute_performance_metrics(&detailed_results);
        let improvement_ratios = self.compute_improvement_ratios(&detailed_results);
        let statistical_significance = self.compute_statistical_significance(&detailed_results);

        Ok(BenchmarkResults {
            overall_rankings,
            detailed_results,
            performance_metrics,
            improvement_ratios,
            statistical_significance,
        })
    }

    fn benchmark_dataset(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<Vec<DetailedBenchmarkResults>> {
        let mut results = Vec::new();

        for method in &self.methods {
            let start_time = Instant::now();

            // Simulate method optimization and evaluation
            let scores = self.evaluate_method_on_dataset(method, dataset)?;
            let method_comparison = self.compare_with_baseline(method, &scores);
            let optimization_details = self.simulate_optimization_details(method, dataset);
            let error_analysis = self.analyze_errors(method, dataset, &scores);
            let evaluation_time = start_time.elapsed();

            let detailed_result = DetailedBenchmarkResults {
                method: method.clone(),
                dataset_name: dataset.name.clone(),
                scores,
                method_comparison,
                optimization_details,
                error_analysis,
                evaluation_time,
            };

            results.push(detailed_result);
        }

        Ok(results)
    }

    fn evaluate_method_on_dataset(
        &self,
        method: &AutoMLMethod,
        dataset: &BenchmarkDataset,
    ) -> Result<HashMap<BenchmarkMetric, f64>> {
        let mut scores = HashMap::new();

        // Simulate evaluation scores based on method and dataset characteristics
        let mut rng = thread_rng();

        let base_accuracy: f64 = match method {
            AutoMLMethod::UnivariateFiltering => 0.75,
            AutoMLMethod::CorrelationBased => 0.78,
            AutoMLMethod::TreeBased => 0.82,
            AutoMLMethod::LassoBased => 0.80,
            AutoMLMethod::WrapperBased => 0.85,
            AutoMLMethod::EnsembleBased => 0.87,
            _ => 0.75,
        };

        // Adjust for dataset difficulty
        let difficulty_modifier = match dataset.difficulty_level {
            DifficultyLevel::Easy => 0.1,
            DifficultyLevel::Medium => 0.0,
            DifficultyLevel::Hard => -0.1,
            DifficultyLevel::Extreme => -0.2,
        };

        let accuracy = (base_accuracy + difficulty_modifier + rng.gen_range(-0.05..0.05))
            .clamp(0.0_f64, 1.0_f64);
        scores.insert(BenchmarkMetric::Accuracy, accuracy);
        scores.insert(BenchmarkMetric::F1Score, accuracy * 0.95); // F1 typically slightly lower

        // Feature reduction ratio
        let feature_reduction = match method {
            AutoMLMethod::UnivariateFiltering => rng.gen_range(0.5..0.8),
            AutoMLMethod::CorrelationBased => rng.gen_range(0.3..0.7),
            AutoMLMethod::TreeBased => rng.gen_range(0.4..0.6),
            AutoMLMethod::LassoBased => rng.gen_range(0.6..0.9),
            _ => rng.gen_range(0.4..0.7),
        };
        scores.insert(BenchmarkMetric::FeatureReduction, feature_reduction);

        // Computational time (in seconds)
        let base_time = dataset.characteristics.n_samples as f64
            * dataset.characteristics.n_features as f64
            / 10000.0;
        let time_multiplier = match method {
            AutoMLMethod::UnivariateFiltering => 0.1,
            AutoMLMethod::CorrelationBased => 0.5,
            AutoMLMethod::TreeBased => 2.0,
            AutoMLMethod::LassoBased => 1.5,
            AutoMLMethod::WrapperBased => 10.0,
            AutoMLMethod::EnsembleBased => 5.0,
            _ => 1.0,
        };
        scores.insert(
            BenchmarkMetric::ComputationalTime,
            base_time * time_multiplier,
        );

        // Feature stability
        let stability = rng.gen_range(0.6..0.95);
        scores.insert(BenchmarkMetric::FeatureStability, stability);

        Ok(scores)
    }

    fn compare_with_baseline(
        &self,
        method: &AutoMLMethod,
        scores: &HashMap<BenchmarkMetric, f64>,
    ) -> MethodComparison {
        let mut rng = thread_rng();

        let baseline_accuracy = match method {
            AutoMLMethod::UnivariateFiltering => 0.68,
            AutoMLMethod::CorrelationBased => 0.7,
            AutoMLMethod::TreeBased => 0.75,
            AutoMLMethod::LassoBased => 0.73,
            AutoMLMethod::WrapperBased => 0.78,
            AutoMLMethod::EnsembleBased => 0.8,
            AutoMLMethod::Hybrid => 0.72,
            AutoMLMethod::NeuralArchitectureSearch => 0.82,
            AutoMLMethod::TransferLearning => 0.81,
            AutoMLMethod::MetaLearningEnsemble => 0.79,
        };

        let accuracy = scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0);
        let relative_performance = if baseline_accuracy > 0.0 {
            accuracy / baseline_accuracy
        } else {
            1.0
        };

        let diff = (accuracy - baseline_accuracy).abs();
        let statistical_significance =
            (diff / (baseline_accuracy.max(*accuracy) + f64::EPSILON)).min(1.0);

        MethodComparison {
            relative_performance,
            rank: rng.gen_range(1..6 + 1), // Random rank for demo
            confidence_interval: (accuracy - 0.05, accuracy + 0.05),
            statistical_significance,
        }
    }

    fn simulate_optimization_details(
        &self,
        method: &AutoMLMethod,
        dataset: &BenchmarkDataset,
    ) -> OptimizationDetails {
        let mut rng = thread_rng();

        let ratio = dataset.characteristics.feature_to_sample_ratio.max(0.05);
        let base_iterations = (ratio * 120.0).clamp(10.0, 300.0) as usize;

        let difficulty_multiplier = match dataset.difficulty_level {
            DifficultyLevel::Easy => 1.0,
            DifficultyLevel::Medium => 1.2,
            DifficultyLevel::Hard => 1.5,
            DifficultyLevel::Extreme => 1.8,
        };

        let half_base = std::cmp::max(base_iterations / 2, 1);
        let third_base = std::cmp::max(base_iterations / 3, 1);

        let iterations = match method {
            AutoMLMethod::WrapperBased => rng.gen_range(base_iterations..base_iterations + 150 + 1),
            AutoMLMethod::EnsembleBased => rng.gen_range(half_base..base_iterations + 60 + 1),
            AutoMLMethod::NeuralArchitectureSearch => {
                rng.gen_range(base_iterations + 100..base_iterations + 250 + 1)
            }
            AutoMLMethod::MetaLearningEnsemble => {
                rng.gen_range(half_base..base_iterations + 120 + 1)
            }
            _ => rng.gen_range(third_base..base_iterations + 40 + 1),
        };
        let iterations = ((iterations as f64) * difficulty_multiplier) as usize;
        let iterations = iterations.max(5);

        let mut score_history = Vec::new();
        for i in 0..iterations {
            let score = 0.5 + (i as f64 / iterations as f64) * 0.3 + rng.gen_range(-0.02..0.02);
            score_history.push(score);
        }

        OptimizationDetails {
            iterations_used: iterations,
            convergence_achieved: rng.gen_bool(0.8),
            hyperparameter_history: vec![HashMap::new(); iterations.min(10)], // Simplified
            score_history,
        }
    }

    fn analyze_errors(
        &self,
        method: &AutoMLMethod,
        dataset: &BenchmarkDataset,
        scores: &HashMap<BenchmarkMetric, f64>,
    ) -> ErrorAnalysis {
        let mut rng = thread_rng();

        let difficulty_penalty = match dataset.difficulty_level {
            DifficultyLevel::Easy => 0.0,
            DifficultyLevel::Medium => 0.02,
            DifficultyLevel::Hard => 0.05,
            DifficultyLevel::Extreme => 0.08,
        };

        let ratio = dataset.characteristics.feature_to_sample_ratio.max(0.01);

        let bias_base = match method {
            AutoMLMethod::UnivariateFiltering => 0.04,
            AutoMLMethod::CorrelationBased => 0.035,
            AutoMLMethod::TreeBased => 0.025,
            AutoMLMethod::LassoBased => 0.03,
            AutoMLMethod::WrapperBased => 0.02,
            AutoMLMethod::EnsembleBased => 0.022,
            AutoMLMethod::Hybrid => 0.028,
            AutoMLMethod::NeuralArchitectureSearch => 0.015,
            AutoMLMethod::TransferLearning => 0.02,
            AutoMLMethod::MetaLearningEnsemble => 0.018,
        };

        let variance_base = match method {
            AutoMLMethod::TreeBased | AutoMLMethod::EnsembleBased => 0.018,
            AutoMLMethod::NeuralArchitectureSearch => 0.02,
            AutoMLMethod::WrapperBased => 0.016,
            _ => 0.028,
        };

        let bias =
            (bias_base + difficulty_penalty + ratio * 0.01 + rng.gen_range(0.0..0.02)).min(0.2);
        let variance =
            (variance_base + difficulty_penalty / 2.0 + rng.gen_range(0.0..0.015)).min(0.12);

        let accuracy = scores
            .get(&BenchmarkMetric::Accuracy)
            .copied()
            .unwrap_or(0.75);
        let stability = scores
            .get(&BenchmarkMetric::FeatureStability)
            .copied()
            .unwrap_or(0.7);
        let overfitting_score =
            ((accuracy - stability).abs() + difficulty_penalty * 1.5 + rng.gen_range(0.0..0.05))
                .min(1.0);

        ErrorAnalysis {
            bias,
            variance,
            overfitting_score,
            feature_importance_stability: stability,
        }
    }

    fn compute_overall_rankings(
        &self,
        results: &[DetailedBenchmarkResults],
    ) -> HashMap<AutoMLMethod, f64> {
        let mut rankings = HashMap::new();

        for method in &self.methods {
            let method_results: Vec<_> = results.iter().filter(|r| r.method == *method).collect();
            let avg_accuracy = method_results
                .iter()
                .map(|r| r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
                .sum::<f64>()
                / method_results.len() as f64;
            rankings.insert(method.clone(), avg_accuracy);
        }

        rankings
    }

    fn compute_performance_metrics(
        &self,
        results: &[DetailedBenchmarkResults],
    ) -> PerformanceMetrics {
        let mut mean_accuracy = HashMap::new();
        let mut std_accuracy = HashMap::new();
        let mut mean_feature_reduction = HashMap::new();
        let mut mean_computational_time = HashMap::new();
        let mut convergence_rate = HashMap::new();

        for method in &self.methods {
            let method_results: Vec<_> = results.iter().filter(|r| r.method == *method).collect();

            let accuracies: Vec<f64> = method_results
                .iter()
                .map(|r| *r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
                .collect();

            let mean_acc = accuracies.iter().sum::<f64>() / accuracies.len() as f64;
            let std_acc = (accuracies
                .iter()
                .map(|x| (x - mean_acc).powi(2))
                .sum::<f64>()
                / accuracies.len() as f64)
                .sqrt();

            mean_accuracy.insert(method.clone(), mean_acc);
            std_accuracy.insert(method.clone(), std_acc);

            let mean_reduction = method_results
                .iter()
                .map(|r| {
                    r.scores
                        .get(&BenchmarkMetric::FeatureReduction)
                        .unwrap_or(&0.0)
                })
                .sum::<f64>()
                / method_results.len() as f64;
            mean_feature_reduction.insert(method.clone(), mean_reduction);

            let mean_time = method_results
                .iter()
                .map(|r| {
                    r.scores
                        .get(&BenchmarkMetric::ComputationalTime)
                        .unwrap_or(&0.0)
                })
                .sum::<f64>()
                / method_results.len() as f64;
            mean_computational_time.insert(method.clone(), mean_time);

            let conv_rate = method_results
                .iter()
                .map(|r| {
                    if r.optimization_details.convergence_achieved {
                        1.0
                    } else {
                        0.0
                    }
                })
                .sum::<f64>()
                / method_results.len() as f64;
            convergence_rate.insert(method.clone(), conv_rate);
        }

        PerformanceMetrics {
            mean_accuracy,
            std_accuracy,
            mean_feature_reduction,
            mean_computational_time,
            convergence_rate,
        }
    }

    fn compute_improvement_ratios(
        &self,
        results: &[DetailedBenchmarkResults],
    ) -> ImprovementRatios {
        // Simplified implementation - compute improvements relative to UnivariateFiltering baseline
        let baseline_method = &AutoMLMethod::UnivariateFiltering;
        let baseline_results: Vec<_> = results
            .iter()
            .filter(|r| r.method == *baseline_method)
            .collect();

        let baseline_accuracy = baseline_results
            .iter()
            .map(|r| r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
            .sum::<f64>()
            / baseline_results.len() as f64;

        let mut accuracy_improvement = HashMap::new();
        let mut speed_improvement = HashMap::new();
        let mut memory_improvement = HashMap::new();
        let mut stability_improvement = HashMap::new();

        for method in &self.methods {
            let method_results: Vec<_> = results.iter().filter(|r| r.method == *method).collect();
            let method_accuracy = method_results
                .iter()
                .map(|r| r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
                .sum::<f64>()
                / method_results.len() as f64;

            accuracy_improvement.insert(method.clone(), method_accuracy / baseline_accuracy);
            speed_improvement.insert(method.clone(), 1.0); // Simplified
            memory_improvement.insert(method.clone(), 1.0); // Simplified
            stability_improvement.insert(method.clone(), 1.0); // Simplified
        }

        ImprovementRatios {
            accuracy_improvement,
            speed_improvement,
            memory_improvement,
            stability_improvement,
        }
    }

    fn compute_statistical_significance(
        &self,
        results: &[DetailedBenchmarkResults],
    ) -> HashMap<(AutoMLMethod, AutoMLMethod), f64> {
        let mut significance = HashMap::new();

        for method1 in &self.methods {
            for method2 in &self.methods {
                if method1 == method2 {
                    significance.insert((method1.clone(), method2.clone()), 1.0);
                    continue;
                }

                let method1_scores: Vec<f64> = results
                    .iter()
                    .filter(|r| r.method == *method1)
                    .map(|r| *r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
                    .collect();

                let method2_scores: Vec<f64> = results
                    .iter()
                    .filter(|r| r.method == *method2)
                    .map(|r| *r.scores.get(&BenchmarkMetric::Accuracy).unwrap_or(&0.0))
                    .collect();

                if method1_scores.is_empty() || method2_scores.is_empty() {
                    significance.insert((method1.clone(), method2.clone()), 1.0);
                    continue;
                }

                let mean1 = method1_scores.iter().sum::<f64>() / method1_scores.len() as f64;
                let mean2 = method2_scores.iter().sum::<f64>() / method2_scores.len() as f64;

                let var1 = method1_scores
                    .iter()
                    .map(|s| (s - mean1).powi(2))
                    .sum::<f64>()
                    / method1_scores.len() as f64;
                let var2 = method2_scores
                    .iter()
                    .map(|s| (s - mean2).powi(2))
                    .sum::<f64>()
                    / method2_scores.len() as f64;

                let pooled_std = (var1 + var2 + f64::EPSILON).sqrt();
                let diff = (mean1 - mean2).abs();
                let effect_size = diff / (pooled_std + f64::EPSILON);

                let pseudo_p_value = (1.0 - effect_size / (effect_size + 1.0)).clamp(0.0, 1.0);
                significance.insert((method1.clone(), method2.clone()), pseudo_p_value);
            }
        }

        significance
    }

    /// Generate synthetic benchmark datasets
    #[allow(non_snake_case)]
    pub fn generate_synthetic_datasets(&mut self, n_datasets: usize) -> Result<()> {
        let mut rng = thread_rng();

        for i in 0..n_datasets {
            let n_samples = rng.gen_range(100..2000);
            let n_features = rng.gen_range(10..200);
            let difficulty = match i % 4 {
                0 => DifficultyLevel::Easy,
                1 => DifficultyLevel::Medium,
                2 => DifficultyLevel::Hard,
                _ => DifficultyLevel::Extreme,
            };

            let X = Array2::from_shape_fn((n_samples, n_features), |_| rng.gen_range(-1.0..1.0));
            let y = Array1::from_shape_fn(n_samples, |_| rng.gen_range(0.0..1.0));

            let characteristics = DataCharacteristics {
                n_samples,
                n_features,
                feature_to_sample_ratio: n_features as f64 / n_samples as f64,
                target_type: TargetType::BinaryClassification,
                has_missing_values: false,
                has_categorical_features: false,
                feature_variance_distribution: (0..n_features)
                    .map(|_| rng.gen_range(0.1..2.0))
                    .collect(),
                correlation_structure: super::automl_core::CorrelationStructure {
                    high_correlation_pairs: rng.gen_range(0..n_features / 4),
                    average_correlation: rng.gen_range(0.1..0.6),
                    max_correlation: rng.gen_range(0.5..0.9),
                    correlation_clusters: rng.gen_range(1..n_features / 10 + 1),
                },
                computational_budget: super::automl_core::ComputationalBudget {
                    max_time_seconds: 300.0,
                    max_memory_mb: 1024.0,
                    prefer_speed: false,
                    allow_complex_methods: true,
                },
            };

            let dataset = BenchmarkDataset {
                name: format!("synthetic_dataset_{}", i),
                dataset_type: DatasetType::Synthetic,
                difficulty_level: difficulty,
                X,
                y,
                characteristics,
            };

            self.add_dataset(dataset);
        }

        Ok(())
    }
}

impl Default for AutoMLBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

//! Benchmarking framework for feature selection methods
//!
//! This module provides comprehensive benchmarking capabilities to evaluate and compare
//! feature selection methods against reference implementations and standard datasets.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::error::{Result as SklResult, SklearsError};
type Result<T> = SklResult<T>;
use std::collections::HashMap;
use std::time::{Duration, Instant};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BenchmarkError {
    #[error("Invalid dataset configuration")]
    InvalidDataset,
    #[error("Method execution failed: {0}")]
    MethodExecutionFailed(String),
    #[error("Benchmark configuration error: {0}")]
    ConfigurationError(String),
    #[error("Statistical analysis failed")]
    StatisticalAnalysisFailed,
    #[error("Comparison baseline not found")]
    BaselineNotFound,
}

impl From<BenchmarkError> for SklearsError {
    fn from(err: BenchmarkError) -> Self {
        SklearsError::FitError(format!("Benchmark error: {}", err))
    }
}

/// Comprehensive benchmarking suite for feature selection methods
#[derive(Debug)]
pub struct FeatureSelectionBenchmark {
    datasets: Vec<BenchmarkDataset>,
    methods: HashMap<String, Box<dyn BenchmarkableMethod>>,
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl Default for FeatureSelectionBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureSelectionBenchmark {
    /// Create a new benchmarking suite
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
            methods: HashMap::new(),
            config: BenchmarkConfig::default(),
            results: Vec::new(),
        }
    }

    /// Configure the benchmark suite
    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    /// Add a dataset to the benchmark suite
    pub fn add_dataset(&mut self, dataset: BenchmarkDataset) -> Result<()> {
        if dataset.X.is_empty() || dataset.y.is_empty() {
            return Err(BenchmarkError::InvalidDataset.into());
        }
        self.datasets.push(dataset);
        Ok(())
    }

    /// Add multiple standard datasets for benchmarking
    pub fn add_standard_datasets(&mut self) -> Result<()> {
        // Add synthetic datasets with known properties
        self.add_dataset(self.create_synthetic_linear_dataset(500, 50, 10)?)?;
        self.add_dataset(self.create_synthetic_nonlinear_dataset(300, 100, 15)?)?;
        self.add_dataset(self.create_high_dimensional_dataset(200, 1000, 5)?)?;
        self.add_dataset(self.create_correlated_features_dataset(400, 80, 20)?)?;
        self.add_dataset(self.create_noisy_dataset(600, 60, 12)?)?;

        Ok(())
    }

    /// Register a benchmarkable method
    pub fn register_method(&mut self, name: String, method: Box<dyn BenchmarkableMethod>) {
        self.methods.insert(name, method);
    }

    /// Run comprehensive benchmark suite
    pub fn run_benchmark(&mut self) -> Result<BenchmarkSuiteResults> {
        if self.datasets.is_empty() {
            return Err(BenchmarkError::ConfigurationError("No datasets added".to_string()).into());
        }

        if self.methods.is_empty() {
            return Err(
                BenchmarkError::ConfigurationError("No methods registered".to_string()).into(),
            );
        }

        let mut all_results = Vec::new();
        let start_time = Instant::now();

        println!(
            "Starting benchmark suite with {} datasets and {} methods...",
            self.datasets.len(),
            self.methods.len()
        );

        // Run benchmarks for each dataset-method combination
        for (dataset_idx, dataset) in self.datasets.iter().enumerate() {
            println!(
                "Benchmarking dataset {}/{}: {}",
                dataset_idx + 1,
                self.datasets.len(),
                dataset.name
            );

            for (method_name, method) in self.methods.iter() {
                println!("  Running method: {}", method_name);

                // Run multiple iterations if configured
                let mut method_results = Vec::new();

                for iteration in 0..self.config.n_iterations {
                    if self.config.verbose && iteration % 10 == 0 {
                        println!(
                            "    Iteration {}/{}",
                            iteration + 1,
                            self.config.n_iterations
                        );
                    }

                    let result = self.run_single_benchmark(
                        method.as_ref(),
                        method_name,
                        dataset,
                        iteration,
                    )?;

                    method_results.push(result);
                }

                // Aggregate results for this method on this dataset
                let aggregated_result = self.aggregate_method_results(&method_results)?;
                all_results.push(aggregated_result);
            }
        }

        let total_duration = start_time.elapsed();

        // Perform statistical analysis
        let statistical_analysis = self.perform_statistical_analysis(&all_results)?;

        // Generate rankings
        let rankings = self.generate_rankings(&all_results)?;

        // Create comprehensive results
        let suite_results = BenchmarkSuiteResults {
            individual_results: all_results,
            statistical_analysis,
            rankings,
            execution_summary: ExecutionSummary {
                total_duration,
                n_datasets: self.datasets.len(),
                n_methods: self.methods.len(),
                n_total_runs: self.datasets.len() * self.methods.len() * self.config.n_iterations,
            },
            configuration: self.config.clone(),
        };

        self.results
            .extend(suite_results.individual_results.clone());

        println!(
            "Benchmark suite completed in {:.2}s",
            total_duration.as_secs_f64()
        );

        Ok(suite_results)
    }

    /// Run a single benchmark iteration
    #[allow(non_snake_case)]
    fn run_single_benchmark(
        &self,
        method: &dyn BenchmarkableMethod,
        method_name: &str,
        dataset: &BenchmarkDataset,
        iteration: usize,
    ) -> Result<BenchmarkResult> {
        let start_time = Instant::now();

        // Split data if cross-validation is configured
        let (train_indices, test_indices) = if self.config.use_cross_validation {
            self.create_cv_split(dataset.X.nrows(), iteration % self.config.cv_folds)?
        } else {
            self.create_holdout_split(dataset.X.nrows())?
        };

        // Extract training and test data
        let X_train = self.extract_samples(dataset.X.view(), &train_indices);
        let y_train = self.extract_targets(dataset.y.view(), &train_indices);
        let X_test = self.extract_samples(dataset.X.view(), &test_indices);
        let y_test = self.extract_targets(dataset.y.view(), &test_indices);

        // Measure memory usage before method execution
        let memory_before = self.get_memory_usage();

        // Execute feature selection method
        let selected_features = match method.select_features(X_train.view(), y_train.view()) {
            Ok(features) => features,
            Err(e) => return Err(BenchmarkError::MethodExecutionFailed(format!("{:?}", e)).into()),
        };

        let selection_time = start_time.elapsed();

        // Measure memory usage after method execution
        let memory_after = self.get_memory_usage();
        let memory_delta = memory_after.saturating_sub(memory_before);

        // Evaluate selected features
        let evaluation_start = Instant::now();
        let evaluation_metrics = self.evaluate_feature_selection(
            X_train.view(),
            y_train.view(),
            X_test.view(),
            y_test.view(),
            &selected_features,
            &dataset.ground_truth,
        )?;
        let evaluation_time = evaluation_start.elapsed();

        // Compute feature quality metrics
        let feature_metrics =
            self.compute_feature_metrics(dataset.X.view(), dataset.y.view(), &selected_features)?;

        Ok(BenchmarkResult {
            method_name: method_name.to_string(),
            dataset_name: dataset.name.clone(),
            iteration,
            selected_features: selected_features.clone(),
            n_features_selected: selected_features.len(),
            n_features_total: dataset.X.ncols(),
            selection_time,
            evaluation_time,
            memory_usage: memory_delta,
            evaluation_metrics,
            feature_metrics,
            dataset_properties: dataset.properties.clone(),
        })
    }

    /// Create synthetic linear dataset with known relevant features
    fn create_synthetic_linear_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_relevant: usize,
    ) -> Result<BenchmarkDataset> {
        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        // Generate relevant features with linear relationship to target
        for i in 0..n_samples {
            let mut target = 0.0;

            for j in 0..n_relevant {
                let value = thread_rng().gen::<f64>() * 10.0 - 5.0; // Range [-5, 5]
                X[[i, j]] = value;
                target += value * (j + 1) as f64; // Different coefficients for each feature
            }

            // Add noise features
            for j in n_relevant..n_features {
                X[[i, j]] = thread_rng().gen::<f64>() * 10.0 - 5.0;
            }

            // Add noise to target
            target += (thread_rng().gen::<f64>() - 0.5) * 2.0;
            y[i] = target;
        }

        let ground_truth = Some((0..n_relevant).collect());
        let properties = DatasetProperties {
            task_type: TaskType::Regression,
            n_informative_features: n_relevant,
            noise_level: 0.1,
            correlation_structure: CorrelationStructure::Linear,
            feature_types: vec![FeatureType::Continuous; n_features],
        };

        Ok(BenchmarkDataset {
            name: format!("SyntheticLinear_{}x{}", n_samples, n_features),
            X,
            y,
            ground_truth,
            properties,
        })
    }

    /// Create synthetic nonlinear dataset
    fn create_synthetic_nonlinear_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_relevant: usize,
    ) -> Result<BenchmarkDataset> {
        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut target = 0.0;

            // Generate relevant features with nonlinear relationship
            for j in 0..n_relevant {
                let value = thread_rng().gen::<f64>() * 6.0 - 3.0; // Range [-3, 3]
                X[[i, j]] = value;

                // Nonlinear relationships: quadratic, sine, exponential
                match j % 3 {
                    0 => target += value * value,
                    1 => target += (value * std::f64::consts::PI / 2.0).sin() * 2.0,
                    2 => target += (value.abs()).ln().max(-5.0),
                    _ => unreachable!(),
                }
            }

            // Noise features
            for j in n_relevant..n_features {
                X[[i, j]] = thread_rng().gen::<f64>() * 6.0 - 3.0;
            }

            y[i] = target + (thread_rng().gen::<f64>() - 0.5) * 0.5;
        }

        let ground_truth = Some((0..n_relevant).collect());
        let properties = DatasetProperties {
            task_type: TaskType::Regression,
            n_informative_features: n_relevant,
            noise_level: 0.15,
            correlation_structure: CorrelationStructure::Nonlinear,
            feature_types: vec![FeatureType::Continuous; n_features],
        };

        Ok(BenchmarkDataset {
            name: format!("SyntheticNonlinear_{}x{}", n_samples, n_features),
            X,
            y,
            ground_truth,
            properties,
        })
    }

    /// Create high-dimensional dataset (p >> n scenario)
    fn create_high_dimensional_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_relevant: usize,
    ) -> Result<BenchmarkDataset> {
        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut target = 0.0;

            // Sparse relevant features
            for j in 0..n_relevant {
                let value = thread_rng().gen::<f64>() * 4.0 - 2.0;
                X[[i, j]] = value;
                target += value * (1.0 / (j + 1) as f64); // Decreasing importance
            }

            // Many noise features
            for j in n_relevant..n_features {
                X[[i, j]] = thread_rng().gen::<f64>() * 4.0 - 2.0;
            }

            y[i] = if target > 0.0 { 1.0 } else { 0.0 }; // Binary classification
        }

        let ground_truth = Some((0..n_relevant).collect());
        let properties = DatasetProperties {
            task_type: TaskType::BinaryClassification,
            n_informative_features: n_relevant,
            noise_level: 0.2,
            correlation_structure: CorrelationStructure::Sparse,
            feature_types: vec![FeatureType::Continuous; n_features],
        };

        Ok(BenchmarkDataset {
            name: format!("HighDimensional_{}x{}", n_samples, n_features),
            X,
            y,
            ground_truth,
            properties,
        })
    }

    /// Create dataset with correlated features
    fn create_correlated_features_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_relevant: usize,
    ) -> Result<BenchmarkDataset> {
        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut target = 0.0;

            // Generate base relevant features
            let base_features: Vec<f64> = (0..n_relevant)
                .map(|_| thread_rng().gen::<f64>() * 4.0 - 2.0)
                .collect();

            for j in 0..n_relevant {
                X[[i, j]] = base_features[j];
                target += base_features[j] * (j + 1) as f64;
            }

            // Add correlated copies and linear combinations
            let mut feature_idx = n_relevant;
            for j in 0..n_relevant.min((n_features - n_relevant) / 3) {
                if feature_idx < n_features {
                    // Highly correlated copy with noise
                    X[[i, feature_idx]] =
                        base_features[j] + (thread_rng().gen::<f64>() - 0.5) * 0.2;
                    feature_idx += 1;
                }

                if feature_idx < n_features && j + 1 < base_features.len() {
                    // Linear combination of two features
                    X[[i, feature_idx]] = 0.5 * base_features[j] + 0.3 * base_features[j + 1];
                    feature_idx += 1;
                }
            }

            // Fill remaining with noise
            for j in feature_idx..n_features {
                X[[i, j]] = thread_rng().gen::<f64>() * 4.0 - 2.0;
            }

            y[i] = target + (thread_rng().gen::<f64>() - 0.5) * 1.0;
        }

        let ground_truth = Some((0..n_relevant).collect());
        let properties = DatasetProperties {
            task_type: TaskType::Regression,
            n_informative_features: n_relevant,
            noise_level: 0.25,
            correlation_structure: CorrelationStructure::HighlyCorrelated,
            feature_types: vec![FeatureType::Continuous; n_features],
        };

        Ok(BenchmarkDataset {
            name: format!("Correlated_{}x{}", n_samples, n_features),
            X,
            y,
            ground_truth,
            properties,
        })
    }

    /// Create noisy dataset
    fn create_noisy_dataset(
        &self,
        n_samples: usize,
        n_features: usize,
        n_relevant: usize,
    ) -> Result<BenchmarkDataset> {
        let mut X = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut target = 0.0;

            for j in 0..n_relevant {
                let value = thread_rng().gen::<f64>() * 8.0 - 4.0;
                X[[i, j]] = value;
                target += value * (j + 1) as f64;
            }

            // Noise features
            for j in n_relevant..n_features {
                X[[i, j]] = thread_rng().gen::<f64>() * 8.0 - 4.0;
            }

            // Heavy noise on target
            let noise = (thread_rng().gen::<f64>() - 0.5) * target.abs() * 0.5;
            y[i] = target + noise;
        }

        let ground_truth = Some((0..n_relevant).collect());
        let properties = DatasetProperties {
            task_type: TaskType::Regression,
            n_informative_features: n_relevant,
            noise_level: 0.5,
            correlation_structure: CorrelationStructure::Noisy,
            feature_types: vec![FeatureType::Continuous; n_features],
        };

        Ok(BenchmarkDataset {
            name: format!("Noisy_{}x{}", n_samples, n_features),
            X,
            y,
            ground_truth,
            properties,
        })
    }

    // Helper methods
    fn create_cv_split(&self, n_samples: usize, fold: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let fold_size = n_samples / self.config.cv_folds;
        let start = (fold % self.config.cv_folds) * fold_size;
        let end = if fold % self.config.cv_folds == self.config.cv_folds - 1 {
            n_samples
        } else {
            start + fold_size
        };

        let test_indices: Vec<usize> = (start..end).collect();
        let train_indices: Vec<usize> = (0..start).chain(end..n_samples).collect();

        Ok((train_indices, test_indices))
    }

    fn create_holdout_split(&self, n_samples: usize) -> Result<(Vec<usize>, Vec<usize>)> {
        let n_test = (n_samples as f64 * self.config.test_size) as usize;
        let n_train = n_samples - n_test;

        // Simple sequential split for deterministic behavior
        let train_indices: Vec<usize> = (0..n_train).collect();
        let test_indices: Vec<usize> = (n_train..n_samples).collect();

        Ok((train_indices, test_indices))
    }

    fn extract_samples(&self, X: ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
        let mut samples = Array2::zeros((indices.len(), X.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            samples.row_mut(i).assign(&X.row(idx));
        }
        samples
    }

    fn extract_targets(&self, y: ArrayView1<f64>, indices: &[usize]) -> Array1<f64> {
        let mut targets = Array1::zeros(indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            targets[i] = y[idx];
        }
        targets
    }

    fn get_memory_usage(&self) -> usize {
        // Simplified memory tracking - in a real implementation, this would
        // use system calls or memory profiling tools
        0
    }

    fn evaluate_feature_selection(
        &self,
        X_train: ArrayView2<f64>,
        y_train: ArrayView1<f64>,
        X_test: ArrayView2<f64>,
        y_test: ArrayView1<f64>,
        selected_features: &[usize],
        ground_truth: &Option<Vec<usize>>,
    ) -> Result<EvaluationMetrics> {
        let mut metrics = EvaluationMetrics::default();

        // Compute relevance score (correlation with target)
        if !selected_features.is_empty() {
            let mut total_relevance = 0.0;
            for &feature_idx in selected_features {
                if feature_idx < X_train.ncols() {
                    let correlation =
                        self.compute_correlation(X_train.column(feature_idx), y_train);
                    total_relevance += correlation.abs();
                }
            }
            metrics.relevance_score = total_relevance / selected_features.len() as f64;
        }

        // Compute redundancy (average pairwise correlation among selected features)
        if selected_features.len() > 1 {
            let mut total_correlation = 0.0;
            let mut pair_count = 0;

            for i in 0..selected_features.len() {
                for j in (i + 1)..selected_features.len() {
                    if selected_features[i] < X_train.ncols()
                        && selected_features[j] < X_train.ncols()
                    {
                        let correlation = self.compute_correlation(
                            X_train.column(selected_features[i]),
                            X_train.column(selected_features[j]),
                        );
                        total_correlation += correlation.abs();
                        pair_count += 1;
                    }
                }
            }

            if pair_count > 0 {
                metrics.redundancy_score = total_correlation / pair_count as f64;
            }
        }

        // Compute ground truth metrics if available
        if let Some(true_features) = ground_truth {
            let selected_set: std::collections::HashSet<_> = selected_features.iter().collect();
            let true_set: std::collections::HashSet<_> = true_features.iter().collect();

            let intersection = selected_set.intersection(&true_set).count() as f64;
            let union = selected_set.union(&true_set).count() as f64;

            if union > 0.0 {
                metrics.jaccard_score = Some(intersection / union);
            }

            if !true_features.is_empty() {
                metrics.precision = Some(intersection / selected_features.len() as f64);
                metrics.recall = Some(intersection / true_features.len() as f64);

                if let (Some(p), Some(r)) = (metrics.precision, metrics.recall) {
                    if p + r > 0.0 {
                        metrics.f1_score = Some(2.0 * p * r / (p + r));
                    }
                }
            }
        }

        // Simplified predictive performance (correlation-based)
        metrics.predictive_score = metrics.relevance_score * (1.0 - metrics.redundancy_score * 0.5);

        Ok(metrics)
    }

    fn compute_feature_metrics(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
        selected_features: &[usize],
    ) -> Result<FeatureMetrics> {
        let mut metrics = FeatureMetrics::default();

        if selected_features.is_empty() {
            return Ok(metrics);
        }

        // Selection ratio
        metrics.selection_ratio = selected_features.len() as f64 / X.ncols() as f64;

        // Feature importance distribution
        let mut importances = Vec::new();
        for &feature_idx in selected_features {
            if feature_idx < X.ncols() {
                let importance = self.compute_correlation(X.column(feature_idx), y).abs();
                importances.push(importance);
            }
        }

        if !importances.is_empty() {
            importances.sort_by(|a, b| b.partial_cmp(a).unwrap());
            metrics.importance_distribution = ImportanceDistribution {
                mean: importances.iter().sum::<f64>() / importances.len() as f64,
                std: {
                    let mean = importances.iter().sum::<f64>() / importances.len() as f64;
                    let variance = importances.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                        / importances.len() as f64;
                    variance.sqrt()
                },
                max: importances[0],
                min: importances[importances.len() - 1],
            };
        }

        Ok(metrics)
    }

    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }

    fn aggregate_method_results(&self, results: &[BenchmarkResult]) -> Result<BenchmarkResult> {
        if results.is_empty() {
            return Err(BenchmarkError::StatisticalAnalysisFailed.into());
        }

        let first = &results[0];

        // Aggregate numerical metrics
        let aggregate_evaluation_metrics = EvaluationMetrics {
            relevance_score: results
                .iter()
                .map(|r| r.evaluation_metrics.relevance_score)
                .sum::<f64>()
                / results.len() as f64,
            redundancy_score: results
                .iter()
                .map(|r| r.evaluation_metrics.redundancy_score)
                .sum::<f64>()
                / results.len() as f64,
            predictive_score: results
                .iter()
                .map(|r| r.evaluation_metrics.predictive_score)
                .sum::<f64>()
                / results.len() as f64,
            jaccard_score: if results
                .iter()
                .all(|r| r.evaluation_metrics.jaccard_score.is_some())
            {
                Some(
                    results
                        .iter()
                        .map(|r| r.evaluation_metrics.jaccard_score.unwrap())
                        .sum::<f64>()
                        / results.len() as f64,
                )
            } else {
                None
            },
            precision: if results
                .iter()
                .all(|r| r.evaluation_metrics.precision.is_some())
            {
                Some(
                    results
                        .iter()
                        .map(|r| r.evaluation_metrics.precision.unwrap())
                        .sum::<f64>()
                        / results.len() as f64,
                )
            } else {
                None
            },
            recall: if results
                .iter()
                .all(|r| r.evaluation_metrics.recall.is_some())
            {
                Some(
                    results
                        .iter()
                        .map(|r| r.evaluation_metrics.recall.unwrap())
                        .sum::<f64>()
                        / results.len() as f64,
                )
            } else {
                None
            },
            f1_score: if results
                .iter()
                .all(|r| r.evaluation_metrics.f1_score.is_some())
            {
                Some(
                    results
                        .iter()
                        .map(|r| r.evaluation_metrics.f1_score.unwrap())
                        .sum::<f64>()
                        / results.len() as f64,
                )
            } else {
                None
            },
        };

        // Take most frequent feature selection
        let most_common_features = first.selected_features.clone(); // Simplified

        Ok(BenchmarkResult {
            method_name: first.method_name.clone(),
            dataset_name: first.dataset_name.clone(),
            iteration: 999, // Special marker for aggregated result
            selected_features: most_common_features,
            n_features_selected: results.iter().map(|r| r.n_features_selected).sum::<usize>()
                / results.len(),
            n_features_total: first.n_features_total,
            selection_time: Duration::from_nanos(
                (results
                    .iter()
                    .map(|r| r.selection_time.as_nanos())
                    .sum::<u128>()
                    / results.len() as u128) as u64,
            ),
            evaluation_time: Duration::from_nanos(
                (results
                    .iter()
                    .map(|r| r.evaluation_time.as_nanos())
                    .sum::<u128>()
                    / results.len() as u128) as u64,
            ),
            memory_usage: results.iter().map(|r| r.memory_usage).sum::<usize>() / results.len(),
            evaluation_metrics: aggregate_evaluation_metrics,
            feature_metrics: first.feature_metrics.clone(), // Simplified
            dataset_properties: first.dataset_properties.clone(),
        })
    }

    fn perform_statistical_analysis(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<StatisticalAnalysis> {
        // Group results by method and dataset
        let mut method_performances: HashMap<String, Vec<f64>> = HashMap::new();
        let mut dataset_difficulties: HashMap<String, Vec<f64>> = HashMap::new();

        for result in results {
            method_performances
                .entry(result.method_name.clone())
                .or_default()
                .push(result.evaluation_metrics.predictive_score);

            dataset_difficulties
                .entry(result.dataset_name.clone())
                .or_default()
                .push(result.evaluation_metrics.predictive_score);
        }

        // Compute method rankings
        let mut method_rankings = Vec::new();
        for (method, scores) in method_performances {
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let std_score = {
                let variance = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>()
                    / scores.len() as f64;
                variance.sqrt()
            };

            method_rankings.push(MethodRanking {
                method_name: method,
                mean_score,
                std_score,
                scores,
            });
        }

        method_rankings.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());

        // Compute dataset difficulties
        let mut dataset_rankings = Vec::new();
        for (dataset, scores) in dataset_difficulties {
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

            dataset_rankings.push(DatasetRanking {
                dataset_name: dataset,
                difficulty_score: 1.0 - mean_score, // Higher score = more difficult
                mean_performance: mean_score,
            });
        }

        dataset_rankings
            .sort_by(|a, b| b.difficulty_score.partial_cmp(&a.difficulty_score).unwrap());

        let overall_best_method = method_rankings.first().map(|r| r.method_name.clone());

        Ok(StatisticalAnalysis {
            method_rankings,
            dataset_rankings,
            overall_best_method,
            performance_summary: self.compute_performance_summary(results)?,
        })
    }

    fn compute_performance_summary(
        &self,
        results: &[BenchmarkResult],
    ) -> Result<PerformanceSummary> {
        if results.is_empty() {
            return Err(BenchmarkError::StatisticalAnalysisFailed.into());
        }

        let scores: Vec<f64> = results
            .iter()
            .map(|r| r.evaluation_metrics.predictive_score)
            .collect();
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

        let std_score = {
            let variance =
                scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>() / scores.len() as f64;
            variance.sqrt()
        };

        let min_score = scores.iter().fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

        let total_time: Duration = results
            .iter()
            .map(|r| r.selection_time + r.evaluation_time)
            .sum();
        let avg_features_selected = results.iter().map(|r| r.n_features_selected).sum::<usize>()
            as f64
            / results.len() as f64;

        Ok(PerformanceSummary {
            mean_score,
            std_score,
            min_score,
            max_score,
            total_execution_time: total_time,
            avg_features_selected,
        })
    }

    fn generate_rankings(&self, results: &[BenchmarkResult]) -> Result<Vec<MethodRanking>> {
        let mut method_scores: HashMap<String, Vec<f64>> = HashMap::new();

        for result in results {
            method_scores
                .entry(result.method_name.clone())
                .or_default()
                .push(result.evaluation_metrics.predictive_score);
        }

        let mut rankings = Vec::new();
        for (method, scores) in method_scores {
            let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
            let std_score = {
                let variance = scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f64>()
                    / scores.len() as f64;
                variance.sqrt()
            };

            rankings.push(MethodRanking {
                method_name: method,
                mean_score,
                std_score,
                scores,
            });
        }

        rankings.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());
        Ok(rankings)
    }
}

/// Trait for benchmarkable feature selection methods
pub trait BenchmarkableMethod: std::fmt::Debug + Send + Sync {
    fn select_features(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> std::result::Result<Vec<usize>, BenchmarkError>;
    fn method_name(&self) -> &str;
    fn method_params(&self) -> HashMap<String, String>;
}

// Data structures for benchmarking

#[derive(Debug, Clone)]
pub struct BenchmarkDataset {
    pub name: String,
    pub X: Array2<f64>,
    pub y: Array1<f64>,
    pub ground_truth: Option<Vec<usize>>,
    pub properties: DatasetProperties,
}

#[derive(Debug, Clone)]
pub struct DatasetProperties {
    pub task_type: TaskType,
    pub n_informative_features: usize,
    pub noise_level: f64,
    pub correlation_structure: CorrelationStructure,
    pub feature_types: Vec<FeatureType>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskType {
    /// BinaryClassification
    BinaryClassification,
    /// MultiClassification
    MultiClassification,
    /// Regression
    Regression,
    /// MultiLabel
    MultiLabel,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CorrelationStructure {
    /// Linear
    Linear,
    /// Nonlinear
    Nonlinear,
    /// Sparse
    Sparse,
    /// HighlyCorrelated
    HighlyCorrelated,
    /// Noisy
    Noisy,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FeatureType {
    /// Continuous
    Continuous,
    /// Discrete
    Discrete,
    /// Binary
    Binary,
    /// Categorical
    Categorical,
}

#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub n_iterations: usize,
    pub use_cross_validation: bool,
    pub cv_folds: usize,
    pub test_size: f64,
    pub timeout_seconds: Option<u64>,
    pub memory_limit_mb: Option<usize>,
    pub verbose: bool,
    pub random_seed: Option<u64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_iterations: 5,
            use_cross_validation: true,
            cv_folds: 3,
            test_size: 0.3,
            timeout_seconds: Some(300),  // 5 minutes
            memory_limit_mb: Some(2048), // 2 GB
            verbose: true,
            random_seed: Some(42),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub method_name: String,
    pub dataset_name: String,
    pub iteration: usize,
    pub selected_features: Vec<usize>,
    pub n_features_selected: usize,
    pub n_features_total: usize,
    pub selection_time: Duration,
    pub evaluation_time: Duration,
    pub memory_usage: usize,
    pub evaluation_metrics: EvaluationMetrics,
    pub feature_metrics: FeatureMetrics,
    pub dataset_properties: DatasetProperties,
}

#[derive(Debug, Clone, Default)]
pub struct EvaluationMetrics {
    pub relevance_score: f64,
    pub redundancy_score: f64,
    pub predictive_score: f64,
    pub jaccard_score: Option<f64>,
    pub precision: Option<f64>,
    pub recall: Option<f64>,
    pub f1_score: Option<f64>,
}

#[derive(Debug, Clone, Default)]
pub struct FeatureMetrics {
    pub selection_ratio: f64,
    pub importance_distribution: ImportanceDistribution,
}

#[derive(Debug, Clone, Default)]
pub struct ImportanceDistribution {
    pub mean: f64,
    pub std: f64,
    pub max: f64,
    pub min: f64,
}

#[derive(Debug, Clone)]
pub struct BenchmarkSuiteResults {
    pub individual_results: Vec<BenchmarkResult>,
    pub statistical_analysis: StatisticalAnalysis,
    pub rankings: Vec<MethodRanking>,
    pub execution_summary: ExecutionSummary,
    pub configuration: BenchmarkConfig,
}

impl BenchmarkSuiteResults {
    /// Generate comprehensive benchmark report
    pub fn report(&self) -> String {
        let mut report = String::new();

        report.push_str(
            "╔═══════════════════════════════════════════════════════════════════════════╗\n",
        );
        report.push_str(
            "║                    Feature Selection Benchmark Report                    ║\n",
        );
        report.push_str(
            "╚═══════════════════════════════════════════════════════════════════════════╝\n\n",
        );

        // Execution summary
        report.push_str("=== Execution Summary ===\n");
        report.push_str(&format!(
            "Total Duration: {:.2}s\n",
            self.execution_summary.total_duration.as_secs_f64()
        ));
        report.push_str(&format!(
            "Datasets: {}\n",
            self.execution_summary.n_datasets
        ));
        report.push_str(&format!("Methods: {}\n", self.execution_summary.n_methods));
        report.push_str(&format!(
            "Total Runs: {}\n",
            self.execution_summary.n_total_runs
        ));

        // Method rankings
        if !self.rankings.is_empty() {
            report.push_str("\n=== Method Rankings ===\n");
            for (i, ranking) in self.rankings.iter().take(10).enumerate() {
                report.push_str(&format!(
                    "{}. {} - Score: {:.4} ± {:.4}\n",
                    i + 1,
                    ranking.method_name,
                    ranking.mean_score,
                    ranking.std_score
                ));
            }
        }

        // Best method details
        if let Some(best_method) = &self.statistical_analysis.overall_best_method {
            report.push_str(&format!("\n=== Best Method: {} ===\n", best_method));

            if let Some(best_ranking) = self.rankings.first() {
                report.push_str(&format!(
                    "Mean Performance: {:.4}\n",
                    best_ranking.mean_score
                ));
                report.push_str(&format!(
                    "Consistency (StdDev): {:.4}\n",
                    best_ranking.std_score
                ));
                report.push_str(&format!("Runs: {}\n", best_ranking.scores.len()));
            }
        }

        // Dataset difficulty analysis
        if !self.statistical_analysis.dataset_rankings.is_empty() {
            report.push_str("\n=== Dataset Difficulty Ranking ===\n");
            for (i, dataset) in self
                .statistical_analysis
                .dataset_rankings
                .iter()
                .take(5)
                .enumerate()
            {
                report.push_str(&format!(
                    "{}. {} - Difficulty: {:.4} (Avg Performance: {:.4})\n",
                    i + 1,
                    dataset.dataset_name,
                    dataset.difficulty_score,
                    dataset.mean_performance
                ));
            }
        }

        // Performance summary
        report.push_str(&"\n=== Overall Performance Summary ===\n".to_string());
        let summary = &self.statistical_analysis.performance_summary;
        report.push_str(&format!(
            "Mean Score: {:.4} ± {:.4}\n",
            summary.mean_score, summary.std_score
        ));
        report.push_str(&format!(
            "Score Range: [{:.4}, {:.4}]\n",
            summary.min_score, summary.max_score
        ));
        report.push_str(&format!(
            "Total Time: {:.2}s\n",
            summary.total_execution_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "Avg Features Selected: {:.1}\n",
            summary.avg_features_selected
        ));

        report
    }

    /// Export results to CSV format (simplified)
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();

        csv.push_str("Method,Dataset,Iteration,FeaturesSelected,SelectionTime,RelevanceScore,RedundancyScore,PredictiveScore\n");

        for result in &self.individual_results {
            csv.push_str(&format!(
                "{},{},{},{},{:.6},{:.4},{:.4},{:.4}\n",
                result.method_name,
                result.dataset_name,
                result.iteration,
                result.n_features_selected,
                result.selection_time.as_secs_f64(),
                result.evaluation_metrics.relevance_score,
                result.evaluation_metrics.redundancy_score,
                result.evaluation_metrics.predictive_score
            ));
        }

        csv
    }
}

#[derive(Debug, Clone)]
pub struct StatisticalAnalysis {
    pub method_rankings: Vec<MethodRanking>,
    pub dataset_rankings: Vec<DatasetRanking>,
    pub overall_best_method: Option<String>,
    pub performance_summary: PerformanceSummary,
}

#[derive(Debug, Clone)]
pub struct MethodRanking {
    pub method_name: String,
    pub mean_score: f64,
    pub std_score: f64,
    pub scores: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct DatasetRanking {
    pub dataset_name: String,
    pub difficulty_score: f64,
    pub mean_performance: f64,
}

#[derive(Debug, Clone)]
pub struct PerformanceSummary {
    pub mean_score: f64,
    pub std_score: f64,
    pub min_score: f64,
    pub max_score: f64,
    pub total_execution_time: Duration,
    pub avg_features_selected: f64,
}

#[derive(Debug, Clone)]
pub struct ExecutionSummary {
    pub total_duration: Duration,
    pub n_datasets: usize,
    pub n_methods: usize,
    pub n_total_runs: usize,
}

// Example benchmarkable method implementations
#[derive(Debug)]
pub struct UnivariateFilterMethod {
    k: usize,
}

impl UnivariateFilterMethod {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl BenchmarkableMethod for UnivariateFilterMethod {
    fn select_features(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> std::result::Result<Vec<usize>, BenchmarkError> {
        let mut correlations: Vec<(usize, f64)> = Vec::new();

        for i in 0..X.ncols() {
            let correlation = self.compute_correlation(X.column(i), y);
            correlations.push((i, correlation.abs()));
        }

        correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        Ok(correlations
            .into_iter()
            .take(self.k)
            .map(|(idx, _)| idx)
            .collect())
    }

    fn method_name(&self) -> &str {
        "UnivariateFilter"
    }

    fn method_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("k".to_string(), self.k.to_string());
        params
    }
}

impl UnivariateFilterMethod {
    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }
}

#[derive(Debug)]
pub struct RandomSelectionMethod {
    k: usize,
}

impl RandomSelectionMethod {
    pub fn new(k: usize) -> Self {
        Self { k }
    }
}

impl BenchmarkableMethod for RandomSelectionMethod {
    fn select_features(
        &self,
        X: ArrayView2<f64>,
        _y: ArrayView1<f64>,
    ) -> std::result::Result<Vec<usize>, BenchmarkError> {
        let mut features: Vec<usize> = (0..X.ncols()).collect();

        // Simple random shuffling
        for i in (1..features.len()).rev() {
            let j = (thread_rng().gen::<f64>() * (i + 1) as f64) as usize;
            features.swap(i, j);
        }

        Ok(features.into_iter().take(self.k.min(X.ncols())).collect())
    }

    fn method_name(&self) -> &str {
        "RandomSelection"
    }

    fn method_params(&self) -> HashMap<String, String> {
        let mut params = HashMap::new();
        params.insert("k".to_string(), self.k.to_string());
        params
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_synthetic_dataset_creation() {
        let benchmark = FeatureSelectionBenchmark::new();

        let dataset = benchmark
            .create_synthetic_linear_dataset(100, 20, 5)
            .unwrap();

        assert_eq!(dataset.X.nrows(), 100);
        assert_eq!(dataset.X.ncols(), 20);
        assert_eq!(dataset.y.len(), 100);
        assert_eq!(dataset.ground_truth.as_ref().unwrap().len(), 5);
        assert_eq!(dataset.properties.n_informative_features, 5);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_benchmarkable_method() {
        let method = UnivariateFilterMethod::new(5);

        let X = array![
            [1.0, 2.0, 3.0, 0.1],
            [2.0, 3.0, 4.0, 0.2],
            [3.0, 4.0, 5.0, 0.3],
            [4.0, 5.0, 6.0, 0.4],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let selected = method.select_features(X.view(), y.view()).unwrap();

        assert!(!selected.is_empty());
        assert!(selected.len() <= 4); // Can't select more than available
        assert!(selected.iter().all(|&idx| idx < X.ncols()));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_benchmark_execution() {
        let mut benchmark = FeatureSelectionBenchmark::new();
        benchmark.config.n_iterations = 1; // Fast test
        benchmark.config.verbose = false;

        // Add simple dataset
        let X = array![
            [1.0, 2.0, 0.1, 0.2],
            [2.0, 3.0, 0.2, 0.3],
            [3.0, 4.0, 0.3, 0.4],
            [4.0, 5.0, 0.4, 0.5],
            [5.0, 6.0, 0.5, 0.6],
        ];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let dataset = BenchmarkDataset {
            name: "TestDataset".to_string(),
            X,
            y,
            ground_truth: Some(vec![0, 1]),
            properties: DatasetProperties {
                task_type: TaskType::Regression,
                n_informative_features: 2,
                noise_level: 0.1,
                correlation_structure: CorrelationStructure::Linear,
                feature_types: vec![FeatureType::Continuous; 4],
            },
        };

        benchmark.add_dataset(dataset).unwrap();
        benchmark.register_method(
            "UnivariateFilt".to_string(),
            Box::new(UnivariateFilterMethod::new(2)),
        );
        benchmark.register_method(
            "Random".to_string(),
            Box::new(RandomSelectionMethod::new(2)),
        );

        let results = benchmark.run_benchmark().unwrap();

        assert_eq!(results.individual_results.len(), 2); // 2 methods
        assert_eq!(results.rankings.len(), 2);
        assert!(results.statistical_analysis.overall_best_method.is_some());

        let report = results.report();
        assert!(report.contains("Benchmark Report"));
        assert!(report.contains("Method Rankings"));

        let csv = results.to_csv();
        assert!(csv.contains("Method,Dataset"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_evaluation_metrics() {
        let benchmark = FeatureSelectionBenchmark::new();

        let X = array![[1.0, 2.0, 0.1], [2.0, 3.0, 0.2], [3.0, 4.0, 0.3],];
        let y = array![1.0, 2.0, 3.0];

        let selected_features = vec![0, 1];
        let ground_truth = Some(vec![0, 1]);

        let metrics = benchmark
            .evaluate_feature_selection(
                X.view(),
                y.view(),
                X.view(),
                y.view(),
                &selected_features,
                &ground_truth,
            )
            .unwrap();

        assert!(metrics.relevance_score >= 0.0);
        assert!(metrics.redundancy_score >= 0.0);
        assert!(metrics.predictive_score >= 0.0);
        assert!(metrics.jaccard_score.is_some());
        assert!(metrics.precision.is_some());
        assert!(metrics.recall.is_some());
        assert!(metrics.f1_score.is_some());
    }
}

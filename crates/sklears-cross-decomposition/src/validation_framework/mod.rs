//! Comprehensive Validation Framework
//!
//! This module provides a systematic validation framework for cross-decomposition algorithms,
//! including real-world case studies, benchmark datasets, and performance evaluation metrics.

mod types;
pub use types::*;

use crate::validation_metrics::{
    empirical_p_value, jaccard_index, kmeans, principal_angles, quantile, rand_index,
    silhouette_coefficient, subspace_recovery,
};
use crate::{PLSRegression, CCA};
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{seeded_rng, CoreRandom, RandNormal, RandUniform};
use sklears_core::traits::{Fit, Predict, Transform};
use sklears_core::types::Float;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Default number of resamples (permutations / bootstrap replicates) used by
/// the statistical significance tests. Large enough for a stable empirical
/// p-value resolution of ~1e-3.
const DEFAULT_N_RESAMPLES: usize = 1000;

/// Comprehensive validation framework for cross-decomposition algorithms
pub struct ValidationFramework {
    /// Benchmark datasets
    benchmark_datasets: Vec<BenchmarkDataset>,
    /// Performance metrics to compute
    pub performance_metrics: Vec<PerformanceMetric>,
    /// Statistical significance tests
    significance_tests: Vec<SignificanceTest>,
    /// Real-world case studies
    case_studies: Vec<CaseStudy>,
    /// Cross-validation settings
    cv_settings: CrossValidationSettings,
    /// Number of resamples used by the permutation / bootstrap / CV tests.
    n_resamples: usize,
}

impl ValidationFramework {
    /// Create a new validation framework
    pub fn new() -> Self {
        Self {
            benchmark_datasets: Vec::new(),
            performance_metrics: vec![
                PerformanceMetric::CanonicalCorrelationAccuracy,
                PerformanceMetric::ComponentRecovery,
                PerformanceMetric::PredictionAccuracy,
                PerformanceMetric::CrossValidationStability,
                PerformanceMetric::ComputationalTime,
            ],
            significance_tests: vec![
                SignificanceTest::PermutationTest,
                SignificanceTest::BootstrapConfidenceIntervals,
                SignificanceTest::CrossValidationSignificance,
            ],
            case_studies: Vec::new(),
            cv_settings: CrossValidationSettings {
                n_folds: 5,
                n_repetitions: 3,
                stratification: true,
                random_seed: Some(42),
            },
            n_resamples: DEFAULT_N_RESAMPLES,
        }
    }

    /// Set the number of resamples used by the statistical significance tests
    /// (permutation / bootstrap / cross-validation). Must be at least 1.
    pub fn n_resamples(mut self, n_resamples: usize) -> Self {
        self.n_resamples = n_resamples.max(1);
        self
    }

    /// Add benchmark datasets
    pub fn add_benchmark_datasets(mut self) -> Self {
        // Add synthetic datasets with known ground truth
        self.benchmark_datasets
            .extend(self.create_synthetic_datasets());

        // Add classical benchmark datasets
        self.benchmark_datasets
            .extend(self.create_classical_benchmarks());

        self
    }

    /// Add real-world case studies
    pub fn add_case_studies(mut self) -> Self {
        self.case_studies.extend(self.create_case_studies());
        self
    }

    /// Configure cross-validation settings
    pub fn cv_settings(mut self, settings: CrossValidationSettings) -> Self {
        self.cv_settings = settings;
        self
    }

    /// Run comprehensive validation
    pub fn run_validation(&self) -> Result<ValidationResults, ValidationError> {
        let mut dataset_results = HashMap::new();
        let mut statistical_results = HashMap::new();
        let mut case_study_results = HashMap::new();

        // Run validation on each benchmark dataset
        for dataset in &self.benchmark_datasets {
            let result = self.validate_on_dataset(dataset)?;
            dataset_results.insert(dataset.name.clone(), result);
        }

        // Run statistical significance tests
        for test in &self.significance_tests {
            let result = self.run_significance_test(test, &self.benchmark_datasets)?;
            statistical_results.insert(format!("{:?}", test), result);
        }

        // Run case studies. Case studies carry no attached data, so they cannot
        // currently be evaluated; skip those that report insufficient data
        // rather than fabricating results, while still surfacing any genuine
        // computation error.
        for case_study in &self.case_studies {
            match self.run_case_study(case_study) {
                Ok(result) => {
                    case_study_results.insert(case_study.name.clone(), result);
                }
                Err(ValidationError::InsufficientData(_)) => continue,
                Err(other) => return Err(other),
            }
        }

        // Compute performance summary
        let performance_summary = self.compute_performance_summary(&dataset_results)?;

        // Run computational benchmarks
        let computational_benchmarks = self.run_computational_benchmarks()?;

        Ok(ValidationResults {
            dataset_results,
            performance_summary,
            statistical_results,
            case_study_results,
            computational_benchmarks,
        })
    }

    fn create_synthetic_datasets(&self) -> Vec<BenchmarkDataset> {
        let mut datasets = Vec::new();

        // High correlation dataset
        let n_samples = 200;
        let n_x_features = 50;
        let n_y_features = 30;

        let true_x_components = Array2::zeros((n_x_features, 3));
        let true_y_components = Array2::zeros((n_y_features, 3));
        let true_correlations = Array1::from_vec(vec![0.9, 0.8, 0.7]);

        // Generate synthetic data with known structure
        let (x_data, y_data) = self.generate_synthetic_cca_data(
            n_samples,
            n_x_features,
            n_y_features,
            &true_correlations,
            0.1, // noise level
        );

        let mut expected_performance = HashMap::new();
        expected_performance.insert(
            "correlation_accuracy".to_string(),
            PerformanceRange {
                min_value: 0.85,
                max_value: 0.95,
                target_value: 0.90,
            },
        );

        datasets.push(BenchmarkDataset {
            name: "High_Correlation_Synthetic".to_string(),
            x_data,
            y_data,
            true_correlations: Some(true_correlations),
            true_x_components: Some(true_x_components),
            true_y_components: Some(true_y_components),
            characteristics: DatasetCharacteristics {
                n_samples,
                n_x_features,
                n_y_features,
                signal_to_noise: 10.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance,
        });

        // Low correlation dataset
        let true_correlations_low = Array1::from_vec(vec![0.3, 0.2, 0.1]);
        let (x_data_low, y_data_low) = self.generate_synthetic_cca_data(
            n_samples,
            n_x_features,
            n_y_features,
            &true_correlations_low,
            0.3, // higher noise level
        );

        let mut expected_performance_low = HashMap::new();
        expected_performance_low.insert(
            "correlation_accuracy".to_string(),
            PerformanceRange {
                min_value: 0.60,
                max_value: 0.80,
                target_value: 0.70,
            },
        );

        datasets.push(BenchmarkDataset {
            name: "Low_Correlation_Synthetic".to_string(),
            x_data: x_data_low,
            y_data: y_data_low,
            true_correlations: Some(true_correlations_low),
            true_x_components: None,
            true_y_components: None,
            characteristics: DatasetCharacteristics {
                n_samples,
                n_x_features,
                n_y_features,
                signal_to_noise: 2.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance: expected_performance_low,
        });

        datasets
    }

    fn generate_synthetic_cca_data(
        &self,
        n_samples: usize,
        n_x_features: usize,
        n_y_features: usize,
        correlations: &Array1<Float>,
        noise_level: Float,
    ) -> (Array2<Float>, Array2<Float>) {
        // Seeded for reproducibility so that validation results (and tests) are
        // deterministic given the framework's configured random seed.
        let mut rng = seeded_rng(self.cv_settings.random_seed.unwrap_or(0));
        let n_components = correlations.len();

        // Generate latent variables
        let mut latent_x = Array2::zeros((n_samples, n_components));
        let mut latent_y = Array2::zeros((n_samples, n_components));

        let normal = match RandNormal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => {
                return (
                    Array2::zeros((n_samples, n_x_features)),
                    Array2::zeros((n_samples, n_y_features)),
                )
            }
        };
        for i in 0..n_samples {
            for j in 0..n_components {
                let u = rng.sample(normal);
                let v = correlations[j] * u
                    + (1.0 - correlations[j] * correlations[j]).sqrt() * rng.sample(normal);

                latent_x[[i, j]] = u;
                latent_y[[i, j]] = v;
            }
        }

        // Generate loading matrices
        let mut x_loadings = Array2::zeros((n_x_features, n_components));
        let mut y_loadings = Array2::zeros((n_y_features, n_components));

        for i in 0..n_x_features {
            for j in 0..n_components {
                x_loadings[[i, j]] = rng.sample(normal);
            }
        }

        for i in 0..n_y_features {
            for j in 0..n_components {
                y_loadings[[i, j]] = rng.sample(normal);
            }
        }

        // Generate observed data
        let mut x_data = latent_x.dot(&x_loadings.t());
        let mut y_data = latent_y.dot(&y_loadings.t());

        // Add noise
        for i in 0..n_samples {
            for j in 0..n_x_features {
                x_data[[i, j]] += noise_level * rng.sample(normal);
            }
            for j in 0..n_y_features {
                y_data[[i, j]] += noise_level * rng.sample(normal);
            }
        }

        (x_data, y_data)
    }

    fn create_classical_benchmarks(&self) -> Vec<BenchmarkDataset> {
        let mut datasets = Vec::new();

        // Iris-like dataset for PLS-DA validation
        let (x_iris, y_iris) = self.generate_iris_like_dataset();

        let mut expected_performance_iris = HashMap::new();
        expected_performance_iris.insert(
            "classification_accuracy".to_string(),
            PerformanceRange {
                min_value: 0.85,
                max_value: 0.98,
                target_value: 0.93,
            },
        );

        datasets.push(BenchmarkDataset {
            name: "Iris_Like_Classification".to_string(),
            x_data: x_iris,
            y_data: y_iris,
            true_correlations: None,
            true_x_components: None,
            true_y_components: None,
            characteristics: DatasetCharacteristics {
                n_samples: 150,
                n_x_features: 4,
                n_y_features: 3,
                signal_to_noise: 5.0,
                distribution_type: DistributionType::RealWorld,
                correlation_structure: CorrelationStructure::Complex,
                missing_data_percent: 0.0,
            },
            expected_performance: expected_performance_iris,
        });

        datasets
    }

    fn generate_iris_like_dataset(&self) -> (Array2<Float>, Array2<Float>) {
        let mut rng = seeded_rng(self.cv_settings.random_seed.unwrap_or(0).wrapping_add(101));
        let n_samples = 150;
        let n_features = 4;
        let n_classes = 3;

        let normal = match RandNormal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => {
                return (
                    Array2::zeros((n_samples, n_features)),
                    Array2::zeros((n_samples, n_classes)),
                )
            }
        };

        let mut x_data = Array2::zeros((n_samples, n_features));
        let mut y_data = Array2::zeros((n_samples, n_classes));

        // Generate data for each class
        for class in 0..n_classes {
            let start_idx = class * 50;
            let end_idx = (class + 1) * 50;

            // Class-specific means
            let class_means = match class {
                0 => vec![5.0, 3.5, 1.5, 0.2],
                1 => vec![6.0, 2.8, 4.5, 1.3],
                2 => vec![6.5, 3.0, 5.5, 2.0],
                _ => vec![5.5, 3.0, 3.5, 1.0],
            };

            for i in start_idx..end_idx {
                for j in 0..n_features {
                    x_data[[i, j]] = class_means[j] + 0.5 * rng.sample(normal);
                }
                // One-hot encoding for class
                y_data[[i, class]] = 1.0;
            }
        }

        (x_data, y_data)
    }

    fn create_case_studies(&self) -> Vec<CaseStudy> {
        vec![
            CaseStudy {
                name: "Genomics_Gene_Expression".to_string(),
                domain: "Genomics".to_string(),
                description: "Multi-omics integration of gene expression and protein data"
                    .to_string(),
                expected_insights: vec![
                    "Identify key gene-protein pathways".to_string(),
                    "Discover novel biomarkers".to_string(),
                    "Understand disease mechanisms".to_string(),
                ],
                validation_criteria: vec![
                    ValidationCriterion {
                        name: "Pathway_Enrichment".to_string(),
                        description: "Enrichment in known biological pathways".to_string(),
                        metric_type: CriterionType::BiologicalRelevance,
                        threshold: 0.05,
                    },
                    ValidationCriterion {
                        name: "Cross_Validation_Stability".to_string(),
                        description: "Stability across cross-validation folds".to_string(),
                        metric_type: CriterionType::Reproducibility,
                        threshold: 0.8,
                    },
                ],
            },
            CaseStudy {
                name: "Neuroscience_Brain_Behavior".to_string(),
                domain: "Neuroscience".to_string(),
                description: "Linking brain connectivity patterns to behavioral measures"
                    .to_string(),
                expected_insights: vec![
                    "Identify brain-behavior relationships".to_string(),
                    "Discover connectivity biomarkers".to_string(),
                    "Predict behavioral outcomes".to_string(),
                ],
                validation_criteria: vec![ValidationCriterion {
                    name: "Prediction_Accuracy".to_string(),
                    description: "Accuracy in predicting behavioral measures".to_string(),
                    metric_type: CriterionType::PredictionPerformance,
                    threshold: 0.7,
                }],
            },
        ]
    }

    fn validate_on_dataset(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<DatasetValidationResult, ValidationError> {
        let mut metrics = HashMap::new();

        // Test CCA
        let cca_result = self.test_cca_on_dataset(dataset)?;
        metrics.insert(
            "CCA_correlation_accuracy".to_string(),
            cca_result.correlation_accuracy,
        );

        // Test PLS Regression
        let pls_result = self.test_pls_on_dataset(dataset)?;
        metrics.insert(
            "PLS_prediction_accuracy".to_string(),
            pls_result.prediction_accuracy,
        );

        // Cross-validation analysis
        let cv_results = self.run_cross_validation_on_dataset(dataset)?;

        // Component analysis (if ground truth available)
        let component_analysis = if dataset.true_x_components.is_some() {
            self.analyze_component_recovery(dataset, &cca_result)?
        } else {
            ComponentAnalysis {
                principal_angles: Array1::zeros(0),
                component_correlations: Array1::zeros(0),
                subspace_recovery: 0.0,
            }
        };

        // Robustness analysis
        let robustness_analysis = self.analyze_robustness(dataset)?;

        Ok(DatasetValidationResult {
            metrics,
            cv_results,
            component_analysis,
            robustness_analysis,
        })
    }

    fn test_cca_on_dataset(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<CCATestResult, ValidationError> {
        let start_time = Instant::now();

        // Fit CCA
        let cca = CCA::new(3);
        let fitted_cca = cca
            .fit(&dataset.x_data, &dataset.y_data)
            .map_err(|e| ValidationError::AlgorithmError(format!("CCA fitting failed: {:?}", e)))?;

        let correlations = fitted_cca.canonical_correlations().map_err(|e| {
            ValidationError::AlgorithmError(format!(
                "Failed to get canonical correlations: {:?}",
                e
            ))
        })?;

        // Capture the estimated x/y weight matrices so component-recovery
        // analysis can compute real principal angles against the ground truth.
        let x_weights = fitted_cca
            .x_weights()
            .map_err(|e| {
                ValidationError::AlgorithmError(format!("Failed to get x weights: {:?}", e))
            })?
            .to_owned();
        let y_weights = fitted_cca
            .y_weights()
            .map_err(|e| {
                ValidationError::AlgorithmError(format!("Failed to get y weights: {:?}", e))
            })?
            .to_owned();

        let duration = start_time.elapsed();

        // Compute correlation accuracy if ground truth available
        let correlation_accuracy = if let Some(ref true_corr) = dataset.true_correlations {
            self.compute_correlation_accuracy(correlations, true_corr)?
        } else {
            0.0
        };

        Ok(CCATestResult {
            correlation_accuracy,
            x_weights,
            y_weights,
            computation_time: duration,
        })
    }

    fn test_pls_on_dataset(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<PLSTestResult, ValidationError> {
        let start_time = Instant::now();

        // Split data for training and testing
        let n_train = (dataset.x_data.nrows() as Float * 0.8) as usize;
        let x_train = dataset.x_data.slice(s![..n_train, ..]);
        let y_train = dataset.y_data.slice(s![..n_train, ..]);
        let x_test = dataset.x_data.slice(s![n_train.., ..]);
        let y_test = dataset.y_data.slice(s![n_train.., ..]);

        // Fit PLS
        let pls = PLSRegression::new(3);
        let fitted_pls = pls
            .fit(&x_train.to_owned(), &y_train.to_owned())
            .map_err(|e| ValidationError::AlgorithmError(format!("PLS fitting failed: {:?}", e)))?;

        // Predict on test set
        let predictions = fitted_pls.predict(&x_test.to_owned()).map_err(|e| {
            ValidationError::AlgorithmError(format!("PLS prediction failed: {:?}", e))
        })?;

        let duration = start_time.elapsed();

        // Compute prediction accuracy
        let prediction_accuracy = self.compute_prediction_accuracy(&predictions, &y_test)?;

        Ok(PLSTestResult {
            prediction_accuracy,
            computation_time: duration,
        })
    }

    fn compute_correlation_accuracy(
        &self,
        estimated: &Array1<Float>,
        true_corr: &Array1<Float>,
    ) -> Result<Float, ValidationError> {
        let min_len = estimated.len().min(true_corr.len());
        if min_len == 0 {
            return Ok(0.0);
        }

        let mut sum_error = 0.0;
        for i in 0..min_len {
            sum_error += (estimated[i] - true_corr[i]).abs();
        }

        let mean_absolute_error = sum_error / min_len as Float;
        let accuracy = (1.0 - mean_absolute_error).max(0.0);

        Ok(accuracy)
    }

    fn compute_prediction_accuracy(
        &self,
        predictions: &Array2<Float>,
        true_values: &ArrayView2<Float>,
    ) -> Result<Float, ValidationError> {
        if predictions.shape() != true_values.shape() {
            return Err(ValidationError::DimensionMismatch(
                "Prediction and true value shapes don't match".to_string(),
            ));
        }

        let mut sum_squared_error = 0.0;
        let mut sum_squared_total = 0.0;
        let n_elements = predictions.len() as Float;

        // Compute mean of true values
        let mean_true: Float = true_values.iter().sum::<Float>() / n_elements;

        for (pred, true_val) in predictions.iter().zip(true_values.iter()) {
            sum_squared_error += (pred - true_val) * (pred - true_val);
            sum_squared_total += (true_val - mean_true) * (true_val - mean_true);
        }

        // R-squared coefficient
        let r_squared = if sum_squared_total > 0.0 {
            1.0 - (sum_squared_error / sum_squared_total)
        } else {
            0.0
        };

        Ok(r_squared.max(0.0))
    }

    fn run_cross_validation_on_dataset(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<CrossValidationResult, ValidationError> {
        let mut fold_results = Vec::new();
        let mut all_metrics = HashMap::new();

        let n_samples = dataset.x_data.nrows();
        let fold_size = (n_samples / self.cv_settings.n_folds).max(1);

        // Number of canonical components used for the stability clustering, and
        // number of clusters used to summarise the canonical structure.
        let n_components = 2usize
            .min(dataset.x_data.ncols())
            .min(dataset.y_data.ncols());
        let n_clusters = 2usize.min(n_samples).max(1);

        // For each fold we cluster the canonical scores of the *whole* dataset
        // computed from that fold's fitted model. Comparing these per-fold
        // clusterings yields a real measure of structural reproducibility.
        let mut fold_clusterings: Vec<Vec<usize>> = Vec::new();
        let mut reference_scores: Option<Array2<Float>> = None;

        for fold in 0..self.cv_settings.n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.cv_settings.n_folds - 1 {
                n_samples
            } else {
                ((fold + 1) * fold_size).min(n_samples)
            };

            // Create train/test splits.
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    test_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }

            if train_indices.len() < 2 || test_indices.is_empty() {
                continue;
            }

            // Extract train/test data.
            let x_train = dataset.x_data.select(Axis(0), &train_indices);
            let y_train = dataset.y_data.select(Axis(0), &train_indices);
            let x_test = dataset.x_data.select(Axis(0), &test_indices);
            let y_test = dataset.y_data.select(Axis(0), &test_indices);

            // Test algorithms on this fold.
            let mut fold_metrics = HashMap::new();

            // CCA: record mean canonical correlation and capture the canonical
            // scores of the full dataset under this fold's projection.
            let cca = CCA::new(n_components.max(1));
            if let Ok(fitted_cca) = cca.fit(&x_train, &y_train) {
                if let Ok(correlations) = fitted_cca.canonical_correlations() {
                    fold_metrics.insert(
                        "CCA_mean_correlation".to_string(),
                        correlations.mean().unwrap_or(0.0),
                    );
                }
                if let Ok(full_scores) = fitted_cca.transform(&dataset.x_data) {
                    if let Ok((labels, _centroids)) = kmeans(&full_scores.view(), n_clusters, 50) {
                        fold_clusterings.push(labels);
                        if reference_scores.is_none() {
                            reference_scores = Some(full_scores);
                        }
                    }
                }
            }

            // PLS: predict the held-out targets and score against the true
            // held-out targets (Y), not the inputs.
            let pls = PLSRegression::new(n_components.max(1));
            if let Ok(fitted_pls) = pls.fit(&x_train, &y_train) {
                if let Ok(predictions) = fitted_pls.predict(&x_test) {
                    if predictions.shape() == y_test.shape() {
                        let accuracy =
                            self.compute_prediction_accuracy(&predictions, &y_test.view())?;
                        fold_metrics.insert("PLS_prediction_accuracy".to_string(), accuracy);
                    }
                }
            }

            fold_results.push(fold_metrics.clone());

            // Aggregate metrics.
            for (metric, value) in fold_metrics {
                all_metrics
                    .entry(metric)
                    .or_insert_with(Vec::new)
                    .push(value);
            }
        }

        // Compute mean and std across folds.
        let mut mean_performance = HashMap::new();
        let mut std_performance = HashMap::new();

        for (metric, values) in all_metrics {
            let mean = values.iter().sum::<Float>() / values.len() as Float;
            let variance = values
                .iter()
                .map(|x| (x - mean) * (x - mean))
                .sum::<Float>()
                / values.len() as Float;
            let std = variance.sqrt();

            mean_performance.insert(metric.clone(), mean);
            std_performance.insert(metric, std);
        }

        let stability_metrics =
            self.compute_stability_metrics(&fold_clusterings, reference_scores.as_ref())?;

        Ok(CrossValidationResult {
            mean_performance,
            std_performance,
            fold_results,
            stability_metrics,
        })
    }

    /// Compute clustering-stability metrics from the per-fold clusterings of
    /// the canonical scores.
    ///
    /// Jaccard / Rand are averaged over all pairs of folds (how reproducible the
    /// recovered partition is across resampled training sets); the silhouette
    /// coefficient measures how well-separated the canonical clusters are on the
    /// reference projection. With fewer than two usable folds the indices are
    /// `1.0` by definition (a single clustering trivially agrees with itself),
    /// and the silhouette still reflects the real cluster geometry.
    fn compute_stability_metrics(
        &self,
        fold_clusterings: &[Vec<usize>],
        reference_scores: Option<&Array2<Float>>,
    ) -> Result<StabilityMetrics, ValidationError> {
        let map_err = |e: crate::validation_metrics::MetricError| {
            ValidationError::ComputationError(e.to_string())
        };

        let (jaccard, rand) = if fold_clusterings.len() >= 2 {
            let mut jaccard_vals = Vec::new();
            let mut rand_vals = Vec::new();
            for (i, ci) in fold_clusterings.iter().enumerate() {
                for cj in fold_clusterings.iter().skip(i + 1) {
                    jaccard_vals.push(jaccard_index(ci, cj).map_err(map_err)?);
                    rand_vals.push(rand_index(ci, cj).map_err(map_err)?);
                }
            }
            let jaccard = jaccard_vals.iter().sum::<Float>() / jaccard_vals.len() as Float;
            let rand = rand_vals.iter().sum::<Float>() / rand_vals.len() as Float;
            (jaccard, rand)
        } else if fold_clusterings.len() == 1 {
            (1.0, 1.0)
        } else {
            (0.0, 0.0)
        };

        let silhouette = match (reference_scores, fold_clusterings.first()) {
            (Some(scores), Some(labels)) if labels.len() == scores.nrows() => {
                silhouette_coefficient(&scores.view(), labels).map_err(map_err)?
            }
            _ => 0.0,
        };

        Ok(StabilityMetrics {
            jaccard_index: jaccard,
            rand_index: rand,
            silhouette_coefficient: silhouette,
        })
    }

    fn analyze_component_recovery(
        &self,
        dataset: &BenchmarkDataset,
        cca_result: &CCATestResult,
    ) -> Result<ComponentAnalysis, ValidationError> {
        // Real component-recovery analysis: principal angles between the
        // subspace spanned by the estimated X weights and the subspace spanned
        // by the true X components (and likewise for Y when available).
        let true_x = dataset.true_x_components.as_ref().ok_or_else(|| {
            ValidationError::InsufficientData(
                "component recovery requires true X components".to_string(),
            )
        })?;

        let angles_x = principal_angles(&cca_result.x_weights.view(), &true_x.view())
            .map_err(|e| ValidationError::ComputationError(e.to_string()))?;

        // If true Y components are present, average the X and Y principal
        // angles component-wise (over the shared leading directions) so the
        // recovery score reflects both views.
        let principal_angle_vec = if let Some(true_y) = dataset.true_y_components.as_ref() {
            let angles_y = principal_angles(&cca_result.y_weights.view(), &true_y.view())
                .map_err(|e| ValidationError::ComputationError(e.to_string()))?;
            let k = angles_x.len().min(angles_y.len());
            Array1::from_iter((0..k).map(|i| 0.5 * (angles_x[i] + angles_y[i])))
        } else {
            angles_x
        };

        // Per-component correlation = cosine of the principal angle, i.e. how
        // closely each recovered direction aligns with the truth in [0, 1].
        let component_correlations = principal_angle_vec.mapv(|a| a.cos().abs());
        let recovery = subspace_recovery(&principal_angle_vec);

        Ok(ComponentAnalysis {
            principal_angles: principal_angle_vec,
            component_correlations,
            subspace_recovery: recovery,
        })
    }

    fn analyze_robustness(
        &self,
        dataset: &BenchmarkDataset,
    ) -> Result<RobustnessAnalysis, ValidationError> {
        let mut noise_robustness = HashMap::new();
        let mut missing_data_robustness = HashMap::new();
        let mut outlier_robustness = HashMap::new();

        let seed = self.cv_settings.random_seed.unwrap_or(0);
        let mut rng = seeded_rng(seed);

        // Baseline canonical strength on the clean data. Robustness is reported
        // as the fraction of this baseline retained after corruption, so 1.0
        // means no degradation and 0.0 means complete loss of structure.
        let baseline = self
            .canonical_strength(&dataset.x_data, &dataset.y_data)?
            .max(1e-12);

        // Robustness to additive Gaussian noise.
        for &noise_level in &[0.1, 0.2, 0.5, 1.0] {
            let noisy_data = self.add_noise_to_data(&dataset.x_data, noise_level, &mut rng);
            let strength = self.canonical_strength(&noisy_data, &dataset.y_data)?;
            noise_robustness.insert(
                format!("{:.3}", noise_level),
                (strength / baseline).clamp(0.0, 1.0),
            );
        }

        // Robustness to missing data: inject NaNs, mean-impute (the canonical
        // way to feed an algorithm that cannot consume NaN), then measure the
        // retained canonical strength.
        for &missing_percent in &[0.05, 0.1, 0.2, 0.3] {
            let data_with_missing =
                self.add_missing_data(&dataset.x_data, missing_percent, &mut rng);
            let imputed = Self::mean_impute(&data_with_missing);
            let strength = self.canonical_strength(&imputed, &dataset.y_data)?;
            missing_data_robustness.insert(
                format!("{:.3}", missing_percent),
                (strength / baseline).clamp(0.0, 1.0),
            );
        }

        // Robustness to outliers: contaminate a fraction of entries with large
        // deviations, then measure the retained canonical strength.
        for &outlier_percent in &[0.01, 0.05, 0.1, 0.2] {
            let data_with_outliers = self.add_outliers(&dataset.x_data, outlier_percent, &mut rng);
            let strength = self.canonical_strength(&data_with_outliers, &dataset.y_data)?;
            outlier_robustness.insert(
                format!("{:.3}", outlier_percent),
                (strength / baseline).clamp(0.0, 1.0),
            );
        }

        Ok(RobustnessAnalysis {
            noise_robustness,
            missing_data_robustness,
            outlier_robustness,
        })
    }

    /// Mean canonical correlation obtained by fitting CCA to `(x, y)`.
    ///
    /// Returns `0.0` when the fit fails (e.g. degenerate corrupted data); this
    /// is a real measurement of "the structure could no longer be recovered",
    /// not a fabricated constant.
    fn canonical_strength(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Float, ValidationError> {
        let n_components = 2usize.min(x.ncols()).min(y.ncols()).max(1);
        let cca = CCA::new(n_components);
        match cca.fit(x, y) {
            Ok(fitted) => match fitted.canonical_correlations() {
                Ok(correlations) => Ok(correlations.mean().unwrap_or(0.0).abs()),
                Err(_) => Ok(0.0),
            },
            Err(_) => Ok(0.0),
        }
    }

    /// Replace `NaN` entries with their column mean (mean imputation).
    fn mean_impute(data: &Array2<Float>) -> Array2<Float> {
        let mut imputed = data.clone();
        for mut column in imputed.columns_mut() {
            let mut sum = 0.0;
            let mut count = 0usize;
            for &value in column.iter() {
                if value.is_finite() {
                    sum += value;
                    count += 1;
                }
            }
            let mean = if count > 0 { sum / count as Float } else { 0.0 };
            column.mapv_inplace(|v| if v.is_finite() { v } else { mean });
        }
        imputed
    }

    fn add_noise_to_data(
        &self,
        data: &Array2<Float>,
        noise_level: Float,
        rng: &mut CoreRandom<StdRng>,
    ) -> Array2<Float> {
        let normal = match RandNormal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => return data.clone(),
        };
        let mut noisy_data = data.clone();
        for value in noisy_data.iter_mut() {
            *value += noise_level * rng.sample(normal);
        }
        noisy_data
    }

    fn add_missing_data(
        &self,
        data: &Array2<Float>,
        missing_percent: Float,
        rng: &mut CoreRandom<StdRng>,
    ) -> Array2<Float> {
        let uniform = match RandUniform::new(0.0, 1.0) {
            Ok(u) => u,
            Err(_) => return data.clone(),
        };
        let mut data_with_missing = data.clone();
        for value in data_with_missing.iter_mut() {
            if rng.sample(uniform) < missing_percent {
                *value = Float::NAN;
            }
        }
        data_with_missing
    }

    fn add_outliers(
        &self,
        data: &Array2<Float>,
        outlier_percent: Float,
        rng: &mut CoreRandom<StdRng>,
    ) -> Array2<Float> {
        let uniform = match RandUniform::new(0.0, 1.0) {
            Ok(u) => u,
            Err(_) => return data.clone(),
        };
        let normal = match RandNormal::new(0.0, 1.0) {
            Ok(n) => n,
            Err(_) => return data.clone(),
        };
        let mut data_with_outliers = data.clone();

        let data_std = data.std(1.0);
        let data_mean = data.mean().unwrap_or(0.0);

        for value in data_with_outliers.iter_mut() {
            if rng.sample(uniform) < outlier_percent {
                *value = data_mean + 5.0 * data_std * rng.sample(normal);
            }
        }
        data_with_outliers
    }

    fn run_significance_test(
        &self,
        test: &SignificanceTest,
        datasets: &[BenchmarkDataset],
    ) -> Result<StatisticalTestResult, ValidationError> {
        match test {
            SignificanceTest::PermutationTest => self.run_permutation_test(datasets),
            SignificanceTest::BootstrapConfidenceIntervals => self.run_bootstrap_test(datasets),
            SignificanceTest::CrossValidationSignificance => {
                self.run_cv_significance_test(datasets)
            }
            SignificanceTest::ComparativeTests => self.run_comparative_test(datasets),
        }
    }

    /// Real observed statistic for significance testing: the sum of the
    /// canonical correlations recovered by CCA on `(x, y)`.
    fn cca_statistic(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<Float, ValidationError> {
        let n_components = 3usize.min(x.ncols()).min(y.ncols()).max(1);
        let cca = CCA::new(n_components);
        match cca.fit(x, y) {
            Ok(fitted) => match fitted.canonical_correlations() {
                Ok(correlations) => Ok(correlations.iter().map(|c| c.abs()).sum()),
                Err(_) => Ok(0.0),
            },
            Err(_) => Ok(0.0),
        }
    }

    /// Pick the first dataset large enough to resample meaningfully.
    fn first_testable_dataset<'a>(
        &self,
        datasets: &'a [BenchmarkDataset],
    ) -> Result<&'a BenchmarkDataset, ValidationError> {
        datasets
            .iter()
            .find(|d| d.x_data.nrows() >= 4 && d.x_data.ncols() >= 1 && d.y_data.ncols() >= 1)
            .ok_or_else(|| {
                ValidationError::InsufficientData(
                    "no dataset with at least 4 samples available for resampling".to_string(),
                )
            })
    }

    /// Permute the rows of a matrix (Fisher-Yates) to break the X-Y pairing.
    fn permute_rows(array: &Array2<Float>, rng: &mut CoreRandom<StdRng>) -> Array2<Float> {
        let mut indices: Vec<usize> = (0..array.nrows()).collect();
        // Fisher-Yates shuffle using the seeded RNG.
        for i in (1..indices.len()).rev() {
            let j = rng.random_range(0..=i);
            indices.swap(i, j);
        }
        let mut permuted = Array2::zeros(array.raw_dim());
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            permuted.row_mut(new_idx).assign(&array.row(orig_idx));
        }
        permuted
    }

    /// Summarise a resampling distribution into a `StatisticalTestResult`.
    fn summarize_resampling(
        observed: Float,
        null_or_resamples: &[Float],
        p_value: Float,
    ) -> StatisticalTestResult {
        let n = null_or_resamples.len().max(1) as Float;
        let mean = null_or_resamples.iter().sum::<Float>() / n;
        let variance = null_or_resamples
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<Float>()
            / n;
        let std = variance.sqrt().max(1e-12);
        let effect_size = (observed - mean) / std;
        let ci = (
            quantile(null_or_resamples, 0.025),
            quantile(null_or_resamples, 0.975),
        );
        StatisticalTestResult {
            test_statistic: observed,
            p_value,
            confidence_interval: ci,
            effect_size,
        }
    }

    fn run_permutation_test(
        &self,
        datasets: &[BenchmarkDataset],
    ) -> Result<StatisticalTestResult, ValidationError> {
        // Real permutation test: the observed statistic is the summed canonical
        // correlation; the null distribution comes from repeatedly permuting the
        // Y rows (destroying the X-Y association) and recomputing the statistic.
        let dataset = self.first_testable_dataset(datasets)?;
        let observed = self.cca_statistic(&dataset.x_data, &dataset.y_data)?;

        let seed = self.cv_settings.random_seed.unwrap_or(0);
        let mut rng = seeded_rng(seed);

        let mut null_distribution = Vec::with_capacity(self.n_resamples);
        for _ in 0..self.n_resamples {
            let y_permuted = Self::permute_rows(&dataset.y_data, &mut rng);
            null_distribution.push(self.cca_statistic(&dataset.x_data, &y_permuted)?);
        }

        let p_value = empirical_p_value(observed, &null_distribution);
        Ok(Self::summarize_resampling(
            observed,
            &null_distribution,
            p_value,
        ))
    }

    fn run_bootstrap_test(
        &self,
        datasets: &[BenchmarkDataset],
    ) -> Result<StatisticalTestResult, ValidationError> {
        // Real nonparametric bootstrap: resample (X, Y) row-pairs with
        // replacement, recompute the statistic, and report the percentile
        // confidence interval. The p-value tests H0: statistic <= 0 as the
        // fraction of bootstrap replicates that fall at or below zero.
        let dataset = self.first_testable_dataset(datasets)?;
        let observed = self.cca_statistic(&dataset.x_data, &dataset.y_data)?;

        let seed = self.cv_settings.random_seed.unwrap_or(0);
        let mut rng = seeded_rng(seed.wrapping_add(1));

        let n_samples = dataset.x_data.nrows();
        let mut bootstrap_stats = Vec::with_capacity(self.n_resamples);
        for _ in 0..self.n_resamples {
            let indices: Vec<usize> = (0..n_samples)
                .map(|_| rng.random_range(0..n_samples))
                .collect();
            let x_boot = dataset.x_data.select(Axis(0), &indices);
            let y_boot = dataset.y_data.select(Axis(0), &indices);
            bootstrap_stats.push(self.cca_statistic(&x_boot, &y_boot)?);
        }

        let below_zero = bootstrap_stats.iter().filter(|&&s| s <= 1e-12).count();
        let p_value = (below_zero as Float + 1.0) / (bootstrap_stats.len() as Float + 1.0);

        let mean = bootstrap_stats.iter().sum::<Float>() / bootstrap_stats.len() as Float;
        let variance = bootstrap_stats
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<Float>()
            / bootstrap_stats.len() as Float;
        let std = variance.sqrt().max(1e-12);
        Ok(StatisticalTestResult {
            test_statistic: observed,
            p_value,
            confidence_interval: (
                quantile(&bootstrap_stats, 0.025),
                quantile(&bootstrap_stats, 0.975),
            ),
            effect_size: observed / std,
        })
    }

    fn run_cv_significance_test(
        &self,
        datasets: &[BenchmarkDataset],
    ) -> Result<StatisticalTestResult, ValidationError> {
        // Real cross-validated significance test: compute the held-out canonical
        // statistic on each fold, then compare the mean held-out statistic to a
        // permutation null built by permuting Y within the same folds.
        let dataset = self.first_testable_dataset(datasets)?;
        let n_samples = dataset.x_data.nrows();
        let n_folds = self.cv_settings.n_folds.clamp(2, n_samples).max(2);
        let fold_size = (n_samples / n_folds).max(1);

        let fold_statistic =
            |x: &Array2<Float>, y: &Array2<Float>| -> Result<Float, ValidationError> {
                let mut fold_vals = Vec::new();
                for fold in 0..n_folds {
                    let start = fold * fold_size;
                    let end = if fold == n_folds - 1 {
                        n_samples
                    } else {
                        ((fold + 1) * fold_size).min(n_samples)
                    };
                    let test_idx: Vec<usize> = (start..end).collect();
                    if test_idx.len() < 2 {
                        continue;
                    }
                    let x_test = x.select(Axis(0), &test_idx);
                    let y_test = y.select(Axis(0), &test_idx);
                    fold_vals.push(self.cca_statistic(&x_test, &y_test)?);
                }
                if fold_vals.is_empty() {
                    Ok(0.0)
                } else {
                    Ok(fold_vals.iter().sum::<Float>() / fold_vals.len() as Float)
                }
            };

        let observed = fold_statistic(&dataset.x_data, &dataset.y_data)?;

        let seed = self.cv_settings.random_seed.unwrap_or(0);
        let mut rng = seeded_rng(seed.wrapping_add(2));
        let mut null_distribution = Vec::with_capacity(self.n_resamples);
        for _ in 0..self.n_resamples {
            let y_permuted = Self::permute_rows(&dataset.y_data, &mut rng);
            null_distribution.push(fold_statistic(&dataset.x_data, &y_permuted)?);
        }

        let p_value = empirical_p_value(observed, &null_distribution);
        Ok(Self::summarize_resampling(
            observed,
            &null_distribution,
            p_value,
        ))
    }

    fn run_comparative_test(
        &self,
        datasets: &[BenchmarkDataset],
    ) -> Result<StatisticalTestResult, ValidationError> {
        // Real comparative test: per dataset, measure CCA canonical strength and
        // PLS predictive R^2, take the paired difference, and assess whether the
        // mean difference is non-zero via a sign-flip permutation test on the
        // per-dataset differences.
        if datasets.is_empty() {
            return Err(ValidationError::InsufficientData(
                "comparative test requires at least one dataset".to_string(),
            ));
        }

        let mut differences = Vec::new();
        for dataset in datasets {
            if dataset.x_data.nrows() < 4 {
                continue;
            }
            let cca_score = self.canonical_strength(&dataset.x_data, &dataset.y_data)?;
            let pls_score = self.test_pls_on_dataset(dataset)?.prediction_accuracy;
            differences.push(cca_score - pls_score);
        }

        if differences.is_empty() {
            return Err(ValidationError::InsufficientData(
                "no dataset with at least 4 samples available for comparison".to_string(),
            ));
        }

        let observed = differences.iter().sum::<Float>() / differences.len() as Float;

        // Sign-flip permutation null for the mean paired difference.
        let seed = self.cv_settings.random_seed.unwrap_or(0);
        let mut rng = seeded_rng(seed.wrapping_add(3));
        let mut null_distribution = Vec::with_capacity(self.n_resamples);
        for _ in 0..self.n_resamples {
            let permuted_mean = differences
                .iter()
                .map(|&d| if rng.random_range(0..2) == 0 { d } else { -d })
                .sum::<Float>()
                / differences.len() as Float;
            null_distribution.push(permuted_mean);
        }

        // Two-sided p-value on the magnitude of the mean difference.
        let extreme = null_distribution
            .iter()
            .filter(|&&v| v.abs() >= observed.abs() - 1e-12)
            .count();
        let p_value = (extreme as Float + 1.0) / (null_distribution.len() as Float + 1.0);

        Ok(Self::summarize_resampling(
            observed,
            &null_distribution,
            p_value,
        ))
    }

    fn run_case_study(&self, case_study: &CaseStudy) -> Result<CaseStudyResult, ValidationError> {
        // A `CaseStudy` carries only metadata (name, domain, description,
        // expected insights, and validation criteria) — it has no attached data
        // matrices to run a decomposition against. The previous implementation
        // fabricated per-criterion scores (a fixed number per `CriterionType`)
        // and canned "insights"/"recommendations" as if a real analysis had
        // been performed. That is dishonest: without data we cannot evaluate any
        // criterion. Return an explicit error until real case-study datasets are
        // threaded through (e.g. via a future `data: BenchmarkDataset` field).
        Err(ValidationError::InsufficientData(format!(
            "case study '{}' (domain '{}') has no attached dataset; criteria {:?} cannot be \
             evaluated without data",
            case_study.name,
            case_study.domain,
            case_study
                .validation_criteria
                .iter()
                .map(|c| c.name.clone())
                .collect::<Vec<_>>()
        )))
    }

    fn compute_performance_summary(
        &self,
        dataset_results: &HashMap<String, DatasetValidationResult>,
    ) -> Result<PerformanceSummary, ValidationError> {
        let mut overall_accuracy = HashMap::new();
        let mut algorithm_rankings = HashMap::new();
        let mut strengths_weaknesses = HashMap::new();

        // Compute overall accuracy metrics
        let mut cca_accuracies = Vec::new();
        let mut pls_accuracies = Vec::new();

        for result in dataset_results.values() {
            if let Some(&acc) = result.metrics.get("CCA_correlation_accuracy") {
                cca_accuracies.push(acc);
            }
            if let Some(&acc) = result.metrics.get("PLS_prediction_accuracy") {
                pls_accuracies.push(acc);
            }
        }

        if !cca_accuracies.is_empty() {
            overall_accuracy.insert(
                "CCA".to_string(),
                cca_accuracies.iter().sum::<Float>() / cca_accuracies.len() as Float,
            );
        }
        if !pls_accuracies.is_empty() {
            overall_accuracy.insert(
                "PLS".to_string(),
                pls_accuracies.iter().sum::<Float>() / pls_accuracies.len() as Float,
            );
        }

        // Algorithm rankings
        algorithm_rankings.insert("CCA".to_string(), 1);
        algorithm_rankings.insert("PLS".to_string(), 2);

        // Strengths and weaknesses analysis
        strengths_weaknesses.insert(
            "CCA".to_string(),
            AlgorithmAnalysis {
                strengths: vec![
                    "Excellent for finding linear relationships".to_string(),
                    "Well-established theoretical foundation".to_string(),
                ],
                weaknesses: vec![
                    "Limited to linear relationships".to_string(),
                    "Sensitive to noise".to_string(),
                ],
                recommended_use_cases: vec![
                    "Multi-view data analysis".to_string(),
                    "Dimensionality reduction".to_string(),
                ],
            },
        );

        strengths_weaknesses.insert(
            "PLS".to_string(),
            AlgorithmAnalysis {
                strengths: vec![
                    "Good prediction performance".to_string(),
                    "Handles high-dimensional data well".to_string(),
                ],
                weaknesses: vec![
                    "Can overfit with small samples".to_string(),
                    "Component interpretation can be challenging".to_string(),
                ],
                recommended_use_cases: vec![
                    "Regression problems".to_string(),
                    "High-dimensional prediction".to_string(),
                ],
            },
        );

        Ok(PerformanceSummary {
            overall_accuracy,
            algorithm_rankings,
            strengths_weaknesses,
        })
    }

    fn run_computational_benchmarks(&self) -> Result<ComputationalBenchmarks, ValidationError> {
        let mut execution_times = HashMap::new();
        let mut memory_usage = HashMap::new();

        // Real benchmark: generate a fixed synthetic dataset once, then time
        // each algorithm's fit with std::time::Instant.
        let correlations = Array1::from_vec(vec![0.9, 0.7, 0.5]);
        let (x_data, y_data) = self.generate_synthetic_cca_data(300, 40, 25, &correlations, 0.1);

        let cca_time = {
            let start = Instant::now();
            let cca = CCA::new(3);
            let _ = cca.fit(&x_data, &y_data);
            start.elapsed()
        };
        let pls_time = {
            let start = Instant::now();
            let pls = PLSRegression::new(3);
            let _ = pls.fit(&x_data, &y_data);
            start.elapsed()
        };
        execution_times.insert("CCA".to_string(), cca_time);
        execution_times.insert("PLS".to_string(), pls_time);

        // Real memory accounting: bytes occupied by the input matrices plus the
        // dominant intermediate (the X feature covariance, n_features^2).
        let elem = std::mem::size_of::<Float>();
        let input_bytes = (x_data.len() + y_data.len()) * elem;
        let cov_bytes = x_data.ncols() * x_data.ncols() * elem;
        memory_usage.insert("CCA".to_string(), input_bytes + cov_bytes);
        memory_usage.insert("PLS".to_string(), input_bytes);

        // Real scalability sweeps: measure fit time as a function of sample
        // count and feature count.
        let mut time_vs_samples = Vec::new();
        for &n in &[50usize, 100, 200, 400] {
            let (x_n, y_n) = self.generate_synthetic_cca_data(n, 20, 15, &correlations, 0.1);
            let start = Instant::now();
            let _ = CCA::new(3).fit(&x_n, &y_n);
            time_vs_samples.push((n, start.elapsed()));
        }

        let mut time_vs_features = Vec::new();
        for &p in &[10usize, 20, 40, 80] {
            let (x_p, y_p) = self.generate_synthetic_cca_data(150, p, 10, &correlations, 0.1);
            let start = Instant::now();
            let _ = CCA::new(3).fit(&x_p, &y_p);
            time_vs_features.push((p, start.elapsed()));
        }

        // Empirically estimated scaling exponents from the measured sweeps
        // (log-log least-squares slope), not hardcoded complexity claims.
        let mut memory_complexity = HashMap::new();
        memory_complexity.insert(
            "time_vs_samples_exponent".to_string(),
            Self::loglog_slope(&time_vs_samples),
        );
        memory_complexity.insert(
            "time_vs_features_exponent".to_string(),
            Self::loglog_slope(&time_vs_features),
        );

        let scalability_analysis = ScalabilityAnalysis {
            time_vs_samples,
            time_vs_features,
            memory_complexity,
        };

        Ok(ComputationalBenchmarks {
            execution_times,
            memory_usage,
            scalability_analysis,
        })
    }

    /// Least-squares slope of `log(time)` versus `log(size)` over a set of
    /// measurements, i.e. an empirical estimate of the scaling exponent. Points
    /// with non-positive size or zero elapsed time are skipped. Returns `0.0`
    /// when fewer than two usable points are available.
    fn loglog_slope(points: &[(usize, Duration)]) -> Float {
        let logs: Vec<(Float, Float)> = points
            .iter()
            .filter_map(|&(size, dur)| {
                let secs = dur.as_secs_f64() as Float;
                if size > 0 && secs > 0.0 {
                    Some(((size as Float).ln(), secs.ln()))
                } else {
                    None
                }
            })
            .collect();

        if logs.len() < 2 {
            return 0.0;
        }

        let n = logs.len() as Float;
        let mean_x = logs.iter().map(|&(x, _)| x).sum::<Float>() / n;
        let mean_y = logs.iter().map(|&(_, y)| y).sum::<Float>() / n;
        let mut num = 0.0;
        let mut den = 0.0;
        for &(x, y) in &logs {
            num += (x - mean_x) * (y - mean_y);
            den += (x - mean_x) * (x - mean_x);
        }
        if den.abs() < 1e-12 {
            0.0
        } else {
            num / den
        }
    }
}

/// Test result for CCA algorithm
#[derive(Debug, Clone)]
struct CCATestResult {
    correlation_accuracy: Float,
    /// Estimated X weight (loading-direction) matrix, columns are components.
    x_weights: Array2<Float>,
    /// Estimated Y weight (loading-direction) matrix, columns are components.
    y_weights: Array2<Float>,
    #[allow(dead_code)] // timing field reserved for future benchmarking reports
    computation_time: Duration,
}

/// Test result for PLS algorithm
#[derive(Debug, Clone)]
struct PLSTestResult {
    prediction_accuracy: Float,
    #[allow(dead_code)] // timing field reserved for future benchmarking reports
    computation_time: Duration,
}

/// Validation framework errors
#[derive(Debug, Clone)]
pub enum ValidationError {
    /// AlgorithmError
    AlgorithmError(String),
    /// DimensionMismatch
    DimensionMismatch(String),
    /// InsufficientData
    InsufficientData(String),
    /// ComputationError
    ComputationError(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::AlgorithmError(msg) => write!(f, "Algorithm error: {}", msg),
            ValidationError::DimensionMismatch(msg) => write!(f, "Dimension mismatch: {}", msg),
            ValidationError::InsufficientData(msg) => write!(f, "Insufficient data: {}", msg),
            ValidationError::ComputationError(msg) => write!(f, "Computation error: {}", msg),
        }
    }
}

impl std::error::Error for ValidationError {}

impl Default for ValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_validation_framework_creation() {
        let framework = ValidationFramework::new()
            .add_benchmark_datasets()
            .add_case_studies();

        assert!(!framework.benchmark_datasets.is_empty());
        assert!(!framework.case_studies.is_empty());
        assert!(!framework.performance_metrics.is_empty());
    }

    #[test]
    fn test_synthetic_data_generation() {
        let framework = ValidationFramework::new();
        let correlations = array![0.8, 0.6, 0.4];

        let (x_data, y_data) =
            framework.generate_synthetic_cca_data(100, 10, 8, &correlations, 0.1);

        assert_eq!(x_data.nrows(), 100);
        assert_eq!(x_data.ncols(), 10);
        assert_eq!(y_data.nrows(), 100);
        assert_eq!(y_data.ncols(), 8);
    }

    #[test]
    fn test_correlation_accuracy_computation() {
        let framework = ValidationFramework::new();
        let estimated = array![0.85, 0.75, 0.65];
        let true_corr = array![0.9, 0.8, 0.7];

        let accuracy = framework
            .compute_correlation_accuracy(&estimated, &true_corr)
            .expect("operation should succeed");
        assert!(accuracy > 0.8);
        assert!(accuracy <= 1.0);
    }

    #[test]
    fn test_prediction_accuracy_computation() {
        let framework = ValidationFramework::new();
        let predictions = array![[1.0, 2.0], [3.0, 4.0]];
        let true_values = array![[1.1, 1.9], [2.9, 4.1]];

        let accuracy = framework
            .compute_prediction_accuracy(&predictions, &true_values.view())
            .expect("operation should succeed");
        assert!(accuracy > 0.8);
        assert!(accuracy <= 1.0);
    }

    #[test]
    fn test_noise_addition() {
        let framework = ValidationFramework::new();
        let original_data = array![[1.0, 2.0], [3.0, 4.0]];
        let mut rng = seeded_rng(7);
        let noisy_data = framework.add_noise_to_data(&original_data, 0.1, &mut rng);

        assert_eq!(noisy_data.shape(), original_data.shape());
        // Data should be different due to noise
        assert_ne!(noisy_data, original_data);
    }

    #[test]
    fn test_case_study_returns_honest_error() {
        // Case studies carry no data, so evaluation must honestly fail rather
        // than fabricate per-criterion scores.
        let framework = ValidationFramework::new().add_case_studies();
        let case_study = &framework.case_studies[0];

        let result = framework.run_case_study(case_study);
        assert!(
            matches!(result, Err(ValidationError::InsufficientData(_))),
            "expected InsufficientData error, got {result:?}"
        );
    }

    #[test]
    fn test_robustness_analysis() {
        let framework = ValidationFramework::new();
        let x_data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0]
        ];
        let y_data = array![[2.0, 3.0], [5.0, 6.0], [8.0, 9.0], [11.0, 12.0]];

        let dataset = BenchmarkDataset {
            name: "Test".to_string(),
            x_data,
            y_data,
            true_correlations: None,
            true_x_components: None,
            true_y_components: None,
            characteristics: DatasetCharacteristics {
                n_samples: 4,
                n_x_features: 3,
                n_y_features: 2,
                signal_to_noise: 5.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance: HashMap::new(),
        };

        let result = framework
            .analyze_robustness(&dataset)
            .expect("operation should succeed");
        assert!(!result.noise_robustness.is_empty());
        assert!(!result.missing_data_robustness.is_empty());
        assert!(!result.outlier_robustness.is_empty());
        // All robustness scores are retained-fraction ratios in [0, 1].
        for (_k, &v) in result
            .noise_robustness
            .iter()
            .chain(result.missing_data_robustness.iter())
            .chain(result.outlier_robustness.iter())
        {
            assert!((0.0..=1.0).contains(&v), "robustness out of range: {v}");
        }
    }

    /// Build a strongly-correlated synthetic dataset with known ground-truth
    /// loading subspaces for component-recovery testing.
    fn correlated_dataset_with_truth() -> BenchmarkDataset {
        let framework = ValidationFramework::new();
        let true_correlations = array![0.95, 0.9, 0.85];
        let (x_data, y_data) =
            framework.generate_synthetic_cca_data(80, 8, 6, &true_correlations, 0.05);
        // Use the leading identity directions as a stand-in ground-truth
        // subspace; the test only checks that real principal angles are computed
        // (finite, in [0, pi/2]) and that recovery is a real number in [0, 1].
        let mut true_x = Array2::zeros((8, 3));
        let mut true_y = Array2::zeros((6, 3));
        for i in 0..3 {
            true_x[[i, i]] = 1.0;
            true_y[[i, i]] = 1.0;
        }
        BenchmarkDataset {
            name: "Correlated".to_string(),
            x_data,
            y_data,
            true_correlations: Some(true_correlations),
            true_x_components: Some(true_x),
            true_y_components: Some(true_y),
            characteristics: DatasetCharacteristics {
                n_samples: 200,
                n_x_features: 12,
                n_y_features: 10,
                signal_to_noise: 20.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance: HashMap::new(),
        }
    }

    #[test]
    fn test_component_recovery_real_principal_angles() {
        let framework = ValidationFramework::new();
        let dataset = correlated_dataset_with_truth();
        let cca_result = framework
            .test_cca_on_dataset(&dataset)
            .expect("cca should fit");
        let analysis = framework
            .analyze_component_recovery(&dataset, &cca_result)
            .expect("component recovery should compute");

        assert!(!analysis.principal_angles.is_empty());
        let half_pi = std::f64::consts::FRAC_PI_2 as Float;
        for &a in analysis.principal_angles.iter() {
            assert!(
                a.is_finite() && (-1e-9..=half_pi + 1e-9).contains(&a),
                "angle {a}"
            );
        }
        for &c in analysis.component_correlations.iter() {
            assert!((0.0..=1.0).contains(&c), "component corr {c}");
        }
        assert!((0.0..=1.0).contains(&analysis.subspace_recovery));
    }

    #[test]
    fn test_component_recovery_identical_subspace_is_perfect() {
        // When the "estimated" weights equal the true components, the principal
        // angles must be zero and recovery must be 1.0 (proves it is computed,
        // not hardcoded to 0.85).
        let framework = ValidationFramework::new();
        let mut weights = Array2::zeros((6, 2));
        weights[[0, 0]] = 1.0;
        weights[[1, 1]] = 1.0;
        let dataset = BenchmarkDataset {
            name: "Identity".to_string(),
            x_data: Array2::zeros((4, 6)),
            y_data: Array2::zeros((4, 4)),
            true_correlations: None,
            true_x_components: Some(weights.clone()),
            true_y_components: None,
            characteristics: DatasetCharacteristics {
                n_samples: 4,
                n_x_features: 6,
                n_y_features: 4,
                signal_to_noise: 1.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance: HashMap::new(),
        };
        let cca_result = CCATestResult {
            correlation_accuracy: 0.0,
            x_weights: weights,
            y_weights: Array2::zeros((4, 2)),
            computation_time: Duration::from_millis(0),
        };
        let analysis = framework
            .analyze_component_recovery(&dataset, &cca_result)
            .expect("recovery");
        for &a in analysis.principal_angles.iter() {
            assert!(a.abs() < 1e-6, "expected zero angle, got {a}");
        }
        assert!((analysis.subspace_recovery - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_stability_metrics_identical_clusterings() {
        let framework = ValidationFramework::new();
        let clustering = vec![0usize, 0, 1, 1, 2, 2];
        let scores = array![
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [10.0, 0.0],
            [10.1, 0.0],
        ];
        let metrics = framework
            .compute_stability_metrics(
                &[clustering.clone(), clustering.clone(), clustering],
                Some(&scores),
            )
            .expect("stability");
        assert!((metrics.jaccard_index - 1.0).abs() < 1e-9);
        assert!((metrics.rand_index - 1.0).abs() < 1e-9);
        // Well-separated reference clusters -> high silhouette.
        assert!(metrics.silhouette_coefficient > 0.5);
    }

    #[test]
    fn test_stability_metrics_disagreeing_clusterings() {
        let framework = ValidationFramework::new();
        let a = vec![0usize, 0, 1, 1];
        let b = vec![0usize, 1, 0, 1];
        let metrics = framework
            .compute_stability_metrics(&[a, b], None)
            .expect("stability");
        assert!(metrics.jaccard_index < 1.0);
        assert!(metrics.rand_index < 1.0);
    }

    #[test]
    fn test_permutation_test_detects_real_signal() {
        let framework = ValidationFramework::new().n_resamples(200);
        let dataset = correlated_dataset_with_truth();
        let result = framework
            .run_permutation_test(std::slice::from_ref(&dataset))
            .expect("permutation test");
        // Strong real association -> small p-value and large positive effect.
        assert!(result.p_value < 0.05, "p_value {}", result.p_value);
        assert!(
            result.effect_size > 1.0,
            "effect_size {}",
            result.effect_size
        );
        assert!(result.test_statistic > 0.0);
    }

    #[test]
    fn test_permutation_test_null_has_large_pvalue() {
        // Independent X and Y: the observed statistic is just another draw from
        // the permutation null, so the test must NOT be significant at 5%. We
        // build genuinely independent data with two *different* seeds and keep
        // many samples relative to the feature count so finite-sample spurious
        // correlation stays small and the observed statistic sits inside the
        // null. (Using the framework's own generator with one seed would make X
        // and Y share latent structure.)
        let framework = ValidationFramework::new().n_resamples(200);
        let n = 400usize;
        let p = 2usize;
        let normal = RandNormal::new(0.0, 1.0).expect("normal");
        let mut rng_x = seeded_rng(11);
        let mut rng_y = seeded_rng(98765);
        let mut x_data = Array2::<Float>::zeros((n, p));
        let mut y_data = Array2::<Float>::zeros((n, p));
        for v in x_data.iter_mut() {
            *v = rng_x.sample(normal);
        }
        for v in y_data.iter_mut() {
            *v = rng_y.sample(normal);
        }
        let dataset = BenchmarkDataset {
            name: "Independent".to_string(),
            x_data,
            y_data,
            true_correlations: None,
            true_x_components: None,
            true_y_components: None,
            characteristics: DatasetCharacteristics {
                n_samples: n,
                n_x_features: p,
                n_y_features: p,
                signal_to_noise: 1.0,
                distribution_type: DistributionType::Gaussian,
                correlation_structure: CorrelationStructure::Linear,
                missing_data_percent: 0.0,
            },
            expected_performance: HashMap::new(),
        };
        let result = framework
            .run_permutation_test(std::slice::from_ref(&dataset))
            .expect("permutation test");
        // No real association -> the observed statistic is unremarkable, so the
        // test is not significant at the 5% level (in stark contrast to the
        // strong-signal case which yields p < 0.05).
        assert!(result.p_value > 0.05, "p_value {}", result.p_value);
    }

    #[test]
    fn test_bootstrap_and_cv_and_comparative_tests_run() {
        let framework = ValidationFramework::new().n_resamples(100);
        let dataset = correlated_dataset_with_truth();
        let datasets = std::slice::from_ref(&dataset);

        let boot = framework.run_bootstrap_test(datasets).expect("bootstrap");
        assert!(boot.test_statistic > 0.0);
        assert!(boot.confidence_interval.0 <= boot.confidence_interval.1);
        assert!((0.0..=1.0).contains(&boot.p_value));

        let cv = framework
            .run_cv_significance_test(datasets)
            .expect("cv significance");
        assert!((0.0..=1.0).contains(&cv.p_value));

        let comp = framework
            .run_comparative_test(datasets)
            .expect("comparative");
        assert!((0.0..=1.0).contains(&comp.p_value));
        assert!(comp.test_statistic.is_finite());
    }

    #[test]
    fn test_significance_tests_error_without_data() {
        let framework = ValidationFramework::new();
        let empty: Vec<BenchmarkDataset> = Vec::new();
        assert!(matches!(
            framework.run_permutation_test(&empty),
            Err(ValidationError::InsufficientData(_))
        ));
    }

    #[test]
    fn test_computational_benchmarks_are_measured() {
        let framework = ValidationFramework::new();
        let bench = framework
            .run_computational_benchmarks()
            .expect("benchmarks");
        // Real timings recorded for both algorithms and a non-empty sweep.
        assert!(bench.execution_times.contains_key("CCA"));
        assert!(bench.execution_times.contains_key("PLS"));
        assert_eq!(bench.scalability_analysis.time_vs_samples.len(), 4);
        assert_eq!(bench.scalability_analysis.time_vs_features.len(), 4);
        // Memory accounting is a positive byte count derived from real shapes.
        assert!(*bench.memory_usage.get("CCA").unwrap_or(&0) > 0);
        // The empirical scaling exponents are present (and finite).
        for (_k, &v) in bench.scalability_analysis.memory_complexity.iter() {
            assert!(v.is_finite());
        }
    }
}

//! Benchmarks against scikit-learn Naive Bayes implementations
//!
//! This module provides benchmarking utilities to compare performance
//! and accuracy against scikit-learn's Naive Bayes classifiers.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::Rng;
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict, PredictProba},
};
use std::time::{Duration, Instant};

use crate::{GaussianNB, MultinomialNB};

/// Type alias for train/test split data: (X_train, y_train, X_test, y_test)
type TrainTestSplit = (Array2<f64>, Array1<i32>, Array2<f64>, Array1<i32>);

/// Benchmark results comparing sklears and scikit-learn
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Dataset name
    pub dataset_name: String,
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Number of classes
    pub n_classes: usize,
    /// sklears fitting time
    pub sklears_fit_time: Duration,
    /// sklears prediction time
    pub sklears_predict_time: Duration,
    /// sklears accuracy
    pub sklears_accuracy: f64,
    /// Expected scikit-learn accuracy (for comparison)
    pub sklearn_accuracy_reference: Option<f64>,
    /// Performance improvement factor
    pub performance_factor: Option<f64>,
    /// Additional metrics
    pub additional_metrics: std::collections::HashMap<String, f64>,
}

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of runs for timing averages
    pub n_runs: usize,
    /// Whether to include memory usage profiling
    pub profile_memory: bool,
    /// Whether to warm up before timing
    pub warmup: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            n_runs: 5,
            profile_memory: false,
            warmup: true,
            random_seed: Some(42),
        }
    }
}

/// Comprehensive benchmark suite
pub struct NaiveBayesBenchmark {
    config: BenchmarkConfig,
}

impl Default for NaiveBayesBenchmark {
    fn default() -> Self {
        Self::new()
    }
}

impl NaiveBayesBenchmark {
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    pub fn with_config(mut self, config: BenchmarkConfig) -> Self {
        self.config = config;
        self
    }

    /// Benchmark Gaussian Naive Bayes
    pub fn benchmark_gaussian_nb(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<i32>,
        x_test: &Array2<f64>,
        y_test: &Array1<i32>,
        dataset_name: &str,
    ) -> Result<BenchmarkResults> {
        let n_samples = x_train.nrows();
        let n_features = x_train.ncols();
        let n_classes = {
            let mut classes = y_train.to_vec();
            classes.sort();
            classes.dedup();
            classes.len()
        };

        // Warmup if requested
        if self.config.warmup {
            let _model = GaussianNB::new().fit(x_train, y_train)?;
        }

        // Benchmark fitting
        let mut fit_times = Vec::new();
        for _ in 0..self.config.n_runs {
            let start = Instant::now();
            let _model = GaussianNB::new().fit(x_train, y_train)?;
            fit_times.push(start.elapsed());
        }
        let avg_fit_time = fit_times.iter().sum::<Duration>() / fit_times.len() as u32;

        // Train model for prediction benchmarking
        let model = GaussianNB::new().fit(x_train, y_train)?;

        // Benchmark prediction
        let mut predict_times = Vec::new();
        for _ in 0..self.config.n_runs {
            let start = Instant::now();
            let _predictions = model.predict(x_test)?;
            predict_times.push(start.elapsed());
        }
        let avg_predict_time = predict_times.iter().sum::<Duration>() / predict_times.len() as u32;

        // Calculate accuracy
        let predictions = model.predict(x_test)?;
        let accuracy = calculate_accuracy(&predictions, y_test);

        // Additional metrics
        let mut additional_metrics = std::collections::HashMap::new();

        // Log likelihood
        let probabilities = model.predict_proba(x_test)?;
        let log_likelihood = calculate_log_likelihood(&probabilities, y_test);
        additional_metrics.insert("log_likelihood".to_string(), log_likelihood);

        // Brier score
        let brier_score = calculate_brier_score(&probabilities, y_test);
        additional_metrics.insert("brier_score".to_string(), brier_score);

        Ok(BenchmarkResults {
            dataset_name: dataset_name.to_string(),
            n_samples,
            n_features,
            n_classes,
            sklears_fit_time: avg_fit_time,
            sklears_predict_time: avg_predict_time,
            sklears_accuracy: accuracy,
            sklearn_accuracy_reference: None,
            performance_factor: None,
            additional_metrics,
        })
    }

    /// Benchmark Multinomial Naive Bayes
    pub fn benchmark_multinomial_nb(
        &self,
        x_train: &Array2<f64>,
        y_train: &Array1<i32>,
        x_test: &Array2<f64>,
        y_test: &Array1<i32>,
        dataset_name: &str,
    ) -> Result<BenchmarkResults> {
        let n_samples = x_train.nrows();
        let n_features = x_train.ncols();
        let n_classes = {
            let mut classes = y_train.to_vec();
            classes.sort();
            classes.dedup();
            classes.len()
        };

        // Warmup
        if self.config.warmup {
            let _model = MultinomialNB::new().fit(x_train, y_train)?;
        }

        // Benchmark fitting
        let mut fit_times = Vec::new();
        for _ in 0..self.config.n_runs {
            let start = Instant::now();
            let _model = MultinomialNB::new().fit(x_train, y_train)?;
            fit_times.push(start.elapsed());
        }
        let avg_fit_time = fit_times.iter().sum::<Duration>() / fit_times.len() as u32;

        // Train model for prediction benchmarking
        let model = MultinomialNB::new().fit(x_train, y_train)?;

        // Benchmark prediction
        let mut predict_times = Vec::new();
        for _ in 0..self.config.n_runs {
            let start = Instant::now();
            let _predictions = model.predict(x_test)?;
            predict_times.push(start.elapsed());
        }
        let avg_predict_time = predict_times.iter().sum::<Duration>() / predict_times.len() as u32;

        // Calculate accuracy
        let predictions = model.predict(x_test)?;
        let accuracy = calculate_accuracy(&predictions, y_test);

        // Additional metrics
        let mut additional_metrics = std::collections::HashMap::new();
        let probabilities = model.predict_proba(x_test)?;
        let log_likelihood = calculate_log_likelihood(&probabilities, y_test);
        additional_metrics.insert("log_likelihood".to_string(), log_likelihood);

        let brier_score = calculate_brier_score(&probabilities, y_test);
        additional_metrics.insert("brier_score".to_string(), brier_score);

        Ok(BenchmarkResults {
            dataset_name: dataset_name.to_string(),
            n_samples,
            n_features,
            n_classes,
            sklears_fit_time: avg_fit_time,
            sklears_predict_time: avg_predict_time,
            sklears_accuracy: accuracy,
            sklearn_accuracy_reference: None,
            performance_factor: None,
            additional_metrics,
        })
    }

    /// Comprehensive benchmark across multiple datasets
    pub fn run_comprehensive_benchmark(&self) -> Result<Vec<BenchmarkResults>> {
        let mut results = Vec::new();

        // Test dataset 1: Synthetic Gaussian data
        let (x_train, y_train, x_test, y_test) = generate_gaussian_dataset(1000, 20, 3, 42)?;
        let gaussian_results =
            self.benchmark_gaussian_nb(&x_train, &y_train, &x_test, &y_test, "synthetic_gaussian")?;
        results.push(gaussian_results);

        // Test dataset 2: Synthetic multinomial data
        let (x_train, y_train, x_test, y_test) = generate_multinomial_dataset(1000, 50, 5, 42)?;
        let multinomial_results = self.benchmark_multinomial_nb(
            &x_train,
            &y_train,
            &x_test,
            &y_test,
            "synthetic_multinomial",
        )?;
        results.push(multinomial_results);

        // Test dataset 3: Large sparse dataset
        let (x_train, y_train, x_test, y_test) = generate_sparse_dataset(5000, 1000, 10, 42)?;
        let sparse_results = self.benchmark_multinomial_nb(
            &x_train,
            &y_train,
            &x_test,
            &y_test,
            "synthetic_sparse",
        )?;
        results.push(sparse_results);

        Ok(results)
    }

    /// Print benchmark results in a formatted table
    pub fn print_results(&self, results: &[BenchmarkResults]) {
        println!("\n{:=<80}", "");
        println!("{:^80}", "Naive Bayes Benchmark Results");
        println!("{:=<80}", "");

        for result in results {
            println!("\nDataset: {}", result.dataset_name);
            println!(
                "  Samples: {}, Features: {}, Classes: {}",
                result.n_samples, result.n_features, result.n_classes
            );
            println!("  Fit time: {:?}", result.sklears_fit_time);
            println!("  Predict time: {:?}", result.sklears_predict_time);
            println!("  Accuracy: {:.4}", result.sklears_accuracy);

            if let Some(ref_acc) = result.sklearn_accuracy_reference {
                println!("  sklearn accuracy (reference): {:.4}", ref_acc);
                println!(
                    "  Accuracy difference: {:.4}",
                    result.sklears_accuracy - ref_acc
                );
            }

            if let Some(perf_factor) = result.performance_factor {
                println!("  Performance factor: {:.2}x", perf_factor);
            }

            println!("  Additional metrics:");
            for (metric, value) in &result.additional_metrics {
                println!("    {}: {:.4}", metric, value);
            }
            println!("{:-<50}", "");
        }
    }

    /// Compare with reference sklearn results
    pub fn compare_with_sklearn(
        &self,
        results: &mut [BenchmarkResults],
        sklearn_results: &[(String, f64, Duration)], // (dataset_name, accuracy, fit_time)
    ) {
        for result in results.iter_mut() {
            if let Some((_, sklearn_acc, sklearn_time)) = sklearn_results
                .iter()
                .find(|(name, _, _)| name == &result.dataset_name)
            {
                result.sklearn_accuracy_reference = Some(*sklearn_acc);
                result.performance_factor =
                    Some(sklearn_time.as_secs_f64() / result.sklears_fit_time.as_secs_f64());
            }
        }
    }
}

/// Calculate classification accuracy
fn calculate_accuracy(predictions: &Array1<i32>, y_true: &Array1<i32>) -> f64 {
    let correct = predictions
        .iter()
        .zip(y_true.iter())
        .filter(|(&pred, &true_val)| pred == true_val)
        .count();
    correct as f64 / y_true.len() as f64
}

/// Calculate log likelihood
fn calculate_log_likelihood(probabilities: &Array2<f64>, y_true: &Array1<i32>) -> f64 {
    let mut log_likelihood = 0.0;
    for (i, &true_class) in y_true.iter().enumerate() {
        let class_idx = true_class as usize;
        if class_idx < probabilities.ncols() {
            log_likelihood += probabilities[[i, class_idx]].max(1e-15).ln();
        }
    }
    log_likelihood / y_true.len() as f64
}

/// Calculate Brier score
fn calculate_brier_score(probabilities: &Array2<f64>, y_true: &Array1<i32>) -> f64 {
    let n_classes = probabilities.ncols();
    let mut brier_score = 0.0;

    for (i, &true_class) in y_true.iter().enumerate() {
        for class_idx in 0..n_classes {
            let target = if class_idx == true_class as usize {
                1.0
            } else {
                0.0
            };
            let prob = probabilities[[i, class_idx]];
            brier_score += (prob - target).powi(2);
        }
    }

    brier_score / (y_true.len() * n_classes) as f64
}

/// Generate synthetic Gaussian dataset
fn generate_gaussian_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> Result<TrainTestSplit> {
    // SciRS2 Policy Compliance - Use scirs2-core for random functionality

    use scirs2_core::random::SeedableRng;
    // SciRS2 Policy Compliance - Use scirs2-core for random distributions
    use scirs2_core::random::essentials::Normal as RandNormal;
    use scirs2_core::random::Distribution;

    let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(seed);

    let samples_per_class = n_samples / n_classes;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for class_idx in 0..n_classes {
        let class_mean = (class_idx as f64) * 3.0;
        let normal = RandNormal::new(class_mean, 1.0).unwrap();

        for _ in 0..samples_per_class {
            let mut sample = Vec::new();
            for _ in 0..n_features {
                sample.push(normal.sample(&mut rng));
            }
            x_data.extend(sample);
            y_data.push(class_idx as i32);
        }
    }

    let total_samples = samples_per_class * n_classes;
    let x_array = Array2::from_shape_vec((total_samples, n_features), x_data)
        .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
    let y_array = Array1::from_vec(y_data);

    // Split into train/test (80/20)
    let split_idx = (total_samples as f64 * 0.8) as usize;
    let x_train = x_array
        .slice(scirs2_core::ndarray::s![..split_idx, ..])
        .to_owned();
    let y_train = y_array
        .slice(scirs2_core::ndarray::s![..split_idx])
        .to_owned();
    let x_test = x_array
        .slice(scirs2_core::ndarray::s![split_idx.., ..])
        .to_owned();
    let y_test = y_array
        .slice(scirs2_core::ndarray::s![split_idx..])
        .to_owned();

    Ok((x_train, y_train, x_test, y_test))
}

/// Generate synthetic multinomial dataset
fn generate_multinomial_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> Result<TrainTestSplit> {
    // SciRS2 Policy Compliance - Use scirs2-core for random functionality

    use scirs2_core::random::SeedableRng;
    // SciRS2 Policy Compliance - Use scirs2-core for random distributions
    use scirs2_core::random::Distribution;
    use scirs2_core::Poisson;

    let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(seed);

    let samples_per_class = n_samples / n_classes;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for class_idx in 0..n_classes {
        let class_rate = 2.0 + (class_idx as f64) * 1.5;
        let poisson = Poisson::new(class_rate).unwrap();

        for _ in 0..samples_per_class {
            let mut sample = Vec::new();
            for _ in 0..n_features {
                sample.push(poisson.sample(&mut rng));
            }
            x_data.extend(sample);
            y_data.push(class_idx as i32);
        }
    }

    let total_samples = samples_per_class * n_classes;
    let x_array = Array2::from_shape_vec((total_samples, n_features), x_data)
        .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
    let y_array = Array1::from_vec(y_data);

    // Split into train/test (80/20)
    let split_idx = (total_samples as f64 * 0.8) as usize;
    let x_train = x_array
        .slice(scirs2_core::ndarray::s![..split_idx, ..])
        .to_owned();
    let y_train = y_array
        .slice(scirs2_core::ndarray::s![..split_idx])
        .to_owned();
    let x_test = x_array
        .slice(scirs2_core::ndarray::s![split_idx.., ..])
        .to_owned();
    let y_test = y_array
        .slice(scirs2_core::ndarray::s![split_idx..])
        .to_owned();

    Ok((x_train, y_train, x_test, y_test))
}

/// Generate synthetic sparse dataset
fn generate_sparse_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
    seed: u64,
) -> Result<TrainTestSplit> {
    // SciRS2 Policy Compliance - Use scirs2-core for random functionality

    use scirs2_core::random::SeedableRng;

    let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(seed);

    let samples_per_class = n_samples / n_classes;
    let mut x_data = Vec::new();
    let mut y_data = Vec::new();

    for class_idx in 0..n_classes {
        for _ in 0..samples_per_class {
            let mut sample = vec![0.0; n_features];

            // Only make 10% of features non-zero
            let n_nonzero = n_features / 10;
            for _ in 0..n_nonzero {
                let feature_idx = rng.gen_range(0..n_features);
                sample[feature_idx] = rng.gen_range(1.0..10.0);
            }

            x_data.extend(sample);
            y_data.push(class_idx as i32);
        }
    }

    let total_samples = samples_per_class * n_classes;
    let x_array = Array2::from_shape_vec((total_samples, n_features), x_data)
        .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
    let y_array = Array1::from_vec(y_data);

    // Split into train/test (80/20)
    let split_idx = (total_samples as f64 * 0.8) as usize;
    let x_train = x_array
        .slice(scirs2_core::ndarray::s![..split_idx, ..])
        .to_owned();
    let y_train = y_array
        .slice(scirs2_core::ndarray::s![..split_idx])
        .to_owned();
    let x_test = x_array
        .slice(scirs2_core::ndarray::s![split_idx.., ..])
        .to_owned();
    let y_test = y_array
        .slice(scirs2_core::ndarray::s![split_idx..])
        .to_owned();

    Ok((x_train, y_train, x_test, y_test))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_gaussian_nb() {
        let benchmark = NaiveBayesBenchmark::new();

        // Generate small test dataset
        let (x_train, y_train, x_test, y_test) = generate_gaussian_dataset(100, 5, 2, 42).unwrap();

        let result = benchmark
            .benchmark_gaussian_nb(&x_train, &y_train, &x_test, &y_test, "test_gaussian")
            .unwrap();

        assert_eq!(result.dataset_name, "test_gaussian");
        assert!(result.sklears_accuracy >= 0.0 && result.sklears_accuracy <= 1.0);
        assert!(result.n_samples > 0);
        assert!(result.n_features > 0);
        assert!(result.n_classes > 0);
    }

    #[test]
    fn test_benchmark_multinomial_nb() {
        let benchmark = NaiveBayesBenchmark::new();

        // Generate small test dataset
        let (x_train, y_train, x_test, y_test) =
            generate_multinomial_dataset(100, 5, 2, 42).unwrap();

        let result = benchmark
            .benchmark_multinomial_nb(&x_train, &y_train, &x_test, &y_test, "test_multinomial")
            .unwrap();

        assert_eq!(result.dataset_name, "test_multinomial");
        assert!(result.sklears_accuracy >= 0.0 && result.sklears_accuracy <= 1.0);
    }

    #[test]
    fn test_accuracy_calculation() {
        let predictions = Array1::from_vec(vec![0, 1, 1, 0, 1]);
        let y_true = Array1::from_vec(vec![0, 1, 0, 0, 1]);

        let accuracy = calculate_accuracy(&predictions, &y_true);
        assert_eq!(accuracy, 0.8); // 4 out of 5 correct
    }
}

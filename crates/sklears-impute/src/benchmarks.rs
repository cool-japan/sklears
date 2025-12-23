//! Benchmarking and comparison utilities for imputation methods
//!
//! This module provides tools for comparing imputation methods against reference implementations
//! and measuring performance across different scenarios and datasets.

use scirs2_core::ndarray::{Array2, ArrayView2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::{Random, Rng};
use sklears_core::{
    error::Result as SklResult,
    traits::{Fit, Transform},
    types::Float,
};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Benchmark results for imputation methods
#[derive(Debug, Clone)]
pub struct ImputationBenchmark {
    /// method_name
    pub method_name: String,
    /// dataset_name
    pub dataset_name: String,
    /// missing_rate
    pub missing_rate: f64,
    /// missing_pattern
    pub missing_pattern: String,
    /// rmse
    pub rmse: f64,
    /// mae
    pub mae: f64,
    /// execution_time
    pub execution_time: Duration,
    /// memory_usage
    pub memory_usage: Option<usize>,
    /// convergence_iterations
    pub convergence_iterations: Option<usize>,
}

/// Comparison results between methods
#[derive(Debug, Clone)]
pub struct ImputationComparison {
    /// benchmarks
    pub benchmarks: Vec<ImputationBenchmark>,
    /// best_rmse_method
    pub best_rmse_method: String,
    /// best_mae_method
    pub best_mae_method: String,
    /// fastest_method
    pub fastest_method: String,
    /// accuracy_rankings
    pub accuracy_rankings: HashMap<String, usize>,
    /// speed_rankings
    pub speed_rankings: HashMap<String, usize>,
}

/// Missing data pattern types for benchmarking
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MissingPattern {
    /// Missing Completely At Random
    MCAR { missing_rate: f64 },
    /// Missing At Random - depends on observed values
    MAR {
        missing_rate: f64,
        dependency_strength: f64,
    },
    /// Missing Not At Random - depends on unobserved values
    MNAR { missing_rate: f64, threshold: f64 },
    /// Block missing pattern
    Block {
        block_size: usize,
        missing_rate: f64,
    },
    /// Monotone missing pattern
    Monotone { missing_rate: f64 },
}

/// Dataset generator for benchmarking
pub struct BenchmarkDatasetGenerator {
    n_samples: usize,
    n_features: usize,
    noise_level: f64,
    correlation_strength: f64,
    random_state: Option<u64>,
}

impl BenchmarkDatasetGenerator {
    /// Create a new dataset generator
    pub fn new(n_samples: usize, n_features: usize) -> Self {
        Self {
            n_samples,
            n_features,
            noise_level: 0.1,
            correlation_strength: 0.5,
            random_state: None,
        }
    }

    /// Set noise level
    pub fn noise_level(mut self, noise_level: f64) -> Self {
        self.noise_level = noise_level;
        self
    }

    /// Set correlation strength between features
    pub fn correlation_strength(mut self, correlation_strength: f64) -> Self {
        self.correlation_strength = correlation_strength;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Generate a correlated multivariate dataset
    pub fn generate_correlated_data(&self) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(_seed) = self.random_state {
            Random::default()
        } else {
            Random::default()
        };

        let mut data = Array2::zeros((self.n_samples, self.n_features));

        // Generate base features
        for i in 0..self.n_samples {
            data[[i, 0]] = rng.gen_range(-3.0..3.0);
        }

        // Generate correlated features
        for j in 1..self.n_features {
            let correlation = self.correlation_strength;
            for i in 0..self.n_samples {
                let base_value = data[[i, 0]];
                let noise = rng.gen_range(-self.noise_level..self.noise_level);
                data[[i, j]] = correlation * base_value
                    + (1.0 - correlation) * rng.gen_range(-2.0..2.0)
                    + noise;
            }
        }

        Ok(data)
    }

    /// Generate linear relationship dataset
    pub fn generate_linear_data(&self) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(_seed) = self.random_state {
            Random::default()
        } else {
            Random::default()
        };

        let mut data = Array2::zeros((self.n_samples, self.n_features));

        // Generate features with linear relationships
        for i in 0..self.n_samples {
            // Base feature
            data[[i, 0]] = rng.gen_range(-5.0..5.0);

            // Linearly related features
            for j in 1..self.n_features {
                let coef = (j as f64) * 0.5;
                let noise = rng.gen_range(-self.noise_level..self.noise_level);
                data[[i, j]] = coef * data[[i, 0]] + noise;
            }
        }

        Ok(data)
    }

    /// Generate non-linear relationship dataset
    pub fn generate_nonlinear_data(&self) -> SklResult<Array2<f64>> {
        let mut rng = if let Some(_seed) = self.random_state {
            Random::default()
        } else {
            Random::default()
        };

        let mut data = Array2::zeros((self.n_samples, self.n_features));

        for i in 0..self.n_samples {
            let x: f64 = rng.gen_range(-2.0..2.0);
            data[[i, 0]] = x;

            // Non-linear relationships
            data[[i, 1]] = x.powi(2) + rng.gen_range(-self.noise_level..self.noise_level);

            if self.n_features > 2 {
                data[[i, 2]] = (x * 1.5).sin() + rng.gen_range(-self.noise_level..self.noise_level);
            }

            if self.n_features > 3 {
                data[[i, 3]] = (x.powi(2) + x).exp() / 10.0
                    + rng.gen_range(-self.noise_level..self.noise_level);
            }

            // Additional features with mixed relationships
            for j in 4..self.n_features {
                let noise = rng.gen_range(-self.noise_level..self.noise_level);
                data[[i, j]] = (x + (j as f64) * 0.2).cos() + noise;
            }
        }

        Ok(data)
    }
}

/// Missing pattern generator
pub struct MissingPatternGenerator {
    random_state: Option<u64>,
}

impl MissingPatternGenerator {
    /// Create a new missing pattern generator
    pub fn new() -> Self {
        Self { random_state: None }
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Introduce missing values according to pattern
    pub fn introduce_missing(
        &self,
        data: &Array2<f64>,
        pattern: &MissingPattern,
    ) -> SklResult<(Array2<f64>, Array2<bool>)> {
        let mut rng = if let Some(_seed) = self.random_state {
            Random::default()
        } else {
            Random::default()
        };

        let (n_samples, n_features) = data.dim();
        let mut data_with_missing = data.clone();
        let mut missing_mask = Array2::from_elem((n_samples, n_features), false);

        match pattern {
            MissingPattern::MCAR { missing_rate } => {
                self.introduce_mcar(
                    &mut data_with_missing,
                    &mut missing_mask,
                    *missing_rate,
                    &mut rng,
                )?;
            }
            MissingPattern::MAR {
                missing_rate,
                dependency_strength,
            } => {
                self.introduce_mar(
                    data,
                    &mut data_with_missing,
                    &mut missing_mask,
                    *missing_rate,
                    *dependency_strength,
                    &mut rng,
                )?;
            }
            MissingPattern::MNAR {
                missing_rate,
                threshold,
            } => {
                self.introduce_mnar(
                    data,
                    &mut data_with_missing,
                    &mut missing_mask,
                    *missing_rate,
                    *threshold,
                    &mut rng,
                )?;
            }
            MissingPattern::Block {
                block_size,
                missing_rate,
            } => {
                self.introduce_block(
                    &mut data_with_missing,
                    &mut missing_mask,
                    *block_size,
                    *missing_rate,
                    &mut rng,
                )?;
            }
            MissingPattern::Monotone { missing_rate } => {
                self.introduce_monotone(
                    &mut data_with_missing,
                    &mut missing_mask,
                    *missing_rate,
                    &mut rng,
                )?;
            }
        }

        Ok((data_with_missing, missing_mask))
    }

    fn introduce_mcar(
        &self,
        data: &mut Array2<f64>,
        missing_mask: &mut Array2<bool>,
        missing_rate: f64,
        rng: &mut Random,
    ) -> SklResult<()> {
        let total_elements = data.len();
        let n_missing = (total_elements as f64 * missing_rate) as usize;

        let mut positions: Vec<(usize, usize)> = Vec::new();
        for i in 0..data.nrows() {
            for j in 0..data.ncols() {
                positions.push((i, j));
            }
        }

        positions.shuffle(rng);

        for &(i, j) in positions.iter().take(n_missing) {
            data[[i, j]] = f64::NAN;
            missing_mask[[i, j]] = true;
        }

        Ok(())
    }

    fn introduce_mar(
        &self,
        original_data: &Array2<f64>,
        data: &mut Array2<f64>,
        missing_mask: &mut Array2<bool>,
        missing_rate: f64,
        dependency_strength: f64,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, n_features) = data.dim();

        if n_features < 2 {
            return self.introduce_mcar(data, missing_mask, missing_rate, rng);
        }

        // Make missingness in columns 1+ depend on column 0
        let column_0_median = {
            let mut sorted: Vec<f64> = original_data.column(0).iter().cloned().collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            sorted[sorted.len() / 2]
        };

        for i in 0..n_samples {
            for j in 1..n_features {
                let base_prob = missing_rate;
                let prob_adjustment = if original_data[[i, 0]] > column_0_median {
                    dependency_strength
                } else {
                    -dependency_strength
                };

                let prob_missing = (base_prob + prob_adjustment).clamp(0.0, 1.0);

                if rng.gen::<f64>() < prob_missing {
                    data[[i, j]] = f64::NAN;
                    missing_mask[[i, j]] = true;
                }
            }
        }

        Ok(())
    }

    fn introduce_mnar(
        &self,
        original_data: &Array2<f64>,
        data: &mut Array2<f64>,
        missing_mask: &mut Array2<bool>,
        missing_rate: f64,
        threshold: f64,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, n_features) = data.dim();

        for j in 0..n_features {
            let column_values: Vec<f64> = original_data.column(j).iter().cloned().collect();
            let column_threshold = {
                let mut sorted = column_values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                sorted[(sorted.len() as f64 * threshold) as usize]
            };

            for i in 0..n_samples {
                // Higher chance of missing if value is above threshold
                let base_prob = missing_rate;
                let prob_missing = if original_data[[i, j]] > column_threshold {
                    base_prob * 2.0
                } else {
                    base_prob * 0.5
                };

                if rng.gen::<f64>() < prob_missing.min(1.0) {
                    data[[i, j]] = f64::NAN;
                    missing_mask[[i, j]] = true;
                }
            }
        }

        Ok(())
    }

    fn introduce_block(
        &self,
        data: &mut Array2<f64>,
        missing_mask: &mut Array2<bool>,
        block_size: usize,
        missing_rate: f64,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, n_features) = data.dim();
        let n_blocks =
            ((n_samples * n_features) as f64 * missing_rate / block_size as f64) as usize;

        for _ in 0..n_blocks {
            let start_i = rng.gen_range(0..n_samples);
            let start_j = rng.gen_range(0..n_features);

            let block_height = (block_size as f64).sqrt() as usize;
            let block_width = block_size / block_height.max(1);

            for di in 0..block_height {
                for dj in 0..block_width {
                    let i = (start_i + di) % n_samples;
                    let j = (start_j + dj) % n_features;
                    data[[i, j]] = f64::NAN;
                    missing_mask[[i, j]] = true;
                }
            }
        }

        Ok(())
    }

    fn introduce_monotone(
        &self,
        data: &mut Array2<f64>,
        missing_mask: &mut Array2<bool>,
        missing_rate: f64,
        rng: &mut Random,
    ) -> SklResult<()> {
        let (n_samples, n_features) = data.dim();

        if n_features == 0 {
            return Ok(());
        }

        let mut samples_to_affect: Vec<usize> = (0..n_samples).collect();
        samples_to_affect.shuffle(rng);
        let n_affected = (n_samples as f64 * missing_rate) as usize;

        for &sample_idx in samples_to_affect.iter().take(n_affected) {
            // Start missing from a random feature onwards
            let start_feature = rng.gen_range(0..n_features);
            for j in start_feature..n_features {
                data[[sample_idx, j]] = f64::NAN;
                missing_mask[[sample_idx, j]] = true;
            }
        }

        Ok(())
    }
}

impl Default for MissingPatternGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Accuracy metrics calculator
pub struct AccuracyMetrics;

impl AccuracyMetrics {
    /// Calculate Root Mean Square Error
    pub fn rmse(
        true_values: &Array2<f64>,
        imputed_values: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> f64 {
        let mut sum_squared_diff = 0.0;
        let mut count = 0;

        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                let diff = true_values[[i, j]] - imputed_values[[i, j]];
                sum_squared_diff += diff * diff;
                count += 1;
            }
        }

        if count > 0 {
            (sum_squared_diff / count as f64).sqrt()
        } else {
            0.0
        }
    }

    /// Calculate Mean Absolute Error
    pub fn mae(
        true_values: &Array2<f64>,
        imputed_values: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> f64 {
        let mut sum_abs_diff = 0.0;
        let mut count = 0;

        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                let diff = (true_values[[i, j]] - imputed_values[[i, j]]).abs();
                sum_abs_diff += diff;
                count += 1;
            }
        }

        if count > 0 {
            sum_abs_diff / count as f64
        } else {
            0.0
        }
    }

    /// Calculate bias (mean error)
    pub fn bias(
        true_values: &Array2<f64>,
        imputed_values: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> f64 {
        let mut sum_diff = 0.0;
        let mut count = 0;

        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                let diff = imputed_values[[i, j]] - true_values[[i, j]];
                sum_diff += diff;
                count += 1;
            }
        }

        if count > 0 {
            sum_diff / count as f64
        } else {
            0.0
        }
    }

    /// Calculate R-squared coefficient
    pub fn r_squared(
        true_values: &Array2<f64>,
        imputed_values: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> f64 {
        let mut missing_true_values = Vec::new();
        let mut missing_imputed_values = Vec::new();

        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                missing_true_values.push(true_values[[i, j]]);
                missing_imputed_values.push(imputed_values[[i, j]]);
            }
        }

        if missing_true_values.is_empty() {
            return 1.0;
        }

        let true_mean = missing_true_values.iter().sum::<f64>() / missing_true_values.len() as f64;

        let ss_tot: f64 = missing_true_values
            .iter()
            .map(|&x| (x - true_mean).powi(2))
            .sum();

        let ss_res: f64 = missing_true_values
            .iter()
            .zip(missing_imputed_values.iter())
            .map(|(&true_val, &imputed_val)| (true_val - imputed_val).powi(2))
            .sum();

        if ss_tot == 0.0 {
            1.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

/// Performance benchmarking suite
pub struct BenchmarkSuite {
    datasets: Vec<(String, Array2<f64>)>,
    missing_patterns: Vec<(String, MissingPattern)>,
    random_state: Option<u64>,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new() -> Self {
        Self {
            datasets: Vec::new(),
            missing_patterns: Vec::new(),
            random_state: None,
        }
    }

    /// Add a dataset to the benchmark suite
    pub fn add_dataset(mut self, name: String, data: Array2<f64>) -> Self {
        self.datasets.push((name, data));
        self
    }

    /// Add a missing pattern to test
    pub fn add_missing_pattern(mut self, name: String, pattern: MissingPattern) -> Self {
        self.missing_patterns.push((name, pattern));
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Add standard benchmark datasets
    pub fn add_standard_datasets(mut self) -> Self {
        // Linear relationship dataset
        let linear_gen = BenchmarkDatasetGenerator::new(100, 4)
            .correlation_strength(0.7)
            .noise_level(0.1)
            .random_state(Some(42));

        if let Ok(linear_data) = linear_gen.generate_linear_data() {
            self.datasets
                .push(("linear_100x4".to_string(), linear_data));
        }

        // Non-linear relationship dataset
        let nonlinear_gen = BenchmarkDatasetGenerator::new(80, 3)
            .noise_level(0.2)
            .random_state(Some(123));

        if let Ok(nonlinear_data) = nonlinear_gen.generate_nonlinear_data() {
            self.datasets
                .push(("nonlinear_80x3".to_string(), nonlinear_data));
        }

        // Correlated dataset
        let correlated_gen = BenchmarkDatasetGenerator::new(120, 5)
            .correlation_strength(0.8)
            .noise_level(0.05)
            .random_state(Some(456));

        if let Ok(correlated_data) = correlated_gen.generate_correlated_data() {
            self.datasets
                .push(("correlated_120x5".to_string(), correlated_data));
        }

        self
    }

    /// Add standard missing patterns
    pub fn add_standard_patterns(mut self) -> Self {
        self.missing_patterns.push((
            "MCAR_15%".to_string(),
            MissingPattern::MCAR { missing_rate: 0.15 },
        ));

        self.missing_patterns.push((
            "MAR_20%".to_string(),
            MissingPattern::MAR {
                missing_rate: 0.20,
                dependency_strength: 0.3,
            },
        ));

        self.missing_patterns.push((
            "MNAR_10%".to_string(),
            MissingPattern::MNAR {
                missing_rate: 0.10,
                threshold: 0.7,
            },
        ));

        self.missing_patterns.push((
            "Block_12%".to_string(),
            MissingPattern::Block {
                block_size: 4,
                missing_rate: 0.12,
            },
        ));

        self
    }

    /// Run benchmark on a specific imputer
    pub fn benchmark_imputer<I, T>(
        &self,
        imputer: I,
        imputer_name: &str,
    ) -> SklResult<Vec<ImputationBenchmark>>
    where
        I: Clone,
        for<'a> I: Fit<ArrayView2<'a, Float>, (), Fitted = T>,
        for<'a> T: Transform<ArrayView2<'a, Float>, Array2<Float>>,
    {
        let mut results = Vec::new();
        let pattern_generator = MissingPatternGenerator::new().random_state(self.random_state);

        for (dataset_name, true_data) in &self.datasets {
            for (pattern_name, pattern) in &self.missing_patterns {
                let (data_with_missing, missing_mask) =
                    pattern_generator.introduce_missing(true_data, pattern)?;

                let data_float = data_with_missing.mapv(|x| x as Float);

                // Measure execution time
                let start_time = Instant::now();

                let fitted = imputer.clone().fit(&data_float.view(), &())?;
                let imputed_data = fitted.transform(&data_float.view())?;

                let execution_time = start_time.elapsed();

                // Calculate accuracy metrics
                let imputed_f64 = imputed_data.mapv(|x| x);
                let rmse = AccuracyMetrics::rmse(true_data, &imputed_f64, &missing_mask);
                let mae = AccuracyMetrics::mae(true_data, &imputed_f64, &missing_mask);

                let missing_rate =
                    missing_mask.iter().filter(|&&x| x).count() as f64 / missing_mask.len() as f64;

                results.push(ImputationBenchmark {
                    method_name: imputer_name.to_string(),
                    dataset_name: dataset_name.clone(),
                    missing_rate,
                    missing_pattern: pattern_name.clone(),
                    rmse,
                    mae,
                    execution_time,
                    memory_usage: None,
                    convergence_iterations: None,
                });
            }
        }

        Ok(results)
    }

    /// Compare multiple imputers
    pub fn compare_imputers(&self, benchmarks: Vec<ImputationBenchmark>) -> ImputationComparison {
        if benchmarks.is_empty() {
            return ImputationComparison {
                benchmarks: Vec::new(),
                best_rmse_method: String::new(),
                best_mae_method: String::new(),
                fastest_method: String::new(),
                accuracy_rankings: HashMap::new(),
                speed_rankings: HashMap::new(),
            };
        }

        // Find best performing methods
        let best_rmse = benchmarks
            .iter()
            .min_by(|a, b| a.rmse.partial_cmp(&b.rmse).unwrap());
        let best_mae = benchmarks
            .iter()
            .min_by(|a, b| a.mae.partial_cmp(&b.mae).unwrap());
        let fastest = benchmarks.iter().min_by_key(|b| b.execution_time);

        let best_rmse_method = best_rmse.map(|b| b.method_name.clone()).unwrap_or_default();
        let best_mae_method = best_mae.map(|b| b.method_name.clone()).unwrap_or_default();
        let fastest_method = fastest.map(|b| b.method_name.clone()).unwrap_or_default();

        // Calculate rankings
        let mut accuracy_rankings = HashMap::new();
        let mut speed_rankings = HashMap::new();

        // Group benchmarks by method
        let mut method_avg_rmse: HashMap<String, f64> = HashMap::new();
        let mut method_avg_time: HashMap<String, Duration> = HashMap::new();
        let mut method_counts: HashMap<String, usize> = HashMap::new();

        for benchmark in &benchmarks {
            let count = method_counts
                .entry(benchmark.method_name.clone())
                .or_insert(0);
            *count += 1;

            let avg_rmse = method_avg_rmse
                .entry(benchmark.method_name.clone())
                .or_insert(0.0);
            *avg_rmse += benchmark.rmse;

            let avg_time = method_avg_time
                .entry(benchmark.method_name.clone())
                .or_insert(Duration::ZERO);
            *avg_time += benchmark.execution_time;
        }

        // Calculate averages
        for (method, count) in &method_counts {
            if let Some(total_rmse) = method_avg_rmse.get_mut(method) {
                *total_rmse /= *count as f64;
            }
            if let Some(total_time) = method_avg_time.get_mut(method) {
                *total_time /= *count as u32;
            }
        }

        // Create sorted rankings
        let mut rmse_pairs: Vec<_> = method_avg_rmse.into_iter().collect();
        rmse_pairs.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        for (rank, (method, _)) in rmse_pairs.into_iter().enumerate() {
            accuracy_rankings.insert(method, rank + 1);
        }

        let mut time_pairs: Vec<_> = method_avg_time.into_iter().collect();
        time_pairs.sort_by_key(|a| a.1);
        for (rank, (method, _)) in time_pairs.into_iter().enumerate() {
            speed_rankings.insert(method, rank + 1);
        }

        ImputationComparison {
            benchmarks,
            best_rmse_method,
            best_mae_method,
            fastest_method,
            accuracy_rankings,
            speed_rankings,
        }
    }

    /// Generate a comprehensive benchmark report
    pub fn generate_report(&self, comparison: &ImputationComparison) -> String {
        let mut report = String::new();

        report.push_str("# Imputation Methods Benchmark Report\n\n");

        report.push_str("## Summary\n");
        report.push_str(&format!("- Best RMSE: {}\n", comparison.best_rmse_method));
        report.push_str(&format!("- Best MAE: {}\n", comparison.best_mae_method));
        report.push_str(&format!("- Fastest: {}\n\n", comparison.fastest_method));

        report.push_str("## Accuracy Rankings\n");
        let mut accuracy_pairs: Vec<_> = comparison.accuracy_rankings.iter().collect();
        accuracy_pairs.sort_by_key(|&(_, rank)| rank);
        for (method, rank) in accuracy_pairs {
            report.push_str(&format!("{}. {}\n", rank, method));
        }

        report.push_str("\n## Speed Rankings\n");
        let mut speed_pairs: Vec<_> = comparison.speed_rankings.iter().collect();
        speed_pairs.sort_by_key(|&(_, rank)| rank);
        for (method, rank) in speed_pairs {
            report.push_str(&format!("{}. {}\n", rank, method));
        }

        report.push_str("\n## Detailed Results\n");
        for benchmark in &comparison.benchmarks {
            report.push_str(&format!(
                "- {}: {} on {} ({}): RMSE={:.4}, MAE={:.4}, Time={:.2}ms\n",
                benchmark.method_name,
                benchmark.missing_pattern,
                benchmark.dataset_name,
                (benchmark.missing_rate * 100.0).round(),
                benchmark.rmse,
                benchmark.mae,
                benchmark.execution_time.as_secs_f64() * 1000.0
            ));
        }

        report
    }
}

impl Default for BenchmarkSuite {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{KNNImputer, SimpleImputer};

    #[test]
    fn test_dataset_generation() {
        let generator = BenchmarkDatasetGenerator::new(50, 3).random_state(Some(42));

        let linear_data = generator.generate_linear_data().unwrap();
        assert_eq!(linear_data.shape(), &[50, 3]);

        let nonlinear_data = generator.generate_nonlinear_data().unwrap();
        assert_eq!(nonlinear_data.shape(), &[50, 3]);

        let correlated_data = generator.generate_correlated_data().unwrap();
        assert_eq!(correlated_data.shape(), &[50, 3]);
    }

    #[test]
    fn test_missing_pattern_generation() {
        let data = Array2::from_shape_fn((20, 3), |(i, j)| (i + j) as f64);
        let generator = MissingPatternGenerator::new().random_state(Some(123));

        // Test MCAR pattern
        let mcar_pattern = MissingPattern::MCAR { missing_rate: 0.2 };
        let (_data_mcar, mask_mcar) = generator.introduce_missing(&data, &mcar_pattern).unwrap();
        let missing_count = mask_mcar.iter().filter(|&&x| x).count();
        assert!(missing_count > 0);
        assert!(missing_count < data.len());

        // Test MAR pattern
        let mar_pattern = MissingPattern::MAR {
            missing_rate: 0.15,
            dependency_strength: 0.3,
        };
        let (_data_mar, mask_mar) = generator.introduce_missing(&data, &mar_pattern).unwrap();
        let mar_missing_count = mask_mar.iter().filter(|&&x| x).count();
        assert!(mar_missing_count > 0);

        // Test Block pattern
        let block_pattern = MissingPattern::Block {
            block_size: 4,
            missing_rate: 0.1,
        };
        let (_data_block, mask_block) = generator.introduce_missing(&data, &block_pattern).unwrap();
        let block_missing_count = mask_block.iter().filter(|&&x| x).count();
        assert!(block_missing_count > 0);
    }

    #[test]
    fn test_accuracy_metrics() {
        let true_data = Array2::from_shape_fn((10, 2), |(i, j)| (i + j) as f64);
        let mut imputed_data = true_data.clone();
        imputed_data[[0, 0]] = 10.0; // Introduce error
        imputed_data[[1, 1]] = 20.0; // Introduce error

        let mut missing_mask = Array2::from_elem((10, 2), false);
        missing_mask[[0, 0]] = true;
        missing_mask[[1, 1]] = true;

        let rmse = AccuracyMetrics::rmse(&true_data, &imputed_data, &missing_mask);
        let mae = AccuracyMetrics::mae(&true_data, &imputed_data, &missing_mask);
        let bias = AccuracyMetrics::bias(&true_data, &imputed_data, &missing_mask);

        assert!(rmse > 0.0);
        assert!(mae > 0.0);
        assert!(bias > 0.0); // Positive bias since imputed values are higher
    }

    #[test]
    fn test_benchmark_suite() {
        let data = Array2::from_shape_fn((30, 3), |(i, j)| (i + j) as f64);

        let suite = BenchmarkSuite::new()
            .add_dataset("test_data".to_string(), data)
            .add_missing_pattern(
                "test_mcar".to_string(),
                MissingPattern::MCAR { missing_rate: 0.1 },
            )
            .random_state(Some(42));

        // Test simple imputer
        let simple_imputer = SimpleImputer::new().strategy("mean".to_string());
        let simple_results = suite
            .benchmark_imputer(simple_imputer, "SimpleImputer")
            .unwrap();

        assert_eq!(simple_results.len(), 1);
        assert_eq!(simple_results[0].method_name, "SimpleImputer");
        assert!(simple_results[0].rmse >= 0.0);
        assert!(simple_results[0].mae >= 0.0);

        // Test KNN imputer
        let knn_imputer = KNNImputer::new().n_neighbors(3);
        let knn_results = suite.benchmark_imputer(knn_imputer, "KNNImputer").unwrap();

        assert_eq!(knn_results.len(), 1);
        assert_eq!(knn_results[0].method_name, "KNNImputer");

        // Test comparison
        let all_results = [simple_results, knn_results].concat();
        let comparison = suite.compare_imputers(all_results);

        assert!(comparison.accuracy_rankings.contains_key("SimpleImputer"));
        assert!(comparison.accuracy_rankings.contains_key("KNNImputer"));
        assert!(comparison.speed_rankings.contains_key("SimpleImputer"));
        assert!(comparison.speed_rankings.contains_key("KNNImputer"));
    }

    #[test]
    fn test_standard_benchmarks() {
        let suite = BenchmarkSuite::new()
            .add_standard_datasets()
            .add_standard_patterns()
            .random_state(Some(42));

        assert!(!suite.datasets.is_empty());
        assert!(!suite.missing_patterns.is_empty());

        // Test with a simple imputer
        let simple_imputer = SimpleImputer::new().strategy("mean".to_string());
        let results = suite
            .benchmark_imputer(simple_imputer, "SimpleImputer")
            .unwrap();

        // Should have results for each dataset × pattern combination
        let expected_results = suite.datasets.len() * suite.missing_patterns.len();
        assert_eq!(results.len(), expected_results);

        // All results should have valid metrics
        for result in &results {
            assert!(result.rmse >= 0.0);
            assert!(result.mae >= 0.0);
            assert!(result.execution_time > Duration::ZERO);
        }
    }
}

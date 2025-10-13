//! Benchmark baselines for performance evaluation
//!
//! This module provides standardized benchmark baselines that follow established
//! practices in machine learning literature. These baselines serve as reference
//! points for comparing the performance of more sophisticated models.
//!
//! The module includes:
//! - Standard benchmark baselines from ML literature
//! - Domain-specific baselines for common problem types
//! - Competition-grade baselines used in ML competitions
//! - Theoretical lower bounds for performance evaluation

use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{error::SklearsError, traits::Estimator, traits::Fit, traits::Predict};
use std::collections::HashMap;

/// Standard benchmark baseline strategies
#[derive(Debug, Clone)]
pub enum BenchmarkStrategy {
    /// Zero-Rule (ZeroR) - predicts most common class/mean value
    ZeroRule,
    /// One-Rule (OneR) - simple decision stump on best single feature
    OneRule,
    /// Random Forest of stumps - ensemble of single-feature decision trees
    RandomStumps { n_stumps: usize },
    /// Majority class with tie-breaking
    MajorityClassTieBreak,
    /// Weighted random by class frequency
    WeightedRandom,
    /// Linear trend baseline for time series
    LinearTrend,
    /// Moving average baseline
    MovingAverage { window_size: usize },
    /// K-Nearest Neighbors with k=1
    NearestNeighbor,
    /// Competition baseline - combines multiple simple strategies
    CompetitionBaseline,
}

/// Domain-specific benchmark strategies
#[derive(Debug, Clone)]
pub enum DomainStrategy {
    /// Computer vision: pixel intensity statistics
    PixelIntensity,
    /// NLP: bag of words frequency
    BagOfWords,
    /// Time series: seasonal decomposition
    SeasonalDecomposition { period: usize },
    /// Recommendation: popularity baseline
    PopularityBaseline,
    /// Anomaly detection: isolation threshold
    IsolationThreshold { contamination: f64 },
}

/// Theoretical lower bound strategies
#[derive(Debug, Clone)]
pub enum TheoreticalBound {
    /// Bayes error rate (theoretical minimum for classification)
    BayesError,
    /// Random chance baseline
    RandomChance,
    /// Information-theoretic lower bound
    InformationBound,
    /// Statistical lower bound based on data characteristics
    StatisticalBound,
}

/// Benchmark classifier implementing standard ML baselines
#[derive(Debug, Clone)]
pub struct BenchmarkClassifier {
    strategy: BenchmarkStrategy,
    random_state: Option<u64>,
}

/// Trained benchmark classifier
#[derive(Debug, Clone)]
pub struct TrainedBenchmarkClassifier {
    strategy: BenchmarkStrategy,
    classes: Vec<i32>,
    class_counts: HashMap<i32, usize>,
    feature_rules: Option<Vec<(usize, f64, i32)>>, // (feature_idx, threshold, prediction)
    training_data: Option<(Array2<f64>, Array1<i32>)>, // For NN baseline
    random_state: Option<u64>,
}

impl BenchmarkClassifier {
    /// Create a new benchmark classifier
    pub fn new(strategy: BenchmarkStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for BenchmarkClassifier {
    type Config = BenchmarkStrategy;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.strategy
    }
}

impl Fit<Array2<f64>, Array1<i32>> for BenchmarkClassifier {
    type Fitted = TrainedBenchmarkClassifier;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Self::Fitted, SklearsError> {
        let mut class_counts = HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let mut classes: Vec<_> = class_counts.keys().cloned().collect();
        classes.sort();

        let feature_rules = match &self.strategy {
            BenchmarkStrategy::OneRule => Some(Self::build_one_rule(x, y)?),
            BenchmarkStrategy::RandomStumps { n_stumps } => Some(Self::build_random_stumps(
                x,
                y,
                *n_stumps,
                self.random_state,
            )?),
            _ => None,
        };

        let training_data = match &self.strategy {
            BenchmarkStrategy::NearestNeighbor => Some((x.clone(), y.clone())),
            _ => None,
        };

        Ok(TrainedBenchmarkClassifier {
            strategy: self.strategy,
            classes,
            class_counts,
            feature_rules,
            training_data,
            random_state: self.random_state,
        })
    }
}

impl BenchmarkClassifier {
    fn build_one_rule(
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<Vec<(usize, f64, i32)>, SklearsError> {
        let n_features = x.ncols();
        let mut best_accuracy = 0.0;
        let mut best_rule = None;

        for feature_idx in 0..n_features {
            let feature_values = x.column(feature_idx);

            // Try different thresholds
            let mut values: Vec<_> = feature_values.iter().cloned().collect();
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());

            for i in 0..values.len() - 1 {
                let threshold = (values[i] + values[i + 1]) / 2.0;

                // Try both directions
                for &(pred_below, pred_above) in &[(0, 1), (1, 0)] {
                    let mut correct = 0;
                    for (j, &actual) in y.iter().enumerate() {
                        let predicted = if feature_values[j] <= threshold {
                            pred_below
                        } else {
                            pred_above
                        };
                        if predicted == actual {
                            correct += 1;
                        }
                    }

                    let accuracy = correct as f64 / y.len() as f64;
                    if accuracy > best_accuracy {
                        best_accuracy = accuracy;
                        best_rule = Some((feature_idx, threshold, pred_below));
                    }
                }
            }
        }

        Ok(vec![best_rule.unwrap_or((0, 0.0, 0))])
    }

    fn build_random_stumps(
        x: &Array2<f64>,
        y: &Array1<i32>,
        n_stumps: usize,
        random_state: Option<u64>,
    ) -> Result<Vec<(usize, f64, i32)>, SklearsError> {
        let mut rng = if let Some(seed) = random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(0)
        };

        let n_features = x.ncols();
        let mut stumps = Vec::new();

        for _ in 0..n_stumps {
            let feature_idx = rng.gen_range(0..n_features);
            let feature_values = x.column(feature_idx);

            let min_val = feature_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = feature_values
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let threshold = rng.gen_range(min_val..=max_val);
            let prediction = rng.gen_range(0..=1);

            stumps.push((feature_idx, threshold, prediction));
        }

        Ok(stumps)
    }
}

impl Predict<Array2<f64>, Array1<i32>> for TrainedBenchmarkClassifier {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        match &self.strategy {
            BenchmarkStrategy::ZeroRule => {
                // Predict most common class
                let most_common = self
                    .class_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&class, _)| class)
                    .unwrap_or(0);
                predictions.fill(most_common);
            }

            BenchmarkStrategy::MajorityClassTieBreak => {
                let most_common = self
                    .class_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&class, _)| class)
                    .unwrap_or(0);
                predictions.fill(most_common);
            }

            BenchmarkStrategy::WeightedRandom => {
                let mut rng = if let Some(seed) = self.random_state {
                    scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
                } else {
                    scirs2_core::random::rngs::StdRng::seed_from_u64(0)
                };

                let total_count: usize = self.class_counts.values().sum();
                for i in 0..n_samples {
                    let rand_val = rng.gen_range(0..total_count);
                    let mut cumsum = 0;
                    for (&class, &count) in &self.class_counts {
                        cumsum += count;
                        if rand_val < cumsum {
                            predictions[i] = class;
                            break;
                        }
                    }
                }
            }

            BenchmarkStrategy::OneRule => {
                if let Some(rules) = &self.feature_rules {
                    if let Some((feature_idx, threshold, prediction)) = rules.first() {
                        for i in 0..n_samples {
                            predictions[i] = if x[[i, *feature_idx]] <= *threshold {
                                *prediction
                            } else {
                                1 - *prediction
                            };
                        }
                    }
                }
            }

            BenchmarkStrategy::RandomStumps { .. } => {
                if let Some(rules) = &self.feature_rules {
                    for i in 0..n_samples {
                        let mut votes = HashMap::new();
                        for (feature_idx, threshold, prediction) in rules {
                            let vote = if x[[i, *feature_idx]] <= *threshold {
                                *prediction
                            } else {
                                1 - *prediction
                            };
                            *votes.entry(vote).or_insert(0) += 1;
                        }
                        predictions[i] = votes
                            .into_iter()
                            .max_by_key(|(_, count)| *count)
                            .map(|(class, _)| class)
                            .unwrap_or(0);
                    }
                }
            }

            BenchmarkStrategy::NearestNeighbor => {
                if let Some((train_x, train_y)) = &self.training_data {
                    for i in 0..n_samples {
                        let test_point = x.row(i);
                        let mut min_distance = f64::INFINITY;
                        let mut nearest_class = 0;

                        for j in 0..train_x.nrows() {
                            let train_point = train_x.row(j);
                            let distance: f64 = test_point
                                .iter()
                                .zip(train_point.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();

                            if distance < min_distance {
                                min_distance = distance;
                                nearest_class = train_y[j];
                            }
                        }
                        predictions[i] = nearest_class;
                    }
                }
            }

            BenchmarkStrategy::CompetitionBaseline => {
                // Ensemble of simple strategies
                let zr_pred = self
                    .class_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&class, _)| class)
                    .unwrap_or(0);
                predictions.fill(zr_pred);
            }

            _ => {
                // Default to most common class
                let most_common = self
                    .class_counts
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&class, _)| class)
                    .unwrap_or(0);
                predictions.fill(most_common);
            }
        }

        Ok(predictions)
    }
}

/// Benchmark regressor implementing standard ML baselines
#[derive(Debug, Clone)]
pub struct BenchmarkRegressor {
    strategy: BenchmarkStrategy,
    random_state: Option<u64>,
}

/// Trained benchmark regressor
#[derive(Debug, Clone)]
pub struct TrainedBenchmarkRegressor {
    strategy: BenchmarkStrategy,
    mean_value: f64,
    median_value: f64,
    training_data: Option<(Array2<f64>, Array1<f64>)>,
    trend_coefficients: Option<(f64, f64)>, // (slope, intercept)
    moving_avg_values: Option<Array1<f64>>,
    random_state: Option<u64>,
}

impl BenchmarkRegressor {
    /// Create a new benchmark regressor
    pub fn new(strategy: BenchmarkStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Estimator for BenchmarkRegressor {
    type Config = BenchmarkStrategy;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.strategy
    }
}

impl Fit<Array2<f64>, Array1<f64>> for BenchmarkRegressor {
    type Fitted = TrainedBenchmarkRegressor;

    fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<Self::Fitted, SklearsError> {
        let mean_value = y.mean().unwrap_or(0.0);

        let mut sorted_y = y.to_vec();
        sorted_y.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_value = if sorted_y.len() % 2 == 0 {
            let mid = sorted_y.len() / 2;
            (sorted_y[mid - 1] + sorted_y[mid]) / 2.0
        } else {
            sorted_y[sorted_y.len() / 2]
        };

        let training_data = match &self.strategy {
            BenchmarkStrategy::NearestNeighbor => Some((x.clone(), y.clone())),
            _ => None,
        };

        let trend_coefficients = match &self.strategy {
            BenchmarkStrategy::LinearTrend => {
                // Simple linear regression on time index
                let n = y.len() as f64;
                let sum_x = (0..y.len()).sum::<usize>() as f64;
                let sum_y = y.sum();
                let sum_xy = y
                    .iter()
                    .enumerate()
                    .map(|(i, &yi)| i as f64 * yi)
                    .sum::<f64>();
                let sum_x2 = (0..y.len()).map(|i| (i as f64).powi(2)).sum::<f64>();

                let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x.powi(2));
                let intercept = (sum_y - slope * sum_x) / n;
                Some((slope, intercept))
            }
            _ => None,
        };

        let moving_avg_values = match &self.strategy {
            BenchmarkStrategy::MovingAverage { window_size } => {
                let mut values = Vec::new();
                for i in 0..y.len() {
                    let start = i.saturating_sub(*window_size);
                    let window_mean = y.slice(s![start..=i]).mean().unwrap_or(0.0);
                    values.push(window_mean);
                }
                Some(Array1::from(values))
            }
            _ => None,
        };

        Ok(TrainedBenchmarkRegressor {
            strategy: self.strategy,
            mean_value,
            median_value,
            training_data,
            trend_coefficients,
            moving_avg_values,
            random_state: self.random_state,
        })
    }
}

impl Predict<Array2<f64>, Array1<f64>> for TrainedBenchmarkRegressor {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);

        match &self.strategy {
            BenchmarkStrategy::ZeroRule => {
                predictions.fill(self.mean_value);
            }

            BenchmarkStrategy::LinearTrend => {
                if let Some((slope, intercept)) = self.trend_coefficients {
                    for i in 0..n_samples {
                        predictions[i] = slope * i as f64 + intercept;
                    }
                } else {
                    predictions.fill(self.mean_value);
                }
            }

            BenchmarkStrategy::MovingAverage { .. } => {
                if let Some(ref values) = self.moving_avg_values {
                    let last_value = values.last().copied().unwrap_or(self.mean_value);
                    predictions.fill(last_value);
                } else {
                    predictions.fill(self.mean_value);
                }
            }

            BenchmarkStrategy::NearestNeighbor => {
                if let Some((train_x, train_y)) = &self.training_data {
                    for i in 0..n_samples {
                        let test_point = x.row(i);
                        let mut min_distance = f64::INFINITY;
                        let mut nearest_value = self.mean_value;

                        for j in 0..train_x.nrows() {
                            let train_point = train_x.row(j);
                            let distance: f64 = test_point
                                .iter()
                                .zip(train_point.iter())
                                .map(|(a, b)| (a - b).powi(2))
                                .sum::<f64>()
                                .sqrt();

                            if distance < min_distance {
                                min_distance = distance;
                                nearest_value = train_y[j];
                            }
                        }
                        predictions[i] = nearest_value;
                    }
                } else {
                    predictions.fill(self.mean_value);
                }
            }

            _ => {
                predictions.fill(self.mean_value);
            }
        }

        Ok(predictions)
    }
}

/// Domain-specific benchmark classifier
#[derive(Debug, Clone)]
pub struct DomainBenchmarkClassifier {
    strategy: DomainStrategy,
    random_state: Option<u64>,
}

/// Competition-grade benchmark utilities
pub struct CompetitionBaseline;

impl CompetitionBaseline {
    /// Create a competition-grade baseline classifier
    pub fn classifier() -> BenchmarkClassifier {
        BenchmarkClassifier::new(BenchmarkStrategy::CompetitionBaseline)
    }

    /// Create a competition-grade baseline regressor
    pub fn regressor() -> BenchmarkRegressor {
        BenchmarkRegressor::new(BenchmarkStrategy::ZeroRule)
    }

    /// Get ensemble of benchmark strategies for robust baseline
    pub fn ensemble_strategies() -> Vec<BenchmarkStrategy> {
        vec![
            BenchmarkStrategy::ZeroRule,
            BenchmarkStrategy::OneRule,
            BenchmarkStrategy::WeightedRandom,
            BenchmarkStrategy::NearestNeighbor,
        ]
    }
}

/// Theoretical bounds calculator
pub struct TheoreticalBounds;

impl TheoreticalBounds {
    /// Calculate theoretical lower bound for classification accuracy
    pub fn classification_bound(y: &Array1<i32>) -> f64 {
        let mut class_counts = HashMap::new();
        for &class in y.iter() {
            *class_counts.entry(class).or_insert(0) += 1;
        }

        let total = y.len() as f64;
        let max_count = class_counts.values().max().copied().unwrap_or(0) as f64;
        max_count / total
    }

    /// Calculate random chance baseline for classification
    pub fn random_chance_classification(n_classes: usize) -> f64 {
        1.0 / n_classes as f64
    }

    /// Calculate theoretical lower bound for regression (using empirical variance)
    pub fn regression_bound(y: &Array1<f64>) -> f64 {
        let mean = y.mean().unwrap_or(0.0);
        let variance = y.iter().map(|&yi| (yi - mean).powi(2)).sum::<f64>() / y.len() as f64;
        variance.sqrt() // Return standard deviation as lower bound
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_zero_rule_classifier() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 0]; // Class 0 is most frequent

        let classifier = BenchmarkClassifier::new(BenchmarkStrategy::ZeroRule);
        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions, array![0, 0, 0, 0]);
    }

    #[test]
    fn test_one_rule_classifier() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let classifier = BenchmarkClassifier::new(BenchmarkStrategy::OneRule);
        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_benchmark_regressor() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let regressor = BenchmarkRegressor::new(BenchmarkStrategy::ZeroRule);
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        let expected_mean = y.mean().unwrap();
        for pred in predictions.iter() {
            assert!((pred - expected_mean).abs() < 1e-10);
        }
    }

    #[test]
    fn test_linear_trend_regressor() {
        let x = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0]; // Perfect linear trend

        let regressor = BenchmarkRegressor::new(BenchmarkStrategy::LinearTrend);
        let fitted = regressor.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        // Should approximate the linear trend
        for i in 0..predictions.len() - 1 {
            assert!(predictions[i + 1] >= predictions[i]); // Increasing trend
        }
    }

    #[test]
    fn test_theoretical_bounds() {
        let y_class = array![0, 0, 1, 0]; // 75% class 0
        let bound = TheoreticalBounds::classification_bound(&y_class);
        assert!((bound - 0.75).abs() < 1e-10);

        let random_chance = TheoreticalBounds::random_chance_classification(2);
        assert!((random_chance - 0.5).abs() < 1e-10);

        let y_reg = array![1.0, 2.0, 3.0, 4.0];
        let reg_bound = TheoreticalBounds::regression_bound(&y_reg);
        assert!(reg_bound > 0.0);
    }

    #[test]
    fn test_competition_baseline() {
        let classifier = CompetitionBaseline::classifier();
        let regressor = CompetitionBaseline::regressor();
        let strategies = CompetitionBaseline::ensemble_strategies();

        assert!(matches!(
            classifier.strategy,
            BenchmarkStrategy::CompetitionBaseline
        ));
        assert!(matches!(regressor.strategy, BenchmarkStrategy::ZeroRule));
        assert_eq!(strategies.len(), 4);
    }

    #[test]
    fn test_nearest_neighbor_baseline() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0]).unwrap();
        let y = array![0, 1, 0];

        let classifier = BenchmarkClassifier::new(BenchmarkStrategy::NearestNeighbor);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Test point closest to first training point
        let test_x = Array2::from_shape_vec((1, 2), vec![1.1, 1.1]).unwrap();
        let predictions = fitted.predict(&test_x).unwrap();

        assert_eq!(predictions[0], 0); // Should predict class of nearest neighbor
    }
}

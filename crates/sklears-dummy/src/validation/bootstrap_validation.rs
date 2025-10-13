use super::validation_core::*;
use super::validation_metrics::*;

use scirs2_core::ndarray::{Array1, Axis};
use scirs2_core::random::{Rng, RngCore};
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

use crate::{ClassifierStrategy, DummyClassifier, DummyRegressor, RegressorStrategy};

/// Bootstrap validation result
#[derive(Debug, Clone)]
pub struct BootstrapValidationResult {
    /// bootstrap_scores
    pub bootstrap_scores: Vec<Float>,
    /// mean_score
    pub mean_score: Float,
    /// std_score
    pub std_score: Float,
    /// confidence_interval
    pub confidence_interval: (Float, Float),
    /// bias
    pub bias: Float,
    /// strategy
    pub strategy: String,
    /// n_bootstrap_samples
    pub n_bootstrap_samples: usize,
}

impl BootstrapValidationResult {
    pub fn new(bootstrap_scores: Vec<Float>, strategy: String, confidence_level: Float) -> Self {
        let n_bootstrap_samples = bootstrap_scores.len();
        let mean_score = bootstrap_scores.iter().sum::<Float>() / n_bootstrap_samples as Float;

        let variance = bootstrap_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / n_bootstrap_samples as Float;
        let std_score = variance.sqrt();

        // Calculate confidence interval using percentile method
        let mut sorted_scores = bootstrap_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let alpha = 1.0 - confidence_level;
        let lower_idx = (alpha / 2.0 * n_bootstrap_samples as Float) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap_samples as Float) as usize;

        let lower_bound = sorted_scores[lower_idx.min(n_bootstrap_samples - 1)];
        let upper_bound = sorted_scores[upper_idx.min(n_bootstrap_samples - 1)];
        let confidence_interval = (lower_bound, upper_bound);

        // Bias calculation (simplified)
        let bias = 0.0; // Would require original score for proper bias calculation

        Self {
            bootstrap_scores,
            mean_score,
            std_score,
            confidence_interval,
            bias,
            strategy,
            n_bootstrap_samples,
        }
    }

    pub fn percentile(&self, p: Float) -> Float {
        let mut sorted_scores = self.bootstrap_scores.clone();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let idx = (p * self.n_bootstrap_samples as Float) as usize;
        sorted_scores[idx.min(self.n_bootstrap_samples - 1)]
    }

    pub fn bootstrap_distribution_summary(&self) -> StatisticalSummary {
        StatisticalSummary::from_scores(&self.bootstrap_scores)
    }
}

/// Perform bootstrap validation for a dummy classifier
pub fn bootstrap_validate_classifier(
    classifier: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<BootstrapValidationResult> {
    if n_bootstrap < 1 {
        return Err(SklearsError::InvalidInput(
            "Number of bootstrap samples must be at least 1".to_string(),
        ));
    }

    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "Cannot perform bootstrap validation on empty dataset".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let mut bootstrap_scores = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Create bootstrap sample
        let bootstrap_indices = create_bootstrap_sample(n_samples, &mut *rng);
        let oob_indices = create_out_of_bag_indices(&bootstrap_indices, n_samples);

        if oob_indices.is_empty() {
            continue; // Skip if no out-of-bag samples
        }

        // Extract bootstrap training data
        let x_bootstrap = x.select(Axis(0), &bootstrap_indices);
        let y_bootstrap = y.select(Axis(0), &bootstrap_indices);

        // Extract out-of-bag test data
        let x_oob = x.select(Axis(0), &oob_indices);
        let y_oob = y.select(Axis(0), &oob_indices);

        // Fit on bootstrap sample and predict on out-of-bag
        let fitted = classifier.clone().fit(&x_bootstrap, &y_bootstrap)?;
        let predictions = fitted.predict(&x_oob)?;

        // Calculate accuracy
        let correct = predictions
            .iter()
            .zip(y_oob.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        let accuracy = correct as Float / oob_indices.len() as Float;
        bootstrap_scores.push(accuracy);
    }

    if bootstrap_scores.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid bootstrap samples created".to_string(),
        ));
    }

    Ok(BootstrapValidationResult::new(
        bootstrap_scores,
        format!("{:?}", classifier.strategy),
        0.95,
    ))
}

/// Perform bootstrap validation for a dummy regressor
pub fn bootstrap_validate_regressor(
    regressor: DummyRegressor,
    x: &Features,
    y: &Array1<Float>,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<BootstrapValidationResult> {
    if n_bootstrap < 1 {
        return Err(SklearsError::InvalidInput(
            "Number of bootstrap samples must be at least 1".to_string(),
        ));
    }

    let n_samples = x.nrows();
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "Cannot perform bootstrap validation on empty dataset".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);
    let mut bootstrap_scores = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Create bootstrap sample
        let bootstrap_indices = create_bootstrap_sample(n_samples, &mut *rng);
        let oob_indices = create_out_of_bag_indices(&bootstrap_indices, n_samples);

        if oob_indices.is_empty() {
            continue; // Skip if no out-of-bag samples
        }

        // Extract bootstrap training data
        let x_bootstrap = x.select(Axis(0), &bootstrap_indices);
        let y_bootstrap = y.select(Axis(0), &bootstrap_indices);

        // Extract out-of-bag test data
        let x_oob = x.select(Axis(0), &oob_indices);
        let y_oob = y.select(Axis(0), &oob_indices);

        // Fit on bootstrap sample and predict on out-of-bag
        let fitted = regressor.clone().fit(&x_bootstrap, &y_bootstrap)?;
        let predictions = fitted.predict(&x_oob)?;

        // Calculate negative MSE
        let mse = predictions
            .iter()
            .zip(y_oob.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<Float>()
            / oob_indices.len() as Float;
        bootstrap_scores.push(-mse);
    }

    if bootstrap_scores.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid bootstrap samples created".to_string(),
        ));
    }

    Ok(BootstrapValidationResult::new(
        bootstrap_scores,
        format!("{:?}", regressor.strategy),
        0.95,
    ))
}

/// Create a bootstrap sample of indices
fn create_bootstrap_sample(n_samples: usize, rng: &mut dyn RngCore) -> Vec<usize> {
    (0..n_samples)
        .map(|_| rng.gen_range(0..n_samples))
        .collect()
}

/// Create out-of-bag indices (samples not in bootstrap sample)
fn create_out_of_bag_indices(bootstrap_indices: &[usize], n_samples: usize) -> Vec<usize> {
    let mut in_bootstrap = vec![false; n_samples];
    for &idx in bootstrap_indices {
        in_bootstrap[idx] = true;
    }

    (0..n_samples).filter(|&i| !in_bootstrap[i]).collect()
}

/// Perform bootstrap validation with multiple strategies
pub fn bootstrap_compare_strategies(
    strategies: &[String],
    x: &Features,
    y: &Array1<Float>,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<Vec<BootstrapValidationResult>> {
    if strategies.is_empty() {
        return Err(SklearsError::InvalidInput(
            "At least one strategy must be provided".to_string(),
        ));
    }

    let mut results = Vec::new();
    let is_classification = is_classification_task(y);

    if is_classification {
        let y_int: Array1<Int> = y.mapv(|x| x as Int);

        for strategy_name in strategies {
            let strategy = parse_classifier_strategy(strategy_name)?;
            let classifier = DummyClassifier::new(strategy);
            let result =
                bootstrap_validate_classifier(classifier, x, &y_int, n_bootstrap, random_state)?;
            results.push(result);
        }
    } else {
        for strategy_name in strategies {
            let strategy = parse_regressor_strategy(strategy_name)?;
            let regressor = DummyRegressor::new(strategy);
            let result = bootstrap_validate_regressor(regressor, x, y, n_bootstrap, random_state)?;
            results.push(result);
        }
    }

    Ok(results)
}

/// Bootstrap hypothesis test for comparing two strategies
pub fn bootstrap_hypothesis_test(
    strategy1: DummyClassifier,
    strategy2: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<BootstrapHypothesisTest> {
    let mut rng = create_rng(random_state);
    let n_samples = x.nrows();

    let mut differences = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        // Create bootstrap sample
        let bootstrap_indices = create_bootstrap_sample(n_samples, &mut *rng);
        let oob_indices = create_out_of_bag_indices(&bootstrap_indices, n_samples);

        if oob_indices.is_empty() {
            continue;
        }

        // Extract data
        let x_bootstrap = x.select(Axis(0), &bootstrap_indices);
        let y_bootstrap = y.select(Axis(0), &bootstrap_indices);
        let x_oob = x.select(Axis(0), &oob_indices);
        let y_oob = y.select(Axis(0), &oob_indices);

        // Evaluate both strategies
        let fitted1 = strategy1.clone().fit(&x_bootstrap, &y_bootstrap)?;
        let predictions1 = fitted1.predict(&x_oob)?;
        let score1 = calculate_classification_score(&predictions1, &y_oob, "accuracy")?;

        let fitted2 = strategy2.clone().fit(&x_bootstrap, &y_bootstrap)?;
        let predictions2 = fitted2.predict(&x_oob)?;
        let score2 = calculate_classification_score(&predictions2, &y_oob, "accuracy")?;

        differences.push(score1 - score2);
    }

    Ok(BootstrapHypothesisTest::new(differences))
}

/// Bootstrap hypothesis test result
#[derive(Debug, Clone)]
pub struct BootstrapHypothesisTest {
    /// differences
    pub differences: Vec<Float>,
    /// mean_difference
    pub mean_difference: Float,
    /// std_difference
    pub std_difference: Float,
    /// p_value
    pub p_value: Float,
    /// confidence_interval
    pub confidence_interval: (Float, Float),
}

impl BootstrapHypothesisTest {
    pub fn new(differences: Vec<Float>) -> Self {
        let n = differences.len();
        let mean_difference = differences.iter().sum::<Float>() / n as Float;

        let variance = differences
            .iter()
            .map(|&d| (d - mean_difference).powi(2))
            .sum::<Float>()
            / n as Float;
        let std_difference = variance.sqrt();

        // Calculate p-value (two-tailed test for H0: difference = 0)
        let negative_count = differences.iter().filter(|&&d| d < 0.0).count();
        let positive_count = differences.iter().filter(|&&d| d > 0.0).count();
        let p_value = 2.0 * (negative_count.min(positive_count) as Float / n as Float);

        // Calculate 95% confidence interval
        let mut sorted_diffs = differences.clone();
        sorted_diffs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let lower_idx = (0.025 * n as Float) as usize;
        let upper_idx = (0.975 * n as Float) as usize;
        let confidence_interval = (
            sorted_diffs[lower_idx.min(n - 1)],
            sorted_diffs[upper_idx.min(n - 1)],
        );

        Self {
            differences,
            mean_difference,
            std_difference,
            p_value,
            confidence_interval,
        }
    }

    pub fn is_significant(&self, alpha: Float) -> bool {
        self.p_value < alpha
    }

    pub fn effect_size(&self) -> Float {
        if self.std_difference > 0.0 {
            self.mean_difference / self.std_difference
        } else {
            0.0
        }
    }
}

/// Stratified bootstrap validation
pub fn stratified_bootstrap_validate_classifier(
    classifier: DummyClassifier,
    x: &Features,
    y: &Array1<Int>,
    n_bootstrap: usize,
    random_state: Option<u64>,
) -> Result<BootstrapValidationResult> {
    let mut rng = create_rng(random_state);
    let n_samples = x.nrows();

    // Group indices by class
    let mut class_indices: HashMap<Int, Vec<usize>> = HashMap::new();
    for (i, &class) in y.iter().enumerate() {
        class_indices.entry(class).or_default().push(i);
    }

    let mut bootstrap_scores = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let mut bootstrap_indices = Vec::new();
        let mut oob_indices = Vec::new();

        // Create stratified bootstrap sample
        for indices in class_indices.values() {
            let class_bootstrap = create_bootstrap_sample(indices.len(), &mut *rng);
            let class_bootstrap_indices: Vec<usize> =
                class_bootstrap.iter().map(|&i| indices[i]).collect();

            let class_oob = create_out_of_bag_indices(&class_bootstrap, indices.len());
            let class_oob_indices: Vec<usize> = class_oob.iter().map(|&i| indices[i]).collect();

            bootstrap_indices.extend(class_bootstrap_indices);
            oob_indices.extend(class_oob_indices);
        }

        if oob_indices.is_empty() {
            continue;
        }

        // Extract data and evaluate
        let x_bootstrap = x.select(Axis(0), &bootstrap_indices);
        let y_bootstrap = y.select(Axis(0), &bootstrap_indices);
        let x_oob = x.select(Axis(0), &oob_indices);
        let y_oob = y.select(Axis(0), &oob_indices);

        let fitted = classifier.clone().fit(&x_bootstrap, &y_bootstrap)?;
        let predictions = fitted.predict(&x_oob)?;

        let correct = predictions
            .iter()
            .zip(y_oob.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count();
        let accuracy = correct as Float / oob_indices.len() as Float;
        bootstrap_scores.push(accuracy);
    }

    if bootstrap_scores.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid bootstrap samples created".to_string(),
        ));
    }

    Ok(BootstrapValidationResult::new(
        bootstrap_scores,
        format!("{:?}", classifier.strategy),
        0.95,
    ))
}

/// Helper functions for parsing strategies (simplified versions)
fn parse_classifier_strategy(strategy: &str) -> Result<ClassifierStrategy> {
    match strategy.to_lowercase().as_str() {
        "mostfrequent" | "most_frequent" => Ok(ClassifierStrategy::MostFrequent),
        "stratified" => Ok(ClassifierStrategy::Stratified),
        "uniform" => Ok(ClassifierStrategy::Uniform),
        "constant" => Ok(ClassifierStrategy::Constant),
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown classifier strategy: {}",
            strategy
        ))),
    }
}

fn parse_regressor_strategy(strategy: &str) -> Result<RegressorStrategy> {
    match strategy.to_lowercase().as_str() {
        "mean" => Ok(RegressorStrategy::Mean),
        "median" => Ok(RegressorStrategy::Median),
        "quantile" => Ok(RegressorStrategy::Quantile(0.5)),
        "constant" => Ok(RegressorStrategy::Constant(0.0)),
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown regressor strategy: {}",
            strategy
        ))),
    }
}

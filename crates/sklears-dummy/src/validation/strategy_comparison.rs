use super::cross_validation::*;
use super::validation_core::*;

use scirs2_core::ndarray::Array1;
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

use crate::{ClassifierStrategy, DummyClassifier, DummyRegressor, RegressorStrategy};

/// Compare multiple dummy strategies and return results for each
pub fn compare_dummy_strategies(
    strategies: &[String],
    x: &Features,
    y: &Array1<Float>,
    cv: usize,
) -> Result<Vec<DummyValidationResult>> {
    if strategies.is_empty() {
        return Err(SklearsError::InvalidInput(
            "At least one strategy must be provided".to_string(),
        ));
    }

    let mut results = Vec::new();

    // Determine if this is a classification or regression task
    let is_classification = is_classification_task(y);

    if is_classification {
        // Convert to Int for classification
        let y_int: Array1<Int> = y.mapv(|x| x as Int);

        for strategy_name in strategies {
            let strategy = parse_classifier_strategy(strategy_name)?;
            let classifier = DummyClassifier::new(strategy);
            let result = cross_validate_dummy_classifier(classifier, x, &y_int, cv)?;
            results.push(result);
        }
    } else {
        for strategy_name in strategies {
            let strategy = parse_regressor_strategy(strategy_name)?;
            let regressor = DummyRegressor::new(strategy);
            let result = cross_validate_dummy_regressor(regressor, x, y, cv)?;
            results.push(result);
        }
    }

    Ok(results)
}

/// Compare classifier strategies with detailed analysis
pub fn compare_classifier_strategies(
    strategies: &[ClassifierStrategy],
    x: &Features,
    y: &Array1<Int>,
    config: &ValidationConfig,
) -> Result<StrategyComparisonResult> {
    let mut results = Vec::new();
    let mut strategy_names = Vec::new();

    for strategy in strategies {
        let classifier = DummyClassifier::new(strategy.clone());
        let result = comprehensive_cross_validate_classifier(classifier, x, y, config)?;
        strategy_names.push(format!("{:?}", strategy));
        results.push(result);
    }

    let analysis = StrategyAnalysis::from_results(&results, &strategy_names)?;

    Ok(StrategyComparisonResult {
        results,
        analysis,
        config: config.clone(),
    })
}

/// Compare regressor strategies with detailed analysis
pub fn compare_regressor_strategies(
    strategies: &[RegressorStrategy],
    x: &Features,
    y: &Array1<Float>,
    config: &ValidationConfig,
) -> Result<StrategyComparisonResult> {
    let mut results = Vec::new();
    let mut strategy_names = Vec::new();

    for strategy in strategies {
        let regressor = DummyRegressor::new(*strategy);
        let result = comprehensive_cross_validate_regressor(regressor, x, y, config)?;
        strategy_names.push(format!("{:?}", strategy));
        results.push(result);
    }

    let analysis = StrategyAnalysis::from_results(&results, &strategy_names)?;

    Ok(StrategyComparisonResult {
        results,
        analysis,
        config: config.clone(),
    })
}

/// Result of strategy comparison analysis
#[derive(Debug, Clone)]
pub struct StrategyComparisonResult {
    /// results
    pub results: Vec<ComprehensiveValidationResult>,
    /// analysis
    pub analysis: StrategyAnalysis,
    /// config
    pub config: ValidationConfig,
}

/// Analysis of strategy comparison
#[derive(Debug, Clone)]
pub struct StrategyAnalysis {
    /// best_strategy
    pub best_strategy: String,
    /// worst_strategy
    pub worst_strategy: String,
    /// ranking
    pub ranking: Vec<(String, Float)>,
    /// statistical_tests
    pub statistical_tests: Vec<StatisticalTest>,
    /// performance_matrix
    pub performance_matrix: HashMap<String, Float>,
}

impl StrategyAnalysis {
    pub fn from_results(
        results: &[ComprehensiveValidationResult],
        strategy_names: &[String],
    ) -> Result<Self> {
        if results.is_empty() || strategy_names.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one result must be provided".to_string(),
            ));
        }

        // Create ranking based on mean scores
        let mut ranking: Vec<(String, Float)> = results
            .iter()
            .zip(strategy_names.iter())
            .map(|(result, name)| (name.clone(), result.validation_result.mean_score))
            .collect();

        ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let best_strategy = ranking[0].0.clone();
        let worst_strategy = ranking[ranking.len() - 1].0.clone();

        // Create performance matrix
        let mut performance_matrix = HashMap::new();
        for (name, score) in &ranking {
            performance_matrix.insert(name.clone(), *score);
        }

        // Perform statistical tests
        let statistical_tests = perform_pairwise_tests(results, strategy_names)?;

        Ok(Self {
            best_strategy,
            worst_strategy,
            ranking,
            statistical_tests,
            performance_matrix,
        })
    }

    pub fn get_significant_differences(&self, alpha: Float) -> Vec<&StatisticalTest> {
        self.statistical_tests
            .iter()
            .filter(|test| test.p_value < alpha)
            .collect()
    }

    pub fn print_summary(&self) {
        println!("Strategy Comparison Summary:");
        println!(
            "Best Strategy: {} (Score: {:.4})",
            self.best_strategy, self.ranking[0].1
        );
        println!(
            "Worst Strategy: {} (Score: {:.4})",
            self.worst_strategy,
            self.ranking[self.ranking.len() - 1].1
        );
        println!("\nRanking:");
        for (i, (name, score)) in self.ranking.iter().enumerate() {
            println!("  {}. {} (Score: {:.4})", i + 1, name, score);
        }
    }
}

/// Statistical test result
#[derive(Debug, Clone)]
pub struct StatisticalTest {
    /// strategy1
    pub strategy1: String,
    /// strategy2
    pub strategy2: String,
    /// test_statistic
    pub test_statistic: Float,
    /// p_value
    pub p_value: Float,
    /// effect_size
    pub effect_size: Float,
    /// test_type
    pub test_type: String,
}

/// Perform pairwise statistical tests between strategies
fn perform_pairwise_tests(
    results: &[ComprehensiveValidationResult],
    strategy_names: &[String],
) -> Result<Vec<StatisticalTest>> {
    let mut tests = Vec::new();

    for i in 0..results.len() {
        for j in (i + 1)..results.len() {
            let result1 = &results[i];
            let result2 = &results[j];

            let test = perform_paired_t_test(
                &result1.validation_result.fold_scores,
                &result2.validation_result.fold_scores,
                &strategy_names[i],
                &strategy_names[j],
            )?;

            tests.push(test);
        }
    }

    Ok(tests)
}

/// Perform paired t-test between two sets of fold scores
pub fn perform_paired_t_test(
    scores1: &[Float],
    scores2: &[Float],
    name1: &str,
    name2: &str,
) -> Result<StatisticalTest> {
    if scores1.len() != scores2.len() {
        return Err(SklearsError::InvalidInput(
            "Score arrays must have the same length".to_string(),
        ));
    }

    let n = scores1.len() as Float;
    if n < 2.0 {
        return Err(SklearsError::InvalidInput(
            "At least 2 observations required for t-test".to_string(),
        ));
    }

    // Calculate differences
    let differences: Vec<Float> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(&s1, &s2)| s1 - s2)
        .collect();

    let mean_diff = differences.iter().sum::<Float>() / n;
    let var_diff = differences
        .iter()
        .map(|&d| (d - mean_diff).powi(2))
        .sum::<Float>()
        / (n - 1.0);
    let std_diff = var_diff.sqrt();

    let t_statistic = if std_diff > 0.0 {
        mean_diff / (std_diff / n.sqrt())
    } else {
        0.0
    };

    // Approximate p-value using normal distribution (for large samples)
    let p_value = if t_statistic.abs() < 1e-10 {
        1.0
    } else {
        2.0 * (1.0 - normal_cdf(t_statistic.abs()))
    };

    // Calculate effect size (Cohen's d)
    let pooled_std = ((scores1.iter().map(|&x| x.powi(2)).sum::<Float>()
        + scores2.iter().map(|&x| x.powi(2)).sum::<Float>())
        / (2.0 * n))
        .sqrt();
    let effect_size = if pooled_std > 0.0 {
        mean_diff / pooled_std
    } else {
        0.0
    };

    Ok(StatisticalTest {
        strategy1: name1.to_string(),
        strategy2: name2.to_string(),
        test_statistic: t_statistic,
        p_value,
        effect_size,
        test_type: "paired_t_test".to_string(),
    })
}

/// Approximate normal CDF for p-value calculation
fn normal_cdf(x: Float) -> Float {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Approximate error function
fn erf(x: Float) -> Float {
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

/// Parse classifier strategy from string
fn parse_classifier_strategy(strategy: &str) -> Result<ClassifierStrategy> {
    match strategy.to_lowercase().as_str() {
        "mostfrequent" | "most_frequent" => Ok(ClassifierStrategy::MostFrequent),
        "stratified" => Ok(ClassifierStrategy::Stratified),
        "uniform" => Ok(ClassifierStrategy::Uniform),
        "constant" => Ok(ClassifierStrategy::Constant), // Default constant value
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown classifier strategy: {}",
            strategy
        ))),
    }
}

/// Parse regressor strategy from string
fn parse_regressor_strategy(strategy: &str) -> Result<RegressorStrategy> {
    match strategy.to_lowercase().as_str() {
        "mean" => Ok(RegressorStrategy::Mean),
        "median" => Ok(RegressorStrategy::Median),
        "quantile" => Ok(RegressorStrategy::Quantile(0.5)), // Default to median
        "constant" => Ok(RegressorStrategy::Constant(0.0)), // Default constant value
        _ => Err(SklearsError::InvalidInput(format!(
            "Unknown regressor strategy: {}",
            strategy
        ))),
    }
}

/// Find the best strategy from a list of validation results
pub fn find_best_strategy(results: &[DummyValidationResult]) -> Option<&DummyValidationResult> {
    results.iter().max_by(|a, b| {
        a.mean_score
            .partial_cmp(&b.mean_score)
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Compare strategies with multiple metrics
pub fn compare_strategies_multi_metric(
    strategies: &[String],
    x: &Features,
    y: &Array1<Float>,
    cv: usize,
    metrics: &[String],
) -> Result<HashMap<String, Vec<DummyValidationResult>>> {
    let mut results = HashMap::new();

    for metric in metrics {
        let mut config = ValidationConfig::default();
        config.scoring_metric = metric.clone();
        config.cv_folds = cv;

        let metric_results = compare_dummy_strategies(strategies, x, y, cv)?;
        results.insert(metric.clone(), metric_results);
    }

    Ok(results)
}

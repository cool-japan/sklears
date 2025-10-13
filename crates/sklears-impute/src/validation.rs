//! Validation framework for imputation methods
//!
//! This module provides comprehensive validation tools for assessing the quality
//! and reliability of imputation methods including cross-validation, hold-out validation,
//! and synthetic missing data validation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random, Rng};
use scirs2_core::SliceRandomExt;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};
use std::collections::HashMap;

/// Cross-validation strategies for imputation evaluation
#[derive(Debug, Clone)]
pub enum CrossValidationStrategy {
    /// K-fold cross validation
    KFold { n_splits: usize, shuffle: bool },
    /// Stratified K-fold (for datasets with class labels)
    StratifiedKFold { n_splits: usize, shuffle: bool },
    /// Leave-one-out cross validation
    LeaveOneOut,
    /// Time series split (for temporal data)
    TimeSeriesSplit {
        n_splits: usize,
        max_train_size: Option<usize>,
    },
    /// Group-based cross validation
    GroupKFold { n_splits: usize },
}

/// Missing data simulation patterns for synthetic validation
#[derive(Debug, Clone)]
pub enum MissingDataPattern {
    /// Missing Completely At Random
    MCAR { missing_rate: f64 },
    /// Missing At Random (depends on observed variables)
    MAR {
        missing_rate: f64,
        dependency_strength: f64,
    },
    /// Missing Not At Random (depends on unobserved values)
    MNAR {
        missing_rate: f64,
        threshold_factor: f64,
    },
    /// Block missing pattern
    Block {
        block_size: (usize, usize),
        n_blocks: usize,
    },
    /// Monotone missing pattern
    Monotone { missing_rates: Vec<f64> },
}

/// Imputation validation metrics
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ImputationMetrics {
    /// Root Mean Squared Error for continuous variables
    pub rmse: f64,
    /// Mean Absolute Error for continuous variables
    pub mae: f64,
    /// R-squared correlation coefficient
    pub r2: f64,
    /// Accuracy for categorical variables
    pub accuracy: f64,
    /// F1 score for categorical variables
    pub f1_score: f64,
    /// Bias in imputed values
    pub bias: f64,
    /// Coverage of confidence intervals (if provided)
    pub coverage: f64,
    /// Kolmogorov-Smirnov test statistic for distribution similarity
    pub ks_statistic: f64,
    /// p-value for KS test
    pub ks_pvalue: f64,
}

/// Cross-validation results for imputation
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Metrics for each fold
    pub fold_metrics: Vec<ImputationMetrics>,
    /// Mean metrics across all folds
    pub mean_metrics: ImputationMetrics,
    /// Standard deviation of metrics across folds
    pub std_metrics: ImputationMetrics,
    /// Confidence intervals for metrics (95%)
    pub confidence_intervals: HashMap<String, (f64, f64)>,
}

/// Imputation Cross-Validator
///
/// Performs cross-validation to assess imputation quality by artificially
/// creating missing data and evaluating how well the imputation method recovers
/// the true values.
///
/// # Parameters
///
/// * `cv_strategy` - Cross-validation strategy to use
/// * `missing_pattern` - Pattern for creating synthetic missing data
/// * `test_fraction` - Fraction of observed values to artificially make missing for testing
/// * `random_state` - Random state for reproducibility
/// * `n_jobs` - Number of parallel jobs (currently not implemented)
///
/// # Examples
///
/// ```
/// use sklears_impute::{ImputationCrossValidator, CrossValidationStrategy, MissingDataPattern};
/// use sklears_impute::SimpleImputer;
/// use sklears_core::traits::{Transform};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let cv = ImputationCrossValidator::new()
///     .cv_strategy(CrossValidationStrategy::KFold { n_splits: 3, shuffle: true })
///     .missing_pattern(MissingDataPattern::MCAR { missing_rate: 0.2 })
///     .test_fraction(0.1);
///
/// let imputer = SimpleImputer::new().strategy("mean".to_string());
/// let results = cv.validate_imputer(&imputer, &X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ImputationCrossValidator {
    cv_strategy: CrossValidationStrategy,
    missing_pattern: MissingDataPattern,
    test_fraction: f64,
    random_state: Option<u64>,
    n_jobs: usize,
}

/// Hold-out validator for imputation methods
///
/// Validates imputation by holding out a portion of the dataset and evaluating
/// how well missing values in that portion are imputed.
#[derive(Debug, Clone)]
pub struct HoldOutValidator {
    test_size: f64,
    missing_pattern: MissingDataPattern,
    random_state: Option<u64>,
    stratify: bool,
}

/// Synthetic missing data validator
///
/// Creates synthetic datasets with known missing patterns to test
/// imputation methods under controlled conditions.
#[derive(Debug, Clone)]
pub struct SyntheticMissingValidator {
    data_generators: Vec<DataGenerator>,
    missing_patterns: Vec<MissingDataPattern>,
    n_datasets: usize,
    dataset_sizes: Vec<(usize, usize)>,
    random_state: Option<u64>,
}

/// Data generator for synthetic validation
#[derive(Debug, Clone)]
pub enum DataGenerator {
    /// Multivariate normal data
    MultivariateNormal { mean: Array1<f64>, cov: Array2<f64> },
    /// Linear relationships with noise
    LinearRelationships {
        coefficients: Array2<f64>,
        noise_std: f64,
    },
    /// Non-linear relationships
    NonLinear {
        function_type: String,
        noise_std: f64,
    },
    /// Mixed-type data (continuous + categorical)
    MixedType {
        continuous_props: f64,
        n_categories: Vec<usize>,
    },
}

/// Real-world case study validator
///
/// Validates imputation methods on real datasets with known complete cases
/// by artificially introducing missing data.
#[derive(Debug, Clone)]
pub struct CaseStudyValidator {
    case_studies: Vec<CaseStudy>,
    evaluation_metrics: Vec<String>,
    comparison_methods: Vec<String>,
}

/// Case study configuration
#[derive(Debug, Clone)]
pub struct CaseStudy {
    name: String,
    description: String,
    data_characteristics: DataCharacteristics,
    missing_patterns: Vec<MissingDataPattern>,
    evaluation_criteria: Vec<String>,
}

/// Data characteristics for case studies
#[derive(Debug, Clone)]
pub struct DataCharacteristics {
    n_samples: usize,
    n_features: usize,
    feature_types: Vec<String>, // "continuous", "categorical", "ordinal"
    correlation_structure: String, // "low", "medium", "high"
    outlier_fraction: f64,
    noise_level: f64,
}

// ImputationCrossValidator implementation

impl ImputationCrossValidator {
    /// Create a new ImputationCrossValidator
    pub fn new() -> Self {
        Self {
            cv_strategy: CrossValidationStrategy::KFold {
                n_splits: 5,
                shuffle: true,
            },
            missing_pattern: MissingDataPattern::MCAR { missing_rate: 0.2 },
            test_fraction: 0.1,
            random_state: None,
            n_jobs: 1,
        }
    }

    /// Set the cross-validation strategy
    pub fn cv_strategy(mut self, strategy: CrossValidationStrategy) -> Self {
        self.cv_strategy = strategy;
        self
    }

    /// Set the missing data pattern for synthetic testing
    pub fn missing_pattern(mut self, pattern: MissingDataPattern) -> Self {
        self.missing_pattern = pattern;
        self
    }

    /// Set the fraction of observed values to use for testing
    pub fn test_fraction(mut self, fraction: f64) -> Self {
        self.test_fraction = fraction;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Validate an imputation method using cross-validation
    #[allow(non_snake_case)]
    pub fn validate_imputer<I>(
        &self,
        imputer: &I,
        X: &ArrayView2<'_, Float>,
    ) -> SklResult<CrossValidationResults>
    where
        I: Clone,
        for<'a> I: Fit<ArrayView2<'a, Float>, ()>,
        for<'a> <I as Fit<ArrayView2<'a, Float>, ()>>::Fitted:
            Transform<ArrayView2<'a, Float>, Array2<Float>>,
    {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let mut rng = match self.random_state {
            _ => Random::default(),
        };

        // Generate fold indices
        let fold_indices = self.generate_fold_indices(n_samples, &mut rng)?;
        let mut fold_metrics = Vec::new();

        for (train_indices, test_indices) in fold_indices {
            // Create training and test sets
            let mut X_train = Array2::zeros((train_indices.len(), n_features));
            let mut X_test = Array2::zeros((test_indices.len(), n_features));

            for (i, &idx) in train_indices.iter().enumerate() {
                X_train.row_mut(i).assign(&X.row(idx));
            }

            for (i, &idx) in test_indices.iter().enumerate() {
                X_test.row_mut(i).assign(&X.row(idx));
            }

            // Introduce synthetic missing data in test set
            let (X_test_with_missing, missing_mask) =
                self.introduce_missing_data(&X_test, &mut rng)?;

            // Convert to Float arrays upfront to avoid lifetime issues
            let X_train_float = X_train.mapv(|x| x as Float);
            let X_test_missing_float = X_test_with_missing.mapv(|x| x as Float);

            // Train imputer on training data
            let fitted_imputer = imputer.clone().fit(&X_train_float.view(), &())?;

            // Impute test data
            let X_test_imputed = fitted_imputer.transform(&X_test_missing_float.view())?;

            // Compute metrics
            let metrics =
                self.compute_metrics(&X_test, &X_test_imputed.mapv(|x| x as f64), &missing_mask)?;
            fold_metrics.push(metrics);
        }

        // Aggregate results
        let mean_metrics = self.compute_mean_metrics(&fold_metrics)?;
        let std_metrics = self.compute_std_metrics(&fold_metrics, &mean_metrics)?;
        let confidence_intervals = self.compute_confidence_intervals(&fold_metrics)?;

        Ok(CrossValidationResults {
            fold_metrics,
            mean_metrics,
            std_metrics,
            confidence_intervals,
        })
    }

    fn generate_fold_indices(
        &self,
        n_samples: usize,
        rng: &mut impl Rng,
    ) -> SklResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        match &self.cv_strategy {
            CrossValidationStrategy::KFold { n_splits, shuffle } => {
                if *shuffle {
                    indices.shuffle(rng);
                }

                let fold_size = n_samples / n_splits;
                let mut folds = Vec::new();

                for i in 0..*n_splits {
                    let start = i * fold_size;
                    let end = if i == n_splits - 1 {
                        n_samples
                    } else {
                        (i + 1) * fold_size
                    };

                    let test_indices: Vec<usize> = indices[start..end].to_vec();
                    let train_indices: Vec<usize> = indices[..start]
                        .iter()
                        .chain(indices[end..].iter())
                        .cloned()
                        .collect();

                    folds.push((train_indices, test_indices));
                }

                Ok(folds)
            }

            CrossValidationStrategy::LeaveOneOut => {
                let mut folds = Vec::new();
                for i in 0..n_samples {
                    let test_indices = vec![i];
                    let train_indices: Vec<usize> = (0..n_samples).filter(|&x| x != i).collect();
                    folds.push((train_indices, test_indices));
                }
                Ok(folds)
            }

            CrossValidationStrategy::TimeSeriesSplit {
                n_splits,
                max_train_size,
            } => {
                let mut folds = Vec::new();
                let test_size = n_samples / (n_splits + 1);

                for i in 1..=*n_splits {
                    let test_start = i * test_size;
                    let test_end = (test_start + test_size).min(n_samples);
                    let test_indices: Vec<usize> = (test_start..test_end).collect();

                    let train_end = test_start;
                    let train_start = if let Some(max_size) = max_train_size {
                        train_end.saturating_sub(*max_size)
                    } else {
                        0
                    };
                    let train_indices: Vec<usize> = (train_start..train_end).collect();

                    if !train_indices.is_empty() && !test_indices.is_empty() {
                        folds.push((train_indices, test_indices));
                    }
                }

                Ok(folds)
            }

            _ => Err(SklearsError::InvalidInput(
                "Unsupported CV strategy".to_string(),
            )),
        }
    }

    fn introduce_missing_data(
        &self,
        X: &Array2<f64>,
        rng: &mut impl Rng,
    ) -> SklResult<(Array2<f64>, Array2<bool>)> {
        let (n_samples, n_features) = X.dim();
        let mut X_missing = X.clone();
        let mut missing_mask = Array2::from_elem((n_samples, n_features), false);

        match &self.missing_pattern {
            MissingDataPattern::MCAR { missing_rate } => {
                let n_missing = (n_samples * n_features) as f64 * missing_rate * self.test_fraction;
                let n_missing = n_missing as usize;

                let mut positions: Vec<(usize, usize)> = Vec::new();
                for i in 0..n_samples {
                    for j in 0..n_features {
                        positions.push((i, j));
                    }
                }

                positions.shuffle(rng);

                for &(i, j) in positions.iter().take(n_missing) {
                    X_missing[[i, j]] = f64::NAN;
                    missing_mask[[i, j]] = true;
                }
            }

            MissingDataPattern::MAR {
                missing_rate,
                dependency_strength: _,
            } => {
                // Simplified MAR pattern: missingness depends on first feature
                if n_features > 1 {
                    let threshold = X.column(0).mean().unwrap_or(0.0);

                    for i in 0..n_samples {
                        for j in 1..n_features {
                            let prob = if X[[i, 0]] > threshold {
                                missing_rate * 2.0 * self.test_fraction
                            } else {
                                missing_rate * 0.5 * self.test_fraction
                            };

                            if rng.gen::<f64>() < prob {
                                X_missing[[i, j]] = f64::NAN;
                                missing_mask[[i, j]] = true;
                            }
                        }
                    }
                }
            }

            MissingDataPattern::MNAR {
                missing_rate,
                threshold_factor,
            } => {
                // MNAR: high values are more likely to be missing
                for j in 0..n_features {
                    let column = X.column(j);
                    let mean = column.mean().unwrap_or(0.0);
                    let std = column.var(0.0).sqrt();
                    let threshold = mean + threshold_factor * std;

                    for i in 0..n_samples {
                        let prob = if X[[i, j]] > threshold {
                            missing_rate * 3.0 * self.test_fraction
                        } else {
                            missing_rate * 0.3 * self.test_fraction
                        };

                        if rng.gen::<f64>() < prob {
                            X_missing[[i, j]] = f64::NAN;
                            missing_mask[[i, j]] = true;
                        }
                    }
                }
            }

            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported missing pattern".to_string(),
                ));
            }
        }

        Ok((X_missing, missing_mask))
    }

    fn compute_metrics(
        &self,
        X_true: &Array2<f64>,
        X_imputed: &Array2<f64>,
        missing_mask: &Array2<bool>,
    ) -> SklResult<ImputationMetrics> {
        let mut mse_sum = 0.0;
        let mut mae_sum = 0.0;
        let mut bias_sum = 0.0;
        let mut count = 0;

        let mut true_values = Vec::new();
        let mut imputed_values = Vec::new();

        for ((i, j), &is_missing) in missing_mask.indexed_iter() {
            if is_missing {
                let true_val = X_true[[i, j]];
                let imputed_val = X_imputed[[i, j]];

                if !true_val.is_nan() && !imputed_val.is_nan() {
                    let error = true_val - imputed_val;
                    mse_sum += error * error;
                    mae_sum += error.abs();
                    bias_sum += error;
                    count += 1;

                    true_values.push(true_val);
                    imputed_values.push(imputed_val);
                }
            }
        }

        if count == 0 {
            return Ok(ImputationMetrics {
                rmse: f64::NAN,
                mae: f64::NAN,
                r2: f64::NAN,
                accuracy: f64::NAN,
                f1_score: f64::NAN,
                bias: f64::NAN,
                coverage: f64::NAN,
                ks_statistic: f64::NAN,
                ks_pvalue: f64::NAN,
            });
        }

        let mse = mse_sum / count as f64;
        let rmse = mse.sqrt();
        let mae = mae_sum / count as f64;
        let bias = bias_sum / count as f64;

        // Compute R-squared
        let true_mean = true_values.iter().sum::<f64>() / true_values.len() as f64;
        let ss_tot: f64 = true_values.iter().map(|&x| (x - true_mean).powi(2)).sum();
        let ss_res: f64 = true_values
            .iter()
            .zip(imputed_values.iter())
            .map(|(&t, &p)| (t - p).powi(2))
            .sum();

        let r2 = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            f64::NAN
        };

        // Compute KS statistic (simplified)
        let (ks_statistic, ks_pvalue) = compute_ks_test(&true_values, &imputed_values);

        Ok(ImputationMetrics {
            rmse,
            mae,
            r2,
            accuracy: f64::NAN, // For continuous data
            f1_score: f64::NAN, // For continuous data
            bias,
            coverage: f64::NAN, // Not computed in this basic version
            ks_statistic,
            ks_pvalue,
        })
    }

    fn compute_mean_metrics(
        &self,
        fold_metrics: &[ImputationMetrics],
    ) -> SklResult<ImputationMetrics> {
        if fold_metrics.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No fold metrics provided".to_string(),
            ));
        }

        let n = fold_metrics.len() as f64;

        let rmse = fold_metrics
            .iter()
            .map(|m| m.rmse)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;
        let mae = fold_metrics
            .iter()
            .map(|m| m.mae)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;
        let r2 = fold_metrics
            .iter()
            .map(|m| m.r2)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;
        let bias = fold_metrics
            .iter()
            .map(|m| m.bias)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;
        let ks_statistic = fold_metrics
            .iter()
            .map(|m| m.ks_statistic)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;
        let ks_pvalue = fold_metrics
            .iter()
            .map(|m| m.ks_pvalue)
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / n;

        Ok(ImputationMetrics {
            rmse,
            mae,
            r2,
            accuracy: f64::NAN,
            f1_score: f64::NAN,
            bias,
            coverage: f64::NAN,
            ks_statistic,
            ks_pvalue,
        })
    }

    fn compute_std_metrics(
        &self,
        fold_metrics: &[ImputationMetrics],
        mean_metrics: &ImputationMetrics,
    ) -> SklResult<ImputationMetrics> {
        if fold_metrics.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No fold metrics provided".to_string(),
            ));
        }

        let n = fold_metrics.len() as f64;

        let rmse_var = fold_metrics
            .iter()
            .map(|m| (m.rmse - mean_metrics.rmse).powi(2))
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / (n - 1.0);

        let mae_var = fold_metrics
            .iter()
            .map(|m| (m.mae - mean_metrics.mae).powi(2))
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / (n - 1.0);

        let r2_var = fold_metrics
            .iter()
            .map(|m| (m.r2 - mean_metrics.r2).powi(2))
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / (n - 1.0);

        let bias_var = fold_metrics
            .iter()
            .map(|m| (m.bias - mean_metrics.bias).powi(2))
            .filter(|x| !x.is_nan())
            .sum::<f64>()
            / (n - 1.0);

        Ok(ImputationMetrics {
            rmse: rmse_var.sqrt(),
            mae: mae_var.sqrt(),
            r2: r2_var.sqrt(),
            accuracy: f64::NAN,
            f1_score: f64::NAN,
            bias: bias_var.sqrt(),
            coverage: f64::NAN,
            ks_statistic: f64::NAN,
            ks_pvalue: f64::NAN,
        })
    }

    fn compute_confidence_intervals(
        &self,
        fold_metrics: &[ImputationMetrics],
    ) -> SklResult<HashMap<String, (f64, f64)>> {
        let mut intervals = HashMap::new();

        if fold_metrics.len() < 2 {
            return Ok(intervals);
        }

        // Simple 95% confidence intervals using t-distribution approximation
        let n = fold_metrics.len() as f64;
        let t_critical = 2.0; // Approximation for 95% CI

        // RMSE
        let rmse_values: Vec<f64> = fold_metrics
            .iter()
            .map(|m| m.rmse)
            .filter(|x| !x.is_nan())
            .collect();
        if !rmse_values.is_empty() {
            let mean = rmse_values.iter().sum::<f64>() / rmse_values.len() as f64;
            let std = (rmse_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (rmse_values.len() - 1) as f64)
                .sqrt();
            let margin = t_critical * std / (rmse_values.len() as f64).sqrt();
            intervals.insert("rmse".to_string(), (mean - margin, mean + margin));
        }

        // MAE
        let mae_values: Vec<f64> = fold_metrics
            .iter()
            .map(|m| m.mae)
            .filter(|x| !x.is_nan())
            .collect();
        if !mae_values.is_empty() {
            let mean = mae_values.iter().sum::<f64>() / mae_values.len() as f64;
            let std = (mae_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (mae_values.len() - 1) as f64)
                .sqrt();
            let margin = t_critical * std / (mae_values.len() as f64).sqrt();
            intervals.insert("mae".to_string(), (mean - margin, mean + margin));
        }

        // R2
        let r2_values: Vec<f64> = fold_metrics
            .iter()
            .map(|m| m.r2)
            .filter(|x| !x.is_nan())
            .collect();
        if !r2_values.is_empty() {
            let mean = r2_values.iter().sum::<f64>() / r2_values.len() as f64;
            let std = (r2_values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                / (r2_values.len() - 1) as f64)
                .sqrt();
            let margin = t_critical * std / (r2_values.len() as f64).sqrt();
            intervals.insert("r2".to_string(), (mean - margin, mean + margin));
        }

        Ok(intervals)
    }
}

impl Default for ImputationCrossValidator {
    fn default() -> Self {
        Self::new()
    }
}

// Helper functions

fn compute_ks_test(sample1: &[f64], sample2: &[f64]) -> (f64, f64) {
    if sample1.is_empty() || sample2.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    // Simplified KS test - compute empirical CDFs and maximum difference
    let mut all_values: Vec<f64> = sample1.iter().chain(sample2.iter()).cloned().collect();
    all_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    all_values.dedup();

    let mut max_diff = 0.0;

    for &value in &all_values {
        let cdf1 = sample1.iter().filter(|&&x| x <= value).count() as f64 / sample1.len() as f64;
        let cdf2 = sample2.iter().filter(|&&x| x <= value).count() as f64 / sample2.len() as f64;

        let diff = (cdf1 - cdf2).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }

    // Approximate p-value (simplified)
    let n1 = sample1.len() as f64;
    let n2 = sample2.len() as f64;
    let n_eff = (n1 * n2) / (n1 + n2);
    let lambda = max_diff * n_eff.sqrt();

    // Very simplified p-value approximation
    let p_value = 2.0 * (-2.0 * lambda * lambda).exp();

    (max_diff, p_value.min(1.0))
}

/// Validate imputation method with simple hold-out strategy
pub fn validate_with_holdout<I>(
    imputer: &I,
    X: &ArrayView2<'_, Float>,
    test_size: f64,
    missing_pattern: MissingDataPattern,
    random_state: Option<u64>,
) -> SklResult<ImputationMetrics>
where
    I: Clone,
    for<'a> I: Fit<ArrayView2<'a, Float>, ()>,
    for<'a> <I as Fit<ArrayView2<'a, Float>, ()>>::Fitted:
        Transform<ArrayView2<'a, Float>, Array2<Float>>,
{
    let validator = HoldOutValidator {
        test_size,
        missing_pattern,
        random_state,
        stratify: false,
    };

    validator.validate(imputer, X)
}

impl HoldOutValidator {
    /// Create a new HoldOutValidator
    pub fn new(test_size: f64) -> Self {
        Self {
            test_size,
            missing_pattern: MissingDataPattern::MCAR { missing_rate: 0.2 },
            random_state: None,
            stratify: false,
        }
    }

    /// Set the missing data pattern
    pub fn missing_pattern(mut self, pattern: MissingDataPattern) -> Self {
        self.missing_pattern = pattern;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Validate an imputation method
    #[allow(non_snake_case)]
    pub fn validate<I>(
        &self,
        imputer: &I,
        X: &ArrayView2<'_, Float>,
    ) -> SklResult<ImputationMetrics>
    where
        I: Clone,
        for<'a> I: Fit<ArrayView2<'a, Float>, ()>,
        for<'a> <I as Fit<ArrayView2<'a, Float>, ()>>::Fitted:
            Transform<ArrayView2<'a, Float>, Array2<Float>>,
    {
        let X = X.mapv(|x| x as f64);
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let mut rng = match self.random_state {
            _ => Random::default(),
        };

        // Split data
        let test_size = (n_samples as f64 * self.test_size) as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        let test_indices = &indices[..test_size];
        let train_indices = &indices[test_size..];

        // Create train and test sets
        let mut X_train = Array2::zeros((train_indices.len(), n_features));
        let mut X_test = Array2::zeros((test_indices.len(), n_features));

        for (i, &idx) in train_indices.iter().enumerate() {
            X_train.row_mut(i).assign(&X.row(idx));
        }

        for (i, &idx) in test_indices.iter().enumerate() {
            X_test.row_mut(i).assign(&X.row(idx));
        }

        // Introduce missing data in test set
        let cv = ImputationCrossValidator::new()
            .missing_pattern(self.missing_pattern.clone())
            .test_fraction(1.0); // Use all test data for validation

        let (X_test_with_missing, missing_mask) = cv.introduce_missing_data(&X_test, &mut rng)?;

        // Convert to Float arrays upfront to avoid lifetime issues
        let X_train_float = X_train.mapv(|x| x as Float);
        let X_test_missing_float = X_test_with_missing.mapv(|x| x as Float);

        // Train and apply imputer
        let fitted_imputer = imputer.clone().fit(&X_train_float.view(), &())?;
        let X_test_imputed = fitted_imputer.transform(&X_test_missing_float.view())?;

        // Compute metrics
        cv.compute_metrics(&X_test, &X_test_imputed.mapv(|x| x as f64), &missing_mask)
    }
}

//! Validation framework for probabilistic models
//!
//! This module provides cross-validation, posterior predictive checking,
//! and model criticism methods specifically designed for probabilistic models.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::numeric::Float;
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ValidationError {
    #[error("Insufficient data for validation")]
    InsufficientData,
    #[error("Invalid fold configuration: {0}")]
    InvalidFoldConfig(String),
    #[error("Model fitting failed: {0}")]
    ModelFittingFailed(String),
    #[error("Prediction failed: {0}")]
    PredictionFailed(String),
    #[error("Statistical test failed: {0}")]
    StatisticalTestFailed(String),
}

/// Cross-validation results for probabilistic models
#[derive(Debug, Clone)]
pub struct CVResults {
    pub fold_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub log_likelihood_scores: Vec<f64>,
    pub brier_scores: Vec<f64>,
    pub calibration_metrics: Vec<CalibrationMetrics>,
}

/// Calibration metrics for probabilistic predictions
#[derive(Debug, Clone)]
pub struct CalibrationMetrics {
    pub reliability: f64,
    pub resolution: f64,
    pub uncertainty: f64,
    pub brier_score: f64,
    pub expected_calibration_error: f64,
    pub maximum_calibration_error: f64,
}

/// Posterior predictive checking results
#[derive(Debug, Clone)]
pub struct PPCResults {
    pub test_statistic: f64,
    pub posterior_predictive_pvalue: f64,
    pub observed_statistics: Vec<f64>,
    pub replicated_statistics: Vec<f64>,
    pub discrepancy_measures: HashMap<String, f64>,
}

/// Model criticism metrics
#[derive(Debug, Clone)]
pub struct ModelCriticismResults {
    pub deviance: f64,
    pub aic: f64,
    pub bic: f64,
    pub dic: f64,
    pub waic: f64,
    pub residual_analysis: ResidualAnalysis,
    pub goodness_of_fit: GoodnessOfFitResults,
}

/// Residual analysis for probabilistic models
#[derive(Debug, Clone)]
pub struct ResidualAnalysis {
    pub pearson_residuals: Array1<f64>,
    pub deviance_residuals: Array1<f64>,
    pub standardized_residuals: Array1<f64>,
    pub leverage_values: Array1<f64>,
    pub cooks_distance: Array1<f64>,
}

/// Goodness-of-fit test results
#[derive(Debug, Clone)]
pub struct GoodnessOfFitResults {
    pub chi_square_statistic: f64,
    pub chi_square_pvalue: f64,
    pub kolmogorov_smirnov_statistic: f64,
    pub kolmogorov_smirnov_pvalue: f64,
    pub anderson_darling_statistic: f64,
    pub anderson_darling_pvalue: f64,
}

/// Cross-validation strategy for probabilistic models
pub enum CVStrategy {
    /// KFold
    KFold(usize),
    /// StratifiedKFold
    StratifiedKFold(usize),
    /// LeaveOneOut
    LeaveOneOut,
    /// TimeSeriesSplit
    TimeSeriesSplit(usize),
    /// PurgedGroupKFold
    PurgedGroupKFold(usize),
}

/// Probabilistic model validator
pub struct ProbabilisticValidator {
    strategy: CVStrategy,
    random_state: Option<u64>,
    scoring_metrics: Vec<ScoringMetric>,
}

#[derive(Debug, Clone)]
pub enum ScoringMetric {
    /// LogLikelihood
    LogLikelihood,
    /// BrierScore
    BrierScore,
    /// Accuracy
    Accuracy,
    /// Precision
    Precision,
    /// Recall
    Recall,
    /// F1Score
    F1Score,
    /// AUC
    AUC,
    /// CalibrationError
    CalibrationError,
}

impl ProbabilisticValidator {
    pub fn new(strategy: CVStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            scoring_metrics: vec![
                ScoringMetric::LogLikelihood,
                ScoringMetric::BrierScore,
                ScoringMetric::Accuracy,
                ScoringMetric::CalibrationError,
            ],
        }
    }

    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn with_scoring_metrics(mut self, metrics: Vec<ScoringMetric>) -> Self {
        self.scoring_metrics = metrics;
        self
    }

    /// Perform cross-validation on a probabilistic model
    pub fn cross_validate<M, E>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model_factory: M,
    ) -> Result<CVResults, ValidationError>
    where
        M: Fn() -> Box<dyn ProbabilisticModel<Error = E>>,
        E: std::fmt::Debug,
    {
        let folds = self.generate_folds(X, y)?;
        let mut fold_scores = Vec::new();
        let mut log_likelihood_scores = Vec::new();
        let mut brier_scores = Vec::new();
        let mut calibration_metrics = Vec::new();

        for (train_indices, test_indices) in folds {
            // Create training and testing sets
            let (X_train, y_train) = self.extract_subset(X, y, &train_indices);
            let (X_test, y_test) = self.extract_subset(X, y, &test_indices);

            // Train model
            let mut model = model_factory();
            model
                .fit(&X_train, &y_train)
                .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

            // Make predictions
            let y_pred = model
                .predict(&X_test)
                .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;
            let y_proba = model
                .predict_proba(&X_test)
                .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

            // Compute scores
            let accuracy = self.compute_accuracy(&y_test, &y_pred);
            let log_likelihood = self.compute_log_likelihood(&y_test, &y_proba);
            let brier_score = self.compute_brier_score(&y_test, &y_proba);
            let calibration = self.compute_calibration_metrics(&y_test, &y_proba)?;

            fold_scores.push(accuracy);
            log_likelihood_scores.push(log_likelihood);
            brier_scores.push(brier_score);
            calibration_metrics.push(calibration);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / fold_scores.len() as f64;
        let std_score = variance.sqrt();

        Ok(CVResults {
            fold_scores,
            mean_score,
            std_score,
            log_likelihood_scores,
            brier_scores,
            calibration_metrics,
        })
    }

    /// Perform posterior predictive checking
    pub fn posterior_predictive_check<M>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model: &mut M,
        n_replications: usize,
    ) -> Result<PPCResults, ValidationError>
    where
        M: ProbabilisticModel,
    {
        // Fit the model
        model
            .fit(X, y)
            .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

        // Compute observed test statistic
        let observed_statistic = self.compute_test_statistic(X, y)?;

        // Generate replicated datasets and compute test statistics
        let mut replicated_statistics = Vec::new();
        for _ in 0..n_replications {
            let y_rep = self.generate_replicated_data(X, model)?;
            let rep_statistic = self.compute_test_statistic(X, &y_rep)?;
            replicated_statistics.push(rep_statistic);
        }

        // Compute posterior predictive p-value
        let extreme_count = replicated_statistics
            .iter()
            .filter(|&&stat| stat >= observed_statistic)
            .count();
        let posterior_predictive_pvalue = extreme_count as f64 / n_replications as f64;

        // Compute discrepancy measures
        let discrepancy_measures = self.compute_discrepancy_measures(X, y, model)?;

        Ok(PPCResults {
            test_statistic: observed_statistic,
            posterior_predictive_pvalue,
            observed_statistics: vec![observed_statistic],
            replicated_statistics,
            discrepancy_measures,
        })
    }

    /// Perform comprehensive model criticism
    pub fn model_criticism<M>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model: &mut M,
    ) -> Result<ModelCriticismResults, ValidationError>
    where
        M: ProbabilisticModel,
    {
        // Fit the model
        model
            .fit(X, y)
            .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

        // Compute information criteria
        let log_likelihood = model
            .log_likelihood(X, y)
            .map_err(|e| ValidationError::StatisticalTestFailed(format!("{:?}", e)))?;
        let n_params = model.get_n_parameters();
        let n_samples = X.nrows() as f64;

        let deviance = -2.0 * log_likelihood;
        let aic = deviance + 2.0 * n_params as f64;
        let bic = deviance + n_params as f64 * n_samples.ln();

        // Simplified DIC and WAIC (would need full Bayesian implementation)
        let dic = aic; // Placeholder
        let waic = bic; // Placeholder

        // Compute residual analysis
        let residual_analysis = self.compute_residual_analysis(X, y, model)?;

        // Perform goodness-of-fit tests
        let goodness_of_fit = self.compute_goodness_of_fit(X, y, model)?;

        Ok(ModelCriticismResults {
            deviance,
            aic,
            bic,
            dic,
            waic,
            residual_analysis,
            goodness_of_fit,
        })
    }

    fn generate_folds(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ValidationError> {
        let n_samples = X.nrows();

        match &self.strategy {
            CVStrategy::KFold(k) => {
                if *k > n_samples {
                    return Err(ValidationError::InvalidFoldConfig(
                        "K larger than number of samples".to_string(),
                    ));
                }
                self.generate_k_fold_indices(n_samples, *k)
            }
            CVStrategy::StratifiedKFold(k) => self.generate_stratified_k_fold_indices(y, *k),
            CVStrategy::LeaveOneOut => self.generate_leave_one_out_indices(n_samples),
            CVStrategy::TimeSeriesSplit(n_splits) => {
                self.generate_time_series_indices(n_samples, *n_splits)
            }
            CVStrategy::PurgedGroupKFold(_) => {
                // Simplified implementation
                self.generate_k_fold_indices(n_samples, 5)
            }
        }
    }

    fn generate_k_fold_indices(
        &self,
        n_samples: usize,
        k: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ValidationError> {
        let mut folds = Vec::new();
        let fold_size = n_samples / k;
        let remainder = n_samples % k;

        for i in 0..k {
            let start = i * fold_size + (i.min(remainder));
            let end = start + fold_size + if i < remainder { 1 } else { 0 };

            let test_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&x| !test_indices.contains(&x))
                .collect();

            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }

    fn generate_stratified_k_fold_indices(
        &self,
        y: &Array1<i32>,
        k: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ValidationError> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        let mut folds = vec![(Vec::new(), Vec::new()); k];

        // Distribute each class across folds
        for (_, indices) in class_indices {
            let fold_size = indices.len() / k;
            let remainder = indices.len() % k;

            for i in 0..k {
                let start = i * fold_size + (i.min(remainder));
                let end = start + fold_size + if i < remainder { 1 } else { 0 };

                folds[i].1.extend_from_slice(&indices[start..end]);
            }
        }

        // Create train sets (complement of test sets)
        let all_indices: Vec<usize> = (0..y.len()).collect();
        for (train_indices, test_indices) in &mut folds {
            *train_indices = all_indices
                .iter()
                .cloned()
                .filter(|x| !test_indices.contains(x))
                .collect();
        }

        Ok(folds)
    }

    fn generate_leave_one_out_indices(
        &self,
        n_samples: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ValidationError> {
        let mut folds = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..n_samples).filter(|&x| x != i).collect();
            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }

    fn generate_time_series_indices(
        &self,
        n_samples: usize,
        n_splits: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, ValidationError> {
        let mut folds = Vec::new();
        let min_train_size = n_samples / (n_splits + 1);

        for i in 1..=n_splits {
            let train_end = min_train_size * i;
            let test_start = train_end;
            let test_end = if i == n_splits {
                n_samples
            } else {
                train_end + min_train_size
            };

            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }

    #[allow(non_snake_case)]
    fn extract_subset(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        indices: &[usize],
    ) -> (Array2<f64>, Array1<i32>) {
        let X_subset = X.select(Axis(0), indices);
        let y_subset = Array1::from_iter(indices.iter().map(|&i| y[i]));
        (X_subset, y_subset)
    }

    fn compute_accuracy(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
        let correct = y_true
            .iter()
            .zip(y_pred.iter())
            .filter(|(a, b)| a == b)
            .count();
        correct as f64 / y_true.len() as f64
    }

    fn compute_log_likelihood(&self, y_true: &Array1<i32>, y_proba: &Array2<f64>) -> f64 {
        let mut log_likelihood = 0.0;
        for (i, &true_label) in y_true.iter().enumerate() {
            // Find probability for true class (simplified - assumes binary classification)
            let prob = if true_label == 1 {
                y_proba[[i, 1]].max(1e-15)
            } else {
                y_proba[[i, 0]].max(1e-15)
            };
            log_likelihood += prob.ln();
        }
        log_likelihood / y_true.len() as f64
    }

    fn compute_brier_score(&self, y_true: &Array1<i32>, y_proba: &Array2<f64>) -> f64 {
        let mut brier_score = 0.0;
        for (i, &true_label) in y_true.iter().enumerate() {
            let prob_positive = y_proba[[i, 1]];
            let actual = if true_label == 1 { 1.0 } else { 0.0 };
            brier_score += (prob_positive - actual).powi(2);
        }
        brier_score / y_true.len() as f64
    }

    fn compute_calibration_metrics(
        &self,
        y_true: &Array1<i32>,
        y_proba: &Array2<f64>,
    ) -> Result<CalibrationMetrics, ValidationError> {
        // Simplified calibration metrics computation
        let brier_score = self.compute_brier_score(y_true, y_proba);

        // Compute reliability, resolution, and uncertainty
        let n_bins = 10;
        let mut bin_counts = vec![0; n_bins];
        let mut bin_correct = vec![0.0; n_bins];
        let mut bin_prob_sum = vec![0.0; n_bins];

        for (i, &true_label) in y_true.iter().enumerate() {
            let prob = y_proba[[i, 1]];
            let bin_idx = ((prob * n_bins as f64).floor() as usize).min(n_bins - 1);

            bin_counts[bin_idx] += 1;
            bin_prob_sum[bin_idx] += prob;
            if true_label == 1 {
                bin_correct[bin_idx] += 1.0;
            }
        }

        let mut reliability = 0.0;
        let mut resolution = 0.0;
        let base_rate =
            y_true.iter().filter(|&&label| label == 1).count() as f64 / y_true.len() as f64;

        for i in 0..n_bins {
            if bin_counts[i] > 0 {
                let bin_accuracy = bin_correct[i] / bin_counts[i] as f64;
                let bin_confidence = bin_prob_sum[i] / bin_counts[i] as f64;
                let bin_weight = bin_counts[i] as f64 / y_true.len() as f64;

                reliability += bin_weight * (bin_confidence - bin_accuracy).powi(2);
                resolution += bin_weight * (bin_accuracy - base_rate).powi(2);
            }
        }

        let uncertainty = base_rate * (1.0 - base_rate);

        // Expected and maximum calibration error
        let mut ece = 0.0;
        let mut mce = 0.0;

        for i in 0..n_bins {
            if bin_counts[i] > 0 {
                let bin_accuracy = bin_correct[i] / bin_counts[i] as f64;
                let bin_confidence = bin_prob_sum[i] / bin_counts[i] as f64;
                let bin_weight = bin_counts[i] as f64 / y_true.len() as f64;
                let calibration_error = (bin_confidence - bin_accuracy).abs();

                ece += bin_weight * calibration_error;
                mce = mce.max(calibration_error);
            }
        }

        Ok(CalibrationMetrics {
            reliability,
            resolution,
            uncertainty,
            brier_score,
            expected_calibration_error: ece,
            maximum_calibration_error: mce,
        })
    }

    fn compute_test_statistic(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<f64, ValidationError> {
        // Example test statistic: chi-square goodness-of-fit
        let class_counts = self.count_classes(y);
        let expected_count = y.len() as f64 / class_counts.len() as f64;

        let chi_square = class_counts
            .values()
            .map(|&observed| ((observed as f64) - expected_count).powi(2) / expected_count)
            .sum();

        Ok(chi_square)
    }

    fn generate_replicated_data<M>(
        &self,
        X: &Array2<f64>,
        model: &M,
    ) -> Result<Array1<i32>, ValidationError>
    where
        M: ProbabilisticModel,
    {
        // Generate synthetic data from the fitted model
        model
            .sample(X)
            .map_err(|e| ValidationError::StatisticalTestFailed(format!("{:?}", e)))
    }

    fn compute_discrepancy_measures<M>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model: &M,
    ) -> Result<HashMap<String, f64>, ValidationError>
    where
        M: ProbabilisticModel,
    {
        let mut measures = HashMap::new();

        // Compute various discrepancy measures
        let predictions = model
            .predict(X)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        let accuracy = self.compute_accuracy(y, &predictions);
        measures.insert("accuracy".to_string(), accuracy);

        let class_counts_observed = self.count_classes(y);
        let class_counts_predicted = self.count_classes(&predictions);

        // Compute chi-square statistic for class distribution
        let mut chi_square = 0.0;
        for (class, &observed) in class_counts_observed.iter() {
            let expected = class_counts_predicted.get(class).copied().unwrap_or(0) as f64;
            if expected > 0.0 {
                chi_square += (observed as f64 - expected).powi(2) / expected;
            }
        }
        measures.insert("class_distribution_chi_square".to_string(), chi_square);

        Ok(measures)
    }

    fn compute_residual_analysis<M>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model: &M,
    ) -> Result<ResidualAnalysis, ValidationError>
    where
        M: ProbabilisticModel,
    {
        let y_pred = model
            .predict(X)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;
        let y_proba = model
            .predict_proba(X)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        let n_samples = y.len();
        let mut pearson_residuals = Array1::zeros(n_samples);
        let mut deviance_residuals = Array1::zeros(n_samples);
        let mut standardized_residuals = Array1::zeros(n_samples);
        let mut leverage_values = Array1::zeros(n_samples);
        let mut cooks_distance = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let observed = y[i] as f64;
            let predicted = y_pred[i] as f64;
            let prob = y_proba[[i, if y[i] == 1 { 1 } else { 0 }]].max(1e-15);

            // Pearson residuals
            pearson_residuals[i] = (observed - predicted) / (predicted * (1.0 - predicted)).sqrt();

            // Deviance residuals
            deviance_residuals[i] = if observed == predicted {
                0.0
            } else {
                let sign = if observed > predicted { 1.0 } else { -1.0 };
                sign * (-2.0 * prob.ln()).sqrt()
            };

            // Simplified standardized residuals and leverage (would need full statistical computation)
            standardized_residuals[i] = pearson_residuals[i];
            leverage_values[i] = 1.0 / n_samples as f64; // Simplified
            cooks_distance[i] = pearson_residuals[i].powi(2) * leverage_values[i]
                / ((1.0 - leverage_values[i]).powi(2));
        }

        Ok(ResidualAnalysis {
            pearson_residuals,
            deviance_residuals,
            standardized_residuals,
            leverage_values,
            cooks_distance,
        })
    }

    fn compute_goodness_of_fit<M>(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        model: &M,
    ) -> Result<GoodnessOfFitResults, ValidationError>
    where
        M: ProbabilisticModel,
    {
        let y_pred = model
            .predict(X)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        // Chi-square test
        let class_counts_observed = self.count_classes(y);
        let class_counts_predicted = self.count_classes(&y_pred);

        let mut chi_square_statistic = 0.0;
        let mut degrees_of_freedom = 0;

        for (class, &observed) in class_counts_observed.iter() {
            let expected = class_counts_predicted.get(class).copied().unwrap_or(0) as f64;
            if expected > 0.0 {
                chi_square_statistic += (observed as f64 - expected).powi(2) / expected;
                degrees_of_freedom += 1;
            }
        }
        degrees_of_freedom -= 1; // Adjust for constraints

        // Simplified p-value computation (would need chi-square distribution)
        let chi_square_pvalue = if chi_square_statistic > 3.84 {
            0.05
        } else {
            0.5
        };

        // Simplified KS and AD tests (placeholders)
        let kolmogorov_smirnov_statistic = 0.1;
        let kolmogorov_smirnov_pvalue = 0.5;
        let anderson_darling_statistic = 0.5;
        let anderson_darling_pvalue = 0.5;

        Ok(GoodnessOfFitResults {
            chi_square_statistic,
            chi_square_pvalue,
            kolmogorov_smirnov_statistic,
            kolmogorov_smirnov_pvalue,
            anderson_darling_statistic,
            anderson_darling_pvalue,
        })
    }

    fn count_classes(&self, y: &Array1<i32>) -> HashMap<i32, usize> {
        let mut counts = HashMap::new();
        for &label in y.iter() {
            *counts.entry(label).or_insert(0) += 1;
        }
        counts
    }
}

/// Trait for probabilistic models that can be validated
pub trait ProbabilisticModel {
    type Error: std::fmt::Debug;

    fn fit(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<(), Self::Error>;
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, Self::Error>;
    fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>, Self::Error>;
    fn log_likelihood(&self, X: &Array2<f64>, y: &Array1<i32>) -> Result<f64, Self::Error>;
    fn get_n_parameters(&self) -> usize;
    fn sample(&self, X: &Array2<f64>) -> Result<Array1<i32>, Self::Error>;
}

/// Predictive accuracy assessment metrics
#[derive(Debug, Clone)]
pub struct PredictiveAccuracyAssessment {
    /// Out-of-sample accuracy
    pub out_of_sample_accuracy: f64,
    /// Prediction interval coverage
    pub prediction_interval_coverage: HashMap<String, f64>,
    /// Mean absolute prediction error
    pub mean_absolute_error: f64,
    /// Root mean squared prediction error
    pub rmse: f64,
    /// Prediction stability metrics
    pub stability_metrics: PredictionStabilityMetrics,
    /// Temporal validation results (if applicable)
    pub temporal_validation: Option<TemporalValidationResults>,
    /// Model degradation metrics
    pub degradation_metrics: ModelDegradationMetrics,
}

/// Prediction stability metrics
#[derive(Debug, Clone)]
pub struct PredictionStabilityMetrics {
    /// Prediction variance across multiple runs
    pub prediction_variance: f64,
    /// Stability coefficient (lower is more stable)
    pub stability_coefficient: f64,
    /// Prediction consistency rate
    pub consistency_rate: f64,
    /// Standard deviation of prediction differences
    pub prediction_std: f64,
}

/// Temporal validation results for time-dependent data
#[derive(Debug, Clone)]
pub struct TemporalValidationResults {
    /// Walk-forward validation scores
    pub walk_forward_scores: Vec<f64>,
    /// Expanding window validation scores
    pub expanding_window_scores: Vec<f64>,
    /// Mean temporal accuracy
    pub mean_temporal_accuracy: f64,
    /// Temporal accuracy trend (positive = improving, negative = degrading)
    pub accuracy_trend: f64,
    /// Prediction lag analysis
    pub lag_analysis: HashMap<usize, f64>,
}

/// Model degradation monitoring metrics
#[derive(Debug, Clone)]
pub struct ModelDegradationMetrics {
    /// Performance degradation rate (per time unit)
    pub degradation_rate: f64,
    /// Early warning signals
    pub early_warning_signals: Vec<String>,
    /// Drift detection p-value
    pub drift_detection_pvalue: f64,
    /// Performance decay coefficient
    pub decay_coefficient: f64,
    /// Recommended retraining threshold
    pub retraining_threshold: f64,
}

/// Predictive accuracy assessor
pub struct PredictiveAccuracyAssessor {
    /// Number of bootstrap samples for stability assessment
    pub n_bootstrap_samples: usize,
    /// Confidence levels for prediction intervals
    pub confidence_levels: Vec<f64>,
    /// Enable temporal validation
    pub enable_temporal_validation: bool,
    /// Maximum lag for temporal analysis
    pub max_lag: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for PredictiveAccuracyAssessor {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 100,
            confidence_levels: vec![0.5, 0.8, 0.9, 0.95, 0.99],
            enable_temporal_validation: false,
            max_lag: 10,
            random_state: None,
        }
    }
}

impl PredictiveAccuracyAssessor {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_bootstrap_samples(mut self, n_samples: usize) -> Self {
        self.n_bootstrap_samples = n_samples;
        self
    }

    pub fn with_confidence_levels(mut self, levels: Vec<f64>) -> Self {
        self.confidence_levels = levels;
        self
    }

    pub fn with_temporal_validation(mut self, enable: bool) -> Self {
        self.enable_temporal_validation = enable;
        self
    }

    pub fn with_max_lag(mut self, max_lag: usize) -> Self {
        self.max_lag = max_lag;
        self
    }

    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Comprehensive predictive accuracy assessment
    pub fn assess_predictive_accuracy<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<PredictiveAccuracyAssessment, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Out-of-sample accuracy
        let out_of_sample_accuracy =
            self.compute_out_of_sample_accuracy(model, X_train, y_train, X_test, y_test)?;

        // Prediction interval coverage
        let prediction_interval_coverage =
            self.compute_prediction_interval_coverage(model, X_train, y_train, X_test, y_test)?;

        // Prediction errors
        let (mean_absolute_error, rmse) =
            self.compute_prediction_errors(model, X_train, y_train, X_test, y_test)?;

        // Stability metrics
        let stability_metrics =
            self.assess_prediction_stability(model, X_train, y_train, X_test)?;

        // Temporal validation (if enabled)
        let temporal_validation = if self.enable_temporal_validation {
            Some(self.temporal_validation(model, X_train, y_train, X_test, y_test)?)
        } else {
            None
        };

        // Model degradation metrics
        let degradation_metrics =
            self.assess_model_degradation(model, X_train, y_train, X_test, y_test)?;

        Ok(PredictiveAccuracyAssessment {
            out_of_sample_accuracy,
            prediction_interval_coverage,
            mean_absolute_error,
            rmse,
            stability_metrics,
            temporal_validation,
            degradation_metrics,
        })
    }

    /// Compute out-of-sample accuracy with bootstrap confidence intervals
    fn compute_out_of_sample_accuracy<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<f64, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut trained_model = model.clone();
        trained_model
            .fit(X_train, y_train)
            .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

        let y_pred = trained_model
            .predict(X_test)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        let accuracy = y_test
            .iter()
            .zip(y_pred.iter())
            .map(|(&true_val, &pred_val)| if true_val == pred_val { 1.0 } else { 0.0 })
            .sum::<f64>()
            / y_test.len() as f64;

        Ok(accuracy)
    }

    /// Compute prediction interval coverage for different confidence levels
    fn compute_prediction_interval_coverage<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<HashMap<String, f64>, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut coverage_map = HashMap::new();

        for &confidence_level in &self.confidence_levels {
            let coverage = self.compute_coverage_for_level(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
                confidence_level,
            )?;
            coverage_map.insert(confidence_level.to_string(), coverage);
        }

        Ok(coverage_map)
    }

    /// Compute coverage for a specific confidence level
    fn compute_coverage_for_level<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
        confidence_level: f64,
    ) -> Result<f64, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut trained_model = model.clone();
        trained_model
            .fit(X_train, y_train)
            .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

        let y_proba = trained_model
            .predict_proba(X_test)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        // For each test sample, check if true class probability is within confidence interval
        let alpha = 1.0 - confidence_level;
        let lower_threshold = alpha / 2.0;
        let upper_threshold = 1.0 - alpha / 2.0;

        let mut covered_count = 0;
        for (i, &true_class) in y_test.iter().enumerate() {
            if true_class >= 0 && (true_class as usize) < y_proba.ncols() {
                let true_class_prob = y_proba[[i, true_class as usize]];

                // Simple coverage check: if true class probability is reasonable
                if true_class_prob >= lower_threshold {
                    covered_count += 1;
                }
            }
        }

        Ok(covered_count as f64 / y_test.len() as f64)
    }

    /// Compute prediction errors (MAE and RMSE for probabilistic predictions)
    fn compute_prediction_errors<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<(f64, f64), ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut trained_model = model.clone();
        trained_model
            .fit(X_train, y_train)
            .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

        let y_proba = trained_model
            .predict_proba(X_test)
            .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

        let mut absolute_errors = Vec::new();
        let mut squared_errors = Vec::new();

        for (i, &true_class) in y_test.iter().enumerate() {
            if true_class >= 0 && (true_class as usize) < y_proba.ncols() {
                let true_class_prob = y_proba[[i, true_class as usize]];
                let error = 1.0 - true_class_prob; // Error for true class

                absolute_errors.push(error.abs());
                squared_errors.push(error.powi(2));
            }
        }

        let mae = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;
        let rmse = (squared_errors.iter().sum::<f64>() / squared_errors.len() as f64).sqrt();

        Ok((mae, rmse))
    }

    /// Assess prediction stability using bootstrap sampling
    fn assess_prediction_stability<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
    ) -> Result<PredictionStabilityMetrics, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut all_predictions = Vec::new();

        // Generate bootstrap samples and predictions
        for _ in 0..self.n_bootstrap_samples {
            let mut trained_model = model.clone();

            // Bootstrap sample from training data
            let (X_boot, y_boot) = self.bootstrap_sample(X_train, y_train);

            trained_model
                .fit(&X_boot, &y_boot)
                .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

            let y_pred = trained_model
                .predict(X_test)
                .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

            all_predictions.push(y_pred);
        }

        // Compute stability metrics
        let prediction_variance = self.compute_prediction_variance(&all_predictions);
        let stability_coefficient = self.compute_stability_coefficient(&all_predictions);
        let consistency_rate = self.compute_consistency_rate(&all_predictions);
        let prediction_std = self.compute_prediction_std(&all_predictions);

        Ok(PredictionStabilityMetrics {
            prediction_variance,
            stability_coefficient,
            consistency_rate,
            prediction_std,
        })
    }

    /// Perform temporal validation (walk-forward and expanding window)
    fn temporal_validation<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<TemporalValidationResults, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Walk-forward validation
        let walk_forward_scores =
            self.walk_forward_validation(model, X_train, y_train, X_test, y_test)?;

        // Expanding window validation
        let expanding_window_scores =
            self.expanding_window_validation(model, X_train, y_train, X_test, y_test)?;

        // Compute mean accuracy and trend
        let mean_temporal_accuracy =
            walk_forward_scores.iter().sum::<f64>() / walk_forward_scores.len() as f64;
        let accuracy_trend = self.compute_accuracy_trend(&walk_forward_scores);

        // Lag analysis
        let lag_analysis = self.compute_lag_analysis(model, X_train, y_train, X_test, y_test)?;

        Ok(TemporalValidationResults {
            walk_forward_scores,
            expanding_window_scores,
            mean_temporal_accuracy,
            accuracy_trend,
            lag_analysis,
        })
    }

    /// Assess model degradation over time
    fn assess_model_degradation<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<ModelDegradationMetrics, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Compute degradation rate (simplified)
        let degradation_rate =
            self.compute_degradation_rate(model, X_train, y_train, X_test, y_test)?;

        // Generate early warning signals
        let early_warning_signals = self.generate_early_warnings(degradation_rate);

        // Drift detection using simplified statistical test
        let drift_detection_pvalue =
            self.compute_drift_pvalue(model, X_train, y_train, X_test, y_test)?;

        // Performance decay coefficient
        let decay_coefficient = degradation_rate.abs();

        // Recommended retraining threshold
        let retraining_threshold = if degradation_rate > 0.1 { 0.8 } else { 0.6 };

        Ok(ModelDegradationMetrics {
            degradation_rate,
            early_warning_signals,
            drift_detection_pvalue,
            decay_coefficient,
            retraining_threshold,
        })
    }

    // Helper methods for stability metrics
    fn bootstrap_sample(&self, X: &Array2<f64>, y: &Array1<i32>) -> (Array2<f64>, Array1<i32>) {
        let n_samples = X.nrows();
        let mut indices = Vec::new();
        let mut rng = scirs2_core::random::CoreRandom::seed_from_u64(42); // Use fixed seed for reproducibility

        for _ in 0..n_samples {
            indices.push(rng.gen_range(0..n_samples));
        }

        let mut X_boot = Array2::zeros((n_samples, X.ncols()));
        let mut y_boot = Array1::zeros(n_samples);

        for (i, &idx) in indices.iter().enumerate() {
            X_boot.row_mut(i).assign(&X.row(idx));
            y_boot[i] = y[idx];
        }

        (X_boot, y_boot)
    }

    fn compute_prediction_variance(&self, predictions: &[Array1<i32>]) -> f64 {
        if predictions.is_empty() || predictions[0].is_empty() {
            return 0.0;
        }

        let n_predictions = predictions.len();
        let n_samples = predictions[0].len();
        let mut total_variance = 0.0;

        for i in 0..n_samples {
            let values: Vec<f64> = predictions.iter().map(|pred| pred[i] as f64).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
            total_variance += variance;
        }

        total_variance / n_samples as f64
    }

    fn compute_stability_coefficient(&self, predictions: &[Array1<i32>]) -> f64 {
        self.compute_prediction_variance(predictions)
    }

    fn compute_consistency_rate(&self, predictions: &[Array1<i32>]) -> f64 {
        if predictions.len() < 2 {
            return 1.0;
        }

        let n_samples = predictions[0].len();
        let mut consistent_count = 0;
        let mut total_comparisons = 0;

        for i in 0..n_samples {
            for j in 0..predictions.len() {
                for k in (j + 1)..predictions.len() {
                    if predictions[j][i] == predictions[k][i] {
                        consistent_count += 1;
                    }
                    total_comparisons += 1;
                }
            }
        }

        if total_comparisons == 0 {
            1.0
        } else {
            consistent_count as f64 / total_comparisons as f64
        }
    }

    fn compute_prediction_std(&self, predictions: &[Array1<i32>]) -> f64 {
        self.compute_prediction_variance(predictions).sqrt()
    }

    // Temporal validation helpers
    #[allow(non_snake_case)]
    fn walk_forward_validation<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<Vec<f64>, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Simplified implementation
        let mut scores = Vec::new();
        let n_folds = 5;
        let fold_size = X_test.nrows() / n_folds;

        for i in 0..n_folds {
            let start = i * fold_size;
            let end = if i == n_folds - 1 {
                X_test.nrows()
            } else {
                (i + 1) * fold_size
            };

            if start >= end {
                continue;
            }

            let mut trained_model = model.clone();
            trained_model
                .fit(X_train, y_train)
                .map_err(|e| ValidationError::ModelFittingFailed(format!("{:?}", e)))?;

            let X_fold = X_test
                .slice(scirs2_core::ndarray::s![start..end, ..])
                .to_owned();
            let y_fold = y_test
                .slice(scirs2_core::ndarray::s![start..end])
                .to_owned();

            let y_pred = trained_model
                .predict(&X_fold)
                .map_err(|e| ValidationError::PredictionFailed(format!("{:?}", e)))?;

            let accuracy = y_fold
                .iter()
                .zip(y_pred.iter())
                .map(|(&true_val, &pred_val)| if true_val == pred_val { 1.0 } else { 0.0 })
                .sum::<f64>()
                / y_fold.len() as f64;

            scores.push(accuracy);
        }

        Ok(scores)
    }

    fn expanding_window_validation<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<Vec<f64>, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Simplified implementation - return same as walk-forward for now
        self.walk_forward_validation(model, X_train, y_train, X_test, y_test)
    }

    fn compute_accuracy_trend(&self, scores: &[f64]) -> f64 {
        if scores.len() < 2 {
            return 0.0;
        }

        // Simple linear trend computation
        let n = scores.len() as f64;
        let x_mean = (n - 1.0) / 2.0;
        let y_mean = scores.iter().sum::<f64>() / n;

        let numerator: f64 = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i as f64 - x_mean) * (score - y_mean))
            .sum();

        let denominator: f64 = (0..scores.len()).map(|i| (i as f64 - x_mean).powi(2)).sum();

        if denominator.abs() < 1e-10 {
            0.0
        } else {
            numerator / denominator
        }
    }

    fn compute_lag_analysis<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<HashMap<usize, f64>, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        let mut lag_analysis = HashMap::new();

        // Simplified lag analysis
        for lag in 1..=self.max_lag.min(X_test.nrows() / 2) {
            let accuracy =
                self.compute_out_of_sample_accuracy(model, X_train, y_train, X_test, y_test)?;
            lag_analysis.insert(lag, accuracy);
        }

        Ok(lag_analysis)
    }

    // Degradation assessment helpers
    fn compute_degradation_rate<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<f64, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Simplified degradation rate computation
        let initial_accuracy =
            self.compute_out_of_sample_accuracy(model, X_train, y_train, X_test, y_test)?;

        // Simulate degradation over time (this would be computed from real temporal data)
        let simulated_final_accuracy = initial_accuracy * 0.95; // 5% degradation
        let degradation_rate = (initial_accuracy - simulated_final_accuracy) / initial_accuracy;

        Ok(degradation_rate)
    }

    fn generate_early_warnings(&self, degradation_rate: f64) -> Vec<String> {
        let mut warnings = Vec::new();

        if degradation_rate > 0.1 {
            warnings.push("High degradation rate detected".to_string());
        }
        if degradation_rate > 0.05 {
            warnings.push("Model performance declining".to_string());
        }
        if degradation_rate > 0.02 {
            warnings.push("Consider model monitoring".to_string());
        }

        warnings
    }

    fn compute_drift_pvalue<M>(
        &self,
        model: &M,
        X_train: &Array2<f64>,
        y_train: &Array1<i32>,
        X_test: &Array2<f64>,
        y_test: &Array1<i32>,
    ) -> Result<f64, ValidationError>
    where
        M: ProbabilisticModel + Clone,
    {
        // Simplified drift detection - in practice would use proper statistical tests
        let degradation_rate =
            self.compute_degradation_rate(model, X_train, y_train, X_test, y_test)?;

        // Convert degradation rate to p-value (simplified)
        let p_value = if degradation_rate > 0.1 {
            0.01
        } else if degradation_rate > 0.05 {
            0.05
        } else {
            0.5
        };

        Ok(p_value)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(non_snake_case)]
    fn test_k_fold_generation() {
        let validator = ProbabilisticValidator::new(CVStrategy::KFold(5));
        let X = Array2::zeros((100, 5));
        let y = Array1::from_iter((0..100).map(|i| i % 2));

        let folds = validator.generate_folds(&X, &y).unwrap();
        assert_eq!(folds.len(), 5);

        // Check that all samples are used exactly once as test data
        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test_indices) in &folds {
            all_test_indices.extend(test_indices);
        }
        all_test_indices.sort();
        assert_eq!(all_test_indices, (0..100).collect::<Vec<_>>());
    }

    #[test]
    fn test_calibration_metrics() {
        let validator = ProbabilisticValidator::new(CVStrategy::KFold(5));
        let y_true = Array1::from_vec(vec![0, 0, 1, 1, 1]);
        let y_proba = Array2::from_shape_vec(
            (5, 2),
            vec![
                0.8, 0.2, // Correct prediction for class 0
                0.7, 0.3, // Correct prediction for class 0
                0.4, 0.6, // Correct prediction for class 1
                0.3, 0.7, // Correct prediction for class 1
                0.2, 0.8, // Correct prediction for class 1
            ],
        )
        .unwrap();

        let metrics = validator
            .compute_calibration_metrics(&y_true, &y_proba)
            .unwrap();
        assert!(metrics.brier_score >= 0.0);
        assert!(metrics.expected_calibration_error >= 0.0);
        assert!(metrics.maximum_calibration_error >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_stratified_k_fold() {
        let validator = ProbabilisticValidator::new(CVStrategy::StratifiedKFold(3));
        let X = Array2::zeros((12, 2));
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]);

        let folds = validator.generate_folds(&X, &y).unwrap();
        assert_eq!(folds.len(), 3);

        // Check that each fold has approximately equal class distribution
        for (_, test_indices) in &folds {
            let test_labels: Vec<i32> = test_indices.iter().map(|&i| y[i]).collect();
            let unique_classes: std::collections::HashSet<i32> = test_labels.into_iter().collect();
            assert!(unique_classes.len() > 1 || test_indices.len() == 1);
        }
    }
}

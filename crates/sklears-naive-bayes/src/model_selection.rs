//! Model selection and validation utilities for Naive Bayes classifiers
//!
//! This module provides advanced model selection capabilities including
//! cross-validation, hyperparameter tuning, and model comparison.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use rayon::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict, PredictProba},
};
use std::collections::HashMap;

use crate::{BernoulliNB, ComplementNB, GaussianNB, MultinomialNB};

/// Information criteria for model selection
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InformationCriterion {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Corrected AIC for small samples
    AICc,
    /// Deviance Information Criterion
    DIC,
    /// Widely Applicable Information Criterion
    WAIC,
    /// Leave-One-Out Information Criterion
    LOOIC,
}

/// Cross-validation strategy
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// K-fold cross-validation
    KFold {
        k: usize,

        shuffle: bool,

        random_state: Option<u64>,
    },
    /// Stratified K-fold (maintains class proportions)
    StratifiedKFold {
        k: usize,

        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series split (for temporal data)
    TimeSeriesSplit { n_splits: usize },
}

/// Grid search parameter space
#[derive(Debug, Clone)]
pub struct ParameterGrid {
    /// Parameter combinations to try
    pub parameters: Vec<HashMap<String, ParameterValue>>,
}

/// Parameter value types for grid search
#[derive(Debug, Clone)]
pub enum ParameterValue {
    /// Float
    Float(f64),
    /// Int
    Int(i32),
    /// Bool
    Bool(bool),
    /// String
    String(String),
    /// FloatArray
    FloatArray(Vec<f64>),
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

impl ParameterGrid {
    pub fn new() -> Self {
        Self {
            parameters: Vec::new(),
        }
    }

    /// Add a parameter combination
    pub fn add_params(&mut self, params: HashMap<String, ParameterValue>) {
        self.parameters.push(params);
    }

    /// Create grid from parameter ranges
    pub fn from_ranges(param_ranges: HashMap<String, Vec<ParameterValue>>) -> Self {
        let mut grid = Self::new();

        // Generate all combinations
        let param_names: Vec<String> = param_ranges.keys().cloned().collect();
        let param_values: Vec<Vec<ParameterValue>> = param_names
            .iter()
            .map(|name| param_ranges[name].clone())
            .collect();

        fn generate_combinations(
            param_names: &[String],
            param_values: &[Vec<ParameterValue>],
            current: &mut HashMap<String, ParameterValue>,
            index: usize,
            results: &mut Vec<HashMap<String, ParameterValue>>,
        ) {
            if index == param_names.len() {
                results.push(current.clone());
                return;
            }

            for value in &param_values[index] {
                current.insert(param_names[index].clone(), value.clone());
                generate_combinations(param_names, param_values, current, index + 1, results);
            }
        }

        let mut current = HashMap::new();
        generate_combinations(
            &param_names,
            &param_values,
            &mut current,
            0,
            &mut grid.parameters,
        );

        grid
    }
}

/// Model selection results
#[derive(Debug, Clone)]
pub struct ModelSelectionResults {
    /// Best parameters found
    pub best_params: HashMap<String, ParameterValue>,
    /// Best cross-validation score
    pub best_score: f64,
    /// All parameter combinations and their scores
    pub cv_results: Vec<(HashMap<String, ParameterValue>, CVResults)>,
    /// Information criteria values
    pub information_criteria: HashMap<InformationCriterion, f64>,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CVResults {
    /// Scores for each fold
    pub test_scores: Vec<f64>,
    /// Training scores for each fold
    pub train_scores: Vec<f64>,
    /// Mean test score
    pub mean_test_score: f64,
    /// Standard deviation of test scores
    pub std_test_score: f64,
    /// Mean training score
    pub mean_train_score: f64,
    /// Standard deviation of training scores
    pub std_train_score: f64,
    /// Fit times for each fold
    pub fit_times: Vec<f64>,
    /// Score times for each fold
    pub score_times: Vec<f64>,
}

/// Scoring metrics for evaluation
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    /// Accuracy
    Accuracy,
    /// Precision (macro average)
    Precision,
    /// Recall (macro average)
    Recall,
    /// F1 score (macro average)
    F1,
    /// Log loss
    LogLoss,
    /// ROC AUC (for binary classification)
    ROCAUC,
    /// Custom scoring function
    Custom(fn(&Array1<i32>, &Array1<i32>) -> f64),
}

/// Model comparison results
#[derive(Debug, Clone)]
pub struct ModelComparison {
    /// Model names
    pub model_names: Vec<String>,
    /// Cross-validation results for each model
    pub cv_results: Vec<CVResults>,
    /// Statistical significance tests
    pub significance_tests: HashMap<(String, String), f64>,
    /// Best model name
    pub best_model: String,
    /// Bayesian model comparison results
    pub bayesian_results: Option<BayesianModelComparison>,
}

/// Bayesian model selection and comparison methods
#[derive(Debug, Clone)]
pub struct BayesianModelComparison {
    /// Model names
    pub model_names: Vec<String>,
    /// Log marginal likelihoods (model evidence)
    pub log_marginal_likelihoods: Vec<f64>,
    /// Bayes factors relative to first model
    pub bayes_factors: Vec<f64>,
    /// Posterior model probabilities
    pub posterior_model_probs: Vec<f64>,
    /// WAIC values for each model
    pub waic_values: Vec<f64>,
    /// LOOIC values for each model  
    pub looic_values: Vec<f64>,
    /// Best model according to Bayesian criteria
    pub best_bayesian_model: String,
}

/// Bayesian model selector for principled model comparison
#[derive(Debug)]
pub struct BayesianModelSelector {
    /// Prior probabilities over models (uniform if None)
    pub model_priors: Option<Vec<f64>>,
    /// Number of samples for marginal likelihood estimation
    pub n_samples: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to compute WAIC
    pub compute_waic: bool,
    /// Whether to compute LOOIC
    pub compute_looic: bool,
}

impl Default for BayesianModelSelector {
    fn default() -> Self {
        Self {
            model_priors: None,
            n_samples: 1000,
            random_state: None,
            compute_waic: true,
            compute_looic: true,
        }
    }
}

impl BayesianModelSelector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn model_priors(mut self, priors: Vec<f64>) -> Self {
        self.model_priors = Some(priors);
        self
    }

    pub fn n_samples(mut self, n_samples: usize) -> Self {
        self.n_samples = n_samples;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    pub fn with_waic(mut self, compute_waic: bool) -> Self {
        self.compute_waic = compute_waic;
        self
    }

    pub fn with_looic(mut self, compute_looic: bool) -> Self {
        self.compute_looic = compute_looic;
        self
    }

    /// Compute log marginal likelihood using harmonic mean estimator
    /// Note: This is a simplified implementation. In practice, more sophisticated
    /// methods like bridge sampling or nested sampling would be preferred.
    pub fn log_marginal_likelihood<M>(
        &self,
        model: &M,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<f64>
    where
        M: Clone + Send + Sync,
        M: Fit<Array2<f64>, Array1<i32>>,
        <M as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
    {
        // Fit the model
        let fitted_model = model.clone().fit(x, y)?;

        // Compute log likelihood
        let proba = fitted_model.predict_proba(x)?;
        let mut log_likelihood = 0.0;

        for (i, &true_class) in y.iter().enumerate() {
            // Find the probability for the true class
            let class_prob = if true_class >= 0 && (true_class as usize) < proba.ncols() {
                proba[[i, true_class as usize]]
            } else {
                1e-10 // Small probability for unseen classes
            };
            log_likelihood += class_prob.max(1e-10).ln();
        }

        // For Naive Bayes, we can compute the marginal likelihood analytically
        // This is a simplified approximation
        let n_params = self.estimate_n_parameters(x, y);
        let bic_penalty = n_params as f64 * (y.len() as f64).ln() / 2.0;
        let log_marginal_likelihood = log_likelihood - bic_penalty;

        Ok(log_marginal_likelihood)
    }

    /// Estimate number of parameters for the model
    fn estimate_n_parameters(&self, x: &Array2<f64>, y: &Array1<i32>) -> usize {
        let n_features = x.ncols();
        let n_classes = y.iter().collect::<std::collections::HashSet<_>>().len();

        // For Gaussian NB: mean and variance for each feature-class combination + class priors
        n_classes * n_features * 2 + n_classes - 1
    }

    /// Compute Bayes factors relative to the first model
    pub fn compute_bayes_factors(&self, log_marginal_likelihoods: &[f64]) -> Vec<f64> {
        if log_marginal_likelihoods.is_empty() {
            return Vec::new();
        }

        let reference_lml = log_marginal_likelihoods[0];
        log_marginal_likelihoods
            .iter()
            .map(|&lml| (lml - reference_lml).exp())
            .collect()
    }

    /// Compute posterior model probabilities
    pub fn compute_posterior_model_probs(&self, log_marginal_likelihoods: &[f64]) -> Vec<f64> {
        if log_marginal_likelihoods.is_empty() {
            return Vec::new();
        }

        let n_models = log_marginal_likelihoods.len();

        // Use uniform priors if not specified
        let priors = self
            .model_priors
            .clone()
            .unwrap_or_else(|| vec![1.0 / n_models as f64; n_models]);

        // Compute log posterior unnormalized
        let log_posterior_unnorm: Vec<f64> = log_marginal_likelihoods
            .iter()
            .zip(priors.iter())
            .map(|(&lml, &prior)| lml + prior.ln())
            .collect();

        // Normalize using log-sum-exp trick
        let max_log_post = log_posterior_unnorm
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);

        let sum_exp: f64 = log_posterior_unnorm
            .iter()
            .map(|&x| (x - max_log_post).exp())
            .sum();

        let log_normalizer = max_log_post + sum_exp.ln();

        log_posterior_unnorm
            .iter()
            .map(|&x| (x - log_normalizer).exp())
            .collect()
    }

    /// Compute WAIC (Widely Applicable Information Criterion)
    pub fn compute_waic<M>(&self, model: &M, x: &Array2<f64>, y: &Array1<i32>) -> Result<f64>
    where
        M: Clone + Send + Sync,
        M: Fit<Array2<f64>, Array1<i32>>,
        <M as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
    {
        // For simplicity, use a single model fit
        // In practice, WAIC requires multiple posterior samples
        let fitted_model = model.clone().fit(x, y)?;
        let proba = fitted_model.predict_proba(x)?;

        let mut lppd = 0.0; // Log pointwise predictive density
        let mut p_waic = 0.0; // Effective number of parameters

        for (i, &true_class) in y.iter().enumerate() {
            let class_prob = if true_class >= 0 && (true_class as usize) < proba.ncols() {
                proba[[i, true_class as usize]].max(1e-10)
            } else {
                1e-10
            };

            lppd += class_prob.ln();
            // Simplified variance calculation (would need multiple posterior samples)
            p_waic += 0.1; // Placeholder - in practice compute variance of log probabilities
        }

        let waic = -2.0 * (lppd - p_waic);
        Ok(waic)
    }

    /// Compute LOOIC (Leave-One-Out Information Criterion)
    pub fn compute_looic<M>(&self, model: &M, x: &Array2<f64>, y: &Array1<i32>) -> Result<f64>
    where
        M: Clone + Send + Sync,
        M: Fit<Array2<f64>, Array1<i32>>,
        <M as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
    {
        let n_samples = x.nrows();
        let mut log_lik_loo = 0.0;

        for i in 0..n_samples {
            // Create training set without sample i
            let train_indices: Vec<usize> = (0..n_samples).filter(|&idx| idx != i).collect();
            let x_train = x.select(Axis(0), &train_indices);
            let y_train = y.select(Axis(0), &train_indices);

            // Fit model and predict left-out sample
            let fitted_model = model.clone().fit(&x_train, &y_train)?;
            let x_test = x.slice(scirs2_core::ndarray::s![i..i + 1, ..]).to_owned();
            let proba = fitted_model.predict_proba(&x_test)?;

            let true_class = y[i];
            let class_prob = if true_class >= 0 && (true_class as usize) < proba.ncols() {
                proba[[0, true_class as usize]].max(1e-10)
            } else {
                1e-10
            };

            log_lik_loo += class_prob.ln();
        }

        let looic = -2.0 * log_lik_loo;
        Ok(looic)
    }

    /// Perform comprehensive Bayesian model comparison
    pub fn compare_models_bayesian<M>(
        &self,
        models: Vec<(&str, M)>,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<BayesianModelComparison>
    where
        M: Clone + Send + Sync,
        M: Fit<Array2<f64>, Array1<i32>>,
        <M as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
    {
        let mut model_names = Vec::new();
        let mut log_marginal_likelihoods = Vec::new();
        let mut waic_values = Vec::new();
        let mut looic_values = Vec::new();

        for (name, model) in models {
            model_names.push(name.to_string());

            // Compute log marginal likelihood
            let lml = self.log_marginal_likelihood(&model, x, y)?;
            log_marginal_likelihoods.push(lml);

            // Compute WAIC if requested
            if self.compute_waic {
                let waic = self.compute_waic(&model, x, y)?;
                waic_values.push(waic);
            }

            // Compute LOOIC if requested
            if self.compute_looic {
                let looic = self.compute_looic(&model, x, y)?;
                looic_values.push(looic);
            }
        }

        // Compute Bayes factors
        let bayes_factors = self.compute_bayes_factors(&log_marginal_likelihoods);

        // Compute posterior model probabilities
        let posterior_model_probs = self.compute_posterior_model_probs(&log_marginal_likelihoods);

        // Find best model based on highest posterior probability
        let best_idx = posterior_model_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best_bayesian_model = model_names[best_idx].clone();

        Ok(BayesianModelComparison {
            model_names,
            log_marginal_likelihoods,
            bayes_factors,
            posterior_model_probs,
            waic_values: if self.compute_waic {
                waic_values
            } else {
                Vec::new()
            },
            looic_values: if self.compute_looic {
                looic_values
            } else {
                Vec::new()
            },
            best_bayesian_model,
        })
    }
}

/// Advanced model selector for Naive Bayes classifiers
#[derive(Debug)]
pub struct NaiveBayesModelSelector {
    /// Scoring metric for evaluation
    pub scoring: ScoringMetric,
    /// Cross-validation strategy
    pub cv_strategy: CVStrategy,
    /// Whether to refit on full dataset
    pub refit: bool,
    /// Number of parallel jobs
    pub n_jobs: Option<usize>,
    /// Verbose output
    pub verbose: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for NaiveBayesModelSelector {
    fn default() -> Self {
        Self {
            scoring: ScoringMetric::Accuracy,
            cv_strategy: CVStrategy::KFold {
                k: 5,
                shuffle: true,
                random_state: None,
            },
            refit: true,
            n_jobs: None,
            verbose: false,
            random_state: None,
        }
    }
}

impl NaiveBayesModelSelector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn scoring(mut self, scoring: ScoringMetric) -> Self {
        self.scoring = scoring;
        self
    }

    pub fn cv_strategy(mut self, cv_strategy: CVStrategy) -> Self {
        self.cv_strategy = cv_strategy;
        self
    }

    pub fn n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Generate cross-validation splits
    fn generate_cv_splits(
        &self,
        n_samples: usize,
        y: &Array1<i32>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        match &self.cv_strategy {
            CVStrategy::KFold {
                k,
                shuffle,
                random_state,
            } => self.kfold_splits(n_samples, *k, *shuffle, *random_state),
            CVStrategy::StratifiedKFold {
                k,
                shuffle,
                random_state,
            } => self.stratified_kfold_splits(y, *k, *shuffle, *random_state),
            CVStrategy::LeaveOneOut => self.leave_one_out_splits(n_samples),
            CVStrategy::TimeSeriesSplit { n_splits } => {
                self.time_series_splits(n_samples, *n_splits)
            }
        }
    }

    /// K-fold cross-validation splits
    fn kfold_splits(
        &self,
        n_samples: usize,
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            let mut rng = match random_state {
                Some(seed) => scirs2_core::random::CoreRandom::seed_from_u64(seed),
                None => {
                    scirs2_core::random::CoreRandom::from_rng(&mut scirs2_core::random::thread_rng())
                }
            };

            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        let fold_size = n_samples / k;
        let mut splits = Vec::new();

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices = indices[start..end].to_vec();
            let train_indices = [&indices[..start], &indices[end..]].concat();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Stratified K-fold splits (maintains class proportions)
    fn stratified_kfold_splits(
        &self,
        y: &Array1<i32>,
        k: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        // Shuffle indices within each class if requested
        if shuffle {
            let mut rng = match random_state {
                Some(seed) => scirs2_core::random::CoreRandom::seed_from_u64(seed),
                None => {
                    scirs2_core::random::CoreRandom::from_rng(&mut scirs2_core::random::thread_rng())
                }
            };

            for indices in class_indices.values_mut() {
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
            }
        }

        let mut splits = Vec::new();

        for fold in 0..k {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            // For each class, split indices into folds
            for indices in class_indices.values() {
                let fold_size = indices.len() / k;
                let start = fold * fold_size;
                let end = if fold == k - 1 {
                    indices.len()
                } else {
                    (fold + 1) * fold_size
                };

                test_indices.extend_from_slice(&indices[start..end]);
                train_indices.extend_from_slice(&indices[..start]);
                train_indices.extend_from_slice(&indices[end..]);
            }

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Leave-one-out cross-validation splits
    fn leave_one_out_splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..n_samples).filter(|&x| x != i).collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Time series cross-validation splits
    fn time_series_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        let min_train_size = n_samples / (n_splits + 1);

        for i in 0..n_splits {
            let train_end = min_train_size * (i + 2);
            let test_start = train_end;
            let test_end = if i == n_splits - 1 {
                n_samples
            } else {
                min_train_size * (i + 3)
            };

            if test_end > n_samples {
                break;
            }

            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Compute scoring metric
    fn compute_score(&self, y_true: &Array1<i32>, y_pred: &Array1<i32>) -> f64 {
        match &self.scoring {
            ScoringMetric::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_val, &pred_val)| true_val == pred_val)
                    .count();
                correct as f64 / y_true.len() as f64
            }
            ScoringMetric::Custom(func) => func(y_true, y_pred),
            _ => {
                // For other metrics, implement or return accuracy as fallback
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .filter(|(&true_val, &pred_val)| true_val == pred_val)
                    .count();
                correct as f64 / y_true.len() as f64
            }
        }
    }

    /// Perform cross-validation for a single model
    pub fn cross_validate<M>(&self, model: M, x: &Array2<f64>, y: &Array1<i32>) -> Result<CVResults>
    where
        M: Clone + Send + Sync,
        M: Fit<Array2<f64>, Array1<i32>>,
        <M as Fit<Array2<f64>, Array1<i32>>>::Fitted: Predict<Array2<f64>, Array1<i32>>,
    {
        let splits = self.generate_cv_splits(x.nrows(), y)?;
        let mut test_scores = Vec::new();
        let mut train_scores = Vec::new();
        let mut fit_times = Vec::new();
        let mut score_times = Vec::new();

        for (train_idx, test_idx) in splits {
            let start_time = std::time::Instant::now();

            // Create training and test sets
            let x_train = x.select(Axis(0), &train_idx);
            let y_train = y.select(Axis(0), &train_idx);
            let x_test = x.select(Axis(0), &test_idx);
            let y_test = y.select(Axis(0), &test_idx);

            // Fit model
            let fitted_model = model.clone().fit(&x_train, &y_train)?;
            let fit_time = start_time.elapsed().as_secs_f64();
            fit_times.push(fit_time);

            let score_start = std::time::Instant::now();

            // Predict on test set
            let y_pred_test = fitted_model.predict(&x_test)?;
            let test_score = self.compute_score(&y_test, &y_pred_test);
            test_scores.push(test_score);

            // Predict on training set
            let y_pred_train = fitted_model.predict(&x_train)?;
            let train_score = self.compute_score(&y_train, &y_pred_train);
            train_scores.push(train_score);

            let score_time = score_start.elapsed().as_secs_f64();
            score_times.push(score_time);
        }

        let mean_test_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let std_test_score = {
            let variance = test_scores
                .iter()
                .map(|score| (score - mean_test_score).powi(2))
                .sum::<f64>()
                / test_scores.len() as f64;
            variance.sqrt()
        };

        let mean_train_score = train_scores.iter().sum::<f64>() / train_scores.len() as f64;
        let std_train_score = {
            let variance = train_scores
                .iter()
                .map(|score| (score - mean_train_score).powi(2))
                .sum::<f64>()
                / train_scores.len() as f64;
            variance.sqrt()
        };

        Ok(CVResults {
            test_scores,
            train_scores,
            mean_test_score,
            std_test_score,
            mean_train_score,
            std_train_score,
            fit_times,
            score_times,
        })
    }

    /// Compare multiple Naive Bayes models
    pub fn compare_models(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<ModelComparison> {
        let models = vec![
            (
                "GaussianNB",
                Box::new(GaussianNB::new()) as Box<dyn std::any::Any + Send + Sync>,
            ),
            (
                "MultinomialNB",
                Box::new(MultinomialNB::new()) as Box<dyn std::any::Any + Send + Sync>,
            ),
            (
                "BernoulliNB",
                Box::new(BernoulliNB::new()) as Box<dyn std::any::Any + Send + Sync>,
            ),
            (
                "ComplementNB",
                Box::new(ComplementNB::new()) as Box<dyn std::any::Any + Send + Sync>,
            ),
        ];

        let mut model_names = Vec::new();
        let mut cv_results = Vec::new();

        for (name, _model) in models {
            model_names.push(name.to_string());

            // For now, use GaussianNB as a placeholder
            // In a real implementation, we would need trait objects or enums
            let model = GaussianNB::new();
            let results = self.cross_validate(model, x, y)?;
            cv_results.push(results);
        }

        // Find best model
        let best_idx = cv_results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.mean_test_score.partial_cmp(&b.mean_test_score).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let best_model = model_names[best_idx].clone();

        // Placeholder for significance tests
        let significance_tests = HashMap::new();

        Ok(ModelComparison {
            model_names,
            cv_results,
            significance_tests,
            best_model,
            bayesian_results: None,
        })
    }

    /// Compute information criteria for model selection
    pub fn compute_information_criteria(
        &self,
        log_likelihood: f64,
        n_params: usize,
        n_samples: usize,
    ) -> HashMap<InformationCriterion, f64> {
        let mut criteria = HashMap::new();

        // AIC = -2 * log_likelihood + 2 * n_params
        let aic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        criteria.insert(InformationCriterion::AIC, aic);

        // BIC = -2 * log_likelihood + n_params * log(n_samples)
        let bic = -2.0 * log_likelihood + n_params as f64 * (n_samples as f64).ln();
        criteria.insert(InformationCriterion::BIC, bic);

        // AICc = AIC + 2 * n_params * (n_params + 1) / (n_samples - n_params - 1)
        if n_samples > n_params + 1 {
            let aicc = aic
                + 2.0 * n_params as f64 * (n_params as f64 + 1.0)
                    / (n_samples as f64 - n_params as f64 - 1.0);
            criteria.insert(InformationCriterion::AICc, aicc);
        }

        // DIC (simplified - would need posterior samples for full implementation)
        let dic = -2.0 * log_likelihood + 2.0 * n_params as f64;
        criteria.insert(InformationCriterion::DIC, dic);

        criteria
    }

    /// Enhanced model comparison with Bayesian methods
    pub fn compare_models_enhanced(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<ModelComparison> {
        // Traditional frequentist comparison
        let mut traditional_comparison = self.compare_models(x, y)?;

        // Add Bayesian comparison if requested
        let bayesian_selector = BayesianModelSelector::new();

        // For now, just use GaussianNB for Bayesian comparison
        // In a full implementation, we would need a unified trait object approach
        let models = vec![("GaussianNB", GaussianNB::new())];

        match bayesian_selector.compare_models_bayesian(models, x, y) {
            Ok(bayesian_results) => {
                traditional_comparison.bayesian_results = Some(bayesian_results);
            }
            Err(_) => {
                // If Bayesian comparison fails, continue with traditional results only
                traditional_comparison.bayesian_results = None;
            }
        }

        Ok(traditional_comparison)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parameter_grid() {
        let mut param_ranges = HashMap::new();
        param_ranges.insert(
            "alpha".to_string(),
            vec![
                ParameterValue::Float(0.1),
                ParameterValue::Float(1.0),
                ParameterValue::Float(10.0),
            ],
        );
        param_ranges.insert(
            "fit_prior".to_string(),
            vec![ParameterValue::Bool(true), ParameterValue::Bool(false)],
        );

        let grid = ParameterGrid::from_ranges(param_ranges);
        assert_eq!(grid.parameters.len(), 6); // 3 * 2 combinations
    }

    #[test]
    fn test_kfold_splits() {
        let selector = NaiveBayesModelSelector::new();
        let y = Array1::from_vec(vec![0, 1, 0, 1, 0, 1, 0, 1]);
        let splits = selector.kfold_splits(8, 4, false, None).unwrap();

        assert_eq!(splits.len(), 4);
        for (train, test) in splits {
            assert_eq!(train.len() + test.len(), 8);
        }
    }

    #[test]
    fn test_cross_validate() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0]).unwrap();
        let y = Array1::from_vec(vec![0, 1, 1, 0]);

        let selector = NaiveBayesModelSelector::new().cv_strategy(CVStrategy::KFold {
            k: 2,
            shuffle: false,
            random_state: Some(42),
        });

        let model = GaussianNB::new();
        let results = selector.cross_validate(model, &x, &y).unwrap();

        assert_eq!(results.test_scores.len(), 2);
        assert!(results.mean_test_score >= 0.0 && results.mean_test_score <= 1.0);
    }

    #[test]
    fn test_information_criteria() {
        let selector = NaiveBayesModelSelector::new();
        let criteria = selector.compute_information_criteria(-100.0, 5, 100);

        assert!(criteria.contains_key(&InformationCriterion::AIC));
        assert!(criteria.contains_key(&InformationCriterion::BIC));
        assert!(criteria.contains_key(&InformationCriterion::AICc));
        assert!(criteria.contains_key(&InformationCriterion::DIC));
    }

    #[test]
    fn test_bayesian_model_selector() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0,
                4.0, 3.0, 3.0, 1.0, 4.0, 1.0, 3.0, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1, 1, 0, 1, 0, 1, 0]);

        let bayesian_selector = BayesianModelSelector::new();

        // Test log marginal likelihood computation
        let model = GaussianNB::new();
        let lml = bayesian_selector.log_marginal_likelihood(&model, &x, &y);
        assert!(lml.is_ok());

        // Test WAIC computation
        let waic = bayesian_selector.compute_waic(&model, &x, &y);
        assert!(waic.is_ok());
    }

    #[test]
    fn test_bayes_factors() {
        let bayesian_selector = BayesianModelSelector::new();
        let log_marginal_likelihoods = vec![-10.0, -12.0, -8.0];
        let bayes_factors = bayesian_selector.compute_bayes_factors(&log_marginal_likelihoods);

        assert_eq!(bayes_factors.len(), 3);
        assert_abs_diff_eq!(bayes_factors[0], 1.0, epsilon = 1e-10); // Reference model
        assert!(bayes_factors[2] > bayes_factors[1]); // Better model has higher BF
    }

    #[test]
    fn test_posterior_model_probs() {
        let bayesian_selector = BayesianModelSelector::new();
        let log_marginal_likelihoods = vec![-10.0, -12.0, -8.0];
        let posterior_probs =
            bayesian_selector.compute_posterior_model_probs(&log_marginal_likelihoods);

        assert_eq!(posterior_probs.len(), 3);
        let sum: f64 = posterior_probs.iter().sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10); // Should sum to 1

        // Model with highest log marginal likelihood should have highest posterior prob
        let max_idx = posterior_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        assert_eq!(max_idx, 2); // Third model has highest log marginal likelihood
    }

    #[test]
    fn test_bayesian_model_comparison() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![1.0, 2.0, 2.0, 1.0, 3.0, 3.0, 1.0, 1.0, 4.0, 2.0, 2.0, 4.0],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1, 1, 0, 1, 0]);

        let bayesian_selector = BayesianModelSelector::new()
            .with_waic(false) // Disable WAIC for faster testing
            .with_looic(false); // Disable LOOIC for faster testing

        let models = vec![("GaussianNB", GaussianNB::new())];

        let comparison = bayesian_selector.compare_models_bayesian(models, &x, &y);
        assert!(comparison.is_ok());

        let results = comparison.unwrap();
        assert_eq!(results.model_names.len(), 1);
        assert_eq!(results.log_marginal_likelihoods.len(), 1);
        assert_eq!(results.bayes_factors.len(), 1);
        assert_eq!(results.posterior_model_probs.len(), 1);
        assert!(!results.best_bayesian_model.is_empty());
    }

    #[test]
    fn test_nested_model_comparison() {
        let x = Array2::from_shape_vec(
            (8, 3),
            vec![
                1.0, 2.0, 3.0, 2.0, 1.0, 4.0, 3.0, 3.0, 1.0, 1.0, 1.0, 2.0, 4.0, 2.0, 1.0, 2.0,
                4.0, 3.0, 3.0, 1.0, 4.0, 1.0, 3.0, 2.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1, 1, 0, 1, 0, 1, 0]);

        let selector = NaiveBayesModelSelector::new();

        // Compare simple vs complex models
        let simple_model = GaussianNB::new();
        let complex_model = MultinomialNB::new();

        let nested_results = selector
            .nested_model_comparison(&simple_model, &complex_model, &x, &y, 0.05)
            .unwrap();

        assert!(!nested_results.simple_model_name.is_empty());
        assert!(!nested_results.complex_model_name.is_empty());
        // Likelihood ratio can be negative if simple model is better
        assert!(nested_results.likelihood_ratio.is_finite());
        assert!(nested_results.p_value >= 0.0 && nested_results.p_value <= 1.0);
    }
}

/// Nested model comparison results
#[derive(Debug, Clone)]
pub struct NestedModelComparison {
    /// Name of the simpler (nested) model
    pub simple_model_name: String,
    /// Name of the more complex model
    pub complex_model_name: String,
    /// Log-likelihood of the simple model
    pub simple_log_likelihood: f64,
    /// Log-likelihood of the complex model
    pub complex_log_likelihood: f64,
    /// Likelihood ratio test statistic
    pub likelihood_ratio: f64,
    /// Degrees of freedom difference
    pub df_difference: usize,
    /// P-value from likelihood ratio test
    pub p_value: f64,
    /// Whether the complex model is significantly better
    pub is_significant: bool,
    /// Information criteria comparison
    pub information_criteria_comparison: HashMap<InformationCriterion, (f64, f64)>,
    /// Recommendation
    pub recommendation: String,
}

/// Nested model validation metrics
#[derive(Debug, Clone)]
pub struct NestedModelValidation {
    /// Cross-validation results for simple model
    pub simple_cv_results: CVResults,
    /// Cross-validation results for complex model
    pub complex_cv_results: CVResults,
    /// Statistical significance of performance difference
    pub performance_significance: f64,
    /// Effect size of performance difference
    pub effect_size: f64,
    /// Confidence interval for performance difference
    pub confidence_interval: (f64, f64),
}

impl NaiveBayesModelSelector {
    /// Compare nested models using likelihood ratio test and information criteria
    pub fn nested_model_comparison<S, C>(
        &self,
        simple_model: &S,
        complex_model: &C,
        x: &Array2<f64>,
        y: &Array1<i32>,
        alpha: f64,
    ) -> Result<NestedModelComparison>
    where
        S: Clone + Send + Sync,
        S: Fit<Array2<f64>, Array1<i32>>,
        <S as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
        C: Clone + Send + Sync,
        C: Fit<Array2<f64>, Array1<i32>>,
        <C as Fit<Array2<f64>, Array1<i32>>>::Fitted: PredictProba<Array2<f64>, Array2<f64>>,
    {
        // Fit both models
        let simple_fitted = simple_model.clone().fit(x, y)?;
        let complex_fitted = complex_model.clone().fit(x, y)?;

        // Compute log-likelihoods
        let simple_log_likelihood = self.compute_log_likelihood(&simple_fitted, x, y)?;
        let complex_log_likelihood = self.compute_log_likelihood(&complex_fitted, x, y)?;

        // Estimate parameters for both models
        let simple_n_params = self.estimate_model_parameters(x, y, "simple");
        let complex_n_params = self.estimate_model_parameters(x, y, "complex");

        // Ensure complex model has more parameters
        if complex_n_params <= simple_n_params {
            return Err(SklearsError::InvalidParameter {
                name: "model_comparison".to_string(),
                reason: "Complex model must have more parameters than simple model".to_string(),
            });
        }

        let df_difference = complex_n_params - simple_n_params;

        // Likelihood ratio test
        let likelihood_ratio = 2.0 * (complex_log_likelihood - simple_log_likelihood);
        let p_value = self.chi_squared_p_value(likelihood_ratio, df_difference);
        let is_significant = p_value < alpha;

        // Compute information criteria for both models
        let simple_criteria =
            self.compute_information_criteria(simple_log_likelihood, simple_n_params, y.len());
        let complex_criteria =
            self.compute_information_criteria(complex_log_likelihood, complex_n_params, y.len());

        let mut information_criteria_comparison = HashMap::new();
        for criterion in &[
            InformationCriterion::AIC,
            InformationCriterion::BIC,
            InformationCriterion::AICc,
        ] {
            if let (Some(&simple_ic), Some(&complex_ic)) = (
                simple_criteria.get(criterion),
                complex_criteria.get(criterion),
            ) {
                information_criteria_comparison.insert(criterion.clone(), (simple_ic, complex_ic));
            }
        }

        // Generate recommendation
        let recommendation = self.generate_nested_model_recommendation(
            likelihood_ratio,
            p_value,
            &information_criteria_comparison,
            alpha,
        );

        Ok(NestedModelComparison {
            simple_model_name: "Simple Model".to_string(),
            complex_model_name: "Complex Model".to_string(),
            simple_log_likelihood,
            complex_log_likelihood,
            likelihood_ratio,
            df_difference,
            p_value,
            is_significant,
            information_criteria_comparison,
            recommendation,
        })
    }

    /// Perform nested model validation using cross-validation
    pub fn nested_model_validation<S, C>(
        &self,
        simple_model: &S,
        complex_model: &C,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<NestedModelValidation>
    where
        S: Clone + Send + Sync,
        S: Fit<Array2<f64>, Array1<i32>>,
        <S as Fit<Array2<f64>, Array1<i32>>>::Fitted:
            PredictProba<Array2<f64>, Array2<f64>> + Predict<Array2<f64>, Array1<i32>>,
        C: Clone + Send + Sync,
        C: Fit<Array2<f64>, Array1<i32>>,
        <C as Fit<Array2<f64>, Array1<i32>>>::Fitted:
            PredictProba<Array2<f64>, Array2<f64>> + Predict<Array2<f64>, Array1<i32>>,
    {
        // Perform cross-validation for both models
        let simple_cv_results = self.cross_validate(simple_model.clone(), x, y)?;
        let complex_cv_results = self.cross_validate(complex_model.clone(), x, y)?;

        // Compute statistical significance of performance difference
        let performance_significance = self.paired_t_test(
            &simple_cv_results.test_scores,
            &complex_cv_results.test_scores,
        )?;

        // Compute effect size (Cohen's d)
        let effect_size = self.compute_cohens_d(
            &simple_cv_results.test_scores,
            &complex_cv_results.test_scores,
        );

        // Compute confidence interval for difference
        let confidence_interval = self.compute_confidence_interval(
            &simple_cv_results.test_scores,
            &complex_cv_results.test_scores,
            0.95,
        );

        Ok(NestedModelValidation {
            simple_cv_results,
            complex_cv_results,
            performance_significance,
            effect_size,
            confidence_interval,
        })
    }

    /// Compute log-likelihood for a fitted model
    fn compute_log_likelihood<F>(
        &self,
        fitted_model: &F,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<f64>
    where
        F: PredictProba<Array2<f64>, Array2<f64>>,
    {
        let proba = fitted_model.predict_proba(x)?;
        let mut log_likelihood = 0.0;

        for (i, &true_class) in y.iter().enumerate() {
            let class_prob = if true_class >= 0 && (true_class as usize) < proba.ncols() {
                proba[[i, true_class as usize]]
            } else {
                1e-10
            };
            log_likelihood += class_prob.max(1e-10).ln();
        }

        Ok(log_likelihood)
    }

    /// Estimate number of parameters for different model types
    fn estimate_model_parameters(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        model_type: &str,
    ) -> usize {
        let n_features = x.ncols();
        let n_classes = y.iter().collect::<std::collections::HashSet<_>>().len();

        match model_type {
            "simple" => {
                // Simpler parameterization (e.g., shared variance)
                n_classes * n_features + n_classes - 1
            }
            "complex" => {
                // More complex parameterization (e.g., separate variance per feature-class)
                n_classes * n_features * 2 + n_classes - 1
            }
            _ => n_classes * n_features + n_classes - 1,
        }
    }

    /// Compute Chi-squared p-value (simplified approximation)
    fn chi_squared_p_value(&self, chi_squared_stat: f64, df: usize) -> f64 {
        if chi_squared_stat <= 0.0 || df == 0 {
            return 1.0;
        }

        // Simplified approximation using normal distribution for large df
        if df > 30 {
            let z =
                ((2.0 * chi_squared_stat).sqrt() - (2.0 * df as f64 - 1.0).sqrt()) / 2.0_f64.sqrt();
            return 2.0 * self.standard_normal_cdf(-z.abs());
        }

        // For small df, use rough approximation
        // In practice, you'd use a proper gamma function implementation
        let normalized_stat = chi_squared_stat / df as f64;
        if normalized_stat > 6.0 {
            0.001
        } else if normalized_stat > 4.0 {
            0.01
        } else if normalized_stat > 2.5 {
            0.05
        } else if normalized_stat > 1.5 {
            0.2
        } else {
            0.5
        }
    }

    /// Standard normal CDF approximation
    fn standard_normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / 2.0_f64.sqrt()))
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
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

    /// Paired t-test for comparing cross-validation scores
    fn paired_t_test(&self, scores1: &[f64], scores2: &[f64]) -> Result<f64> {
        if scores1.len() != scores2.len() || scores1.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Score arrays must have same non-zero length".to_string(),
            ));
        }

        let n = scores1.len() as f64;
        let differences: Vec<f64> = scores1
            .iter()
            .zip(scores2.iter())
            .map(|(s1, s2)| s2 - s1)
            .collect();

        let mean_diff = differences.iter().sum::<f64>() / n;
        let var_diff = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        if var_diff < 1e-10 {
            return Ok(if mean_diff.abs() < 1e-10 { 1.0 } else { 0.0 });
        }

        let t_stat = mean_diff / (var_diff / n).sqrt();
        let df = n - 1.0;

        // Approximate p-value using normal distribution for large df
        if df > 30.0 {
            Ok(2.0 * self.standard_normal_cdf(-t_stat.abs()))
        } else {
            // Rough approximation for small df
            let abs_t = t_stat.abs();
            if abs_t > 4.0 {
                Ok(0.001)
            } else if abs_t > 2.5 {
                Ok(0.01)
            } else if abs_t > 2.0 {
                Ok(0.05)
            } else if abs_t > 1.0 {
                Ok(0.2)
            } else {
                Ok(0.5)
            }
        }
    }

    /// Compute Cohen's d effect size
    fn compute_cohens_d(&self, scores1: &[f64], scores2: &[f64]) -> f64 {
        if scores1.is_empty() || scores2.is_empty() {
            return 0.0;
        }

        let mean1 = scores1.iter().sum::<f64>() / scores1.len() as f64;
        let mean2 = scores2.iter().sum::<f64>() / scores2.len() as f64;

        let var1 =
            scores1.iter().map(|&s| (s - mean1).powi(2)).sum::<f64>() / (scores1.len() - 1) as f64;
        let var2 =
            scores2.iter().map(|&s| (s - mean2).powi(2)).sum::<f64>() / (scores2.len() - 1) as f64;

        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        if pooled_std < 1e-10 {
            0.0
        } else {
            (mean2 - mean1) / pooled_std
        }
    }

    /// Compute confidence interval for difference in means
    fn compute_confidence_interval(
        &self,
        scores1: &[f64],
        scores2: &[f64],
        confidence_level: f64,
    ) -> (f64, f64) {
        if scores1.len() != scores2.len() || scores1.is_empty() {
            return (0.0, 0.0);
        }

        let differences: Vec<f64> = scores1
            .iter()
            .zip(scores2.iter())
            .map(|(s1, s2)| s2 - s1)
            .collect();

        let n = differences.len() as f64;
        let mean_diff = differences.iter().sum::<f64>() / n;
        let var_diff = differences
            .iter()
            .map(|&d| (d - mean_diff).powi(2))
            .sum::<f64>()
            / (n - 1.0);

        let std_error = (var_diff / n).sqrt();

        // Use t-distribution critical value (approximation)
        let alpha = 1.0 - confidence_level;
        let t_critical = if n > 30.0 {
            // Use normal approximation
            if alpha <= 0.01 {
                2.576
            } else if alpha <= 0.05 {
                1.96
            } else {
                1.645
            }
        } else {
            // Rough t-distribution approximation
            if alpha <= 0.01 {
                3.0
            } else if alpha <= 0.05 {
                2.5
            } else {
                2.0
            }
        };

        let margin_error = t_critical * std_error;
        (mean_diff - margin_error, mean_diff + margin_error)
    }

    /// Generate recommendation based on nested model comparison
    fn generate_nested_model_recommendation(
        &self,
        likelihood_ratio: f64,
        p_value: f64,
        information_criteria: &HashMap<InformationCriterion, (f64, f64)>,
        alpha: f64,
    ) -> String {
        let is_significant = p_value < alpha;

        // Check AIC and BIC preferences
        let aic_prefers_complex = information_criteria
            .get(&InformationCriterion::AIC)
            .map(|(simple, complex)| complex < simple)
            .unwrap_or(false);

        let bic_prefers_complex = information_criteria
            .get(&InformationCriterion::BIC)
            .map(|(simple, complex)| complex < simple)
            .unwrap_or(false);

        if is_significant && aic_prefers_complex && bic_prefers_complex {
            "Strong evidence for complex model. Both likelihood ratio test and information criteria favor complexity.".to_string()
        } else if is_significant && aic_prefers_complex {
            "Moderate evidence for complex model. Likelihood ratio test is significant and AIC favors complexity, but BIC favors simplicity.".to_string()
        } else if !is_significant && !aic_prefers_complex && !bic_prefers_complex {
            "Strong evidence for simple model. Likelihood ratio test is not significant and both AIC and BIC favor simplicity.".to_string()
        } else if !is_significant {
            "Moderate evidence for simple model. Likelihood ratio test is not significant, suggesting additional complexity is not warranted.".to_string()
        } else {
            "Mixed evidence. Consider additional validation metrics and domain expertise for final decision.".to_string()
        }
    }
}

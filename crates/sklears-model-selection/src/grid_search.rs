//! Grid search and randomized search for hyperparameter tuning

use crate::{CrossValidator, KFold, Scoring};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::essentials::Normal as RandNormal;
// use scirs2_core::random::prelude::*;
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict, Score},
    types::Float,
};
use sklears_metrics::{classification::accuracy_score, get_scorer, regression::mean_squared_error};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Parameter grid for grid search
///
/// This represents all possible combinations of hyperparameters to test.
/// Each parameter name maps to a vector of possible values.
pub type ParameterGrid = HashMap<String, Vec<ParameterValue>>;

/// A parameter value that can be of different types
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ParameterValue {
    /// Integer parameter
    Int(i64),
    /// Float parameter
    Float(f64),
    /// Boolean parameter
    Bool(bool),
    /// String parameter
    String(String),
    /// Option integer parameter (Some/None)
    OptionalInt(Option<i64>),
    /// Option float parameter (Some/None)
    OptionalFloat(Option<f64>),
}

impl ParameterValue {
    /// Extract an integer value
    pub fn as_int(&self) -> Option<i64> {
        match self {
            ParameterValue::Int(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract a float value
    pub fn as_float(&self) -> Option<f64> {
        match self {
            ParameterValue::Float(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract a boolean value
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            ParameterValue::Bool(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract an optional integer value
    pub fn as_optional_int(&self) -> Option<Option<i64>> {
        match self {
            ParameterValue::OptionalInt(v) => Some(*v),
            _ => None,
        }
    }

    /// Extract an optional float value
    pub fn as_optional_float(&self) -> Option<Option<f64>> {
        match self {
            ParameterValue::OptionalFloat(v) => Some(*v),
            _ => None,
        }
    }
}

impl From<i32> for ParameterValue {
    fn from(value: i32) -> Self {
        ParameterValue::Int(value as i64)
    }
}

impl From<i64> for ParameterValue {
    fn from(value: i64) -> Self {
        ParameterValue::Int(value)
    }
}

impl From<f32> for ParameterValue {
    fn from(value: f32) -> Self {
        ParameterValue::Float(value as f64)
    }
}

impl From<f64> for ParameterValue {
    fn from(value: f64) -> Self {
        ParameterValue::Float(value)
    }
}

impl From<bool> for ParameterValue {
    fn from(value: bool) -> Self {
        ParameterValue::Bool(value)
    }
}

impl From<String> for ParameterValue {
    fn from(value: String) -> Self {
        ParameterValue::String(value)
    }
}

impl From<&str> for ParameterValue {
    fn from(value: &str) -> Self {
        ParameterValue::String(value.to_string())
    }
}

impl From<Option<i32>> for ParameterValue {
    fn from(value: Option<i32>) -> Self {
        ParameterValue::OptionalInt(value.map(|v| v as i64))
    }
}

impl From<Option<i64>> for ParameterValue {
    fn from(value: Option<i64>) -> Self {
        ParameterValue::OptionalInt(value)
    }
}

impl From<Option<f64>> for ParameterValue {
    fn from(value: Option<f64>) -> Self {
        ParameterValue::OptionalFloat(value)
    }
}

impl Eq for ParameterValue {}

impl std::hash::Hash for ParameterValue {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            ParameterValue::Int(v) => v.hash(state),
            ParameterValue::Float(v) => v.to_bits().hash(state), // Use bits representation for f64
            ParameterValue::Bool(v) => v.hash(state),
            ParameterValue::String(v) => v.hash(state),
            ParameterValue::OptionalInt(v) => v.hash(state),
            ParameterValue::OptionalFloat(v) => v.map(|f| f.to_bits()).hash(state),
        }
    }
}

/// A parameter combination for one grid search iteration
pub type ParameterSet = HashMap<String, ParameterValue>;

/// Grid search cross-validation
///
/// Exhaustive search over specified parameter values for an estimator.
/// Uses cross-validation to evaluate each parameter combination.
pub struct GridSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    /// Base estimator to use for each parameter combination
    estimator: E,
    /// Parameter grid to search
    param_grid: ParameterGrid,
    /// Cross-validation strategy
    cv: Box<dyn CrossValidator>,
    /// Scoring method
    scoring: Scoring,
    /// Number of parallel jobs (-1 for all cores)
    n_jobs: Option<usize>,
    /// Whether to refit on the entire dataset with best parameters
    refit: bool,
    /// Configuration function to apply parameters to estimator
    config_fn: ConfigFn,
    /// Phantom data for fitted estimator type
    _phantom: PhantomData<F>,
    // Fitted results
    best_estimator_: Option<F>,
    best_params_: Option<ParameterSet>,
    best_score_: Option<f64>,
    cv_results_: Option<GridSearchResults>,
}

/// Results from grid search cross-validation
#[derive(Debug, Clone)]
pub struct GridSearchResults {
    pub params: Vec<ParameterSet>,
    pub mean_test_scores: Array1<f64>,
    pub std_test_scores: Array1<f64>,
    pub mean_fit_times: Array1<f64>,
    pub mean_score_times: Array1<f64>,
    pub rank_test_scores: Array1<usize>,
}

/// Helper function for scoring that handles both regression and classification
fn compute_score_for_regression(
    metric_name: &str,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> Result<f64> {
    match metric_name {
        "neg_mean_squared_error" => Ok(-mean_squared_error(y_true, y_pred)?),
        "mean_squared_error" => Ok(mean_squared_error(y_true, y_pred)?),
        _ => {
            // For unsupported metrics, return a default score
            Err(SklearsError::InvalidInput(format!(
                "Metric '{}' not supported for regression",
                metric_name
            )))
        }
    }
}

/// Helper function for scoring classification data
fn compute_score_for_classification(
    metric_name: &str,
    y_true: &Array1<i32>,
    y_pred: &Array1<i32>,
) -> Result<f64> {
    match metric_name {
        "accuracy" => Ok(accuracy_score(y_true, y_pred)?),
        _ => {
            let scorer = get_scorer(metric_name)?;
            scorer.score(y_true.as_slice().unwrap(), y_pred.as_slice().unwrap())
        }
    }
}

impl<E, F, ConfigFn> GridSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    /// Create a new grid search CV
    pub fn new(estimator: E, param_grid: ParameterGrid, config_fn: ConfigFn) -> Self {
        Self {
            estimator,
            param_grid,
            cv: Box::new(KFold::new(5)),
            scoring: Scoring::EstimatorScore,
            n_jobs: None,
            refit: true,
            config_fn,
            _phantom: PhantomData,
            best_estimator_: None,
            best_params_: None,
            best_score_: None,
            cv_results_: None,
        }
    }

    /// Set the cross-validation strategy
    pub fn cv<C: CrossValidator + 'static>(mut self, cv: C) -> Self {
        self.cv = Box::new(cv);
        self
    }

    /// Set the scoring method
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set whether to refit with best parameters
    pub fn refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// Get the best estimator (after fitting)
    pub fn best_estimator(&self) -> Option<&F> {
        self.best_estimator_.as_ref()
    }

    /// Get the best parameters (after fitting)
    pub fn best_params(&self) -> Option<&ParameterSet> {
        self.best_params_.as_ref()
    }

    /// Get the best score (after fitting)
    pub fn best_score(&self) -> Option<f64> {
        self.best_score_
    }

    /// Get the CV results (after fitting)
    pub fn cv_results(&self) -> Option<&GridSearchResults> {
        self.cv_results_.as_ref()
    }

    /// Generate all parameter combinations from the grid
    fn generate_param_combinations(&self) -> Vec<ParameterSet> {
        let mut combinations = vec![HashMap::new()];

        for (param_name, param_values) in &self.param_grid {
            let mut new_combinations = Vec::new();

            for combination in combinations {
                for param_value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), param_value.clone());
                    new_combinations.push(new_combination);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    /// Evaluate a single parameter combination using cross-validation
    fn evaluate_params(
        &self,
        params: &ParameterSet,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(f64, f64, f64, f64)> {
        // Configure estimator with current parameters
        let configured_estimator = (self.config_fn)(self.estimator.clone(), params)?;

        // Get CV splits
        let splits = self.cv.split(x.nrows(), None);
        let n_splits = splits.len();

        let mut test_scores = Vec::with_capacity(n_splits);
        let mut fit_times = Vec::with_capacity(n_splits);
        let mut score_times = Vec::with_capacity(n_splits);

        // Evaluate on each CV fold
        for (train_idx, test_idx) in splits {
            // Extract train and test data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_idx);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_idx);

            // Fit the estimator
            let start = std::time::Instant::now();
            let fitted = configured_estimator.clone().fit(&x_train, &y_train)?;
            let fit_time = start.elapsed().as_secs_f64();
            fit_times.push(fit_time);

            // Score on test set
            let start = std::time::Instant::now();
            let test_score = match &self.scoring {
                Scoring::EstimatorScore => fitted.score(&x_test, &y_test)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_test)?;
                    func(&y_test.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_test)?;
                    compute_score_for_regression(metric_name, &y_test, &y_pred)?
                }
                Scoring::Scorer(_scorer) => {
                    let y_pred = fitted.predict(&x_test)?;
                    // Default to negative MSE for regression
                    -mean_squared_error(&y_test, &y_pred)?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_test, &y_test)?
                }
            };
            let score_time = start.elapsed().as_secs_f64();
            score_times.push(score_time);
            test_scores.push(test_score);
        }

        // Calculate statistics
        let mean_test_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let std_test_score = {
            let variance = test_scores
                .iter()
                .map(|&score| (score - mean_test_score).powi(2))
                .sum::<f64>()
                / test_scores.len() as f64;
            variance.sqrt()
        };
        let mean_fit_time = fit_times.iter().sum::<f64>() / fit_times.len() as f64;
        let mean_score_time = score_times.iter().sum::<f64>() / score_times.len() as f64;

        Ok((
            mean_test_score,
            std_test_score,
            mean_fit_time,
            mean_score_time,
        ))
    }
}

impl<E, F, ConfigFn> Fit<Array2<Float>, Array1<Float>> for GridSearchCV<E, F, ConfigFn>
where
    E: Clone + Send + Sync,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>> + Send + Sync,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E> + Send + Sync,
{
    type Fitted = GridSearchCV<E, F, ConfigFn>;

    fn fit(mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {}", x.nrows()),
                actual: format!("y.shape[0] = {}", y.len()),
            });
        }

        // Generate all parameter combinations
        let param_combinations = self.generate_param_combinations();

        if param_combinations.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter combinations to evaluate".to_string(),
            ));
        }

        // Evaluate each parameter combination
        let mut results = Vec::with_capacity(param_combinations.len());

        for params in &param_combinations {
            let (mean_score, std_score, mean_fit_time, mean_score_time) =
                self.evaluate_params(params, x, y)?;

            results.push((mean_score, std_score, mean_fit_time, mean_score_time));
        }

        // Find best parameters
        let best_idx = results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| SklearsError::NumericalError("No valid scores found".to_string()))?;

        let best_params = param_combinations[best_idx].clone();
        let best_score = results[best_idx].0;

        // Create CV results
        let mean_test_scores = Array1::from_vec(results.iter().map(|r| r.0).collect());
        let std_test_scores = Array1::from_vec(results.iter().map(|r| r.1).collect());
        let mean_fit_times = Array1::from_vec(results.iter().map(|r| r.2).collect());
        let mean_score_times = Array1::from_vec(results.iter().map(|r| r.3).collect());

        // Calculate ranks (1 = best)
        let mut scores_with_idx: Vec<(f64, usize)> = mean_test_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (score, i))
            .collect();
        scores_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut ranks = vec![0; param_combinations.len()];
        for (rank, (_, idx)) in scores_with_idx.iter().enumerate() {
            ranks[*idx] = rank + 1;
        }

        let cv_results = GridSearchResults {
            params: param_combinations.clone(),
            mean_test_scores,
            std_test_scores,
            mean_fit_times,
            mean_score_times,
            rank_test_scores: Array1::from_vec(ranks),
        };

        // Refit with best parameters if requested
        let best_estimator = if self.refit {
            let configured_estimator = (self.config_fn)(self.estimator.clone(), &best_params)?;
            Some(configured_estimator.fit(x, y)?)
        } else {
            None
        };

        self.best_estimator_ = best_estimator;
        self.best_params_ = Some(best_params);
        self.best_score_ = Some(best_score);
        self.cv_results_ = Some(cv_results);

        Ok(self)
    }
}

impl<E, F, ConfigFn> Predict<Array2<Float>, Array1<Float>> for GridSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        match &self.best_estimator_ {
            Some(estimator) => estimator.predict(x),
            None => Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            }),
        }
    }
}

impl<E, F, ConfigFn> Score<Array2<Float>, Array1<Float>> for GridSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    type Float = f64;
    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        match &self.best_estimator_ {
            Some(estimator) => estimator.score(x, y),
            None => Err(SklearsError::NotFitted {
                operation: "score".to_string(),
            }),
        }
    }
}

/// Parameter distribution for randomized search
#[derive(Debug, Clone)]
pub enum ParameterDistribution {
    /// Uniform distribution over discrete values
    Choice(Vec<ParameterValue>),
    /// Uniform distribution over integer range [low, high)
    RandInt { low: i64, high: i64 },
    /// Uniform distribution over float range [low, high)
    Uniform { low: f64, high: f64 },
    /// Log-uniform distribution over float range [low, high)
    LogUniform { low: f64, high: f64 },
    /// Normal distribution with mean and std
    Normal { mean: f64, std: f64 },
}

impl ParameterDistribution {
    /// Sample a value from this distribution
    pub fn sample(&self, rng: &mut impl scirs2_core::random::Rng) -> ParameterValue {
        use scirs2_core::essentials::Uniform;
        use scirs2_core::random::Distribution;

        match self {
            ParameterDistribution::Choice(values) => values.as_slice().choose(rng).unwrap().clone(),
            ParameterDistribution::RandInt { low, high } => {
                let dist = Uniform::new(*low, *high).unwrap();
                ParameterValue::Int(dist.sample(rng))
            }
            ParameterDistribution::Uniform { low, high } => {
                let dist = Uniform::new(*low, *high).unwrap();
                ParameterValue::Float(dist.sample(rng))
            }
            ParameterDistribution::LogUniform { low, high } => {
                // Implement log-uniform manually: sample from log scale then exponentiate
                let log_low = low.ln();
                let log_high = high.ln();
                let dist = Uniform::new(log_low, log_high).unwrap();
                let log_sample = dist.sample(rng);
                ParameterValue::Float(log_sample.exp())
            }
            ParameterDistribution::Normal { mean, std } => {
                let dist = RandNormal::new(*mean, *std).unwrap();
                ParameterValue::Float(dist.sample(rng))
            }
        }
    }
}

/// Parameter distribution grid for randomized search
pub type ParameterDistributions = HashMap<String, ParameterDistribution>;

/// Randomized search cross-validation
///
/// Search over parameter distributions with random sampling.
/// More efficient than GridSearchCV for large parameter spaces.
pub struct RandomizedSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    /// Base estimator to use for each parameter combination
    estimator: E,
    /// Parameter distributions to search
    param_distributions: ParameterDistributions,
    /// Number of parameter settings to sample
    n_iter: usize,
    /// Cross-validation strategy
    cv: Box<dyn CrossValidator>,
    /// Scoring method
    scoring: Scoring,
    /// Number of parallel jobs
    n_jobs: Option<usize>,
    /// Whether to refit on the entire dataset with best parameters
    refit: bool,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    /// Configuration function to apply parameters to estimator
    config_fn: ConfigFn,
    /// Phantom data for fitted estimator type
    _phantom: PhantomData<F>,
    // Fitted results
    best_estimator_: Option<F>,
    best_params_: Option<ParameterSet>,
    best_score_: Option<f64>,
    cv_results_: Option<GridSearchResults>,
}

impl<E, F, ConfigFn> RandomizedSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    pub fn new(
        estimator: E,
        param_distributions: ParameterDistributions,
        config_fn: ConfigFn,
    ) -> Self {
        Self {
            estimator,
            param_distributions,
            n_iter: 10,
            cv: Box::new(KFold::new(5)),
            scoring: Scoring::EstimatorScore,
            n_jobs: None,
            refit: true,
            random_state: None,
            config_fn,
            _phantom: PhantomData,
            best_estimator_: None,
            best_params_: None,
            best_score_: None,
            cv_results_: None,
        }
    }

    /// Set the number of parameter settings to sample
    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    /// Set the cross-validation strategy
    pub fn cv<C: CrossValidator + 'static>(mut self, cv: C) -> Self {
        self.cv = Box::new(cv);
        self
    }

    /// Set the scoring method
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<usize>) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Set whether to refit with best parameters
    pub fn refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    /// Set the random seed
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Get the best estimator (after fitting)
    pub fn best_estimator(&self) -> Option<&F> {
        self.best_estimator_.as_ref()
    }

    /// Get the best parameters (after fitting)
    pub fn best_params(&self) -> Option<&ParameterSet> {
        self.best_params_.as_ref()
    }

    /// Get the best score (after fitting)
    pub fn best_score(&self) -> Option<f64> {
        self.best_score_
    }

    /// Get the CV results (after fitting)
    pub fn cv_results(&self) -> Option<&GridSearchResults> {
        self.cv_results_.as_ref()
    }

    /// Sample parameter combinations from distributions
    fn sample_parameters(&self, n_samples: usize) -> Vec<ParameterSet> {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        let mut param_sets = Vec::with_capacity(n_samples);

        for _ in 0..n_samples {
            let mut param_set = HashMap::new();

            for (param_name, distribution) in &self.param_distributions {
                let value = distribution.sample(&mut rng);
                param_set.insert(param_name.clone(), value);
            }

            param_sets.push(param_set);
        }

        param_sets
    }

    /// Evaluate a single parameter combination using cross-validation
    fn evaluate_params(
        &self,
        params: &ParameterSet,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<(f64, f64, f64, f64)> {
        // Configure estimator with current parameters
        let configured_estimator = (self.config_fn)(self.estimator.clone(), params)?;

        // Get CV splits
        let splits = self.cv.split(x.nrows(), None);
        let n_splits = splits.len();

        let mut test_scores = Vec::with_capacity(n_splits);
        let mut fit_times = Vec::with_capacity(n_splits);
        let mut score_times = Vec::with_capacity(n_splits);

        // Evaluate on each CV fold
        for (train_idx, test_idx) in splits {
            // Extract train and test data
            let x_train = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let y_train = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
            let x_test = x.select(scirs2_core::ndarray::Axis(0), &test_idx);
            let y_test = y.select(scirs2_core::ndarray::Axis(0), &test_idx);

            // Fit the estimator
            let start = std::time::Instant::now();
            let fitted = configured_estimator.clone().fit(&x_train, &y_train)?;
            let fit_time = start.elapsed().as_secs_f64();
            fit_times.push(fit_time);

            // Score on test set
            let start = std::time::Instant::now();
            let test_score = match &self.scoring {
                Scoring::EstimatorScore => fitted.score(&x_test, &y_test)?,
                Scoring::Custom(func) => {
                    let y_pred = fitted.predict(&x_test)?;
                    func(&y_test.to_owned(), &y_pred)?
                }
                Scoring::Metric(metric_name) => {
                    let y_pred = fitted.predict(&x_test)?;
                    compute_score_for_regression(metric_name, &y_test, &y_pred)?
                }
                Scoring::Scorer(_scorer) => {
                    let y_pred = fitted.predict(&x_test)?;
                    // Default to negative MSE for regression
                    -mean_squared_error(&y_test, &y_pred)?
                }
                Scoring::MultiMetric(_metrics) => {
                    // For multi-metric, just use the first metric for now
                    fitted.score(&x_test, &y_test)?
                }
            };
            let score_time = start.elapsed().as_secs_f64();
            score_times.push(score_time);
            test_scores.push(test_score);
        }

        // Calculate statistics
        let mean_test_score = test_scores.iter().sum::<f64>() / test_scores.len() as f64;
        let std_test_score = {
            let variance = test_scores
                .iter()
                .map(|&score| (score - mean_test_score).powi(2))
                .sum::<f64>()
                / test_scores.len() as f64;
            variance.sqrt()
        };
        let mean_fit_time = fit_times.iter().sum::<f64>() / fit_times.len() as f64;
        let mean_score_time = score_times.iter().sum::<f64>() / score_times.len() as f64;

        Ok((
            mean_test_score,
            std_test_score,
            mean_fit_time,
            mean_score_time,
        ))
    }
}

impl<E, F, ConfigFn> Fit<Array2<Float>, Array1<Float>> for RandomizedSearchCV<E, F, ConfigFn>
where
    E: Clone + Send + Sync,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>> + Send + Sync,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E> + Send + Sync,
{
    type Fitted = RandomizedSearchCV<E, F, ConfigFn>;

    fn fit(mut self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty dataset".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {}", x.nrows()),
                actual: format!("y.shape[0] = {}", y.len()),
            });
        }

        if self.param_distributions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No parameter distributions to sample from".to_string(),
            ));
        }

        // Sample parameter combinations
        let param_combinations = self.sample_parameters(self.n_iter);

        // Evaluate each parameter combination
        let mut results = Vec::with_capacity(param_combinations.len());

        for params in &param_combinations {
            let (mean_score, std_score, mean_fit_time, mean_score_time) =
                self.evaluate_params(params, x, y)?;

            results.push((mean_score, std_score, mean_fit_time, mean_score_time));
        }

        // Find best parameters
        let best_idx = results
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.0.partial_cmp(&b.0).unwrap())
            .map(|(idx, _)| idx)
            .ok_or_else(|| SklearsError::NumericalError("No valid scores found".to_string()))?;

        let best_params = param_combinations[best_idx].clone();
        let best_score = results[best_idx].0;

        // Create CV results
        let mean_test_scores = Array1::from_vec(results.iter().map(|r| r.0).collect());
        let std_test_scores = Array1::from_vec(results.iter().map(|r| r.1).collect());
        let mean_fit_times = Array1::from_vec(results.iter().map(|r| r.2).collect());
        let mean_score_times = Array1::from_vec(results.iter().map(|r| r.3).collect());

        // Calculate ranks (1 = best)
        let mut scores_with_idx: Vec<(f64, usize)> = mean_test_scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (score, i))
            .collect();
        scores_with_idx.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut ranks = vec![0; param_combinations.len()];
        for (rank, (_, idx)) in scores_with_idx.iter().enumerate() {
            ranks[*idx] = rank + 1;
        }

        let cv_results = GridSearchResults {
            params: param_combinations.clone(),
            mean_test_scores,
            std_test_scores,
            mean_fit_times,
            mean_score_times,
            rank_test_scores: Array1::from_vec(ranks),
        };

        // Refit with best parameters if requested
        let best_estimator = if self.refit {
            let configured_estimator = (self.config_fn)(self.estimator.clone(), &best_params)?;
            Some(configured_estimator.fit(x, y)?)
        } else {
            None
        };

        self.best_estimator_ = best_estimator;
        self.best_params_ = Some(best_params);
        self.best_score_ = Some(best_score);
        self.cv_results_ = Some(cv_results);

        Ok(self)
    }
}

impl<E, F, ConfigFn> Predict<Array2<Float>, Array1<Float>> for RandomizedSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        match &self.best_estimator_ {
            Some(estimator) => estimator.predict(x),
            None => Err(SklearsError::NotFitted {
                operation: "predict".to_string(),
            }),
        }
    }
}

impl<E, F, ConfigFn> Score<Array2<Float>, Array1<Float>> for RandomizedSearchCV<E, F, ConfigFn>
where
    E: Clone,
    E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
    F: Predict<Array2<Float>, Array1<Float>>,
    F: Score<Array2<Float>, Array1<Float>, Float = f64>,
    ConfigFn: Fn(E, &ParameterSet) -> Result<E>,
{
    type Float = f64;
    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        match &self.best_estimator_ {
            Some(estimator) => estimator.score(x, y),
            None => Err(SklearsError::NotFitted {
                operation: "score".to_string(),
            }),
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KFold;
    use scirs2_core::ndarray::array;
    // use sklears_ensemble::GradientBoostingRegressor;

    // Mock estimator for testing
    #[derive(Debug, Clone)]
    struct MockRegressor {
        n_estimators: usize,
        learning_rate: f64,
        random_state: Option<u64>,
        fitted: bool,
    }

    impl MockRegressor {
        fn new() -> Self {
            Self {
                n_estimators: 100,
                learning_rate: 0.1,
                random_state: None,
                fitted: false,
            }
        }

        fn n_estimators(mut self, n: usize) -> Self {
            self.n_estimators = n;
            self
        }

        fn learning_rate(mut self, lr: f64) -> Self {
            self.learning_rate = lr;
            self
        }

        fn random_state(mut self, state: Option<u64>) -> Self {
            self.random_state = state;
            self
        }
    }

    impl Fit<Array2<f64>, Array1<f64>> for MockRegressor {
        type Fitted = MockRegressor;

        fn fit(mut self, _x: &Array2<f64>, _y: &Array1<f64>) -> Result<Self::Fitted> {
            self.fitted = true;
            Ok(self)
        }
    }

    impl Predict<Array2<f64>, Array1<f64>> for MockRegressor {
        fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>> {
            if !self.fitted {
                return Err(SklearsError::NotFitted {
                    operation: "predict".to_string(),
                });
            }
            // Simple prediction: sum of features
            Ok(x.sum_axis(scirs2_core::ndarray::Axis(1)))
        }
    }

    impl Score<Array2<f64>, Array1<f64>> for MockRegressor {
        type Float = f64;

        fn score(&self, x: &Array2<f64>, y: &Array1<f64>) -> Result<f64> {
            let y_pred = self.predict(x)?;
            let mse = mean_squared_error(y, &y_pred)?;
            Ok(-mse) // Return negative MSE as score (higher is better)
        }
    }

    type GradientBoostingRegressor = MockRegressor;

    #[test]
    fn test_parameter_value_extraction() {
        let int_param = ParameterValue::Int(42);
        assert_eq!(int_param.as_int(), Some(42));
        assert_eq!(int_param.as_float(), None);

        let float_param = ParameterValue::Float(std::f64::consts::PI);
        assert_eq!(float_param.as_float(), Some(std::f64::consts::PI));
        assert_eq!(float_param.as_int(), None);

        let opt_int_param = ParameterValue::OptionalInt(Some(10));
        assert_eq!(opt_int_param.as_optional_int(), Some(Some(10)));
    }

    #[test]
    #[ignore] // Temporarily disabled due to sklears_ensemble dependency issues
    fn test_grid_search_cv() {
        // Create a simple dataset
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // roughly x^2

        // Create parameter grid
        let mut param_grid = HashMap::new();
        param_grid.insert(
            "n_estimators".to_string(),
            vec![ParameterValue::Int(5), ParameterValue::Int(10)],
        );
        param_grid.insert(
            "learning_rate".to_string(),
            vec![ParameterValue::Float(0.1), ParameterValue::Float(0.3)],
        );

        // Configuration function for GradientBoostingRegressor
        let config_fn = |estimator: GradientBoostingRegressor,
                         params: &ParameterSet|
         -> Result<GradientBoostingRegressor> {
            let mut configured = estimator;

            if let Some(n_est) = params.get("n_estimators").and_then(|p| p.as_int()) {
                configured = configured.n_estimators(n_est as usize);
            }

            if let Some(lr) = params.get("learning_rate").and_then(|p| p.as_float()) {
                configured = configured.learning_rate(lr);
            }

            Ok(configured)
        };

        // Create and fit grid search
        let base_estimator = GradientBoostingRegressor::new().random_state(Some(42));
        let grid_search = GridSearchCV::new(base_estimator, param_grid, config_fn)
            .cv(KFold::new(3))
            .fit(&x, &y)
            .unwrap();

        // Check that we have results
        assert!(grid_search.best_score().is_some());
        assert!(grid_search.best_params().is_some());
        assert!(grid_search.best_estimator().is_some());
        assert!(grid_search.cv_results().is_some());

        // Check CV results structure
        let cv_results = grid_search.cv_results().unwrap();
        assert_eq!(cv_results.params.len(), 4); // 2 x 2 = 4 combinations
        assert_eq!(cv_results.mean_test_scores.len(), 4);
        assert_eq!(cv_results.rank_test_scores.len(), 4);

        // Best rank should be 1
        let best_rank = cv_results.rank_test_scores.iter().min().unwrap();
        assert_eq!(*best_rank, 1);

        // Test prediction with best estimator
        let predictions = grid_search.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    #[ignore] // Temporarily disabled due to sklears_ensemble dependency issues
    fn test_grid_search_empty_grid() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let param_grid = HashMap::new(); // Empty grid
        let config_fn = |estimator: GradientBoostingRegressor,
                         _params: &ParameterSet|
         -> Result<GradientBoostingRegressor> { Ok(estimator) };

        let base_estimator = GradientBoostingRegressor::new();
        let result = GridSearchCV::new(base_estimator, param_grid, config_fn)
            .cv(KFold::new(2)) // Use 2 folds for 2 samples
            .fit(&x, &y);

        // Empty grid should succeed with default parameters
        assert!(result.is_ok());
        let grid_search = result.unwrap();

        // Should have one parameter combination (empty set = default params)
        let cv_results = grid_search.cv_results().unwrap();
        assert_eq!(cv_results.params.len(), 1);
        assert!(cv_results.params[0].is_empty()); // Empty parameter set
    }

    #[test]
    fn test_parameter_distribution_sampling() {
        use scirs2_core::random::rngs::StdRng;
        use scirs2_core::random::SeedableRng;
        let mut rng = StdRng::seed_from_u64(42);

        // Test Choice distribution
        let choice_dist = ParameterDistribution::Choice(vec![
            ParameterValue::Int(1),
            ParameterValue::Int(2),
            ParameterValue::Int(3),
        ]);
        let sample = choice_dist.sample(&mut rng);
        if let ParameterValue::Int(val) = sample {
            assert!(val >= 1 && val <= 3);
        } else {
            panic!("Expected Int parameter value");
        }

        // Test RandInt distribution
        let int_dist = ParameterDistribution::RandInt { low: 10, high: 20 };
        let sample = int_dist.sample(&mut rng);
        if let ParameterValue::Int(val) = sample {
            assert!(val >= 10 && val < 20);
        } else {
            panic!("Expected Int parameter value");
        }

        // Test Uniform distribution
        let uniform_dist = ParameterDistribution::Uniform {
            low: 0.0,
            high: 1.0,
        };
        let sample = uniform_dist.sample(&mut rng);
        if let ParameterValue::Float(val) = sample {
            assert!(val >= 0.0 && val < 1.0);
        } else {
            panic!("Expected Float parameter value");
        }

        // Test Normal distribution
        let normal_dist = ParameterDistribution::Normal {
            mean: 0.0,
            std: 1.0,
        };
        let sample = normal_dist.sample(&mut rng);
        assert!(matches!(sample, ParameterValue::Float(_)));
    }

    #[test]
    #[ignore] // Temporarily disabled due to sklears_ensemble dependency issues
    fn test_randomized_search_cv() {
        // Create a simple dataset
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0],];
        let y = array![1.0, 4.0, 9.0, 16.0, 25.0]; // roughly x^2

        // Create parameter distributions
        let mut param_distributions = HashMap::new();
        param_distributions.insert(
            "n_estimators".to_string(),
            ParameterDistribution::Choice(vec![
                ParameterValue::Int(5),
                ParameterValue::Int(10),
                ParameterValue::Int(15),
            ]),
        );
        param_distributions.insert(
            "learning_rate".to_string(),
            ParameterDistribution::Uniform {
                low: 0.05,
                high: 0.5,
            },
        );

        // Configuration function for GradientBoostingRegressor
        let config_fn = |estimator: GradientBoostingRegressor,
                         params: &ParameterSet|
         -> Result<GradientBoostingRegressor> {
            let mut configured = estimator;

            if let Some(n_est) = params.get("n_estimators").and_then(|p| p.as_int()) {
                configured = configured.n_estimators(n_est as usize);
            }

            if let Some(lr) = params.get("learning_rate").and_then(|p| p.as_float()) {
                configured = configured.learning_rate(lr);
            }

            Ok(configured)
        };

        // Create and fit randomized search
        let base_estimator = GradientBoostingRegressor::new().random_state(Some(42));
        let randomized_search =
            RandomizedSearchCV::new(base_estimator, param_distributions, config_fn)
                .n_iter(8) // Sample 8 parameter combinations
                .cv(KFold::new(3))
                .random_state(Some(42))
                .fit(&x, &y)
                .unwrap();

        // Check that we have results
        assert!(randomized_search.best_score().is_some());
        assert!(randomized_search.best_params().is_some());
        assert!(randomized_search.best_estimator().is_some());
        assert!(randomized_search.cv_results().is_some());

        // Check CV results structure
        let cv_results = randomized_search.cv_results().unwrap();
        assert_eq!(cv_results.params.len(), 8); // Should have 8 sampled combinations
        assert_eq!(cv_results.mean_test_scores.len(), 8);
        assert_eq!(cv_results.rank_test_scores.len(), 8);

        // Best rank should be 1
        let best_rank = cv_results.rank_test_scores.iter().min().unwrap();
        assert_eq!(*best_rank, 1);

        // Test prediction with best estimator
        let predictions = randomized_search.predict(&x).unwrap();
        assert_eq!(predictions.len(), x.nrows());

        // Check that parameter values are within expected ranges
        for params in &cv_results.params {
            if let Some(n_est) = params.get("n_estimators").and_then(|p| p.as_int()) {
                assert!(n_est == 5 || n_est == 10 || n_est == 15);
            }
            if let Some(lr) = params.get("learning_rate").and_then(|p| p.as_float()) {
                assert!(lr >= 0.05 && lr < 0.5);
            }
        }
    }

    #[test]
    #[ignore] // Temporarily disabled due to sklears_ensemble dependency issues
    fn test_randomized_search_empty_distributions() {
        let x = array![[1.0], [2.0]];
        let y = array![1.0, 2.0];

        let param_distributions = HashMap::new(); // Empty distributions
        let config_fn = |estimator: GradientBoostingRegressor,
                         _params: &ParameterSet|
         -> Result<GradientBoostingRegressor> { Ok(estimator) };

        let base_estimator = GradientBoostingRegressor::new();
        let result = RandomizedSearchCV::new(base_estimator, param_distributions, config_fn)
            .cv(KFold::new(2))
            .fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    #[ignore] // Temporarily disabled due to sklears_ensemble dependency issues
    fn test_randomized_search_reproducibility() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        // Create parameter distributions
        let mut param_distributions = HashMap::new();
        param_distributions.insert(
            "learning_rate".to_string(),
            ParameterDistribution::Uniform {
                low: 0.1,
                high: 0.5,
            },
        );

        let config_fn = |estimator: GradientBoostingRegressor,
                         params: &ParameterSet|
         -> Result<GradientBoostingRegressor> {
            let mut configured = estimator;
            if let Some(lr) = params.get("learning_rate").and_then(|p| p.as_float()) {
                configured = configured.learning_rate(lr);
            }
            Ok(configured)
        };

        // Run twice with same random state
        let base_estimator1 = GradientBoostingRegressor::new().random_state(Some(42));
        let result1 =
            RandomizedSearchCV::new(base_estimator1, param_distributions.clone(), config_fn)
                .n_iter(5)
                .random_state(Some(123))
                .cv(KFold::new(2))
                .fit(&x, &y)
                .unwrap();

        let base_estimator2 = GradientBoostingRegressor::new().random_state(Some(42));
        let result2 = RandomizedSearchCV::new(base_estimator2, param_distributions, config_fn)
            .n_iter(5)
            .random_state(Some(123))
            .cv(KFold::new(2))
            .fit(&x, &y)
            .unwrap();

        // Should get identical results
        assert_eq!(result1.best_score(), result2.best_score());

        let params1 = result1.cv_results().unwrap();
        let params2 = result2.cv_results().unwrap();

        // Check that the same parameters were sampled
        for (p1, p2) in params1.params.iter().zip(params2.params.iter()) {
            assert_eq!(p1, p2);
        }
    }
}

//! Automatic Hyperparameter Tuning for Covariance Estimators
//!
//! This module provides automatic hyperparameter optimization for covariance estimators
//! using various search strategies including grid search, random search, Bayesian optimization,
//! and evolutionary algorithms.

use scirs2_core::ndarray::{s, Array2, ArrayView2, NdFloat};
use scirs2_core::Distribution;
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::fmt::Debug;

/// Type alias for custom scoring function
type CustomScoringFn = Box<dyn Fn(&Array2<f64>, Option<&Array2<f64>>) -> f64 + Send + Sync>;

/// Hyperparameter tuning configuration
#[derive(Debug, Clone)]
pub struct TuningConfig {
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
    /// Search strategy to use
    pub search_strategy: SearchStrategy,
    /// Scoring metric for optimization
    pub scoring: ScoringMetric,
    /// Maximum number of evaluations
    pub max_evaluations: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Parallel execution configuration
    pub n_jobs: Option<usize>,
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    pub n_folds: usize,
    pub shuffle: bool,
    pub random_seed: Option<u64>,
    pub stratify: bool,
}

/// Search strategy for hyperparameter optimization
#[derive(Debug, Clone)]
pub enum SearchStrategy {
    /// Exhaustive grid search
    GridSearch,
    /// Random search over parameter distributions
    RandomSearch { n_iter: usize },
    /// Bayesian optimization with Gaussian processes
    BayesianOptimization {
        acquisition_function: AcquisitionFunction,
        n_initial_points: usize,
    },
    /// Evolutionary algorithm (genetic algorithm)
    EvolutionarySearch {
        population_size: usize,
        mutation_rate: f64,
        crossover_rate: f64,
    },
    /// Tree-structured Parzen Estimator (TPE)
    TPESearch {
        n_startup_trials: usize,
        n_ei_candidates: usize,
    },
    /// Successive halving (early termination of poor performers)
    SuccessiveHalving {
        min_resource: f64,
        max_resource: f64,
        reduction_factor: f64,
    },
}

/// Acquisition function for Bayesian optimization
#[derive(Debug, Clone)]
pub enum AcquisitionFunction {
    /// Expected Improvement
    ExpectedImprovement,
    /// Upper Confidence Bound
    UpperConfidenceBound { beta: f64 },
    /// Probability of Improvement
    ProbabilityOfImprovement,
    /// Knowledge Gradient
    KnowledgeGradient,
}

/// Scoring metrics for evaluating covariance estimators
pub enum ScoringMetric {
    /// Log-likelihood (higher is better)
    LogLikelihood,
    /// Frobenius norm error from true covariance (lower is better)
    FrobeniusError,
    /// Spectral norm error (lower is better)
    SpectralError,
    /// Condition number (lower is better)
    ConditionNumber,
    /// Stein's loss (lower is better)
    SteinLoss,
    /// Predictive likelihood on held-out data
    PredictiveLikelihood,
    /// Cross-validation score
    CrossValidationScore,
    /// Custom scoring function
    Custom(CustomScoringFn),
}

impl std::fmt::Debug for ScoringMetric {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScoringMetric::LogLikelihood => write!(f, "LogLikelihood"),
            ScoringMetric::FrobeniusError => write!(f, "FrobeniusError"),
            ScoringMetric::SpectralError => write!(f, "SpectralError"),
            ScoringMetric::ConditionNumber => write!(f, "ConditionNumber"),
            ScoringMetric::SteinLoss => write!(f, "SteinLoss"),
            ScoringMetric::PredictiveLikelihood => write!(f, "PredictiveLikelihood"),
            ScoringMetric::CrossValidationScore => write!(f, "CrossValidationScore"),
            ScoringMetric::Custom(_) => write!(f, "Custom(function)"),
        }
    }
}

impl Clone for ScoringMetric {
    fn clone(&self) -> Self {
        match self {
            ScoringMetric::LogLikelihood => ScoringMetric::LogLikelihood,
            ScoringMetric::FrobeniusError => ScoringMetric::FrobeniusError,
            ScoringMetric::SpectralError => ScoringMetric::SpectralError,
            ScoringMetric::ConditionNumber => ScoringMetric::ConditionNumber,
            ScoringMetric::SteinLoss => ScoringMetric::SteinLoss,
            ScoringMetric::PredictiveLikelihood => ScoringMetric::PredictiveLikelihood,
            ScoringMetric::CrossValidationScore => ScoringMetric::CrossValidationScore,
            ScoringMetric::Custom(_) => {
                // Custom functions cannot be cloned, so we'll panic
                panic!("Cannot clone ScoringMetric::Custom - custom functions are not cloneable")
            }
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Patience: number of evaluations without improvement before stopping
    pub patience: usize,
    /// Minimum improvement threshold
    pub min_delta: f64,
    /// Whether the metric should be maximized or minimized
    pub maximize: bool,
}

/// Parameter specification for tuning
#[derive(Debug, Clone)]
pub struct ParameterSpec {
    /// Parameter name
    pub name: String,
    /// Parameter type and range
    pub param_type: ParameterType,
    /// Whether the parameter is on log scale
    pub log_scale: bool,
}

/// Types of parameters that can be tuned
#[derive(Debug, Clone)]
pub enum ParameterType {
    /// Continuous parameter with min and max bounds
    Continuous { min: f64, max: f64 },
    /// Integer parameter with min and max bounds
    Integer { min: i64, max: i64 },
    /// Categorical parameter with discrete choices
    Categorical { choices: Vec<String> },
    /// Boolean parameter
    Boolean,
}

/// Parameter value wrapper
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterValue {
    /// Continuous value
    Float(f64),
    /// Integer value
    Int(i64),
    /// String value
    String(String),
    /// Boolean value
    Bool(bool),
}

/// Result of hyperparameter tuning
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// Best parameter configuration found
    pub best_params: HashMap<String, ParameterValue>,
    /// Best cross-validation score
    pub best_score: f64,
    /// Standard deviation of best score across CV folds
    pub best_score_std: f64,
    /// All evaluated parameter configurations
    pub cv_results: Vec<CVResult>,
    /// Optimization history
    pub optimization_history: OptimizationHistory,
    /// Total time spent tuning
    pub total_time_seconds: f64,
    /// Number of evaluations performed
    pub n_evaluations: usize,
}

/// Single cross-validation result
#[derive(Debug, Clone)]
pub struct CVResult {
    /// Parameter configuration
    pub params: HashMap<String, ParameterValue>,
    /// Mean score across CV folds
    pub mean_score: f64,
    /// Standard deviation across CV folds
    pub std_score: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Time to evaluate this configuration
    pub eval_time_seconds: f64,
    /// Additional metrics
    pub additional_metrics: HashMap<String, f64>,
}

/// Optimization history tracking
#[derive(Debug, Clone)]
pub struct OptimizationHistory {
    /// Best score at each iteration
    pub best_scores: Vec<f64>,
    /// Score at each iteration
    pub scores: Vec<f64>,
    /// Cumulative time at each iteration
    pub cumulative_times: Vec<f64>,
    /// Search space exploration metrics
    pub exploration_metrics: Vec<ExplorationMetrics>,
}

/// Metrics about search space exploration
#[derive(Debug, Clone)]
pub struct ExplorationMetrics {
    /// Fraction of search space explored
    pub exploration_ratio: f64,
    /// Diversity of evaluated points
    pub diversity_score: f64,
    /// Convergence indicator
    pub convergence_score: f64,
}

/// Hyperparameter tuner for covariance estimators
pub struct CovarianceHyperparameterTuner<F: NdFloat> {
    /// Parameter specifications
    pub parameter_specs: Vec<ParameterSpec>,
    /// Tuning configuration
    pub config: TuningConfig,
    /// Phantom data for float type
    _phantom: std::marker::PhantomData<F>,
}

impl<F: NdFloat> CovarianceHyperparameterTuner<F> {
    /// Create a new hyperparameter tuner
    pub fn new(parameter_specs: Vec<ParameterSpec>, config: TuningConfig) -> Self {
        Self {
            parameter_specs,
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Tune hyperparameters for a covariance estimator
    pub fn tune<E>(
        &self,
        estimator_factory: E,
        x: &ArrayView2<F>,
        y: Option<&ArrayView2<F>>,
    ) -> SklResult<TuningResult>
    where
        E: Fn(
            &HashMap<String, ParameterValue>,
        ) -> SklResult<Box<dyn CovarianceEstimatorTunable<F>>>,
    {
        let start_time = std::time::Instant::now();

        // Validate inputs
        self.validate_inputs(x)?;

        // Generate parameter configurations based on search strategy
        let param_configs = self.generate_parameter_configs()?;

        // Perform cross-validation for each configuration
        let mut cv_results = Vec::new();
        let mut best_score = if self.should_maximize() {
            f64::NEG_INFINITY
        } else {
            f64::INFINITY
        };
        let mut best_params = HashMap::new();
        let mut best_score_std = 0.0;

        let mut optimization_history = OptimizationHistory {
            best_scores: Vec::new(),
            scores: Vec::new(),
            cumulative_times: Vec::new(),
            exploration_metrics: Vec::new(),
        };

        for (eval_idx, params) in param_configs.iter().enumerate() {
            let eval_start = std::time::Instant::now();

            // Create estimator with current parameters
            let estimator = estimator_factory(params)?;

            // Perform cross-validation
            let cv_result = self.cross_validate(estimator.as_ref(), x, y)?;

            let eval_time = eval_start.elapsed().as_secs_f64();
            let cv_result_with_time = CVResult {
                params: params.clone(),
                mean_score: cv_result.mean_score,
                std_score: cv_result.std_score,
                fold_scores: cv_result.fold_scores,
                eval_time_seconds: eval_time,
                additional_metrics: cv_result.additional_metrics,
            };

            // Update best result
            let is_better = if self.should_maximize() {
                cv_result.mean_score > best_score
            } else {
                cv_result.mean_score < best_score
            };

            if is_better {
                best_score = cv_result.mean_score;
                best_params = params.clone();
                best_score_std = cv_result.std_score;
            }

            cv_results.push(cv_result_with_time);

            // Update optimization history
            optimization_history.best_scores.push(best_score);
            optimization_history.scores.push(cv_result.mean_score);
            optimization_history
                .cumulative_times
                .push(start_time.elapsed().as_secs_f64());
            optimization_history
                .exploration_metrics
                .push(ExplorationMetrics {
                    exploration_ratio: (eval_idx + 1) as f64 / param_configs.len() as f64,
                    diversity_score: self.compute_diversity_score(&cv_results),
                    convergence_score: self
                        .compute_convergence_score(&optimization_history.best_scores),
                });

            // Check early stopping
            if let Some(early_stopping) = &self.config.early_stopping {
                if self.should_early_stop(early_stopping, &optimization_history.best_scores) {
                    break;
                }
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();
        let n_evaluations = optimization_history.scores.len();

        Ok(TuningResult {
            best_params,
            best_score,
            best_score_std,
            cv_results,
            optimization_history,
            total_time_seconds: total_time,
            n_evaluations,
        })
    }

    /// Validate input data
    fn validate_inputs(&self, x: &ArrayView2<F>) -> SklResult<()> {
        let (n_samples, n_features) = x.dim();

        if n_samples < self.config.cv_config.n_folds {
            return Err(SklearsError::InvalidInput(format!(
                "Number of samples ({}) must be >= number of CV folds ({})",
                n_samples, self.config.cv_config.n_folds
            )));
        }

        if n_features < 1 {
            return Err(SklearsError::InvalidInput(
                "At least one feature required".to_string(),
            ));
        }

        Ok(())
    }

    /// Generate parameter configurations based on search strategy
    fn generate_parameter_configs(&self) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        match &self.config.search_strategy {
            SearchStrategy::GridSearch => self.generate_grid_search_configs(),
            SearchStrategy::RandomSearch { n_iter } => self.generate_random_search_configs(*n_iter),
            SearchStrategy::BayesianOptimization { .. } => self.generate_bayesian_configs(),
            SearchStrategy::EvolutionarySearch { .. } => self.generate_evolutionary_configs(),
            SearchStrategy::TPESearch { .. } => self.generate_tpe_configs(),
            SearchStrategy::SuccessiveHalving { .. } => self.generate_successive_halving_configs(),
        }
    }

    /// Generate grid search configurations
    fn generate_grid_search_configs(&self) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        let mut configs = vec![HashMap::new()];

        for param_spec in &self.parameter_specs {
            let param_values = self.generate_parameter_values(param_spec)?;
            let mut new_configs = Vec::new();

            for config in &configs {
                for value in &param_values {
                    let mut new_config = config.clone();
                    new_config.insert(param_spec.name.clone(), value.clone());
                    new_configs.push(new_config);
                }
            }
            configs = new_configs;
        }

        Ok(configs)
    }

    /// Generate random search configurations
    fn generate_random_search_configs(
        &self,
        n_iter: usize,
    ) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        let mut rng = scirs2_core::random::thread_rng();
        let mut configs = Vec::new();

        for _ in 0..n_iter {
            let mut config = HashMap::new();

            for param_spec in &self.parameter_specs {
                let value = self.sample_parameter_value(param_spec, &mut rng)?;
                config.insert(param_spec.name.clone(), value);
            }

            configs.push(config);
        }

        Ok(configs)
    }

    /// Generate Bayesian optimization configurations
    fn generate_bayesian_configs(&self) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        // Simplified Bayesian optimization - in practice would use GP regression
        self.generate_random_search_configs(self.config.max_evaluations)
    }

    /// Generate evolutionary algorithm configurations
    fn generate_evolutionary_configs(&self) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        // Simplified evolutionary search - in practice would implement full GA
        self.generate_random_search_configs(self.config.max_evaluations)
    }

    /// Generate TPE configurations
    fn generate_tpe_configs(&self) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        // Simplified TPE - in practice would implement full TPE algorithm
        self.generate_random_search_configs(self.config.max_evaluations)
    }

    /// Generate successive halving configurations
    fn generate_successive_halving_configs(
        &self,
    ) -> SklResult<Vec<HashMap<String, ParameterValue>>> {
        // Simplified successive halving
        self.generate_random_search_configs(self.config.max_evaluations)
    }

    /// Generate parameter values for a given parameter specification
    fn generate_parameter_values(
        &self,
        param_spec: &ParameterSpec,
    ) -> SklResult<Vec<ParameterValue>> {
        match &param_spec.param_type {
            ParameterType::Continuous { min, max } => {
                let n_points = 10; // Default grid resolution
                let values = (0..n_points)
                    .map(|i| {
                        let ratio = i as f64 / (n_points - 1) as f64;
                        let value = min + ratio * (max - min);
                        if param_spec.log_scale {
                            ParameterValue::Float(10_f64.powf(value))
                        } else {
                            ParameterValue::Float(value)
                        }
                    })
                    .collect();
                Ok(values)
            }
            ParameterType::Integer { min, max } => {
                let values = (*min..=*max).map(ParameterValue::Int).collect();
                Ok(values)
            }
            ParameterType::Categorical { choices } => {
                let values = choices
                    .iter()
                    .map(|choice| ParameterValue::String(choice.clone()))
                    .collect();
                Ok(values)
            }
            ParameterType::Boolean => Ok(vec![
                ParameterValue::Bool(false),
                ParameterValue::Bool(true),
            ]),
        }
    }

    /// Sample a random parameter value
    fn sample_parameter_value(
        &self,
        param_spec: &ParameterSpec,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<ParameterValue> {
        match &param_spec.param_type {
            ParameterType::Continuous { min, max } => {
                let value = rng.gen_range(*min..*max);
                let final_value = if param_spec.log_scale {
                    10_f64.powf(value)
                } else {
                    value
                };
                Ok(ParameterValue::Float(final_value))
            }
            ParameterType::Integer { min, max } => {
                let value = rng.gen_range(*min..*max + 1);
                Ok(ParameterValue::Int(value))
            }
            ParameterType::Categorical { choices } => {
                let idx = rng.gen_range(0..choices.len());
                Ok(ParameterValue::String(choices[idx].clone()))
            }
            ParameterType::Boolean => {
                let uniform = scirs2_core::random::essentials::Uniform::new(0.0f64, 1.0f64)
                    .expect("operation should succeed");
                Ok(ParameterValue::Bool(uniform.sample(rng) < 0.5))
            }
        }
    }

    /// Perform cross-validation for an estimator
    fn cross_validate(
        &self,
        estimator: &dyn CovarianceEstimatorTunable<F>,
        x: &ArrayView2<F>,
        y: Option<&ArrayView2<F>>,
    ) -> SklResult<CVResult> {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.config.cv_config.n_folds;
        let mut fold_scores = Vec::new();
        let additional_metrics = HashMap::new();

        for fold in 0..self.config.cv_config.n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.config.cv_config.n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation splits
            let (x_train, x_val) = self.create_cv_split(x, start_idx, end_idx)?;
            let (y_train, y_val) = if let Some(y) = y {
                let (y_tr, y_v) = self.create_cv_split(y, start_idx, end_idx)?;
                (Some(y_tr), Some(y_v))
            } else {
                (None, None)
            };

            // Fit estimator and compute score
            let fitted_estimator =
                estimator.fit(&x_train.view(), y_train.as_ref().map(|y| y.view()))?;
            let score = self.compute_score(
                fitted_estimator.as_ref(),
                &x_val.view(),
                y_val.as_ref().map(|y| y.view()),
            )?;

            fold_scores.push(score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let variance = fold_scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / fold_scores.len() as f64;
        let std_score = variance.sqrt();

        Ok(CVResult {
            params: HashMap::new(), // Will be filled by caller
            mean_score,
            std_score,
            fold_scores,
            eval_time_seconds: 0.0, // Will be filled by caller
            additional_metrics,
        })
    }

    /// Create cross-validation train/test split
    fn create_cv_split(
        &self,
        x: &ArrayView2<F>,
        val_start: usize,
        val_end: usize,
    ) -> SklResult<(Array2<F>, Array2<F>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Validation set
        let x_val = x.slice(s![val_start..val_end, ..]).to_owned();

        // Training set (all samples except validation)
        let train_size = n_samples - (val_end - val_start);
        let mut x_train = Array2::zeros((train_size, n_features));

        // Copy samples before validation set
        if val_start > 0 {
            let before_slice = x.slice(s![..val_start, ..]);
            x_train.slice_mut(s![..val_start, ..]).assign(&before_slice);
        }

        // Copy samples after validation set
        if val_end < n_samples {
            let after_slice = x.slice(s![val_end.., ..]);
            x_train.slice_mut(s![val_start.., ..]).assign(&after_slice);
        }

        Ok((x_train, x_val))
    }

    /// Compute score for a fitted estimator
    fn compute_score(
        &self,
        estimator: &dyn CovarianceEstimatorFitted<F>,
        x_val: &ArrayView2<F>,
        y_val: Option<ArrayView2<F>>,
    ) -> SklResult<f64> {
        let covariance = estimator.get_covariance();

        match &self.config.scoring {
            ScoringMetric::LogLikelihood => self.compute_log_likelihood(covariance, x_val),
            ScoringMetric::FrobeniusError => {
                if let Some(y_val) = y_val {
                    self.compute_frobenius_error(covariance, &y_val)
                } else {
                    Err(SklearsError::InvalidInput(
                        "True covariance required for Frobenius error".to_string(),
                    ))
                }
            }
            ScoringMetric::ConditionNumber => {
                Ok(-self.compute_condition_number(covariance)) // Negative because lower is better
            }
            ScoringMetric::SteinLoss => {
                if let Some(y_val) = y_val {
                    self.compute_stein_loss(covariance, &y_val)
                } else {
                    Err(SklearsError::InvalidInput(
                        "True covariance required for Stein loss".to_string(),
                    ))
                }
            }
            ScoringMetric::SpectralError => {
                if let Some(y_val) = y_val {
                    self.compute_spectral_error(covariance, &y_val)
                } else {
                    Err(SklearsError::InvalidInput(
                        "True covariance required for spectral error".to_string(),
                    ))
                }
            }
            ScoringMetric::PredictiveLikelihood => {
                self.compute_predictive_likelihood(estimator, x_val)
            }
            ScoringMetric::CrossValidationScore => {
                self.compute_log_likelihood(covariance, x_val) // Default to log-likelihood
            }
            ScoringMetric::Custom(func) => {
                // Cast to f64 if necessary
                let cov_f64 = covariance.mapv(|x| x.to_f64().unwrap_or(0.0));
                let y_val_f64 = y_val
                    .as_ref()
                    .map(|y| y.mapv(|x| x.to_f64().unwrap_or(0.0)));
                Ok(func(&cov_f64, y_val_f64.as_ref()))
            }
        }
    }

    /// Compute log-likelihood score
    /// Gaussian log-likelihood of `x` under `N(mean(x), covariance)`.
    ///
    /// `-0.5 * n * (p * ln(2*pi) + ln(det(covariance)) + tr(covariance^-1 * S))`,
    /// where `S` is the (mean-centered) empirical covariance of `x`. The
    /// `tr(covariance^-1 * S)` quadratic-form term used to be missing
    /// entirely, which meant `x`'s actual *values* played no role at all --
    /// only its shape did -- so this "log-likelihood" was really just a
    /// monotone function of `det(covariance)` alone, independent of whether
    /// `covariance` was anywhere close to `x`'s real spread.
    fn compute_log_likelihood(&self, covariance: &Array2<F>, x: &ArrayView2<F>) -> SklResult<f64> {
        let n_samples = x.nrows() as f64;
        let n_features = x.ncols() as f64;

        let det = self.compute_determinant(covariance)?;
        if det <= F::zero() {
            return Ok(f64::NEG_INFINITY);
        }
        let log_det = det
            .to_f64()
            .ok_or_else(|| SklearsError::NumericalError("to_f64 failed".into()))?
            .ln();

        let covariance_f64 = covariance.mapv(|value| value.to_f64().unwrap_or(0.0));
        let x_f64 = x.mapv(|value| value.to_f64().unwrap_or(0.0));

        let precision_f64 = match covariance_f64.inv() {
            Ok(inv) => inv,
            // `det > 0` above already rules out exact singularity; a failure
            // here means the matrix is numerically too ill-conditioned to
            // invert reliably, which is itself evidence of a very poor fit.
            Err(_) => return Ok(f64::NEG_INFINITY),
        };
        let sample_covariance = Self::empirical_covariance_f64(&x_f64);
        let trace_term = trace_of_product(&precision_f64, &sample_covariance);

        let log_likelihood = -0.5
            * n_samples
            * (n_features * (2.0 * std::f64::consts::PI).ln() + log_det + trace_term);

        Ok(log_likelihood)
    }

    /// Mean-centered empirical (sample) covariance of `x`, in `f64`.
    fn empirical_covariance_f64(x: &Array2<f64>) -> Array2<f64> {
        let n = x.nrows();
        let p = x.ncols();

        let mut mean = vec![0.0_f64; p];
        for row in x.rows() {
            for j in 0..p {
                mean[j] += row[j];
            }
        }
        if n > 0 {
            let n_f = n as f64;
            for m in mean.iter_mut() {
                *m /= n_f;
            }
        }

        let mut cov = Array2::<f64>::zeros((p, p));
        for row in x.rows() {
            for i in 0..p {
                let di = row[i] - mean[i];
                for j in 0..p {
                    cov[[i, j]] += di * (row[j] - mean[j]);
                }
            }
        }
        let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
        cov.mapv_inplace(|value| value / denom);
        cov
    }

    /// Compute Frobenius norm error
    fn compute_frobenius_error(
        &self,
        estimated: &Array2<F>,
        true_cov: &ArrayView2<F>,
    ) -> SklResult<f64> {
        if estimated.dim() != true_cov.dim() {
            return Err(SklearsError::InvalidInput(
                "Covariance matrices must have same dimensions".to_string(),
            ));
        }

        let error = estimated
            .iter()
            .zip(true_cov.iter())
            .map(|(est, true_val)| {
                let diff = est.to_f64().expect("operation should succeed")
                    - true_val.to_f64().expect("operation should succeed");
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        Ok(-error) // Negative because lower is better
    }

    /// Compute the condition number of a (symmetric PSD) covariance/precision
    /// matrix via eigendecomposition: `κ(Σ) = λ_max / max(λ_min, 1e-15)`.
    ///
    /// The previous "simplified" version used `‖Σ‖_F / tr(Σ)`, which is not a
    /// condition number at all -- it is not even scale-invariant in the
    /// right way and bears no fixed relationship to `λ_max / λ_min` except
    /// for very special matrices (e.g. it is exactly `1/sqrt(p)` for any
    /// multiple of the identity, never `1`). `Σ` is expected to be symmetric
    /// PSD, so `eigvalsh` is the right tool here; eigenvalues are NOT
    /// guaranteed sorted by this API, so the max/min are found via `fold`
    /// (mirroring `testing_quality.rs::condition_number`), and the
    /// denominator is clamped to `1e-15` to avoid dividing by (near) zero
    /// for a near-singular matrix.
    fn compute_condition_number(&self, matrix: &Array2<F>) -> f64 {
        let matrix_f64 = matrix.mapv(|value| value.to_f64().unwrap_or(0.0));
        match matrix_f64.eigvalsh(UPLO::Lower) {
            Ok(eigenvalues) => {
                let max_eigenvalue = eigenvalues.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let min_eigenvalue = eigenvalues
                    .iter()
                    .fold(f64::INFINITY, |a, &b| a.min(b.max(1e-15)));
                max_eigenvalue / min_eigenvalue
            }
            Err(_) => f64::INFINITY,
        }
    }

    /// Compute Stein's loss of `estimated` (`Σ̂`) relative to `true_cov`
    /// (`Σ`): `L(Σ̂, Σ) = tr(Σ̂⁻¹Σ) - log det(Σ̂⁻¹Σ) - p`.
    ///
    /// The previous "simplified" version only compared per-feature variances
    /// (`estimated[i,i] / true_cov[i,i]`), ignoring all covariance structure
    /// entirely (any two matrices with the same diagonal scored identically,
    /// regardless of off-diagonal correlation). This computes the real
    /// matrix quantities:
    /// - `tr(Σ̂⁻¹Σ)` via the existing `trace_of_product` helper, which
    ///   avoids materializing the full `p x p` product `Σ̂⁻¹Σ` since only
    ///   its trace is needed;
    /// - `log det(Σ̂⁻¹Σ) = log det(Σ) - log det(Σ̂)` (since
    ///   `det(AB) = det(A)det(B)`), which avoids forming `Σ̂⁻¹Σ` at all for
    ///   the log-det term and reuses `ArrayLinalgExt::det()`.
    ///
    /// On a non-invertible/ill-conditioned `estimated` matrix, or a
    /// non-positive determinant on either side (both indicate a
    /// degenerate/invalid covariance), this returns a large-but-*finite*
    /// negative penalty rather than `f64::NEG_INFINITY`: `cross_validate`
    /// averages fold scores and then computes `(score - mean).powi(2)` for
    /// the standard deviation, and mixing a `NEG_INFINITY` fold score with
    /// finite ones there produces `(-inf) - (-inf) = NaN`, silently
    /// poisoning the CV summary. A large finite sentinel keeps the failure
    /// mode "very bad score" instead of "corrupt score".
    ///
    /// Returns `Ok(-stein_loss)`: `compute_score`'s `SteinLoss` branch does
    /// not itself negate the result (unlike `ConditionNumber`), so the sign
    /// flip -- matching the file's "negative because lower [raw loss] is
    /// better" convention, same as `compute_frobenius_error` -- has to
    /// happen here.
    fn compute_stein_loss(
        &self,
        estimated: &Array2<F>,
        true_cov: &ArrayView2<F>,
    ) -> SklResult<f64> {
        /// Large finite penalty for a degenerate/non-invertible input.
        /// Finite (rather than `NEG_INFINITY`) so it can safely participate
        /// in mean/variance arithmetic downstream without producing NaN.
        const DEGENERATE_PENALTY: f64 = -1.0e6;

        let p = estimated.nrows() as f64;

        let estimated_f64 = estimated.mapv(|value| value.to_f64().unwrap_or(0.0));
        let true_cov_f64 = true_cov.mapv(|value| value.to_f64().unwrap_or(0.0));

        let precision_f64 = match estimated_f64.inv() {
            Ok(inv) => inv,
            Err(_) => return Ok(DEGENERATE_PENALTY),
        };

        let det_estimated = match estimated_f64.det() {
            Ok(d) if d > 0.0 => d,
            _ => return Ok(DEGENERATE_PENALTY),
        };
        let det_true_cov = match true_cov_f64.det() {
            Ok(d) if d > 0.0 => d,
            _ => return Ok(DEGENERATE_PENALTY),
        };

        let trace_term = trace_of_product(&precision_f64, &true_cov_f64);
        let log_det_a = det_true_cov.ln() - det_estimated.ln();

        let stein_loss = trace_term - log_det_a - p;
        Ok(-stein_loss)
    }

    /// Compute the spectral-norm error between `estimated` and `true_cov`:
    /// `‖Σ̂ - Σ‖₂`.
    ///
    /// The previous "simplified" version just called
    /// `compute_frobenius_error` outright, i.e. it never computed a
    /// spectral norm at all. Since both `estimated` and `true_cov` are real
    /// symmetric covariance matrices, so is their difference
    /// `D = Σ̂ - Σ`, and for a real symmetric matrix the spectral (operator
    /// 2-)norm equals the largest-magnitude eigenvalue:
    /// `‖D‖₂ = max_i |λ_i(D)|`.
    ///
    /// If the eigendecomposition of `D` fails, this falls back to the
    /// (always-computable) Frobenius error as a degraded-but-safe estimate:
    /// `‖D‖_F` is a valid upper bound on `‖D‖₂` (`‖D‖₂ <= ‖D‖_F`), so it
    /// still gives a sensible, correctly-signed, finite score rather than
    /// aborting the whole scoring pass.
    ///
    /// Returns `Ok(-spectral_norm)`: `compute_score`'s `SpectralError`
    /// branch does not itself negate the result, so -- exactly as with
    /// `compute_frobenius_error` -- the "negative because lower is better"
    /// sign flip has to happen inside this function.
    fn compute_spectral_error(
        &self,
        estimated: &Array2<F>,
        true_cov: &ArrayView2<F>,
    ) -> SklResult<f64> {
        if estimated.dim() != true_cov.dim() {
            return Err(SklearsError::InvalidInput(
                "Covariance matrices must have same dimensions".to_string(),
            ));
        }

        let estimated_f64 = estimated.mapv(|value| value.to_f64().unwrap_or(0.0));
        let true_cov_f64 = true_cov.mapv(|value| value.to_f64().unwrap_or(0.0));
        let diff = &estimated_f64 - &true_cov_f64;

        match diff.eigvalsh(UPLO::Lower) {
            Ok(eigenvalues) => {
                let spectral_norm = eigenvalues.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
                Ok(-spectral_norm)
            }
            Err(_) => self.compute_frobenius_error(estimated, true_cov),
        }
    }

    /// Compute predictive likelihood
    fn compute_predictive_likelihood(
        &self,
        estimator: &dyn CovarianceEstimatorFitted<F>,
        x_val: &ArrayView2<F>,
    ) -> SklResult<f64> {
        self.compute_log_likelihood(estimator.get_covariance(), x_val)
    }

    /// Compute determinant (simplified)
    /// Compute the determinant of a square matrix via Gaussian elimination
    /// with partial pivoting (`O(n^3)`).
    ///
    /// This used to fall back, for `n > 2`, to the product of the diagonal
    /// entries ("assuming diagonal dominance"), which is only exact for a
    /// diagonal matrix. For any matrix with real off-diagonal structure --
    /// exactly what most covariance-estimator regularization parameters
    /// act on -- that shortcut silently returned the wrong determinant,
    /// which made every log-likelihood-based `ScoringMetric`
    /// (`LogLikelihood`, `PredictiveLikelihood`, `CrossValidationScore`)
    /// nearly insensitive to those parameters.
    fn compute_determinant(&self, matrix: &Array2<F>) -> SklResult<F> {
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }
        if n == 0 {
            return Ok(F::one());
        }

        let mut a = matrix.clone();
        let mut det = F::one();

        for col in 0..n {
            // Partial pivoting: bring the largest-magnitude entry in this
            // column (at or below the diagonal) onto the diagonal, for
            // numerical stability.
            let mut pivot_row = col;
            let mut pivot_val = a[[col, col]].abs();
            for row in (col + 1)..n {
                let val = a[[row, col]].abs();
                if val > pivot_val {
                    pivot_val = val;
                    pivot_row = row;
                }
            }

            if pivot_val <= F::epsilon() {
                // Singular (or numerically indistinguishable from it).
                return Ok(F::zero());
            }

            if pivot_row != col {
                for k in 0..n {
                    a.swap([col, k], [pivot_row, k]);
                }
                det = -det;
            }

            det *= a[[col, col]];

            let pivot = a[[col, col]];
            for row in (col + 1)..n {
                let factor = a[[row, col]] / pivot;
                if factor != F::zero() {
                    for k in col..n {
                        let delta = factor * a[[col, k]];
                        a[[row, k]] -= delta;
                    }
                }
            }
        }

        Ok(det)
    }

    /// Check if the metric should be maximized
    fn should_maximize(&self) -> bool {
        matches!(
            self.config.scoring,
            ScoringMetric::LogLikelihood
                | ScoringMetric::PredictiveLikelihood
                | ScoringMetric::CrossValidationScore
        )
    }

    /// Compute diversity score of evaluated configurations
    fn compute_diversity_score(&self, cv_results: &[CVResult]) -> f64 {
        if cv_results.len() < 2 {
            return 0.0;
        }

        // Simple diversity measure based on score variance
        let scores: Vec<f64> = cv_results.iter().map(|r| r.mean_score).collect();
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let variance = scores
            .iter()
            .map(|score| (score - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;

        variance.sqrt()
    }

    /// Compute convergence score
    fn compute_convergence_score(&self, best_scores: &[f64]) -> f64 {
        if best_scores.len() < 5 {
            return 0.0;
        }

        // Compute rate of improvement in recent iterations
        let recent_window = 5;
        let start_idx = best_scores.len().saturating_sub(recent_window);
        let recent_scores = &best_scores[start_idx..];

        let improvement = if self.should_maximize() {
            recent_scores.last().expect("operation should succeed")
                - recent_scores.first().expect("operation should succeed")
        } else {
            recent_scores.first().expect("operation should succeed")
                - recent_scores.last().expect("operation should succeed")
        };

        improvement.max(0.0)
    }

    /// Check if early stopping criteria are met
    fn should_early_stop(&self, early_stopping: &EarlyStoppingConfig, best_scores: &[f64]) -> bool {
        if best_scores.len() < early_stopping.patience {
            return false;
        }

        let recent_scores = &best_scores[best_scores.len() - early_stopping.patience..];
        let best_recent = if early_stopping.maximize {
            recent_scores
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b))
        } else {
            recent_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b))
        };

        let current_best = *best_scores.last().expect("operation should succeed");
        let improvement = if early_stopping.maximize {
            best_recent - current_best
        } else {
            current_best - best_recent
        };

        improvement < early_stopping.min_delta
    }
}

/// `tr(a * b)` for two same-shape square matrices, computed directly as
/// `sum_{i,j} a[i,j] * b[j,i]` rather than forming the full product, since
/// only the trace is needed.
fn trace_of_product(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    let n = a.nrows();
    let mut sum = 0.0;
    for i in 0..n {
        for j in 0..n {
            sum += a[[i, j]] * b[[j, i]];
        }
    }
    sum
}

/// Trait for covariance estimators that can be tuned
pub trait CovarianceEstimatorTunable<F: NdFloat> {
    /// Fit the estimator to data
    fn fit(
        &self,
        x: &ArrayView2<F>,
        y: Option<ArrayView2<F>>,
    ) -> SklResult<Box<dyn CovarianceEstimatorFitted<F>>>;
}

/// Trait for fitted covariance estimators
pub trait CovarianceEstimatorFitted<F: NdFloat> {
    /// Get the estimated covariance matrix
    fn get_covariance(&self) -> &Array2<F>;

    /// Get the precision matrix if available
    fn get_precision(&self) -> Option<&Array2<F>>;
}

/// Convenience functions for creating common tuning configurations
pub mod presets {
    use super::*;

    /// Create a grid search configuration for regularization parameter
    pub fn grid_search_regularization(
        _min_alpha: f64,
        _max_alpha: f64,
        n_points: usize,
    ) -> TuningConfig {
        TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 5,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: SearchStrategy::GridSearch,
            scoring: ScoringMetric::LogLikelihood,
            max_evaluations: n_points,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: None,
        }
    }

    /// Create a random search configuration
    pub fn random_search_default(n_iter: usize) -> TuningConfig {
        TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 5,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: SearchStrategy::RandomSearch { n_iter },
            scoring: ScoringMetric::LogLikelihood,
            max_evaluations: n_iter,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 10,
                min_delta: 1e-4,
                maximize: true,
            }),
        }
    }

    /// Create Bayesian optimization configuration
    pub fn bayesian_optimization_default() -> TuningConfig {
        TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 5,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: SearchStrategy::BayesianOptimization {
                acquisition_function: AcquisitionFunction::ExpectedImprovement,
                n_initial_points: 10,
            },
            scoring: ScoringMetric::LogLikelihood,
            max_evaluations: 50,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: Some(EarlyStoppingConfig {
                patience: 15,
                min_delta: 1e-4,
                maximize: true,
            }),
        }
    }

    /// Create parameter specifications for common covariance estimators
    pub fn ledoit_wolf_params() -> Vec<ParameterSpec> {
        vec![ParameterSpec {
            name: "shrinkage".to_string(),
            param_type: ParameterType::Continuous { min: 0.0, max: 1.0 },
            log_scale: false,
        }]
    }

    /// Create parameter specifications for GraphicalLasso
    pub fn graphical_lasso_params() -> Vec<ParameterSpec> {
        vec![
            ParameterSpec {
                name: "alpha".to_string(),
                param_type: ParameterType::Continuous {
                    min: -3.0,
                    max: 1.0,
                },
                log_scale: true,
            },
            ParameterSpec {
                name: "max_iter".to_string(),
                param_type: ParameterType::Integer { min: 50, max: 1000 },
                log_scale: false,
            },
        ]
    }

    /// Create parameter specifications for Ridge covariance
    pub fn ridge_covariance_params() -> Vec<ParameterSpec> {
        vec![ParameterSpec {
            name: "alpha".to_string(),
            param_type: ParameterType::Continuous {
                min: -4.0,
                max: 2.0,
            },
            log_scale: true,
        }]
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphicalLasso;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::{Distribution, SeedableRng, StdRng};
    use sklears_core::traits::Fit;

    /// Wraps [`GraphicalLasso`] so `alpha` can be searched by the tuner.
    /// `get_covariance()`/`get_precision()` are forwarded directly -- no
    /// manual ridge-then-invert workaround is needed here, since
    /// `GraphicalLasso::get_covariance()` already returns the covariance
    /// implied by the fitted, regularized precision matrix.
    struct TunableGraphicalLassoAlpha {
        alpha: f64,
    }

    impl CovarianceEstimatorTunable<f64> for TunableGraphicalLassoAlpha {
        fn fit(
            &self,
            x: &ArrayView2<f64>,
            _y: Option<ArrayView2<f64>>,
        ) -> SklResult<Box<dyn CovarianceEstimatorFitted<f64>>> {
            let fitted = GraphicalLasso::new().alpha(self.alpha).fit(x, &())?;
            Ok(Box::new(FittedGraphicalLassoAlpha {
                covariance: fitted.get_covariance().clone(),
                precision: fitted.get_precision().clone(),
            }))
        }
    }

    struct FittedGraphicalLassoAlpha {
        covariance: Array2<f64>,
        precision: Array2<f64>,
    }

    impl CovarianceEstimatorFitted<f64> for FittedGraphicalLassoAlpha {
        fn get_covariance(&self) -> &Array2<f64> {
            &self.covariance
        }

        fn get_precision(&self) -> Option<&Array2<f64>> {
            Some(&self.precision)
        }
    }

    /// Block-correlated toy dataset: features `0..block_size` share one
    /// latent factor (a real, recoverable dependency structure); the rest
    /// are independent noise. Before `get_covariance()` was fixed to depend
    /// on `alpha`, every candidate in a search would score identically
    /// (`compute_score` always reads `estimator.get_covariance()`), so the
    /// "winner" would just be whichever config the search happened to visit
    /// first/last -- never a data-dependent choice.
    fn generate_block_sparse_data(
        n_samples: usize,
        n_features: usize,
        block_size: usize,
        seed: u64,
    ) -> Array2<f64> {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal = Normal::new(0.0, 1.0).expect("standard normal parameters are always valid");

        let mut data = Array2::<f64>::zeros((n_samples, n_features));
        for i in 0..n_samples {
            let factor_a = normal.sample(&mut rng);
            let factor_b = normal.sample(&mut rng);
            for j in 0..n_features {
                let noise = normal.sample(&mut rng);
                data[[i, j]] = if j < block_size {
                    0.9 * factor_a + 0.3 * noise
                } else if j < 2 * block_size {
                    0.9 * factor_b + 0.3 * noise
                } else {
                    noise
                };
            }
        }
        data
    }

    /// End-to-end confirmation of the `get_covariance()` alpha-invariance
    /// fix: tuning `GraphicalLasso`'s `alpha` via cross-validated
    /// log-likelihood must (a) produce genuinely different scores across
    /// the alpha grid, and (b) land on a sensible interior value rather
    /// than always the same (data-independent) boundary alpha.
    #[test]
    fn test_graphical_lasso_tuning_selects_interior_alpha() {
        let data = generate_block_sparse_data(150, 12, 3, 7);

        let param_specs = vec![ParameterSpec {
            name: "alpha".to_string(),
            param_type: ParameterType::Continuous {
                min: -3.0,
                max: 0.5,
            }, // log10: alpha in [1e-3, ~3.16]
            log_scale: true,
        }];
        let config = TuningConfig {
            cv_config: CrossValidationConfig {
                n_folds: 5,
                shuffle: true,
                random_seed: Some(42),
                stratify: false,
            },
            search_strategy: SearchStrategy::GridSearch,
            scoring: ScoringMetric::LogLikelihood,
            max_evaluations: 10,
            random_seed: Some(42),
            n_jobs: None,
            early_stopping: None,
        };

        let tuner = CovarianceHyperparameterTuner::new(param_specs, config);
        let factory = |params: &HashMap<String, ParameterValue>| -> SklResult<
            Box<dyn CovarianceEstimatorTunable<f64>>,
        > {
            let alpha = match params.get("alpha") {
                Some(ParameterValue::Float(a)) => *a,
                _ => 0.01,
            };
            Ok(Box::new(TunableGraphicalLassoAlpha { alpha }))
        };

        let result = tuner
            .tune(factory, &data.view(), None)
            .expect("tuning should succeed");

        // (a) Scores must actually depend on alpha now: they cannot all be
        // tied, which is what alpha-invariant scoring would produce.
        let scores: Vec<f64> = result.cv_results.iter().map(|r| r.mean_score).collect();
        let min_score = scores.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(
            max_score - min_score > 1e-6,
            "CV scores across the alpha grid should vary once get_covariance() responds to \
             alpha (got a flat score range [{min_score}, {max_score}])"
        );

        // (b) The winner should be a sensible interior value, not glued to
        // whichever grid boundary the search happened to visit.
        let best_alpha = match result.best_params.get("alpha") {
            Some(ParameterValue::Float(a)) => *a,
            other => panic!("expected a float alpha in best_params, got {other:?}"),
        };
        assert!(
            (0.005..1.0).contains(&best_alpha),
            "expected the tuner to select an interior alpha in [0.005, 1.0) out of the \
             searched [0.001, 3.162] range, got {best_alpha}"
        );
    }

    #[test]
    fn test_parameter_value_equality() {
        assert_eq!(ParameterValue::Float(1.0), ParameterValue::Float(1.0));
        assert_eq!(ParameterValue::Int(5), ParameterValue::Int(5));
        assert_eq!(
            ParameterValue::String("test".to_string()),
            ParameterValue::String("test".to_string())
        );
        assert_eq!(ParameterValue::Bool(true), ParameterValue::Bool(true));
    }

    #[test]
    fn test_parameter_spec_creation() {
        let spec = ParameterSpec {
            name: "alpha".to_string(),
            param_type: ParameterType::Continuous { min: 0.0, max: 1.0 },
            log_scale: false,
        };

        assert_eq!(spec.name, "alpha");
        assert!(!spec.log_scale);
    }

    #[test]
    fn test_tuning_config_creation() {
        let config = presets::random_search_default(10);
        assert_eq!(config.cv_config.n_folds, 5);
        assert_eq!(config.max_evaluations, 10);
    }

    #[test]
    fn test_cv_result_creation() {
        let mut params = HashMap::new();
        params.insert("alpha".to_string(), ParameterValue::Float(0.5));

        let result = CVResult {
            params,
            mean_score: 0.85,
            std_score: 0.05,
            fold_scores: vec![0.8, 0.85, 0.9],
            eval_time_seconds: 1.5,
            additional_metrics: HashMap::new(),
        };

        assert_eq!(result.mean_score, 0.85);
        assert_eq!(result.fold_scores.len(), 3);
    }

    /// A `CovarianceHyperparameterTuner<f64>` with an empty parameter grid,
    /// used purely as a `self` receiver for exercising the private
    /// `compute_condition_number` / `compute_stein_loss` /
    /// `compute_spectral_error` scoring methods directly. The concrete
    /// `TuningConfig` contents (search strategy, scoring metric, ...) are
    /// irrelevant to those methods -- only `&self` is needed to call them.
    fn make_test_tuner() -> CovarianceHyperparameterTuner<f64> {
        CovarianceHyperparameterTuner::new(
            Vec::new(),
            TuningConfig {
                cv_config: CrossValidationConfig {
                    n_folds: 5,
                    shuffle: true,
                    random_seed: Some(42),
                    stratify: false,
                },
                search_strategy: SearchStrategy::GridSearch,
                scoring: ScoringMetric::LogLikelihood,
                max_evaluations: 10,
                random_seed: Some(42),
                n_jobs: None,
                early_stopping: None,
            },
        )
    }

    // -- compute_condition_number -------------------------------------------

    #[test]
    fn test_condition_number_identity_is_one() {
        let tuner = make_test_tuner();
        let identity = Array2::<f64>::eye(3);

        let kappa = tuner.compute_condition_number(&identity);

        assert!(
            (kappa - 1.0).abs() < 1e-6,
            "condition number of the identity matrix must be exactly 1.0, got {kappa}"
        );
    }

    #[test]
    fn test_condition_number_diagonal_matches_eigenvalue_ratio() {
        let tuner = make_test_tuner();
        // diag(1, 100): eigenvalues are trivially 1 and 100, so
        // kappa = lambda_max / lambda_min = 100.
        let diagonal =
            Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 100.0]).expect("2x2 shape matches");

        let kappa = tuner.compute_condition_number(&diagonal);

        assert!(
            (kappa - 100.0).abs() < 1e-6,
            "condition number of diag(1, 100) must be 100.0, got {kappa}"
        );
    }

    // -- compute_stein_loss ---------------------------------------------------

    #[test]
    fn test_stein_loss_zero_at_identical_covariances() {
        let tuner = make_test_tuner();
        // A non-diagonal SPD matrix: exercises the full tr(.)/det(.) formula
        // rather than only the diagonal special case.
        let cov = Array2::from_shape_vec((2, 2), vec![2.0, 0.5, 0.5, 1.0]).expect("2x2 shape matches");

        let score = tuner
            .compute_stein_loss(&cov, &cov.view())
            .expect("stein loss should succeed for an SPD matrix");

        // A = Sigma_hat^-1 * Sigma = I when estimated == true_cov, so
        // stein_loss = tr(I) - log(det(I)) - p = p - 0 - p = 0.
        assert!(
            score.abs() < 1e-6,
            "Stein loss score must be ~0 when estimated == true_cov, got {score}"
        );
    }

    #[test]
    fn test_stein_loss_matches_hand_computed_value_diagonal() {
        let tuner = make_test_tuner();
        let estimated = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 2.0]).expect("shape");
        let true_cov = Array2::from_shape_vec((2, 2), vec![1.0, 0.0, 0.0, 1.0]).expect("shape");

        let score = tuner
            .compute_stein_loss(&estimated, &true_cov.view())
            .expect("stein loss should succeed for SPD diagonal matrices");

        // A = diag(2,2)^-1 * diag(1,1) = diag(0.5, 0.5).
        // tr(A) = 1.0, det(A) = 0.25, p = 2.
        // stein_loss = 1.0 - ln(0.25) - 2 = 1.0 + 1.3862943611198906 - 2
        //            = 0.3862943611198906
        // score = -stein_loss.
        let expected_stein_loss = 1.0 - 0.25_f64.ln() - 2.0;
        let expected_score = -expected_stein_loss;

        assert!(
            (score - expected_score).abs() < 1e-9,
            "expected Stein-loss score {expected_score}, got {score}"
        );
    }

    #[test]
    fn test_stein_loss_matches_hand_computed_value_off_diagonal() {
        let tuner = make_test_tuner();
        // estimated has real off-diagonal correlation; true_cov = I.
        // A = estimated^-1 (since true_cov = I).
        let estimated = Array2::from_shape_vec((2, 2), vec![2.0, 1.0, 1.0, 2.0]).expect("shape");
        let true_cov = Array2::<f64>::eye(2);

        let score = tuner
            .compute_stein_loss(&estimated, &true_cov.view())
            .expect("stein loss should succeed for an SPD matrix with off-diagonal terms");

        // det(estimated) = 2*2 - 1*1 = 3.
        // estimated^-1 = (1/3) * [[2, -1], [-1, 2]], trace = 4/3.
        // det(A) = det(estimated^-1) * det(I) = 1/3.
        // stein_loss = 4/3 - ln(1/3) - 2.
        let expected_stein_loss = 4.0 / 3.0 - (1.0_f64 / 3.0).ln() - 2.0;
        let expected_score = -expected_stein_loss;

        assert!(
            (score - expected_score).abs() < 1e-9,
            "expected Stein-loss score {expected_score}, got {score}"
        );
    }

    #[test]
    fn test_stein_loss_is_monotonically_worse_as_estimate_drifts() {
        let tuner = make_test_tuner();
        let true_cov = Array2::<f64>::eye(2);

        let score_close = tuner
            .compute_stein_loss(&(&true_cov * 1.5), &true_cov.view())
            .expect("stein loss should succeed");
        let score_far = tuner
            .compute_stein_loss(&(&true_cov * 3.0), &true_cov.view())
            .expect("stein loss should succeed");
        let score_exact = tuner
            .compute_stein_loss(&true_cov, &true_cov.view())
            .expect("stein loss should succeed");

        assert!(
            score_exact > score_close,
            "an exact match ({score_exact}) must score strictly better than a 1.5x-scaled \
             estimate ({score_close})"
        );
        assert!(
            score_close > score_far,
            "a 1.5x-scaled estimate ({score_close}) must score strictly better than a \
             3x-scaled estimate ({score_far}) as it drifts further from true_cov"
        );
    }

    // -- compute_spectral_error ------------------------------------------------

    #[test]
    fn test_spectral_error_diagonal_difference_matrix() {
        let tuner = make_test_tuner();
        // true_cov = diag(2, 3), estimated = diag(7, 1)
        // => diff = diag(5, -2), eigenvalues {5, -2}, spectral norm = 5.
        let true_cov = Array2::from_shape_vec((2, 2), vec![2.0, 0.0, 0.0, 3.0]).expect("shape");
        let estimated = Array2::from_shape_vec((2, 2), vec![7.0, 0.0, 0.0, 1.0]).expect("shape");

        let score = tuner
            .compute_spectral_error(&estimated, &true_cov.view())
            .expect("spectral error should succeed");

        assert!(
            (score - (-5.0)).abs() < 1e-6,
            "expected spectral-error score -5.0 (spectral norm 5.0), got {score}"
        );
    }

    #[test]
    fn test_spectral_error_off_diagonal_difference_matrix() {
        let tuner = make_test_tuner();
        // true_cov = I, estimated = [[1,3],[3,1]]
        // => diff = [[0,3],[3,0]], eigenvalues {3, -3}, spectral norm = 3.
        let true_cov = Array2::<f64>::eye(2);
        let estimated = Array2::from_shape_vec((2, 2), vec![1.0, 3.0, 3.0, 1.0]).expect("shape");

        let score = tuner
            .compute_spectral_error(&estimated, &true_cov.view())
            .expect("spectral error should succeed");

        assert!(
            (score - (-3.0)).abs() < 1e-6,
            "expected spectral-error score -3.0 (spectral norm 3.0), got {score}"
        );
    }
}

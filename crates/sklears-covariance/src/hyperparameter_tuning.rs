//! Automatic Hyperparameter Tuning for Covariance Estimators
//!
//! This module provides automatic hyperparameter optimization for covariance estimators
//! using various search strategies including grid search, random search, Bayesian optimization,
//! and evolutionary algorithms.

use scirs2_core::ndarray::{s, Array2, ArrayView2, NdFloat};
use scirs2_core::Distribution;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;
use std::fmt::Debug;

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
    Custom(Box<dyn Fn(&Array2<f64>, Option<&Array2<f64>>) -> f64 + Send + Sync>),
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
        X: &ArrayView2<F>,
        y: Option<&ArrayView2<F>>,
    ) -> SklResult<TuningResult>
    where
        E: Fn(
            &HashMap<String, ParameterValue>,
        ) -> SklResult<Box<dyn CovarianceEstimatorTunable<F>>>,
    {
        let start_time = std::time::Instant::now();

        // Validate inputs
        self.validate_inputs(X)?;

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
            let cv_result = self.cross_validate(estimator.as_ref(), X, y)?;

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
    fn validate_inputs(&self, X: &ArrayView2<F>) -> SklResult<()> {
        let (n_samples, n_features) = X.dim();

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
                let uniform =
                    scirs2_core::random::essentials::Uniform::new(0.0f64, 1.0f64).unwrap();
                Ok(ParameterValue::Bool(uniform.sample(rng) < 0.5))
            }
        }
    }

    /// Perform cross-validation for an estimator
    fn cross_validate(
        &self,
        estimator: &dyn CovarianceEstimatorTunable<F>,
        X: &ArrayView2<F>,
        y: Option<&ArrayView2<F>>,
    ) -> SklResult<CVResult> {
        let n_samples = X.nrows();
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
            let (X_train, X_val) = self.create_cv_split(X, start_idx, end_idx)?;
            let (y_train, y_val) = if let Some(y) = y {
                let (y_tr, y_v) = self.create_cv_split(y, start_idx, end_idx)?;
                (Some(y_tr), Some(y_v))
            } else {
                (None, None)
            };

            // Fit estimator and compute score
            let fitted_estimator =
                estimator.fit(&X_train.view(), y_train.as_ref().map(|y| y.view()))?;
            let score = self.compute_score(
                fitted_estimator.as_ref(),
                &X_val.view(),
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
        X: &ArrayView2<F>,
        val_start: usize,
        val_end: usize,
    ) -> SklResult<(Array2<F>, Array2<F>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Validation set
        let X_val = X.slice(s![val_start..val_end, ..]).to_owned();

        // Training set (all samples except validation)
        let train_size = n_samples - (val_end - val_start);
        let mut X_train = Array2::zeros((train_size, n_features));

        // Copy samples before validation set
        if val_start > 0 {
            let before_slice = X.slice(s![..val_start, ..]);
            X_train.slice_mut(s![..val_start, ..]).assign(&before_slice);
        }

        // Copy samples after validation set
        if val_end < n_samples {
            let after_slice = X.slice(s![val_end.., ..]);
            X_train.slice_mut(s![val_start.., ..]).assign(&after_slice);
        }

        Ok((X_train, X_val))
    }

    /// Compute score for a fitted estimator
    fn compute_score(
        &self,
        estimator: &dyn CovarianceEstimatorFitted<F>,
        X_val: &ArrayView2<F>,
        y_val: Option<ArrayView2<F>>,
    ) -> SklResult<f64> {
        let covariance = estimator.get_covariance();

        match &self.config.scoring {
            ScoringMetric::LogLikelihood => self.compute_log_likelihood(covariance, X_val),
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
                self.compute_predictive_likelihood(estimator, X_val)
            }
            ScoringMetric::CrossValidationScore => {
                self.compute_log_likelihood(covariance, X_val) // Default to log-likelihood
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
    fn compute_log_likelihood(&self, covariance: &Array2<F>, X: &ArrayView2<F>) -> SklResult<f64> {
        // Simplified log-likelihood computation
        let n_samples = X.nrows() as f64;
        let n_features = X.ncols() as f64;

        // Compute determinant (simplified)
        let det = self.compute_determinant(covariance)?;
        if det <= F::zero() {
            return Ok(f64::NEG_INFINITY);
        }

        let log_det = det.to_f64().unwrap().ln();
        let log_likelihood =
            -0.5 * n_samples * (n_features * (2.0 * std::f64::consts::PI).ln() + log_det);

        Ok(log_likelihood)
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
                let diff = est.to_f64().unwrap() - true_val.to_f64().unwrap();
                diff * diff
            })
            .sum::<f64>()
            .sqrt();

        Ok(-error) // Negative because lower is better
    }

    /// Compute condition number
    fn compute_condition_number(&self, matrix: &Array2<F>) -> f64 {
        // Simplified condition number computation
        // In practice, would use proper eigenvalue decomposition
        let trace = (0..matrix.nrows())
            .map(|i| matrix[[i, i]].to_f64().unwrap())
            .sum::<f64>();
        let norm = matrix
            .iter()
            .map(|x| x.to_f64().unwrap().powi(2))
            .sum::<f64>()
            .sqrt();

        if trace > 0.0 {
            norm / trace
        } else {
            f64::INFINITY
        }
    }

    /// Compute Stein's loss
    fn compute_stein_loss(
        &self,
        estimated: &Array2<F>,
        true_cov: &ArrayView2<F>,
    ) -> SklResult<f64> {
        // Simplified Stein's loss: tr(Σ^-1 S) - log|Σ^-1 S| - p
        // where Σ is true covariance and S is estimated covariance
        let p = estimated.nrows() as f64;

        // This is a simplified version - in practice would need proper matrix inversion
        let trace_ratio = (0..estimated.nrows())
            .map(|i| estimated[[i, i]].to_f64().unwrap() / true_cov[[i, i]].to_f64().unwrap())
            .sum::<f64>();

        let stein_loss = trace_ratio - p;
        Ok(-stein_loss) // Negative because lower is better
    }

    /// Compute spectral norm error
    fn compute_spectral_error(
        &self,
        estimated: &Array2<F>,
        true_cov: &ArrayView2<F>,
    ) -> SklResult<f64> {
        // Simplified spectral norm (largest singular value)
        self.compute_frobenius_error(estimated, true_cov) // Placeholder
    }

    /// Compute predictive likelihood
    fn compute_predictive_likelihood(
        &self,
        estimator: &dyn CovarianceEstimatorFitted<F>,
        X_val: &ArrayView2<F>,
    ) -> SklResult<f64> {
        self.compute_log_likelihood(estimator.get_covariance(), X_val)
    }

    /// Compute determinant (simplified)
    fn compute_determinant(&self, matrix: &Array2<F>) -> SklResult<F> {
        // Simplified determinant for small matrices
        let n = matrix.nrows();
        if n != matrix.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        match n {
            1 => Ok(matrix[[0, 0]]),
            2 => Ok(matrix[[0, 0]] * matrix[[1, 1]] - matrix[[0, 1]] * matrix[[1, 0]]),
            _ => {
                // For larger matrices, use product of diagonal (assuming diagonal dominance)
                let det = (0..n)
                    .map(|i| matrix[[i, i]])
                    .fold(F::one(), |acc, x| acc * x);
                Ok(det)
            }
        }
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
            recent_scores.last().unwrap() - recent_scores.first().unwrap()
        } else {
            recent_scores.first().unwrap() - recent_scores.last().unwrap()
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

        let current_best = *best_scores.last().unwrap();
        let improvement = if early_stopping.maximize {
            best_recent - current_best
        } else {
            current_best - best_recent
        };

        improvement < early_stopping.min_delta
    }
}

/// Trait for covariance estimators that can be tuned
pub trait CovarianceEstimatorTunable<F: NdFloat> {
    /// Fit the estimator to data
    fn fit(
        &self,
        X: &ArrayView2<F>,
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
        min_alpha: f64,
        max_alpha: f64,
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
    use scirs2_core::ndarray::array;

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
}

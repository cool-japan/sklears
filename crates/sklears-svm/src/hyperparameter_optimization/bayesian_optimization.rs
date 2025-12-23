//! Bayesian Optimization for hyperparameter tuning

use std::time::Instant;

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;

use crate::kernels::KernelType;
use crate::svc::SVC;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};

use super::{
    OptimizationConfig, OptimizationResult, ParameterSet, ParameterSpec, ScoringMetric, SearchSpace,
};

// Type aliases for compatibility
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;

/// Bayesian Optimization hyperparameter optimizer
pub struct BayesianOptimizationCV {
    config: OptimizationConfig,
    search_space: SearchSpace,
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl BayesianOptimizationCV {
    /// Create a new Bayesian optimization optimizer
    pub fn new(config: OptimizationConfig, search_space: SearchSpace) -> Self {
        let rng = if let Some(seed) = config.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42) // Default seed for reproducibility
        };

        Self {
            config,
            search_space,
            rng,
        }
    }

    /// Run Bayesian optimization
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<OptimizationResult> {
        let start_time = Instant::now();

        if self.config.verbose {
            println!(
                "Bayesian optimization with {} iterations",
                self.config.n_iterations
            );
        }

        // Initialize with random samples
        let n_initial = (self.config.n_iterations / 5).clamp(5, 20);
        let mut evaluated_params: Vec<(ParameterSet, f64)> = Vec::new();
        let mut best_score = -f64::INFINITY;
        let mut best_params = ParameterSet::new();
        let mut score_history = Vec::new();
        let mut iterations_without_improvement = 0;

        // Phase 1: Random exploration
        if self.config.verbose {
            println!("Phase 1: Random exploration ({} samples)", n_initial);
        }

        for i in 0..n_initial {
            let params = self.sample_random_params()?;
            let score = self.evaluate_params(&params, x, y)?;

            evaluated_params.push((params.clone(), score));
            score_history.push(score);

            if score > best_score {
                best_score = score;
                best_params = params.clone();
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement += 1;
            }

            if self.config.verbose {
                println!("Sample {}/{}: Score {:.6}", i + 1, n_initial, score);
            }
        }

        // Phase 2: Bayesian optimization with Expected Improvement
        if self.config.verbose {
            println!("Phase 2: Bayesian optimization");
        }

        for iteration in n_initial..self.config.n_iterations {
            // Build surrogate model (Gaussian Process approximation)
            // Select next point using Expected Improvement acquisition function
            let next_params = self.select_next_point(&evaluated_params)?;

            // Evaluate the selected point
            let score = self.evaluate_params(&next_params, x, y)?;

            evaluated_params.push((next_params.clone(), score));
            score_history.push(score);

            if score > best_score {
                best_score = score;
                best_params = next_params.clone();
                iterations_without_improvement = 0;

                if self.config.verbose {
                    println!(
                        "Iteration {}/{}: NEW BEST Score {:.6}",
                        iteration + 1,
                        self.config.n_iterations,
                        score
                    );
                }
            } else {
                iterations_without_improvement += 1;

                if self.config.verbose && (iteration + 1) % 10 == 0 {
                    println!(
                        "Iteration {}/{}: Score {:.6} (best: {:.6})",
                        iteration + 1,
                        self.config.n_iterations,
                        score,
                        best_score
                    );
                }
            }

            // Early stopping
            if let Some(patience) = self.config.early_stopping_patience {
                if iterations_without_improvement >= patience {
                    if self.config.verbose {
                        println!("Early stopping at iteration {}", iteration + 1);
                    }
                    break;
                }
            }
        }

        if self.config.verbose {
            println!("Best score: {:.6}", best_score);
            println!("Best params: {:?}", best_params);
        }

        Ok(OptimizationResult {
            best_params,
            best_score,
            cv_results: evaluated_params,
            n_iterations: score_history.len(),
            optimization_time: start_time.elapsed().as_secs_f64(),
            score_history,
        })
    }

    /// Sample random parameters from search space
    fn sample_random_params(&mut self) -> Result<ParameterSet> {
        // Clone search space specs to avoid borrow checker issues
        let c_spec = self.search_space.c.clone();
        let kernel_spec = self.search_space.kernel.clone();
        let tol_spec = self.search_space.tol.clone();
        let max_iter_spec = self.search_space.max_iter.clone();

        let c = self.sample_value(&c_spec)?;

        let kernel = if let Some(ref spec) = kernel_spec {
            self.sample_kernel(spec)?
        } else {
            KernelType::Rbf { gamma: 1.0 }
        };

        let tol = if let Some(ref spec) = tol_spec {
            self.sample_value(spec)?
        } else {
            1e-3
        };

        let max_iter = if let Some(ref spec) = max_iter_spec {
            self.sample_value(spec)? as usize
        } else {
            1000
        };

        Ok(ParameterSet {
            c,
            kernel,
            tol,
            max_iter,
        })
    }

    /// Select next point to evaluate using Expected Improvement
    fn select_next_point(&mut self, evaluated: &[(ParameterSet, f64)]) -> Result<ParameterSet> {
        // Generate candidate points
        let n_candidates = 100;
        let mut candidates = Vec::with_capacity(n_candidates);

        for _ in 0..n_candidates {
            candidates.push(self.sample_random_params()?);
        }

        // Calculate Expected Improvement for each candidate
        let best_observed = evaluated
            .iter()
            .map(|(_, score)| *score)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let mut best_ei = -f64::INFINITY;
        let mut best_candidate = candidates[0].clone();

        for candidate in &candidates {
            let ei = self.expected_improvement(candidate, evaluated, best_observed)?;
            if ei > best_ei {
                best_ei = ei;
                best_candidate = candidate.clone();
            }
        }

        Ok(best_candidate)
    }

    /// Calculate Expected Improvement acquisition function
    fn expected_improvement(
        &self,
        candidate: &ParameterSet,
        evaluated: &[(ParameterSet, f64)],
        best_observed: f64,
    ) -> Result<f64> {
        // Simplified Gaussian Process prediction using RBF kernel
        // Mean prediction: weighted average of observed values
        // Std prediction: based on distance to nearest neighbors

        let (mean, std) = self.gp_predict(candidate, evaluated)?;

        if std < 1e-10 {
            return Ok(0.0);
        }

        // Expected Improvement formula
        let z = (mean - best_observed - 0.01) / std; // 0.01 is exploration parameter (xi)
        let ei = (mean - best_observed - 0.01) * self.normal_cdf(z) + std * self.normal_pdf(z);

        Ok(ei.max(0.0))
    }

    /// Simplified Gaussian Process prediction
    fn gp_predict(
        &self,
        candidate: &ParameterSet,
        evaluated: &[(ParameterSet, f64)],
    ) -> Result<(f64, f64)> {
        if evaluated.is_empty() {
            return Ok((0.0, 1.0));
        }

        // Calculate distances and weights using RBF kernel
        let length_scale = 1.0;
        let mut total_weight = 0.0;
        let mut weighted_mean = 0.0;

        for (params, score) in evaluated {
            let dist = self.parameter_distance(candidate, params)?;
            let weight = (-0.5 * (dist / length_scale).powi(2)).exp();
            total_weight += weight;
            weighted_mean += weight * score;
        }

        let mean = if total_weight > 1e-10 {
            weighted_mean / total_weight
        } else {
            evaluated.iter().map(|(_, score)| score).sum::<f64>() / evaluated.len() as f64
        };

        // Estimate uncertainty based on distance to nearest neighbor
        let min_dist = evaluated
            .iter()
            .map(|(params, _)| self.parameter_distance(candidate, params).unwrap_or(1.0))
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(1.0);

        let std = (0.1 + 0.9 * (1.0 - (-min_dist.powi(2)).exp())).min(1.0);

        Ok((mean, std))
    }

    /// Calculate distance between two parameter sets
    fn parameter_distance(&self, a: &ParameterSet, b: &ParameterSet) -> Result<f64> {
        // Normalized Euclidean distance in parameter space
        let c_dist = ((a.c.ln() - b.c.ln()) / 3.0).powi(2); // Normalized log distance
        let tol_dist = ((a.tol.ln() - b.tol.ln()) / 3.0).powi(2);
        let max_iter_dist = ((a.max_iter as f64 - b.max_iter as f64) / 2500.0).powi(2);

        // Kernel distance (0 if same, 1 if different)
        let kernel_dist = if std::mem::discriminant(&a.kernel) == std::mem::discriminant(&b.kernel)
        {
            0.0
        } else {
            1.0
        };

        Ok((c_dist + tol_dist + max_iter_dist + kernel_dist).sqrt())
    }

    /// Standard normal CDF (cumulative distribution function)
    fn normal_cdf(&self, x: f64) -> f64 {
        0.5 * (1.0 + self.erf(x / std::f64::consts::SQRT_2))
    }

    /// Standard normal PDF (probability density function)
    fn normal_pdf(&self, x: f64) -> f64 {
        (1.0 / (2.0 * std::f64::consts::PI).sqrt()) * (-0.5 * x.powi(2)).exp()
    }

    /// Error function approximation
    fn erf(&self, x: f64) -> f64 {
        // Abramowitz and Stegun approximation
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

    /// Sample a single value from parameter specification
    fn sample_value(&mut self, spec: &ParameterSpec) -> Result<f64> {
        match spec {
            ParameterSpec::Fixed(value) => Ok(*value),
            ParameterSpec::Uniform { min, max } => {
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(*min, *max).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                Ok(self.rng.sample(dist))
            }
            ParameterSpec::LogUniform { min, max } => {
                use scirs2_core::random::essentials::Uniform;
                let log_min = min.ln();
                let log_max = max.ln();
                let dist = Uniform::new(log_min, log_max).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let log_val = self.rng.sample(dist);
                Ok(log_val.exp())
            }
            ParameterSpec::Choice(choices) => {
                if choices.is_empty() {
                    return Err(SklearsError::InvalidInput("Empty choice list".to_string()));
                }
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(0, choices.len()).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let idx = self.rng.sample(dist);
                Ok(choices[idx])
            }
            ParameterSpec::KernelChoice(_) => Err(SklearsError::InvalidInput(
                "Use sample_kernel for kernel specs".to_string(),
            )),
        }
    }

    /// Sample a kernel from kernel specification
    fn sample_kernel(&mut self, spec: &ParameterSpec) -> Result<KernelType> {
        match spec {
            ParameterSpec::KernelChoice(kernels) => {
                if kernels.is_empty() {
                    return Err(SklearsError::InvalidInput(
                        "Empty kernel choice list".to_string(),
                    ));
                }
                use scirs2_core::random::essentials::Uniform;
                let dist = Uniform::new(0, kernels.len()).map_err(|e| {
                    SklearsError::InvalidInput(format!(
                        "Failed to create uniform distribution: {}",
                        e
                    ))
                })?;
                let idx = self.rng.sample(dist);
                Ok(kernels[idx].clone())
            }
            _ => Err(SklearsError::InvalidInput(
                "Invalid kernel specification".to_string(),
            )),
        }
    }

    /// Evaluate parameter set using cross-validation
    fn evaluate_params(
        &self,
        params: &ParameterSet,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<f64> {
        let scores = self.cross_validate(params, x, y)?;
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Perform cross-validation
    fn cross_validate(
        &self,
        params: &ParameterSet,
        x: &DMatrix<f64>,
        y: &DVector<f64>,
    ) -> Result<Vec<f64>> {
        let n_samples = x.nrows();
        let fold_size = n_samples / self.config.cv_folds;
        let mut scores = Vec::new();

        for fold in 0..self.config.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.config.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/test splits
            let mut x_train_data = Vec::new();
            let mut y_train_vals = Vec::new();
            let mut x_test_data = Vec::new();
            let mut y_test_vals = Vec::new();

            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    // Test set
                    for j in 0..x.ncols() {
                        x_test_data.push(x[[i, j]]);
                    }
                    y_test_vals.push(y[i]);
                } else {
                    // Training set
                    for j in 0..x.ncols() {
                        x_train_data.push(x[[i, j]]);
                    }
                    y_train_vals.push(y[i]);
                }
            }

            let n_train = y_train_vals.len();
            let n_test = y_test_vals.len();
            let n_features = x.ncols();

            let x_train = Array2::from_shape_vec((n_train, n_features), x_train_data)?;
            let y_train = Array1::from_vec(y_train_vals);
            let x_test = Array2::from_shape_vec((n_test, n_features), x_test_data)?;
            let y_test = Array1::from_vec(y_test_vals);

            // Train and evaluate model
            let svm = SVC::new()
                .c(params.c)
                .kernel(params.kernel.clone())
                .tol(params.tol)
                .max_iter(params.max_iter);

            let fitted_svm = svm.fit(&x_train, &y_train)?;
            let y_pred = fitted_svm.predict(&x_test)?;

            let score = self.calculate_score(&y_test, &y_pred)?;
            scores.push(score);
        }

        Ok(scores)
    }

    /// Calculate score based on scoring metric
    fn calculate_score(&self, y_true: &DVector<f64>, y_pred: &DVector<f64>) -> Result<f64> {
        match self.config.scoring {
            ScoringMetric::Accuracy => {
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| if (t - p).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                Ok(correct / y_true.len() as f64)
            }
            ScoringMetric::MeanSquaredError => {
                let mse = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| (t - p).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mse) // Negative because we want to maximize
            }
            ScoringMetric::MeanAbsoluteError => {
                let mae = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| (t - p).abs())
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mae) // Negative because we want to maximize
            }
            _ => {
                // For now, default to accuracy for other metrics
                let correct = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&t, &p)| if (t - p).abs() < 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>();
                Ok(correct / y_true.len() as f64)
            }
        }
    }
}

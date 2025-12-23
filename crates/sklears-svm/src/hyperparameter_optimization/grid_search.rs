//! Grid Search Cross-Validation for hyperparameter optimization

use std::time::Instant;

#[cfg(feature = "parallel")]
use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::Random;

// Type aliases for compatibility
type DMatrix<T> = Array2<T>;
type DVector<T> = Array1<T>;

use crate::kernels::KernelType;
use crate::svc::SVC;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict};

use super::{
    OptimizationConfig, OptimizationResult, ParameterSet, ParameterSpec, ScoringMetric, SearchSpace,
};

/// Grid Search hyperparameter optimizer
pub struct GridSearchCV {
    config: OptimizationConfig,
    search_space: SearchSpace,
    rng: Random<scirs2_core::random::rngs::StdRng>,
}

impl GridSearchCV {
    /// Create a new grid search optimizer
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

    /// Run grid search optimization
    pub fn fit(&mut self, x: &DMatrix<f64>, y: &DVector<f64>) -> Result<OptimizationResult> {
        let start_time = Instant::now();

        // Generate parameter grid
        let param_grid = self.generate_parameter_grid()?;

        if self.config.verbose {
            println!(
                "Grid search with {} parameter combinations",
                param_grid.len()
            );
        }

        // Evaluate all parameter combinations
        let cv_results: Vec<(ParameterSet, f64)> = {
            #[cfg(feature = "parallel")]
            if self.config.n_jobs.is_some() {
                // Parallel evaluation
                param_grid
                    .into_par_iter()
                    .map(|params| {
                        let score = self
                            .evaluate_params(&params, x, y)
                            .unwrap_or(-f64::INFINITY);
                        (params, score)
                    })
                    .collect()
            } else {
                // Sequential evaluation
                param_grid
                    .into_iter()
                    .map(|params| {
                        let score = self
                            .evaluate_params(&params, x, y)
                            .unwrap_or(-f64::INFINITY);
                        if self.config.verbose {
                            println!("Params: {:?}, Score: {:.6}", params, score);
                        }
                        (params, score)
                    })
                    .collect()
            }

            #[cfg(not(feature = "parallel"))]
            {
                // Sequential evaluation (parallel feature disabled)
                param_grid
                    .into_iter()
                    .map(|params| {
                        let score = self
                            .evaluate_params(&params, x, y)
                            .unwrap_or(-f64::INFINITY);
                        if self.config.verbose {
                            println!("Params: {:?}, Score: {:.6}", params, score);
                        }
                        (params, score)
                    })
                    .collect()
            }
        };

        // Find best parameters
        let (best_params, best_score) = cv_results
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(p, s)| (p.clone(), *s))
            .ok_or_else(|| {
                SklearsError::Other("No valid parameter combinations found".to_string())
            })?;

        let score_history: Vec<f64> = cv_results.iter().map(|(_, score)| *score).collect();
        let n_iterations = cv_results.len();

        Ok(OptimizationResult {
            best_params,
            best_score,
            cv_results,
            n_iterations,
            optimization_time: start_time.elapsed().as_secs_f64(),
            score_history,
        })
    }

    /// Generate parameter grid for grid search
    fn generate_parameter_grid(&mut self) -> Result<Vec<ParameterSet>> {
        let mut param_grid = Vec::new();

        // Clone specs to avoid borrowing conflicts
        let c_spec = self.search_space.c.clone();
        let kernel_spec = self.search_space.kernel.clone();
        let tol_spec = self.search_space.tol.clone();
        let max_iter_spec = self.search_space.max_iter.clone();

        // Generate C values
        let c_values = self.generate_values(&c_spec, 10)?;

        // Generate kernel values
        let kernel_values = if let Some(kernel_spec) = kernel_spec {
            self.generate_kernel_values(&kernel_spec)?
        } else {
            vec![KernelType::Rbf { gamma: 1.0 }]
        };

        // Generate tolerance values
        let tol_values = if let Some(tol_spec) = tol_spec {
            self.generate_values(&tol_spec, 5)?
        } else {
            vec![1e-3]
        };

        // Generate max_iter values
        let max_iter_values = if let Some(max_iter_spec) = max_iter_spec {
            self.generate_values(&max_iter_spec, 3)?
                .into_iter()
                .map(|v| v as usize)
                .collect()
        } else {
            vec![1000]
        };

        // Generate all combinations
        for &c in &c_values {
            for kernel in &kernel_values {
                for &tol in &tol_values {
                    for &max_iter in &max_iter_values {
                        param_grid.push(ParameterSet {
                            c,
                            kernel: kernel.clone(),
                            tol,
                            max_iter,
                        });
                    }
                }
            }
        }

        Ok(param_grid)
    }

    /// Generate values from parameter specification
    fn generate_values(&mut self, spec: &ParameterSpec, n_values: usize) -> Result<Vec<f64>> {
        match spec {
            ParameterSpec::Fixed(value) => Ok(vec![*value]),
            ParameterSpec::Uniform { min, max } => Ok((0..n_values)
                .map(|i| min + (max - min) * i as f64 / (n_values - 1) as f64)
                .collect()),
            ParameterSpec::LogUniform { min, max } => {
                let log_min = min.ln();
                let log_max = max.ln();
                Ok((0..n_values)
                    .map(|i| {
                        let log_val =
                            log_min + (log_max - log_min) * i as f64 / (n_values - 1) as f64;
                        log_val.exp()
                    })
                    .collect())
            }
            ParameterSpec::Choice(choices) => Ok(choices.clone()),
            ParameterSpec::KernelChoice(_) => Err(SklearsError::InvalidInput(
                "Use generate_kernel_values for kernel specs".to_string(),
            )),
        }
    }

    /// Generate kernel values from kernel specification
    fn generate_kernel_values(&mut self, spec: &ParameterSpec) -> Result<Vec<KernelType>> {
        match spec {
            ParameterSpec::KernelChoice(kernels) => Ok(kernels.clone()),
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

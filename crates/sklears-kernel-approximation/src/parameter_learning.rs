//! Automatic kernel parameter learning and optimization
//!
//! This module provides automated methods for learning optimal kernel parameters
//! using various optimization strategies like grid search and Bayesian optimization.

use crate::{Nystroem, RBFSampler};
use rayon::prelude::*;
use scirs2_core::ndarray::ndarray_linalg::SVD;
use scirs2_core::ndarray::{s, Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
};

/// Parameter search strategies
#[derive(Debug, Clone)]
/// SearchStrategy
pub enum SearchStrategy {
    /// Grid search over parameter space
    GridSearch {
        /// Number of points per parameter
        n_points: usize,
    },
    /// Random search
    RandomSearch {
        /// Number of random samples
        n_samples: usize,
    },
    /// Bayesian optimization (simplified)
    BayesianOptimization {
        /// Number of initial random samples
        n_initial: usize,
        /// Number of optimization iterations
        n_iterations: usize,
        /// Acquisition function parameter
        exploration_factor: f64,
    },
    /// Coordinate descent
    CoordinateDescent {
        /// Maximum iterations
        max_iterations: usize,
        /// Convergence tolerance
        tolerance: f64,
    },
}

/// Parameter bounds for optimization
#[derive(Debug, Clone)]
/// ParameterBounds
pub struct ParameterBounds {
    /// Gamma parameter bounds
    pub gamma_bounds: (f64, f64),
    /// Number of components bounds
    pub n_components_bounds: (usize, usize),
    /// Degree bounds (for polynomial kernels)
    pub degree_bounds: Option<(i32, i32)>,
    /// Coef0 bounds (for polynomial kernels)
    pub coef0_bounds: Option<(f64, f64)>,
}

impl Default for ParameterBounds {
    fn default() -> Self {
        Self {
            gamma_bounds: (1e-6, 1e2),
            n_components_bounds: (10, 1000),
            degree_bounds: Some((2, 5)),
            coef0_bounds: Some((-1.0, 1.0)),
        }
    }
}

/// Objective functions for parameter optimization
#[derive(Debug, Clone)]
/// ObjectiveFunction
pub enum ObjectiveFunction {
    /// Kernel alignment score
    KernelAlignment,
    /// Cross-validation error
    CrossValidationError { n_folds: usize },
    /// Approximation quality (Frobenius norm)
    ApproximationQuality,
    /// Effective rank
    EffectiveRank,
    /// Custom objective function
    Custom,
}

/// Configuration for parameter learning
#[derive(Debug, Clone)]
/// ParameterLearningConfig
pub struct ParameterLearningConfig {
    /// Search strategy
    pub search_strategy: SearchStrategy,
    /// Parameter bounds
    pub parameter_bounds: ParameterBounds,
    /// Objective function
    pub objective_function: ObjectiveFunction,
    /// Validation fraction for evaluation
    pub validation_fraction: f64,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Parallel processing
    pub n_jobs: usize,
    /// Verbose output
    pub verbose: bool,
}

impl Default for ParameterLearningConfig {
    fn default() -> Self {
        Self {
            search_strategy: SearchStrategy::GridSearch { n_points: 10 },
            parameter_bounds: ParameterBounds::default(),
            objective_function: ObjectiveFunction::KernelAlignment,
            validation_fraction: 0.2,
            random_seed: None,
            n_jobs: num_cpus::get(),
            verbose: false,
        }
    }
}

/// Parameter set for optimization
#[derive(Debug, Clone)]
/// ParameterSet
pub struct ParameterSet {
    /// Gamma parameter
    pub gamma: f64,
    /// Number of components
    pub n_components: usize,
    /// Degree (for polynomial kernels)
    pub degree: Option<i32>,
    /// Coef0 (for polynomial kernels)
    pub coef0: Option<f64>,
}

impl PartialEq for ParameterSet {
    fn eq(&self, other: &Self) -> bool {
        (self.gamma - other.gamma).abs() < f64::EPSILON
            && self.n_components == other.n_components
            && self.degree == other.degree
            && match (self.coef0, other.coef0) {
                (Some(a), Some(b)) => (a - b).abs() < f64::EPSILON,
                (None, None) => true,
                _ => false,
            }
    }
}

impl Eq for ParameterSet {}

impl std::hash::Hash for ParameterSet {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Convert f64 to bits for hashing
        self.gamma.to_bits().hash(state);
        self.n_components.hash(state);
        self.degree.hash(state);
        if let Some(coef0) = self.coef0 {
            coef0.to_bits().hash(state);
        }
    }
}

/// Results from parameter optimization
#[derive(Debug, Clone)]
/// OptimizationResult
pub struct OptimizationResult {
    /// Best parameter set found
    pub best_parameters: ParameterSet,
    /// Best objective value
    pub best_score: f64,
    /// All evaluated parameter sets and scores
    pub parameter_history: Vec<(ParameterSet, f64)>,
    /// Convergence information
    pub converged: bool,
    /// Number of function evaluations
    pub n_evaluations: usize,
}

/// Automatic parameter learning for RBF kernels
pub struct ParameterLearner {
    config: ParameterLearningConfig,
}

impl ParameterLearner {
    /// Create a new parameter learner
    pub fn new(config: ParameterLearningConfig) -> Self {
        Self { config }
    }

    /// Optimize RBF sampler parameters
    pub fn optimize_rbf_parameters(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<OptimizationResult> {
        match &self.config.search_strategy {
            SearchStrategy::GridSearch { n_points } => self.grid_search_rbf(x, y, *n_points),
            SearchStrategy::RandomSearch { n_samples } => self.random_search_rbf(x, y, *n_samples),
            SearchStrategy::BayesianOptimization {
                n_initial,
                n_iterations,
                exploration_factor,
            } => {
                self.bayesian_optimization_rbf(x, y, *n_initial, *n_iterations, *exploration_factor)
            }
            SearchStrategy::CoordinateDescent {
                max_iterations,
                tolerance,
            } => self.coordinate_descent_rbf(x, y, *max_iterations, *tolerance),
        }
    }

    /// Optimize Nyström parameters
    pub fn optimize_nystroem_parameters(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
    ) -> Result<OptimizationResult> {
        // Similar to RBF optimization but for Nyström method
        match &self.config.search_strategy {
            SearchStrategy::GridSearch { n_points } => self.grid_search_nystroem(x, y, *n_points),
            SearchStrategy::RandomSearch { n_samples } => {
                self.random_search_nystroem(x, y, *n_samples)
            }
            _ => {
                // Fallback to grid search for unsupported strategies
                self.grid_search_nystroem(x, y, 10)
            }
        }
    }

    fn grid_search_rbf(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        n_points: usize,
    ) -> Result<OptimizationResult> {
        let gamma_values = self.create_parameter_grid(
            self.config.parameter_bounds.gamma_bounds.0,
            self.config.parameter_bounds.gamma_bounds.1,
            n_points,
            true, // log scale
        );

        let n_components_values = self
            .create_parameter_grid(
                self.config.parameter_bounds.n_components_bounds.0 as f64,
                self.config.parameter_bounds.n_components_bounds.1 as f64,
                n_points,
                false, // linear scale
            )
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>();

        let mut parameter_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = ParameterSet {
            gamma: gamma_values[0],
            n_components: n_components_values[0],
            degree: None,
            coef0: None,
        };

        // Generate all parameter combinations
        let parameter_combinations: Vec<_> = gamma_values
            .iter()
            .flat_map(|&gamma| {
                n_components_values
                    .iter()
                    .map(move |&n_components| ParameterSet {
                        gamma,
                        n_components,
                        degree: None,
                        coef0: None,
                    })
            })
            .collect();

        if self.config.verbose {
            println!(
                "Evaluating {} parameter combinations",
                parameter_combinations.len()
            );
        }

        // Parallel evaluation of parameter combinations
        let scores: Result<Vec<_>> = parameter_combinations
            .par_iter()
            .map(|params| {
                let score = self.evaluate_rbf_parameters(x, y, params)?;
                Ok((params.clone(), score))
            })
            .collect();

        let scores = scores?;

        for (params, score) in scores {
            parameter_history.push((params.clone(), score));

            if score > best_score {
                best_score = score;
                best_parameters = params;
            }
        }

        let n_evaluations = parameter_history.len();
        Ok(OptimizationResult {
            best_parameters,
            best_score,
            parameter_history,
            converged: true,
            n_evaluations,
        })
    }

    fn random_search_rbf(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        n_samples: usize,
    ) -> Result<OptimizationResult> {
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_seed(thread_rng().gen())
        };

        let mut parameter_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = ParameterSet {
            gamma: 1.0,
            n_components: 100,
            degree: None,
            coef0: None,
        };

        for _ in 0..n_samples {
            // Sample random parameters
            let gamma = self.sample_log_uniform(
                &mut rng,
                self.config.parameter_bounds.gamma_bounds.0,
                self.config.parameter_bounds.gamma_bounds.1,
            );

            let n_components = rng.gen_range(
                self.config.parameter_bounds.n_components_bounds.0
                    ..=self.config.parameter_bounds.n_components_bounds.1,
            );

            let params = ParameterSet {
                gamma,
                n_components,
                degree: None,
                coef0: None,
            };

            let score = self.evaluate_rbf_parameters(x, y, &params)?;
            parameter_history.push((params.clone(), score));

            if score > best_score {
                best_score = score;
                best_parameters = params;
            }
        }

        Ok(OptimizationResult {
            best_parameters,
            best_score,
            parameter_history,
            converged: true,
            n_evaluations: n_samples,
        })
    }

    fn bayesian_optimization_rbf(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        n_initial: usize,
        n_iterations: usize,
        exploration_factor: f64,
    ) -> Result<OptimizationResult> {
        // Simplified Bayesian optimization using Gaussian process surrogate
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_seed(thread_rng().gen())
        };

        let mut parameter_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = ParameterSet {
            gamma: 1.0,
            n_components: 100,
            degree: None,
            coef0: None,
        };

        // Initial random sampling
        for _ in 0..n_initial {
            let gamma = self.sample_log_uniform(
                &mut rng,
                self.config.parameter_bounds.gamma_bounds.0,
                self.config.parameter_bounds.gamma_bounds.1,
            );

            let n_components = rng.gen_range(
                self.config.parameter_bounds.n_components_bounds.0
                    ..=self.config.parameter_bounds.n_components_bounds.1,
            );

            let params = ParameterSet {
                gamma,
                n_components,
                degree: None,
                coef0: None,
            };

            let score = self.evaluate_rbf_parameters(x, y, &params)?;
            parameter_history.push((params.clone(), score));

            if score > best_score {
                best_score = score;
                best_parameters = params;
            }
        }

        // Bayesian optimization iterations
        for iteration in 0..n_iterations {
            // Simplified acquisition function (Upper Confidence Bound)
            let next_params =
                self.acquisition_function_rbf(&parameter_history, exploration_factor, &mut rng);

            let score = self.evaluate_rbf_parameters(x, y, &next_params)?;
            parameter_history.push((next_params.clone(), score));

            if score > best_score {
                best_score = score;
                best_parameters = next_params;
            }

            if self.config.verbose {
                println!(
                    "Iteration {}: Best score = {:.6}",
                    iteration + 1,
                    best_score
                );
            }
        }

        Ok(OptimizationResult {
            best_parameters,
            best_score,
            parameter_history,
            converged: true,
            n_evaluations: n_initial + n_iterations,
        })
    }

    fn coordinate_descent_rbf(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        max_iterations: usize,
        tolerance: f64,
    ) -> Result<OptimizationResult> {
        let mut current_params = ParameterSet {
            gamma: (self.config.parameter_bounds.gamma_bounds.0
                * self.config.parameter_bounds.gamma_bounds.1)
                .sqrt(),
            n_components: (self.config.parameter_bounds.n_components_bounds.0
                + self.config.parameter_bounds.n_components_bounds.1)
                / 2,
            degree: None,
            coef0: None,
        };

        let mut current_score = self.evaluate_rbf_parameters(x, y, &current_params)?;
        let mut parameter_history = vec![(current_params.clone(), current_score)];
        let mut converged = false;

        for iteration in 0..max_iterations {
            let prev_score = current_score;

            // Optimize gamma
            current_params = self.optimize_gamma_coordinate(x, y, &current_params)?;
            current_score = self.evaluate_rbf_parameters(x, y, &current_params)?;
            parameter_history.push((current_params.clone(), current_score));

            // Optimize n_components
            current_params = self.optimize_n_components_coordinate(x, y, &current_params)?;
            current_score = self.evaluate_rbf_parameters(x, y, &current_params)?;
            parameter_history.push((current_params.clone(), current_score));

            // Check convergence
            if (current_score - prev_score).abs() < tolerance {
                converged = true;
                if self.config.verbose {
                    println!(
                        "Converged at iteration {} with score {:.6}",
                        iteration + 1,
                        current_score
                    );
                }
                break;
            }

            if self.config.verbose {
                println!("Iteration {}: Score = {:.6}", iteration + 1, current_score);
            }
        }

        let n_evaluations = parameter_history.len();
        Ok(OptimizationResult {
            best_parameters: current_params,
            best_score: current_score,
            parameter_history,
            converged,
            n_evaluations,
        })
    }

    fn grid_search_nystroem(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        n_points: usize,
    ) -> Result<OptimizationResult> {
        // Similar to RBF grid search but for Nyström method
        let gamma_values = self.create_parameter_grid(
            self.config.parameter_bounds.gamma_bounds.0,
            self.config.parameter_bounds.gamma_bounds.1,
            n_points,
            true,
        );

        let n_components_values = self
            .create_parameter_grid(
                self.config.parameter_bounds.n_components_bounds.0 as f64,
                self.config.parameter_bounds.n_components_bounds.1 as f64,
                n_points,
                false,
            )
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>();

        let mut parameter_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = ParameterSet {
            gamma: gamma_values[0],
            n_components: n_components_values[0],
            degree: None,
            coef0: None,
        };

        for &gamma in &gamma_values {
            for &n_components in &n_components_values {
                let params = ParameterSet {
                    gamma,
                    n_components,
                    degree: None,
                    coef0: None,
                };

                let score = self.evaluate_nystroem_parameters(x, y, &params)?;
                parameter_history.push((params.clone(), score));

                if score > best_score {
                    best_score = score;
                    best_parameters = params;
                }
            }
        }

        let n_evaluations = parameter_history.len();
        Ok(OptimizationResult {
            best_parameters,
            best_score,
            parameter_history,
            converged: true,
            n_evaluations,
        })
    }

    fn random_search_nystroem(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        n_samples: usize,
    ) -> Result<OptimizationResult> {
        let mut rng = if let Some(seed) = self.config.random_seed {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::from_seed(thread_rng().gen())
        };

        let mut parameter_history = Vec::new();
        let mut best_score = f64::NEG_INFINITY;
        let mut best_parameters = ParameterSet {
            gamma: 1.0,
            n_components: 100,
            degree: None,
            coef0: None,
        };

        for _ in 0..n_samples {
            let gamma = self.sample_log_uniform(
                &mut rng,
                self.config.parameter_bounds.gamma_bounds.0,
                self.config.parameter_bounds.gamma_bounds.1,
            );

            let n_components = rng.gen_range(
                self.config.parameter_bounds.n_components_bounds.0
                    ..=self.config.parameter_bounds.n_components_bounds.1,
            );

            let params = ParameterSet {
                gamma,
                n_components,
                degree: None,
                coef0: None,
            };

            let score = self.evaluate_nystroem_parameters(x, y, &params)?;
            parameter_history.push((params.clone(), score));

            if score > best_score {
                best_score = score;
                best_parameters = params;
            }
        }

        Ok(OptimizationResult {
            best_parameters,
            best_score,
            parameter_history,
            converged: true,
            n_evaluations: n_samples,
        })
    }

    fn evaluate_rbf_parameters(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        params: &ParameterSet,
    ) -> Result<f64> {
        let sampler = RBFSampler::new(params.n_components).gamma(params.gamma);
        let fitted = sampler.fit(x, &())?;
        let x_transformed = fitted.transform(x)?;

        match &self.config.objective_function {
            ObjectiveFunction::KernelAlignment => {
                self.compute_kernel_alignment(x, &x_transformed, params.gamma)
            }
            ObjectiveFunction::CrossValidationError { n_folds } => {
                if let Some(y_data) = y {
                    self.compute_cross_validation_score(&x_transformed, y_data, *n_folds)
                } else {
                    // Fallback to kernel alignment
                    self.compute_kernel_alignment(x, &x_transformed, params.gamma)
                }
            }
            ObjectiveFunction::ApproximationQuality => {
                self.compute_approximation_quality(x, &x_transformed, params.gamma)
            }
            ObjectiveFunction::EffectiveRank => self.compute_effective_rank(&x_transformed),
            ObjectiveFunction::Custom => {
                // Placeholder for custom objective function
                Ok(0.0)
            }
        }
    }

    fn evaluate_nystroem_parameters(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        params: &ParameterSet,
    ) -> Result<f64> {
        use crate::nystroem::Kernel;

        let kernel = Kernel::Rbf {
            gamma: params.gamma,
        };
        let nystroem = Nystroem::new(kernel, params.n_components);
        let fitted = nystroem.fit(x, &())?;
        let x_transformed = fitted.transform(x)?;

        match &self.config.objective_function {
            ObjectiveFunction::KernelAlignment => {
                self.compute_kernel_alignment(x, &x_transformed, params.gamma)
            }
            ObjectiveFunction::CrossValidationError { n_folds } => {
                if let Some(y_data) = y {
                    self.compute_cross_validation_score(&x_transformed, y_data, *n_folds)
                } else {
                    self.compute_kernel_alignment(x, &x_transformed, params.gamma)
                }
            }
            ObjectiveFunction::ApproximationQuality => {
                self.compute_approximation_quality(x, &x_transformed, params.gamma)
            }
            ObjectiveFunction::EffectiveRank => self.compute_effective_rank(&x_transformed),
            ObjectiveFunction::Custom => Ok(0.0),
        }
    }

    fn compute_kernel_alignment(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
        gamma: f64,
    ) -> Result<f64> {
        let n_samples = x.nrows().min(100); // Limit for efficiency
        let x_subset = x.slice(s![..n_samples, ..]);

        // Compute exact kernel matrix
        let mut k_exact = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-gamma * squared_norm).exp();
            }
        }

        // Compute approximate kernel matrix
        let x_transformed_subset = x_transformed.slice(s![..n_samples, ..]);
        let k_approx = x_transformed_subset.dot(&x_transformed_subset.t());

        // Compute alignment
        let k_exact_frobenius = k_exact.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let k_approx_frobenius = k_approx.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let k_product = (&k_exact * &k_approx).sum();

        let alignment = k_product / (k_exact_frobenius * k_approx_frobenius);
        Ok(alignment)
    }

    fn compute_cross_validation_score(
        &self,
        x_transformed: &Array2<f64>,
        y: &Array1<f64>,
        n_folds: usize,
    ) -> Result<f64> {
        let n_samples = x_transformed.nrows();
        let fold_size = n_samples / n_folds;
        let mut cv_scores = Vec::new();

        for fold in 0..n_folds {
            let start = fold * fold_size;
            let end = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Simple correlation-based score (placeholder for more sophisticated CV)
            let val_features = x_transformed.slice(s![start..end, ..]);
            let val_targets = y.slice(s![start..end]);

            // Compute mean correlation between features and targets
            let mut correlations = Vec::new();
            for j in 0..val_features.ncols() {
                let feature_col = val_features.column(j);
                let correlation =
                    self.compute_correlation(feature_col.into_owned().view(), val_targets);
                correlations.push(correlation.abs());
            }

            let mean_correlation = correlations.iter().sum::<f64>() / correlations.len() as f64;
            cv_scores.push(mean_correlation);
        }

        Ok(cv_scores.iter().sum::<f64>() / cv_scores.len() as f64)
    }

    fn compute_approximation_quality(
        &self,
        x: &Array2<f64>,
        x_transformed: &Array2<f64>,
        gamma: f64,
    ) -> Result<f64> {
        // Compute reconstruction error
        let n_samples = x.nrows().min(50); // Limit for efficiency
        let x_subset = x.slice(s![..n_samples, ..]);
        let x_transformed_subset = x_transformed.slice(s![..n_samples, ..]);

        // Exact kernel matrix
        let mut k_exact = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in 0..n_samples {
                let diff = &x_subset.row(i) - &x_subset.row(j);
                let squared_norm = diff.dot(&diff);
                k_exact[[i, j]] = (-gamma * squared_norm).exp();
            }
        }

        // Approximate kernel matrix
        let k_approx = x_transformed_subset.dot(&x_transformed_subset.t());

        // Frobenius norm error
        let error_matrix = &k_exact - &k_approx;
        let frobenius_error = error_matrix.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let exact_frobenius = k_exact.iter().map(|&x| x * x).sum::<f64>().sqrt();

        let relative_error = frobenius_error / exact_frobenius;
        Ok(1.0 / (1.0 + relative_error)) // Convert error to quality score
    }

    fn compute_effective_rank(&self, x_transformed: &Array2<f64>) -> Result<f64> {
        // Compute SVD
        let (_, s, _) = x_transformed
            .svd(true, true)
            .map_err(|_| SklearsError::InvalidInput("SVD computation failed".to_string()))?;

        // Compute effective rank using entropy
        let s_sum = s.sum();
        if s_sum == 0.0 {
            return Ok(0.0);
        }

        let s_normalized = &s / s_sum;
        let entropy = -s_normalized
            .iter()
            .filter(|&&x| x > 1e-12)
            .map(|&x| x * x.ln())
            .sum::<f64>();

        Ok(entropy.exp())
    }

    fn compute_correlation(
        &self,
        x: scirs2_core::ndarray::ArrayView1<f64>,
        y: scirs2_core::ndarray::ArrayView1<f64>,
    ) -> f64 {
        let x_mean = x.mean().unwrap_or(0.0);
        let y_mean = y.mean().unwrap_or(0.0);

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum();

        let x_var: f64 = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum();
        let y_var: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();

        let denominator = (x_var * y_var).sqrt();

        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }

    // Helper methods
    fn create_parameter_grid(
        &self,
        min_val: f64,
        max_val: f64,
        n_points: usize,
        log_scale: bool,
    ) -> Vec<f64> {
        if log_scale {
            let log_min = min_val.ln();
            let log_max = max_val.ln();
            (0..n_points)
                .map(|i| {
                    let t = i as f64 / (n_points - 1) as f64;
                    (log_min + t * (log_max - log_min)).exp()
                })
                .collect()
        } else {
            (0..n_points)
                .map(|i| {
                    let t = i as f64 / (n_points - 1) as f64;
                    min_val + t * (max_val - min_val)
                })
                .collect()
        }
    }

    fn sample_log_uniform(&self, rng: &mut StdRng, min_val: f64, max_val: f64) -> f64 {
        let log_min = min_val.ln();
        let log_max = max_val.ln();
        let log_val = rng.gen_range(log_min..=log_max);
        log_val.exp()
    }

    fn acquisition_function_rbf(
        &self,
        parameter_history: &[(ParameterSet, f64)],
        exploration_factor: f64,
        rng: &mut StdRng,
    ) -> ParameterSet {
        // Simplified Upper Confidence Bound acquisition function
        // In practice, this would use a more sophisticated GP surrogate model

        let best_score = parameter_history
            .iter()
            .map(|(_, score)| *score)
            .fold(f64::NEG_INFINITY, f64::max);

        // Generate candidate points and evaluate acquisition function
        let mut best_acquisition = f64::NEG_INFINITY;
        let mut best_candidate = parameter_history[0].0.clone();

        for _ in 0..100 {
            // Sample 100 candidates
            let gamma = self.sample_log_uniform(
                rng,
                self.config.parameter_bounds.gamma_bounds.0,
                self.config.parameter_bounds.gamma_bounds.1,
            );

            let n_components = rng.gen_range(
                self.config.parameter_bounds.n_components_bounds.0
                    ..=self.config.parameter_bounds.n_components_bounds.1,
            );

            let candidate = ParameterSet {
                gamma,
                n_components,
                degree: None,
                coef0: None,
            };

            // Simple acquisition function: exploration bonus
            let mean_score = self.predict_score_from_history(parameter_history, &candidate);
            let uncertainty =
                exploration_factor * self.estimate_uncertainty(parameter_history, &candidate);
            let acquisition = mean_score + uncertainty;

            if acquisition > best_acquisition {
                best_acquisition = acquisition;
                best_candidate = candidate;
            }
        }

        best_candidate
    }

    fn predict_score_from_history(
        &self,
        parameter_history: &[(ParameterSet, f64)],
        candidate: &ParameterSet,
    ) -> f64 {
        // Simple distance-weighted average
        let mut weighted_sum = 0.0;
        let mut weight_sum = 0.0;

        for (params, score) in parameter_history {
            let distance = self.parameter_distance(candidate, params);
            let weight = (-distance).exp();
            weighted_sum += weight * score;
            weight_sum += weight;
        }

        if weight_sum > 0.0 {
            weighted_sum / weight_sum
        } else {
            0.0
        }
    }

    fn estimate_uncertainty(
        &self,
        parameter_history: &[(ParameterSet, f64)],
        candidate: &ParameterSet,
    ) -> f64 {
        // Simple uncertainty estimation based on distance to nearest neighbor
        let min_distance = parameter_history
            .iter()
            .map(|(params, _)| self.parameter_distance(candidate, params))
            .fold(f64::INFINITY, f64::min);

        min_distance
    }

    fn parameter_distance(&self, p1: &ParameterSet, p2: &ParameterSet) -> f64 {
        let gamma_diff = (p1.gamma.ln() - p2.gamma.ln()).powi(2);
        let n_components_diff =
            ((p1.n_components as f64).ln() - (p2.n_components as f64).ln()).powi(2);
        (gamma_diff + n_components_diff).sqrt()
    }

    fn optimize_gamma_coordinate(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        current_params: &ParameterSet,
    ) -> Result<ParameterSet> {
        let gamma_values = self.create_parameter_grid(
            self.config.parameter_bounds.gamma_bounds.0,
            self.config.parameter_bounds.gamma_bounds.1,
            10,
            true,
        );

        let mut best_gamma = current_params.gamma;
        let mut best_score = f64::NEG_INFINITY;

        for &gamma in &gamma_values {
            let test_params = ParameterSet {
                gamma,
                n_components: current_params.n_components,
                degree: current_params.degree,
                coef0: current_params.coef0,
            };

            let score = self.evaluate_rbf_parameters(x, y, &test_params)?;
            if score > best_score {
                best_score = score;
                best_gamma = gamma;
            }
        }

        Ok(ParameterSet {
            gamma: best_gamma,
            n_components: current_params.n_components,
            degree: current_params.degree,
            coef0: current_params.coef0,
        })
    }

    fn optimize_n_components_coordinate(
        &self,
        x: &Array2<f64>,
        y: Option<&Array1<f64>>,
        current_params: &ParameterSet,
    ) -> Result<ParameterSet> {
        let n_components_values = self
            .create_parameter_grid(
                self.config.parameter_bounds.n_components_bounds.0 as f64,
                self.config.parameter_bounds.n_components_bounds.1 as f64,
                10,
                false,
            )
            .into_iter()
            .map(|x| x as usize)
            .collect::<Vec<_>>();

        let mut best_n_components = current_params.n_components;
        let mut best_score = f64::NEG_INFINITY;

        for &n_components in &n_components_values {
            let test_params = ParameterSet {
                gamma: current_params.gamma,
                n_components,
                degree: current_params.degree,
                coef0: current_params.coef0,
            };

            let score = self.evaluate_rbf_parameters(x, y, &test_params)?;
            if score > best_score {
                best_score = score;
                best_n_components = n_components;
            }
        }

        Ok(ParameterSet {
            gamma: current_params.gamma,
            n_components: best_n_components,
            degree: current_params.degree,
            coef0: current_params.coef0,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_parameter_learner_grid_search() {
        let x =
            Array2::from_shape_vec((50, 5), (0..250).map(|i| i as f64 * 0.01).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::GridSearch { n_points: 3 },
            parameter_bounds: ParameterBounds {
                gamma_bounds: (0.1, 10.0),
                n_components_bounds: (10, 50),
                ..Default::default()
            },
            objective_function: ObjectiveFunction::KernelAlignment,
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert!(result.best_parameters.gamma >= 0.1);
        assert!(result.best_parameters.gamma <= 10.0);
        assert!(result.best_parameters.n_components >= 10);
        assert!(result.best_parameters.n_components <= 50);
        assert_eq!(result.parameter_history.len(), 9); // 3x3 grid
        assert!(result.converged);
    }

    #[test]
    fn test_parameter_learner_random_search() {
        let x =
            Array2::from_shape_vec((30, 4), (0..120).map(|i| i as f64 * 0.05).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::RandomSearch { n_samples: 5 },
            parameter_bounds: ParameterBounds {
                gamma_bounds: (0.01, 1.0),
                n_components_bounds: (5, 25),
                ..Default::default()
            },
            random_seed: Some(42),
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert_eq!(result.parameter_history.len(), 5);
        assert_eq!(result.n_evaluations, 5);
    }

    #[test]
    fn test_parameter_learner_bayesian_optimization() {
        let x = Array2::from_shape_vec((25, 3), (0..75).map(|i| i as f64 * 0.1).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::BayesianOptimization {
                n_initial: 3,
                n_iterations: 2,
                exploration_factor: 1.0,
            },
            parameter_bounds: ParameterBounds {
                gamma_bounds: (0.1, 5.0),
                n_components_bounds: (10, 30),
                ..Default::default()
            },
            random_seed: Some(123),
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert_eq!(result.parameter_history.len(), 5); // 3 initial + 2 iterations
        assert_eq!(result.n_evaluations, 5);
    }

    #[test]
    fn test_parameter_learner_coordinate_descent() {
        let x =
            Array2::from_shape_vec((40, 6), (0..240).map(|i| i as f64 * 0.02).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::CoordinateDescent {
                max_iterations: 3,
                tolerance: 1e-6,
            },
            parameter_bounds: ParameterBounds {
                gamma_bounds: (0.1, 2.0),
                n_components_bounds: (15, 35),
                ..Default::default()
            },
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert!(result.parameter_history.len() >= 1);
        // Note: May converge early, so we don't check exact length
    }

    #[test]
    fn test_parameter_learner_nystroem() {
        let x =
            Array2::from_shape_vec((35, 4), (0..140).map(|i| i as f64 * 0.03).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::GridSearch { n_points: 2 },
            parameter_bounds: ParameterBounds {
                gamma_bounds: (0.5, 2.0),
                n_components_bounds: (10, 20),
                ..Default::default()
            },
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_nystroem_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert_eq!(result.parameter_history.len(), 4); // 2x2 grid
    }

    #[test]
    fn test_cross_validation_objective() {
        let x =
            Array2::from_shape_vec((50, 3), (0..150).map(|i| i as f64 * 0.01).collect()).unwrap();
        let y = Array1::from_shape_fn(50, |i| (i as f64 * 0.1).sin());

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::GridSearch { n_points: 2 },
            objective_function: ObjectiveFunction::CrossValidationError { n_folds: 3 },
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, Some(&y)).unwrap();

        assert!(result.best_score >= 0.0);
        assert!(result.parameter_history.len() > 0);
    }

    #[test]
    fn test_effective_rank_objective() {
        let x =
            Array2::from_shape_vec((30, 5), (0..150).map(|i| i as f64 * 0.02).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::RandomSearch { n_samples: 3 },
            objective_function: ObjectiveFunction::EffectiveRank,
            random_seed: Some(456),
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        assert!(result.best_score > 0.0);
        assert_eq!(result.parameter_history.len(), 3);
    }

    #[test]
    fn test_parameter_grid_creation() {
        let learner = ParameterLearner::new(ParameterLearningConfig::default());

        // Test linear scale
        let linear_grid = learner.create_parameter_grid(0.0, 10.0, 5, false);
        assert_eq!(linear_grid.len(), 5);
        assert_abs_diff_eq!(linear_grid[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(linear_grid[4], 10.0, epsilon = 1e-10);

        // Test log scale
        let log_grid = learner.create_parameter_grid(0.1, 10.0, 3, true);
        assert_eq!(log_grid.len(), 3);
        assert_abs_diff_eq!(log_grid[0], 0.1, epsilon = 1e-10);
        assert_abs_diff_eq!(log_grid[2], 10.0, epsilon = 1e-10);
        assert!(log_grid[1] > log_grid[0]);
        assert!(log_grid[1] < log_grid[2]);
    }

    #[test]
    fn test_optimization_result() {
        let x = Array2::from_shape_vec((20, 3), (0..60).map(|i| i as f64 * 0.1).collect()).unwrap();

        let config = ParameterLearningConfig {
            search_strategy: SearchStrategy::GridSearch { n_points: 2 },
            ..Default::default()
        };

        let learner = ParameterLearner::new(config);
        let result = learner.optimize_rbf_parameters(&x, None).unwrap();

        // Verify result structure
        assert!(result.best_score > 0.0);
        assert!(result.best_parameters.gamma > 0.0);
        assert!(result.best_parameters.n_components > 0);
        assert!(result.converged);
        assert_eq!(result.n_evaluations, result.parameter_history.len());

        // Verify that best score is actually the best
        let max_score = result
            .parameter_history
            .iter()
            .map(|(_, score)| *score)
            .fold(f64::NEG_INFINITY, f64::max);
        assert_abs_diff_eq!(result.best_score, max_score, epsilon = 1e-10);
    }
}

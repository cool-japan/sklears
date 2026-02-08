//! Population-based Training (PBT) for hyperparameter optimization
//!
//! Population-based Training is a method for training a population of neural networks (or ML models)
//! in parallel while optimizing hyperparameters. It combines parallel search with evolutionary
//! methods to exploit and explore the hyperparameter space dynamically.
//!
//! The key idea is to periodically evaluate the performance of all models in the population,
//! then replace the worst-performing models with copies of the best-performing models,
//! with perturbed hyperparameters to continue exploration.

use crate::{CrossValidator, KFold, Scoring};
use scirs2_core::ndarray::{Array1, Array2, ArrayBase, Dim, OwnedRepr};
use scirs2_core::numeric::ToPrimitive;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Score},
};
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Configuration for Population-based Training
#[derive(Debug, Clone)]
pub struct PBTConfig {
    /// Population size (number of parallel workers)
    pub population_size: usize,
    /// How often to perform exploitation and exploration (in training steps)
    pub perturbation_interval: usize,
    /// Fraction of population to replace in each perturbation
    pub replacement_fraction: f64,
    /// Standard deviation for hyperparameter perturbation
    pub perturbation_factor: f64,
    /// Maximum number of training iterations
    pub max_iterations: usize,
    /// Early stopping patience
    pub patience: Option<usize>,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

impl Default for PBTConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            perturbation_interval: 10,
            replacement_fraction: 0.25,
            perturbation_factor: 0.2,
            max_iterations: 100,
            patience: Some(10),
            random_state: None,
        }
    }
}

/// Parameter space definition for PBT
#[derive(Debug, Clone)]
pub struct PBTParameterSpace {
    /// Continuous parameters with their bounds
    pub continuous: HashMap<String, (f64, f64)>,
    /// Discrete parameters with their possible values
    pub discrete: HashMap<String, Vec<f64>>,
}

impl Default for PBTParameterSpace {
    fn default() -> Self {
        Self::new()
    }
}

impl PBTParameterSpace {
    pub fn new() -> Self {
        Self {
            continuous: HashMap::new(),
            discrete: HashMap::new(),
        }
    }

    /// Add a continuous parameter with bounds
    pub fn add_continuous(mut self, name: &str, low: f64, high: f64) -> Self {
        self.continuous.insert(name.to_string(), (low, high));
        self
    }

    /// Add a discrete parameter with possible values
    pub fn add_discrete(mut self, name: &str, values: Vec<f64>) -> Self {
        self.discrete.insert(name.to_string(), values);
        self
    }

    /// Sample a random parameter configuration
    pub fn sample<R: Rng>(&self, rng: &mut R) -> PBTParameters {
        let mut params = PBTParameters::new();

        for (name, (low, high)) in &self.continuous {
            let value = rng.gen_range(*low..*high + 1.0);
            params.set(name.clone(), value);
        }

        for (name, values) in &self.discrete {
            let idx = rng.gen_range(0..values.len());
            params.set(name.clone(), values[idx]);
        }

        params
    }

    /// Perturb parameters with exploration
    pub fn perturb<R: Rng>(
        &self,
        params: &PBTParameters,
        factor: f64,
        rng: &mut R,
    ) -> PBTParameters {
        let mut new_params = params.clone();

        for (name, (low, high)) in &self.continuous {
            if let Some(&current_value) = params.get(name) {
                let range = high - low;
                let perturbation = rng.gen_range(-factor..factor + 1.0) * range;
                let new_value = (current_value + perturbation).clamp(*low, *high);
                new_params.set(name.clone(), new_value);
            }
        }

        for (name, values) in &self.discrete {
            if rng.gen_bool(factor.min(1.0)) {
                // Probability of changing discrete parameter
                let idx = rng.gen_range(0..values.len());
                new_params.set(name.clone(), values[idx]);
            }
        }

        new_params
    }
}

/// Parameter configuration for a single worker
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PBTParameters {
    params: HashMap<String, f64>,
}

impl Default for PBTParameters {
    fn default() -> Self {
        Self::new()
    }
}

impl PBTParameters {
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    pub fn set(&mut self, name: String, value: f64) {
        self.params.insert(name, value);
    }

    pub fn get(&self, name: &str) -> Option<&f64> {
        self.params.get(name)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &f64)> {
        self.params.iter()
    }
}

/// A worker in the PBT population
#[derive(Debug, Clone)]
pub struct PBTWorker<E> {
    /// Worker ID
    pub id: usize,
    /// Current hyperparameters
    pub parameters: PBTParameters,
    /// Current model state
    pub estimator: Option<E>,
    /// Performance history
    pub score_history: Vec<f64>,
    /// Current performance score
    pub current_score: f64,
    /// Training step count
    pub step: usize,
}

impl<E> PBTWorker<E> {
    pub fn new(id: usize, parameters: PBTParameters) -> Self {
        Self {
            id,
            parameters,
            estimator: None,
            score_history: Vec::new(),
            current_score: f64::NEG_INFINITY,
            step: 0,
        }
    }

    /// Check if this worker should be replaced based on performance
    pub fn should_be_replaced(&self, threshold_percentile: f64, all_scores: &[f64]) -> bool {
        if all_scores.is_empty() {
            return false;
        }

        let mut sorted_scores = all_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let threshold_idx = (threshold_percentile * sorted_scores.len() as f64) as usize;
        let threshold = sorted_scores
            .get(threshold_idx)
            .unwrap_or(&f64::NEG_INFINITY);

        self.current_score < *threshold
    }
}

/// Population-based Training optimizer
pub struct PopulationBasedTraining<E, X, Y> {
    config: PBTConfig,
    parameter_space: PBTParameterSpace,
    population: Vec<PBTWorker<E>>,
    rng: StdRng,
    _phantom: PhantomData<(X, Y)>,
}

impl<E, X, Y> PopulationBasedTraining<E, X, Y>
where
    E: Clone + Debug,
{
    /// Create a new PBT optimizer
    pub fn new(config: PBTConfig, parameter_space: PBTParameterSpace) -> Self {
        let rng = match config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };

        Self {
            config,
            parameter_space,
            population: Vec::new(),
            rng,
            _phantom: PhantomData,
        }
    }

    /// Initialize the population with random hyperparameters
    pub fn initialize_population(&mut self) {
        self.population.clear();

        for i in 0..self.config.population_size {
            let parameters = self.parameter_space.sample(&mut self.rng);
            let worker = PBTWorker::new(i, parameters);
            self.population.push(worker);
        }
    }

    /// Get the current population
    pub fn population(&self) -> &[PBTWorker<E>] {
        &self.population
    }

    /// Get the best worker in the current population
    pub fn best_worker(&self) -> Option<&PBTWorker<E>> {
        self.population
            .iter()
            .max_by(|a, b| a.current_score.partial_cmp(&b.current_score).unwrap())
    }

    /// Update worker performance
    pub fn update_worker_score(&mut self, worker_id: usize, score: f64) -> Result<()> {
        let worker = self
            .population
            .get_mut(worker_id)
            .ok_or_else(|| SklearsError::InvalidInput(format!("Worker {} not found", worker_id)))?;

        worker.current_score = score;
        worker.score_history.push(score);
        worker.step += 1;

        Ok(())
    }

    /// Perform exploitation and exploration step
    pub fn exploit_and_explore(&mut self) -> Result<()> {
        let population_size = self.population.len();
        if population_size == 0 {
            return Err(SklearsError::InvalidOperation(
                "Empty population".to_string(),
            ));
        }

        // Get current scores for ranking
        let scores: Vec<f64> = self.population.iter().map(|w| w.current_score).collect();

        // Identify workers to replace (bottom fraction)
        let num_to_replace = (self.config.replacement_fraction * population_size as f64) as usize;
        let mut worker_scores: Vec<(usize, f64)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        worker_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Get indices of worst and best performers
        let worst_indices: Vec<usize> = worker_scores
            .iter()
            .take(num_to_replace)
            .map(|(i, _)| *i)
            .collect();
        let best_indices: Vec<usize> = worker_scores
            .iter()
            .rev()
            .take(num_to_replace)
            .map(|(i, _)| *i)
            .collect();

        // Replace worst performers with perturbed copies of best performers
        for (&worst_idx, &best_idx) in worst_indices.iter().zip(best_indices.iter()) {
            if worst_idx != best_idx {
                // Copy the best performer's parameters
                let best_params = self.population[best_idx].parameters.clone();

                // Perturb the parameters for exploration
                let perturbed_params = self.parameter_space.perturb(
                    &best_params,
                    self.config.perturbation_factor,
                    &mut self.rng,
                );

                // Update the worst performer
                self.population[worst_idx].parameters = perturbed_params;
                self.population[worst_idx].current_score = f64::NEG_INFINITY;
                self.population[worst_idx].score_history.clear();
                self.population[worst_idx].step = 0;
                self.population[worst_idx].estimator = None;
            }
        }

        Ok(())
    }

    /// Check if any worker has converged
    pub fn check_convergence(&self) -> bool {
        if let Some(patience) = self.config.patience {
            for worker in &self.population {
                if worker.score_history.len() >= patience {
                    let recent_scores =
                        &worker.score_history[worker.score_history.len() - patience..];
                    let max_recent = recent_scores
                        .iter()
                        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                    let current = worker.current_score;

                    // If no improvement in patience steps, consider converged
                    if current <= max_recent {
                        return true;
                    }
                }
            }
        }
        false
    }

    /// Get optimization statistics
    pub fn get_statistics(&self) -> PBTStatistics {
        let scores: Vec<f64> = self.population.iter().map(|w| w.current_score).collect();

        let best_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_score = scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let mean_score = scores.iter().sum::<f64>() / scores.len() as f64;

        let variance = scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>()
            / scores.len() as f64;
        let std_dev = variance.sqrt();

        PBTStatistics {
            generation: self.population.iter().map(|w| w.step).max().unwrap_or(0),
            population_size: self.population.len(),
            best_score,
            worst_score,
            mean_score,
            std_dev,
        }
    }
}

/// Statistics for PBT optimization
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PBTStatistics {
    pub generation: usize,
    pub population_size: usize,
    pub best_score: f64,
    pub worst_score: f64,
    pub mean_score: f64,
    pub std_dev: f64,
}

/// Result of PBT optimization
#[derive(Debug, Clone)]
pub struct PBTResult<E> {
    /// Best hyperparameters found
    pub best_params: PBTParameters,
    /// Best score achieved
    pub best_score: f64,
    /// Best model/estimator
    pub best_estimator: Option<E>,
    /// Optimization history
    pub history: Vec<PBTStatistics>,
    /// Final population
    pub final_population: Vec<PBTWorker<E>>,
}

/// Configuration function type for PBT
pub type PBTConfigFn<E> = Box<dyn Fn(E, &PBTParameters) -> Result<E>>;

/// Population-based Training with cross-validation
pub struct PopulationBasedTrainingCV<E, X, Y> {
    base_estimator: E,
    config: PBTConfig,
    parameter_space: PBTParameterSpace,
    config_fn: PBTConfigFn<E>,
    cv: Box<dyn CrossValidator>,
    scoring: Option<Scoring>,
    _phantom: PhantomData<(X, Y)>,
}

impl<E, X, Y> PopulationBasedTrainingCV<E, X, Y>
where
    E: Clone + Debug,
{
    /// Create a new PBT with cross-validation
    pub fn new(
        base_estimator: E,
        config: PBTConfig,
        parameter_space: PBTParameterSpace,
        config_fn: PBTConfigFn<E>,
    ) -> Self {
        Self {
            base_estimator,
            config,
            parameter_space,
            config_fn,
            cv: Box::new(KFold::new(5)),
            scoring: None,
            _phantom: PhantomData,
        }
    }

    /// Set cross-validation strategy
    pub fn cv<CV>(mut self, cv: CV) -> Self
    where
        CV: CrossValidator + 'static,
    {
        self.cv = Box::new(cv);
        self
    }

    /// Set scoring function
    pub fn scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = Some(scoring);
        self
    }
}

impl<E> PopulationBasedTrainingCV<E, Array2<f64>, Array1<f64>>
where
    E: Clone
        + Debug
        + Fit<
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>, f64>,
        > + Score<
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>, f64>,
        >,
    E::Fitted: Clone
        + Debug
        + Predict<
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>, f64>,
        > + Score<
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 2]>, f64>,
            ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>, f64>,
        >,
{
    /// Fit the PBT optimizer with cross-validation
    pub fn fit(self, x: &Array2<f64>, y: &Array1<f64>) -> Result<PBTResult<E::Fitted>> {
        let mut pbt: PopulationBasedTraining<E::Fitted, Array2<f64>, Array1<f64>> =
            PopulationBasedTraining::new(self.config.clone(), self.parameter_space.clone());
        pbt.initialize_population();

        let mut history = Vec::new();
        let mut best_global_score = f64::NEG_INFINITY;
        let mut best_global_params = None;
        let mut best_global_estimator = None;

        for iteration in 0..self.config.max_iterations {
            // Evaluate all workers using cross-validation
            let population_size = pbt.population.len();
            for worker_id in 0..population_size {
                let worker_params = pbt.population[worker_id].parameters.clone();

                // Configure estimator with current parameters
                let configured_estimator =
                    (self.config_fn)(self.base_estimator.clone(), &worker_params)?;

                // Perform cross-validation
                let cv_splits = self.cv.split(x.nrows(), None);
                let mut cv_scores = Vec::new();

                for (train_idx, test_idx) in cv_splits {
                    let x_train_view = x.select(scirs2_core::ndarray::Axis(0), &train_idx);
                    let y_train_view = y.select(scirs2_core::ndarray::Axis(0), &train_idx);
                    let x_test_view = x.select(scirs2_core::ndarray::Axis(0), &test_idx);
                    let y_test_view = y.select(scirs2_core::ndarray::Axis(0), &test_idx);

                    let x_train = Array2::from_shape_vec(
                        (x_train_view.nrows(), x_train_view.ncols()),
                        x_train_view.iter().copied().collect(),
                    )?;
                    let y_train = Array1::from_vec(y_train_view.iter().copied().collect());
                    let x_test = Array2::from_shape_vec(
                        (x_test_view.nrows(), x_test_view.ncols()),
                        x_test_view.iter().copied().collect(),
                    )?;
                    let y_test = Array1::from_vec(y_test_view.iter().copied().collect());

                    let fitted_estimator = configured_estimator.clone().fit(&x_train, &y_train)?;
                    let score = fitted_estimator.score(&x_test, &y_test)?;
                    cv_scores.push(score);
                }

                let mean_score = cv_scores
                    .iter()
                    .copied()
                    .map(|x| x.to_f64().unwrap_or(0.0))
                    .sum::<f64>()
                    / cv_scores.len() as f64;
                pbt.update_worker_score(worker_id, mean_score)?;

                // Track global best
                if mean_score > best_global_score {
                    best_global_score = mean_score;
                    best_global_params = Some(worker_params);

                    // Train final model on full dataset
                    let final_estimator = configured_estimator.fit(x, y)?;
                    best_global_estimator = Some(final_estimator);
                }
            }

            // Record statistics
            let stats = pbt.get_statistics();
            history.push(stats);

            // Perform exploitation and exploration
            if iteration > 0 && iteration % self.config.perturbation_interval == 0 {
                pbt.exploit_and_explore()?;
            }

            // Check for convergence
            if pbt.check_convergence() {
                break;
            }
        }

        Ok(PBTResult {
            best_params: best_global_params.unwrap_or_else(PBTParameters::new),
            best_score: best_global_score,
            best_estimator: best_global_estimator,
            history,
            final_population: pbt.population,
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pbt_parameter_space() {
        let space = PBTParameterSpace::new()
            .add_continuous("learning_rate", 0.001, 0.1)
            .add_discrete("n_estimators", vec![10.0, 50.0, 100.0]);

        let mut rng = StdRng::seed_from_u64(42);
        let params = space.sample(&mut rng);

        assert!(params.get("learning_rate").is_some());
        assert!(params.get("n_estimators").is_some());
    }

    #[test]
    fn test_pbt_worker() {
        let mut params = PBTParameters::new();
        params.set("learning_rate".to_string(), 0.01);

        let worker = PBTWorker::<i32>::new(0, params);
        assert_eq!(worker.id, 0);
        assert_eq!(worker.current_score, f64::NEG_INFINITY);
    }

    #[test]
    fn test_pbt_config() {
        let config = PBTConfig::default();
        assert_eq!(config.population_size, 20);
        assert_eq!(config.perturbation_interval, 10);
    }

    #[test]
    fn test_parameter_perturbation() {
        let space = PBTParameterSpace::new().add_continuous("learning_rate", 0.001, 0.1);

        let mut params = PBTParameters::new();
        params.set("learning_rate".to_string(), 0.05);

        let mut rng = StdRng::seed_from_u64(42);
        let perturbed = space.perturb(&params, 0.1, &mut rng);

        let original = params.get("learning_rate").unwrap();
        let new_val = perturbed.get("learning_rate").unwrap();

        // Should be different but within bounds
        assert_ne!(original, new_val);
        assert!(*new_val >= 0.001 && *new_val <= 0.1);
    }
}

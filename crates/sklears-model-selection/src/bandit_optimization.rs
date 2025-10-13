//! Multi-Armed Bandit algorithms for hyperparameter optimization
//!
//! This module implements various bandit-based optimization algorithms that can be used
//! for efficient hyperparameter optimization by treating each parameter configuration
//! as an arm in a multi-armed bandit problem.

use crate::{CrossValidator, ParameterValue, Scoring};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use sklears_core::{
    error::{Result, SklearsError},
    prelude::Predict,
    traits::Fit,
    traits::Score,
    types::Float,
};
use std::collections::HashMap;

/// Configuration for bandit-based optimization
#[derive(Debug, Clone)]
pub struct BanditConfig {
    /// Number of iterations to run
    pub n_iterations: usize,
    /// Number of initial random pulls for each arm
    pub n_initial_random: usize,
    /// Exploration parameter for UCB algorithm
    pub ucb_c: f64,
    /// Temperature parameter for Boltzmann exploration
    pub temperature: f64,
    /// Decay rate for temperature cooling
    pub temperature_decay: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for BanditConfig {
    fn default() -> Self {
        Self {
            n_iterations: 100,
            n_initial_random: 5,
            ucb_c: 1.96, // 95% confidence interval
            temperature: 1.0,
            temperature_decay: 0.95,
            random_state: None,
        }
    }
}

/// Strategy for arm selection in multi-armed bandit
#[derive(Debug, Clone)]
pub enum BanditStrategy {
    /// Upper Confidence Bound
    UCB,
    /// Epsilon-greedy with decaying epsilon
    EpsilonGreedy(f64),
    /// Boltzmann exploration with temperature cooling
    Boltzmann(f64),
    /// Thompson sampling (Bayesian approach)
    ThompsonSampling,
}

/// Arm statistics for bandit algorithms
#[derive(Debug, Clone)]
struct ArmStats {
    /// Number of times this arm was pulled
    n_pulls: usize,
    /// Sum of rewards received
    sum_rewards: f64,
    /// Sum of squared rewards (for variance estimation)
    sum_squared_rewards: f64,
    /// Best score achieved
    best_score: f64,
    /// Recent scores for trend analysis
    recent_scores: Vec<f64>,
}

impl ArmStats {
    fn new() -> Self {
        Self {
            n_pulls: 0,
            sum_rewards: 0.0,
            sum_squared_rewards: 0.0,
            best_score: f64::NEG_INFINITY,
            recent_scores: Vec::new(),
        }
    }

    fn update(&mut self, reward: f64) {
        self.n_pulls += 1;
        self.sum_rewards += reward;
        self.sum_squared_rewards += reward * reward;

        if reward > self.best_score {
            self.best_score = reward;
        }

        // Keep only recent scores for trend analysis
        self.recent_scores.push(reward);
        if self.recent_scores.len() > 10 {
            self.recent_scores.remove(0);
        }
    }

    fn mean_reward(&self) -> f64 {
        if self.n_pulls == 0 {
            0.0
        } else {
            self.sum_rewards / self.n_pulls as f64
        }
    }

    fn variance(&self) -> f64 {
        if self.n_pulls <= 1 {
            0.0
        } else {
            let mean = self.mean_reward();
            (self.sum_squared_rewards / self.n_pulls as f64) - (mean * mean)
        }
    }

    fn confidence_interval(&self, confidence: f64) -> f64 {
        if self.n_pulls == 0 {
            f64::INFINITY
        } else {
            let std_err = (self.variance() / self.n_pulls as f64).sqrt();
            confidence * std_err
        }
    }
}

/// Result of bandit optimization
#[derive(Debug, Clone)]
pub struct BanditOptimizationResult {
    /// Best parameter configuration found
    pub best_params: ParameterValue,
    /// Best score achieved
    pub best_score: f64,
    /// All parameter configurations tried
    pub all_params: Vec<ParameterValue>,
    /// All scores achieved
    pub all_scores: Vec<f64>,
    /// Convergence history
    pub convergence_history: Vec<f64>,
    /// Number of iterations run
    pub n_iterations: usize,
    /// Final arm statistics
    pub arm_stats: HashMap<String, (f64, usize)>, // (mean_reward, n_pulls)
}

/// Multi-Armed Bandit hyperparameter optimizer
pub struct BanditOptimizer {
    /// Parameter space to search
    param_space: Vec<ParameterValue>,
    /// Bandit strategy to use
    strategy: BanditStrategy,
    /// Configuration
    config: BanditConfig,
    /// Arm statistics
    arm_stats: Vec<ArmStats>,
    /// Random number generator
    rng: StdRng,
    /// Current iteration
    current_iteration: usize,
    /// Current temperature (for Boltzmann)
    current_temperature: f64,
}

impl BanditOptimizer {
    pub fn new(
        param_space: Vec<ParameterValue>,
        strategy: BanditStrategy,
        config: BanditConfig,
    ) -> Self {
        let rng = if let Some(seed) = config.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(42)
        };

        let arm_stats = vec![ArmStats::new(); param_space.len()];
        let current_temperature = config.temperature;

        Self {
            param_space,
            strategy,
            config,
            arm_stats,
            rng,
            current_iteration: 0,
            current_temperature,
        }
    }

    /// Select next arm to pull based on strategy
    fn select_arm(&mut self) -> usize {
        // Initial random exploration
        if self.current_iteration < self.config.n_initial_random * self.param_space.len() {
            return self.rng.gen_range(0..self.param_space.len());
        }

        match &self.strategy {
            BanditStrategy::UCB => self.select_ucb_arm(),
            BanditStrategy::EpsilonGreedy(epsilon) => self.select_epsilon_greedy_arm(*epsilon),
            BanditStrategy::Boltzmann(_) => self.select_boltzmann_arm(),
            BanditStrategy::ThompsonSampling => self.select_thompson_sampling_arm(),
        }
    }

    /// UCB arm selection
    fn select_ucb_arm(&self) -> usize {
        let total_pulls = self
            .arm_stats
            .iter()
            .map(|stats| stats.n_pulls)
            .sum::<usize>();
        let log_total = (total_pulls as f64).ln();

        let mut best_arm = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            if stats.n_pulls == 0 {
                return i; // Prefer unvisited arms
            }

            let mean_reward = stats.mean_reward();
            let confidence_bonus = self.config.ucb_c * (log_total / stats.n_pulls as f64).sqrt();
            let ucb_value = mean_reward + confidence_bonus;

            if ucb_value > best_ucb {
                best_ucb = ucb_value;
                best_arm = i;
            }
        }

        best_arm
    }

    /// Epsilon-greedy arm selection with decaying epsilon
    fn select_epsilon_greedy_arm(&mut self, base_epsilon: f64) -> usize {
        let decayed_epsilon = base_epsilon / (1.0 + self.current_iteration as f64 * 0.01);

        if self.rng.gen::<f64>() < decayed_epsilon {
            // Explore: random arm
            self.rng.gen_range(0..self.param_space.len())
        } else {
            // Exploit: best arm
            let mut best_arm = 0;
            let mut best_reward = f64::NEG_INFINITY;

            for (i, stats) in self.arm_stats.iter().enumerate() {
                let reward = if stats.n_pulls == 0 {
                    f64::INFINITY // Prefer unvisited arms
                } else {
                    stats.mean_reward()
                };

                if reward > best_reward {
                    best_reward = reward;
                    best_arm = i;
                }
            }

            best_arm
        }
    }

    /// Boltzmann exploration arm selection
    fn select_boltzmann_arm(&mut self) -> usize {
        let mut unnormalized_probs = Vec::with_capacity(self.param_space.len());

        for stats in &self.arm_stats {
            let reward = if stats.n_pulls == 0 {
                1.0 // Give unvisited arms high probability
            } else {
                stats.mean_reward()
            };

            unnormalized_probs.push((reward / self.current_temperature).exp());
        }

        // Normalize probabilities
        let total: f64 = unnormalized_probs.iter().sum();
        let probs: Vec<f64> = unnormalized_probs.iter().map(|p| p / total).collect();

        // Sample according to probabilities
        let mut cumsum = 0.0;
        let random_val = self.rng.gen::<f64>();

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob;
            if random_val <= cumsum {
                return i;
            }
        }

        // Fallback
        self.param_space.len() - 1
    }

    /// Thompson sampling arm selection
    fn select_thompson_sampling_arm(&mut self) -> usize {
        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            let sample = if stats.n_pulls == 0 {
                // Sample from uninformative prior
                self.rng.gen::<f64>()
            } else {
                // Sample from posterior (assuming Gaussian with known variance)
                let mean = stats.mean_reward();
                let std = (stats.variance() / stats.n_pulls as f64).sqrt().max(0.1);

                use scirs2_core::random::RandNormal;
                let normal = RandNormal::new(mean, std).unwrap();
                self.rng.sample(normal)
            };

            if sample > best_sample {
                best_sample = sample;
                best_arm = i;
            }
        }

        best_arm
    }

    /// Update arm statistics with new reward
    fn update_arm(&mut self, arm: usize, reward: f64) {
        self.arm_stats[arm].update(reward);

        // Update temperature for Boltzmann
        if matches!(self.strategy, BanditStrategy::Boltzmann(_)) {
            self.current_temperature *= self.config.temperature_decay;
        }

        self.current_iteration += 1;
    }

    /// Get the best arm so far
    fn best_arm(&self) -> (usize, f64) {
        let mut best_arm = 0;
        let mut best_score = f64::NEG_INFINITY;

        for (i, stats) in self.arm_stats.iter().enumerate() {
            if stats.best_score > best_score {
                best_score = stats.best_score;
                best_arm = i;
            }
        }

        (best_arm, best_score)
    }
}

/// Bandit-based cross-validation optimizer
pub struct BanditSearchCV<E> {
    /// Base estimator
    estimator: E,
    /// Parameter space
    param_space: Vec<ParameterValue>,
    /// Bandit strategy
    strategy: BanditStrategy,
    /// Configuration
    config: BanditConfig,
    /// Scoring method
    scoring: Option<Scoring>,
    /// Parameter configuration function
    param_config_fn: Option<Box<dyn Fn(E, &ParameterValue) -> Result<E>>>,
}

impl<E> BanditSearchCV<E>
where
    E: Clone,
{
    /// Create a new bandit search CV
    pub fn new(estimator: E, param_space: Vec<ParameterValue>) -> Self {
        Self {
            estimator,
            param_space,
            strategy: BanditStrategy::UCB,
            config: BanditConfig::default(),
            scoring: None,
            param_config_fn: None,
        }
    }

    /// Set the bandit strategy
    pub fn with_strategy(mut self, strategy: BanditStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the configuration
    pub fn with_config(mut self, config: BanditConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the scoring method
    pub fn with_scoring(mut self, scoring: Scoring) -> Self {
        self.scoring = Some(scoring);
        self
    }

    /// Set the parameter configuration function
    pub fn with_param_config<F>(mut self, func: F) -> Self
    where
        F: Fn(E, &ParameterValue) -> Result<E> + 'static,
    {
        self.param_config_fn = Some(Box::new(func));
        self
    }

    /// Fit the bandit search optimizer
    pub fn fit<F, C>(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        cv: &C,
    ) -> Result<BanditOptimizationResult>
    where
        F: Clone,
        E: Fit<Array2<Float>, Array1<Float>, Fitted = F>,
        F: Predict<Array2<Float>, Array1<Float>>,
        F: Score<Array2<Float>, Array1<Float>, Float = f64>,
        C: CrossValidator,
    {
        if self.param_space.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Parameter space cannot be empty".to_string(),
            ));
        }

        let param_config_fn = self.param_config_fn.as_ref().ok_or_else(|| {
            SklearsError::InvalidInput("Parameter configuration function not set".to_string())
        })?;

        let mut optimizer = BanditOptimizer::new(
            self.param_space.clone(),
            self.strategy.clone(),
            self.config.clone(),
        );

        let mut all_params = Vec::new();
        let mut all_scores = Vec::new();
        let mut convergence_history = Vec::new();

        for _ in 0..self.config.n_iterations {
            // Select arm (parameter configuration)
            let arm_idx = optimizer.select_arm();
            let param = &self.param_space[arm_idx];

            // Configure estimator
            let configured_estimator = param_config_fn(self.estimator.clone(), param)?;

            // Evaluate using cross-validation
            let scores = crate::validation::cross_val_score(
                configured_estimator,
                x,
                y,
                cv,
                self.scoring.clone(),
                None,
            )?;

            let mean_score = scores.mean().unwrap_or(0.0);

            // Update bandit statistics
            optimizer.update_arm(arm_idx, mean_score);

            // Track progress
            all_params.push(param.clone());
            all_scores.push(mean_score);

            let (_, current_best_score) = optimizer.best_arm();
            convergence_history.push(current_best_score);
        }

        // Get final results
        let (best_arm_idx, best_score) = optimizer.best_arm();
        let best_params = self.param_space[best_arm_idx].clone();

        // Collect arm statistics
        let mut arm_stats = HashMap::new();
        for (i, stats) in optimizer.arm_stats.iter().enumerate() {
            arm_stats.insert(format!("arm_{}", i), (stats.mean_reward(), stats.n_pulls));
        }

        Ok(BanditOptimizationResult {
            best_params,
            best_score,
            all_params,
            all_scores,
            convergence_history,
            n_iterations: self.config.n_iterations,
            arm_stats,
        })
    }
}

/// Bandit-based hyperparameter optimization with cross-validation
#[derive(Debug, Clone)]
pub struct BanditOptimization<E, S> {
    estimator: E,
    parameter_space: Vec<ParameterValue>,
    scorer: Box<S>,
    cv_folds: usize,
    n_iter: usize,
    strategy: BanditStrategy,
    random_state: Option<u64>,
    arm_stats: HashMap<usize, ArmStats>,
}

impl<E, S> BanditOptimization<E, S>
where
    E: Clone + Fit<Array2<Float>, Array1<Float>> + Predict<Array2<Float>, Array1<Float>>,
    S: Fn(&E::Fitted, &Array2<Float>, &Array1<Float>) -> Result<f64>,
{
    /// Create a new bandit optimization
    pub fn new(
        estimator: E,
        parameter_space: Vec<ParameterValue>,
        scorer: Box<S>,
        cv_folds: usize,
    ) -> Self {
        Self {
            estimator,
            parameter_space,
            scorer,
            cv_folds,
            n_iter: 100,
            strategy: BanditStrategy::UCB,
            random_state: None,
            arm_stats: HashMap::new(),
        }
    }

    /// Set number of iterations
    pub fn set_n_iter(&mut self, n_iter: usize) {
        self.n_iter = n_iter;
    }

    /// Set bandit strategy
    pub fn set_strategy(&mut self, strategy: BanditStrategy) {
        self.strategy = strategy;
    }

    /// Set random state for reproducibility
    pub fn set_random_state(&mut self, seed: u64) {
        self.random_state = Some(seed);
    }

    /// Run bandit optimization
    pub fn fit(
        &mut self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BanditOptimizationResult> {
        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        let mut best_score = f64::NEG_INFINITY;
        let mut best_arm = 0;
        let n_arms = self.parameter_space.len();

        // Initialize arm statistics
        for i in 0..n_arms {
            self.arm_stats.insert(i, ArmStats::new());
        }

        // Bandit optimization loop
        for iteration in 0..self.n_iter {
            // Select arm based on strategy
            let selected_arm = self.select_arm(iteration, &mut rng)?;

            // Evaluate selected arm
            let score = self.evaluate_arm(selected_arm, x, y)?;

            // Update arm statistics
            if let Some(stats) = self.arm_stats.get_mut(&selected_arm) {
                stats.update(score);

                if score > best_score {
                    best_score = score;
                    best_arm = selected_arm;
                }
            }
        }

        // Convert arm_stats to the expected format
        let arm_stats_converted: HashMap<String, (f64, usize)> = self
            .arm_stats
            .iter()
            .map(|(&idx, stats)| (format!("arm_{}", idx), (stats.mean_reward(), stats.n_pulls)))
            .collect();

        Ok(BanditOptimizationResult {
            best_params: self.parameter_space[best_arm].clone(),
            best_score,
            all_params: self.parameter_space.clone(),
            all_scores: self.arm_stats.values().map(|s| s.best_score).collect(),
            convergence_history: vec![best_score], // Simplified
            n_iterations: self.n_iter,
            arm_stats: arm_stats_converted,
        })
    }

    /// Select arm based on bandit strategy
    fn select_arm(&self, iteration: usize, rng: &mut StdRng) -> Result<usize> {
        let n_arms = self.parameter_space.len();

        match &self.strategy {
            BanditStrategy::UCB => {
                let mut best_value = f64::NEG_INFINITY;
                let mut best_arm = 0;

                for arm in 0..n_arms {
                    if let Some(stats) = self.arm_stats.get(&arm) {
                        let value = if stats.n_pulls == 0 {
                            f64::INFINITY
                        } else {
                            let confidence =
                                (2.0 * (iteration as f64 + 1.0).ln() / stats.n_pulls as f64).sqrt();
                            stats.mean_reward() + 1.96 * confidence // UCB with c=1.96
                        };

                        if value > best_value {
                            best_value = value;
                            best_arm = arm;
                        }
                    }
                }

                Ok(best_arm)
            }
            BanditStrategy::EpsilonGreedy(epsilon) => {
                if rng.gen::<f64>() < *epsilon {
                    // Explore: random arm
                    Ok(rng.gen_range(0..n_arms))
                } else {
                    // Exploit: best arm so far
                    let mut best_mean = f64::NEG_INFINITY;
                    let mut best_arm = 0;

                    for arm in 0..n_arms {
                        if let Some(stats) = self.arm_stats.get(&arm) {
                            let mean = stats.mean_reward();
                            if mean > best_mean {
                                best_mean = mean;
                                best_arm = arm;
                            }
                        }
                    }

                    Ok(best_arm)
                }
            }
            BanditStrategy::ThompsonSampling => {
                // Simplified Thompson sampling (assuming beta distribution)
                let mut best_sample = f64::NEG_INFINITY;
                let mut best_arm = 0;

                for arm in 0..n_arms {
                    if let Some(stats) = self.arm_stats.get(&arm) {
                        // Sample from posterior (simplified)
                        let alpha = stats.sum_rewards + 1.0;
                        let beta = (stats.n_pulls as f64 - stats.sum_rewards) + 1.0;
                        let sample = rng.gen::<f64>().powf(1.0 / alpha); // Simplified beta sampling

                        if sample > best_sample {
                            best_sample = sample;
                            best_arm = arm;
                        }
                    }
                }

                Ok(best_arm)
            }
            BanditStrategy::Boltzmann(temperature) => {
                // Softmax selection
                let mut weights = Vec::new();
                let mut max_mean = f64::NEG_INFINITY;

                // Find max for numerical stability
                for arm in 0..n_arms {
                    if let Some(stats) = self.arm_stats.get(&arm) {
                        let mean = stats.mean_reward();
                        if mean > max_mean {
                            max_mean = mean;
                        }
                    }
                }

                // Calculate softmax weights
                for arm in 0..n_arms {
                    if let Some(stats) = self.arm_stats.get(&arm) {
                        let mean = stats.mean_reward();
                        weights.push(((mean - max_mean) / temperature).exp());
                    } else {
                        weights.push(1.0);
                    }
                }

                // Sample from categorical distribution
                let sum: f64 = weights.iter().sum();
                let mut cumsum = 0.0;
                let threshold = rng.gen::<f64>() * sum;

                for (arm, weight) in weights.iter().enumerate() {
                    cumsum += weight;
                    if cumsum >= threshold {
                        return Ok(arm);
                    }
                }

                Ok(n_arms - 1) // Fallback
            }
        }
    }

    /// Evaluate a specific arm (parameter configuration)
    fn evaluate_arm(&self, arm: usize, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        // Simple train-test split for evaluation (placeholder for full CV)
        let n_samples = x.nrows();
        let train_size = (n_samples as f64 * 0.8) as usize;

        let x_train = x
            .slice(scirs2_core::ndarray::s![..train_size, ..])
            .to_owned();
        let y_train = y.slice(scirs2_core::ndarray::s![..train_size]).to_owned();
        let x_test = x
            .slice(scirs2_core::ndarray::s![train_size.., ..])
            .to_owned();
        let y_test = y.slice(scirs2_core::ndarray::s![train_size..]).to_owned();

        // Configure estimator with selected parameter (simplified)
        let estimator = self.estimator.clone();
        let fitted = estimator.fit(&x_train, &y_train)?;

        (self.scorer)(&fitted, &x_test, &y_test)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::KFold;
    use scirs2_core::ndarray::array;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator {
        param: f64,
    }

    #[derive(Clone)]
    struct MockFitted {
        param: f64,
    }

    impl Fit<Array2<Float>, Array1<Float>> for MockEstimator {
        type Fitted = MockFitted;

        fn fit(self, _x: &Array2<Float>, _y: &Array1<Float>) -> Result<Self::Fitted> {
            Ok(MockFitted { param: self.param })
        }
    }

    impl Predict<Array2<Float>, Array1<Float>> for MockFitted {
        fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
            // Simple prediction based on parameter value
            Ok(Array1::from_elem(x.nrows(), self.param))
        }
    }

    impl Score<Array2<Float>, Array1<Float>> for MockFitted {
        type Float = Float;

        fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
            let y_pred = self.predict(x)?;
            // Score is higher when predictions stay close to targets
            let mean_abs_error = (&y_pred - y).mapv(|diff| diff.abs()).mean().unwrap_or(0.0);
            Ok(1.0 - mean_abs_error) // Higher score for lower error
        }
    }

    #[test]
    fn test_bandit_optimization_ucb() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Mean = 3.5

        let estimator = MockEstimator { param: 0.0 };
        let param_space = vec![
            ParameterValue::Float(1.0),
            ParameterValue::Float(3.5), // Should be best
            ParameterValue::Float(5.0),
        ];

        let param_config_fn = |_estimator: MockEstimator, param_value: &ParameterValue| {
            if let ParameterValue::Float(val) = param_value {
                Ok(MockEstimator { param: *val })
            } else {
                Err(SklearsError::InvalidInput(
                    "Expected float parameter".to_string(),
                ))
            }
        };

        let config = BanditConfig {
            n_iterations: 30,
            n_initial_random: 2,
            ..Default::default()
        };

        let search = BanditSearchCV::new(estimator, param_space)
            .with_strategy(BanditStrategy::UCB)
            .with_config(config)
            .with_param_config(param_config_fn);

        let cv = KFold::new(3);
        let result = search.fit(&x, &y, &cv).unwrap();

        // The best parameter should be close to 3.5
        if let ParameterValue::Float(best_val) = result.best_params {
            assert!((best_val - 3.5).abs() < 2.0); // Allow some tolerance
        }

        assert_eq!(result.n_iterations, 30);
        assert_eq!(result.all_scores.len(), 30);
        assert_eq!(result.convergence_history.len(), 30);

        // Convergence should improve over time
        let early_best = result.convergence_history[..10]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let late_best = result.convergence_history[20..]
            .iter()
            .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        assert!(late_best >= early_best);
    }

    #[test]
    fn test_bandit_optimization_epsilon_greedy() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let estimator = MockEstimator { param: 0.0 };
        let param_space = vec![
            ParameterValue::Float(1.0),
            ParameterValue::Float(3.5),
            ParameterValue::Float(5.0),
        ];

        let param_config_fn = |_estimator: MockEstimator, param_value: &ParameterValue| {
            if let ParameterValue::Float(val) = param_value {
                Ok(MockEstimator { param: *val })
            } else {
                Err(SklearsError::InvalidInput(
                    "Expected float parameter".to_string(),
                ))
            }
        };

        let config = BanditConfig {
            n_iterations: 20,
            n_initial_random: 1,
            random_state: Some(42),
            ..Default::default()
        };

        let search = BanditSearchCV::new(estimator, param_space)
            .with_strategy(BanditStrategy::EpsilonGreedy(0.1))
            .with_config(config)
            .with_param_config(param_config_fn);

        let cv = KFold::new(2);
        let result = search.fit(&x, &y, &cv).unwrap();

        assert_eq!(result.n_iterations, 20);
        assert!(result.best_score.is_finite());
        assert!(!result.arm_stats.is_empty());
    }

    #[test]
    fn test_arm_stats() {
        let mut stats = ArmStats::new();

        assert_eq!(stats.n_pulls, 0);
        assert_eq!(stats.mean_reward(), 0.0);

        stats.update(1.0);
        stats.update(2.0);
        stats.update(3.0);

        assert_eq!(stats.n_pulls, 3);
        assert_eq!(stats.mean_reward(), 2.0);
        assert_eq!(stats.best_score, 3.0);
        assert!(stats.variance() > 0.0);
    }
}

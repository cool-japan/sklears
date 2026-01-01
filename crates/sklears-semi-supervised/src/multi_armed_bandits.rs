//! Multi-armed bandits for active learning
//!
//! This module implements various multi-armed bandit strategies for adaptive
//! sample selection in active learning scenarios, treating different query
//! strategies as arms in a bandit problem.

use scirs2_core::ndarray_ext::{Array1, ArrayView1};
use scirs2_core::random::{Random, Rng};
use sklears_core::error::{Result, SklearsError};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum BanditError {
    #[error("Invalid epsilon parameter: {0}")]
    InvalidEpsilon(f64),
    #[error("Invalid temperature parameter: {0}")]
    InvalidTemperature(f64),
    #[error("Invalid confidence parameter: {0}")]
    InvalidConfidence(f64),
    #[error("Invalid exploration parameter: {0}")]
    InvalidExploration(f64),
    #[error("Invalid arm index: {arm_idx} for {n_arms} arms")]
    InvalidArmIndex { arm_idx: usize, n_arms: usize },
    #[error("No arms available")]
    NoArmsAvailable,
    #[error("Insufficient data for arm: {0}")]
    InsufficientDataForArm(usize),
    #[error("Bandit computation failed: {0}")]
    BanditComputationFailed(String),
}

impl From<BanditError> for SklearsError {
    fn from(err: BanditError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}

/// Epsilon-Greedy strategy for multi-armed bandit active learning
///
/// This strategy selects the best-performing query strategy with probability (1-ε)
/// and explores a random strategy with probability ε.
#[derive(Debug, Clone)]
pub struct EpsilonGreedy {
    /// epsilon
    pub epsilon: f64,
    /// decay_rate
    pub decay_rate: f64,
    /// min_epsilon
    pub min_epsilon: f64,
    /// random_state
    pub random_state: Option<u64>,
    arm_counts: Vec<usize>,
    arm_rewards: Vec<f64>,
    total_rounds: usize,
}

impl EpsilonGreedy {
    pub fn new(epsilon: f64) -> Result<Self> {
        if !(0.0..=1.0).contains(&epsilon) {
            return Err(BanditError::InvalidEpsilon(epsilon).into());
        }
        Ok(Self {
            epsilon,
            decay_rate: 0.995,
            min_epsilon: 0.01,
            random_state: None,
            arm_counts: Vec::new(),
            arm_rewards: Vec::new(),
            total_rounds: 0,
        })
    }

    pub fn decay_rate(mut self, decay_rate: f64) -> Self {
        self.decay_rate = decay_rate;
        self
    }

    pub fn min_epsilon(mut self, min_epsilon: f64) -> Self {
        self.min_epsilon = min_epsilon;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn initialize(&mut self, n_arms: usize) {
        self.arm_counts = vec![0; n_arms];
        self.arm_rewards = vec![0.0; n_arms];
        self.total_rounds = 0;
    }

    pub fn select_arm(&mut self) -> Result<usize> {
        if self.arm_counts.is_empty() {
            return Err(BanditError::NoArmsAvailable.into());
        }

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let current_epsilon =
            (self.epsilon * self.decay_rate.powi(self.total_rounds as i32)).max(self.min_epsilon);

        if rng.gen::<f64>() < current_epsilon {
            // Explore: select random arm
            Ok(rng.gen_range(0..self.arm_counts.len()))
        } else {
            // Exploit: select arm with highest average reward
            let mut best_arm = 0;
            let mut best_reward = f64::NEG_INFINITY;

            for (arm_idx, &count) in self.arm_counts.iter().enumerate() {
                let avg_reward = if count > 0 {
                    self.arm_rewards[arm_idx] / count as f64
                } else {
                    0.0
                };

                if avg_reward > best_reward {
                    best_reward = avg_reward;
                    best_arm = arm_idx;
                }
            }

            Ok(best_arm)
        }
    }

    pub fn update(&mut self, arm_idx: usize, reward: f64) -> Result<()> {
        if arm_idx >= self.arm_counts.len() {
            return Err(BanditError::InvalidArmIndex {
                arm_idx,
                n_arms: self.arm_counts.len(),
            }
            .into());
        }

        self.arm_counts[arm_idx] += 1;
        self.arm_rewards[arm_idx] += reward;
        self.total_rounds += 1;

        Ok(())
    }

    pub fn get_arm_statistics(&self) -> Vec<(usize, f64, f64)> {
        self.arm_counts
            .iter()
            .enumerate()
            .map(|(idx, &count)| {
                let avg_reward = if count > 0 {
                    self.arm_rewards[idx] / count as f64
                } else {
                    0.0
                };
                (count, avg_reward, self.arm_rewards[idx])
            })
            .collect()
    }
}

/// Upper Confidence Bound (UCB) strategy for multi-armed bandit active learning
///
/// This strategy selects arms based on an upper confidence bound that balances
/// exploitation of good arms with exploration of uncertain arms.
#[derive(Debug, Clone)]
pub struct UpperConfidenceBound {
    /// confidence
    pub confidence: f64,
    /// random_state
    pub random_state: Option<u64>,
    arm_counts: Vec<usize>,
    arm_rewards: Vec<f64>,
    total_rounds: usize,
}

impl UpperConfidenceBound {
    pub fn new(confidence: f64) -> Result<Self> {
        if confidence <= 0.0 {
            return Err(BanditError::InvalidConfidence(confidence).into());
        }
        Ok(Self {
            confidence,
            random_state: None,
            arm_counts: Vec::new(),
            arm_rewards: Vec::new(),
            total_rounds: 0,
        })
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn initialize(&mut self, n_arms: usize) {
        self.arm_counts = vec![0; n_arms];
        self.arm_rewards = vec![0.0; n_arms];
        self.total_rounds = 0;
    }

    pub fn select_arm(&mut self) -> Result<usize> {
        if self.arm_counts.is_empty() {
            return Err(BanditError::NoArmsAvailable.into());
        }

        // If any arm has not been tried, select it first
        for (arm_idx, &count) in self.arm_counts.iter().enumerate() {
            if count == 0 {
                return Ok(arm_idx);
            }
        }

        // Calculate UCB values for all arms
        let mut best_arm = 0;
        let mut best_ucb = f64::NEG_INFINITY;

        for (arm_idx, &count) in self.arm_counts.iter().enumerate() {
            let avg_reward = self.arm_rewards[arm_idx] / count as f64;
            let confidence_interval =
                self.confidence * ((self.total_rounds as f64).ln() / count as f64).sqrt();
            let ucb_value = avg_reward + confidence_interval;

            if ucb_value > best_ucb {
                best_ucb = ucb_value;
                best_arm = arm_idx;
            }
        }

        Ok(best_arm)
    }

    pub fn update(&mut self, arm_idx: usize, reward: f64) -> Result<()> {
        if arm_idx >= self.arm_counts.len() {
            return Err(BanditError::InvalidArmIndex {
                arm_idx,
                n_arms: self.arm_counts.len(),
            }
            .into());
        }

        self.arm_counts[arm_idx] += 1;
        self.arm_rewards[arm_idx] += reward;
        self.total_rounds += 1;

        Ok(())
    }

    pub fn get_arm_statistics(&self) -> Vec<(usize, f64, f64, f64)> {
        self.arm_counts
            .iter()
            .enumerate()
            .map(|(idx, &count)| {
                let avg_reward = if count > 0 {
                    self.arm_rewards[idx] / count as f64
                } else {
                    0.0
                };
                let confidence_interval = if count > 0 && self.total_rounds > 0 {
                    self.confidence * ((self.total_rounds as f64).ln() / count as f64).sqrt()
                } else {
                    f64::INFINITY
                };
                let ucb_value = avg_reward + confidence_interval;
                (count, avg_reward, confidence_interval, ucb_value)
            })
            .collect()
    }
}

/// Thompson Sampling strategy for multi-armed bandit active learning
///
/// This strategy uses Bayesian inference to maintain probability distributions
/// over the reward rates of each arm and samples from these distributions.
#[derive(Debug, Clone)]
pub struct ThompsonSampling {
    /// random_state
    pub random_state: Option<u64>,
    alpha_params: Vec<f64>,
    beta_params: Vec<f64>,
    total_rounds: usize,
}

impl Default for ThompsonSampling {
    fn default() -> Self {
        Self::new()
    }
}

impl ThompsonSampling {
    pub fn new() -> Self {
        Self {
            random_state: None,
            alpha_params: Vec::new(),
            beta_params: Vec::new(),
            total_rounds: 0,
        }
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn initialize(&mut self, n_arms: usize) {
        // Initialize with Beta(1, 1) priors (uniform distribution)
        self.alpha_params = vec![1.0; n_arms];
        self.beta_params = vec![1.0; n_arms];
        self.total_rounds = 0;
    }

    pub fn select_arm(&mut self) -> Result<usize> {
        if self.alpha_params.is_empty() {
            return Err(BanditError::NoArmsAvailable.into());
        }

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let mut best_arm = 0;
        let mut best_sample = f64::NEG_INFINITY;

        // Sample from Beta distributions for each arm
        for (arm_idx, (&alpha, &beta)) in self
            .alpha_params
            .iter()
            .zip(self.beta_params.iter())
            .enumerate()
        {
            // Simplified Thompson sampling using uniform distribution
            // In practice, this would use proper Beta(alpha, beta) sampling
            let sample = alpha / (alpha + beta) + rng.random_range(-0.1..0.1);

            if sample > best_sample {
                best_sample = sample;
                best_arm = arm_idx;
            }
        }

        Ok(best_arm)
    }

    pub fn update(&mut self, arm_idx: usize, reward: f64) -> Result<()> {
        if arm_idx >= self.alpha_params.len() {
            return Err(BanditError::InvalidArmIndex {
                arm_idx,
                n_arms: self.alpha_params.len(),
            }
            .into());
        }

        // Update Beta distribution parameters
        // Reward should be normalized to [0, 1] for Beta distribution
        let normalized_reward = reward.clamp(0.0, 1.0);

        if normalized_reward > 0.5 {
            // Treat as success
            self.alpha_params[arm_idx] += 1.0;
        } else {
            // Treat as failure
            self.beta_params[arm_idx] += 1.0;
        }

        self.total_rounds += 1;
        Ok(())
    }

    pub fn get_arm_statistics(&self) -> Vec<(f64, f64, f64, f64)> {
        self.alpha_params
            .iter()
            .zip(self.beta_params.iter())
            .map(|(&alpha, &beta)| {
                let mean = alpha / (alpha + beta);
                let variance = (alpha * beta) / ((alpha + beta).powi(2) * (alpha + beta + 1.0));
                (alpha, beta, mean, variance)
            })
            .collect()
    }
}

/// Contextual Bandit for active learning with context features
///
/// This extends multi-armed bandits to include contextual information,
/// making decisions based on both historical performance and current context.
#[derive(Debug, Clone)]
pub struct ContextualBandit {
    /// learning_rate
    pub learning_rate: f64,
    /// exploration
    pub exploration: f64,
    /// random_state
    pub random_state: Option<u64>,
    arm_weights: Vec<Array1<f64>>,
    context_dim: usize,
    total_rounds: usize,
}

impl ContextualBandit {
    pub fn new(exploration: f64) -> Result<Self> {
        if exploration < 0.0 {
            return Err(BanditError::InvalidExploration(exploration).into());
        }
        Ok(Self {
            learning_rate: 0.1,
            exploration,
            random_state: None,
            arm_weights: Vec::new(),
            context_dim: 0,
            total_rounds: 0,
        })
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn initialize(&mut self, n_arms: usize, context_dim: usize) {
        self.context_dim = context_dim;
        self.arm_weights = vec![Array1::zeros(context_dim); n_arms];
        self.total_rounds = 0;
    }

    pub fn select_arm(&mut self, context: &ArrayView1<f64>) -> Result<usize> {
        if self.arm_weights.is_empty() {
            return Err(BanditError::NoArmsAvailable.into());
        }

        if context.len() != self.context_dim {
            return Err(BanditError::BanditComputationFailed(format!(
                "Context dimension mismatch: expected {}, got {}",
                self.context_dim,
                context.len()
            ))
            .into());
        }

        let mut rng = match self.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Epsilon-greedy with linear contextual bandits
        if rng.gen::<f64>() < self.exploration {
            // Explore: select random arm
            Ok(rng.gen_range(0..self.arm_weights.len()))
        } else {
            // Exploit: select arm with highest predicted reward
            let mut best_arm = 0;
            let mut best_reward = f64::NEG_INFINITY;

            for (arm_idx, weights) in self.arm_weights.iter().enumerate() {
                let predicted_reward = weights.dot(context);
                if predicted_reward > best_reward {
                    best_reward = predicted_reward;
                    best_arm = arm_idx;
                }
            }

            Ok(best_arm)
        }
    }

    pub fn update(&mut self, arm_idx: usize, context: &ArrayView1<f64>, reward: f64) -> Result<()> {
        if arm_idx >= self.arm_weights.len() {
            return Err(BanditError::InvalidArmIndex {
                arm_idx,
                n_arms: self.arm_weights.len(),
            }
            .into());
        }

        if context.len() != self.context_dim {
            return Err(BanditError::BanditComputationFailed(format!(
                "Context dimension mismatch: expected {}, got {}",
                self.context_dim,
                context.len()
            ))
            .into());
        }

        // Gradient update for linear contextual bandit
        let predicted_reward = self.arm_weights[arm_idx].dot(context);
        let error = reward - predicted_reward;

        // Update weights: w = w + α * error * context
        for i in 0..self.context_dim {
            self.arm_weights[arm_idx][i] += self.learning_rate * error * context[i];
        }

        self.total_rounds += 1;
        Ok(())
    }

    pub fn get_arm_weights(&self) -> &Vec<Array1<f64>> {
        &self.arm_weights
    }

    pub fn predict_rewards(&self, context: &ArrayView1<f64>) -> Result<Array1<f64>> {
        if context.len() != self.context_dim {
            return Err(BanditError::BanditComputationFailed(format!(
                "Context dimension mismatch: expected {}, got {}",
                self.context_dim,
                context.len()
            ))
            .into());
        }

        let mut predicted_rewards = Array1::zeros(self.arm_weights.len());
        for (arm_idx, weights) in self.arm_weights.iter().enumerate() {
            predicted_rewards[arm_idx] = weights.dot(context);
        }

        Ok(predicted_rewards)
    }
}

/// Bandit-based Active Learning coordinator
///
/// This coordinates multiple query strategies using bandit algorithms,
/// treating each strategy as an arm and adaptively selecting strategies
/// based on their performance.
#[derive(Debug, Clone)]
pub struct BanditBasedActiveLearning {
    /// strategy_names
    pub strategy_names: Vec<String>,
    /// bandit_algorithm
    pub bandit_algorithm: String,
    /// reward_function
    pub reward_function: String,
    /// random_state
    pub random_state: Option<u64>,
    epsilon_greedy: Option<EpsilonGreedy>,
    ucb: Option<UpperConfidenceBound>,
    thompson: Option<ThompsonSampling>,
    contextual: Option<ContextualBandit>,
}

impl BanditBasedActiveLearning {
    pub fn new(strategy_names: Vec<String>, bandit_algorithm: String) -> Self {
        Self {
            strategy_names,
            bandit_algorithm,
            reward_function: "accuracy_improvement".to_string(),
            random_state: None,
            epsilon_greedy: None,
            ucb: None,
            thompson: None,
            contextual: None,
        }
    }

    pub fn reward_function(mut self, reward_function: String) -> Self {
        self.reward_function = reward_function;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn initialize(
        &mut self,
        epsilon: Option<f64>,
        confidence: Option<f64>,
        exploration: Option<f64>,
    ) -> Result<()> {
        let n_arms = self.strategy_names.len();

        match self.bandit_algorithm.as_str() {
            "epsilon_greedy" => {
                let eps = epsilon.unwrap_or(0.1);
                let mut eg = EpsilonGreedy::new(eps)?;
                if let Some(seed) = self.random_state {
                    eg = eg.random_state(seed);
                }
                eg.initialize(n_arms);
                self.epsilon_greedy = Some(eg);
            }
            "ucb" => {
                let conf = confidence.unwrap_or(2.0);
                let mut ucb = UpperConfidenceBound::new(conf)?;
                if let Some(seed) = self.random_state {
                    ucb = ucb.random_state(seed);
                }
                ucb.initialize(n_arms);
                self.ucb = Some(ucb);
            }
            "thompson_sampling" => {
                let mut ts = ThompsonSampling::new();
                if let Some(seed) = self.random_state {
                    ts = ts.random_state(seed);
                }
                ts.initialize(n_arms);
                self.thompson = Some(ts);
            }
            "contextual" => {
                let exp = exploration.unwrap_or(0.1);
                let mut cb = ContextualBandit::new(exp)?;
                if let Some(seed) = self.random_state {
                    cb = cb.random_state(seed);
                }
                // Context dimension will be set when first context is provided
                self.contextual = Some(cb);
            }
            _ => {
                return Err(BanditError::BanditComputationFailed(format!(
                    "Unknown bandit algorithm: {}",
                    self.bandit_algorithm
                ))
                .into())
            }
        }

        Ok(())
    }

    pub fn select_strategy(&mut self, context: Option<&ArrayView1<f64>>) -> Result<usize> {
        match self.bandit_algorithm.as_str() {
            "epsilon_greedy" => {
                if let Some(ref mut eg) = self.epsilon_greedy {
                    eg.select_arm()
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Epsilon-greedy not initialized".to_string(),
                    )
                    .into())
                }
            }
            "ucb" => {
                if let Some(ref mut ucb) = self.ucb {
                    ucb.select_arm()
                } else {
                    Err(
                        BanditError::BanditComputationFailed("UCB not initialized".to_string())
                            .into(),
                    )
                }
            }
            "thompson_sampling" => {
                if let Some(ref mut ts) = self.thompson {
                    ts.select_arm()
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Thompson sampling not initialized".to_string(),
                    )
                    .into())
                }
            }
            "contextual" => {
                if let Some(context) = context {
                    if let Some(ref mut cb) = self.contextual {
                        if cb.context_dim == 0 {
                            cb.initialize(self.strategy_names.len(), context.len());
                        }
                        cb.select_arm(context)
                    } else {
                        Err(BanditError::BanditComputationFailed(
                            "Contextual bandit not initialized".to_string(),
                        )
                        .into())
                    }
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Context required for contextual bandit".to_string(),
                    )
                    .into())
                }
            }
            _ => Err(BanditError::BanditComputationFailed(format!(
                "Unknown bandit algorithm: {}",
                self.bandit_algorithm
            ))
            .into()),
        }
    }

    pub fn update_strategy(
        &mut self,
        strategy_idx: usize,
        reward: f64,
        context: Option<&ArrayView1<f64>>,
    ) -> Result<()> {
        match self.bandit_algorithm.as_str() {
            "epsilon_greedy" => {
                if let Some(ref mut eg) = self.epsilon_greedy {
                    eg.update(strategy_idx, reward)
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Epsilon-greedy not initialized".to_string(),
                    )
                    .into())
                }
            }
            "ucb" => {
                if let Some(ref mut ucb) = self.ucb {
                    ucb.update(strategy_idx, reward)
                } else {
                    Err(
                        BanditError::BanditComputationFailed("UCB not initialized".to_string())
                            .into(),
                    )
                }
            }
            "thompson_sampling" => {
                if let Some(ref mut ts) = self.thompson {
                    ts.update(strategy_idx, reward)
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Thompson sampling not initialized".to_string(),
                    )
                    .into())
                }
            }
            "contextual" => {
                if let Some(context) = context {
                    if let Some(ref mut cb) = self.contextual {
                        cb.update(strategy_idx, context, reward)
                    } else {
                        Err(BanditError::BanditComputationFailed(
                            "Contextual bandit not initialized".to_string(),
                        )
                        .into())
                    }
                } else {
                    Err(BanditError::BanditComputationFailed(
                        "Context required for contextual bandit".to_string(),
                    )
                    .into())
                }
            }
            _ => Err(BanditError::BanditComputationFailed(format!(
                "Unknown bandit algorithm: {}",
                self.bandit_algorithm
            ))
            .into()),
        }
    }

    pub fn get_strategy_performance(&self) -> Result<HashMap<String, f64>> {
        let mut performance = HashMap::new();

        match self.bandit_algorithm.as_str() {
            "epsilon_greedy" => {
                if let Some(ref eg) = self.epsilon_greedy {
                    let stats = eg.get_arm_statistics();
                    for (idx, (_, avg_reward, _)) in stats.iter().enumerate() {
                        if idx < self.strategy_names.len() {
                            performance.insert(self.strategy_names[idx].clone(), *avg_reward);
                        }
                    }
                }
            }
            "ucb" => {
                if let Some(ref ucb) = self.ucb {
                    let stats = ucb.get_arm_statistics();
                    for (idx, (_, avg_reward, _, _)) in stats.iter().enumerate() {
                        if idx < self.strategy_names.len() {
                            performance.insert(self.strategy_names[idx].clone(), *avg_reward);
                        }
                    }
                }
            }
            "thompson_sampling" => {
                if let Some(ref ts) = self.thompson {
                    let stats = ts.get_arm_statistics();
                    for (idx, (_, _, mean, _)) in stats.iter().enumerate() {
                        if idx < self.strategy_names.len() {
                            performance.insert(self.strategy_names[idx].clone(), *mean);
                        }
                    }
                }
            }
            "contextual" => {
                // For contextual bandits, performance is context-dependent
                // Return uniform performance for now
                for name in self.strategy_names.iter() {
                    performance.insert(name.clone(), 0.5);
                }
            }
            _ => {
                return Err(BanditError::BanditComputationFailed(format!(
                    "Unknown bandit algorithm: {}",
                    self.bandit_algorithm
                ))
                .into())
            }
        }

        Ok(performance)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_epsilon_greedy_creation() {
        let eg = EpsilonGreedy::new(0.1).unwrap();
        assert_eq!(eg.epsilon, 0.1);
        assert_eq!(eg.decay_rate, 0.995);
        assert_eq!(eg.min_epsilon, 0.01);
    }

    #[test]
    fn test_epsilon_greedy_invalid_epsilon() {
        assert!(EpsilonGreedy::new(-0.1).is_err());
        assert!(EpsilonGreedy::new(1.5).is_err());
    }

    #[test]
    fn test_epsilon_greedy_basic_functionality() {
        let mut eg = EpsilonGreedy::new(0.5).unwrap().random_state(42);
        eg.initialize(3);

        // Select arms and update rewards
        for _ in 0..10 {
            let arm = eg.select_arm().unwrap();
            assert!(arm < 3);

            let reward = if arm == 0 { 1.0 } else { 0.0 }; // Arm 0 is best
            eg.update(arm, reward).unwrap();
        }

        let stats = eg.get_arm_statistics();
        assert_eq!(stats.len(), 3);

        // Check that arm 0 has highest average reward (it should be selected more often)
        if stats[0].0 > 0 {
            assert!(stats[0].1 >= stats[1].1 && stats[0].1 >= stats[2].1);
        }
    }

    #[test]
    fn test_upper_confidence_bound_creation() {
        let ucb = UpperConfidenceBound::new(2.0).unwrap();
        assert_eq!(ucb.confidence, 2.0);
    }

    #[test]
    fn test_upper_confidence_bound_invalid_confidence() {
        assert!(UpperConfidenceBound::new(0.0).is_err());
        assert!(UpperConfidenceBound::new(-1.0).is_err());
    }

    #[test]
    fn test_upper_confidence_bound_basic_functionality() {
        let mut ucb = UpperConfidenceBound::new(2.0).unwrap().random_state(42);
        ucb.initialize(3);

        // Select arms and update rewards
        for _ in 0..10 {
            let arm = ucb.select_arm().unwrap();
            assert!(arm < 3);

            let reward = if arm == 0 { 0.8 } else { 0.2 }; // Arm 0 is best
            ucb.update(arm, reward).unwrap();
        }

        let stats = ucb.get_arm_statistics();
        assert_eq!(stats.len(), 3);

        // All arms should have been tried at least once
        for (count, _, _, _) in stats.iter() {
            assert!(*count > 0);
        }
    }

    #[test]
    fn test_thompson_sampling_creation() {
        let ts = ThompsonSampling::new();
        assert!(ts.alpha_params.is_empty());
        assert!(ts.beta_params.is_empty());
    }

    #[test]
    fn test_thompson_sampling_basic_functionality() {
        let mut ts = ThompsonSampling::new().random_state(42);
        ts.initialize(3);

        // Select arms and update rewards
        for _ in 0..10 {
            let arm = ts.select_arm().unwrap();
            assert!(arm < 3);

            let reward = if arm == 0 { 0.9 } else { 0.1 }; // Arm 0 is best
            ts.update(arm, reward).unwrap();
        }

        let stats = ts.get_arm_statistics();
        assert_eq!(stats.len(), 3);

        // Check that parameters have been updated
        for (alpha, beta, mean, _) in stats.iter() {
            assert!(*alpha >= 1.0);
            assert!(*beta >= 1.0);
            assert!(*mean >= 0.0 && *mean <= 1.0);
        }
    }

    #[test]
    fn test_contextual_bandit_creation() {
        let cb = ContextualBandit::new(0.1).unwrap();
        assert_eq!(cb.exploration, 0.1);
        assert_eq!(cb.learning_rate, 0.1);
    }

    #[test]
    fn test_contextual_bandit_invalid_exploration() {
        assert!(ContextualBandit::new(-0.1).is_err());
    }

    #[test]
    fn test_contextual_bandit_basic_functionality() {
        let mut cb = ContextualBandit::new(0.1).unwrap().random_state(42);
        cb.initialize(2, 3);

        let context1 = array![1.0, 0.0, 0.0];
        let context2 = array![0.0, 1.0, 0.0];

        // Select arms and update rewards
        for i in 0..10 {
            let context = if i % 2 == 0 { &context1 } else { &context2 };
            let arm = cb.select_arm(&context.view()).unwrap();
            assert!(arm < 2);

            let reward = if (arm == 0 && i % 2 == 0) || (arm == 1 && i % 2 == 1) {
                1.0
            } else {
                0.0
            };
            cb.update(arm, &context.view(), reward).unwrap();
        }

        let weights = cb.get_arm_weights();
        assert_eq!(weights.len(), 2);
        assert_eq!(weights[0].len(), 3);
        assert_eq!(weights[1].len(), 3);

        // Test prediction
        let predicted = cb.predict_rewards(&context1.view()).unwrap();
        assert_eq!(predicted.len(), 2);
    }

    #[test]
    fn test_bandit_based_active_learning() {
        let strategies = vec![
            "entropy".to_string(),
            "margin".to_string(),
            "random".to_string(),
        ];
        let mut bbal =
            BanditBasedActiveLearning::new(strategies.clone(), "epsilon_greedy".to_string())
                .random_state(42);

        bbal.initialize(Some(0.2), None, None).unwrap();

        // Select strategies and update rewards
        for _ in 0..10 {
            let strategy_idx = bbal.select_strategy(None).unwrap();
            assert!(strategy_idx < strategies.len());

            let reward = if strategy_idx == 0 { 0.8 } else { 0.3 }; // Entropy is best
            bbal.update_strategy(strategy_idx, reward, None).unwrap();
        }

        let performance = bbal.get_strategy_performance().unwrap();
        assert_eq!(performance.len(), strategies.len());

        for strategy in strategies.iter() {
            assert!(performance.contains_key(strategy));
        }
    }

    #[test]
    fn test_bandit_based_active_learning_ucb() {
        let strategies = vec!["uncertainty".to_string(), "diversity".to_string()];
        let mut bbal =
            BanditBasedActiveLearning::new(strategies.clone(), "ucb".to_string()).random_state(42);

        bbal.initialize(None, Some(1.5), None).unwrap();

        // Test basic functionality
        let strategy_idx = bbal.select_strategy(None).unwrap();
        assert!(strategy_idx < strategies.len());

        bbal.update_strategy(strategy_idx, 0.5, None).unwrap();

        let performance = bbal.get_strategy_performance().unwrap();
        assert_eq!(performance.len(), strategies.len());
    }

    #[test]
    fn test_bandit_based_active_learning_thompson() {
        let strategies = vec!["query1".to_string(), "query2".to_string()];
        let mut bbal =
            BanditBasedActiveLearning::new(strategies.clone(), "thompson_sampling".to_string())
                .random_state(42);

        bbal.initialize(None, None, None).unwrap();

        // Test basic functionality
        let strategy_idx = bbal.select_strategy(None).unwrap();
        assert!(strategy_idx < strategies.len());

        bbal.update_strategy(strategy_idx, 0.7, None).unwrap();

        let performance = bbal.get_strategy_performance().unwrap();
        assert_eq!(performance.len(), strategies.len());
    }

    #[test]
    fn test_bandit_based_active_learning_contextual() {
        let strategies = vec![
            "context_strategy1".to_string(),
            "context_strategy2".to_string(),
        ];
        let mut bbal = BanditBasedActiveLearning::new(strategies.clone(), "contextual".to_string())
            .random_state(42);

        bbal.initialize(None, None, Some(0.15)).unwrap();

        let context = array![0.5, 1.0, 0.2];

        // Test basic functionality
        let strategy_idx = bbal.select_strategy(Some(&context.view())).unwrap();
        assert!(strategy_idx < strategies.len());

        bbal.update_strategy(strategy_idx, 0.6, Some(&context.view()))
            .unwrap();

        let performance = bbal.get_strategy_performance().unwrap();
        assert_eq!(performance.len(), strategies.len());
    }
}

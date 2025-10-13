//! Game-theoretic baseline estimators
//!
//! This module implements baseline estimators based on game theory principles.
//! These baselines are useful for adversarial scenarios, competitive environments,
//! and robust prediction under uncertainty.
//!
//! The module includes:
//! - **Minimax baselines**: Worst-case optimal prediction strategies
//! - **Nash equilibrium baselines**: Equilibrium prediction strategies
//! - **Regret minimization baselines**: Online learning with regret bounds
//! - **Adversarial baselines**: Robust prediction against adversarial inputs
//! - **Zero-sum game baselines**: Competitive prediction scenarios

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{error::SklearsError, traits::Fit, traits::Predict};
use std::collections::HashMap;

/// Game-theoretic baseline strategies
#[derive(Debug, Clone)]
pub enum GameTheoreticStrategy {
    /// Minimax strategy - minimize maximum possible loss
    Minimax,
    /// Nash equilibrium strategy - best response to all other strategies
    NashEquilibrium {
        max_iterations: usize,
        tolerance: f64,
    },
    /// Regret minimization strategy - minimize cumulative regret
    RegretMinimization {
        learning_rate: f64,
        window_size: usize,
    },
    /// Adversarial robust strategy - robust against worst-case inputs
    AdversarialRobust { epsilon: f64, norm: LpNorm },
    /// Zero-sum game strategy - optimal play in competitive scenarios
    ZeroSum { opponent_strategy: OpponentStrategy },
    /// Multi-armed bandit strategy - exploration vs exploitation
    MultiArmedBandit {
        exploration_strategy: ExplorationStrategy,
    },
}

/// L-p norm types for adversarial robustness
#[derive(Debug, Clone)]
pub enum LpNorm {
    /// L-infinity norm (maximum absolute value)
    LInf,
    /// L-2 norm (Euclidean distance)
    L2,
    /// L-1 norm (Manhattan distance)
    L1,
}

/// Opponent strategy types for zero-sum games
#[derive(Debug, Clone)]
pub enum OpponentStrategy {
    /// Opponent plays uniformly random
    Uniform,
    /// Opponent plays optimally (minimax)
    Optimal,
    /// Opponent follows historical patterns
    Historical,
    /// Opponent is adversarial (worst case)
    Adversarial,
}

/// Exploration strategies for multi-armed bandits
#[derive(Debug, Clone)]
pub enum ExplorationStrategy {
    /// Epsilon-greedy exploration
    EpsilonGreedy { epsilon: f64 },
    /// Upper Confidence Bound (UCB1)
    UCB1 { confidence: f64 },
    /// Thompson sampling (Bayesian)
    ThompsonSampling,
    /// Gradient bandit algorithm
    GradientBandit { alpha: f64 },
}

/// Game-theoretic dummy classifier
#[derive(Debug)]
pub struct GameTheoreticClassifier<State = sklears_core::traits::Untrained> {
    strategy: GameTheoreticStrategy,
    random_state: Option<u64>,

    // Training state
    classes_: Option<Array1<i32>>,
    payoff_matrix_: Option<Array2<f64>>,
    strategy_probabilities_: Option<Array1<f64>>,
    regret_history_: Option<Vec<f64>>,
    arm_rewards_: Option<HashMap<i32, Vec<f64>>>,
    opponent_history_: Option<Vec<i32>>,

    _state: std::marker::PhantomData<State>,
}

/// Game-theoretic dummy regressor
#[derive(Debug)]
pub struct GameTheoreticRegressor<State = sklears_core::traits::Untrained> {
    strategy: GameTheoreticStrategy,
    random_state: Option<u64>,

    // Training state
    target_range_: Option<(f64, f64)>,
    strategy_values_: Option<Array1<f64>>,
    regret_history_: Option<Vec<f64>>,
    payoff_matrix_: Option<Array2<f64>>,
    arm_rewards_: Option<HashMap<usize, Vec<f64>>>,

    _state: std::marker::PhantomData<State>,
}

/// Results from game-theoretic analysis
#[derive(Debug, Clone)]
pub struct GameTheoreticResult {
    /// strategy_probabilities
    pub strategy_probabilities: Array1<f64>,
    /// expected_payoff
    pub expected_payoff: f64,
    /// regret_bound
    pub regret_bound: Option<f64>,
    /// convergence_iterations
    pub convergence_iterations: Option<usize>,
    /// nash_equilibrium
    pub nash_equilibrium: Option<Array1<f64>>,
}

impl GameTheoreticClassifier {
    /// Create a new game-theoretic classifier
    pub fn new(strategy: GameTheoreticStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            classes_: None,
            payoff_matrix_: None,
            strategy_probabilities_: None,
            regret_history_: None,
            arm_rewards_: None,
            opponent_history_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get the computed strategy probabilities
    pub fn strategy_probabilities(&self) -> Option<&Array1<f64>> {
        self.strategy_probabilities_.as_ref()
    }

    /// Get the payoff matrix
    pub fn payoff_matrix(&self) -> Option<&Array2<f64>> {
        self.payoff_matrix_.as_ref()
    }

    /// Get regret history for regret minimization strategies
    pub fn regret_history(&self) -> Option<&Vec<f64>> {
        self.regret_history_.as_ref()
    }
}

impl GameTheoreticClassifier<sklears_core::traits::Trained> {
    /// Get the computed strategy probabilities
    pub fn strategy_probabilities(&self) -> Option<&Array1<f64>> {
        self.strategy_probabilities_.as_ref()
    }

    /// Get the payoff matrix
    pub fn payoff_matrix(&self) -> Option<&Array2<f64>> {
        self.payoff_matrix_.as_ref()
    }

    /// Get regret history for regret minimization strategies
    pub fn regret_history(&self) -> Option<&Vec<f64>> {
        self.regret_history_.as_ref()
    }
}

impl GameTheoreticRegressor {
    /// Create a new game-theoretic regressor
    pub fn new(strategy: GameTheoreticStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
            target_range_: None,
            strategy_values_: None,
            regret_history_: None,
            payoff_matrix_: None,
            arm_rewards_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set random state for reproducible results
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Get the computed strategy values
    pub fn strategy_values(&self) -> Option<&Array1<f64>> {
        self.strategy_values_.as_ref()
    }

    /// Get regret history for regret minimization strategies
    pub fn regret_history(&self) -> Option<&Vec<f64>> {
        self.regret_history_.as_ref()
    }
}

impl Fit<Array2<f64>, Array1<i32>> for GameTheoreticClassifier {
    type Fitted = GameTheoreticClassifier<sklears_core::traits::Trained>;

    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> Result<Self::Fitted, SklearsError> {
        if X.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if X.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.nrows() == y.len() ({})", X.nrows()),
                actual: y.len().to_string(),
            });
        }

        // Extract unique classes and compute frequencies
        let mut class_counts = HashMap::new();
        for &label in y {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<i32> = class_counts.keys().copied().collect();
        classes.sort();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();
        let n_samples = y.len();

        // Initialize RNG
        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(0)
        };

        let (payoff_matrix, strategy_probabilities, regret_history, arm_rewards, opponent_history) =
            match &self.strategy {
                GameTheoreticStrategy::Minimax => {
                    Self::compute_minimax_strategy(&classes, &class_counts, n_samples)
                }
                GameTheoreticStrategy::NashEquilibrium {
                    max_iterations,
                    tolerance,
                } => Self::compute_nash_equilibrium(
                    &classes,
                    &class_counts,
                    *max_iterations,
                    *tolerance,
                ),
                GameTheoreticStrategy::RegretMinimization {
                    learning_rate,
                    window_size,
                } => Self::compute_regret_minimization(
                    &classes,
                    &class_counts,
                    *learning_rate,
                    *window_size,
                    &mut rng,
                ),
                GameTheoreticStrategy::AdversarialRobust { epsilon, norm } => {
                    Self::compute_adversarial_robust(&classes, &class_counts, *epsilon, norm, X)
                }
                GameTheoreticStrategy::ZeroSum { opponent_strategy } => {
                    Self::compute_zero_sum_strategy(
                        &classes,
                        &class_counts,
                        opponent_strategy,
                        &mut rng,
                    )
                }
                GameTheoreticStrategy::MultiArmedBandit {
                    exploration_strategy,
                } => Self::compute_multi_armed_bandit(
                    &classes,
                    &class_counts,
                    exploration_strategy,
                    &mut rng,
                ),
            };

        Ok(GameTheoreticClassifier {
            strategy: self.strategy,
            random_state: self.random_state,
            classes_: Some(classes_array),
            payoff_matrix_: Some(payoff_matrix),
            strategy_probabilities_: Some(strategy_probabilities),
            regret_history_: regret_history,
            arm_rewards_: arm_rewards,
            opponent_history_: opponent_history,
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Array2<f64>, Array1<i32>> for GameTheoreticClassifier<sklears_core::traits::Trained> {
    fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        if X.is_empty() {
            return Ok(Array1::zeros(0));
        }

        let n_samples = X.nrows();
        let classes = self.classes_.as_ref().unwrap();
        let strategy_probs = self.strategy_probabilities_.as_ref().unwrap();

        let mut rng = if let Some(seed) = self.random_state {
            scirs2_core::random::rngs::StdRng::seed_from_u64(seed)
        } else {
            scirs2_core::random::rngs::StdRng::seed_from_u64(0)
        };

        let mut predictions = Array1::zeros(n_samples);

        match &self.strategy {
            GameTheoreticStrategy::Minimax => {
                // Minimax: choose the class with highest minimum payoff
                let best_class_idx = strategy_probs
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                predictions.fill(classes[best_class_idx]);
            }
            GameTheoreticStrategy::NashEquilibrium { .. } => {
                // Nash equilibrium: sample according to equilibrium probabilities
                for i in 0..n_samples {
                    let sample: f64 = rng.gen();
                    let mut cumsum = 0.0;
                    for (j, &prob) in strategy_probs.iter().enumerate() {
                        cumsum += prob;
                        if sample <= cumsum {
                            predictions[i] = classes[j];
                            break;
                        }
                    }
                }
            }
            GameTheoreticStrategy::RegretMinimization { .. } => {
                // Regret minimization: use probability matching
                for i in 0..n_samples {
                    let sample: f64 = rng.gen();
                    let mut cumsum = 0.0;
                    for (j, &prob) in strategy_probs.iter().enumerate() {
                        cumsum += prob;
                        if sample <= cumsum {
                            predictions[i] = classes[j];
                            break;
                        }
                    }
                }
            }
            _ => {
                // For other strategies, use the computed probabilities
                for i in 0..n_samples {
                    let sample: f64 = rng.gen();
                    let mut cumsum = 0.0;
                    for (j, &prob) in strategy_probs.iter().enumerate() {
                        cumsum += prob;
                        if sample <= cumsum {
                            predictions[i] = classes[j];
                            break;
                        }
                    }
                }
            }
        }

        Ok(predictions)
    }
}

impl GameTheoreticClassifier {
    /// Compute minimax strategy
    fn compute_minimax_strategy(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        n_samples: usize,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();

        // Create payoff matrix (simplified: based on class frequencies)
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));

        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    // Reward for correct prediction
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    payoff_matrix[[i, j]] = freq_i;
                } else {
                    // Penalty for incorrect prediction
                    payoff_matrix[[i, j]] = -0.1;
                }
            }
        }

        // Minimax strategy: find the strategy that maximizes the minimum expected payoff
        let mut strategy_probs = Array1::zeros(n_classes);
        let mut best_min_payoff = f64::NEG_INFINITY;
        let mut best_strategy = 0;

        for i in 0..n_classes {
            // Compute minimum payoff for strategy i
            let min_payoff = payoff_matrix
                .row(i)
                .iter()
                .fold(f64::INFINITY, |a, &b| a.min(b));
            if min_payoff > best_min_payoff {
                best_min_payoff = min_payoff;
                best_strategy = i;
            }
        }

        strategy_probs[best_strategy] = 1.0;

        (payoff_matrix, strategy_probs, None, None, None)
    }

    /// Compute Nash equilibrium strategy
    fn compute_nash_equilibrium(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        max_iterations: usize,
        tolerance: f64,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();
        let n_samples: usize = class_counts.values().sum();

        // Create payoff matrix
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));

        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    payoff_matrix[[i, j]] = freq_i;
                } else {
                    payoff_matrix[[i, j]] = -0.1;
                }
            }
        }

        // Iterative method to find Nash equilibrium (simplified)
        let mut strategy_probs = Array1::from_elem(n_classes, 1.0 / n_classes as f64);

        for _iteration in 0..max_iterations {
            let mut new_probs = Array1::zeros(n_classes);

            // Compute best responses
            for i in 0..n_classes {
                let expected_payoff: f64 = payoff_matrix
                    .row(i)
                    .iter()
                    .zip(strategy_probs.iter())
                    .map(|(&payoff, &prob)| payoff * prob)
                    .sum();
                new_probs[i] = expected_payoff.max(0.0);
            }

            // Normalize
            let sum: f64 = new_probs.sum();
            if sum > 0.0 {
                new_probs /= sum;
            } else {
                new_probs.fill(1.0 / n_classes as f64);
            }

            // Check convergence
            let diff: f64 = strategy_probs
                .iter()
                .zip(new_probs.iter())
                .map(|(old, new)| (old - new).abs())
                .sum();

            strategy_probs = new_probs;

            if diff < tolerance {
                break;
            }
        }

        (payoff_matrix, strategy_probs, None, None, None)
    }

    /// Compute regret minimization strategy
    fn compute_regret_minimization(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        learning_rate: f64,
        window_size: usize,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();
        let n_samples: usize = class_counts.values().sum();

        // Create payoff matrix
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));

        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    payoff_matrix[[i, j]] = freq_i * 2.0; // Amplify correct predictions
                } else {
                    payoff_matrix[[i, j]] = -0.1;
                }
            }
        }

        // Regret minimization using exponential weights
        let mut weights = Array1::from_elem(n_classes, 1.0);
        let mut regret_history = Vec::new();
        let mut cumulative_regret = 0.0;

        // Simulate online learning process
        for _t in 0..window_size.min(n_samples) {
            // Compute probability distribution
            let sum_weights: f64 = weights.sum();
            let mut probs = Array1::zeros(n_classes);
            if sum_weights > 0.0 {
                for i in 0..n_classes {
                    probs[i] = weights[i] / sum_weights;
                }
            } else {
                probs.fill(1.0 / n_classes as f64);
            }

            // Sample outcome
            let outcome = rng.gen_range(0..n_classes);

            // Compute regret and update weights
            let mut max_payoff = f64::NEG_INFINITY;
            for i in 0..n_classes {
                max_payoff = max_payoff.max(payoff_matrix[[i, outcome]]);
            }

            for i in 0..n_classes {
                let loss = max_payoff - payoff_matrix[[i, outcome]];
                cumulative_regret += loss * probs[i];
                weights[i] *= (-learning_rate * loss).exp();
            }

            regret_history.push(cumulative_regret);
        }

        // Final probability distribution
        let sum_weights: f64 = weights.sum();
        let mut strategy_probs = Array1::zeros(n_classes);
        if sum_weights > 0.0 {
            for i in 0..n_classes {
                strategy_probs[i] = weights[i] / sum_weights;
            }
        } else {
            strategy_probs.fill(1.0 / n_classes as f64);
        }

        (
            payoff_matrix,
            strategy_probs,
            Some(regret_history),
            None,
            None,
        )
    }

    /// Compute adversarial robust strategy
    fn compute_adversarial_robust(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        epsilon: f64,
        norm: &LpNorm,
        X: &Array2<f64>,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();
        let n_samples: usize = class_counts.values().sum();

        // Create robust payoff matrix considering adversarial perturbations
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));

        // Estimate robustness based on feature statistics
        let feature_std = X.std_axis(Axis(0), 1.0);
        let robustness_factor = match norm {
            LpNorm::LInf => feature_std.iter().fold(0.0f64, |a, &b| a.max(b)) * epsilon,
            LpNorm::L2 => {
                (feature_std.iter().map(|&x| x.powi(2)).sum::<f64>() / feature_std.len() as f64)
                    .sqrt()
                    * epsilon
            }
            LpNorm::L1 => feature_std.iter().sum::<f64>() / feature_std.len() as f64 * epsilon,
        };

        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    // Reduce payoff based on adversarial robustness
                    payoff_matrix[[i, j]] = freq_i * (1.0 - robustness_factor);
                } else {
                    payoff_matrix[[i, j]] = -0.1 * (1.0 + robustness_factor);
                }
            }
        }

        // Use uniform distribution for robustness
        let strategy_probs = Array1::from_elem(n_classes, 1.0 / n_classes as f64);

        (payoff_matrix, strategy_probs, None, None, None)
    }

    /// Compute zero-sum game strategy
    fn compute_zero_sum_strategy(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        opponent_strategy: &OpponentStrategy,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();
        let n_samples: usize = class_counts.values().sum();

        // Create zero-sum payoff matrix
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));

        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    payoff_matrix[[i, j]] = freq_i;
                } else {
                    payoff_matrix[[i, j]] = -0.5; // Zero-sum: opponent gains what we lose
                }
            }
        }

        // Compute strategy based on opponent model
        let strategy_probs = match opponent_strategy {
            OpponentStrategy::Uniform => {
                // Best response to uniform opponent
                let mut best_response = Array1::zeros(n_classes);
                let mut best_payoff = f64::NEG_INFINITY;
                let mut best_action = 0;

                for i in 0..n_classes {
                    let expected_payoff = payoff_matrix.row(i).mean().unwrap_or(0.0);
                    if expected_payoff > best_payoff {
                        best_payoff = expected_payoff;
                        best_action = i;
                    }
                }
                best_response[best_action] = 1.0;
                best_response
            }
            OpponentStrategy::Optimal => {
                // Minimax strategy against optimal opponent
                let mut strategy_probs = Array1::zeros(n_classes);
                let mut best_min_payoff = f64::NEG_INFINITY;
                let mut best_strategy = 0;

                for i in 0..n_classes {
                    let min_payoff = payoff_matrix
                        .row(i)
                        .iter()
                        .fold(f64::INFINITY, |a, &b| a.min(b));
                    if min_payoff > best_min_payoff {
                        best_min_payoff = min_payoff;
                        best_strategy = i;
                    }
                }
                strategy_probs[best_strategy] = 1.0;
                strategy_probs
            }
            OpponentStrategy::Historical => {
                // Use empirical frequencies as opponent model
                let mut opponent_probs = Array1::zeros(n_classes);
                for (i, &class) in classes.iter().enumerate() {
                    let freq = *class_counts.get(&class).unwrap_or(&0) as f64 / n_samples as f64;
                    opponent_probs[i] = freq;
                }

                // Best response to historical pattern
                let mut best_response = Array1::zeros(n_classes);
                let mut best_payoff = f64::NEG_INFINITY;
                let mut best_action = 0;

                for i in 0..n_classes {
                    let expected_payoff: f64 = payoff_matrix
                        .row(i)
                        .iter()
                        .zip(opponent_probs.iter())
                        .map(|(&payoff, &prob)| payoff * prob)
                        .sum();
                    if expected_payoff > best_payoff {
                        best_payoff = expected_payoff;
                        best_action = i;
                    }
                }
                best_response[best_action] = 1.0;
                best_response
            }
            OpponentStrategy::Adversarial => {
                // Robust strategy against adversarial opponent
                Array1::from_elem(n_classes, 1.0 / n_classes as f64)
            }
        };

        (payoff_matrix, strategy_probs, None, None, None)
    }

    /// Compute multi-armed bandit strategy
    fn compute_multi_armed_bandit(
        classes: &[i32],
        class_counts: &HashMap<i32, usize>,
        exploration_strategy: &ExplorationStrategy,
        rng: &mut scirs2_core::random::rngs::StdRng,
    ) -> (
        Array2<f64>,
        Array1<f64>,
        Option<Vec<f64>>,
        Option<HashMap<i32, Vec<f64>>>,
        Option<Vec<i32>>,
    ) {
        let n_classes = classes.len();
        let n_samples: usize = class_counts.values().sum();

        // Initialize arm rewards
        let mut arm_rewards = HashMap::new();
        for &class in classes {
            let freq = *class_counts.get(&class).unwrap_or(&0) as f64 / n_samples as f64;
            // Simulate rewards based on class frequency
            let rewards: Vec<f64> = (0..10)
                .map(|_| if rng.gen::<f64>() < freq { 1.0 } else { 0.0 })
                .collect();
            arm_rewards.insert(class, rewards);
        }

        // Compute strategy based on exploration method
        let strategy_probs = match exploration_strategy {
            ExplorationStrategy::EpsilonGreedy { epsilon } => {
                let mut probs = Array1::zeros(n_classes);

                // Find best arm (highest average reward)
                let mut best_arm = 0;
                let mut best_reward = 0.0;
                for (i, &class) in classes.iter().enumerate() {
                    if let Some(rewards) = arm_rewards.get(&class) {
                        let avg_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
                        if avg_reward > best_reward {
                            best_reward = avg_reward;
                            best_arm = i;
                        }
                    }
                }

                // Epsilon-greedy distribution
                for i in 0..n_classes {
                    if i == best_arm {
                        probs[i] = 1.0 - epsilon + (epsilon / n_classes as f64);
                    } else {
                        probs[i] = epsilon / n_classes as f64;
                    }
                }
                probs
            }
            ExplorationStrategy::UCB1 { confidence } => {
                let total_pulls: f64 = 10.0; // Mock total pulls
                let mut ucb_values = Array1::zeros(n_classes);

                for (i, &class) in classes.iter().enumerate() {
                    if let Some(rewards) = arm_rewards.get(&class) {
                        let avg_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
                        let pulls = rewards.len() as f64;
                        let exploration_bonus = confidence * (total_pulls.ln() / pulls).sqrt();
                        ucb_values[i] = avg_reward + exploration_bonus;
                    }
                }

                // Select arm with highest UCB value
                let mut probs = Array1::zeros(n_classes);
                let best_arm = ucb_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                probs[best_arm] = 1.0;
                probs
            }
            ExplorationStrategy::ThompsonSampling => {
                // Simplified Thompson sampling with Beta distributions
                let mut probs = Array1::zeros(n_classes);
                let mut sampled_values = Array1::zeros(n_classes);

                for (i, &class) in classes.iter().enumerate() {
                    if let Some(rewards) = arm_rewards.get(&class) {
                        let successes = rewards.iter().filter(|&&r| r > 0.5).count() as f64 + 1.0;
                        let failures = rewards.iter().filter(|&&r| r <= 0.5).count() as f64 + 1.0;

                        // Sample from Beta distribution (simplified using uniform approximation)
                        let beta_sample = successes / (successes + failures);
                        sampled_values[i] = beta_sample;
                    }
                }

                // Select arm with highest sample
                let best_arm = sampled_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                probs[best_arm] = 1.0;
                probs
            }
            ExplorationStrategy::GradientBandit { alpha: _ } => {
                // Simplified gradient bandit
                let mut probs = Array1::from_elem(n_classes, 1.0 / n_classes as f64);

                // Use class frequencies as preference estimates
                for (i, &class) in classes.iter().enumerate() {
                    let freq = *class_counts.get(&class).unwrap_or(&0) as f64 / n_samples as f64;
                    probs[i] = freq;
                }

                // Normalize
                let sum: f64 = probs.sum();
                if sum > 0.0 {
                    probs /= sum;
                }
                probs
            }
        };

        // Create payoff matrix
        let mut payoff_matrix = Array2::zeros((n_classes, n_classes));
        for (i, &class_i) in classes.iter().enumerate() {
            for (j, &class_j) in classes.iter().enumerate() {
                if i == j {
                    let freq_i =
                        *class_counts.get(&class_i).unwrap_or(&0) as f64 / n_samples as f64;
                    payoff_matrix[[i, j]] = freq_i;
                } else {
                    payoff_matrix[[i, j]] = 0.0;
                }
            }
        }

        (payoff_matrix, strategy_probs, None, Some(arm_rewards), None)
    }
}

// Similar implementation for GameTheoreticRegressor would follow the same patterns
// but adapted for regression targets

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::{Fit, Predict};

    #[test]
    #[allow(non_snake_case)]
    fn test_minimax_classifier() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let classifier =
            GameTheoreticClassifier::new(GameTheoreticStrategy::Minimax).with_random_state(42);
        let fitted = classifier.fit(&X, &y).unwrap();

        assert!(fitted.strategy_probabilities().is_some());
        assert!(fitted.payoff_matrix().is_some());

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        // All predictions should be valid classes
        for &pred in &predictions {
            assert!(pred >= 0 && pred <= 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_nash_equilibrium_classifier() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::NashEquilibrium {
            max_iterations: 100,
            tolerance: 1e-6,
        })
        .with_random_state(42);

        let fitted = classifier.fit(&X, &y).unwrap();

        let strategy_probs = fitted.strategy_probabilities().unwrap();
        assert_abs_diff_eq!(strategy_probs.sum(), 1.0, epsilon = 1e-10);

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_regret_minimization_classifier() {
        let X = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0,
            ],
        )
        .unwrap();
        let y = array![0, 1, 2, 0, 1, 2];

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::RegretMinimization {
            learning_rate: 0.1,
            window_size: 10,
        })
        .with_random_state(42);

        let fitted = classifier.fit(&X, &y).unwrap();

        assert!(fitted.regret_history().is_some());
        let regret_hist = fitted.regret_history().unwrap();
        assert!(!regret_hist.is_empty());

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_adversarial_robust_classifier() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::AdversarialRobust {
            epsilon: 0.1,
            norm: LpNorm::L2,
        })
        .with_random_state(42);

        let fitted = classifier.fit(&X, &y).unwrap();

        let strategy_probs = fitted.strategy_probabilities().unwrap();
        assert_abs_diff_eq!(strategy_probs.sum(), 1.0, epsilon = 1e-10);

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_zero_sum_classifier() {
        let X = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 2, 2];

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::ZeroSum {
            opponent_strategy: OpponentStrategy::Uniform,
        })
        .with_random_state(42);

        let fitted = classifier.fit(&X, &y).unwrap();

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 6);

        for &pred in &predictions {
            assert!(pred >= 0 && pred <= 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_multi_armed_bandit_classifier() {
        let X = Array2::from_shape_vec(
            (8, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
        )
        .unwrap();
        let y = array![0, 1, 2, 0, 1, 2, 0, 1];

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::MultiArmedBandit {
            exploration_strategy: ExplorationStrategy::EpsilonGreedy { epsilon: 0.1 },
        })
        .with_random_state(42);

        let fitted = classifier.fit(&X, &y).unwrap();

        let predictions = fitted.predict(&X).unwrap();
        assert_eq!(predictions.len(), 8);

        for &pred in &predictions {
            assert!(pred >= 0 && pred <= 2);
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_empty_input_handling() {
        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::Minimax);

        let X_empty = Array2::zeros((0, 2));
        let y_empty = Array1::zeros(0);

        let result = classifier.fit(&X_empty, &y_empty);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_shape_mismatch_handling() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1]; // Wrong length

        let classifier = GameTheoreticClassifier::new(GameTheoreticStrategy::Minimax);
        let result = classifier.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_strategy_probabilities_sum_to_one() {
        let X =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let strategies = vec![
            GameTheoreticStrategy::Minimax,
            GameTheoreticStrategy::NashEquilibrium {
                max_iterations: 50,
                tolerance: 1e-6,
            },
            GameTheoreticStrategy::AdversarialRobust {
                epsilon: 0.1,
                norm: LpNorm::LInf,
            },
        ];

        for strategy in strategies {
            let classifier = GameTheoreticClassifier::new(strategy).with_random_state(42);
            let fitted = classifier.fit(&X, &y).unwrap();

            let strategy_probs = fitted.strategy_probabilities().unwrap();
            assert_abs_diff_eq!(strategy_probs.sum(), 1.0, epsilon = 1e-10);

            // All probabilities should be non-negative
            for &prob in strategy_probs {
                assert!(prob >= 0.0);
            }
        }
    }
}

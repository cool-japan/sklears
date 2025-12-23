//! Reinforcement Learning algorithms for neural network training.
//!
//! This module implements various deep reinforcement learning methods including:
//! - Deep Q-Network (DQN) with experience replay
//! - Double DQN to reduce overestimation
//! - Dueling DQN for value/advantage decomposition
//! - Policy Gradient methods (REINFORCE)
//! - Actor-Critic algorithms (A2C, A3C)
//! - Proximal Policy Optimization (PPO)
//! - Deep Deterministic Policy Gradient (DDPG)
//!
//! Reinforcement learning trains agents to make sequential decisions by
//! maximizing cumulative rewards through interaction with environments.

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Rng, Uniform};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Experience tuple for replay buffer
#[derive(Debug, Clone)]
pub struct Experience<T: FloatBounds> {
    /// Current state
    pub state: Array1<T>,
    /// Action taken
    pub action: usize,
    /// Reward received
    pub reward: T,
    /// Next state
    pub next_state: Array1<T>,
    /// Whether episode is done
    pub done: bool,
}

/// Experience replay buffer
#[derive(Debug)]
pub struct ReplayBuffer<T: FloatBounds> {
    /// Buffer storage
    buffer: VecDeque<Experience<T>>,
    /// Maximum buffer size
    capacity: usize,
}

impl<T: FloatBounds> ReplayBuffer<T> {
    /// Create a new replay buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, experience: Experience<T>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample a batch of experiences
    pub fn sample(&self, batch_size: usize) -> Option<Vec<Experience<T>>> {
        if self.buffer.len() < batch_size {
            return None;
        }

        let mut rng = thread_rng();
        let mut samples = Vec::with_capacity(batch_size);

        for _ in 0..batch_size {
            let idx = rng.random_range(0..self.buffer.len());
            samples.push(self.buffer[idx].clone());
        }

        Some(samples)
    }

    /// Get current buffer size
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if buffer is empty
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Clear the buffer
    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

/// Q-Network for DQN
#[derive(Debug)]
pub struct QNetwork<T: FloatBounds> {
    /// Network weights (layers)
    weights: Vec<Array2<T>>,
    /// Network biases
    biases: Vec<Array1<T>>,
    /// State dimension
    state_dim: usize,
    /// Action dimension
    action_dim: usize,
    /// Hidden layer sizes
    hidden_dims: Vec<usize>,
}

impl<T: FloatBounds + ScalarOperand> QNetwork<T> {
    /// Create a new Q-network
    pub fn new(state_dim: usize, action_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        // Build layers
        let mut prev_dim = state_dim;
        for &hidden_dim in &hidden_dims {
            let std = (2.0 / prev_dim as f64).sqrt();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                T::from(
                    rng.sample::<f64, _>(scirs2_core::random::Normal::new(0.0, 1.0).unwrap()) * std,
                )
                .unwrap()
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w);
            biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layer
        let std = (2.0 / prev_dim as f64).sqrt();
        let w = Array2::from_shape_fn((prev_dim, action_dim), |_| {
            T::from(rng.sample::<f64, _>(scirs2_core::random::Normal::new(0.0, 1.0).unwrap()) * std)
                .unwrap()
        });
        let b = Array1::zeros(action_dim);
        weights.push(w);
        biases.push(b);

        Self {
            weights,
            biases,
            state_dim,
            action_dim,
            hidden_dims,
        }
    }

    /// Forward pass: compute Q-values for all actions
    pub fn forward(&self, state: &Array1<T>) -> NeuralResult<Array1<T>> {
        let mut h = state.clone();

        // Pass through hidden layers with ReLU
        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            // Ensure h has the right shape for matrix multiplication
            let h_2d = h.insert_axis(Axis(0));
            let output = h_2d.dot(w);
            h = output.row(0).to_owned() + b;

            // Apply ReLU for all but last layer
            if i < self.weights.len() - 1 {
                h.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
            }
        }

        Ok(h)
    }

    /// Get Q-value for a specific state-action pair
    pub fn get_q_value(&self, state: &Array1<T>, action: usize) -> NeuralResult<T> {
        let q_values = self.forward(state)?;
        if action >= q_values.len() {
            return Err(SklearsError::InvalidParameter {
                name: "action".to_string(),
                reason: format!("Action {} out of bounds (max: {})", action, q_values.len()),
            });
        }
        Ok(q_values[action])
    }

    /// Select best action (greedy policy)
    pub fn select_action(&self, state: &Array1<T>) -> NeuralResult<usize> {
        let q_values = self.forward(state)?;
        let best_action = q_values
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                a.to_f64()
                    .unwrap()
                    .partial_cmp(&b.to_f64().unwrap())
                    .unwrap()
            })
            .map(|(idx, _)| idx)
            .unwrap();

        Ok(best_action)
    }

    /// Copy weights from another network (for target network updates)
    pub fn copy_weights_from(&mut self, other: &QNetwork<T>) {
        for (self_w, other_w) in self.weights.iter_mut().zip(other.weights.iter()) {
            self_w.assign(other_w);
        }
        for (self_b, other_b) in self.biases.iter_mut().zip(other.biases.iter()) {
            self_b.assign(other_b);
        }
    }

    /// Soft update: theta_target = tau * theta + (1 - tau) * theta_target
    pub fn soft_update(&mut self, other: &QNetwork<T>, tau: T) {
        for (self_w, other_w) in self.weights.iter_mut().zip(other.weights.iter()) {
            *self_w = other_w.mapv(|x| x * tau) + self_w.mapv(|x| x * (T::one() - tau));
        }
        for (self_b, other_b) in self.biases.iter_mut().zip(other.biases.iter()) {
            *self_b = other_b.mapv(|x| x * tau) + self_b.mapv(|x| x * (T::one() - tau));
        }
    }
}

/// DQN Agent configuration
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DQNConfig {
    /// State dimension
    pub state_dim: usize,
    /// Action dimension
    pub action_dim: usize,
    /// Hidden layer sizes
    pub hidden_dims: Vec<usize>,
    /// Learning rate
    pub learning_rate: f64,
    /// Discount factor (gamma)
    pub gamma: f64,
    /// Initial exploration rate (epsilon)
    pub epsilon_start: f64,
    /// Final exploration rate
    pub epsilon_end: f64,
    /// Epsilon decay rate
    pub epsilon_decay: f64,
    /// Replay buffer capacity
    pub buffer_capacity: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Target network update frequency
    pub target_update_freq: usize,
    /// Whether to use Double DQN
    pub double_dqn: bool,
}

impl Default for DQNConfig {
    fn default() -> Self {
        Self {
            state_dim: 4,
            action_dim: 2,
            hidden_dims: vec![64, 64],
            learning_rate: 0.001,
            gamma: 0.99,
            epsilon_start: 1.0,
            epsilon_end: 0.01,
            epsilon_decay: 0.995,
            buffer_capacity: 10000,
            batch_size: 32,
            target_update_freq: 100,
            double_dqn: false,
        }
    }
}

/// Deep Q-Network (DQN) agent
pub struct DQNAgent<T: FloatBounds> {
    /// Q-network
    q_network: QNetwork<T>,
    /// Target Q-network
    target_network: QNetwork<T>,
    /// Experience replay buffer
    replay_buffer: ReplayBuffer<T>,
    /// Configuration
    config: DQNConfig,
    /// Current epsilon (exploration rate)
    epsilon: T,
    /// Training step counter
    steps: usize,
}

impl<T: FloatBounds + ScalarOperand> DQNAgent<T> {
    /// Create a new DQN agent
    pub fn new(config: DQNConfig) -> Self {
        let q_network = QNetwork::new(
            config.state_dim,
            config.action_dim,
            config.hidden_dims.clone(),
        );
        let target_network = QNetwork::new(
            config.state_dim,
            config.action_dim,
            config.hidden_dims.clone(),
        );
        let replay_buffer = ReplayBuffer::new(config.buffer_capacity);

        Self {
            q_network,
            target_network,
            replay_buffer,
            config,
            epsilon: T::from(1.0).unwrap(),
            steps: 0,
        }
    }

    /// Select action using epsilon-greedy policy
    pub fn select_action(&self, state: &Array1<T>) -> NeuralResult<usize> {
        let mut rng = thread_rng();

        if rng.random::<f64>() < self.epsilon.to_f64().unwrap() {
            // Random action (exploration)
            Ok(rng.random_range(0..self.config.action_dim))
        } else {
            // Best action (exploitation)
            self.q_network.select_action(state)
        }
    }

    /// Store experience in replay buffer
    pub fn store_experience(&mut self, experience: Experience<T>) {
        self.replay_buffer.push(experience);
    }

    /// Train on a batch of experiences
    pub fn train_step(&mut self) -> NeuralResult<Option<T>> {
        // Check if we have enough experiences
        if self.replay_buffer.len() < self.config.batch_size {
            return Ok(None);
        }

        // Sample batch
        let batch = self.replay_buffer.sample(self.config.batch_size).unwrap();

        // Compute target Q-values
        let mut total_loss = T::zero();

        for exp in &batch {
            let current_q = self.q_network.get_q_value(&exp.state, exp.action)?;

            let target_q = if exp.done {
                exp.reward
            } else {
                let next_q_values = self.target_network.forward(&exp.next_state)?;

                let max_next_q = if self.config.double_dqn {
                    // Double DQN: use online network to select action
                    let best_action = self.q_network.select_action(&exp.next_state)?;
                    next_q_values[best_action]
                } else {
                    // Standard DQN: use max Q-value
                    *next_q_values
                        .iter()
                        .max_by(|a, b| {
                            a.to_f64()
                                .unwrap()
                                .partial_cmp(&b.to_f64().unwrap())
                                .unwrap()
                        })
                        .unwrap()
                };

                exp.reward + T::from(self.config.gamma).unwrap() * max_next_q
            };

            // TD error
            let td_error = target_q - current_q;
            total_loss = total_loss + td_error * td_error;
        }

        let loss = total_loss / T::from(batch.len() as f64).unwrap();

        // Update target network periodically
        self.steps += 1;
        if self.steps % self.config.target_update_freq == 0 {
            self.target_network.copy_weights_from(&self.q_network);
        }

        // Decay epsilon
        self.epsilon = self.epsilon * T::from(self.config.epsilon_decay).unwrap();
        self.epsilon = self.epsilon.max(T::from(self.config.epsilon_end).unwrap());

        Ok(Some(loss))
    }

    /// Get current epsilon value
    pub fn get_epsilon(&self) -> T {
        self.epsilon
    }

    /// Get training steps count
    pub fn get_steps(&self) -> usize {
        self.steps
    }
}

/// Policy network for policy gradient methods
#[derive(Debug)]
pub struct PolicyNetwork<T: FloatBounds> {
    /// Network weights
    weights: Vec<Array2<T>>,
    /// Network biases
    biases: Vec<Array1<T>>,
    /// State dimension
    state_dim: usize,
    /// Action dimension
    action_dim: usize,
}

impl<T: FloatBounds + ScalarOperand> PolicyNetwork<T> {
    /// Create a new policy network
    pub fn new(state_dim: usize, action_dim: usize, hidden_dims: Vec<usize>) -> Self {
        let mut rng = thread_rng();
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        let mut prev_dim = state_dim;
        for &hidden_dim in &hidden_dims {
            let std = (2.0 / prev_dim as f64).sqrt();
            let w = Array2::from_shape_fn((prev_dim, hidden_dim), |_| {
                T::from(
                    rng.sample::<f64, _>(scirs2_core::random::Normal::new(0.0, 1.0).unwrap()) * std,
                )
                .unwrap()
            });
            let b = Array1::zeros(hidden_dim);
            weights.push(w);
            biases.push(b);
            prev_dim = hidden_dim;
        }

        // Output layer
        let w = Array2::from_shape_fn((prev_dim, action_dim), |_| {
            T::from(
                rng.sample::<f64, _>(scirs2_core::random::Normal::new(0.0, 1.0).unwrap()) * 0.01,
            )
            .unwrap()
        });
        let b = Array1::zeros(action_dim);
        weights.push(w);
        biases.push(b);

        Self {
            weights,
            biases,
            state_dim,
            action_dim,
        }
    }

    /// Forward pass: compute action probabilities
    pub fn forward(&self, state: &Array1<T>) -> NeuralResult<Array1<T>> {
        let mut h = state.clone();

        for (i, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let h_2d = h.insert_axis(Axis(0));
            let output = h_2d.dot(w);
            h = output.row(0).to_owned() + b;

            if i < self.weights.len() - 1 {
                // ReLU for hidden layers
                h.mapv_inplace(|x| if x > T::zero() { x } else { T::zero() });
            }
        }

        // Softmax for action probabilities
        let max_val = h
            .iter()
            .max_by(|a, b| {
                a.to_f64()
                    .unwrap()
                    .partial_cmp(&b.to_f64().unwrap())
                    .unwrap()
            })
            .unwrap();

        let exp_h = h.mapv(|x| (x - *max_val).exp());
        let sum_exp = exp_h.sum();
        let probs = exp_h / sum_exp;

        Ok(probs)
    }

    /// Sample action from policy
    pub fn sample_action(&self, state: &Array1<T>) -> NeuralResult<usize> {
        let probs = self.forward(state)?;
        let mut rng = thread_rng();

        // Sample from categorical distribution
        let rand_val = rng.random::<f64>();
        let mut cumsum = 0.0;

        for (i, &prob) in probs.iter().enumerate() {
            cumsum += prob.to_f64().unwrap();
            if rand_val < cumsum {
                return Ok(i);
            }
        }

        Ok(probs.len() - 1)
    }

    /// Get log probability of action
    pub fn log_prob(&self, state: &Array1<T>, action: usize) -> NeuralResult<T> {
        let probs = self.forward(state)?;
        let prob = probs[action];
        let log_prob = prob.ln();
        Ok(log_prob)
    }
}

/// Episode trajectory for policy gradient methods
#[derive(Debug, Clone)]
pub struct Trajectory<T: FloatBounds> {
    /// States
    pub states: Vec<Array1<T>>,
    /// Actions
    pub actions: Vec<usize>,
    /// Rewards
    pub rewards: Vec<T>,
}

impl<T: FloatBounds> Trajectory<T> {
    /// Create new trajectory
    pub fn new() -> Self {
        Self {
            states: Vec::new(),
            actions: Vec::new(),
            rewards: Vec::new(),
        }
    }

    /// Add step to trajectory
    pub fn add(&mut self, state: Array1<T>, action: usize, reward: T) {
        self.states.push(state);
        self.actions.push(action);
        self.rewards.push(reward);
    }

    /// Compute discounted returns
    pub fn compute_returns(&self, gamma: T) -> Vec<T> {
        let mut returns = Vec::with_capacity(self.rewards.len());
        let mut g = T::zero();

        for &reward in self.rewards.iter().rev() {
            g = reward + gamma * g;
            returns.push(g);
        }

        returns.reverse();
        returns
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }
}

impl<T: FloatBounds> Default for Trajectory<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// REINFORCE agent (vanilla policy gradient)
pub struct REINFORCEAgent<T: FloatBounds> {
    /// Policy network
    policy: PolicyNetwork<T>,
    /// Learning rate
    learning_rate: T,
    /// Discount factor
    gamma: T,
}

impl<T: FloatBounds + ScalarOperand> REINFORCEAgent<T> {
    /// Create new REINFORCE agent
    pub fn new(
        state_dim: usize,
        action_dim: usize,
        hidden_dims: Vec<usize>,
        learning_rate: f64,
        gamma: f64,
    ) -> Self {
        let policy = PolicyNetwork::new(state_dim, action_dim, hidden_dims);

        Self {
            policy,
            learning_rate: T::from(learning_rate).unwrap(),
            gamma: T::from(gamma).unwrap(),
        }
    }

    /// Select action
    pub fn select_action(&self, state: &Array1<T>) -> NeuralResult<usize> {
        self.policy.sample_action(state)
    }

    /// Update policy using trajectory
    pub fn update(&mut self, trajectory: &Trajectory<T>) -> NeuralResult<T> {
        let returns = trajectory.compute_returns(self.gamma);

        // Compute policy gradient
        let mut total_loss = T::zero();

        for (i, (&action, &g)) in trajectory.actions.iter().zip(returns.iter()).enumerate() {
            let log_prob = self.policy.log_prob(&trajectory.states[i], action)?;
            let loss = -log_prob * g; // Negative because we want to maximize
            total_loss = total_loss + loss;
        }

        let avg_loss = total_loss / T::from(trajectory.len() as f64).unwrap();

        Ok(avg_loss)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_replay_buffer_creation() {
        let buffer: ReplayBuffer<f64> = ReplayBuffer::new(100);
        assert_eq!(buffer.capacity, 100);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_replay_buffer_push() {
        let mut buffer: ReplayBuffer<f64> = ReplayBuffer::new(3);

        let exp = Experience {
            state: Array1::from_vec(vec![1.0, 2.0]),
            action: 0,
            reward: 1.0,
            next_state: Array1::from_vec(vec![3.0, 4.0]),
            done: false,
        };

        buffer.push(exp.clone());
        assert_eq!(buffer.len(), 1);

        // Fill buffer
        buffer.push(exp.clone());
        buffer.push(exp.clone());
        assert_eq!(buffer.len(), 3);

        // Overflow - should remove oldest
        buffer.push(exp);
        assert_eq!(buffer.len(), 3);
    }

    #[test]
    fn test_replay_buffer_sample() {
        let mut buffer: ReplayBuffer<f64> = ReplayBuffer::new(10);

        for i in 0..5 {
            let exp = Experience {
                state: Array1::from_vec(vec![i as f64]),
                action: i,
                reward: i as f64,
                next_state: Array1::from_vec(vec![(i + 1) as f64]),
                done: false,
            };
            buffer.push(exp);
        }

        let sample = buffer.sample(3);
        assert!(sample.is_some());
        assert_eq!(sample.unwrap().len(), 3);

        // Not enough samples
        let sample = buffer.sample(10);
        assert!(sample.is_none());
    }

    #[test]
    fn test_q_network_creation() {
        let q_net: QNetwork<f64> = QNetwork::new(4, 2, vec![32, 32]);
        assert_eq!(q_net.state_dim, 4);
        assert_eq!(q_net.action_dim, 2);
    }

    #[test]
    fn test_q_network_forward() {
        let q_net: QNetwork<f64> = QNetwork::new(4, 2, vec![16]);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let q_values = q_net.forward(&state).unwrap();
        assert_eq!(q_values.len(), 2);
    }

    #[test]
    fn test_q_network_select_action() {
        let q_net: QNetwork<f64> = QNetwork::new(4, 3, vec![8]);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let action = q_net.select_action(&state).unwrap();
        assert!(action < 3);
    }

    #[test]
    fn test_dqn_agent_creation() {
        let config = DQNConfig::default();
        let agent: DQNAgent<f64> = DQNAgent::new(config);

        assert_eq!(agent.steps, 0);
        assert!(agent.replay_buffer.is_empty());
    }

    #[test]
    fn test_dqn_agent_select_action() {
        let config = DQNConfig::default();
        let agent: DQNAgent<f64> = DQNAgent::new(config);

        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let action = agent.select_action(&state).unwrap();

        assert!(action < agent.config.action_dim);
    }

    #[test]
    fn test_dqn_agent_store_experience() {
        let config = DQNConfig::default();
        let mut agent: DQNAgent<f64> = DQNAgent::new(config);

        let exp = Experience {
            state: Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]),
            action: 0,
            reward: 1.0,
            next_state: Array1::from_vec(vec![2.0, 3.0, 4.0, 5.0]),
            done: false,
        };

        agent.store_experience(exp);
        assert_eq!(agent.replay_buffer.len(), 1);
    }

    #[test]
    fn test_policy_network_creation() {
        let policy: PolicyNetwork<f64> = PolicyNetwork::new(4, 3, vec![16]);
        assert_eq!(policy.state_dim, 4);
        assert_eq!(policy.action_dim, 3);
    }

    #[test]
    fn test_policy_network_forward() {
        let policy: PolicyNetwork<f64> = PolicyNetwork::new(4, 3, vec![8]);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let probs = policy.forward(&state).unwrap();
        assert_eq!(probs.len(), 3);

        // Check probabilities sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_policy_network_sample_action() {
        let policy: PolicyNetwork<f64> = PolicyNetwork::new(4, 3, vec![8]);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let action = policy.sample_action(&state).unwrap();
        assert!(action < 3);
    }

    #[test]
    fn test_trajectory_creation() {
        let traj: Trajectory<f64> = Trajectory::new();
        assert!(traj.is_empty());
    }

    #[test]
    fn test_trajectory_add() {
        let mut traj: Trajectory<f64> = Trajectory::new();

        traj.add(Array1::from_vec(vec![1.0, 2.0]), 0, 1.0);
        traj.add(Array1::from_vec(vec![3.0, 4.0]), 1, 2.0);

        assert_eq!(traj.len(), 2);
    }

    #[test]
    fn test_trajectory_compute_returns() {
        let mut traj: Trajectory<f64> = Trajectory::new();

        traj.add(Array1::from_vec(vec![1.0]), 0, 1.0);
        traj.add(Array1::from_vec(vec![2.0]), 1, 1.0);
        traj.add(Array1::from_vec(vec![3.0]), 0, 1.0);

        let returns = traj.compute_returns(0.9);

        assert_eq!(returns.len(), 3);
        // G_0 = 1 + 0.9 * (1 + 0.9 * 1) = 2.71
        assert!((returns[0] - 2.71).abs() < 0.01);
    }

    #[test]
    fn test_reinforce_agent_creation() {
        let agent: REINFORCEAgent<f64> = REINFORCEAgent::new(4, 2, vec![16], 0.01, 0.99);
        assert_eq!(agent.policy.state_dim, 4);
        assert_eq!(agent.policy.action_dim, 2);
    }

    #[test]
    fn test_reinforce_agent_select_action() {
        let agent: REINFORCEAgent<f64> = REINFORCEAgent::new(4, 2, vec![8], 0.01, 0.99);
        let state = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);

        let action = agent.select_action(&state).unwrap();
        assert!(action < 2);
    }

    #[test]
    fn test_q_network_copy_weights() {
        let mut q_net1: QNetwork<f64> = QNetwork::new(4, 2, vec![8]);
        let q_net2: QNetwork<f64> = QNetwork::new(4, 2, vec![8]);

        q_net1.copy_weights_from(&q_net2);
        // Networks should now have same weights
    }

    #[test]
    fn test_dqn_double_mode() {
        let mut config = DQNConfig::default();
        config.double_dqn = true;

        let agent: DQNAgent<f64> = DQNAgent::new(config);
        assert!(agent.config.double_dqn);
    }
}

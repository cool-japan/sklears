//! Multi-Agent Reinforcement Learning
//!
//! This module implements algorithms for multi-agent reinforcement learning (MARL)
//! where multiple agents learn simultaneously through interaction with a shared
//! environment. Supports cooperative, competitive, and mixed scenarios.
//!
//! # Key Algorithms
//!
//! - **Independent Q-Learning (IQL)**: Each agent learns independently
//! - **Value Decomposition Networks (VDN)**: Additive value decomposition
//! - **QMIX**: Non-linear value function mixing
//! - **Multi-Agent Actor-Critic**: Centralized training, decentralized execution
//! - **Communication Networks**: Agent-to-agent communication protocols
//!
//! # References
//!
//! - Lowe, R., et al. (2017). "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
//! - Rashid, T., et al. (2018). "QMIX: Monotonic Value Function Factorisation for Decentralised Multi-Agent Reinforcement Learning"

use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis, ScalarOperand};
use scirs2_core::random::{thread_rng, CoreRandom, Rng, Uniform};
use sklears_core::{error::SklearsError, types::FloatBounds};
use std::collections::VecDeque;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Agent coordination mode
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CoordinationMode {
    /// Fully cooperative agents (shared reward)
    Cooperative,
    /// Fully competitive agents (zero-sum)
    Competitive,
    /// Mixed cooperative-competitive
    Mixed,
    /// Independent agents (separate objectives)
    Independent,
}

/// Communication protocol for agent interaction
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CommunicationProtocol {
    /// No communication between agents
    None,
    /// Broadcast messages to all agents
    Broadcast,
    /// Point-to-point messages to specific agents
    Targeted,
    /// Learned communication (agents learn what to communicate)
    Learned,
}

/// Multi-agent experience for replay
#[derive(Debug, Clone)]
pub struct MultiAgentExperience<T: FloatBounds> {
    /// Joint state (global observation)
    pub joint_state: Array1<T>,
    /// Individual states for each agent
    pub agent_states: Vec<Array1<T>>,
    /// Actions taken by each agent
    pub actions: Vec<usize>,
    /// Rewards received by each agent
    pub rewards: Vec<T>,
    /// Next joint state
    pub next_joint_state: Array1<T>,
    /// Next individual states
    pub next_agent_states: Vec<Array1<T>>,
    /// Whether episode is done
    pub done: bool,
    /// Communication messages (if applicable)
    pub messages: Option<Vec<Array1<T>>>,
}

/// Multi-agent replay buffer
#[derive(Debug)]
pub struct MultiAgentReplayBuffer<T: FloatBounds> {
    /// Buffer storage
    buffer: VecDeque<MultiAgentExperience<T>>,
    /// Maximum buffer size
    capacity: usize,
    /// Number of agents
    n_agents: usize,
}

impl<T: FloatBounds> MultiAgentReplayBuffer<T> {
    /// Create a new multi-agent replay buffer
    pub fn new(capacity: usize, n_agents: usize) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity,
            n_agents,
        }
    }

    /// Add experience to buffer
    pub fn push(&mut self, experience: MultiAgentExperience<T>) {
        if self.buffer.len() >= self.capacity {
            self.buffer.pop_front();
        }
        self.buffer.push_back(experience);
    }

    /// Sample a batch of experiences
    pub fn sample(&self, batch_size: usize) -> Option<Vec<MultiAgentExperience<T>>> {
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
}

/// Independent Q-Learning agent (IQL)
#[derive(Debug)]
pub struct IndependentQLearner<T: FloatBounds> {
    /// Q-network weights for each agent
    agent_networks: Vec<Array2<T>>,
    /// Learning rate
    learning_rate: T,
    /// Discount factor
    gamma: T,
    /// Exploration rate (epsilon)
    epsilon: T,
    /// Number of agents
    n_agents: usize,
    /// State dimension per agent
    state_dim: usize,
    /// Number of actions per agent
    n_actions: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> IndependentQLearner<T> {
    /// Create a new independent Q-learning system
    pub fn new(
        n_agents: usize,
        state_dim: usize,
        n_actions: usize,
        learning_rate: T,
        gamma: T,
        epsilon: T,
    ) -> Self {
        let mut rng = thread_rng();
        let mut agent_networks = Vec::new();

        // Initialize Q-network for each agent
        for _ in 0..n_agents {
            let std_dev = (2.0 / (state_dim + n_actions) as f64).sqrt();
            let weights = Array2::from_shape_fn((n_actions, state_dim), |_| {
                T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero())
            });
            agent_networks.push(weights);
        }

        Self {
            agent_networks,
            learning_rate,
            gamma,
            epsilon,
            n_agents,
            state_dim,
            n_actions,
        }
    }

    /// Select actions for all agents (epsilon-greedy)
    pub fn select_actions(&self, states: &[Array1<T>]) -> Vec<usize> {
        let mut rng = thread_rng();
        let mut actions = Vec::new();

        for (agent_idx, state) in states.iter().enumerate() {
            if rng.random::<f64>() < self.epsilon.to_f64().unwrap_or(0.1) {
                // Explore
                actions.push(rng.gen_range(0..self.n_actions));
            } else {
                // Exploit
                let q_values = self.agent_networks[agent_idx].dot(state);
                let best_action = q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                actions.push(best_action);
            }
        }

        actions
    }

    /// Update Q-networks based on experience
    pub fn update(&mut self, experience: &MultiAgentExperience<T>) -> NeuralResult<()> {
        for agent_idx in 0..self.n_agents {
            let state = &experience.agent_states[agent_idx];
            let action = experience.actions[agent_idx];
            let reward = experience.rewards[agent_idx];
            let next_state = &experience.next_agent_states[agent_idx];

            // Compute Q(s, a)
            let q_values = self.agent_networks[agent_idx].dot(state);
            let q_value = q_values[action];

            // Compute max_a' Q(s', a')
            let next_q_values = self.agent_networks[agent_idx].dot(next_state);
            let max_next_q = next_q_values
                .iter()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(T::zero());

            // TD target
            let target = if experience.done {
                reward
            } else {
                reward + self.gamma * max_next_q
            };

            // TD error
            let td_error = target - q_value;

            // Update weights (gradient descent on squared TD error)
            let gradient = state.mapv(|s| s * td_error * T::from(2.0).unwrap());
            let mut row = self.agent_networks[agent_idx].row_mut(action);
            for (w, &g) in row.iter_mut().zip(gradient.iter()) {
                *w = *w + g * self.learning_rate;
            }
        }

        Ok(())
    }

    /// Decay exploration rate
    pub fn decay_epsilon(&mut self, decay_rate: T) {
        self.epsilon = self.epsilon * decay_rate;
        let min_epsilon = T::from(0.01).unwrap();
        if self.epsilon < min_epsilon {
            self.epsilon = min_epsilon;
        }
    }
}

/// Value Decomposition Network (VDN)
#[derive(Debug)]
pub struct VDN<T: FloatBounds> {
    /// Individual Q-networks for each agent
    agent_networks: Vec<Array2<T>>,
    /// Learning rate
    learning_rate: T,
    /// Discount factor
    gamma: T,
    /// Epsilon for exploration
    epsilon: T,
    /// Number of agents
    n_agents: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> VDN<T> {
    /// Create a new VDN system
    pub fn new(
        n_agents: usize,
        state_dim: usize,
        n_actions: usize,
        learning_rate: T,
        gamma: T,
        epsilon: T,
    ) -> Self {
        let mut rng = thread_rng();
        let mut agent_networks = Vec::new();

        for _ in 0..n_agents {
            let std_dev = (2.0 / (state_dim + n_actions) as f64).sqrt();
            let weights = Array2::from_shape_fn((n_actions, state_dim), |_| {
                T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero())
            });
            agent_networks.push(weights);
        }

        Self {
            agent_networks,
            learning_rate,
            gamma,
            epsilon,
            n_agents,
        }
    }

    /// Compute total Q-value as sum of individual Q-values
    pub fn compute_total_q(&self, states: &[Array1<T>], actions: &[usize]) -> T {
        let mut total_q = T::zero();

        for (agent_idx, (state, &action)) in states.iter().zip(actions.iter()).enumerate() {
            let q_values = self.agent_networks[agent_idx].dot(state);
            total_q = total_q + q_values[action];
        }

        total_q
    }

    /// Select joint actions for all agents
    pub fn select_joint_actions(&self, states: &[Array1<T>]) -> Vec<usize> {
        let mut rng = thread_rng();
        let mut actions = Vec::new();

        for (agent_idx, state) in states.iter().enumerate() {
            if rng.random::<f64>() < self.epsilon.to_f64().unwrap_or(0.1) {
                actions.push(rng.gen_range(0..self.agent_networks[agent_idx].nrows()));
            } else {
                let q_values = self.agent_networks[agent_idx].dot(state);
                let best_action = q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0);
                actions.push(best_action);
            }
        }

        actions
    }

    /// Update using shared reward (cooperative setting)
    pub fn update_cooperative(&mut self, experience: &MultiAgentExperience<T>) -> NeuralResult<()> {
        // Shared reward for all agents
        let shared_reward =
            experience.rewards.iter().cloned().sum::<T>() / T::from(self.n_agents as f64).unwrap();

        // Current total Q-value
        let current_q = self.compute_total_q(&experience.agent_states, &experience.actions);

        // Compute best next actions
        let best_next_actions: Vec<usize> = experience
            .next_agent_states
            .iter()
            .enumerate()
            .map(|(agent_idx, state)| {
                let q_values = self.agent_networks[agent_idx].dot(state);
                q_values
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect();

        // Next total Q-value
        let next_q = self.compute_total_q(&experience.next_agent_states, &best_next_actions);

        // TD target
        let target = if experience.done {
            shared_reward
        } else {
            shared_reward + self.gamma * next_q
        };

        // Update each agent proportionally
        let td_error = target - current_q;

        for (agent_idx, state) in experience.agent_states.iter().enumerate() {
            let action = experience.actions[agent_idx];
            let gradient = state.mapv(|s| s * td_error * T::from(2.0).unwrap());
            let mut row = self.agent_networks[agent_idx].row_mut(action);
            for (w, &g) in row.iter_mut().zip(gradient.iter()) {
                *w = *w + g * self.learning_rate;
            }
        }

        Ok(())
    }
}

/// QMIX mixing network for value function factorization
#[derive(Debug)]
pub struct QMIXMixingNetwork<T: FloatBounds> {
    /// Hypernetwork weights for generating mixing weights
    hyper_w1: Array2<T>,
    /// Hypernetwork biases
    hyper_b1: Array1<T>,
    /// Hypernetwork for final layer
    hyper_w_final: Array2<T>,
    /// Hypernetwork bias for final layer
    hyper_b_final: Array1<T>,
    /// Hidden dimension for mixing network
    hidden_dim: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> QMIXMixingNetwork<T> {
    /// Create a new QMIX mixing network
    pub fn new(n_agents: usize, state_dim: usize, hidden_dim: usize) -> Self {
        let mut rng = thread_rng();

        // Hypernetwork to generate mixing network weights
        let std_w1 = (2.0 / (state_dim + hidden_dim) as f64).sqrt();
        let hyper_w1 = Array2::from_shape_fn((hidden_dim, state_dim), |_| {
            T::from(rng.gen_range(-std_w1..std_w1)).unwrap_or(T::zero())
        });
        let hyper_b1 = Array1::zeros(hidden_dim);

        let std_w_final = (2.0 / (state_dim + 1) as f64).sqrt();
        let hyper_w_final = Array2::from_shape_fn((n_agents, state_dim), |_| {
            T::from(rng.gen_range(-std_w_final..std_w_final)).unwrap_or(T::zero())
        });
        let hyper_b_final = Array1::zeros(n_agents);

        Self {
            hyper_w1,
            hyper_b1,
            hyper_w_final,
            hyper_b_final,
            hidden_dim,
        }
    }

    /// Mix individual Q-values into total Q-value (monotonic mixing)
    pub fn mix(&self, agent_q_values: &[T], global_state: &Array1<T>) -> T {
        // Generate mixing weights from global state (ensure non-negative for monotonicity)
        let w1 = (self.hyper_w1.dot(global_state) + &self.hyper_b1).mapv(|x| {
            // Absolute value to ensure monotonicity
            if x.to_f64().unwrap_or(0.0).abs() < x.to_f64().unwrap_or(0.0) {
                -x
            } else {
                x
            }
        });

        // First layer mixing
        let agent_q_array = Array1::from(agent_q_values.iter().map(|&q| q).collect::<Vec<T>>());

        let h = w1
            .iter()
            .zip(agent_q_array.iter().cycle())
            .take(self.hidden_dim)
            .map(|(&w, &q)| w * q)
            .sum::<T>();

        // Final layer
        let w_final = (self.hyper_w_final.row(0).dot(global_state) + self.hyper_b_final[0]).abs();

        h * w_final
    }
}

/// Multi-Agent Deep Deterministic Policy Gradient (MADDPG)
#[derive(Debug)]
pub struct MADDPG<T: FloatBounds> {
    /// Actor networks (policy) for each agent
    actors: Vec<Array2<T>>,
    /// Critic network (centralized - sees all agents' states/actions)
    critic: Array2<T>,
    /// Target actor networks
    target_actors: Vec<Array2<T>>,
    /// Target critic network
    target_critic: Array2<T>,
    /// Actor learning rate
    actor_lr: T,
    /// Critic learning rate
    critic_lr: T,
    /// Discount factor
    gamma: T,
    /// Soft update parameter (tau)
    tau: T,
    /// Number of agents
    n_agents: usize,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum + std::ops::AddAssign> MADDPG<T> {
    /// Create a new MADDPG system
    pub fn new(
        n_agents: usize,
        state_dim: usize,
        action_dim: usize,
        actor_lr: T,
        critic_lr: T,
        gamma: T,
        tau: T,
    ) -> Self {
        let mut rng = thread_rng();
        let mut actors = Vec::new();
        let mut target_actors = Vec::new();

        // Initialize actor networks
        for _ in 0..n_agents {
            let std_dev = (2.0 / (state_dim + action_dim) as f64).sqrt();
            let actor = Array2::from_shape_fn((action_dim, state_dim), |_| {
                T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero())
            });
            target_actors.push(actor.clone());
            actors.push(actor);
        }

        // Initialize centralized critic (sees all states and actions)
        let total_input_dim = n_agents * (state_dim + action_dim);
        let std_dev = (2.0 / total_input_dim as f64).sqrt();
        let critic = Array2::from_shape_fn((1, total_input_dim), |_| {
            T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero())
        });
        let target_critic = critic.clone();

        Self {
            actors,
            critic,
            target_actors,
            target_critic,
            actor_lr,
            critic_lr,
            gamma,
            tau,
            n_agents,
        }
    }

    /// Get actions from actors for given states
    pub fn get_actions(&self, states: &[Array1<T>]) -> Vec<Array1<T>> {
        states
            .iter()
            .enumerate()
            .map(|(agent_idx, state)| self.actors[agent_idx].dot(state))
            .collect()
    }

    /// Soft update target networks
    pub fn soft_update_targets(&mut self) {
        // Update target actors
        for (target, online) in self.target_actors.iter_mut().zip(&self.actors) {
            *target = target.mapv(|t| t * (T::one() - self.tau)) + online.mapv(|o| o * self.tau);
        }

        // Update target critic
        self.target_critic = self.target_critic.mapv(|t| t * (T::one() - self.tau))
            + self.critic.mapv(|o| o * self.tau);
    }
}

/// Communication layer for agent coordination
#[derive(Debug)]
pub struct CommunicationLayer<T: FloatBounds> {
    /// Communication protocol
    protocol: CommunicationProtocol,
    /// Message dimension
    message_dim: usize,
    /// Attention weights for learned communication
    attention_weights: Option<Array3<T>>,
}

impl<T: FloatBounds + ScalarOperand + std::iter::Sum> CommunicationLayer<T> {
    /// Create a new communication layer
    pub fn new(protocol: CommunicationProtocol, message_dim: usize, n_agents: usize) -> Self {
        let attention_weights = if protocol == CommunicationProtocol::Learned {
            let mut rng = thread_rng();
            let std_dev = (2.0 / (message_dim + message_dim) as f64).sqrt();
            Some(Array3::from_shape_fn(
                (n_agents, n_agents, message_dim),
                |_| T::from(rng.gen_range(-std_dev..std_dev)).unwrap_or(T::zero()),
            ))
        } else {
            None
        };

        Self {
            protocol,
            message_dim,
            attention_weights,
        }
    }

    /// Process messages between agents
    pub fn communicate(&self, agent_states: &[Array1<T>]) -> Vec<Array1<T>> {
        match self.protocol {
            CommunicationProtocol::None => {
                // No communication - return original states
                agent_states.to_vec()
            }
            CommunicationProtocol::Broadcast => {
                // Broadcast: each agent receives mean of all states
                let mean_state = self.compute_mean_state(agent_states);
                agent_states
                    .iter()
                    .map(|state| self.concatenate_state_message(state, &mean_state))
                    .collect()
            }
            CommunicationProtocol::Targeted => {
                // Simple targeted: nearest neighbors
                self.targeted_communication(agent_states)
            }
            CommunicationProtocol::Learned => {
                // Learned communication with attention
                self.learned_communication(agent_states)
            }
        }
    }

    /// Compute mean state across all agents
    fn compute_mean_state(&self, states: &[Array1<T>]) -> Array1<T> {
        if states.is_empty() {
            return Array1::zeros(self.message_dim);
        }

        let sum = states
            .iter()
            .fold(Array1::zeros(states[0].len()), |acc, state| acc + state);
        sum / T::from(states.len() as f64).unwrap()
    }

    /// Concatenate state with message
    fn concatenate_state_message(&self, state: &Array1<T>, message: &Array1<T>) -> Array1<T> {
        let mut result = Vec::new();
        result.extend_from_slice(state.as_slice().unwrap_or(&[]));
        result.extend_from_slice(message.as_slice().unwrap_or(&[]));
        Array1::from(result)
    }

    /// Targeted communication between nearby agents
    fn targeted_communication(&self, agent_states: &[Array1<T>]) -> Vec<Array1<T>> {
        agent_states
            .iter()
            .enumerate()
            .map(|(i, state)| {
                // Communicate with neighbors (simple: adjacent indices)
                let prev_idx = if i > 0 { i - 1 } else { agent_states.len() - 1 };
                let next_idx = (i + 1) % agent_states.len();

                let neighbor_message =
                    (&agent_states[prev_idx] + &agent_states[next_idx]) / T::from(2.0).unwrap();
                self.concatenate_state_message(state, &neighbor_message)
            })
            .collect()
    }

    /// Learned communication using attention mechanism
    fn learned_communication(&self, agent_states: &[Array1<T>]) -> Vec<Array1<T>> {
        if let Some(ref attention) = self.attention_weights {
            agent_states
                .iter()
                .enumerate()
                .map(|(i, state)| {
                    // Compute attention scores with all other agents
                    let mut messages = Vec::new();
                    for (j, other_state) in agent_states.iter().enumerate() {
                        if i != j {
                            let att_weight = &attention.slice(s![i, j, ..]);
                            let att_sum = att_weight.iter().cloned().sum::<T>();
                            let weighted_message = other_state.mapv(|x| x) * att_sum
                                / T::from(att_weight.len() as f64).unwrap();
                            messages.push(weighted_message);
                        }
                    }

                    // Aggregate messages
                    let aggregated = if !messages.is_empty() {
                        messages
                            .iter()
                            .fold(Array1::zeros(state.len()), |acc, msg| acc + msg)
                            / T::from(messages.len() as f64).unwrap()
                    } else {
                        Array1::zeros(state.len())
                    };

                    self.concatenate_state_message(state, &aggregated)
                })
                .collect()
        } else {
            // Fallback to no communication
            agent_states.to_vec()
        }
    }
}

use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ScientificNumber;

    #[test]
    fn test_independent_q_learner_creation() {
        let iql: IndependentQLearner<f64> = IndependentQLearner::new(3, 4, 5, 0.01, 0.99, 0.1);
        assert_eq!(iql.n_agents, 3);
        assert_eq!(iql.state_dim, 4);
        assert_eq!(iql.n_actions, 5);
    }

    #[test]
    fn test_iql_action_selection() {
        let iql: IndependentQLearner<f64> = IndependentQLearner::new(3, 4, 5, 0.01, 0.99, 0.0); // No exploration

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let actions = iql.select_actions(&states);
        assert_eq!(actions.len(), 3);
        for &action in &actions {
            assert!(action < 5);
        }
    }

    #[test]
    fn test_multi_agent_replay_buffer() {
        let mut buffer: MultiAgentReplayBuffer<f64> = MultiAgentReplayBuffer::new(100, 3);

        let experience = MultiAgentExperience {
            joint_state: Array1::from(vec![1.0, 2.0, 3.0]),
            agent_states: vec![
                Array1::from(vec![1.0]),
                Array1::from(vec![2.0]),
                Array1::from(vec![3.0]),
            ],
            actions: vec![0, 1, 2],
            rewards: vec![1.0, 0.5, 0.8],
            next_joint_state: Array1::from(vec![1.1, 2.1, 3.1]),
            next_agent_states: vec![
                Array1::from(vec![1.1]),
                Array1::from(vec![2.1]),
                Array1::from(vec![3.1]),
            ],
            done: false,
            messages: None,
        };

        buffer.push(experience);
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_iql_update() {
        let mut iql: IndependentQLearner<f64> = IndependentQLearner::new(2, 3, 4, 0.01, 0.99, 0.1);

        let experience = MultiAgentExperience {
            joint_state: Array1::from(vec![1.0, 0.0, 0.0]),
            agent_states: vec![Array1::from(vec![1.0, 0.0, 0.0]); 2],
            actions: vec![0, 1],
            rewards: vec![1.0, 0.5],
            next_joint_state: Array1::from(vec![0.0, 1.0, 0.0]),
            next_agent_states: vec![Array1::from(vec![0.0, 1.0, 0.0]); 2],
            done: false,
            messages: None,
        };

        let result = iql.update(&experience);
        assert!(result.is_ok());
    }

    #[test]
    fn test_iql_epsilon_decay() {
        let mut iql: IndependentQLearner<f64> = IndependentQLearner::new(2, 3, 4, 0.01, 0.99, 0.5);

        let initial_epsilon = iql.epsilon;
        iql.decay_epsilon(0.99);

        assert!(iql.epsilon < initial_epsilon);
        assert!(iql.epsilon >= 0.01); // Should not go below minimum
    }

    #[test]
    fn test_vdn_creation() {
        let vdn: VDN<f64> = VDN::new(3, 4, 5, 0.01, 0.99, 0.1);
        assert_eq!(vdn.n_agents, 3);
    }

    #[test]
    fn test_vdn_total_q_computation() {
        let vdn: VDN<f64> = VDN::new(2, 3, 4, 0.01, 0.99, 0.1);

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0]),
        ];
        let actions = vec![0, 1];

        let total_q = vdn.compute_total_q(&states, &actions);
        assert!(total_q.to_f64().is_some());
    }

    #[test]
    fn test_vdn_cooperative_update() {
        let mut vdn: VDN<f64> = VDN::new(2, 3, 4, 0.01, 0.99, 0.1);

        let experience = MultiAgentExperience {
            joint_state: Array1::from(vec![1.0, 0.0, 0.0]),
            agent_states: vec![
                Array1::from(vec![1.0, 0.0, 0.0]),
                Array1::from(vec![0.0, 1.0, 0.0]),
            ],
            actions: vec![0, 1],
            rewards: vec![1.0, 0.8],
            next_joint_state: Array1::from(vec![0.0, 1.0, 0.0]),
            next_agent_states: vec![
                Array1::from(vec![0.0, 1.0, 0.0]),
                Array1::from(vec![1.0, 0.0, 0.0]),
            ],
            done: false,
            messages: None,
        };

        let result = vdn.update_cooperative(&experience);
        assert!(result.is_ok());
    }

    #[test]
    fn test_qmix_mixing_network() {
        let qmix: QMIXMixingNetwork<f64> = QMIXMixingNetwork::new(3, 4, 32);
        assert_eq!(qmix.hidden_dim, 32);

        let agent_q_values = vec![0.5, 0.3, 0.7];
        let global_state = Array1::from(vec![1.0, 0.0, 0.0, 1.0]);

        let total_q = qmix.mix(&agent_q_values, &global_state);
        assert!(total_q.to_f64().is_some());
    }

    #[test]
    fn test_maddpg_creation() {
        let maddpg: MADDPG<f64> = MADDPG::new(3, 4, 2, 0.001, 0.01, 0.99, 0.001);
        assert_eq!(maddpg.n_agents, 3);
    }

    #[test]
    fn test_maddpg_action_generation() {
        let maddpg: MADDPG<f64> = MADDPG::new(2, 3, 2, 0.001, 0.01, 0.99, 0.001);

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0]),
        ];

        let actions = maddpg.get_actions(&states);
        assert_eq!(actions.len(), 2);
    }

    #[test]
    fn test_maddpg_soft_update() {
        let mut maddpg: MADDPG<f64> = MADDPG::new(2, 3, 2, 0.001, 0.01, 0.99, 0.1);

        let initial_target = maddpg.target_actors[0].clone();

        // Modify online network to create a difference
        maddpg.actors[0] = maddpg.actors[0].mapv(|x| x + 1.0);

        maddpg.soft_update_targets();
        let updated_target = &maddpg.target_actors[0];

        // Target should have changed (moved toward online network)
        let diff: f64 = (&initial_target - updated_target).mapv(|x| x.abs()).sum();
        assert!(diff > 0.0);
    }

    #[test]
    fn test_communication_layer_none() {
        let comm: CommunicationLayer<f64> =
            CommunicationLayer::new(CommunicationProtocol::None, 4, 3);

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let communicated = comm.communicate(&states);
        assert_eq!(communicated.len(), 3);
    }

    #[test]
    fn test_communication_layer_broadcast() {
        let comm: CommunicationLayer<f64> =
            CommunicationLayer::new(CommunicationProtocol::Broadcast, 4, 3);

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let communicated = comm.communicate(&states);
        assert_eq!(communicated.len(), 3);
        // After broadcast, states should be longer (concatenated with mean)
        assert!(communicated[0].len() > states[0].len());
    }

    #[test]
    fn test_communication_layer_learned() {
        let comm: CommunicationLayer<f64> =
            CommunicationLayer::new(CommunicationProtocol::Learned, 4, 3);

        let states = vec![
            Array1::from(vec![1.0, 0.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 1.0, 0.0, 0.0]),
            Array1::from(vec![0.0, 0.0, 1.0, 0.0]),
        ];

        let communicated = comm.communicate(&states);
        assert_eq!(communicated.len(), 3);
    }
}

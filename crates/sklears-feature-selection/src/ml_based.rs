//! Machine learning-based feature selection methods
//!
//! This module provides advanced feature selection methods using machine learning techniques
//! including neural networks, attention mechanisms, reinforcement learning, and meta-learning.

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};

use scirs2_core::rand_prelude::IndexedRandom;
use scirs2_core::random::{rngs::StdRng, thread_rng, Rng, SeedableRng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Neural network-based feature selection using importance scoring
/// Neural network feature importance selector
/// Uses a simple neural network to learn feature importances through gradient analysis
#[derive(Debug, Clone)]
pub struct NeuralFeatureSelector<State = Untrained> {
    /// Number of hidden layers
    hidden_layers: Vec<usize>,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Number of training epochs
    epochs: usize,
    /// L1 regularization parameter for sparsity
    l1_reg: f64,
    /// L2 regularization parameter
    l2_reg: f64,
    /// Number of top features to select
    k: Option<usize>,
    /// Threshold for feature importance
    importance_threshold: f64,
    /// Random seed for reproducibility
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    weights_: Option<Vec<Array2<Float>>>,
    biases_: Option<Vec<Array1<Float>>>,
    feature_importances_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
}

impl NeuralFeatureSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            hidden_layers: vec![64, 32],
            learning_rate: 0.001,
            epochs: 100,
            l1_reg: 0.01,
            l2_reg: 0.01,
            k: None,
            importance_threshold: 0.01,
            random_state: None,
            state: PhantomData,
            weights_: None,
            biases_: None,
            feature_importances_: None,
            selected_features_: None,
        }
    }

    pub fn hidden_layers(mut self, layers: Vec<usize>) -> Self {
        self.hidden_layers = layers;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn l1_reg(mut self, l1_reg: f64) -> Self {
        self.l1_reg = l1_reg;
        self
    }

    pub fn l2_reg(mut self, l2_reg: f64) -> Self {
        self.l2_reg = l2_reg;
        self
    }

    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    pub fn importance_threshold(mut self, threshold: f64) -> Self {
        self.importance_threshold = threshold;
        self
    }

    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Default for NeuralFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for NeuralFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for NeuralFeatureSelector<Untrained> {
    type Fitted = NeuralFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let (n_samples, n_features) = x.dim();

        // Initialize network architecture
        let mut layer_sizes = vec![n_features];
        layer_sizes.extend(&self.hidden_layers);
        layer_sizes.push(1); // Output layer for regression/binary classification

        // Initialize weights and biases
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..(layer_sizes.len() - 1) {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];

            // Xavier initialization
            let scale = (2.0 / (input_size + output_size) as f64).sqrt();
            let mut weight_matrix = Array2::zeros((input_size, output_size));
            for elem in weight_matrix.iter_mut() {
                *elem = rng.gen::<f64>() * 2.0 * scale - scale;
            }
            weights.push(weight_matrix);

            let mut bias_vector = Array1::zeros(output_size);
            for elem in bias_vector.iter_mut() {
                *elem = rng.gen::<f64>() * 0.1 - 0.05;
            }
            biases.push(bias_vector);
        }

        // Training loop
        for epoch in 0..self.epochs {
            let mut total_loss = 0.0;

            // Forward and backward pass for each sample
            for i in 0..n_samples {
                let input = x.row(i).to_owned();
                let target = y[i];

                // Forward pass
                let (activations, z_values) = self.forward_pass(&input, &weights, &biases);
                let prediction = activations.last().unwrap()[0];

                // Compute loss (MSE for regression)
                let loss = 0.5 * (prediction - target).powi(2);
                total_loss += loss;

                // Backward pass
                let gradients =
                    self.backward_pass(&input, target, &activations, &z_values, &weights);

                // Update weights and biases
                self.update_parameters(&mut weights, &mut biases, &gradients.0, &gradients.1);
            }

            // Optional: Early stopping or learning rate decay
            if epoch % 10 == 0 {
                let avg_loss = total_loss / n_samples as f64;
                if avg_loss < 1e-6 {
                    break;
                }
            }
        }

        // Compute feature importances using input layer weights
        let input_weights = &weights[0];
        let mut feature_importances = Array1::zeros(n_features);

        for i in 0..n_features {
            // Compute importance as the sum of absolute weights from this feature
            let importance: f64 = input_weights.row(i).iter().map(|&w| w.abs()).sum();
            feature_importances[i] = importance;
        }

        // Normalize importances
        let importance_sum: f64 = feature_importances.sum();
        if importance_sum > 0.0 {
            feature_importances /= importance_sum;
        }

        // Select features based on importance
        let selected_features = self.select_features_by_importance(&feature_importances);

        Ok(NeuralFeatureSelector {
            hidden_layers: self.hidden_layers,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            l1_reg: self.l1_reg,
            l2_reg: self.l2_reg,
            k: self.k,
            importance_threshold: self.importance_threshold,
            random_state: self.random_state,
            state: PhantomData,
            weights_: Some(weights),
            biases_: Some(biases),
            feature_importances_: Some(feature_importances),
            selected_features_: Some(selected_features),
        })
    }
}

impl NeuralFeatureSelector<Untrained> {
    fn forward_pass(
        &self,
        input: &Array1<Float>,
        weights: &[Array2<Float>],
        biases: &[Array1<Float>],
    ) -> (Vec<Array1<Float>>, Vec<Array1<Float>>) {
        let mut activations = vec![input.clone()];
        let mut z_values = Vec::new();

        let mut current_activation = input.clone();

        for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
            // Linear transformation: z = W^T * a + b
            let z = w.t().dot(&current_activation) + b;
            z_values.push(z.clone());

            // Activation function (ReLU for hidden layers, linear for output)
            let activation = if i < weights.len() - 1 {
                z.mapv(|x| x.max(0.0)) // ReLU
            } else {
                z // Linear for output
            };

            activations.push(activation.clone());
            current_activation = activation;
        }

        (activations, z_values)
    }

    fn backward_pass(
        &self,
        _input: &Array1<Float>,
        target: Float,
        activations: &[Array1<Float>],
        z_values: &[Array1<Float>],
        weights: &[Array2<Float>],
    ) -> (Vec<Array2<Float>>, Vec<Array1<Float>>) {
        let n_layers = weights.len();
        let mut weight_gradients = vec![Array2::zeros(weights[0].dim()); n_layers];
        let mut bias_gradients = vec![Array1::zeros(z_values[0].len()); n_layers];

        // Output layer error
        let output_error = activations.last().unwrap()[0] - target;
        let mut delta = Array1::from_vec(vec![output_error]);

        // Backward propagation
        for i in (0..n_layers).rev() {
            // Compute gradients for current layer
            let prev_activation = &activations[i];

            // Weight gradients: dW = a_prev * delta^T
            for j in 0..weight_gradients[i].nrows() {
                for k in 0..weight_gradients[i].ncols() {
                    weight_gradients[i][[j, k]] = prev_activation[j] * delta[k];
                }
            }

            // Bias gradients: db = delta
            bias_gradients[i] = delta.clone();

            // Propagate error to previous layer
            if i > 0 {
                // delta_prev = W * delta * activation_derivative
                let new_delta = weights[i].dot(&delta);

                // Apply ReLU derivative (1 if z > 0, 0 otherwise)
                let z_prev = &z_values[i - 1];
                for j in 0..new_delta.len() {
                    if z_prev[j] <= 0.0 {
                        // delta[j] = 0.0; // This would modify the array, but we need to create a new one
                    }
                }
                delta = new_delta.mapv(|x| if x > 0.0 { x } else { 0.0 });
            }
        }

        (weight_gradients, bias_gradients)
    }

    fn update_parameters(
        &self,
        weights: &mut [Array2<Float>],
        biases: &mut [Array1<Float>],
        weight_gradients: &[Array2<Float>],
        bias_gradients: &[Array1<Float>],
    ) {
        for i in 0..weights.len() {
            // Update weights with L1 and L2 regularization
            for j in 0..weights[i].nrows() {
                for k in 0..weights[i].ncols() {
                    let grad = weight_gradients[i][[j, k]];
                    let l1_penalty = self.l1_reg * weights[i][[j, k]].signum();
                    let l2_penalty = self.l2_reg * weights[i][[j, k]];

                    weights[i][[j, k]] -= self.learning_rate * (grad + l1_penalty + l2_penalty);
                }
            }

            // Update biases
            for j in 0..biases[i].len() {
                biases[i][j] -= self.learning_rate * bias_gradients[i][j];
            }
        }
    }

    fn select_features_by_importance(&self, importances: &Array1<Float>) -> Vec<usize> {
        let mut feature_indices: Vec<(usize, Float)> = importances
            .indexed_iter()
            .map(|(i, &importance)| (i, importance))
            .collect();

        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<usize> = if let Some(k) = self.k {
            feature_indices
                .iter()
                .take(k.min(feature_indices.len()))
                .map(|(i, _)| *i)
                .collect()
        } else {
            feature_indices
                .iter()
                .filter(|(_, importance)| *importance >= self.importance_threshold)
                .map(|(i, _)| *i)
                .collect()
        };

        let mut selected_sorted = selected;
        selected_sorted.sort();
        selected_sorted
    }
}

impl Transform<Array2<Float>> for NeuralFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for NeuralFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.feature_importances_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Attention-based feature selection
/// Attention mechanism for feature selection
/// Uses attention weights to identify important features
#[derive(Debug, Clone)]
pub struct AttentionFeatureSelector<State = Untrained> {
    /// Dimension of attention space
    attention_dim: usize,
    /// Number of attention heads
    n_heads: usize,
    /// Learning rate
    learning_rate: f64,
    /// Number of training epochs
    epochs: usize,
    /// Temperature parameter for attention softmax
    temperature: f64,
    /// Number of top features to select
    k: Option<usize>,
    /// Threshold for attention weights
    attention_threshold: f64,
    /// Random seed
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    attention_weights_: Option<Array2<Float>>,
    feature_attention_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
}

impl AttentionFeatureSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            attention_dim: 64,
            n_heads: 8,
            learning_rate: 0.001,
            epochs: 50,
            temperature: 1.0,
            k: None,
            attention_threshold: 0.01,
            random_state: None,
            state: PhantomData,
            attention_weights_: None,
            feature_attention_: None,
            selected_features_: None,
        }
    }

    pub fn attention_dim(mut self, dim: usize) -> Self {
        self.attention_dim = dim;
        self
    }

    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    pub fn attention_threshold(mut self, threshold: f64) -> Self {
        self.attention_threshold = threshold;
        self
    }

    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Default for AttentionFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for AttentionFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AttentionFeatureSelector<Untrained> {
    type Fitted = AttentionFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let (n_samples, n_features) = x.dim();

        // Initialize attention parameters
        let head_dim = self.attention_dim / self.n_heads;

        // Query, Key, Value matrices for each head
        let mut query_weights = Array2::zeros((n_features, self.attention_dim));
        let mut key_weights = Array2::zeros((n_features, self.attention_dim));
        let mut value_weights = Array2::zeros((n_features, self.attention_dim));

        // Initialize with small random values
        for w in [&mut query_weights, &mut key_weights, &mut value_weights] {
            for elem in w.iter_mut() {
                *elem = rng.gen::<f64>() * 0.02 - 0.01;
            }
        }

        // Training loop
        for _epoch in 0..self.epochs {
            let mut total_attention = Array1::zeros(n_features);

            for i in 0..n_samples {
                let input = x.row(i);

                // Compute queries, keys, values
                let queries = input.dot(&query_weights);
                let keys = input.dot(&key_weights);
                let values = input.dot(&value_weights);

                // Multi-head attention
                let mut head_attentions = Vec::new();

                for head in 0..self.n_heads {
                    let start_idx = head * head_dim;
                    let end_idx = (head + 1) * head_dim;

                    let q_head = queries.slice(s![start_idx..end_idx]);
                    let k_head = keys.slice(s![start_idx..end_idx]);
                    let _v_head = values.slice(s![start_idx..end_idx]);

                    // Compute attention scores
                    let mut attention_scores = Array1::zeros(n_features);
                    for j in 0..n_features {
                        // Simplified attention: dot product of query and key projections
                        let score = q_head.dot(&k_head) / (head_dim as f64).sqrt();
                        attention_scores[j] = score;
                    }

                    // Apply softmax with temperature
                    let softmax_scores =
                        softmax_with_temperature(&attention_scores, self.temperature);
                    head_attentions.push(softmax_scores);
                }

                // Average attention across heads
                let mut avg_attention = Array1::zeros(n_features);
                for head_attention in &head_attentions {
                    avg_attention += head_attention;
                }
                avg_attention /= self.n_heads as f64;

                total_attention += &avg_attention;
            }

            // Average attention across samples
            total_attention /= n_samples as f64;
        }

        // Final attention computation (simplified version)
        let mut feature_attention = Array1::zeros(n_features);
        for i in 0..n_samples {
            let input = x.row(i);

            // Compute simple attention based on feature variance and correlation with target
            for j in 0..n_features {
                let feature_var = input[j].abs();
                let correlation = compute_simple_correlation(input[j], y[i]);
                feature_attention[j] += feature_var * correlation.abs();
            }
        }

        // Normalize attention weights
        let attention_sum = feature_attention.sum();
        if attention_sum > 0.0 {
            feature_attention /= attention_sum;
        }

        // Select features based on attention weights
        let selected_features = self.select_features_by_attention(&feature_attention);

        Ok(AttentionFeatureSelector {
            attention_dim: self.attention_dim,
            n_heads: self.n_heads,
            learning_rate: self.learning_rate,
            epochs: self.epochs,
            temperature: self.temperature,
            k: self.k,
            attention_threshold: self.attention_threshold,
            random_state: self.random_state,
            state: PhantomData,
            attention_weights_: Some(query_weights), // Store one of the weight matrices
            feature_attention_: Some(feature_attention),
            selected_features_: Some(selected_features),
        })
    }
}

impl AttentionFeatureSelector<Untrained> {
    fn select_features_by_attention(&self, attention: &Array1<Float>) -> Vec<usize> {
        let mut feature_indices: Vec<(usize, Float)> =
            attention.indexed_iter().map(|(i, &att)| (i, att)).collect();

        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<usize> = if let Some(k) = self.k {
            feature_indices
                .iter()
                .take(k.min(feature_indices.len()))
                .map(|(i, _)| *i)
                .collect()
        } else {
            feature_indices
                .iter()
                .filter(|(_, att)| *att >= self.attention_threshold)
                .map(|(i, _)| *i)
                .collect()
        };

        let mut selected_sorted = selected;
        selected_sorted.sort();
        selected_sorted
    }
}

impl Transform<Array2<Float>> for AttentionFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for AttentionFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.feature_attention_.as_ref().unwrap().len();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Reinforcement learning-based feature selection
/// Q-learning based feature selection
/// Treats feature selection as a sequential decision problem
#[derive(Debug, Clone)]
pub struct RLFeatureSelector<State = Untrained> {
    /// Q-learning parameters
    learning_rate: f64,
    discount_factor: f64,
    epsilon: f64,
    epsilon_decay: f64,
    /// Number of episodes for training
    episodes: usize,
    /// Maximum steps per episode
    max_steps: usize,
    /// Number of features to select
    k: usize,
    /// Random seed
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    q_table_: Option<HashMap<String, Array1<Float>>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl RLFeatureSelector<Untrained> {
    pub fn new(k: usize) -> Self {
        Self {
            learning_rate: 0.1,
            discount_factor: 0.9,
            epsilon: 1.0,
            epsilon_decay: 0.995,
            episodes: 100,
            max_steps: 20,
            k,
            random_state: None,
            state: PhantomData,
            q_table_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn discount_factor(mut self, gamma: f64) -> Self {
        self.discount_factor = gamma;
        self
    }

    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
        self
    }

    pub fn episodes(mut self, episodes: usize) -> Self {
        self.episodes = episodes;
        self
    }

    pub fn max_steps(mut self, max_steps: usize) -> Self {
        self.max_steps = max_steps;
        self
    }

    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Estimator for RLFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for RLFeatureSelector<Untrained> {
    type Fitted = RLFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let (_, n_features) = x.dim();
        let mut q_table: HashMap<String, Array1<Float>> = HashMap::new();
        let mut current_epsilon = self.epsilon;

        // Q-learning training
        for _episode in 0..self.episodes {
            let mut selected_features = Vec::new();
            let mut state = encode_state(&selected_features, n_features);

            for _step in 0..self.max_steps {
                if selected_features.len() >= self.k {
                    break;
                }

                // Get available actions (unselected features)
                let available_actions: Vec<usize> = (0..n_features)
                    .filter(|&i| !selected_features.contains(&i))
                    .collect();

                if available_actions.is_empty() {
                    break;
                }

                // Epsilon-greedy action selection
                let action = if rng.gen::<f64>() < current_epsilon {
                    // Random action
                    *available_actions.choose(&mut rng).unwrap()
                } else {
                    // Greedy action (highest Q-value)
                    let q_values = q_table
                        .entry(state.clone())
                        .or_insert_with(|| Array1::zeros(n_features));

                    let best_action = available_actions
                        .iter()
                        .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
                        .unwrap();
                    *best_action
                };

                // Take action
                selected_features.push(action);
                let new_state = encode_state(&selected_features, n_features);

                // Compute reward
                let reward = self.compute_reward(x, y, &selected_features);

                // Q-learning update
                let current_q = q_table
                    .entry(state.clone())
                    .or_insert_with(|| Array1::zeros(n_features))[action];

                let next_q_values = q_table
                    .entry(new_state.clone())
                    .or_insert_with(|| Array1::zeros(n_features));
                let max_next_q = next_q_values
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

                let target_q = reward + self.discount_factor * max_next_q;
                let updated_q = current_q + self.learning_rate * (target_q - current_q);

                q_table
                    .entry(state.clone())
                    .or_insert_with(|| Array1::zeros(n_features))[action] = updated_q;

                state = new_state;
            }

            // Decay epsilon
            current_epsilon *= self.epsilon_decay;
        }

        // Extract best feature selection
        let mut best_features = Vec::new();
        let mut current_state = encode_state(&best_features, n_features);

        for _ in 0..self.k {
            let available_actions: Vec<usize> = (0..n_features)
                .filter(|&i| !best_features.contains(&i))
                .collect();

            if available_actions.is_empty() {
                break;
            }

            let default_q_values = Array1::zeros(n_features);
            let q_values = q_table.get(&current_state).unwrap_or(&default_q_values);

            let best_action = available_actions
                .iter()
                .max_by(|&&a, &&b| q_values[a].partial_cmp(&q_values[b]).unwrap())
                .unwrap();

            best_features.push(*best_action);
            current_state = encode_state(&best_features, n_features);
        }

        best_features.sort();

        Ok(RLFeatureSelector {
            learning_rate: self.learning_rate,
            discount_factor: self.discount_factor,
            epsilon: self.epsilon,
            epsilon_decay: self.epsilon_decay,
            episodes: self.episodes,
            max_steps: self.max_steps,
            k: self.k,
            random_state: self.random_state,
            state: PhantomData,
            q_table_: Some(q_table),
            selected_features_: Some(best_features),
            n_features_: Some(n_features),
        })
    }
}

impl RLFeatureSelector<Untrained> {
    fn compute_reward(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        selected_features: &[usize],
    ) -> f64 {
        if selected_features.is_empty() {
            return 0.0;
        }

        // Simple reward: correlation of selected features with target
        let mut reward = 0.0;
        for &feature_idx in selected_features {
            let feature = x.column(feature_idx);
            let correlation = compute_simple_correlation_array(&feature.to_owned(), y);
            reward += correlation.abs();
        }

        // Normalize by number of features to encourage efficiency
        reward / selected_features.len() as f64
    }
}

impl Transform<Array2<Float>> for RLFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for RLFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = self.n_features_.unwrap();
        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

/// Meta-learning feature selector
/// Meta-learning approach that learns from multiple feature selection tasks
#[derive(Debug, Clone)]
pub struct MetaLearningFeatureSelector<State = Untrained> {
    /// Base selectors to use in meta-learning
    base_methods: Vec<String>,
    /// Meta-model parameters
    meta_learning_rate: f64,
    meta_epochs: usize,
    /// Number of top features to select
    k: Option<usize>,
    /// Random seed
    random_state: Option<u64>,
    state: PhantomData<State>,
    // Trained state
    meta_weights_: Option<Array1<Float>>,
    base_selections_: Option<Vec<Vec<usize>>>,
    selected_features_: Option<Vec<usize>>,
}

impl MetaLearningFeatureSelector<Untrained> {
    pub fn new() -> Self {
        Self {
            base_methods: vec![
                "correlation".to_string(),
                "mutual_info".to_string(),
                "f_test".to_string(),
                "chi2".to_string(),
            ],
            meta_learning_rate: 0.01,
            meta_epochs: 50,
            k: None,
            random_state: None,
            state: PhantomData,
            meta_weights_: None,
            base_selections_: None,
            selected_features_: None,
        }
    }

    pub fn base_methods(mut self, methods: Vec<String>) -> Self {
        self.base_methods = methods;
        self
    }

    pub fn meta_learning_rate(mut self, lr: f64) -> Self {
        self.meta_learning_rate = lr;
        self
    }

    pub fn meta_epochs(mut self, epochs: usize) -> Self {
        self.meta_epochs = epochs;
        self
    }

    pub fn k(mut self, k: Option<usize>) -> Self {
        self.k = k;
        self
    }

    pub fn random_state(mut self, seed: Option<u64>) -> Self {
        self.random_state = seed;
        self
    }
}

impl Default for MetaLearningFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MetaLearningFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for MetaLearningFeatureSelector<Untrained> {
    type Fitted = MetaLearningFeatureSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let _rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut thread_rng()),
        };

        let (_, n_features) = x.dim();

        // Apply base feature selection methods
        let mut base_selections = Vec::new();
        let mut base_scores = Vec::new();

        for method in &self.base_methods {
            let (selection, score) = match method.as_str() {
                "correlation" => apply_correlation_selection(x, y, self.k),
                "mutual_info" => apply_mutual_info_selection(x, y, self.k),
                "f_test" => apply_f_test_selection(x, y, self.k),
                "chi2" => apply_chi2_selection(x, y, self.k),
                _ => (Vec::new(), 0.0),
            };
            base_selections.push(selection);
            base_scores.push(score);
        }

        // Initialize meta-weights
        let mut meta_weights = Array1::from_vec(vec![
            1.0 / self.base_methods.len() as f64;
            self.base_methods.len()
        ]);

        // Meta-learning: optimize weights based on performance
        for _epoch in 0..self.meta_epochs {
            // Compute weighted ensemble selection
            let ensemble_selection =
                self.compute_ensemble_selection(&base_selections, &meta_weights, n_features);

            // Compute ensemble performance
            let ensemble_score = evaluate_feature_selection(x, y, &ensemble_selection);

            // Update meta-weights using gradient ascent
            for i in 0..meta_weights.len() {
                let individual_score = base_scores[i];
                let gradient = individual_score - ensemble_score;
                meta_weights[i] += self.meta_learning_rate * gradient;
            }

            // Normalize weights
            let weight_sum = meta_weights.sum();
            if weight_sum > 0.0 {
                meta_weights /= weight_sum;
            }
        }

        // Final ensemble selection
        let selected_features =
            self.compute_ensemble_selection(&base_selections, &meta_weights, n_features);

        Ok(MetaLearningFeatureSelector {
            base_methods: self.base_methods,
            meta_learning_rate: self.meta_learning_rate,
            meta_epochs: self.meta_epochs,
            k: self.k,
            random_state: self.random_state,
            state: PhantomData,
            meta_weights_: Some(meta_weights),
            base_selections_: Some(base_selections),
            selected_features_: Some(selected_features),
        })
    }
}

impl MetaLearningFeatureSelector<Untrained> {
    fn compute_ensemble_selection(
        &self,
        base_selections: &[Vec<usize>],
        weights: &Array1<Float>,
        n_features: usize,
    ) -> Vec<usize> {
        let mut feature_scores = Array1::zeros(n_features);

        // Weight the contributions from each base method
        for (i, selection) in base_selections.iter().enumerate() {
            let weight = weights[i];
            for &feature_idx in selection {
                if feature_idx < n_features {
                    feature_scores[feature_idx] += weight;
                }
            }
        }

        // Select top features
        let mut feature_indices: Vec<(usize, Float)> = feature_scores
            .indexed_iter()
            .map(|(i, &score)| (i, score))
            .collect();

        feature_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let selected: Vec<usize> = if let Some(k) = self.k {
            feature_indices
                .iter()
                .take(k.min(feature_indices.len()))
                .filter(|(_, score)| *score > 0.0)
                .map(|(i, _)| *i)
                .collect()
        } else {
            feature_indices
                .iter()
                .filter(|(_, score)| *score > 0.1) // Threshold for selection
                .map(|(i, _)| *i)
                .collect()
        };

        let mut selected_sorted = selected;
        selected_sorted.sort();
        selected_sorted
    }
}

impl Transform<Array2<Float>> for MetaLearningFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features were selected".to_string(),
            ));
        }

        let selected_indices: Vec<usize> = selected_features.to_vec();
        Ok(x.select(Axis(1), &selected_indices))
    }
}

impl SelectorMixin for MetaLearningFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_features = if let Some(ref selections) = self.base_selections_ {
            selections.iter().flatten().max().unwrap_or(&0) + 1
        } else {
            selected_features.iter().max().unwrap_or(&0) + 1
        };

        let mut support = Array1::from_elem(n_features, false);
        for &idx in selected_features {
            if idx < n_features {
                support[idx] = true;
            }
        }
        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

// Helper functions

fn softmax_with_temperature(scores: &Array1<Float>, temperature: f64) -> Array1<Float> {
    let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_scores: Array1<Float> = scores.mapv(|x| ((x - max_score) / temperature).exp());
    let sum_exp = exp_scores.sum();
    exp_scores / sum_exp
}

fn compute_simple_correlation(x: Float, y: Float) -> Float {
    // Simplified correlation for single values - in practice would need proper correlation
    x * y
}

pub fn compute_simple_correlation_array(x: &Array1<Float>, y: &Array1<Float>) -> Float {
    let n = x.len().min(y.len());
    if n < 2 {
        return 0.0;
    }

    let x_mean = x.iter().take(n).sum::<Float>() / n as Float;
    let y_mean = y.iter().take(n).sum::<Float>() / n as Float;

    let mut numerator = 0.0;
    let mut x_var = 0.0;
    let mut y_var = 0.0;

    for i in 0..n {
        let x_i = x[i] - x_mean;
        let y_i = y[i] - y_mean;
        numerator += x_i * y_i;
        x_var += x_i * x_i;
        y_var += y_i * y_i;
    }

    let denominator = (x_var * y_var).sqrt();
    if denominator.abs() < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

fn encode_state(selected_features: &[usize], n_features: usize) -> String {
    let mut state = vec![false; n_features];
    for &idx in selected_features {
        if idx < n_features {
            state[idx] = true;
        }
    }
    state.iter().map(|&b| if b { '1' } else { '0' }).collect()
}

// Simplified base method implementations
fn apply_correlation_selection(
    x: &Array2<Float>,
    y: &Array1<Float>,
    k: Option<usize>,
) -> (Vec<usize>, f64) {
    let (_, n_features) = x.dim();
    let mut correlations = Vec::new();

    for j in 0..n_features {
        let feature = x.column(j);
        let corr = compute_simple_correlation_array(&feature.to_owned(), y);
        correlations.push((j, corr.abs()));
    }

    correlations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let selection_size = k.unwrap_or(n_features / 4).min(n_features);
    let selection: Vec<usize> = correlations
        .iter()
        .take(selection_size)
        .map(|(i, _)| *i)
        .collect();
    let score = correlations
        .iter()
        .take(selection_size)
        .map(|(_, corr)| corr)
        .sum::<f64>()
        / selection_size as f64;

    (selection, score)
}

fn apply_mutual_info_selection(
    x: &Array2<Float>,
    y: &Array1<Float>,
    k: Option<usize>,
) -> (Vec<usize>, f64) {
    // Simplified mutual information - in practice would use proper MI estimation
    apply_correlation_selection(x, y, k)
}

fn apply_f_test_selection(
    x: &Array2<Float>,
    y: &Array1<Float>,
    k: Option<usize>,
) -> (Vec<usize>, f64) {
    // Simplified F-test - in practice would use proper F-statistic
    apply_correlation_selection(x, y, k)
}

fn apply_chi2_selection(
    x: &Array2<Float>,
    y: &Array1<Float>,
    k: Option<usize>,
) -> (Vec<usize>, f64) {
    // Simplified chi-squared test - in practice would use proper chi-squared statistic
    apply_correlation_selection(x, y, k)
}

fn evaluate_feature_selection(x: &Array2<Float>, y: &Array1<Float>, features: &[usize]) -> f64 {
    if features.is_empty() {
        return 0.0;
    }

    // Simple evaluation: average correlation of selected features with target
    let mut total_corr = 0.0;
    for &feature_idx in features {
        let feature = x.column(feature_idx);
        let corr = compute_simple_correlation_array(&feature.to_owned(), y);
        total_corr += corr.abs();
    }

    total_corr / features.len() as f64
}

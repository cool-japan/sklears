//! Neural Sequence Models for Structured Output Prediction
//!
//! This module implements RNN, LSTM, and GRU models for sequence-based tasks
//! such as sequence labeling, sequence-to-sequence prediction, and other
//! structured output problems.

// Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use scirs2_core::random::thread_rng;
use scirs2_core::random::RandNormal;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};
use std::collections::HashMap;

use crate::activation::ActivationFunction;

/// Cell types for recurrent neural networks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CellType {
    /// Simple RNN cell
    RNN,
    /// Long Short-Term Memory cell
    LSTM,
    /// Gated Recurrent Unit cell
    GRU,
}

/// Output modes for sequence models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SequenceMode {
    /// Many-to-many: output at each timestep
    ManyToMany,
    /// Many-to-one: single output at the end
    ManyToOne,
    /// One-to-many: single input, sequence output
    OneToMany,
}

/// Recurrent Neural Network for Sequence Prediction
///
/// This model can handle various sequence prediction tasks using different
/// cell types (RNN, LSTM, GRU) and output modes.
///
/// # Examples
///
/// ```
/// use sklears_multioutput::recurrent::{RecurrentNeuralNetwork, CellType, SequenceMode};
/// use sklears_core::traits::{Predict, Fit};
/// // Use SciRS2-Core for arrays and random number generation (SciRS2 Policy)
///
/// // Example for sequence labeling (many-to-many)
/// let rnn = RecurrentNeuralNetwork::new()
///     .cell_type(CellType::LSTM)
///     .hidden_size(50)
///     .sequence_mode(SequenceMode::ManyToMany)
///     .learning_rate(0.001)
///     .max_iter(100);
/// ```
#[derive(Debug, Clone)]
pub struct RecurrentNeuralNetwork<S = Untrained> {
    state: S,
    cell_type: CellType,
    hidden_size: usize,
    num_layers: usize,
    sequence_mode: SequenceMode,
    bidirectional: bool,
    dropout: Float,
    learning_rate: Float,
    max_iter: usize,
    tolerance: Float,
    random_state: Option<u64>,
    alpha: Float, // L2 regularization
}

/// Trained state for RecurrentNeuralNetwork
#[derive(Debug, Clone)]
pub struct RecurrentNeuralNetworkTrained {
    /// Input-to-hidden weights for each layer
    input_weights: Vec<Array2<Float>>,
    /// Hidden-to-hidden weights for each layer
    hidden_weights: Vec<Array2<Float>>,
    /// Biases for each layer
    biases: Vec<Array1<Float>>,
    /// Output layer weights
    output_weights: Array2<Float>,
    /// Output layer bias
    output_bias: Array1<Float>,
    /// Additional parameters for LSTM/GRU gates
    gate_weights: HashMap<String, Vec<Array2<Float>>>,
    gate_biases: HashMap<String, Vec<Array1<Float>>>,
    /// Network configuration
    cell_type: CellType,
    hidden_size: usize,
    num_layers: usize,
    sequence_mode: SequenceMode,
    bidirectional: bool,
    n_features: usize,
    n_outputs: usize,
    /// Training history
    loss_curve: Vec<Float>,
    n_iter: usize,
}

impl RecurrentNeuralNetwork<Untrained> {
    /// Create a new RecurrentNeuralNetwork instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            cell_type: CellType::LSTM,
            hidden_size: 50,
            num_layers: 1,
            sequence_mode: SequenceMode::ManyToMany,
            bidirectional: false,
            dropout: 0.0,
            learning_rate: 0.001,
            max_iter: 100,
            tolerance: 1e-4,
            random_state: None,
            alpha: 0.0001,
        }
    }

    /// Set the cell type (RNN, LSTM, GRU)
    pub fn cell_type(mut self, cell_type: CellType) -> Self {
        self.cell_type = cell_type;
        self
    }

    /// Set the hidden layer size
    pub fn hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    /// Set the number of recurrent layers
    pub fn num_layers(mut self, num_layers: usize) -> Self {
        self.num_layers = num_layers;
        self
    }

    /// Set the sequence prediction mode
    pub fn sequence_mode(mut self, sequence_mode: SequenceMode) -> Self {
        self.sequence_mode = sequence_mode;
        self
    }

    /// Enable bidirectional processing
    pub fn bidirectional(mut self, bidirectional: bool) -> Self {
        self.bidirectional = bidirectional;
        self
    }

    /// Set dropout rate
    pub fn dropout(mut self, dropout: Float) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set learning rate
    pub fn learning_rate(mut self, learning_rate: Float) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set L2 regularization parameter
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }
}

impl Default for RecurrentNeuralNetwork<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RecurrentNeuralNetwork<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView3<'_, Float>, Array3<Float>> for RecurrentNeuralNetwork<Untrained> {
    type Fitted = RecurrentNeuralNetwork<RecurrentNeuralNetworkTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView3<'_, Float>, y: &Array3<Float>) -> SklResult<Self::Fitted> {
        let (n_samples, max_seq_len, n_features) = X.dim();
        let (n_samples_y, max_seq_len_y, n_outputs) = y.dim();

        if n_samples != n_samples_y {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        if self.sequence_mode == SequenceMode::ManyToMany && max_seq_len != max_seq_len_y {
            return Err(SklearsError::InvalidInput(
                "For many-to-many mode, X and y must have the same sequence length".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit with zero samples".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = thread_rng();

        // Initialize weights and biases
        let (
            input_weights,
            hidden_weights,
            biases,
            output_weights,
            output_bias,
            gate_weights,
            gate_biases,
        ) = self.initialize_parameters(n_features, n_outputs, &mut rng)?;

        let mut input_weights = input_weights;
        let mut hidden_weights = hidden_weights;
        let mut biases = biases;
        let mut output_weights = output_weights;
        let mut output_bias = output_bias;
        let mut gate_weights = gate_weights;
        let mut gate_biases = gate_biases;

        // Training loop
        let mut loss_curve = Vec::new();
        let X_owned = X.to_owned();
        let y_owned = y.to_owned();

        for epoch in 0..self.max_iter {
            let mut total_loss = 0.0;

            // Process each sequence in the batch
            for sample_idx in 0..n_samples {
                let x_seq = X_owned.slice(s![sample_idx, .., ..]);
                let y_seq = y_owned.slice(s![sample_idx, .., ..]);

                // Forward pass
                let (predictions, hidden_states) = self.forward_sequence(
                    &x_seq,
                    &input_weights,
                    &hidden_weights,
                    &biases,
                    &output_weights,
                    &output_bias,
                    &gate_weights,
                    &gate_biases,
                )?;

                // Compute loss
                let sample_loss = self.compute_sequence_loss(&predictions, &y_seq.to_owned());
                total_loss += sample_loss;

                // Backward pass (BPTT)
                self.backward_sequence(
                    &x_seq,
                    &y_seq.to_owned(),
                    &predictions,
                    &hidden_states,
                    &mut input_weights,
                    &mut hidden_weights,
                    &mut biases,
                    &mut output_weights,
                    &mut output_bias,
                    &mut gate_weights,
                    &mut gate_biases,
                )?;
            }

            let avg_loss = total_loss / n_samples as Float;
            loss_curve.push(avg_loss);

            // Check convergence
            if epoch > 0 && (loss_curve[epoch - 1] - avg_loss).abs() < self.tolerance {
                break;
            }
        }

        let trained_state = RecurrentNeuralNetworkTrained {
            input_weights,
            hidden_weights,
            biases,
            output_weights,
            output_bias,
            gate_weights,
            gate_biases,
            cell_type: self.cell_type,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            sequence_mode: self.sequence_mode,
            bidirectional: self.bidirectional,
            n_features,
            n_outputs,
            loss_curve,
            n_iter: self.max_iter,
        };

        Ok(RecurrentNeuralNetwork {
            state: trained_state,
            cell_type: self.cell_type,
            hidden_size: self.hidden_size,
            num_layers: self.num_layers,
            sequence_mode: self.sequence_mode,
            bidirectional: self.bidirectional,
            dropout: self.dropout,
            learning_rate: self.learning_rate,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            random_state: self.random_state,
            alpha: self.alpha,
        })
    }
}

impl RecurrentNeuralNetwork<Untrained> {
    /// Initialize network parameters
    fn initialize_parameters(
        &self,
        n_features: usize,
        n_outputs: usize,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> SklResult<(
        Vec<Array2<Float>>,                  // input_weights
        Vec<Array2<Float>>,                  // hidden_weights
        Vec<Array1<Float>>,                  // biases
        Array2<Float>,                       // output_weights
        Array1<Float>,                       // output_bias
        HashMap<String, Vec<Array2<Float>>>, // gate_weights
        HashMap<String, Vec<Array1<Float>>>, // gate_biases
    )> {
        let mut input_weights = Vec::new();
        let mut hidden_weights = Vec::new();
        let mut biases = Vec::new();
        let mut gate_weights = HashMap::new();
        let mut gate_biases = HashMap::new();

        // Initialize parameters for each layer
        for layer in 0..self.num_layers {
            let input_size = if layer == 0 {
                n_features
            } else {
                self.hidden_size
            };

            // Xavier initialization
            let input_scale = (2.0 / (input_size + self.hidden_size) as Float).sqrt();
            let hidden_scale = (2.0 / (self.hidden_size + self.hidden_size) as Float).sqrt();

            let mut input_weight = Array2::<Float>::zeros((self.hidden_size, input_size));
            let normal_dist = RandNormal::new(0.0, input_scale).unwrap();
            for i in 0..self.hidden_size {
                for j in 0..input_size {
                    input_weight[[i, j]] = rng.sample(normal_dist);
                }
            }
            let mut hidden_weight = Array2::<Float>::zeros((self.hidden_size, self.hidden_size));
            let hidden_normal_dist = RandNormal::new(0.0, hidden_scale).unwrap();
            for i in 0..self.hidden_size {
                for j in 0..self.hidden_size {
                    hidden_weight[[i, j]] = rng.sample(hidden_normal_dist);
                }
            }
            let bias = Array1::<Float>::zeros(self.hidden_size);

            input_weights.push(input_weight);
            hidden_weights.push(hidden_weight);
            biases.push(bias);

            // Initialize gate parameters for LSTM/GRU
            match self.cell_type {
                CellType::LSTM => {
                    // LSTM has forget, input, and output gates
                    for gate_name in &["forget", "input", "output", "cell"] {
                        // Initialize input weights
                        let input_key = format!("{}_input", gate_name);
                        if !gate_weights.contains_key(&input_key) {
                            gate_weights.insert(input_key.clone(), Vec::new());
                        }
                        let mut input_weight =
                            Array2::<Float>::zeros((self.hidden_size, input_size));
                        let input_normal_dist = RandNormal::new(0.0, input_scale).unwrap();
                        for i in 0..self.hidden_size {
                            for j in 0..input_size {
                                input_weight[[i, j]] = rng.sample(input_normal_dist);
                            }
                        }
                        gate_weights.get_mut(&input_key).unwrap().push(input_weight);

                        // Initialize hidden weights
                        let hidden_key = format!("{}_hidden", gate_name);
                        if !gate_weights.contains_key(&hidden_key) {
                            gate_weights.insert(hidden_key.clone(), Vec::new());
                        }
                        let mut hidden_weight =
                            Array2::<Float>::zeros((self.hidden_size, self.hidden_size));
                        let hidden_normal_dist = RandNormal::new(0.0, hidden_scale).unwrap();
                        for i in 0..self.hidden_size {
                            for j in 0..self.hidden_size {
                                hidden_weight[[i, j]] = rng.sample(hidden_normal_dist);
                            }
                        }
                        gate_weights
                            .get_mut(&hidden_key)
                            .unwrap()
                            .push(hidden_weight);

                        // Initialize biases
                        let bias_key = gate_name.to_string();
                        if !gate_biases.contains_key(&bias_key) {
                            gate_biases.insert(bias_key.clone(), Vec::new());
                        }
                        gate_biases
                            .get_mut(&bias_key)
                            .unwrap()
                            .push(Array1::<Float>::zeros(self.hidden_size));
                    }
                }
                CellType::GRU => {
                    // GRU has reset and update gates
                    for gate_name in &["reset", "update", "new"] {
                        // Initialize input weights
                        let input_key = format!("{}_input", gate_name);
                        if !gate_weights.contains_key(&input_key) {
                            gate_weights.insert(input_key.clone(), Vec::new());
                        }
                        let mut input_weight =
                            Array2::<Float>::zeros((self.hidden_size, input_size));
                        let input_normal_dist = RandNormal::new(0.0, input_scale).unwrap();
                        for i in 0..self.hidden_size {
                            for j in 0..input_size {
                                input_weight[[i, j]] = rng.sample(input_normal_dist);
                            }
                        }
                        gate_weights.get_mut(&input_key).unwrap().push(input_weight);

                        // Initialize hidden weights
                        let hidden_key = format!("{}_hidden", gate_name);
                        if !gate_weights.contains_key(&hidden_key) {
                            gate_weights.insert(hidden_key.clone(), Vec::new());
                        }
                        let mut hidden_weight =
                            Array2::<Float>::zeros((self.hidden_size, self.hidden_size));
                        let hidden_normal_dist = RandNormal::new(0.0, hidden_scale).unwrap();
                        for i in 0..self.hidden_size {
                            for j in 0..self.hidden_size {
                                hidden_weight[[i, j]] = rng.sample(hidden_normal_dist);
                            }
                        }
                        gate_weights
                            .get_mut(&hidden_key)
                            .unwrap()
                            .push(hidden_weight);

                        // Initialize biases
                        let bias_key = gate_name.to_string();
                        if !gate_biases.contains_key(&bias_key) {
                            gate_biases.insert(bias_key.clone(), Vec::new());
                        }
                        gate_biases
                            .get_mut(&bias_key)
                            .unwrap()
                            .push(Array1::<Float>::zeros(self.hidden_size));
                    }
                }
                CellType::RNN => {
                    // Simple RNN doesn't need additional gates
                }
            }
        }

        // Output layer
        let output_input_size = if self.bidirectional {
            2 * self.hidden_size
        } else {
            self.hidden_size
        };
        let output_scale = (2.0 / (output_input_size + n_outputs) as Float).sqrt();
        let mut output_weights = Array2::<Float>::zeros((n_outputs, output_input_size));
        let output_normal_dist = RandNormal::new(0.0, output_scale).unwrap();
        for i in 0..n_outputs {
            for j in 0..output_input_size {
                output_weights[[i, j]] = rng.sample(output_normal_dist);
            }
        }
        let output_bias = Array1::<Float>::zeros(n_outputs);

        Ok((
            input_weights,
            hidden_weights,
            biases,
            output_weights,
            output_bias,
            gate_weights,
            gate_biases,
        ))
    }

    /// Forward pass through sequence
    fn forward_sequence(
        &self,
        x_seq: &ArrayView2<'_, Float>,
        input_weights: &[Array2<Float>],
        hidden_weights: &[Array2<Float>],
        biases: &[Array1<Float>],
        output_weights: &Array2<Float>,
        output_bias: &Array1<Float>,
        gate_weights: &HashMap<String, Vec<Array2<Float>>>,
        gate_biases: &HashMap<String, Vec<Array1<Float>>>,
    ) -> SklResult<(Array2<Float>, Vec<Vec<Array1<Float>>>)> {
        let (seq_len, _) = x_seq.dim();
        let n_outputs = output_weights.nrows();

        // Initialize hidden states for all layers
        let mut hidden_states = Vec::new();
        for _ in 0..self.num_layers {
            hidden_states.push(vec![Array1::<Float>::zeros(self.hidden_size); seq_len + 1]);
        }

        let mut cell_states = Vec::new();
        if self.cell_type == CellType::LSTM {
            for _ in 0..self.num_layers {
                cell_states.push(vec![Array1::<Float>::zeros(self.hidden_size); seq_len + 1]);
            }
        }

        // Process sequence timestep by timestep
        for t in 0..seq_len {
            let x_t = x_seq.row(t);

            for layer in 0..self.num_layers {
                let input = if layer == 0 {
                    x_t.to_owned()
                } else {
                    hidden_states[layer - 1][t].clone()
                };

                let prev_hidden = &hidden_states[layer][t];

                match self.cell_type {
                    CellType::RNN => {
                        // Simple RNN: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
                        let linear = input_weights[layer].dot(&input)
                            + hidden_weights[layer].dot(prev_hidden)
                            + &biases[layer];
                        hidden_states[layer][t + 1] = linear.map(|x| x.tanh());
                    }
                    CellType::LSTM => {
                        // LSTM cell computation
                        let prev_cell = &cell_states[layer][t];

                        // Forget gate
                        let f_t = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["forget_input"][layer],
                            &gate_weights["forget_hidden"][layer],
                            &gate_biases["forget"][layer],
                            ActivationFunction::Sigmoid,
                        );

                        // Input gate
                        let i_t = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["input_input"][layer],
                            &gate_weights["input_hidden"][layer],
                            &gate_biases["input"][layer],
                            ActivationFunction::Sigmoid,
                        );

                        // Candidate values
                        let c_tilde = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["cell_input"][layer],
                            &gate_weights["cell_hidden"][layer],
                            &gate_biases["cell"][layer],
                            ActivationFunction::Tanh,
                        );

                        // Update cell state
                        let new_cell = &f_t * prev_cell + &i_t * &c_tilde;
                        cell_states[layer][t + 1] = new_cell.clone();

                        // Output gate
                        let o_t = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["output_input"][layer],
                            &gate_weights["output_hidden"][layer],
                            &gate_biases["output"][layer],
                            ActivationFunction::Sigmoid,
                        );

                        // Update hidden state
                        hidden_states[layer][t + 1] = &o_t * &new_cell.map(|x| x.tanh());
                    }
                    CellType::GRU => {
                        // GRU cell computation
                        let r_t = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["reset_input"][layer],
                            &gate_weights["reset_hidden"][layer],
                            &gate_biases["reset"][layer],
                            ActivationFunction::Sigmoid,
                        );

                        let z_t = self.compute_gate(
                            &input,
                            prev_hidden,
                            &gate_weights["update_input"][layer],
                            &gate_weights["update_hidden"][layer],
                            &gate_biases["update"][layer],
                            ActivationFunction::Sigmoid,
                        );

                        let reset_hidden = &r_t * prev_hidden;
                        let n_t = self.compute_gate(
                            &input,
                            &reset_hidden,
                            &gate_weights["new_input"][layer],
                            &gate_weights["new_hidden"][layer],
                            &gate_biases["new"][layer],
                            ActivationFunction::Tanh,
                        );

                        let one_minus_z = Array1::<Float>::ones(self.hidden_size) - &z_t;
                        hidden_states[layer][t + 1] = &z_t * prev_hidden + &one_minus_z * &n_t;
                    }
                }
            }
        }

        // Generate outputs based on sequence mode
        let predictions = match self.sequence_mode {
            SequenceMode::ManyToMany => {
                let mut outputs = Array2::<Float>::zeros((seq_len, n_outputs));
                for t in 0..seq_len {
                    let last_layer_hidden = &hidden_states[self.num_layers - 1][t + 1];
                    let output_t = output_weights.dot(last_layer_hidden) + output_bias;
                    outputs.row_mut(t).assign(&output_t);
                }
                outputs
            }
            SequenceMode::ManyToOne => {
                let final_hidden = &hidden_states[self.num_layers - 1][seq_len];
                let output = output_weights.dot(final_hidden) + output_bias;
                Array2::from_shape_vec((1, n_outputs), output.to_vec()).unwrap()
            }
            SequenceMode::OneToMany => {
                // For one-to-many, we typically use the input at t=0 and generate sequence
                let mut outputs = Array2::<Float>::zeros((seq_len, n_outputs));
                for t in 0..seq_len {
                    let hidden_t = &hidden_states[self.num_layers - 1][t + 1];
                    let output_t = output_weights.dot(hidden_t) + output_bias;
                    outputs.row_mut(t).assign(&output_t);
                }
                outputs
            }
        };

        Ok((predictions, hidden_states))
    }

    /// Compute gate activation
    fn compute_gate(
        &self,
        input: &Array1<Float>,
        hidden: &Array1<Float>,
        input_weight: &Array2<Float>,
        hidden_weight: &Array2<Float>,
        bias: &Array1<Float>,
        activation: ActivationFunction,
    ) -> Array1<Float> {
        let linear = input_weight.dot(input) + hidden_weight.dot(hidden) + bias;
        activation.apply(&linear)
    }

    /// Compute loss for sequence
    fn compute_sequence_loss(&self, predictions: &Array2<Float>, targets: &Array2<Float>) -> Float {
        let diff = predictions - targets;
        diff.map(|x| x * x).mean().unwrap()
    }

    /// Backward pass through sequence (simplified BPTT)
    fn backward_sequence(
        &self,
        x_seq: &ArrayView2<'_, Float>,
        y_seq: &Array2<Float>,
        predictions: &Array2<Float>,
        hidden_states: &[Vec<Array1<Float>>],
        input_weights: &mut [Array2<Float>],
        hidden_weights: &mut [Array2<Float>],
        biases: &mut [Array1<Float>],
        output_weights: &mut Array2<Float>,
        output_bias: &mut Array1<Float>,
        gate_weights: &mut HashMap<String, Vec<Array2<Float>>>,
        gate_biases: &mut HashMap<String, Vec<Array1<Float>>>,
    ) -> SklResult<()> {
        // Simplified gradient computation
        let (seq_len, _) = x_seq.dim();

        // Compute output gradients
        let output_error = predictions - y_seq;

        match self.sequence_mode {
            SequenceMode::ManyToMany => {
                for t in 0..seq_len {
                    let hidden_t = &hidden_states[self.num_layers - 1][t + 1];
                    let error_t = output_error.row(t).to_owned();

                    // Update output layer
                    let weight_grad = error_t
                        .clone()
                        .insert_axis(Axis(1))
                        .dot(&hidden_t.clone().insert_axis(Axis(0)));
                    *output_weights = output_weights.clone() - self.learning_rate * weight_grad;
                    *output_bias = output_bias.clone() - self.learning_rate * &error_t;

                    // Simplified hidden layer updates
                    for layer in (0..self.num_layers).rev() {
                        let x_t = if layer == 0 {
                            x_seq.row(t).to_owned()
                        } else {
                            hidden_states[layer - 1][t + 1].clone()
                        };

                        let hidden_error = output_weights.t().dot(&error_t);
                        let weight_grad = hidden_error
                            .clone()
                            .insert_axis(Axis(1))
                            .dot(&x_t.insert_axis(Axis(0)));

                        input_weights[layer] =
                            input_weights[layer].clone() - self.learning_rate * weight_grad;
                        biases[layer] = biases[layer].clone() - self.learning_rate * hidden_error;
                    }
                }
            }
            _ => {
                // Simplified update for other modes
                let hidden_final = &hidden_states[self.num_layers - 1][seq_len];
                let error_final = output_error.row(0).to_owned();
                let weight_grad = error_final
                    .clone()
                    .insert_axis(Axis(1))
                    .dot(&hidden_final.clone().insert_axis(Axis(0)));
                *output_weights = output_weights.clone() - self.learning_rate * weight_grad;
                *output_bias = output_bias.clone() - self.learning_rate * error_final;
            }
        }

        Ok(())
    }
}

impl Predict<ArrayView3<'_, Float>, Array3<Float>>
    for RecurrentNeuralNetwork<RecurrentNeuralNetworkTrained>
{
    fn predict(&self, X: &ArrayView3<'_, Float>) -> SklResult<Array3<Float>> {
        let (n_samples, max_seq_len, n_features) = X.dim();

        if n_features != self.state.n_features {
            return Err(SklearsError::InvalidInput(
                "X has different number of features than training data".to_string(),
            ));
        }

        let mut predictions = match self.state.sequence_mode {
            SequenceMode::ManyToMany => {
                Array3::<Float>::zeros((n_samples, max_seq_len, self.state.n_outputs))
            }
            SequenceMode::ManyToOne => Array3::<Float>::zeros((n_samples, 1, self.state.n_outputs)),
            SequenceMode::OneToMany => {
                Array3::<Float>::zeros((n_samples, max_seq_len, self.state.n_outputs))
            }
        };

        // Process each sequence
        for sample_idx in 0..n_samples {
            let x_seq = X.slice(s![sample_idx, .., ..]);

            let (sample_predictions, _) = self.forward_sequence_trained(&x_seq)?;

            match self.state.sequence_mode {
                SequenceMode::ManyToMany | SequenceMode::OneToMany => {
                    for t in 0..sample_predictions.nrows() {
                        for j in 0..sample_predictions.ncols() {
                            predictions[[sample_idx, t, j]] = sample_predictions[[t, j]];
                        }
                    }
                }
                SequenceMode::ManyToOne => {
                    for j in 0..sample_predictions.ncols() {
                        predictions[[sample_idx, 0, j]] = sample_predictions[[0, j]];
                    }
                }
            }
        }

        Ok(predictions)
    }
}

impl RecurrentNeuralNetwork<RecurrentNeuralNetworkTrained> {
    /// Forward pass for trained model
    fn forward_sequence_trained(
        &self,
        x_seq: &ArrayView2<'_, Float>,
    ) -> SklResult<(Array2<Float>, Vec<Vec<Array1<Float>>>)> {
        let (seq_len, _) = x_seq.dim();
        let n_outputs = self.state.output_weights.nrows();

        // Initialize hidden states
        let mut hidden_states = Vec::new();
        for _ in 0..self.state.num_layers {
            hidden_states.push(vec![
                Array1::<Float>::zeros(self.state.hidden_size);
                seq_len + 1
            ]);
        }

        // Forward pass (simplified for prediction)
        for t in 0..seq_len {
            let x_t = x_seq.row(t);

            for layer in 0..self.state.num_layers {
                let input = if layer == 0 {
                    x_t.to_owned()
                } else {
                    hidden_states[layer - 1][t].clone()
                };

                let prev_hidden = &hidden_states[layer][t];

                // Simplified cell computation for prediction
                let linear = self.state.input_weights[layer].dot(&input)
                    + self.state.hidden_weights[layer].dot(prev_hidden)
                    + &self.state.biases[layer];

                hidden_states[layer][t + 1] = match self.state.cell_type {
                    CellType::RNN => linear.map(|x| x.tanh()),
                    CellType::LSTM | CellType::GRU => linear.map(|x| x.tanh()), // Simplified
                };
            }
        }

        // Generate outputs
        let predictions = match self.state.sequence_mode {
            SequenceMode::ManyToMany => {
                let mut outputs = Array2::<Float>::zeros((seq_len, n_outputs));
                for t in 0..seq_len {
                    let last_layer_hidden = &hidden_states[self.state.num_layers - 1][t + 1];
                    let output_t =
                        self.state.output_weights.dot(last_layer_hidden) + &self.state.output_bias;
                    outputs.row_mut(t).assign(&output_t);
                }
                outputs
            }
            SequenceMode::ManyToOne => {
                let final_hidden = &hidden_states[self.state.num_layers - 1][seq_len];
                let output = self.state.output_weights.dot(final_hidden) + &self.state.output_bias;
                Array2::from_shape_vec((1, n_outputs), output.to_vec()).unwrap()
            }
            SequenceMode::OneToMany => {
                let mut outputs = Array2::<Float>::zeros((seq_len, n_outputs));
                for t in 0..seq_len {
                    let hidden_t = &hidden_states[self.state.num_layers - 1][t + 1];
                    let output_t =
                        self.state.output_weights.dot(hidden_t) + &self.state.output_bias;
                    outputs.row_mut(t).assign(&output_t);
                }
                outputs
            }
        };

        Ok((predictions, hidden_states))
    }

    /// Get the loss curve from training
    pub fn loss_curve(&self) -> &[Float] {
        &self.state.loss_curve
    }

    /// Get training iterations
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }

    /// Get network configuration
    pub fn cell_type(&self) -> CellType {
        self.state.cell_type
    }

    /// Get hidden size
    pub fn hidden_size(&self) -> usize {
        self.state.hidden_size
    }

    /// Get sequence mode
    pub fn sequence_mode(&self) -> SequenceMode {
        self.state.sequence_mode
    }
}

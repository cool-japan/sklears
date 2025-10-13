//! Sequence-to-sequence models for neural machine translation and similar tasks.
//!
//! This module provides implementations of encoder-decoder architectures for
//! sequence-to-sequence learning tasks like machine translation, text summarization,
//! and conversational AI.

use crate::{
    layers::attention::MultiHeadAttention,
    layers::rnn::{GRUCell, LSTMCell},
    layers::{Layer, ParameterizedLayer},
    NeuralResult,
};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use sklears_core::types::FloatBounds;

/// Encoder-Decoder architecture for sequence-to-sequence tasks
pub struct Seq2SeqModel<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> {
    encoder: Encoder<T>,
    decoder: Decoder<T>,
    attention: Option<AttentionMechanism<T>>,
    config: Seq2SeqConfig<T>,
}

/// Configuration for sequence-to-sequence models
#[derive(Debug, Clone)]
pub struct Seq2SeqConfig<T: FloatBounds> {
    pub input_vocab_size: usize,
    pub output_vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub dropout_rate: T,
    pub use_attention: bool,
    pub attention_heads: usize,
    pub bidirectional: bool,
    pub cell_type: RNNCellType,
    pub max_length: usize,
}

impl<T: FloatBounds> Default for Seq2SeqConfig<T> {
    fn default() -> Self {
        Self {
            input_vocab_size: 1000,
            output_vocab_size: 1000,
            hidden_size: 256,
            num_layers: 2,
            dropout_rate: T::from(0.1).unwrap_or_else(|| T::one() / T::from(10).unwrap()),
            use_attention: true,
            attention_heads: 8,
            bidirectional: false,
            cell_type: RNNCellType::LSTM,
            max_length: 100,
        }
    }
}

/// RNN cell types for sequence-to-sequence models
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RNNCellType {
    LSTM,
    GRU,
}

/// Encoder component of the sequence-to-sequence model
pub struct Encoder<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> {
    embedding: EmbeddingLayer<T>,
    rnn_layers: Vec<Box<dyn Layer<T>>>,
    hidden_size: usize,
    num_layers: usize,
    bidirectional: bool,
    cell_type: RNNCellType,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Encoder<T> {
    pub fn new(config: &Seq2SeqConfig<T>) -> NeuralResult<Self> {
        let embedding = EmbeddingLayer::new(config.input_vocab_size, config.hidden_size);
        let mut rnn_layers = Vec::new();

        for layer_idx in 0..config.num_layers {
            let input_size = if layer_idx == 0 {
                config.hidden_size
            } else {
                if config.bidirectional {
                    config.hidden_size * 2
                } else {
                    config.hidden_size
                }
            };

            match config.cell_type {
                RNNCellType::LSTM => {
                    let lstm = LSTMCell::new(input_size, config.hidden_size)?;
                    rnn_layers.push(Box::new(lstm) as Box<dyn Layer<T>>);
                }
                RNNCellType::GRU => {
                    let gru = GRUCell::new(input_size, config.hidden_size)?;
                    rnn_layers.push(Box::new(gru) as Box<dyn Layer<T>>);
                }
            }
        }

        Ok(Self {
            embedding,
            rnn_layers,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            bidirectional: config.bidirectional,
            cell_type: config.cell_type,
        })
    }

    /// Encode input sequence and return final hidden states
    pub fn encode(&mut self, input_seq: &Array2<usize>) -> NeuralResult<EncoderOutput<T>> {
        let (batch_size, seq_len) = input_seq.dim();

        // Embedding lookup
        let embedded = self.embedding.forward(input_seq)?;

        // Initialize hidden states
        let mut hidden_states = Vec::new();
        let mut cell_states = Vec::new();

        for _ in 0..self.num_layers {
            hidden_states.push(Array2::zeros((batch_size, self.hidden_size)));
            if self.cell_type == RNNCellType::LSTM {
                cell_states.push(Array2::zeros((batch_size, self.hidden_size)));
            }
        }

        // Forward pass through RNN layers
        let mut layer_input = embedded;
        let mut all_outputs = Vec::new();

        for time_step in 0..seq_len {
            let step_input = layer_input.slice(s![.., time_step, ..]).to_owned();
            let mut step_output = step_input;

            for (layer_idx, layer) in self.rnn_layers.iter_mut().enumerate() {
                // Forward through the RNN layer - LSTM/GRU cells expect 2D input
                step_output = layer.forward(&step_output, true)?;
            }

            all_outputs.push(step_output.clone());
        }

        // Stack outputs for all time steps
        let encoder_outputs = if all_outputs.is_empty() {
            Array3::zeros((batch_size, seq_len, self.hidden_size))
        } else {
            let mut outputs_3d = Array3::zeros((batch_size, seq_len, self.hidden_size));
            for (t, output) in all_outputs.iter().enumerate() {
                outputs_3d.slice_mut(s![.., t, ..]).assign(output);
            }
            outputs_3d
        };

        Ok(EncoderOutput {
            outputs: encoder_outputs,
            final_hidden: hidden_states,
            final_cell: cell_states,
        })
    }
}

/// Decoder component of the sequence-to-sequence model
pub struct Decoder<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> {
    embedding: EmbeddingLayer<T>,
    rnn_layers: Vec<Box<dyn Layer<T>>>,
    output_projection: LinearLayer<T>,
    hidden_size: usize,
    num_layers: usize,
    output_vocab_size: usize,
    cell_type: RNNCellType,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Decoder<T> {
    pub fn new(config: &Seq2SeqConfig<T>) -> NeuralResult<Self> {
        let embedding = EmbeddingLayer::new(config.output_vocab_size, config.hidden_size);
        let output_projection = LinearLayer::new(config.hidden_size, config.output_vocab_size);
        let mut rnn_layers = Vec::new();

        for layer_idx in 0..config.num_layers {
            let input_size = if layer_idx == 0 {
                config.hidden_size
            } else {
                config.hidden_size
            };

            match config.cell_type {
                RNNCellType::LSTM => {
                    let lstm = LSTMCell::new(input_size, config.hidden_size)?;
                    rnn_layers.push(Box::new(lstm) as Box<dyn Layer<T>>);
                }
                RNNCellType::GRU => {
                    let gru = GRUCell::new(input_size, config.hidden_size)?;
                    rnn_layers.push(Box::new(gru) as Box<dyn Layer<T>>);
                }
            }
        }

        Ok(Self {
            embedding,
            rnn_layers,
            output_projection,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            output_vocab_size: config.output_vocab_size,
            cell_type: config.cell_type,
        })
    }

    /// Decode one step given input token and previous hidden states
    pub fn decode_step(
        &mut self,
        input_token: &Array1<usize>,
        hidden_states: &mut Vec<Array2<T>>,
        cell_states: &mut Vec<Array2<T>>,
        encoder_output: Option<&EncoderOutput<T>>,
        attention: Option<&mut AttentionMechanism<T>>,
    ) -> NeuralResult<Array2<T>> {
        let batch_size = input_token.len();

        // Embedding lookup
        let embedded = self.embedding.forward_token(input_token)?;

        // Forward pass through RNN layers
        let mut layer_input = embedded;

        for (layer_idx, layer) in self.rnn_layers.iter_mut().enumerate() {
            // Forward through the RNN layer - LSTM/GRU cells expect 2D input
            layer_input = layer.forward(&layer_input, true)?;
        }

        // Apply attention if available
        let context_output =
            if let (Some(attention_layer), Some(enc_output)) = (attention, encoder_output) {
                attention_layer.apply_attention(&layer_input, &enc_output.outputs)?
            } else {
                layer_input.clone()
            };

        // Output projection
        let logits = self.output_projection.forward(&context_output)?;

        Ok(logits)
    }

    /// Decode full sequence using greedy decoding
    pub fn decode_greedy(
        &mut self,
        encoder_output: &EncoderOutput<T>,
        max_length: usize,
        start_token: usize,
        end_token: usize,
        mut attention: Option<&mut AttentionMechanism<T>>,
    ) -> NeuralResult<Array2<usize>> {
        let batch_size = encoder_output.final_hidden[0].nrows();
        let mut output_tokens = Vec::new();

        // Initialize decoder states with encoder final states
        let mut hidden_states = encoder_output.final_hidden.clone();
        let mut cell_states = encoder_output.final_cell.clone();

        // Start with start token
        let mut current_token = Array1::from_elem(batch_size, start_token);

        for _ in 0..max_length {
            let logits = self.decode_step(
                &current_token,
                &mut hidden_states,
                &mut cell_states,
                Some(encoder_output),
                attention.as_mut().map(|a| &mut **a),
            )?;

            // Greedy selection (argmax)
            let next_tokens = logits.map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            });

            output_tokens.push(next_tokens.clone());
            current_token = next_tokens;

            // Check if all sequences have ended
            if current_token.iter().all(|&token| token == end_token) {
                break;
            }
        }

        // Stack output tokens
        let mut output_seq = Array2::zeros((batch_size, output_tokens.len()));
        for (t, tokens) in output_tokens.iter().enumerate() {
            output_seq.column_mut(t).assign(tokens);
        }

        Ok(output_seq)
    }
}

/// Attention mechanism for sequence-to-sequence models
#[derive(Debug)]
pub struct AttentionMechanism<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> {
    attention_type: AttentionType,
    multi_head_attention: Option<MultiHeadAttention<T>>,
    linear_attention: Option<LinearAttention<T>>,
}

#[derive(Debug, Clone, Copy)]
pub enum AttentionType {
    MultiHead,
    Linear,
    Dot,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> AttentionMechanism<T> {
    pub fn new(
        attention_type: AttentionType,
        hidden_size: usize,
        num_heads: usize,
    ) -> NeuralResult<Self> {
        match attention_type {
            AttentionType::MultiHead => {
                let multi_head_attention =
                    Some(MultiHeadAttention::new(num_heads, hidden_size, None)?);
                Ok(Self {
                    attention_type,
                    multi_head_attention,
                    linear_attention: None,
                })
            }
            AttentionType::Linear => {
                let linear_attention = Some(LinearAttention::new(hidden_size));
                Ok(Self {
                    attention_type,
                    multi_head_attention: None,
                    linear_attention,
                })
            }
            AttentionType::Dot => Ok(Self {
                attention_type,
                multi_head_attention: None,
                linear_attention: None,
            }),
        }
    }

    pub fn apply_attention(
        &mut self,
        query: &Array2<T>,
        encoder_outputs: &Array3<T>,
    ) -> NeuralResult<Array2<T>> {
        match self.attention_type {
            AttentionType::MultiHead => {
                if let Some(ref mut mha) = self.multi_head_attention {
                    // Convert 3D encoder outputs to 2D for attention
                    let (batch_size, seq_len, hidden_size) = encoder_outputs.dim();
                    let key_value = encoder_outputs
                        .to_shape((batch_size * seq_len, hidden_size))
                        .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                            name: "shape_conversion".to_string(),
                            reason: "Failed to reshape encoder outputs".to_string(),
                        })?
                        .to_owned();

                    // First convert 2D inputs to 3D for attention
                    let batch_size = query.nrows();
                    let seq_len = 1; // Single timestep
                    let query_3d = query
                        .clone()
                        .into_shape((batch_size, seq_len, query.ncols()))
                        .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                            name: "shape_conversion".to_string(),
                            reason: "Failed to reshape query".to_string(),
                        })?;
                    let key_3d = key_value
                        .clone()
                        .into_shape((batch_size, seq_len, key_value.ncols()))
                        .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                            name: "shape_conversion".to_string(),
                            reason: "Failed to reshape key".to_string(),
                        })?;
                    let value_3d = key_value
                        .clone()
                        .into_shape((batch_size, seq_len, key_value.ncols()))
                        .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                            name: "shape_conversion".to_string(),
                            reason: "Failed to reshape value".to_string(),
                        })?;

                    let result_3d = mha.apply(&query_3d, &key_3d, &value_3d, None, true)?;
                    // Convert back to 2D
                    let total_elements = result_3d.len();
                    result_3d
                        .into_shape((batch_size, total_elements / batch_size))
                        .map_err(|_| sklears_core::error::SklearsError::InvalidParameter {
                            name: "attention_reshape".to_string(),
                            reason: "Failed to reshape attention output".to_string(),
                        })
                } else {
                    Err(sklears_core::error::SklearsError::InvalidParameter {
                        name: "attention".to_string(),
                        reason: "Multi-head attention not initialized".to_string(),
                    })
                }
            }
            AttentionType::Linear => {
                if let Some(ref mut linear_attn) = self.linear_attention {
                    linear_attn.forward(query, encoder_outputs)
                } else {
                    Err(sklears_core::error::SklearsError::InvalidParameter {
                        name: "attention".to_string(),
                        reason: "Linear attention not initialized".to_string(),
                    })
                }
            }
            AttentionType::Dot => self.dot_attention(query, encoder_outputs),
        }
    }

    fn dot_attention(
        &self,
        query: &Array2<T>,
        encoder_outputs: &Array3<T>,
    ) -> NeuralResult<Array2<T>> {
        let (batch_size, seq_len, hidden_size) = encoder_outputs.dim();
        let query_size = query.ncols();

        if query_size != hidden_size {
            return Err(sklears_core::error::SklearsError::InvalidParameter {
                name: "dimensions".to_string(),
                reason: "Query and encoder output dimensions must match".to_string(),
            });
        }

        // Compute attention scores
        let mut attention_scores = Array2::zeros((batch_size, seq_len));
        for b in 0..batch_size {
            for t in 0..seq_len {
                let key = encoder_outputs.slice(s![b, t, ..]);
                let query_slice = query.slice(s![b, ..]);

                // Dot product attention
                let score = query_slice.dot(&key);
                attention_scores[[b, t]] = score;
            }
        }

        // Apply softmax to attention scores
        let attention_weights = softmax(&attention_scores, Axis(1))?;

        // Compute weighted context vector
        let mut context = Array2::zeros((batch_size, hidden_size));
        for b in 0..batch_size {
            for t in 0..seq_len {
                let weight = attention_weights[[b, t]];
                let encoder_hidden = encoder_outputs.slice(s![b, t, ..]);

                for h in 0..hidden_size {
                    context[[b, h]] = context[[b, h]] + weight * encoder_hidden[h];
                }
            }
        }

        Ok(context)
    }
}

/// Linear attention mechanism
#[derive(Debug)]
pub struct LinearAttention<T: FloatBounds> {
    query_projection: LinearLayer<T>,
    key_projection: LinearLayer<T>,
    value_projection: LinearLayer<T>,
    output_projection: LinearLayer<T>,
}

impl<T: FloatBounds> LinearAttention<T> {
    pub fn new(hidden_size: usize) -> Self {
        Self {
            query_projection: LinearLayer::new(hidden_size, hidden_size),
            key_projection: LinearLayer::new(hidden_size, hidden_size),
            value_projection: LinearLayer::new(hidden_size, hidden_size),
            output_projection: LinearLayer::new(hidden_size, hidden_size),
        }
    }

    pub fn forward(
        &mut self,
        query: &Array2<T>,
        encoder_outputs: &Array3<T>,
    ) -> NeuralResult<Array2<T>> {
        let (batch_size, seq_len, hidden_size) = encoder_outputs.dim();

        // Project query
        let projected_query = self.query_projection.forward(query)?;

        // Project keys and values for each time step
        let mut projected_keys = Array3::zeros((batch_size, seq_len, hidden_size));
        let mut projected_values = Array3::zeros((batch_size, seq_len, hidden_size));

        for t in 0..seq_len {
            let encoder_step = encoder_outputs.slice(s![.., t, ..]).to_owned();
            let key_step = self.key_projection.forward(&encoder_step)?;
            let value_step = self.value_projection.forward(&encoder_step)?;

            projected_keys.slice_mut(s![.., t, ..]).assign(&key_step);
            projected_values
                .slice_mut(s![.., t, ..])
                .assign(&value_step);
        }

        // Compute attention scores
        let mut attention_scores = Array2::zeros((batch_size, seq_len));
        for b in 0..batch_size {
            for t in 0..seq_len {
                let key = projected_keys.slice(s![b, t, ..]);
                let query_slice = projected_query.slice(s![b, ..]);
                attention_scores[[b, t]] = query_slice.dot(&key);
            }
        }

        // Apply softmax and compute context
        let attention_weights = softmax(&attention_scores, Axis(1))?;
        let mut context = Array2::zeros((batch_size, hidden_size));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let weight = attention_weights[[b, t]];
                let value = projected_values.slice(s![b, t, ..]);

                for h in 0..hidden_size {
                    context[[b, h]] = context[[b, h]] + weight * value[h];
                }
            }
        }

        // Final output projection
        self.output_projection.forward(&context)
    }
}

/// Output of the encoder
#[derive(Debug)]
pub struct EncoderOutput<T: FloatBounds> {
    pub outputs: Array3<T>,           // (batch_size, seq_len, hidden_size)
    pub final_hidden: Vec<Array2<T>>, // Per layer hidden states
    pub final_cell: Vec<Array2<T>>,   // Per layer cell states (LSTM only)
}

/// Simplified embedding layer
#[derive(Debug)]
pub struct EmbeddingLayer<T: FloatBounds> {
    weights: Array2<T>,
    vocab_size: usize,
    embedding_dim: usize,
}

impl<T: FloatBounds> EmbeddingLayer<T> {
    pub fn new(vocab_size: usize, embedding_dim: usize) -> Self {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        // Xavier initialization
        let bound = (6.0 / (vocab_size + embedding_dim) as f64).sqrt();
        let weights = Array2::from_shape_fn((vocab_size, embedding_dim), |_| {
            T::from(rng.gen_range(-bound..bound)).unwrap_or_else(T::zero)
        });

        Self {
            weights,
            vocab_size,
            embedding_dim,
        }
    }

    pub fn forward(&self, input_tokens: &Array2<usize>) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len) = input_tokens.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.embedding_dim));

        for b in 0..batch_size {
            for t in 0..seq_len {
                let token_id = input_tokens[[b, t]];
                if token_id >= self.vocab_size {
                    return Err(sklears_core::error::SklearsError::InvalidParameter {
                        name: "token_id".to_string(),
                        reason: format!(
                            "Token ID {} exceeds vocabulary size {}",
                            token_id, self.vocab_size
                        ),
                    });
                }

                let embedding = self.weights.row(token_id);
                output.slice_mut(s![b, t, ..]).assign(&embedding);
            }
        }

        Ok(output)
    }

    pub fn forward_token(&self, input_tokens: &Array1<usize>) -> NeuralResult<Array2<T>> {
        let batch_size = input_tokens.len();
        let mut output = Array2::zeros((batch_size, self.embedding_dim));

        for b in 0..batch_size {
            let token_id = input_tokens[b];
            if token_id >= self.vocab_size {
                return Err(sklears_core::error::SklearsError::InvalidParameter {
                    name: "token_id".to_string(),
                    reason: format!(
                        "Token ID {} exceeds vocabulary size {}",
                        token_id, self.vocab_size
                    ),
                });
            }

            let embedding = self.weights.row(token_id);
            output.row_mut(b).assign(&embedding);
        }

        Ok(output)
    }
}

/// Simplified linear layer
#[derive(Debug)]
pub struct LinearLayer<T: FloatBounds> {
    weights: Array2<T>,
    bias: Array1<T>,
    input_size: usize,
    output_size: usize,
}

impl<T: FloatBounds> LinearLayer<T> {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        use scirs2_core::random::Rng;
        let mut rng = scirs2_core::random::thread_rng();

        // Xavier initialization
        let bound = (6.0 / (input_size + output_size) as f64).sqrt();
        let weights = Array2::from_shape_fn((input_size, output_size), |_| {
            T::from(rng.gen_range(-bound..bound)).unwrap_or_else(T::zero)
        });

        let bias = Array1::zeros(output_size);

        Self {
            weights,
            bias,
            input_size,
            output_size,
        }
    }

    pub fn forward(&self, input: &Array2<T>) -> NeuralResult<Array2<T>> {
        let output = input.dot(&self.weights);
        let mut result = output;

        // Add bias
        for mut row in result.rows_mut() {
            row += &self.bias;
        }

        Ok(result)
    }
}

/// Utility function for softmax
fn softmax<T: FloatBounds>(input: &Array2<T>, axis: Axis) -> NeuralResult<Array2<T>> {
    let mut result = input.clone();

    match axis {
        Axis(1) => {
            for mut row in result.rows_mut() {
                // Find max for numerical stability
                let max_val = row
                    .iter()
                    .fold(T::neg_infinity(), |acc, &x| if x > acc { x } else { acc });

                // Subtract max and compute exp
                row.mapv_inplace(|x| (x - max_val).exp());

                // Normalize
                let sum: T = row.sum();
                if sum > T::zero() {
                    row.mapv_inplace(|x| x / sum);
                }
            }
        }
        _ => {
            return Err(sklears_core::error::SklearsError::InvalidParameter {
                name: "axis".to_string(),
                reason: "Unsupported axis for softmax".to_string(),
            });
        }
    }

    Ok(result)
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Seq2SeqModel<T> {
    /// Create a new sequence-to-sequence model
    pub fn new(config: Seq2SeqConfig<T>) -> NeuralResult<Self> {
        let encoder = Encoder::new(&config)?;
        let decoder = Decoder::new(&config)?;

        let attention = if config.use_attention {
            Some(AttentionMechanism::new(
                AttentionType::MultiHead,
                config.hidden_size,
                config.attention_heads,
            )?)
        } else {
            None
        };

        Ok(Self {
            encoder,
            decoder,
            attention,
            config,
        })
    }

    /// Train the model on a batch of sequence pairs
    pub fn forward(
        &mut self,
        input_seq: &Array2<usize>,
        target_seq: &Array2<usize>,
    ) -> NeuralResult<Array3<T>> {
        // Encode input sequence
        let encoder_output = self.encoder.encode(input_seq)?;

        // Decode target sequence (teacher forcing during training)
        let (batch_size, target_len) = target_seq.dim();
        let mut decoder_outputs = Vec::new();

        // Initialize decoder states
        let mut hidden_states = encoder_output.final_hidden.clone();
        let mut cell_states = encoder_output.final_cell.clone();

        // Decode each step
        for t in 0..target_len - 1 {
            let input_token = target_seq.column(t).to_owned();
            let logits = self.decoder.decode_step(
                &input_token,
                &mut hidden_states,
                &mut cell_states,
                Some(&encoder_output),
                self.attention.as_mut(),
            )?;
            decoder_outputs.push(logits);
        }

        // Stack decoder outputs
        let output_len = decoder_outputs.len();
        let vocab_size = self.config.output_vocab_size;
        let mut output = Array3::zeros((batch_size, output_len, vocab_size));

        for (t, logits) in decoder_outputs.into_iter().enumerate() {
            output.slice_mut(s![.., t, ..]).assign(&logits);
        }

        Ok(output)
    }

    /// Generate sequences using the trained model
    pub fn generate(
        &mut self,
        input_seq: &Array2<usize>,
        start_token: usize,
        end_token: usize,
        max_length: Option<usize>,
    ) -> NeuralResult<Array2<usize>> {
        let max_len = max_length.unwrap_or(self.config.max_length);

        // Encode input sequence
        let encoder_output = self.encoder.encode(input_seq)?;

        // Generate output sequence
        self.decoder.decode_greedy(
            &encoder_output,
            max_len,
            start_token,
            end_token,
            self.attention.as_mut(),
        )
    }

    /// Get model configuration
    pub fn config(&self) -> &Seq2SeqConfig<T> {
        &self.config
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_seq2seq_config() {
        let config: Seq2SeqConfig<f32> = Seq2SeqConfig::default();
        assert_eq!(config.input_vocab_size, 1000);
        assert_eq!(config.output_vocab_size, 1000);
        assert_eq!(config.hidden_size, 256);
        assert_eq!(config.num_layers, 2);
        assert!(config.use_attention);
        assert_eq!(config.attention_heads, 8);
        assert_eq!(config.cell_type, RNNCellType::LSTM);
    }

    #[test]
    fn test_embedding_layer() -> NeuralResult<()> {
        let embedding = EmbeddingLayer::<f32>::new(100, 64);
        assert_eq!(embedding.vocab_size, 100);
        assert_eq!(embedding.embedding_dim, 64);

        let input = Array1::from_vec(vec![0, 1, 2]);
        let output = embedding.forward_token(&input)?;
        assert_eq!(output.shape(), &[3, 64]);

        Ok(())
    }

    #[test]
    fn test_linear_layer() -> NeuralResult<()> {
        let linear = LinearLayer::<f32>::new(10, 5);
        let input = Array2::ones((3, 10));
        let output = linear.forward(&input)?;
        assert_eq!(output.shape(), &[3, 5]);

        Ok(())
    }

    #[test]
    fn test_seq2seq_model_creation() -> NeuralResult<()> {
        let config = Seq2SeqConfig {
            input_vocab_size: 50,
            output_vocab_size: 60,
            hidden_size: 128,
            num_layers: 1,
            dropout_rate: 0.1,
            use_attention: false,
            attention_heads: 4,
            bidirectional: false,
            cell_type: RNNCellType::GRU,
            max_length: 50,
        };

        let model: Seq2SeqModel<f32> = Seq2SeqModel::new(config)?;
        assert_eq!(model.config().hidden_size, 128);
        assert_eq!(model.config().cell_type, RNNCellType::GRU);
        assert!(!model.config().use_attention);

        Ok(())
    }
}

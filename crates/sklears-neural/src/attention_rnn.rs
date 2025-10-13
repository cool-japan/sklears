//! Attention-based Recurrent Neural Networks
//!
//! This module provides RNN architectures enhanced with attention mechanisms,
//! including Attention-LSTM, Attention-GRU, and Hierarchical Attention Networks.

use crate::layers::attention::{MultiHeadAttention, ScaledDotProductAttention};
use crate::layers::rnn::{GRUCell, LSTMCell};
use crate::layers::Layer;
use crate::weight_init::{InitStrategy, WeightInitializer};
use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::SeedableRng;
use sklears_core::types::FloatBounds;

/// Configuration for attention-based RNN layers
#[derive(Debug, Clone)]
pub struct AttentionRNNConfig<T: FloatBounds> {
    /// Number of attention heads
    pub num_heads: usize,
    /// Attention dimension (d_k and d_v)
    pub attention_dim: usize,
    /// Dropout rate for attention weights
    pub attention_dropout: Option<T>,
    /// Whether to use hierarchical attention
    pub hierarchical: bool,
    /// Context vector dimension
    pub context_dim: Option<usize>,
    /// Whether to use global context
    pub use_global_context: bool,
    /// Temperature for attention weights
    pub temperature: T,
}

impl<T: FloatBounds> Default for AttentionRNNConfig<T> {
    fn default() -> Self {
        Self {
            num_heads: 8,
            attention_dim: 64,
            attention_dropout: Some(T::from(0.1).unwrap()),
            hierarchical: false,
            context_dim: None,
            use_global_context: false,
            temperature: T::one(),
        }
    }
}

/// Attention mechanism type for RNNs
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AttentionType {
    /// Bahdanau (additive) attention
    Bahdanau,
    /// Luong (multiplicative) attention
    Luong,
    /// Scaled dot-product attention
    ScaledDotProduct,
    /// Multi-head attention
    MultiHead,
}

/// Attention-enhanced LSTM cell
///
/// Combines LSTM with attention mechanism to allow the model to focus on
/// relevant parts of the input sequence during processing.
#[derive(Debug, Clone)]
pub struct AttentionLSTM<T: FloatBounds> {
    /// Base LSTM cell
    lstm: LSTMCell<T>,
    /// Attention mechanism
    attention: MultiHeadAttention<T>,
    /// Attention type
    attention_type: AttentionType,
    /// Configuration
    config: AttentionRNNConfig<T>,
    /// Context vector projection weights
    context_projection: Option<Array2<T>>,
    /// Output projection weights
    output_projection: Array2<T>,
    /// Last attention weights for visualization
    last_attention_weights: Option<Array2<T>>,
    /// Cached encoder states
    encoder_states: Option<Array3<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> AttentionLSTM<T> {
    /// Create a new attention LSTM layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        config: AttentionRNNConfig<T>,
        attention_type: AttentionType,
    ) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let init_strategy = InitStrategy::XavierUniform;
        let initializer = WeightInitializer::new(init_strategy);

        // Create base LSTM
        let lstm = LSTMCell::new(input_size, hidden_size)?;

        // Create attention mechanism
        let attention = match attention_type {
            AttentionType::MultiHead => {
                MultiHeadAttention::new(config.num_heads, hidden_size, config.attention_dropout)?
            }
            _ => {
                // For other attention types, use a single-head attention
                MultiHeadAttention::new(1, hidden_size, config.attention_dropout)?
            }
        };

        // Initialize context projection if needed
        let context_projection = if let Some(context_dim) = config.context_dim {
            let weights = initializer.initialize_2d(&mut rng, (hidden_size, context_dim))?;
            Some(weights)
        } else {
            None
        };

        // Initialize output projection
        let output_projection = initializer
            .initialize_2d(&mut rng, (hidden_size + config.attention_dim, hidden_size))?;

        Ok(Self {
            lstm,
            attention,
            attention_type,
            config,
            context_projection,
            output_projection,
            last_attention_weights: None,
            encoder_states: None,
        })
    }

    /// Set encoder states for attention computation
    pub fn set_encoder_states(&mut self, encoder_states: Array3<T>) {
        self.encoder_states = Some(encoder_states);
    }

    /// Get the last attention weights
    pub fn get_attention_weights(&self) -> Option<&Array2<T>> {
        self.last_attention_weights.as_ref()
    }

    /// Forward pass with attention
    pub fn forward_with_attention(
        &mut self,
        input: &Array2<T>,
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        let batch_size = input.nrows();

        // Standard LSTM forward pass
        let lstm_output = self.lstm.forward(input, training)?;

        // If encoder states are available, apply attention
        if let Some(ref encoder_states) = self.encoder_states {
            // Prepare query from LSTM output
            let query = lstm_output.clone().insert_axis(Axis(1)); // Add sequence dimension

            // Apply attention
            let attention_output =
                self.attention
                    .apply(&query, encoder_states, encoder_states, None, training)?;

            // Store attention weights (get first head's weights)
            let all_weights = self.attention.get_all_attention_weights();
            if let Some(Some(weights)) = all_weights.first() {
                self.last_attention_weights = Some((*weights).clone());
            }

            // Remove sequence dimension and combine with LSTM output
            let attention_flat = attention_output.index_axis(Axis(1), 0).to_owned();
            let combined = self.combine_lstm_attention(&lstm_output, &attention_flat)?;

            Ok(combined)
        } else {
            // No attention, just return LSTM output
            Ok(lstm_output)
        }
    }

    /// Combine LSTM output with attention context
    fn combine_lstm_attention(
        &self,
        lstm_output: &Array2<T>,
        attention_output: &Array2<T>,
    ) -> NeuralResult<Array2<T>> {
        let batch_size = lstm_output.nrows();
        let lstm_dim = lstm_output.ncols();
        let attention_dim = attention_output.ncols();

        // Concatenate LSTM and attention outputs
        let mut combined = Array2::zeros((batch_size, lstm_dim + attention_dim));
        combined.slice_mut(s![.., ..lstm_dim]).assign(lstm_output);
        combined
            .slice_mut(s![.., lstm_dim..])
            .assign(attention_output);

        // Project to output dimension
        let output = combined.dot(&self.output_projection);
        Ok(output)
    }

    /// Process a sequence with attention
    pub fn forward_sequence(
        &mut self,
        inputs: &Array3<T>,
        encoder_states: Option<&Array3<T>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, _) = inputs.dim();

        if let Some(states) = encoder_states {
            self.set_encoder_states(states.clone());
        }

        let mut outputs = Vec::new();

        for t in 0..seq_len {
            let input_t = inputs.slice(s![.., t, ..]).to_owned();
            let output_t = self.forward_with_attention(&input_t, training)?;
            outputs.push(output_t);
        }

        // Stack outputs
        let output_dim = outputs[0].ncols();
        let mut result = Array3::zeros((batch_size, seq_len, output_dim));

        for (t, output) in outputs.iter().enumerate() {
            result.slice_mut(s![.., t, ..]).assign(output);
        }

        Ok(result)
    }

    /// Reset the internal states
    pub fn reset_states(&mut self) {
        self.lstm.reset_state();
        self.last_attention_weights = None;
        self.encoder_states = None;
    }
}

/// Attention-enhanced GRU cell
#[derive(Debug, Clone)]
pub struct AttentionGRU<T: FloatBounds> {
    /// Base GRU cell
    gru: GRUCell<T>,
    /// Attention mechanism
    attention: ScaledDotProductAttention<T>,
    /// Configuration
    config: AttentionRNNConfig<T>,
    /// Context projection weights
    context_projection: Option<Array2<T>>,
    /// Output projection weights
    output_projection: Array2<T>,
    /// Last attention weights
    last_attention_weights: Option<Array2<T>>,
    /// Encoder states for attention
    encoder_states: Option<Array3<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> AttentionGRU<T> {
    /// Create a new attention GRU layer
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        config: AttentionRNNConfig<T>,
    ) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let init_strategy = InitStrategy::XavierUniform;
        let initializer = WeightInitializer::new(init_strategy);

        // Create base GRU
        let gru = GRUCell::new(input_size, hidden_size)?;

        // Create attention mechanism
        let attention = ScaledDotProductAttention::new(hidden_size, config.attention_dropout);

        // Initialize projections
        let context_projection = if let Some(context_dim) = config.context_dim {
            let weights = initializer.initialize_2d(&mut rng, (hidden_size, context_dim))?;
            Some(weights)
        } else {
            None
        };

        let output_projection =
            initializer.initialize_2d(&mut rng, (hidden_size * 2, hidden_size))?;

        Ok(Self {
            gru,
            attention,
            config,
            context_projection,
            output_projection,
            last_attention_weights: None,
            encoder_states: None,
        })
    }

    /// Set encoder states for attention
    pub fn set_encoder_states(&mut self, encoder_states: Array3<T>) {
        self.encoder_states = Some(encoder_states);
    }

    /// Forward pass with attention
    pub fn forward_with_attention(
        &mut self,
        input: &Array2<T>,
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        // GRU forward pass
        let gru_output = self.gru.forward(input, training)?;

        // Apply attention if encoder states available
        if let Some(ref encoder_states) = self.encoder_states {
            let query = gru_output.clone().insert_axis(Axis(1));
            let attention_output = self.attention.apply_attention(
                &query,
                encoder_states,
                encoder_states,
                None,
                training,
            )?;

            let attention_flat = attention_output.index_axis(Axis(1), 0).to_owned();
            self.combine_gru_attention(&gru_output, &attention_flat)
        } else {
            Ok(gru_output)
        }
    }

    /// Combine GRU output with attention
    fn combine_gru_attention(
        &self,
        gru_output: &Array2<T>,
        attention_output: &Array2<T>,
    ) -> NeuralResult<Array2<T>> {
        let batch_size = gru_output.nrows();
        let gru_dim = gru_output.ncols();
        let attention_dim = attention_output.ncols();

        // Concatenate and project
        let mut combined = Array2::zeros((batch_size, gru_dim + attention_dim));
        combined.slice_mut(s![.., ..gru_dim]).assign(gru_output);
        combined
            .slice_mut(s![.., gru_dim..])
            .assign(attention_output);

        let output = combined.dot(&self.output_projection);
        Ok(output)
    }
}

/// Hierarchical Attention Network (HAN)
///
/// Implements a hierarchical attention mechanism that operates at multiple levels,
/// typically used for document classification with word-level and sentence-level attention.
#[derive(Debug, Clone)]
pub struct HierarchicalAttentionNetwork<T: FloatBounds> {
    /// Word-level LSTM
    word_lstm: AttentionLSTM<T>,
    /// Sentence-level LSTM
    sentence_lstm: AttentionLSTM<T>,
    /// Word-level attention
    word_attention: ScaledDotProductAttention<T>,
    /// Sentence-level attention
    sentence_attention: ScaledDotProductAttention<T>,
    /// Word context vector
    word_context: Array1<T>,
    /// Sentence context vector
    sentence_context: Array1<T>,
    /// Configuration
    config: AttentionRNNConfig<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> HierarchicalAttentionNetwork<T> {
    /// Create a new hierarchical attention network
    pub fn new(
        vocab_size: usize,
        word_hidden_size: usize,
        sentence_hidden_size: usize,
        config: AttentionRNNConfig<T>,
    ) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let init_strategy = InitStrategy::XavierUniform;
        let initializer = WeightInitializer::new(init_strategy);

        // Create word-level components
        let word_lstm = AttentionLSTM::new(
            vocab_size,
            word_hidden_size,
            config.clone(),
            AttentionType::ScaledDotProduct,
        )?;

        let word_attention =
            ScaledDotProductAttention::new(word_hidden_size, config.attention_dropout);

        let word_context = initializer.initialize_1d(&mut rng, word_hidden_size)?;

        // Create sentence-level components
        let sentence_lstm = AttentionLSTM::new(
            word_hidden_size,
            sentence_hidden_size,
            config.clone(),
            AttentionType::ScaledDotProduct,
        )?;

        let sentence_attention =
            ScaledDotProductAttention::new(sentence_hidden_size, config.attention_dropout);

        let sentence_context = initializer.initialize_1d(&mut rng, sentence_hidden_size)?;

        Ok(Self {
            word_lstm,
            sentence_lstm,
            word_attention,
            sentence_attention,
            word_context,
            sentence_context,
            config,
        })
    }

    /// Forward pass through hierarchical attention
    pub fn forward_hierarchical(
        &mut self,
        document: &Array3<T>, // (batch_size, num_sentences, max_words)
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        let (batch_size, num_sentences, _) = document.dim();
        let mut sentence_representations = Vec::new();

        // Process each sentence with word-level attention
        for s in 0..num_sentences {
            let sentence = document.slice(s![.., s, ..]).to_owned();
            let sentence_repr = self.word_level_attention(&sentence, training)?;
            sentence_representations.push(sentence_repr);
        }

        // Stack sentence representations
        let hidden_size = sentence_representations[0].ncols();
        let mut sentence_matrix = Array3::zeros((batch_size, num_sentences, hidden_size));
        for (s, repr) in sentence_representations.iter().enumerate() {
            sentence_matrix.slice_mut(s![.., s, ..]).assign(repr);
        }

        // Apply sentence-level attention
        self.sentence_level_attention(&sentence_matrix, training)
    }

    /// Apply word-level attention within sentences
    fn word_level_attention(
        &mut self,
        sentence: &Array2<T>,
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        // Convert to sequence format for LSTM
        let sentence_seq = sentence.clone().insert_axis(Axis(1));
        let lstm_output = self
            .word_lstm
            .forward_sequence(&sentence_seq, None, training)?;

        // Apply word-level attention
        let query = self
            .word_context
            .clone()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0)); // Add batch and seq dims
        let attention_output = self.word_attention.apply_attention(
            &query
                .broadcast((sentence.len_of(Axis(0)), 1, self.word_context.len()))
                .unwrap()
                .to_owned(),
            &lstm_output,
            &lstm_output,
            None,
            training,
        )?;

        // Return attended sentence representation
        Ok(attention_output.index_axis(Axis(1), 0).to_owned())
    }

    /// Apply sentence-level attention across sentences
    fn sentence_level_attention(
        &mut self,
        sentences: &Array3<T>,
        training: bool,
    ) -> NeuralResult<Array2<T>> {
        // Process with sentence-level LSTM
        let lstm_output = self
            .sentence_lstm
            .forward_sequence(sentences, None, training)?;

        // Apply sentence-level attention
        let query = self
            .sentence_context
            .clone()
            .insert_axis(Axis(0))
            .insert_axis(Axis(0));
        let attention_output = self.sentence_attention.apply_attention(
            &query
                .broadcast((sentences.len_of(Axis(0)), 1, self.sentence_context.len()))
                .unwrap()
                .to_owned(),
            &lstm_output,
            &lstm_output,
            None,
            training,
        )?;

        // Return document representation
        Ok(attention_output.index_axis(Axis(1), 0).to_owned())
    }
}

/// Self-attention mechanism for sequence modeling
#[derive(Debug, Clone)]
pub struct SelfAttentionRNN<T: FloatBounds> {
    /// Base RNN (can be LSTM or GRU)
    rnn: AttentionLSTM<T>,
    /// Self-attention mechanism
    self_attention: MultiHeadAttention<T>,
    /// Layer normalization weights
    layer_norm_weights: Array1<T>,
    /// Layer normalization bias
    layer_norm_bias: Array1<T>,
    /// Configuration
    config: AttentionRNNConfig<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> SelfAttentionRNN<T> {
    /// Create a new self-attention RNN
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        config: AttentionRNNConfig<T>,
    ) -> NeuralResult<Self> {
        let rnn = AttentionLSTM::new(
            input_size,
            hidden_size,
            config.clone(),
            AttentionType::MultiHead,
        )?;
        let self_attention =
            MultiHeadAttention::new(config.num_heads, hidden_size, config.attention_dropout)?;

        // Initialize layer normalization parameters
        let layer_norm_weights = Array1::ones(hidden_size);
        let layer_norm_bias = Array1::zeros(hidden_size);

        Ok(Self {
            rnn,
            self_attention,
            layer_norm_weights,
            layer_norm_bias,
            config,
        })
    }

    /// Forward pass with self-attention
    pub fn forward_self_attention(
        &mut self,
        input: &Array3<T>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        // RNN forward pass
        let rnn_output = self.rnn.forward_sequence(input, None, training)?;

        // Self-attention
        let attention_output =
            self.self_attention
                .apply(&rnn_output, &rnn_output, &rnn_output, None, training)?;

        // Residual connection and layer normalization
        let residual = &rnn_output + &attention_output;
        let normalized = self.apply_layer_norm(&residual)?;

        Ok(normalized)
    }

    /// Apply layer normalization
    fn apply_layer_norm(&self, input: &Array3<T>) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, hidden_size) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, hidden_size));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = input.slice(s![b, s, ..]);
                let mean = x.mean().unwrap_or(T::zero());
                let variance = x
                    .mapv(|v| (v - mean) * (v - mean))
                    .mean()
                    .unwrap_or(T::zero());
                let std = (variance + T::from(1e-5).unwrap()).sqrt();

                let normalized = x.mapv(|v| (v - mean) / std);
                let scaled = &normalized * &self.layer_norm_weights + &self.layer_norm_bias;

                output.slice_mut(s![b, s, ..]).assign(&scaled);
            }
        }

        Ok(output)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use scirs2_core::essentials::Normal;
    use scirs2_core::ndarray::Array3;
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_attention_lstm_creation() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use hidden_size=64 which is divisible by num_heads=8
        let lstm = AttentionLSTM::new(10, 64, config, AttentionType::MultiHead);
        assert!(lstm.is_ok());
    }

    #[test]
    fn test_attention_gru_creation() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use hidden_size=64 which is divisible by num_heads=8
        let gru = AttentionGRU::new(10, 64, config);
        assert!(gru.is_ok());
    }

    #[test]
    fn test_attention_lstm_forward() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use hidden_size=64 which is divisible by num_heads=8
        let mut lstm = AttentionLSTM::new(10, 64, config, AttentionType::MultiHead).unwrap();

        let input = Array2::from_shape_fn((5, 10), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let output = lstm.forward_with_attention(&input, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dim(), (5, 64));
    }

    #[test]
    fn test_hierarchical_attention_network() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use dimensions that are divisible by num_heads=8
        // vocab_size=96, word_hidden_size=64, sentence_hidden_size=96
        let mut han = HierarchicalAttentionNetwork::new(96, 64, 96, config).unwrap();

        let document = Array3::from_shape_fn((2, 5, 96), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        }); // 2 docs, 5 sentences, 96 word features
        let output = han.forward_hierarchical(&document, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dim(), (2, 96)); // batch_size x sentence_hidden_size
    }

    #[test]
    fn test_self_attention_rnn() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use hidden_size=64 which is divisible by num_heads=8
        let mut self_attn_rnn = SelfAttentionRNN::new(10, 64, config).unwrap();

        let input = Array3::from_shape_fn((2, 8, 10), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        }); // batch, seq, features
        let output = self_attn_rnn.forward_self_attention(&input, false);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.dim(), (2, 8, 64));
    }

    #[test]
    fn test_attention_weights_storage() {
        let config = AttentionRNNConfig::<f64>::default();
        // Use hidden_size=64 which is divisible by num_heads=8
        let mut lstm = AttentionLSTM::new(10, 64, config, AttentionType::MultiHead).unwrap();

        // Set encoder states
        let encoder_states = Array3::from_shape_fn((1, 5, 64), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        lstm.set_encoder_states(encoder_states);

        let input = Array2::from_shape_fn((1, 10), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let _output = lstm.forward_with_attention(&input, false).unwrap();

        // Check if attention weights are stored
        assert!(lstm.get_attention_weights().is_some());
    }
}

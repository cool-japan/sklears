//! Encoder-Decoder Transformer Architectures
//!
//! This module provides complete transformer architectures including encoder-decoder
//! models for sequence-to-sequence tasks, encoder-only models for classification,
//! and decoder-only models for language generation.

use crate::layers::attention::{MultiHeadAttention, ScaledDotProductAttention};
use crate::layers::transformer::{PositionalEncoding, PositionalEncodingType};
use crate::layers::{Layer, ParameterizedLayer};
use crate::weight_init::{InitStrategy, WeightInitializer};
use crate::NeuralResult;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Axis};
use scirs2_core::random::ChaCha8Rng;
use scirs2_core::random::RandomExt;
use scirs2_core::random::SeedableRng;
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;

/// Configuration for transformer models
#[derive(Debug, Clone)]
pub struct TransformerConfig<T: FloatBounds> {
    /// Model dimension (d_model)
    pub d_model: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension of feed-forward network
    pub d_ff: usize,
    /// Number of encoder layers
    pub num_encoder_layers: usize,
    /// Number of decoder layers
    pub num_decoder_layers: usize,
    /// Maximum sequence length
    pub max_seq_len: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Dropout rate
    pub dropout_rate: T,
    /// Label smoothing for training
    pub label_smoothing: T,
    /// Whether to use pre-normalization
    pub pre_norm: bool,
    /// Activation function for FFN
    pub activation: String,
    /// Whether to share embeddings between encoder and decoder
    pub share_embeddings: bool,
}

impl<T: FloatBounds> Default for TransformerConfig<T> {
    fn default() -> Self {
        Self {
            d_model: 512,
            num_heads: 8,
            d_ff: 2048,
            num_encoder_layers: 6,
            num_decoder_layers: 6,
            max_seq_len: 512,
            vocab_size: 30000,
            dropout_rate: T::from(0.1).unwrap(),
            label_smoothing: T::from(0.1).unwrap(),
            pre_norm: false,
            activation: "relu".to_string(),
            share_embeddings: false,
        }
    }
}

/// Feed-forward network used in transformer blocks
#[derive(Debug, Clone)]
pub struct FeedForwardNetwork<T: FloatBounds> {
    /// First linear transformation
    linear1: Array2<T>,
    /// First bias
    bias1: Array1<T>,
    /// Second linear transformation
    linear2: Array2<T>,
    /// Second bias
    bias2: Array1<T>,
    /// Activation function
    activation: String,
    /// Dropout rate
    dropout_rate: T,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> FeedForwardNetwork<T> {
    /// Create a new feed-forward network
    pub fn new(
        d_model: usize,
        d_ff: usize,
        dropout_rate: T,
        activation: String,
    ) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let initializer = WeightInitializer::new(InitStrategy::XavierUniform);

        let linear1 = initializer.initialize_2d(&mut rng, (d_model, d_ff))?;
        let bias1 = Array1::zeros(d_ff);
        let linear2 = initializer.initialize_2d(&mut rng, (d_ff, d_model))?;
        let bias2 = Array1::zeros(d_model);

        Ok(Self {
            linear1,
            bias1,
            linear2,
            bias2,
            activation,
            dropout_rate,
        })
    }

    /// Forward pass through the feed-forward network
    pub fn forward(&self, input: &Array3<T>, training: bool) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, d_model) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, self.linear2.ncols()));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = input.slice(s![b, s, ..]);

                // First linear transformation
                let hidden = x.dot(&self.linear1) + &self.bias1;

                // Apply activation
                let activated = match self.activation.as_str() {
                    "relu" => hidden.mapv(|x| x.max(T::zero())),
                    "gelu" => hidden.mapv(|x| {
                        let x_f64 = x.to_f64().unwrap_or(0.0);
                        let gelu_val = 0.5 * x_f64 * (1.0 + (x_f64 * 0.7978845608028654).tanh());
                        T::from(gelu_val).unwrap_or(T::zero())
                    }),
                    _ => hidden, // Default to identity
                };

                // Apply dropout if training
                let dropout_applied = if training && self.dropout_rate > T::zero() {
                    // Simple dropout implementation
                    use scirs2_core::random::Rng;
                    let mut rng = scirs2_core::random::thread_rng();
                    activated.mapv(|x| {
                        if rng.gen::<f64>() < self.dropout_rate.to_f64().unwrap_or(0.0) {
                            T::zero()
                        } else {
                            x / (T::one() - self.dropout_rate)
                        }
                    })
                } else {
                    activated
                };

                // Second linear transformation
                let final_output = dropout_applied.dot(&self.linear2) + &self.bias2;
                output.slice_mut(s![b, s, ..]).assign(&final_output);
            }
        }

        Ok(output)
    }
}

/// Layer normalization for transformers
#[derive(Debug, Clone)]
pub struct LayerNorm<T: FloatBounds> {
    /// Normalization weights
    weight: Array1<T>,
    /// Normalization bias
    bias: Array1<T>,
    /// Epsilon for numerical stability
    eps: T,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> LayerNorm<T> {
    /// Create a new layer normalization layer
    pub fn new(d_model: usize) -> Self {
        Self {
            weight: Array1::ones(d_model),
            bias: Array1::zeros(d_model),
            eps: T::from(1e-5).unwrap(),
        }
    }

    /// Apply layer normalization
    pub fn forward(&self, input: &Array3<T>) -> Array3<T> {
        let (batch_size, seq_len, d_model) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let x = input.slice(s![b, s, ..]);
                let mean = x.mean().unwrap_or(T::zero());
                let variance = x
                    .mapv(|v| (v - mean) * (v - mean))
                    .mean()
                    .unwrap_or(T::zero());
                let std = (variance + self.eps).sqrt();

                let normalized = x.mapv(|v| (v - mean) / std);
                let scaled = &normalized * &self.weight + &self.bias;

                output.slice_mut(s![b, s, ..]).assign(&scaled);
            }
        }

        output
    }
}

/// Transformer encoder layer
#[derive(Debug, Clone)]
pub struct TransformerEncoderLayer<T: FloatBounds> {
    /// Multi-head self-attention
    self_attention: MultiHeadAttention<T>,
    /// Feed-forward network
    ffn: FeedForwardNetwork<T>,
    /// Layer normalization for attention
    norm1: LayerNorm<T>,
    /// Layer normalization for FFN
    norm2: LayerNorm<T>,
    /// Dropout rate
    dropout_rate: T,
    /// Whether to use pre-normalization
    pre_norm: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> TransformerEncoderLayer<T> {
    pub fn new(config: &TransformerConfig<T>) -> NeuralResult<Self> {
        let self_attention =
            MultiHeadAttention::new(config.num_heads, config.d_model, Some(config.dropout_rate))?;

        let ffn = FeedForwardNetwork::new(
            config.d_model,
            config.d_ff,
            config.dropout_rate,
            config.activation.clone(),
        )?;

        let norm1 = LayerNorm::new(config.d_model);
        let norm2 = LayerNorm::new(config.d_model);

        Ok(Self {
            self_attention,
            ffn,
            norm1,
            norm2,
            dropout_rate: config.dropout_rate,
            pre_norm: config.pre_norm,
        })
    }

    /// Forward pass through encoder layer
    pub fn forward(
        &mut self,
        input: &Array3<T>,
        mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        if self.pre_norm {
            // Pre-normalization: LayerNorm -> Attention -> Residual
            let norm_input = self.norm1.forward(input);
            let attn_output =
                self.self_attention
                    .apply(&norm_input, &norm_input, &norm_input, mask, training)?;
            let residual1 = input + &attn_output;

            let norm_residual1 = self.norm2.forward(&residual1);
            let ffn_output = self.ffn.forward(&norm_residual1, training)?;
            let residual2 = &residual1 + &ffn_output;

            Ok(residual2)
        } else {
            // Post-normalization: Attention -> Residual -> LayerNorm
            let attn_output = self
                .self_attention
                .apply(input, input, input, mask, training)?;
            let residual1 = input + &attn_output;
            let norm1_output = self.norm1.forward(&residual1);

            let ffn_output = self.ffn.forward(&norm1_output, training)?;
            let residual2 = &norm1_output + &ffn_output;
            let norm2_output = self.norm2.forward(&residual2);

            Ok(norm2_output)
        }
    }
}

/// Transformer decoder layer
#[derive(Debug, Clone)]
pub struct TransformerDecoderLayer<T: FloatBounds> {
    /// Masked self-attention
    self_attention: MultiHeadAttention<T>,
    /// Cross-attention with encoder
    cross_attention: MultiHeadAttention<T>,
    /// Feed-forward network
    ffn: FeedForwardNetwork<T>,
    /// Layer normalization layers
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    norm3: LayerNorm<T>,
    /// Dropout rate
    dropout_rate: T,
    /// Whether to use pre-normalization
    pre_norm: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> TransformerDecoderLayer<T> {
    pub fn new(config: &TransformerConfig<T>) -> NeuralResult<Self> {
        let self_attention =
            MultiHeadAttention::new(config.num_heads, config.d_model, Some(config.dropout_rate))?;

        let cross_attention =
            MultiHeadAttention::new(config.num_heads, config.d_model, Some(config.dropout_rate))?;

        let ffn = FeedForwardNetwork::new(
            config.d_model,
            config.d_ff,
            config.dropout_rate,
            config.activation.clone(),
        )?;

        let norm1 = LayerNorm::new(config.d_model);
        let norm2 = LayerNorm::new(config.d_model);
        let norm3 = LayerNorm::new(config.d_model);

        Ok(Self {
            self_attention,
            cross_attention,
            ffn,
            norm1,
            norm2,
            norm3,
            dropout_rate: config.dropout_rate,
            pre_norm: config.pre_norm,
        })
    }

    /// Forward pass through decoder layer
    pub fn forward(
        &mut self,
        input: &Array3<T>,
        encoder_output: &Array3<T>,
        self_mask: Option<&Array2<bool>>,
        cross_mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        if self.pre_norm {
            // Pre-normalization
            let norm_input = self.norm1.forward(input);
            let self_attn_output = self.self_attention.apply(
                &norm_input,
                &norm_input,
                &norm_input,
                self_mask,
                training,
            )?;
            let residual1 = input + &self_attn_output;

            let norm_residual1 = self.norm2.forward(&residual1);
            let cross_attn_output = self.cross_attention.apply(
                &norm_residual1,
                encoder_output,
                encoder_output,
                cross_mask,
                training,
            )?;
            let residual2 = &residual1 + &cross_attn_output;

            let norm_residual2 = self.norm3.forward(&residual2);
            let ffn_output = self.ffn.forward(&norm_residual2, training)?;
            let residual3 = &residual2 + &ffn_output;

            Ok(residual3)
        } else {
            // Post-normalization
            let self_attn_output = self
                .self_attention
                .apply(input, input, input, self_mask, training)?;
            let residual1 = input + &self_attn_output;
            let norm1_output = self.norm1.forward(&residual1);

            let cross_attn_output = self.cross_attention.apply(
                &norm1_output,
                encoder_output,
                encoder_output,
                cross_mask,
                training,
            )?;
            let residual2 = &norm1_output + &cross_attn_output;
            let norm2_output = self.norm2.forward(&residual2);

            let ffn_output = self.ffn.forward(&norm2_output, training)?;
            let residual3 = &norm2_output + &ffn_output;
            let norm3_output = self.norm3.forward(&residual3);

            Ok(norm3_output)
        }
    }
}

/// Complete Transformer encoder
#[derive(Debug, Clone)]
pub struct TransformerEncoder<T: FloatBounds> {
    /// Stack of encoder layers
    layers: Vec<TransformerEncoderLayer<T>>,
    /// Input embeddings
    embeddings: Array2<T>,
    /// Positional encoding
    positional_encoding: PositionalEncoding<T>,
    /// Final layer normalization
    final_norm: Option<LayerNorm<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + From<f64>> TransformerEncoder<T> {
    /// Create a new transformer encoder
    pub fn new(config: &TransformerConfig<T>) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let initializer = WeightInitializer::new(InitStrategy::XavierUniform);

        // Create encoder layers
        let mut layers = Vec::new();
        for _ in 0..config.num_encoder_layers {
            layers.push(TransformerEncoderLayer::new(config)?);
        }

        // Initialize embeddings
        let embeddings =
            initializer.initialize_2d(&mut rng, (config.vocab_size, config.d_model))?;

        // Create positional encoding
        let positional_encoding = PositionalEncoding::new(
            config.max_seq_len,
            config.d_model,
            PositionalEncodingType::Sinusoidal,
            config.dropout_rate,
            true,
        )?;

        // Final layer normalization for pre-norm architectures
        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(config.d_model))
        } else {
            None
        };

        Ok(Self {
            layers,
            embeddings,
            positional_encoding,
            final_norm,
        })
    }

    /// Forward pass through encoder
    pub fn forward(
        &mut self,
        input_ids: &Array2<usize>,
        mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len) = input_ids.dim();
        let d_model = self.embeddings.ncols();

        // Embedding lookup
        let mut embedded = Array3::zeros((batch_size, seq_len, d_model));
        for b in 0..batch_size {
            for s in 0..seq_len {
                let token_id = input_ids[[b, s]];
                if token_id < self.embeddings.nrows() {
                    embedded
                        .slice_mut(s![b, s, ..])
                        .assign(&self.embeddings.row(token_id));
                }
            }
        }

        // Add positional encoding
        let mut x = self.positional_encoding.encode(&embedded)?;

        // Pass through encoder layers
        for layer in &mut self.layers {
            x = layer.forward(&x, mask, training)?;
        }

        // Apply final normalization if using pre-norm
        if let Some(ref final_norm) = self.final_norm {
            x = final_norm.forward(&x);
        }

        Ok(x)
    }
}

/// Complete Transformer decoder
#[derive(Debug, Clone)]
pub struct TransformerDecoder<T: FloatBounds> {
    /// Stack of decoder layers
    layers: Vec<TransformerDecoderLayer<T>>,
    /// Output embeddings
    embeddings: Array2<T>,
    /// Positional encoding
    positional_encoding: PositionalEncoding<T>,
    /// Final layer normalization
    final_norm: Option<LayerNorm<T>>,
    /// Output projection to vocabulary
    output_projection: Array2<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + From<f64>> TransformerDecoder<T> {
    /// Create a new transformer decoder
    pub fn new(config: &TransformerConfig<T>) -> NeuralResult<Self> {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let initializer = WeightInitializer::new(InitStrategy::XavierUniform);

        // Create decoder layers
        let mut layers = Vec::new();
        for _ in 0..config.num_decoder_layers {
            layers.push(TransformerDecoderLayer::new(config)?);
        }

        // Initialize embeddings
        let embeddings =
            initializer.initialize_2d(&mut rng, (config.vocab_size, config.d_model))?;

        // Create positional encoding
        let positional_encoding = PositionalEncoding::new(
            config.max_seq_len,
            config.d_model,
            PositionalEncodingType::Sinusoidal,
            config.dropout_rate,
            true,
        )?;

        // Final layer normalization
        let final_norm = if config.pre_norm {
            Some(LayerNorm::new(config.d_model))
        } else {
            None
        };

        // Output projection
        let output_projection =
            initializer.initialize_2d(&mut rng, (config.d_model, config.vocab_size))?;

        Ok(Self {
            layers,
            embeddings,
            positional_encoding,
            final_norm,
            output_projection,
        })
    }

    /// Forward pass through decoder
    pub fn forward(
        &mut self,
        input_ids: &Array2<usize>,
        encoder_output: &Array3<T>,
        self_mask: Option<&Array2<bool>>,
        cross_mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len) = input_ids.dim();
        let d_model = self.embeddings.ncols();

        // Embedding lookup
        let mut embedded = Array3::zeros((batch_size, seq_len, d_model));
        for b in 0..batch_size {
            for s in 0..seq_len {
                let token_id = input_ids[[b, s]];
                if token_id < self.embeddings.nrows() {
                    embedded
                        .slice_mut(s![b, s, ..])
                        .assign(&self.embeddings.row(token_id));
                }
            }
        }

        // Add positional encoding
        let mut x = self.positional_encoding.encode(&embedded)?;

        // Pass through decoder layers
        for layer in &mut self.layers {
            x = layer.forward(&x, encoder_output, self_mask, cross_mask, training)?;
        }

        // Apply final normalization if using pre-norm
        if let Some(ref final_norm) = self.final_norm {
            x = final_norm.forward(&x);
        }

        // Project to vocabulary
        let (batch_size, seq_len, d_model) = x.dim();
        let mut logits = Array3::zeros((batch_size, seq_len, self.output_projection.ncols()));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let hidden = x.slice(s![b, s, ..]);
                let output = hidden.dot(&self.output_projection);
                logits.slice_mut(s![b, s, ..]).assign(&output);
            }
        }

        Ok(logits)
    }
}

/// Complete Encoder-Decoder Transformer
#[derive(Debug, Clone)]
pub struct EncoderDecoderTransformer<T: FloatBounds> {
    /// Transformer encoder
    encoder: TransformerEncoder<T>,
    /// Transformer decoder
    decoder: TransformerDecoder<T>,
    /// Configuration
    config: TransformerConfig<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + From<f64>>
    EncoderDecoderTransformer<T>
{
    /// Create a new encoder-decoder transformer
    pub fn new(config: TransformerConfig<T>) -> NeuralResult<Self> {
        let encoder = TransformerEncoder::new(&config)?;
        let decoder = TransformerDecoder::new(&config)?;

        Ok(Self {
            encoder,
            decoder,
            config,
        })
    }

    /// Forward pass for training
    pub fn forward(
        &mut self,
        src_ids: &Array2<usize>,
        tgt_ids: &Array2<usize>,
        src_mask: Option<&Array2<bool>>,
        tgt_mask: Option<&Array2<bool>>,
        cross_mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        // Encode source sequence
        let encoder_output = self.encoder.forward(src_ids, src_mask, training)?;

        // Decode target sequence
        let decoder_output =
            self.decoder
                .forward(tgt_ids, &encoder_output, tgt_mask, cross_mask, training)?;

        Ok(decoder_output)
    }

    /// Generate sequences auto-regressively
    pub fn generate(
        &mut self,
        src_ids: &Array2<usize>,
        max_length: usize,
        start_token: usize,
        end_token: usize,
        src_mask: Option<&Array2<bool>>,
    ) -> NeuralResult<Array2<usize>> {
        let batch_size = src_ids.nrows();

        // Encode source
        let encoder_output = self.encoder.forward(src_ids, src_mask, false)?;

        // Initialize decoder input with start token
        let mut generated = Array2::from_elem((batch_size, 1), start_token);

        for _ in 1..max_length {
            // Create causal mask for decoder
            let seq_len = generated.ncols();
            let mut tgt_mask = Array2::from_elem((seq_len, seq_len), true);
            for i in 0..seq_len {
                for j in i + 1..seq_len {
                    tgt_mask[[i, j]] = false;
                }
            }

            // Forward pass through decoder
            let logits =
                self.decoder
                    .forward(&generated, &encoder_output, Some(&tgt_mask), None, false)?;

            // Get next token predictions (greedy decoding)
            let last_logits = logits.slice(s![.., -1, ..]);
            let mut next_tokens = Array1::zeros(batch_size);

            for b in 0..batch_size {
                let logits_row = last_logits.slice(s![b, ..]);
                let mut max_idx = 0;
                let mut max_val = *logits_row.iter().next().unwrap();

                for (idx, &val) in logits_row.iter().enumerate() {
                    if val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                next_tokens[b] = max_idx;
            }

            // Append next tokens to generated sequence
            let mut new_generated = Array2::zeros((batch_size, generated.ncols() + 1));
            new_generated
                .slice_mut(s![.., ..generated.ncols()])
                .assign(&generated);
            for b in 0..batch_size {
                new_generated[[b, generated.ncols()]] = next_tokens[b];
            }
            generated = new_generated;

            // Check for end tokens (early stopping)
            let mut all_ended = true;
            for b in 0..batch_size {
                if next_tokens[b] != end_token {
                    all_ended = false;
                    break;
                }
            }
            if all_ended {
                break;
            }
        }

        Ok(generated)
    }

    /// Get configuration
    pub fn config(&self) -> &TransformerConfig<T> {
        &self.config
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
    fn test_transformer_config() {
        let config: TransformerConfig<f64> = TransformerConfig::default();
        assert_eq!(config.d_model, 512);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.d_ff, 2048);
    }

    #[test]
    fn test_feed_forward_network() {
        let ffn = FeedForwardNetwork::new(512, 2048, 0.1, "relu".to_string()).unwrap();
        let input = Array3::from_shape_fn((2, 10, 512), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let output = ffn.forward(&input, false).unwrap();
        assert_eq!(output.dim(), (2, 10, 512));
    }

    #[test]
    fn test_layer_norm() {
        let layer_norm = LayerNorm::<f64>::new(512);
        let input = Array3::from_shape_fn((2, 10, 512), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let output = layer_norm.forward(&input);
        assert_eq!(output.dim(), (2, 10, 512));
    }

    #[test]
    fn test_transformer_encoder_layer() {
        let config: TransformerConfig<f64> = TransformerConfig::default();
        let mut encoder_layer = TransformerEncoderLayer::new(&config).unwrap();

        let input = Array3::from_shape_fn((2, 10, 512), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let output = encoder_layer.forward(&input, None, false).unwrap();
        assert_eq!(output.dim(), (2, 10, 512));
    }

    #[test]
    fn test_transformer_decoder_layer() {
        let config: TransformerConfig<f64> = TransformerConfig::default();
        let mut decoder_layer = TransformerDecoderLayer::new(&config).unwrap();

        let input = Array3::from_shape_fn((2, 10, 512), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let encoder_output = Array3::from_shape_fn((2, 15, 512), |_| {
            let mut rng = thread_rng();
            rng.sample(&Normal::new(0.0, 1.0).unwrap())
        });
        let output = decoder_layer
            .forward(&input, &encoder_output, None, None, false)
            .unwrap();
        assert_eq!(output.dim(), (2, 10, 512));
    }

    #[test]
    fn test_encoder_decoder_transformer() {
        let config: TransformerConfig<f64> = TransformerConfig {
            vocab_size: 1000,
            max_seq_len: 20,
            d_model: 128,
            num_heads: 4,
            d_ff: 256,
            num_encoder_layers: 2,
            num_decoder_layers: 2,
            ..Default::default()
        };

        let mut transformer = EncoderDecoderTransformer::new(config).unwrap();

        let src_ids = Array2::from_elem((2, 10), 1); // Simple input
        let tgt_ids = Array2::from_elem((2, 8), 2); // Simple target

        let output = transformer
            .forward(&src_ids, &tgt_ids, None, None, None, false)
            .unwrap();
        assert_eq!(output.dim(), (2, 8, 1000)); // batch, seq, vocab
    }

    #[test]
    fn test_transformer_generation() {
        let config: TransformerConfig<f64> = TransformerConfig {
            vocab_size: 100,
            max_seq_len: 20,
            d_model: 64,
            num_heads: 2,
            d_ff: 128,
            num_encoder_layers: 1,
            num_decoder_layers: 1,
            ..Default::default()
        };

        let mut transformer = EncoderDecoderTransformer::new(config).unwrap();

        let src_ids = Array2::from_elem((1, 5), 1);
        let generated = transformer.generate(&src_ids, 10, 2, 3, None).unwrap();

        assert_eq!(generated.nrows(), 1);
        assert!(generated.ncols() <= 10);
        assert_eq!(generated[[0, 0]], 2); // Should start with start token
    }
}

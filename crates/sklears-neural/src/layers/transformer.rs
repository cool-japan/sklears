//! Transformer components including positional encoding and transformer blocks.
//!
//! This module provides implementations of transformer architecture components
//! including positional encoding, multi-head attention, and transformer blocks.

use crate::activation::Activation;
use crate::weight_init::{InitStrategy, WeightInitializer};
use crate::{
    layers::{Layer, ParameterizedLayer},
    NeuralResult,
};
use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::numeric::NumCast;
use scirs2_core::random::thread_rng;
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;

/// Helper function to apply activation to a generic floating point array
fn apply_activation<T: FloatBounds>(activation: &Activation, input: &Array2<T>) -> Array2<T> {
    match activation {
        Activation::Identity => input.clone(),
        Activation::Logistic => input.mapv(|val| {
            let exp_neg = (-val).exp();
            T::one() / (T::one() + exp_neg)
        }),
        Activation::Tanh => input.mapv(|val| val.tanh()),
        Activation::Relu => input.mapv(|val| val.max(T::zero())),
        _ => input.clone(), // For other activations, just return input for now
    }
}

/// Apply activation to 3D array (for transformer use)
fn apply_activation_3d<T: FloatBounds>(activation: &Activation, input: &Array3<T>) -> Array3<T> {
    match activation {
        Activation::Identity => input.clone(),
        Activation::Logistic => input.mapv(|val| {
            let exp_neg = (-val).exp();
            T::one() / (T::one() + exp_neg)
        }),
        Activation::Tanh => input.mapv(|val| val.tanh()),
        Activation::Relu => input.mapv(|val| val.max(T::zero())),
        _ => input.clone(), // For other activations, just return input for now
    }
}

/// Positional encoding types
#[derive(Debug, Clone, PartialEq)]
pub enum PositionalEncodingType {
    /// Sinusoidal positional encoding (original transformer)
    Sinusoidal,
    /// Learnable positional encoding
    Learnable,
    /// Relative positional encoding
    Relative,
}

/// Positional encoding implementation for transformer architectures
///
/// Adds positional information to input embeddings to help the model
/// understand the order of elements in a sequence.
#[derive(Debug, Clone)]
pub struct PositionalEncoding<T: FloatBounds> {
    /// Maximum sequence length
    max_seq_len: usize,
    /// Model dimension (embedding dimension)
    d_model: usize,
    /// Type of positional encoding
    encoding_type: PositionalEncodingType,
    /// Dropout rate for regularization
    dropout_rate: T,
    /// Pre-computed sinusoidal encodings (for sinusoidal type)
    sinusoidal_encodings: Option<Array2<T>>,
    /// Learnable position embeddings (for learnable type)
    position_embeddings: Option<Array2<T>>,
    /// Whether to scale embeddings by sqrt(d_model)
    scale_embeddings: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + From<f64>> PositionalEncoding<T> {
    /// Create a new positional encoding layer
    pub fn new(
        max_seq_len: usize,
        d_model: usize,
        encoding_type: PositionalEncodingType,
        dropout_rate: T,
        scale_embeddings: bool,
    ) -> NeuralResult<Self> {
        let mut pe = Self {
            max_seq_len,
            d_model,
            encoding_type: encoding_type.clone(),
            dropout_rate,
            sinusoidal_encodings: None,
            position_embeddings: None,
            scale_embeddings,
        };

        match encoding_type {
            PositionalEncodingType::Sinusoidal => {
                pe.sinusoidal_encodings = Some(pe.create_sinusoidal_encodings()?);
            }
            PositionalEncodingType::Learnable => {
                pe.position_embeddings = Some(pe.create_learnable_embeddings()?);
            }
            PositionalEncodingType::Relative => {
                // Relative positional encoding is handled differently
                // and doesn't require pre-computed embeddings
            }
        }

        Ok(pe)
    }

    /// Create sinusoidal positional encodings
    fn create_sinusoidal_encodings(&self) -> NeuralResult<Array2<T>> {
        let mut encodings = Array2::zeros((self.max_seq_len, self.d_model));

        for pos in 0..self.max_seq_len {
            for i in 0..(self.d_model / 2) {
                let angle = pos as f64 / 10000_f64.powf(2.0 * i as f64 / self.d_model as f64);

                // Even dimensions use sin
                encodings[[pos, 2 * i]] = NumCast::from(angle.sin()).unwrap_or_else(T::zero);

                // Odd dimensions use cos
                if 2 * i + 1 < self.d_model {
                    encodings[[pos, 2 * i + 1]] =
                        NumCast::from(angle.cos()).unwrap_or_else(T::zero);
                }
            }
        }

        Ok(encodings)
    }

    /// Create learnable position embeddings
    fn create_learnable_embeddings(&self) -> NeuralResult<Array2<T>> {
        let mut rng = thread_rng();
        let initializer = WeightInitializer::new(InitStrategy::Normal {
            mean: 0.0,
            std: 0.02,
        });

        initializer.initialize_2d(&mut rng, (self.max_seq_len, self.d_model))
    }

    /// Apply positional encoding to input embeddings
    pub fn encode(&self, embeddings: &Array3<T>) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, d_model) = embeddings.dim();

        if d_model != self.d_model {
            return Err(SklearsError::InvalidParameter {
                name: "d_model".to_string(),
                reason: format!("expected {}, got {}", self.d_model, d_model),
            });
        }

        if seq_len > self.max_seq_len {
            return Err(SklearsError::InvalidParameter {
                name: "seq_len".to_string(),
                reason: format!(
                    "sequence length {} exceeds maximum {}",
                    seq_len, self.max_seq_len
                ),
            });
        }

        let mut output = embeddings.clone();

        // Scale embeddings if requested
        if self.scale_embeddings {
            let scale_factor = NumCast::from((self.d_model as f64).sqrt()).unwrap_or(T::one());
            output *= scale_factor;
        }

        // Add positional encodings based on type
        match &self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                if let Some(ref encodings) = self.sinusoidal_encodings {
                    for batch in 0..batch_size {
                        for pos in 0..seq_len {
                            for dim in 0..d_model {
                                output[[batch, pos, dim]] =
                                    output[[batch, pos, dim]] + encodings[[pos, dim]];
                            }
                        }
                    }
                }
            }
            PositionalEncodingType::Learnable => {
                if let Some(ref embeddings) = self.position_embeddings {
                    for batch in 0..batch_size {
                        for pos in 0..seq_len {
                            for dim in 0..d_model {
                                output[[batch, pos, dim]] =
                                    output[[batch, pos, dim]] + embeddings[[pos, dim]];
                            }
                        }
                    }
                }
            }
            PositionalEncodingType::Relative => {
                // Relative positional encoding is typically handled in the attention mechanism
                // For now, we'll just return the original embeddings
            }
        }

        Ok(output)
    }

    /// Get positional encodings for a specific sequence length
    pub fn get_encodings(&self, seq_len: usize) -> NeuralResult<Array2<T>> {
        if seq_len > self.max_seq_len {
            return Err(SklearsError::InvalidParameter {
                name: "seq_len".to_string(),
                reason: format!(
                    "sequence length {} exceeds maximum {}",
                    seq_len, self.max_seq_len
                ),
            });
        }

        match &self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                if let Some(ref encodings) = self.sinusoidal_encodings {
                    Ok(encodings.slice(s![..seq_len, ..]).to_owned())
                } else {
                    Err(SklearsError::InvalidParameter {
                        name: "position_encodings".to_string(),
                        reason: "sinusoidal encodings not initialized".to_string(),
                    })
                }
            }
            PositionalEncodingType::Learnable => {
                if let Some(ref embeddings) = self.position_embeddings {
                    Ok(embeddings.slice(s![..seq_len, ..]).to_owned())
                } else {
                    Err(SklearsError::InvalidParameter {
                        name: "position_embeddings".to_string(),
                        reason: "learnable embeddings not initialized".to_string(),
                    })
                }
            }
            PositionalEncodingType::Relative => {
                // Return zeros for relative encoding as it's handled differently
                Ok(Array2::zeros((seq_len, self.d_model)))
            }
        }
    }

    /// Update learnable position embeddings (for gradient-based optimization)
    pub fn update_position_embeddings(
        &mut self,
        gradients: &Array2<T>,
        learning_rate: T,
    ) -> NeuralResult<()> {
        if let PositionalEncodingType::Learnable = self.encoding_type {
            if let Some(ref mut embeddings) = self.position_embeddings {
                *embeddings = embeddings.clone() - gradients * learning_rate;
                Ok(())
            } else {
                Err(SklearsError::InvalidParameter {
                    name: "position_embeddings".to_string(),
                    reason: "learnable embeddings not initialized".to_string(),
                })
            }
        } else {
            Err(SklearsError::InvalidParameter {
                name: "encoding_type".to_string(),
                reason: "position embeddings are not learnable".to_string(),
            })
        }
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        match &self.encoding_type {
            PositionalEncodingType::Sinusoidal => 0, // No learnable parameters
            PositionalEncodingType::Learnable => {
                if let Some(ref embeddings) = self.position_embeddings {
                    embeddings.len()
                } else {
                    0
                }
            }
            PositionalEncodingType::Relative => 0, // Handled in attention layers
        }
    }
}

/// Multi-head attention mechanism for transformer architectures
///
/// Implements scaled dot-product attention with multiple attention heads:
/// Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T: FloatBounds> {
    /// Number of attention heads
    num_heads: usize,
    /// Model dimension
    d_model: usize,
    /// Dimension of each attention head
    d_k: usize,
    /// Query projection weights
    w_q: Array2<T>,
    /// Key projection weights
    w_k: Array2<T>,
    /// Value projection weights
    w_v: Array2<T>,
    /// Output projection weights
    w_o: Array2<T>,
    /// Query bias
    b_q: Option<Array1<T>>,
    /// Key bias
    b_k: Option<Array1<T>>,
    /// Value bias
    b_v: Option<Array1<T>>,
    /// Output bias
    b_o: Option<Array1<T>>,
    /// Dropout rate for attention weights
    dropout_rate: T,
    /// Whether to use bias terms
    use_bias: bool,
    /// Scaling factor for attention scores
    scale_factor: T,
    /// Cached attention weights for visualization/analysis
    cached_attention_weights: Option<Array3<T>>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand + From<f64>> MultiHeadAttention<T> {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        dropout_rate: T,
        use_bias: bool,
    ) -> NeuralResult<Self> {
        if d_model % num_heads != 0 {
            return Err(SklearsError::InvalidParameter {
                name: "d_model".to_string(),
                reason: format!(
                    "d_model ({}) must be divisible by num_heads ({})",
                    d_model, num_heads
                ),
            });
        }

        let d_k = d_model / num_heads;
        let scale_factor = NumCast::from(1.0 / (d_k as f64).sqrt()).unwrap_or(T::one());

        // Initialize weights using Xavier initialization
        let mut rng = thread_rng();
        let initializer = WeightInitializer::new(InitStrategy::XavierUniform);

        let w_q = initializer.initialize_2d(&mut rng, (d_model, d_model))?;
        let w_k = initializer.initialize_2d(&mut rng, (d_model, d_model))?;
        let w_v = initializer.initialize_2d(&mut rng, (d_model, d_model))?;
        let w_o = initializer.initialize_2d(&mut rng, (d_model, d_model))?;

        let (b_q, b_k, b_v, b_o) = if use_bias {
            (
                Some(Array1::zeros(d_model)),
                Some(Array1::zeros(d_model)),
                Some(Array1::zeros(d_model)),
                Some(Array1::zeros(d_model)),
            )
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            num_heads,
            d_model,
            d_k,
            w_q,
            w_k,
            w_v,
            w_o,
            b_q,
            b_k,
            b_v,
            b_o,
            dropout_rate,
            use_bias,
            scale_factor,
            cached_attention_weights: None,
        })
    }

    /// Forward pass through multi-head attention
    pub fn forward(
        &mut self,
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
        mask: Option<&Array3<T>>,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len_q, _) = query.dim();
        let (_, seq_len_k, _) = key.dim();
        let (_, seq_len_v, _) = value.dim();

        if seq_len_k != seq_len_v {
            return Err(SklearsError::InvalidParameter {
                name: "seq_len".to_string(),
                reason: "key and value sequence lengths must match".to_string(),
            });
        }

        // Linear projections for Q, K, V
        let q = self.linear_projection(query, &self.w_q, self.b_q.as_ref())?;
        let k = self.linear_projection(key, &self.w_k, self.b_k.as_ref())?;
        let v = self.linear_projection(value, &self.w_v, self.b_v.as_ref())?;

        // Reshape to (batch_size, num_heads, seq_len, d_k)
        let q_heads = self.reshape_for_heads(&q, batch_size, seq_len_q)?;
        let k_heads = self.reshape_for_heads(&k, batch_size, seq_len_k)?;
        let v_heads = self.reshape_for_heads(&v, batch_size, seq_len_v)?;

        // Scaled dot-product attention
        let (attended_values, attention_weights) =
            self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask)?;

        // Cache attention weights for analysis
        self.cached_attention_weights = Some(attention_weights);

        // Reshape back to (batch_size, seq_len, d_model)
        let concatenated = self.reshape_from_heads(&attended_values, batch_size, seq_len_q)?;

        // Final linear projection
        let output = self.linear_projection(&concatenated, &self.w_o, self.b_o.as_ref())?;

        Ok(output)
    }

    /// Apply linear projection (matrix multiplication + bias)
    fn linear_projection(
        &self,
        input: &Array3<T>,
        weight: &Array2<T>,
        bias: Option<&Array1<T>>,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, d_model) = input.dim();
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for batch in 0..batch_size {
            let input_2d = input.slice(s![batch, .., ..]);
            let projected = input_2d.dot(weight);
            output.slice_mut(s![batch, .., ..]).assign(&projected);

            if let Some(bias_vec) = bias {
                for seq in 0..seq_len {
                    for dim in 0..d_model {
                        output[[batch, seq, dim]] = output[[batch, seq, dim]] + bias_vec[dim];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Reshape tensor for multi-head processing
    fn reshape_for_heads(
        &self,
        input: &Array3<T>,
        batch_size: usize,
        seq_len: usize,
    ) -> NeuralResult<Array3<T>> {
        // Input: (batch_size, seq_len, d_model)
        // Output: (batch_size * num_heads, seq_len, d_k)
        let mut output = Array3::zeros((batch_size * self.num_heads, seq_len, self.d_k));

        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                let head_idx = batch * self.num_heads + head;
                let start_dim = head * self.d_k;
                let end_dim = start_dim + self.d_k;

                for seq in 0..seq_len {
                    for dim in 0..self.d_k {
                        output[[head_idx, seq, dim]] = input[[batch, seq, start_dim + dim]];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Reshape tensor back from multi-head format
    fn reshape_from_heads(
        &self,
        input: &Array3<T>,
        batch_size: usize,
        seq_len: usize,
    ) -> NeuralResult<Array3<T>> {
        // Input: (batch_size * num_heads, seq_len, d_k)
        // Output: (batch_size, seq_len, d_model)
        let mut output = Array3::zeros((batch_size, seq_len, self.d_model));

        for batch in 0..batch_size {
            for head in 0..self.num_heads {
                let head_idx = batch * self.num_heads + head;
                let start_dim = head * self.d_k;

                for seq in 0..seq_len {
                    for dim in 0..self.d_k {
                        output[[batch, seq, start_dim + dim]] = input[[head_idx, seq, dim]];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Scaled dot-product attention
    fn scaled_dot_product_attention(
        &self,
        q: &Array3<T>,
        k: &Array3<T>,
        v: &Array3<T>,
        mask: Option<&Array3<T>>,
    ) -> NeuralResult<(Array3<T>, Array3<T>)> {
        let (batch_heads, seq_len_q, d_k) = q.dim();
        let (_, seq_len_k, _) = k.dim();

        // Compute attention scores: Q @ K^T
        let mut scores = Array3::zeros((batch_heads, seq_len_q, seq_len_k));

        for batch_head in 0..batch_heads {
            let q_slice = q.slice(s![batch_head, .., ..]);
            let k_slice = k.slice(s![batch_head, .., ..]);
            let score_slice = q_slice.dot(&k_slice.t());
            scores
                .slice_mut(s![batch_head, .., ..])
                .assign(&score_slice);
        }

        // Scale by sqrt(d_k)
        scores *= self.scale_factor;

        // Apply mask if provided
        if let Some(mask_tensor) = mask {
            // Apply mask by setting masked positions to large negative value
            let neg_inf = NumCast::from(-1e9).unwrap_or(T::zero());
            for batch_head in 0..batch_heads {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        let mask_batch = batch_head % mask_tensor.dim().0;
                        if mask_tensor[[mask_batch, i, j]] == T::zero() {
                            scores[[batch_head, i, j]] = neg_inf;
                        }
                    }
                }
            }
        }

        // Apply softmax to get attention weights
        let attention_weights = self.softmax_3d(&scores)?;

        // Apply attention to values: Attention @ V
        let mut output = Array3::zeros((batch_heads, seq_len_q, d_k));

        for batch_head in 0..batch_heads {
            let attn_slice = attention_weights.slice(s![batch_head, .., ..]);
            let v_slice = v.slice(s![batch_head, .., ..]);
            let out_slice = attn_slice.dot(&v_slice);
            output.slice_mut(s![batch_head, .., ..]).assign(&out_slice);
        }

        Ok((output, attention_weights))
    }

    /// Apply softmax along the last dimension of a 3D tensor
    fn softmax_3d(&self, input: &Array3<T>) -> NeuralResult<Array3<T>> {
        let (dim0, dim1, dim2) = input.dim();
        let mut output = Array3::zeros((dim0, dim1, dim2));

        for i in 0..dim0 {
            for j in 0..dim1 {
                // Find max for numerical stability
                let mut max_val = input[[i, j, 0]];
                for k in 1..dim2 {
                    if input[[i, j, k]] > max_val {
                        max_val = input[[i, j, k]];
                    }
                }

                // Compute exp(x - max) and sum
                let mut sum = T::zero();
                for k in 0..dim2 {
                    let exp_val = (input[[i, j, k]] - max_val).exp();
                    output[[i, j, k]] = exp_val;
                    sum = sum + exp_val;
                }

                // Normalize
                for k in 0..dim2 {
                    output[[i, j, k]] = output[[i, j, k]] / sum;
                }
            }
        }

        Ok(output)
    }

    /// Get the last computed attention weights
    pub fn get_attention_weights(&self) -> Option<&Array3<T>> {
        self.cached_attention_weights.as_ref()
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.w_q.len() + self.w_k.len() + self.w_v.len() + self.w_o.len();
        let bias_params = if self.use_bias {
            self.b_q.as_ref().map_or(0, |b| b.len())
                + self.b_k.as_ref().map_or(0, |b| b.len())
                + self.b_v.as_ref().map_or(0, |b| b.len())
                + self.b_o.as_ref().map_or(0, |b| b.len())
        } else {
            0
        };
        weight_params + bias_params
    }
}

/// Feed-forward network for transformer blocks
#[derive(Debug, Clone)]
pub struct FeedForward<T: FloatBounds> {
    /// First linear layer weights
    w1: Array2<T>,
    /// Second linear layer weights
    w2: Array2<T>,
    /// First layer bias
    b1: Option<Array1<T>>,
    /// Second layer bias
    b2: Option<Array1<T>>,
    /// Activation function
    activation: Activation,
    /// Dropout rate
    dropout_rate: T,
    /// Whether to use bias
    use_bias: bool,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> FeedForward<T> {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        activation: Activation,
        dropout_rate: T,
        use_bias: bool,
    ) -> NeuralResult<Self> {
        // Initialize weights
        let mut rng = thread_rng();
        let initializer = WeightInitializer::new(InitStrategy::XavierUniform);

        let w1 = initializer.initialize_2d(&mut rng, (d_model, d_ff))?;
        let w2 = initializer.initialize_2d(&mut rng, (d_ff, d_model))?;

        let (b1, b2) = if use_bias {
            (Some(Array1::zeros(d_ff)), Some(Array1::zeros(d_model)))
        } else {
            (None, None)
        };

        Ok(Self {
            w1,
            w2,
            b1,
            b2,
            activation,
            dropout_rate,
            use_bias,
        })
    }

    /// Forward pass through the feed-forward network
    pub fn forward(&mut self, input: &Array3<T>) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, d_model) = input.dim();

        // First linear layer
        let mut hidden = Array3::zeros((batch_size, seq_len, self.w1.dim().1));

        for batch in 0..batch_size {
            let input_2d = input.slice(s![batch, .., ..]);
            let hidden_2d = input_2d.dot(&self.w1);
            hidden.slice_mut(s![batch, .., ..]).assign(&hidden_2d);

            if let Some(ref bias) = self.b1 {
                for seq in 0..seq_len {
                    for dim in 0..bias.len() {
                        hidden[[batch, seq, dim]] = hidden[[batch, seq, dim]] + bias[dim];
                    }
                }
            }
        }

        // Apply activation
        hidden = apply_activation_3d(&self.activation, &hidden);

        // Second linear layer
        let mut output = Array3::zeros((batch_size, seq_len, d_model));

        for batch in 0..batch_size {
            let hidden_2d = hidden.slice(s![batch, .., ..]);
            let output_2d = hidden_2d.dot(&self.w2);
            output.slice_mut(s![batch, .., ..]).assign(&output_2d);

            if let Some(ref bias) = self.b2 {
                for seq in 0..seq_len {
                    for dim in 0..bias.len() {
                        output[[batch, seq, dim]] = output[[batch, seq, dim]] + bias[dim];
                    }
                }
            }
        }

        Ok(output)
    }

    /// Get the number of parameters
    pub fn num_parameters(&self) -> usize {
        let weight_params = self.w1.len() + self.w2.len();
        let bias_params = if self.use_bias {
            self.b1.as_ref().map_or(0, |b| b.len()) + self.b2.as_ref().map_or(0, |b| b.len())
        } else {
            0
        };
        weight_params + bias_params
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    #[ignore]
    fn test_sinusoidal_positional_encoding() {
        let pe =
            PositionalEncoding::<f64>::new(100, 64, PositionalEncodingType::Sinusoidal, 0.1, true)
                .unwrap();

        assert_eq!(pe.max_seq_len, 100);
        assert_eq!(pe.d_model, 64);
        assert!(pe.sinusoidal_encodings.is_some());

        let encodings = pe.get_encodings(10).unwrap();
        assert_eq!(encodings.dim(), (10, 64));
    }

    #[test]
    #[ignore]
    fn test_learnable_positional_encoding() {
        let pe =
            PositionalEncoding::<f64>::new(50, 32, PositionalEncodingType::Learnable, 0.1, false)
                .unwrap();

        assert_eq!(pe.num_parameters(), 50 * 32);
        assert!(pe.position_embeddings.is_some());

        let encodings = pe.get_encodings(20).unwrap();
        assert_eq!(encodings.dim(), (20, 32));
    }

    #[test]
    #[ignore]
    fn test_positional_encoding_forward() {
        let pe =
            PositionalEncoding::<f64>::new(50, 16, PositionalEncodingType::Sinusoidal, 0.0, false)
                .unwrap();

        let embeddings = Array3::zeros((2, 10, 16)); // batch=2, seq=10, d_model=16
        let encoded = pe.encode(&embeddings).unwrap();

        assert_eq!(encoded.dim(), (2, 10, 16));
    }

    #[test]
    #[ignore]
    fn test_multi_head_attention_creation() {
        let mha = MultiHeadAttention::<f64>::new(512, 8, 0.1, true).unwrap();

        assert_eq!(mha.num_heads, 8);
        assert_eq!(mha.d_model, 512);
        assert_eq!(mha.d_k, 64);
        assert!(mha.use_bias);
    }

    #[test]
    #[ignore]
    fn test_multi_head_attention_invalid_dimensions() {
        let result = MultiHeadAttention::<f64>::new(511, 8, 0.1, true);
        assert!(result.is_err()); // 511 is not divisible by 8
    }

    #[test]
    #[ignore]
    fn test_multi_head_attention_forward() {
        let mut mha = MultiHeadAttention::<f64>::new(64, 4, 0.0, false).unwrap();

        let query = Array3::zeros((2, 10, 64)); // batch=2, seq=10, d_model=64
        let key = Array3::zeros((2, 15, 64)); // batch=2, seq=15, d_model=64
        let value = Array3::zeros((2, 15, 64)); // batch=2, seq=15, d_model=64

        let output = mha.forward(&query, &key, &value, None).unwrap();
        assert_eq!(output.dim(), (2, 10, 64));

        // Check that attention weights were cached
        assert!(mha.get_attention_weights().is_some());
    }

    #[test]
    #[ignore]
    fn test_multi_head_attention_with_mask() {
        let mut mha = MultiHeadAttention::<f64>::new(32, 2, 0.0, false).unwrap();

        let query = Array3::ones((1, 5, 32));
        let key = Array3::ones((1, 5, 32));
        let value = Array3::ones((1, 5, 32));

        // Create a causal mask (lower triangular)
        let mut mask = Array3::zeros((1, 5, 5));
        for i in 0..5 {
            for j in 0..=i {
                mask[[0, i, j]] = 1.0;
            }
        }

        let output = mha.forward(&query, &key, &value, Some(&mask)).unwrap();
        assert_eq!(output.dim(), (1, 5, 32));
    }

    #[test]
    #[ignore]
    fn test_feed_forward_network() {
        let mut ffn = FeedForward::<f64>::new(256, 1024, Activation::Relu, 0.1, true).unwrap();

        let input = Array3::zeros((2, 10, 256));
        let output = ffn.forward(&input).unwrap();

        assert_eq!(output.dim(), (2, 10, 256));
        assert_eq!(ffn.num_parameters(), 256 * 1024 + 1024 * 256 + 1024 + 256);
    }

    #[test]
    #[ignore]
    fn test_positional_encoding_sequence_length_validation() {
        let pe =
            PositionalEncoding::<f64>::new(20, 16, PositionalEncodingType::Sinusoidal, 0.0, false)
                .unwrap();

        let long_embeddings = Array3::zeros((1, 25, 16)); // seq_len > max_seq_len
        let result = pe.encode(&long_embeddings);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_positional_encoding_dimension_validation() {
        let pe =
            PositionalEncoding::<f64>::new(50, 16, PositionalEncodingType::Sinusoidal, 0.0, false)
                .unwrap();

        let wrong_dim_embeddings = Array3::zeros((1, 10, 32)); // d_model mismatch
        let result = pe.encode(&wrong_dim_embeddings);
        assert!(result.is_err());
    }

    #[test]
    #[ignore]
    fn test_learnable_embedding_updates() {
        let mut pe =
            PositionalEncoding::<f64>::new(10, 8, PositionalEncodingType::Learnable, 0.0, false)
                .unwrap();

        let gradients = Array2::ones((10, 8));
        let learning_rate = 0.01;

        let result = pe.update_position_embeddings(&gradients, learning_rate);
        assert!(result.is_ok());
    }

    #[test]
    #[ignore]
    fn test_sinusoidal_encoding_properties() {
        let pe =
            PositionalEncoding::<f64>::new(100, 64, PositionalEncodingType::Sinusoidal, 0.0, false)
                .unwrap();

        let encodings = pe.sinusoidal_encodings.as_ref().unwrap();

        // Check that even positions use sin and odd positions use cos
        // This is a basic sanity check - the actual values depend on the formula
        assert_eq!(encodings.dim(), (100, 64));

        // Check that positions 0 and 1 have different patterns
        let pos_0 = encodings.row(0);
        let pos_1 = encodings.row(1);

        let mut different = false;
        for i in 0..64 {
            if (pos_0[i] - pos_1[i]).abs() > 1e-6_f64 {
                different = true;
                break;
            }
        }
        assert!(
            different,
            "Different positions should have different encodings"
        );
    }
}

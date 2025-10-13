//! Attention mechanism implementations for neural networks.
//!
//! This module provides various attention mechanisms including scaled dot-product
//! attention, multi-head attention, and self-attention for transformer architectures.

use crate::layers::{Layer, LayerConfig, ParameterizedLayer};
use crate::NeuralResult;
use scirs2_core::ndarray::{Array1, Array2, Array3, Axis};
use sklears_core::error::SklearsError;
use sklears_core::types::FloatBounds;
use std::marker::PhantomData;

/// Scaled Dot-Product Attention implementation.
///
/// # Mathematical Formulation
///
/// Attention(Q, K, V) = softmax(QK^T / √d_k)V
///
/// Where:
/// - Q: Query matrix
/// - K: Key matrix  
/// - V: Value matrix
/// - d_k: Dimension of key vectors
#[derive(Debug, Clone)]
pub struct ScaledDotProductAttention<T: FloatBounds> {
    /// Dimension of key vectors
    d_k: usize,
    /// Whether to apply dropout to attention weights
    dropout_rate: Option<T>,
    /// Last attention weights (for visualization)
    last_attention_weights: Option<Array2<T>>,
    /// Configuration
    config: LayerConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> ScaledDotProductAttention<T> {
    /// Create a new scaled dot-product attention layer.
    ///
    /// # Arguments
    /// * `d_k` - Dimension of key vectors
    /// * `dropout_rate` - Optional dropout rate for attention weights
    pub fn new(d_k: usize, dropout_rate: Option<T>) -> Self {
        Self {
            d_k,
            dropout_rate,
            last_attention_weights: None,
            config: LayerConfig::new(d_k),
            _phantom: PhantomData,
        }
    }

    /// Apply attention mechanism.
    ///
    /// # Arguments
    /// * `query` - Query matrix (batch_size, seq_len_q, d_k)
    /// * `key` - Key matrix (batch_size, seq_len_k, d_k)
    /// * `value` - Value matrix (batch_size, seq_len_v, d_v)
    /// * `mask` - Optional attention mask
    pub fn apply_attention(
        &mut self,
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
        mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len_q, _) = query.dim();
        let (_, seq_len_k, _) = key.dim();
        let (_, seq_len_v, d_v) = value.dim();

        if seq_len_k != seq_len_v {
            return Err(SklearsError::InvalidInput(
                "Key and Value sequences must have the same length".to_string(),
            ));
        }

        let scale = T::one() / T::from(self.d_k).unwrap_or(T::one()).sqrt();
        let mut output = Array3::zeros((batch_size, seq_len_q, d_v));

        for b in 0..batch_size {
            let q = query.index_axis(Axis(0), b);
            let k = key.index_axis(Axis(0), b);
            let v = value.index_axis(Axis(0), b);

            // Compute attention scores: QK^T / √d_k
            let scores = q.dot(&k.t()) * scale;

            // Apply mask if provided
            let mut masked_scores = scores;
            if let Some(mask) = mask {
                for i in 0..seq_len_q {
                    for j in 0..seq_len_k {
                        if !mask[[i, j]] {
                            masked_scores[[i, j]] = T::from(-1e9)
                                .unwrap_or(-T::one() * T::from(1000000000).unwrap_or(T::one()));
                        }
                    }
                }
            }

            // Apply softmax to get attention weights
            let attention_weights = self.softmax_2d(&masked_scores);

            // Store attention weights for visualization
            self.last_attention_weights = Some(attention_weights.clone());

            // Apply dropout if specified and training
            let final_weights = if training && self.dropout_rate.is_some() {
                self.apply_dropout(&attention_weights)
            } else {
                attention_weights
            };

            // Compute output: attention_weights * V
            let batch_output = final_weights.dot(&v);
            output.index_axis_mut(Axis(0), b).assign(&batch_output);
        }

        Ok(output)
    }

    /// Get the last computed attention weights.
    pub fn get_attention_weights(&self) -> Option<&Array2<T>> {
        self.last_attention_weights.as_ref()
    }

    /// Apply softmax to 2D array along the last axis.
    fn softmax_2d(&self, x: &Array2<T>) -> Array2<T> {
        let mut result = Array2::zeros(x.dim());

        for (i, (input_row, mut output_row)) in x
            .axis_iter(Axis(0))
            .zip(result.axis_iter_mut(Axis(0)))
            .enumerate()
        {
            // Find max for numerical stability
            let max_val = input_row.fold(
                T::from(-f64::INFINITY).unwrap_or(-T::one() * T::from(1e10).unwrap_or(T::one())),
                |acc, &x| {
                    if x > acc {
                        x
                    } else {
                        acc
                    }
                },
            );

            // Compute exp values
            let mut sum = T::zero();
            for (j, (&input_val, output_val)) in
                input_row.iter().zip(output_row.iter_mut()).enumerate()
            {
                let exp_val = (input_val - max_val).exp();
                *output_val = exp_val;
                sum = sum + exp_val;
            }

            // Normalize
            for output_val in output_row.iter_mut() {
                *output_val = *output_val / sum;
            }
        }

        result
    }

    /// Apply dropout to attention weights.
    fn apply_dropout(&self, weights: &Array2<T>) -> Array2<T> {
        // Simple dropout implementation - in practice would use proper randomization
        let dropout_rate = self.dropout_rate.unwrap_or(T::zero());
        let keep_prob = T::one() - dropout_rate;
        weights * keep_prob
    }
}

/// Multi-Head Attention layer.
///
/// # Mathematical Formulation
///
/// MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
/// where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
#[derive(Debug, Clone)]
pub struct MultiHeadAttention<T: FloatBounds> {
    /// Number of attention heads
    num_heads: usize,
    /// Dimension of model
    d_model: usize,
    /// Dimension of each head
    d_k: usize,
    /// Dimension of values
    d_v: usize,
    /// Query projection weights for each head
    w_q: Vec<Array2<T>>,
    /// Key projection weights for each head
    w_k: Vec<Array2<T>>,
    /// Value projection weights for each head
    w_v: Vec<Array2<T>>,
    /// Output projection weights
    w_o: Array2<T>,
    /// Scaled dot-product attention layers
    attention_layers: Vec<ScaledDotProductAttention<T>>,
    /// Last input for backward pass
    last_input: Option<(Array3<T>, Array3<T>, Array3<T>)>,
    /// Configuration
    config: LayerConfig<T>,
    _phantom: PhantomData<T>,
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> MultiHeadAttention<T> {
    /// Create a new multi-head attention layer.
    ///
    /// # Arguments
    /// * `num_heads` - Number of attention heads
    /// * `d_model` - Dimension of the model
    /// * `dropout_rate` - Optional dropout rate
    pub fn new(num_heads: usize, d_model: usize, dropout_rate: Option<T>) -> NeuralResult<Self> {
        if d_model % num_heads != 0 {
            return Err(SklearsError::InvalidInput(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                d_model, num_heads
            )));
        }

        let d_k = d_model / num_heads;
        let d_v = d_k; // Typically same as d_k

        // Initialize projection weights
        let mut w_q = Vec::new();
        let mut w_k = Vec::new();
        let mut w_v = Vec::new();
        let mut attention_layers = Vec::new();

        for _ in 0..num_heads {
            w_q.push(Self::initialize_weights(d_model, d_k)?);
            w_k.push(Self::initialize_weights(d_model, d_k)?);
            w_v.push(Self::initialize_weights(d_model, d_v)?);
            attention_layers.push(ScaledDotProductAttention::new(d_k, dropout_rate));
        }

        let w_o = Self::initialize_weights(d_model, d_model)?;

        Ok(Self {
            num_heads,
            d_model,
            d_k,
            d_v,
            w_q,
            w_k,
            w_v,
            w_o,
            attention_layers,
            last_input: None,
            config: LayerConfig::new(d_model),
            _phantom: PhantomData,
        })
    }

    /// Initialize weight matrix with Xavier/Glorot initialization.
    fn initialize_weights(input_size: usize, output_size: usize) -> NeuralResult<Array2<T>> {
        let scale = T::from(2.0).unwrap_or(T::one() + T::one())
            / T::from(input_size + output_size).unwrap_or(T::one());
        let bound = scale.sqrt();

        let mut weights = Array2::zeros((input_size, output_size));
        for elem in weights.iter_mut() {
            // Simple initialization - in practice would use proper random number generation
            *elem = bound * T::from(0.1).unwrap_or(T::one() / T::from(10).unwrap());
        }

        Ok(weights)
    }

    /// Apply multi-head attention.
    pub fn apply(
        &mut self,
        query: &Array3<T>,
        key: &Array3<T>,
        value: &Array3<T>,
        mask: Option<&Array2<bool>>,
        training: bool,
    ) -> NeuralResult<Array3<T>> {
        self.last_input = Some((query.clone(), key.clone(), value.clone()));

        let (batch_size, seq_len, _) = query.dim();
        let mut head_outputs = Vec::new();

        // Apply each attention head
        for h in 0..self.num_heads {
            // Project Q, K, V for this head
            let q_proj = self.project_sequence(query, &self.w_q[h])?;
            let k_proj = self.project_sequence(key, &self.w_k[h])?;
            let v_proj = self.project_sequence(value, &self.w_v[h])?;

            // Apply attention
            let head_output = self.attention_layers[h]
                .apply_attention(&q_proj, &k_proj, &v_proj, mask, training)?;

            head_outputs.push(head_output);
        }

        // Concatenate heads
        let mut concatenated = Array3::zeros((batch_size, seq_len, self.d_model));
        for (h, head_output) in head_outputs.iter().enumerate() {
            let start_idx = h * self.d_v;
            let end_idx = start_idx + self.d_v;

            for b in 0..batch_size {
                for s in 0..seq_len {
                    for d in 0..self.d_v {
                        concatenated[[b, s, start_idx + d]] = head_output[[b, s, d]];
                    }
                }
            }
        }

        // Apply output projection
        let output = self.project_sequence(&concatenated, &self.w_o)?;

        Ok(output)
    }

    /// Project a sequence through a weight matrix.
    fn project_sequence(&self, input: &Array3<T>, weights: &Array2<T>) -> NeuralResult<Array3<T>> {
        let (batch_size, seq_len, input_dim) = input.dim();
        let output_dim = weights.dim().1;

        let mut output = Array3::zeros((batch_size, seq_len, output_dim));

        for b in 0..batch_size {
            for s in 0..seq_len {
                let input_slice = input.index_axis(Axis(0), b);
                let input_vec = input_slice.index_axis(Axis(0), s);
                let output_vec = input_vec.dot(weights);
                output
                    .index_axis_mut(Axis(0), b)
                    .index_axis_mut(Axis(0), s)
                    .assign(&output_vec);
            }
        }

        Ok(output)
    }

    /// Get attention weights from all heads.
    pub fn get_all_attention_weights(&self) -> Vec<Option<&Array2<T>>> {
        self.attention_layers
            .iter()
            .map(|layer| layer.get_attention_weights())
            .collect()
    }
}

impl<T: FloatBounds + scirs2_core::ndarray::ScalarOperand> Layer<T> for MultiHeadAttention<T> {
    fn forward(&mut self, input: &Array2<T>, training: bool) -> NeuralResult<Array2<T>> {
        // Convert 2D input to 3D for sequence processing
        let (batch_size, features) = input.dim();
        let seq_len = 1; // Treat as single timestep

        let input_3d = input
            .clone()
            .into_shape((batch_size, seq_len, features))
            .map_err(|_| SklearsError::InvalidInput("Failed to reshape input".to_string()))?;

        // Self-attention: Q = K = V = input
        let output_3d = self.apply(&input_3d, &input_3d, &input_3d, None, training)?;

        // Convert back to 2D
        let output = output_3d
            .into_shape((batch_size, features))
            .map_err(|_| SklearsError::InvalidInput("Failed to reshape output".to_string()))?;

        Ok(output)
    }

    fn backward(&mut self, _grad_output: &Array2<T>) -> NeuralResult<Array2<T>> {
        // Simplified backward pass - in practice would compute full gradients
        // through all projection matrices and attention mechanisms
        Err(SklearsError::NotImplemented(
            "Full backward pass for MultiHeadAttention not yet implemented".to_string(),
        ))
    }

    fn num_parameters(&self) -> usize {
        let head_params = self.num_heads * (self.d_model * self.d_k * 3); // Q, K, V projections
        let output_params = self.d_model * self.d_model; // Output projection
        head_params + output_params
    }

    fn reset(&mut self) {
        self.last_input = None;
        for layer in &mut self.attention_layers {
            layer.last_attention_weights = None;
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Array3;

    #[test]
    fn test_scaled_dot_product_attention() {
        let mut attention = ScaledDotProductAttention::<f64>::new(4, None);

        let batch_size = 2;
        let seq_len = 3;
        let d_k = 4;
        let d_v = 4;

        let query = Array3::ones((batch_size, seq_len, d_k));
        let key = Array3::ones((batch_size, seq_len, d_k));
        let value = Array3::ones((batch_size, seq_len, d_v));

        let output = attention
            .apply_attention(&query, &key, &value, None, false)
            .unwrap();

        assert_eq!(output.dim(), (batch_size, seq_len, d_v));

        // Check that attention weights were computed
        assert!(attention.get_attention_weights().is_some());
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let attention = MultiHeadAttention::<f64>::new(8, 512, None).unwrap();

        assert_eq!(attention.num_heads, 8);
        assert_eq!(attention.d_model, 512);
        assert_eq!(attention.d_k, 64); // 512 / 8

        // Should have correct number of weight matrices
        assert_eq!(attention.w_q.len(), 8);
        assert_eq!(attention.w_k.len(), 8);
        assert_eq!(attention.w_v.len(), 8);
    }

    #[test]
    fn test_multi_head_attention_invalid_dimensions() {
        // d_model not divisible by num_heads should fail
        let result = MultiHeadAttention::<f64>::new(7, 512, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_head_attention_forward() {
        let mut attention = MultiHeadAttention::<f64>::new(4, 8, None).unwrap();

        let batch_size = 2;
        let seq_len = 3;
        let d_model = 8;

        let query = Array3::ones((batch_size, seq_len, d_model));
        let key = Array3::ones((batch_size, seq_len, d_model));
        let value = Array3::ones((batch_size, seq_len, d_model));

        let output = attention.apply(&query, &key, &value, None, false).unwrap();

        assert_eq!(output.dim(), (batch_size, seq_len, d_model));
    }

    #[test]
    fn test_attention_with_mask() {
        let mut attention = ScaledDotProductAttention::<f64>::new(4, None);

        let batch_size = 1;
        let seq_len = 3;
        let d_k = 4;

        let query = Array3::ones((batch_size, seq_len, d_k));
        let key = Array3::ones((batch_size, seq_len, d_k));
        let value = Array3::ones((batch_size, seq_len, d_k));

        // Create mask that blocks future positions (causal mask)
        let mut mask = Array2::from_elem((seq_len, seq_len), true);
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask[[i, j]] = false;
            }
        }

        let output = attention
            .apply_attention(&query, &key, &value, Some(&mask), false)
            .unwrap();

        assert_eq!(output.dim(), (batch_size, seq_len, d_k));

        // Verify attention weights respect the mask
        let attention_weights = attention.get_attention_weights().unwrap();

        // Upper triangular part should have very small weights due to masking
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                assert!(attention_weights[[i, j]] < 1e-6);
            }
        }
    }

    #[test]
    fn test_softmax_2d() {
        let attention = ScaledDotProductAttention::<f64>::new(4, None);
        let input = scirs2_core::ndarray::array![[1.0, 2.0, 3.0], [0.0, 1.0, 0.0]];

        let output = attention.softmax_2d(&input);

        // Each row should sum to 1
        for row in output.axis_iter(scirs2_core::ndarray::Axis(0)) {
            let sum: f64 = row.sum();
            assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
        }

        // All values should be positive
        for &val in output.iter() {
            assert!(val >= 0.0);
        }
    }
}

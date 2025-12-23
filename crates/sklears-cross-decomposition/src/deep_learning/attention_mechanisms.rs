//! Attention Mechanisms for Cross-Modal Learning
//!
//! This module provides comprehensive attention-based architectures for cross-decomposition,
//! including self-attention, cross-attention, multi-head attention, and transformer blocks.
//!
//! ## Attention Types
//! - Self-Attention: Attention within a single modality
//! - Cross-Attention: Attention between different modalities
//! - Multi-Head Attention: Parallel attention heads for diverse representations
//! - Scaled Dot-Product Attention: Standard attention mechanism with scaling
//!
//! ## Applications
//! - Cross-modal fusion with attention-based weights
//! - Feature alignment across different data modalities
//! - Interpretable cross-modal relationships
//! - Enhanced canonical correlation analysis with attention

use scirs2_core::ndarray::{Array1, Array2, Array3, ArrayView2, Axis};
use scirs2_core::random::{thread_rng, CoreRandom, Rng};
use sklears_core::types::Float;
use std::f64::consts::PI;

/// Attention mechanism type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionType {
    /// Self-attention within a single modality
    SelfAttention,
    /// Cross-attention between modalities
    CrossAttention,
    /// Multi-head attention
    MultiHead,
    /// Additive attention (Bahdanau)
    Additive,
}

/// Activation function for attention layers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttentionActivation {
    /// Softmax (standard for attention)
    Softmax,
    /// Sigmoid
    Sigmoid,
    /// Tanh
    Tanh,
    /// ReLU
    ReLU,
    /// Sparsemax (for sparse attention)
    Sparsemax,
}

/// Configuration for attention mechanisms
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Type of attention mechanism
    pub attention_type: AttentionType,
    /// Number of attention heads (for multi-head attention)
    pub num_heads: usize,
    /// Dimension of queries
    pub query_dim: usize,
    /// Dimension of keys
    pub key_dim: usize,
    /// Dimension of values
    pub value_dim: usize,
    /// Hidden dimension for each head
    pub head_dim: usize,
    /// Dropout rate
    pub dropout: Float,
    /// Whether to use bias in linear layers
    pub use_bias: bool,
    /// Activation function for attention scores
    pub activation: AttentionActivation,
    /// Temperature for softmax (controls sharpness)
    pub temperature: Float,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            attention_type: AttentionType::MultiHead,
            num_heads: 8,
            query_dim: 64,
            key_dim: 64,
            value_dim: 64,
            head_dim: 64,
            dropout: 0.1,
            use_bias: true,
            activation: AttentionActivation::Softmax,
            temperature: 1.0,
        }
    }
}

/// Attention layer with learnable parameters
#[derive(Debug, Clone)]
pub struct AttentionLayer {
    /// Configuration
    config: AttentionConfig,
    /// Query projection matrix (d_model, d_k * num_heads)
    query_weights: Array2<Float>,
    /// Key projection matrix (d_model, d_k * num_heads)
    key_weights: Array2<Float>,
    /// Value projection matrix (d_model, d_v * num_heads)
    value_weights: Array2<Float>,
    /// Output projection matrix (d_v * num_heads, d_model)
    output_weights: Array2<Float>,
    /// Query bias
    query_bias: Option<Array1<Float>>,
    /// Key bias
    key_bias: Option<Array1<Float>>,
    /// Value bias
    value_bias: Option<Array1<Float>>,
    /// Output bias
    output_bias: Option<Array1<Float>>,
}

/// Results from attention computation
#[derive(Debug, Clone)]
pub struct AttentionOutput {
    /// Attention output (batch_size, seq_len, d_model)
    pub output: Array2<Float>,
    /// Attention weights (batch_size, num_heads, seq_len_q, seq_len_k)
    pub attention_weights: Array3<Float>,
    /// Per-head outputs before final projection
    pub head_outputs: Vec<Array2<Float>>,
}

/// Multi-head attention mechanism
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Attention configuration
    config: AttentionConfig,
    /// Attention layer
    layer: AttentionLayer,
}

/// Cross-attention for combining two modalities
#[derive(Debug, Clone)]
pub struct CrossModalAttention {
    /// Attention from X to Y
    x_to_y_attention: MultiHeadAttention,
    /// Attention from Y to X
    y_to_x_attention: MultiHeadAttention,
    /// Fusion weights
    fusion_weights: Array2<Float>,
}

/// Results from cross-modal attention
#[derive(Debug, Clone)]
pub struct CrossModalAttentionOutput {
    /// Fused representation
    pub fused_output: Array2<Float>,
    /// X-to-Y attention output
    pub x_to_y_output: AttentionOutput,
    /// Y-to-X attention output
    pub y_to_x_output: AttentionOutput,
    /// Fusion weights
    pub fusion_attention_weights: Array2<Float>,
}

/// Transformer encoder block
#[derive(Debug, Clone)]
pub struct TransformerEncoderBlock {
    /// Multi-head self-attention
    self_attention: MultiHeadAttention,
    /// Feed-forward network weights (layer 1)
    ffn_weights_1: Array2<Float>,
    /// Feed-forward network weights (layer 2)
    ffn_weights_2: Array2<Float>,
    /// FFN bias 1
    ffn_bias_1: Option<Array1<Float>>,
    /// FFN bias 2
    ffn_bias_2: Option<Array1<Float>>,
    /// Layer normalization parameters
    ln1_gamma: Array1<Float>,
    ln1_beta: Array1<Float>,
    ln2_gamma: Array1<Float>,
    ln2_beta: Array1<Float>,
    /// Dropout rate
    dropout: Float,
    /// FFN hidden dimension
    ffn_dim: usize,
}

/// Transformer decoder block with cross-attention
#[derive(Debug, Clone)]
pub struct TransformerDecoderBlock {
    /// Self-attention on decoder input
    self_attention: MultiHeadAttention,
    /// Cross-attention between encoder and decoder
    cross_attention: MultiHeadAttention,
    /// Feed-forward network weights (layer 1)
    ffn_weights_1: Array2<Float>,
    /// Feed-forward network weights (layer 2)
    ffn_weights_2: Array2<Float>,
    /// FFN bias 1
    ffn_bias_1: Option<Array1<Float>>,
    /// FFN bias 2
    ffn_bias_2: Option<Array1<Float>>,
    /// Layer normalization parameters (3 layers)
    ln1_gamma: Array1<Float>,
    ln1_beta: Array1<Float>,
    ln2_gamma: Array1<Float>,
    ln2_beta: Array1<Float>,
    ln3_gamma: Array1<Float>,
    ln3_beta: Array1<Float>,
    /// Dropout rate
    dropout: Float,
    /// FFN hidden dimension
    ffn_dim: usize,
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(
        attention_type: AttentionType,
        num_heads: usize,
        query_dim: usize,
        key_dim: usize,
        value_dim: usize,
    ) -> Self {
        Self {
            attention_type,
            num_heads,
            query_dim,
            key_dim,
            value_dim,
            head_dim: key_dim / num_heads,
            ..Default::default()
        }
    }

    /// Set dropout rate
    pub fn dropout(mut self, dropout: Float) -> Self {
        self.dropout = dropout;
        self
    }

    /// Set temperature
    pub fn temperature(mut self, temperature: Float) -> Self {
        self.temperature = temperature;
        self
    }

    /// Set activation function
    pub fn activation(mut self, activation: AttentionActivation) -> Self {
        self.activation = activation;
        self
    }
}

impl AttentionLayer {
    /// Create a new attention layer with random initialization
    pub fn new(config: AttentionConfig) -> Self {
        let mut rng = thread_rng();

        // Xavier initialization scale
        let query_scale =
            (2.0 / (config.query_dim + config.key_dim * config.num_heads) as Float).sqrt();
        let key_scale = query_scale;
        let value_scale =
            (2.0 / (config.query_dim + config.value_dim * config.num_heads) as Float).sqrt();
        let output_scale =
            (2.0 / (config.value_dim * config.num_heads + config.query_dim) as Float).sqrt();

        // Total dimensions for all heads
        let total_key_dim = config.head_dim * config.num_heads;
        let total_value_dim = config.head_dim * config.num_heads;

        // Initialize weight matrices
        let query_weights = Array2::from_shape_fn((config.query_dim, total_key_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * query_scale
        });

        let key_weights = Array2::from_shape_fn((config.key_dim, total_key_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * key_scale
        });

        let value_weights = Array2::from_shape_fn((config.value_dim, total_value_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * value_scale
        });

        let output_weights = Array2::from_shape_fn((total_value_dim, config.query_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * output_scale
        });

        // Initialize biases if needed
        let query_bias = if config.use_bias {
            Some(Array1::zeros(total_key_dim))
        } else {
            None
        };

        let key_bias = if config.use_bias {
            Some(Array1::zeros(total_key_dim))
        } else {
            None
        };

        let value_bias = if config.use_bias {
            Some(Array1::zeros(total_value_dim))
        } else {
            None
        };

        let output_bias = if config.use_bias {
            Some(Array1::zeros(config.query_dim))
        } else {
            None
        };

        Self {
            config,
            query_weights,
            key_weights,
            value_weights,
            output_weights,
            query_bias,
            key_bias,
            value_bias,
            output_bias,
        }
    }

    /// Compute scaled dot-product attention
    ///
    /// # Arguments
    /// * `queries` - Query matrix (seq_len_q, d_k)
    /// * `keys` - Key matrix (seq_len_k, d_k)
    /// * `values` - Value matrix (seq_len_k, d_v)
    ///
    /// # Returns
    /// Tuple of (output, attention_weights)
    pub fn scaled_dot_product_attention(
        &self,
        queries: ArrayView2<Float>,
        keys: ArrayView2<Float>,
        values: ArrayView2<Float>,
    ) -> (Array2<Float>, Array2<Float>) {
        let d_k = keys.ncols() as Float;
        let scale = (d_k / self.config.temperature).sqrt();

        // Compute attention scores: Q @ K^T / sqrt(d_k)
        let scores = queries.dot(&keys.t()) / scale;

        // Apply activation (typically softmax)
        let attention_weights = match self.config.activation {
            AttentionActivation::Softmax => Self::softmax(&scores, Axis(1)),
            AttentionActivation::Sigmoid => scores.mapv(|x| 1.0 / (1.0 + (-x).exp())),
            AttentionActivation::Tanh => scores.mapv(|x| x.tanh()),
            AttentionActivation::ReLU => scores.mapv(|x| x.max(0.0)),
            AttentionActivation::Sparsemax => Self::sparsemax(&scores),
        };

        // Compute output: attention_weights @ V
        let output = attention_weights.dot(&values);

        (output, attention_weights)
    }

    /// Softmax activation along a specified axis
    fn softmax(x: &Array2<Float>, axis: Axis) -> Array2<Float> {
        let max_vals = x.map_axis(axis, |row| {
            row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b))
        });

        let exp_x = x - &max_vals.insert_axis(axis);
        let exp_x = exp_x.mapv(|v| v.exp());

        let sum_exp = exp_x.sum_axis(axis);
        exp_x / &sum_exp.insert_axis(axis)
    }

    /// Sparsemax activation (produces sparse attention weights)
    fn sparsemax(x: &Array2<Float>) -> Array2<Float> {
        let mut result = Array2::zeros(x.dim());

        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mut sorted: Vec<Float> = row.iter().copied().collect();
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

            let mut k = 0;
            let mut cumsum = 0.0;

            for (j, &val) in sorted.iter().enumerate() {
                cumsum += val;
                let threshold = (cumsum - 1.0) / ((j + 1) as Float);

                if val > threshold {
                    k = j + 1;
                } else {
                    break;
                }
            }

            let tau = (sorted[..k].iter().sum::<Float>() - 1.0) / (k as Float);

            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = (val - tau).max(0.0);
            }
        }

        result
    }

    /// Apply dropout (training only)
    fn apply_dropout(&self, x: &Array2<Float>, is_training: bool) -> Array2<Float> {
        if !is_training || self.config.dropout == 0.0 {
            return x.clone();
        }

        let mut rng = thread_rng();
        let keep_prob = 1.0 - self.config.dropout;

        x.mapv(|val| {
            if rng.random::<Float>() < keep_prob {
                val / keep_prob
            } else {
                0.0
            }
        })
    }
}

impl MultiHeadAttention {
    /// Create a new multi-head attention mechanism
    pub fn new(config: AttentionConfig) -> Self {
        assert!(
            config.key_dim % config.num_heads == 0,
            "Key dimension must be divisible by number of heads"
        );
        assert!(
            config.value_dim % config.num_heads == 0,
            "Value dimension must be divisible by number of heads"
        );

        let layer = AttentionLayer::new(config.clone());

        Self { config, layer }
    }

    /// Forward pass through multi-head attention
    ///
    /// # Arguments
    /// * `queries` - Query tensor (batch_size, seq_len_q, d_model)
    /// * `keys` - Key tensor (batch_size, seq_len_k, d_model)
    /// * `values` - Value tensor (batch_size, seq_len_k, d_model)
    /// * `is_training` - Whether in training mode (for dropout)
    pub fn forward(
        &self,
        queries: ArrayView2<Float>,
        keys: ArrayView2<Float>,
        values: ArrayView2<Float>,
        is_training: bool,
    ) -> AttentionOutput {
        let seq_len_q = queries.nrows();
        let seq_len_k = keys.nrows();

        // Project queries, keys, and values
        let q = queries.dot(&self.layer.query_weights);
        let k = keys.dot(&self.layer.key_weights);
        let v = values.dot(&self.layer.value_weights);

        // Add bias if present
        let q = if let Some(ref bias) = self.layer.query_bias {
            q + bias
        } else {
            q
        };
        let k = if let Some(ref bias) = self.layer.key_bias {
            k + bias
        } else {
            k
        };
        let v = if let Some(ref bias) = self.layer.value_bias {
            v + bias
        } else {
            v
        };

        // Split into multiple heads
        let head_dim = self.config.head_dim;
        let num_heads = self.config.num_heads;

        let mut head_outputs = Vec::new();
        let mut all_attention_weights = Vec::new();

        for h in 0..num_heads {
            let start = h * head_dim;
            let end = start + head_dim;

            let q_h = q.slice(s![.., start..end]).to_owned();
            let k_h = k.slice(s![.., start..end]).to_owned();
            let v_h = v.slice(s![.., start..end]).to_owned();

            let (head_out, attn_weights) =
                self.layer
                    .scaled_dot_product_attention(q_h.view(), k_h.view(), v_h.view());

            head_outputs.push(head_out);
            all_attention_weights.push(attn_weights);
        }

        // Concatenate heads
        let concat_output = Self::concatenate_heads(&head_outputs);

        // Apply output projection
        let mut output = concat_output.dot(&self.layer.output_weights);
        if let Some(ref bias) = self.layer.output_bias {
            output = output + bias;
        }

        // Apply dropout
        let output = self.layer.apply_dropout(&output, is_training);

        // Stack attention weights
        let attention_weights = Self::stack_attention_weights(&all_attention_weights);

        AttentionOutput {
            output,
            attention_weights,
            head_outputs,
        }
    }

    /// Concatenate outputs from multiple heads
    fn concatenate_heads(heads: &[Array2<Float>]) -> Array2<Float> {
        if heads.is_empty() {
            return Array2::zeros((0, 0));
        }

        let n_rows = heads[0].nrows();
        let total_cols: usize = heads.iter().map(|h| h.ncols()).sum();

        let mut result = Array2::zeros((n_rows, total_cols));
        let mut col_offset = 0;

        for head in heads {
            let n_cols = head.ncols();
            result
                .slice_mut(s![.., col_offset..col_offset + n_cols])
                .assign(head);
            col_offset += n_cols;
        }

        result
    }

    /// Stack attention weights from multiple heads into 3D array
    fn stack_attention_weights(weights: &[Array2<Float>]) -> Array3<Float> {
        if weights.is_empty() {
            return Array3::zeros((0, 0, 0));
        }

        let num_heads = weights.len();
        let seq_len_q = weights[0].nrows();
        let seq_len_k = weights[0].ncols();

        let mut result = Array3::zeros((num_heads, seq_len_q, seq_len_k));

        for (i, weight) in weights.iter().enumerate() {
            result.slice_mut(s![i, .., ..]).assign(weight);
        }

        result
    }
}

impl CrossModalAttention {
    /// Create a new cross-modal attention mechanism
    ///
    /// # Arguments
    /// * `x_dim` - Dimension of modality X
    /// * `y_dim` - Dimension of modality Y
    /// * `num_heads` - Number of attention heads
    pub fn new(x_dim: usize, y_dim: usize, num_heads: usize) -> Self {
        let x_to_y_config = AttentionConfig::new(
            AttentionType::CrossAttention,
            num_heads,
            x_dim,
            y_dim,
            y_dim,
        );

        let y_to_x_config = AttentionConfig::new(
            AttentionType::CrossAttention,
            num_heads,
            y_dim,
            x_dim,
            x_dim,
        );

        let x_to_y_attention = MultiHeadAttention::new(x_to_y_config);
        let y_to_x_attention = MultiHeadAttention::new(y_to_x_config);

        // Fusion weights for combining bidirectional attention
        let mut rng = thread_rng();
        let fusion_scale = (2.0 / (x_dim + y_dim) as Float).sqrt();
        let fusion_weights = Array2::from_shape_fn((x_dim + y_dim, x_dim + y_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * fusion_scale
        });

        Self {
            x_to_y_attention,
            y_to_x_attention,
            fusion_weights,
        }
    }

    /// Forward pass through cross-modal attention
    ///
    /// # Arguments
    /// * `x` - Input from modality X (seq_len, x_dim)
    /// * `y` - Input from modality Y (seq_len, y_dim)
    /// * `is_training` - Whether in training mode
    pub fn forward(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
        is_training: bool,
    ) -> CrossModalAttentionOutput {
        // X attends to Y (X is query, Y is key/value)
        let x_to_y = self.x_to_y_attention.forward(x, y, y, is_training);

        // Y attends to X (Y is query, X is key/value)
        let y_to_x = self.y_to_x_attention.forward(y, x, x, is_training);

        // Concatenate the attended representations
        let x_attended = &x_to_y.output;
        let y_attended = &y_to_x.output;

        let fused_input = Self::concatenate_modalities(x_attended, y_attended);

        // Apply fusion transformation
        let fused_output = fused_input.dot(&self.fusion_weights);

        // Compute fusion attention weights (simple averaging for now)
        let fusion_attention_weights = (&x_to_y.attention_weights.mean_axis(Axis(0)).unwrap()
            + &y_to_x.attention_weights.mean_axis(Axis(0)).unwrap())
            / 2.0;

        CrossModalAttentionOutput {
            fused_output,
            x_to_y_output: x_to_y,
            y_to_x_output: y_to_x,
            fusion_attention_weights,
        }
    }

    /// Concatenate two modalities along feature dimension
    fn concatenate_modalities(x: &Array2<Float>, y: &Array2<Float>) -> Array2<Float> {
        let n_rows = x.nrows();
        let x_cols = x.ncols();
        let y_cols = y.ncols();

        let mut result = Array2::zeros((n_rows, x_cols + y_cols));
        result.slice_mut(s![.., ..x_cols]).assign(x);
        result.slice_mut(s![.., x_cols..]).assign(y);
        result
    }
}

impl TransformerEncoderBlock {
    /// Create a new transformer encoder block
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `ffn_dim` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate
    pub fn new(d_model: usize, num_heads: usize, ffn_dim: usize, dropout: Float) -> Self {
        let config = AttentionConfig::new(
            AttentionType::MultiHead,
            num_heads,
            d_model,
            d_model,
            d_model,
        )
        .dropout(dropout);

        let self_attention = MultiHeadAttention::new(config);

        let mut rng = thread_rng();
        let ffn_scale_1 = (2.0 / (d_model + ffn_dim) as Float).sqrt();
        let ffn_scale_2 = (2.0 / (ffn_dim + d_model) as Float).sqrt();

        let ffn_weights_1 = Array2::from_shape_fn((d_model, ffn_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * ffn_scale_1
        });

        let ffn_weights_2 = Array2::from_shape_fn((ffn_dim, d_model), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * ffn_scale_2
        });

        let ffn_bias_1 = Some(Array1::zeros(ffn_dim));
        let ffn_bias_2 = Some(Array1::zeros(d_model));

        // Layer normalization parameters (gamma = 1, beta = 0 initially)
        let ln1_gamma = Array1::ones(d_model);
        let ln1_beta = Array1::zeros(d_model);
        let ln2_gamma = Array1::ones(d_model);
        let ln2_beta = Array1::zeros(d_model);

        Self {
            self_attention,
            ffn_weights_1,
            ffn_weights_2,
            ffn_bias_1,
            ffn_bias_2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            dropout,
            ffn_dim,
        }
    }

    /// Forward pass through transformer encoder block
    ///
    /// # Arguments
    /// * `x` - Input tensor (seq_len, d_model)
    /// * `is_training` - Whether in training mode
    pub fn forward(&self, x: ArrayView2<Float>, is_training: bool) -> Array2<Float> {
        // Self-attention with residual connection and layer norm
        let attn_output = self.self_attention.forward(x, x, x, is_training);
        let x = x.to_owned() + &attn_output.output; // Residual
        let x = Self::layer_norm(&x, &self.ln1_gamma, &self.ln1_beta);

        // Feed-forward network with residual connection and layer norm
        let ffn_output = self.feed_forward(&x);
        let x = x + &ffn_output; // Residual

        Self::layer_norm(&x, &self.ln2_gamma, &self.ln2_beta)
    }

    /// Feed-forward network (2-layer MLP with ReLU)
    fn feed_forward(&self, x: &Array2<Float>) -> Array2<Float> {
        // First layer
        let mut hidden = x.dot(&self.ffn_weights_1);
        if let Some(ref bias) = self.ffn_bias_1 {
            hidden = hidden + bias;
        }
        hidden = hidden.mapv(|v| v.max(0.0)); // ReLU

        // Second layer
        let mut output = hidden.dot(&self.ffn_weights_2);
        if let Some(ref bias) = self.ffn_bias_2 {
            output = output + bias;
        }

        output
    }

    /// Layer normalization
    fn layer_norm(x: &Array2<Float>, gamma: &Array1<Float>, beta: &Array1<Float>) -> Array2<Float> {
        let eps = 1e-5;
        let mean = x.mean_axis(Axis(1)).unwrap();
        let var = x.var_axis(Axis(1), 0.0);

        let normalized =
            (x - &mean.insert_axis(Axis(1))) / &var.mapv(|v| (v + eps).sqrt()).insert_axis(Axis(1));

        &normalized * &gamma.view().insert_axis(Axis(0)) + &beta.view().insert_axis(Axis(0))
    }
}

impl TransformerDecoderBlock {
    /// Create a new transformer decoder block
    ///
    /// # Arguments
    /// * `d_model` - Model dimension
    /// * `num_heads` - Number of attention heads
    /// * `ffn_dim` - Feed-forward network hidden dimension
    /// * `dropout` - Dropout rate
    pub fn new(d_model: usize, num_heads: usize, ffn_dim: usize, dropout: Float) -> Self {
        let config = AttentionConfig::new(
            AttentionType::MultiHead,
            num_heads,
            d_model,
            d_model,
            d_model,
        )
        .dropout(dropout);

        let self_attention = MultiHeadAttention::new(config.clone());
        let cross_attention = MultiHeadAttention::new(config);

        let mut rng = thread_rng();
        let ffn_scale_1 = (2.0 / (d_model + ffn_dim) as Float).sqrt();
        let ffn_scale_2 = (2.0 / (ffn_dim + d_model) as Float).sqrt();

        let ffn_weights_1 = Array2::from_shape_fn((d_model, ffn_dim), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * ffn_scale_1
        });

        let ffn_weights_2 = Array2::from_shape_fn((ffn_dim, d_model), |_| {
            (rng.random::<Float>() - 0.5) * 2.0 * ffn_scale_2
        });

        let ffn_bias_1 = Some(Array1::zeros(ffn_dim));
        let ffn_bias_2 = Some(Array1::zeros(d_model));

        // Layer normalization parameters
        let ln1_gamma = Array1::ones(d_model);
        let ln1_beta = Array1::zeros(d_model);
        let ln2_gamma = Array1::ones(d_model);
        let ln2_beta = Array1::zeros(d_model);
        let ln3_gamma = Array1::ones(d_model);
        let ln3_beta = Array1::zeros(d_model);

        Self {
            self_attention,
            cross_attention,
            ffn_weights_1,
            ffn_weights_2,
            ffn_bias_1,
            ffn_bias_2,
            ln1_gamma,
            ln1_beta,
            ln2_gamma,
            ln2_beta,
            ln3_gamma,
            ln3_beta,
            dropout,
            ffn_dim,
        }
    }

    /// Forward pass through transformer decoder block
    ///
    /// # Arguments
    /// * `x` - Decoder input (seq_len, d_model)
    /// * `encoder_output` - Encoder output (seq_len, d_model)
    /// * `is_training` - Whether in training mode
    pub fn forward(
        &self,
        x: ArrayView2<Float>,
        encoder_output: ArrayView2<Float>,
        is_training: bool,
    ) -> Array2<Float> {
        // Self-attention on decoder input
        let self_attn_output = self.self_attention.forward(x, x, x, is_training);
        let x = x.to_owned() + &self_attn_output.output; // Residual
        let x = TransformerEncoderBlock::layer_norm(&x, &self.ln1_gamma, &self.ln1_beta);

        // Cross-attention between decoder and encoder
        let cross_attn_output =
            self.cross_attention
                .forward(x.view(), encoder_output, encoder_output, is_training);
        let x = x + &cross_attn_output.output; // Residual
        let x = TransformerEncoderBlock::layer_norm(&x, &self.ln2_gamma, &self.ln2_beta);

        // Feed-forward network
        let ffn_output = self.feed_forward(&x);
        let x = x + &ffn_output; // Residual

        TransformerEncoderBlock::layer_norm(&x, &self.ln3_gamma, &self.ln3_beta)
    }

    /// Feed-forward network (2-layer MLP with ReLU)
    fn feed_forward(&self, x: &Array2<Float>) -> Array2<Float> {
        // First layer
        let mut hidden = x.dot(&self.ffn_weights_1);
        if let Some(ref bias) = self.ffn_bias_1 {
            hidden = hidden + bias;
        }
        hidden = hidden.mapv(|v| v.max(0.0)); // ReLU

        // Second layer
        let mut output = hidden.dot(&self.ffn_weights_2);
        if let Some(ref bias) = self.ffn_bias_2 {
            output = output + bias;
        }

        output
    }
}

// Import s! macro for slicing
use scirs2_core::ndarray::s;

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_attention_config_creation() {
        let config = AttentionConfig::new(AttentionType::MultiHead, 8, 512, 512, 512);

        assert_eq!(config.attention_type, AttentionType::MultiHead);
        assert_eq!(config.num_heads, 8);
        assert_eq!(config.query_dim, 512);
        assert_eq!(config.key_dim, 512);
        assert_eq!(config.value_dim, 512);
    }

    #[test]
    fn test_multi_head_attention_creation() {
        let config = AttentionConfig::new(AttentionType::MultiHead, 4, 64, 64, 64);

        let mha = MultiHeadAttention::new(config);
        assert_eq!(mha.config.num_heads, 4);
        assert_eq!(mha.config.head_dim, 16); // 64 / 4
    }

    #[test]
    fn test_scaled_dot_product_attention() {
        let config = AttentionConfig::new(AttentionType::SelfAttention, 1, 4, 4, 4);

        let layer = AttentionLayer::new(config);

        let queries = array![[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
        let keys = array![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0]
        ];
        let values = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];

        let (output, weights) =
            layer.scaled_dot_product_attention(queries.view(), keys.view(), values.view());

        assert_eq!(output.shape(), &[2, 4]);
        assert_eq!(weights.shape(), &[2, 3]);

        // Attention weights should sum to 1 along each row
        for i in 0..weights.nrows() {
            let row_sum: Float = weights.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_softmax() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let result = AttentionLayer::softmax(&x, Axis(1));

        // Check that rows sum to 1
        for i in 0..result.nrows() {
            let row_sum: Float = result.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cross_modal_attention() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0]
        ];
        let y = array![
            [0.5, 1.5, 2.5, 3.5],
            [1.5, 2.5, 3.5, 4.5],
            [2.5, 3.5, 4.5, 5.5]
        ];

        let cross_attn = CrossModalAttention::new(4, 4, 2);
        let output = cross_attn.forward(x.view(), y.view(), false);

        assert!(output.fused_output.nrows() > 0);
        assert!(output.fused_output.ncols() > 0);
    }

    #[test]
    fn test_transformer_encoder_block() {
        let d_model = 64;
        let num_heads = 4;
        let ffn_dim = 128;
        let dropout = 0.1;

        let encoder = TransformerEncoderBlock::new(d_model, num_heads, ffn_dim, dropout);

        let x = Array2::from_shape_fn((10, d_model), |(i, j)| (i + j) as Float / 100.0);
        let output = encoder.forward(x.view(), false);

        assert_eq!(output.shape(), &[10, d_model]);
    }

    #[test]
    fn test_transformer_decoder_block() {
        let d_model = 64;
        let num_heads = 4;
        let ffn_dim = 128;
        let dropout = 0.1;

        let decoder = TransformerDecoderBlock::new(d_model, num_heads, ffn_dim, dropout);

        let x = Array2::from_shape_fn((10, d_model), |(i, j)| (i + j) as Float / 100.0);
        let enc_output = Array2::from_shape_fn((10, d_model), |(i, j)| (i * j) as Float / 100.0);

        let output = decoder.forward(x.view(), enc_output.view(), false);

        assert_eq!(output.shape(), &[10, d_model]);
    }

    #[test]
    fn test_attention_layer_dimensions() {
        let config = AttentionConfig::new(AttentionType::MultiHead, 8, 512, 512, 512);

        let layer = AttentionLayer::new(config.clone());

        // head_dim = 512 / 8 = 64, total_dim = 64 * 8 = 512
        assert_eq!(layer.query_weights.shape(), &[512, 512]);
        assert_eq!(layer.key_weights.shape(), &[512, 512]);
        assert_eq!(layer.value_weights.shape(), &[512, 512]);
        assert_eq!(layer.output_weights.shape(), &[512, 512]);
    }

    #[test]
    fn test_multihead_attention_forward() {
        let config = AttentionConfig::new(AttentionType::MultiHead, 4, 32, 32, 32);

        let mha = MultiHeadAttention::new(config);

        let queries = Array2::from_shape_fn((10, 32), |(i, j)| (i + j) as Float / 100.0);
        let keys = Array2::from_shape_fn((15, 32), |(i, j)| (i * j) as Float / 100.0);
        let values = Array2::from_shape_fn((15, 32), |(i, j)| ((i + 1) * (j + 1)) as Float / 100.0);

        let output = mha.forward(queries.view(), keys.view(), values.view(), false);

        assert_eq!(output.output.shape(), &[10, 32]);
        assert_eq!(output.attention_weights.shape(), &[4, 10, 15]); // num_heads x seq_len_q x seq_len_k
        assert_eq!(output.head_outputs.len(), 4);
    }

    #[test]
    fn test_layer_norm() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let gamma = array![1.0, 1.0, 1.0];
        let beta = array![0.0, 0.0, 0.0];

        let normalized = TransformerEncoderBlock::layer_norm(&x, &gamma, &beta);

        // Check that each row has mean close to 0
        for i in 0..normalized.nrows() {
            let row_mean = normalized.row(i).mean().unwrap();
            assert!(row_mean.abs() < 1e-5);
        }
    }

    #[test]
    fn test_sparsemax_activation() {
        let x = array![[3.0, 1.0, -2.0, 0.5], [1.0, 1.0, 1.0, 1.0]];
        let result = AttentionLayer::sparsemax(&x);

        // Check that rows sum to 1
        for i in 0..result.nrows() {
            let row_sum: Float = result.row(i).iter().sum();
            assert!((row_sum - 1.0).abs() < 1e-6);
        }

        // Sparsemax should produce sparse outputs (many zeros)
        let zero_count = result.iter().filter(|&&x| x.abs() < 1e-10).count();
        assert!(zero_count > 0, "Sparsemax should produce some zero values");
    }

    #[test]
    fn test_attention_config_builder() {
        let config = AttentionConfig::new(AttentionType::MultiHead, 8, 256, 256, 256)
            .dropout(0.2)
            .temperature(0.5)
            .activation(AttentionActivation::Sparsemax);

        assert_eq!(config.dropout, 0.2);
        assert_eq!(config.temperature, 0.5);
        assert_eq!(config.activation, AttentionActivation::Sparsemax);
    }
}

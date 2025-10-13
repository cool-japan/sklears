use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;
use super::attention_mechanisms::*;
use scirs2_core::ndarray::s;

pub struct TransformerEncoderLayer {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout_rate: f64,
    pub self_attention: MultiHeadAttention,
    pub feed_forward: FeedForwardNetwork,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
    pub dropout: DropoutLayer,
}

impl TransformerEncoderLayer {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
        activation: CNNActivation,
        random_state: Option<u64>,
    ) -> Self {
        let self_attention = MultiHeadAttention::new(d_model, n_heads, dropout_rate, random_state);
        let feed_forward = FeedForwardNetwork::new(d_model, d_ff, activation, dropout_rate, random_state);
        let layer_norm1 = LayerNorm::new(d_model);
        let layer_norm2 = LayerNorm::new(d_model);
        let dropout = DropoutLayer::new(dropout_rate);

        Self {
            d_model,
            n_heads,
            d_ff,
            dropout_rate,
            self_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let (attention_output, _) = self.self_attention.forward(x, x, x, mask);
        let attention_dropout = self.dropout.forward_2d(&attention_output);
        let residual1 = x + &attention_dropout;
        let norm1 = self.layer_norm1.forward(&residual1);

        let ff_output = self.feed_forward.forward(&norm1);
        let ff_dropout = self.dropout.forward_2d(&ff_output);
        let residual2 = &norm1 + &ff_dropout;
        self.layer_norm2.forward(&residual2)
    }

    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
        self.feed_forward.set_training(training);
        self.layer_norm1.set_training(training);
        self.layer_norm2.set_training(training);
    }
}

pub struct TransformerDecoderLayer {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout_rate: f64,
    pub self_attention: MultiHeadAttention,
    pub cross_attention: MultiHeadAttention,
    pub feed_forward: FeedForwardNetwork,
    pub layer_norm1: LayerNorm,
    pub layer_norm2: LayerNorm,
    pub layer_norm3: LayerNorm,
    pub dropout: DropoutLayer,
}

impl TransformerDecoderLayer {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
        activation: CNNActivation,
        random_state: Option<u64>,
    ) -> Self {
        let self_attention = MultiHeadAttention::new(d_model, n_heads, dropout_rate, random_state);
        let cross_attention = MultiHeadAttention::new(d_model, n_heads, dropout_rate, random_state);
        let feed_forward = FeedForwardNetwork::new(d_model, d_ff, activation, dropout_rate, random_state);
        let layer_norm1 = LayerNorm::new(d_model);
        let layer_norm2 = LayerNorm::new(d_model);
        let layer_norm3 = LayerNorm::new(d_model);
        let dropout = DropoutLayer::new(dropout_rate);

        Self {
            d_model,
            n_heads,
            d_ff,
            dropout_rate,
            self_attention,
            cross_attention,
            feed_forward,
            layer_norm1,
            layer_norm2,
            layer_norm3,
            dropout,
        }
    }

    pub fn forward(
        &mut self,
        x: &Array2<f64>,
        encoder_output: &Array2<f64>,
        self_mask: Option<&Array2<bool>>,
        cross_mask: Option<&Array2<bool>>,
    ) -> Array2<f64> {
        let (self_attention_output, _) = self.self_attention.forward(x, x, x, self_mask);
        let self_attention_dropout = self.dropout.forward_2d(&self_attention_output);
        let residual1 = x + &self_attention_dropout;
        let norm1 = self.layer_norm1.forward(&residual1);

        let (cross_attention_output, _) = self.cross_attention.forward(&norm1, encoder_output, encoder_output, cross_mask);
        let cross_attention_dropout = self.dropout.forward_2d(&cross_attention_output);
        let residual2 = &norm1 + &cross_attention_dropout;
        let norm2 = self.layer_norm2.forward(&residual2);

        let ff_output = self.feed_forward.forward(&norm2);
        let ff_dropout = self.dropout.forward_2d(&ff_output);
        let residual3 = &norm2 + &ff_dropout;
        self.layer_norm3.forward(&residual3)
    }

    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
        self.feed_forward.set_training(training);
        self.layer_norm1.set_training(training);
        self.layer_norm2.set_training(training);
        self.layer_norm3.set_training(training);
    }
}

pub struct FeedForwardNetwork {
    pub d_model: usize,
    pub d_ff: usize,
    pub activation: CNNActivation,
    pub w1: Array2<f64>,
    pub b1: Array1<f64>,
    pub w2: Array2<f64>,
    pub b2: Array1<f64>,
    pub dropout: DropoutLayer,
}

impl FeedForwardNetwork {
    pub fn new(
        d_model: usize,
        d_ff: usize,
        activation: CNNActivation,
        dropout_rate: f64,
        random_state: Option<u64>,
    ) -> Self {
        let w1 = xavier_normal_init(d_model, d_ff, random_state);
        let b1 = Array1::zeros(d_ff);
        let w2 = xavier_normal_init(d_ff, d_model, random_state);
        let b2 = Array1::zeros(d_model);
        let dropout = DropoutLayer::new(dropout_rate);

        Self {
            d_model,
            d_ff,
            activation,
            w1,
            b1,
            w2,
            b2,
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let linear1 = x.dot(&self.w1) + &self.b1;
        let activated = self.apply_activation(&linear1);
        let dropout1 = self.dropout.forward_2d(&activated);
        dropout1.dot(&self.w2) + &self.b2
    }

    fn apply_activation(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                result[(i, j)] = self.activation.apply(x[(i, j)]);
            }
        }

        result
    }

    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }
}

pub struct LayerNorm {
    pub num_features: usize,
    pub eps: f64,
    pub weight: Array1<f64>,
    pub bias: Array1<f64>,
    pub training: bool,
}

impl LayerNorm {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            weight: Array1::ones(num_features),
            bias: Array1::zeros(num_features),
            training: true,
        }
    }

    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, features) = x.dim();

        if features != self.num_features {
            panic!("Input features {} do not match layer features {}", features, self.num_features);
        }

        let mut normalized = Array2::zeros((seq_len, features));

        for i in 0..seq_len {
            let row = x.row(i);
            let mean = row.mean().unwrap_or(0.0);
            let variance = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / features as f64;
            let std_dev = (variance + self.eps).sqrt();

            for j in 0..features {
                let normalized_val = (x[(i, j)] - mean) / std_dev;
                normalized[(i, j)] = self.weight[j] * normalized_val + self.bias[j];
            }
        }

        normalized
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

pub struct TransformerBlock {
    pub encoder_layer: TransformerEncoderLayer,
    pub positional_encoding: PositionalEncoding,
}

impl TransformerBlock {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
        max_seq_len: usize,
        activation: CNNActivation,
        random_state: Option<u64>,
    ) -> Self {
        let encoder_layer = TransformerEncoderLayer::new(d_model, n_heads, d_ff, dropout_rate, activation, random_state);
        let positional_encoding = PositionalEncoding::new(d_model, max_seq_len);

        Self {
            encoder_layer,
            positional_encoding,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let pos_encoded = self.positional_encoding.forward(x);
        self.encoder_layer.forward(&pos_encoded, mask)
    }

    pub fn set_training(&mut self, training: bool) {
        self.encoder_layer.set_training(training);
    }
}

pub struct PositionalEncoding {
    pub d_model: usize,
    pub max_seq_len: usize,
    pub encoding: Array2<f64>,
    pub dropout: DropoutLayer,
}

impl PositionalEncoding {
    pub fn new(d_model: usize, max_seq_len: usize) -> Self {
        let encoding = Self::create_positional_encoding(d_model, max_seq_len);
        let dropout = DropoutLayer::new(0.1);

        Self {
            d_model,
            max_seq_len,
            encoding,
            dropout,
        }
    }

    pub fn with_dropout(mut self, dropout_rate: f64) -> Self {
        self.dropout = DropoutLayer::new(dropout_rate);
        self
    }

    fn create_positional_encoding(d_model: usize, max_seq_len: usize) -> Array2<f64> {
        let mut encoding = Array2::zeros((max_seq_len, d_model));

        for pos in 0..max_seq_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / d_model as f64);
                if i % 2 == 0 {
                    encoding[(pos, i)] = angle.sin();
                } else {
                    encoding[(pos, i)] = angle.cos();
                }
            }
        }

        encoding
    }

    pub fn forward(&mut self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, d_model) = x.dim();

        if d_model != self.d_model {
            panic!("Input d_model {} does not match encoding d_model {}", d_model, self.d_model);
        }

        if seq_len > self.max_seq_len {
            panic!("Sequence length {} exceeds maximum length {}", seq_len, self.max_seq_len);
        }

        let pos_encoding = self.encoding.slice(s![..seq_len, ..]).to_owned();
        let embedded = x + &pos_encoding;
        self.dropout.forward_2d(&embedded)
    }

    pub fn set_training(&mut self, training: bool) {
        self.dropout.set_training(training);
    }
}

pub struct MultiLayerTransformer {
    pub layers: Vec<TransformerEncoderLayer>,
    pub final_layer_norm: LayerNorm,
    pub dropout: DropoutLayer,
}

impl MultiLayerTransformer {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout_rate: f64,
        activation: CNNActivation,
        random_state: Option<u64>,
    ) -> Self {
        let mut layers = Vec::new();

        for _ in 0..n_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                n_heads,
                d_ff,
                dropout_rate,
                activation,
                random_state,
            ));
        }

        let final_layer_norm = LayerNorm::new(d_model);
        let dropout = DropoutLayer::new(dropout_rate);

        Self {
            layers,
            final_layer_norm,
            dropout,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Vec<Array2<f64>> {
        let mut outputs = Vec::new();
        let mut current = x.clone();

        for layer in &mut self.layers {
            current = layer.forward(&current, mask);
            outputs.push(current.clone());
        }

        let final_output = self.final_layer_norm.forward(&current);
        let dropout_output = self.dropout.forward_2d(&final_output);
        outputs.push(dropout_output);

        outputs
    }

    pub fn forward_single(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let mut current = x.clone();

        for layer in &mut self.layers {
            current = layer.forward(&current, mask);
        }

        let final_output = self.final_layer_norm.forward(&current);
        self.dropout.forward_2d(&final_output)
    }

    pub fn set_training(&mut self, training: bool) {
        for layer in &mut self.layers {
            layer.set_training(training);
        }
        self.final_layer_norm.set_training(training);
        self.dropout.set_training(training);
    }

    pub fn get_layer_outputs(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Vec<Array2<f64>> {
        self.forward(x, mask)
    }

    pub fn get_attention_weights(&mut self, x: &Array2<f64>, layer_idx: usize, mask: Option<&Array2<bool>>) -> Vec<Array2<f64>> {
        if layer_idx >= self.layers.len() {
            return Vec::new();
        }

        let mut current = x.clone();

        for i in 0..layer_idx {
            current = self.layers[i].forward(&current, mask);
        }

        let (_, attention_weights) = self.layers[layer_idx].self_attention.forward(&current, &current, &current, mask);
        attention_weights
    }
}

pub struct TransformerEncoder {
    pub transformer: MultiLayerTransformer,
    pub positional_encoding: PositionalEncoding,
}

impl TransformerEncoder {
    pub fn new(
        n_layers: usize,
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_rate: f64,
        activation: CNNActivation,
        random_state: Option<u64>,
    ) -> Self {
        let transformer = MultiLayerTransformer::new(n_layers, d_model, n_heads, d_ff, dropout_rate, activation, random_state);
        let positional_encoding = PositionalEncoding::new(d_model, max_seq_len).with_dropout(dropout_rate);

        Self {
            transformer,
            positional_encoding,
        }
    }

    pub fn forward(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let pos_encoded = self.positional_encoding.forward(x);
        self.transformer.forward_single(&pos_encoded, mask)
    }

    pub fn forward_with_layers(&mut self, x: &Array2<f64>, mask: Option<&Array2<bool>>) -> Vec<Array2<f64>> {
        let pos_encoded = self.positional_encoding.forward(x);
        self.transformer.forward(&pos_encoded, mask)
    }

    pub fn set_training(&mut self, training: bool) {
        self.transformer.set_training(training);
        self.positional_encoding.set_training(training);
    }
}

pub struct GLU {
    pub input_dim: usize,
    pub output_dim: usize,
    pub w: Array2<f64>,
    pub v: Array2<f64>,
    pub b: Array1<f64>,
    pub c: Array1<f64>,
}

impl GLU {
    pub fn new(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Self {
        let w = xavier_normal_init(input_dim, output_dim, random_state);
        let v = xavier_normal_init(input_dim, output_dim, random_state);
        let b = Array1::zeros(output_dim);
        let c = Array1::zeros(output_dim);

        Self {
            input_dim,
            output_dim,
            w,
            v,
            b,
            c,
        }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let linear1 = x.dot(&self.w) + &self.b;
        let linear2 = x.dot(&self.v) + &self.c;
        let gated = self.apply_sigmoid(&linear2);

        &linear1 * &gated
    }

    fn apply_sigmoid(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                result[(i, j)] = 1.0 / (1.0 + (-x[(i, j)]).exp());
            }
        }

        result
    }
}

pub struct GELU {
    pub approximate: bool,
}

impl GELU {
    pub fn new(approximate: bool) -> Self {
        Self { approximate }
    }

    pub fn forward(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                let val = x[(i, j)];
                result[(i, j)] = if self.approximate {
                    0.5 * val * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * (val + 0.044715 * val.powi(3))).tanh())
                } else {
                    0.5 * val * (1.0 + (val / std::f64::consts::SQRT_2).erf())
                };
            }
        }

        result
    }
}

fn erf(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

trait ErfExt {
    fn erf(self) -> Self;
}

impl ErfExt for f64 {
    fn erf(self) -> f64 {
        erf(self)
    }
}
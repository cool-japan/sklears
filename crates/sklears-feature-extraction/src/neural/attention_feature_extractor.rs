use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;
use scirs2_core::ndarray::s;

pub struct AttentionFeatureExtractor {
    pub attention_type: AttentionType,
    pub d_model: usize,
    pub n_heads: usize,
    pub seq_length: usize,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub regularization: f64,
    pub random_state: Option<u64>,
    pub dropout_rate: f64,
    pub use_layer_norm: bool,
    pub temperature: f64,
}

impl AttentionFeatureExtractor {
    pub fn new() -> Self {
        Self {
            attention_type: AttentionType::Scaled,
            d_model: 512,
            n_heads: 8,
            seq_length: 128,
            learning_rate: 0.0001,
            n_epochs: 50,
            batch_size: 32,
            regularization: 0.01,
            random_state: None,
            dropout_rate: 0.1,
            use_layer_norm: true,
            temperature: 1.0,
        }
    }

    pub fn attention_type(mut self, attention_type: AttentionType) -> Self {
        self.attention_type = attention_type;
        self
    }

    pub fn d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    pub fn seq_length(mut self, length: usize) -> Self {
        self.seq_length = length;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn n_epochs(mut self, epochs: usize) -> Self {
        self.n_epochs = epochs;
        self
    }

    pub fn batch_size(mut self, size: usize) -> Self {
        self.batch_size = size;
        self
    }

    pub fn regularization(mut self, reg: f64) -> Self {
        self.regularization = reg;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    pub fn dropout_rate(mut self, rate: f64) -> Self {
        self.dropout_rate = rate;
        self
    }

    pub fn use_layer_norm(mut self, use_norm: bool) -> Self {
        self.use_layer_norm = use_norm;
        self
    }

    pub fn temperature(mut self, temp: f64) -> Self {
        self.temperature = temp;
        self
    }

    fn initialize_weights(&self, input_dim: usize, output_dim: usize) -> Array2<f64> {
        let mut rng = match self.random_state {
            Some(seed) => Random::seed_from_u64(seed),
            None => Random::seed_from_u64(42),
        };

        let scale = (2.0 / (input_dim + output_dim) as f64).sqrt();
        let mut weights = Array2::zeros((input_dim, output_dim));

        for i in 0..input_dim {
            for j in 0..output_dim {
                weights[(i, j)] = rng.gen_range(-scale..scale);
            }
        }

        weights
    }

    fn scaled_dot_product_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> Array2<f64> {
        let d_k = k.ncols() as f64;
        let scores = q.dot(&k.t()) / (d_k.sqrt() * self.temperature);

        let attention_weights = self.apply_softmax_2d(&scores);
        attention_weights.dot(v)
    }

    fn additive_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        w_a: &Array2<f64>,
        w_b: &Array2<f64>,
        w_v: &Array1<f64>,
    ) -> Array2<f64> {
        let q_proj = q.dot(w_a);
        let k_proj = k.dot(w_b);

        let mut energy = Array2::zeros((q.nrows(), k.nrows()));

        for i in 0..q.nrows() {
            for j in 0..k.nrows() {
                let mut sum = 0.0;
                for l in 0..w_v.len() {
                    sum += w_v[l] * (q_proj[(i, l)] + k_proj[(j, l)]).tanh();
                }
                energy[(i, j)] = sum;
            }
        }

        let attention_weights = self.apply_softmax_2d(&energy);
        attention_weights.dot(v)
    }

    fn multiplicative_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        w: &Array2<f64>,
    ) -> Array2<f64> {
        let q_transformed = q.dot(w);
        let scores = q_transformed.dot(&k.t());

        let attention_weights = self.apply_softmax_2d(&scores);
        attention_weights.dot(v)
    }

    fn multi_head_attention(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        w_q: &Array2<f64>,
        w_k: &Array2<f64>,
        w_v: &Array2<f64>,
        w_o: &Array2<f64>,
    ) -> Array2<f64> {
        let d_k = self.d_model / self.n_heads;
        let mut head_outputs = Vec::new();

        for head in 0..self.n_heads {
            let start_idx = head * d_k;
            let end_idx = start_idx + d_k;

            let q_head = q.dot(&w_q.slice(s![.., start_idx..end_idx]));
            let k_head = k.dot(&w_k.slice(s![.., start_idx..end_idx]));
            let v_head = v.dot(&w_v.slice(s![.., start_idx..end_idx]));

            let head_output = self.scaled_dot_product_attention(&q_head, &k_head, &v_head);
            head_outputs.push(head_output);
        }

        let concatenated = self.concatenate_heads(&head_outputs);
        concatenated.dot(w_o)
    }

    fn concatenate_heads(&self, heads: &[Array2<f64>]) -> Array2<f64> {
        let seq_len = heads[0].nrows();
        let total_dim = heads.iter().map(|h| h.ncols()).sum();
        let mut result = Array2::zeros((seq_len, total_dim));

        let mut col_offset = 0;
        for head in heads {
            let head_dim = head.ncols();
            for i in 0..seq_len {
                for j in 0..head_dim {
                    result[(i, col_offset + j)] = head[(i, j)];
                }
            }
            col_offset += head_dim;
        }

        result
    }

    fn apply_softmax_2d(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(x.raw_dim());

        for i in 0..x.nrows() {
            let row = x.row(i);
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_row: Vec<f64> = row.iter().map(|&val| (val - max_val).exp()).collect();
            let sum_exp: f64 = exp_row.iter().sum();

            for (j, &exp_val) in exp_row.iter().enumerate() {
                result[(i, j)] = exp_val / sum_exp;
            }
        }

        result
    }

    fn apply_dropout(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = Random::seed_from_u64(42);
        let mut result = x.clone();

        for elem in result.iter_mut() {
            if rng.random_range(0.0, 1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
        }

        result
    }

    fn layer_normalize(&self, x: &Array2<f64>) -> Array2<f64> {
        if !self.use_layer_norm {
            return x.clone();
        }

        layer_norm(x)
    }

    fn positional_encoding(&self, seq_len: usize, d_model: usize) -> Array2<f64> {
        let mut pe = Array2::zeros((seq_len, d_model));

        for pos in 0..seq_len {
            for i in 0..d_model {
                let angle = pos as f64 / (10000.0_f64).powf((2 * (i / 2)) as f64 / d_model as f64);
                if i % 2 == 0 {
                    pe[(pos, i)] = angle.sin();
                } else {
                    pe[(pos, i)] = angle.cos();
                }
            }
        }

        pe
    }
}

pub struct FittedAttentionFeatureExtractor {
    extractor: AttentionFeatureExtractor,
    w_q: Array2<f64>,
    w_k: Array2<f64>,
    w_v: Array2<f64>,
    w_o: Array2<f64>,
    w_a: Option<Array2<f64>>,
    w_b: Option<Array2<f64>>,
    w_v_add: Option<Array1<f64>>,
    w_mult: Option<Array2<f64>>,
    positional_encoding: Array2<f64>,
    input_shape: (usize, usize),
}

impl Estimator<Untrained> for AttentionFeatureExtractor {
    fn estimator_type(&self) -> &'static str {
        "AttentionFeatureExtractor"
    }

    fn complexity(&self) -> f64 {
        (self.d_model * self.seq_length) as f64
    }
}

impl Fit<Array2<f64>, ()> for AttentionFeatureExtractor {
    type Output = FittedAttentionFeatureExtractor;

    fn fit(&self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Output> {
        let (seq_len, d_input) = x.dim();

        if seq_len == 0 || d_input == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if self.d_model % self.n_heads != 0 {
            return Err(SklearsError::InvalidInput(
                "d_model must be divisible by n_heads".to_string()
            ));
        }

        let w_q = self.initialize_weights(d_input, self.d_model);
        let w_k = self.initialize_weights(d_input, self.d_model);
        let w_v = self.initialize_weights(d_input, self.d_model);
        let w_o = self.initialize_weights(self.d_model, self.d_model);

        let (w_a, w_b, w_v_add) = match self.attention_type {
            AttentionType::Additive => {
                let w_a = Some(self.initialize_weights(self.d_model, self.d_model));
                let w_b = Some(self.initialize_weights(self.d_model, self.d_model));
                let w_v_add = Some(Array1::zeros(self.d_model));
                (w_a, w_b, w_v_add)
            },
            _ => (None, None, None),
        };

        let w_mult = match self.attention_type {
            AttentionType::Multiplicative => {
                Some(self.initialize_weights(self.d_model, self.d_model))
            },
            _ => None,
        };

        let positional_encoding = self.positional_encoding(seq_len, self.d_model);

        Ok(FittedAttentionFeatureExtractor {
            extractor: self.clone(),
            w_q,
            w_k,
            w_v,
            w_o,
            w_a,
            w_b,
            w_v_add,
            w_mult,
            positional_encoding,
            input_shape: (seq_len, d_input),
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedAttentionFeatureExtractor {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (seq_len, d_input) = x.dim();

        if (seq_len, d_input) != self.input_shape {
            return Err(SklearsError::InvalidInput(format!(
                "Input shape {:?} does not match fitted shape {:?}",
                (seq_len, d_input), self.input_shape
            )));
        }

        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);
        let v = x.dot(&self.w_v);

        let q_with_pos = &q + &self.positional_encoding.slice(s![..seq_len, ..]);
        let k_with_pos = &k + &self.positional_encoding.slice(s![..seq_len, ..]);

        let attention_output = match self.extractor.attention_type {
            AttentionType::Scaled => {
                self.extractor.scaled_dot_product_attention(&q_with_pos, &k_with_pos, &v)
            },
            AttentionType::Additive => {
                let w_a = self.w_a.as_ref().unwrap();
                let w_b = self.w_b.as_ref().unwrap();
                let w_v_add = self.w_v_add.as_ref().unwrap();
                self.extractor.additive_attention(&q_with_pos, &k_with_pos, &v, w_a, w_b, w_v_add)
            },
            AttentionType::Multiplicative => {
                let w_mult = self.w_mult.as_ref().unwrap();
                self.extractor.multiplicative_attention(&q_with_pos, &k_with_pos, &v, w_mult)
            },
            AttentionType::MultiHead => {
                self.extractor.multi_head_attention(&q_with_pos, &k_with_pos, &v, &self.w_q, &self.w_k, &self.w_v, &self.w_o)
            },
        };

        let dropout_output = self.extractor.apply_dropout(&attention_output);
        let normalized_output = self.extractor.layer_normalize(&dropout_output);

        Ok(normalized_output)
    }
}

impl FittedAttentionFeatureExtractor {
    pub fn get_attention_weights(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (seq_len, d_input) = x.dim();

        if (seq_len, d_input) != self.input_shape {
            return Err(SklearsError::InvalidInput("Input shape mismatch".to_string()));
        }

        let q = x.dot(&self.w_q);
        let k = x.dot(&self.w_k);

        let q_with_pos = &q + &self.positional_encoding.slice(s![..seq_len, ..]);
        let k_with_pos = &k + &self.positional_encoding.slice(s![..seq_len, ..]);

        let d_k = k_with_pos.ncols() as f64;
        let scores = q_with_pos.dot(&k_with_pos.t()) / (d_k.sqrt() * self.extractor.temperature);

        Ok(self.extractor.apply_softmax_2d(&scores))
    }

    pub fn get_positional_encoding(&self) -> &Array2<f64> {
        &self.positional_encoding
    }

    pub fn get_output_shape(&self) -> (usize, usize) {
        (self.input_shape.0, self.extractor.d_model)
    }
}
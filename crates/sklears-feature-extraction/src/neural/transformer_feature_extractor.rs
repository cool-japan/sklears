use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;

pub struct TransformerFeatureExtractor {
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub seq_length: usize,
    pub learning_rate: f64,
    pub n_epochs: usize,
    pub batch_size: usize,
    pub regularization: f64,
    pub random_state: Option<u64>,
    pub dropout_rate: f64,
    pub use_layer_norm: bool,
    pub activation: CNNActivation,
    pub max_position_embeddings: usize,
}

impl TransformerFeatureExtractor {
    pub fn new() -> Self {
        Self {
            d_model: 512,
            n_heads: 8,
            n_layers: 6,
            d_ff: 2048,
            seq_length: 128,
            learning_rate: 0.0001,
            n_epochs: 50,
            batch_size: 32,
            regularization: 0.01,
            random_state: None,
            dropout_rate: 0.1,
            use_layer_norm: true,
            activation: CNNActivation::ReLU,
            max_position_embeddings: 1024,
        }
    }

    pub fn d_model(mut self, d_model: usize) -> Self {
        self.d_model = d_model;
        self
    }

    pub fn n_heads(mut self, n_heads: usize) -> Self {
        self.n_heads = n_heads;
        self
    }

    pub fn n_layers(mut self, n_layers: usize) -> Self {
        self.n_layers = n_layers;
        self
    }

    pub fn d_ff(mut self, d_ff: usize) -> Self {
        self.d_ff = d_ff;
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

    pub fn activation(mut self, activation: CNNActivation) -> Self {
        self.activation = activation;
        self
    }

    pub fn max_position_embeddings(mut self, max_pos: usize) -> Self {
        self.max_position_embeddings = max_pos;
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

    fn initialize_layer_weights(&self, input_dim: usize) -> TransformerLayerWeights {
        TransformerLayerWeights {
            w_q: self.initialize_weights(input_dim, self.d_model),
            w_k: self.initialize_weights(input_dim, self.d_model),
            w_v: self.initialize_weights(input_dim, self.d_model),
            w_o: self.initialize_weights(self.d_model, self.d_model),
            w_ff1: self.initialize_weights(self.d_model, self.d_ff),
            w_ff2: self.initialize_weights(self.d_ff, self.d_model),
            layer_norm_scale: Array1::ones(self.d_model),
            layer_norm_bias: Array1::zeros(self.d_model),
        }
    }

    fn multi_head_attention(
        &self,
        x: &Array2<f64>,
        weights: &TransformerLayerWeights,
    ) -> Array2<f64> {
        let q = x.dot(&weights.w_q);
        let k = x.dot(&weights.w_k);
        let v = x.dot(&weights.w_v);

        let d_k = self.d_model / self.n_heads;
        let mut head_outputs = Vec::new();

        for head in 0..self.n_heads {
            let start_idx = head * d_k;
            let end_idx = start_idx + d_k;

            let q_head = self.slice_head(&q, start_idx, end_idx);
            let k_head = self.slice_head(&k, start_idx, end_idx);
            let v_head = self.slice_head(&v, start_idx, end_idx);

            let attention_scores = q_head.dot(&k_head.t()) / (d_k as f64).sqrt();
            let attention_weights = self.apply_softmax_2d(&attention_scores);
            let head_output = attention_weights.dot(&v_head);

            head_outputs.push(head_output);
        }

        let concatenated = self.concatenate_heads(&head_outputs);
        concatenated.dot(&weights.w_o)
    }

    fn slice_head(&self, tensor: &Array2<f64>, start: usize, end: usize) -> Array2<f64> {
        let (seq_len, _) = tensor.dim();
        let mut result = Array2::zeros((seq_len, end - start));

        for i in 0..seq_len {
            for j in start..end {
                result[(i, j - start)] = tensor[(i, j)];
            }
        }

        result
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

    fn feed_forward(&self, x: &Array2<f64>, weights: &TransformerLayerWeights) -> Array2<f64> {
        let hidden = x.dot(&weights.w_ff1);
        let activated = self.apply_activation(&hidden);
        activated.dot(&weights.w_ff2)
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

    fn layer_norm(&self, x: &Array2<f64>, weights: &TransformerLayerWeights) -> Array2<f64> {
        if !self.use_layer_norm {
            return x.clone();
        }

        let normalized = layer_norm(x);
        let mut result = Array2::zeros(normalized.raw_dim());

        for i in 0..normalized.nrows() {
            for j in 0..normalized.ncols() {
                result[(i, j)] = normalized[(i, j)] * weights.layer_norm_scale[j] + weights.layer_norm_bias[j];
            }
        }

        result
    }

    fn apply_dropout(&self, x: &Array2<f64>) -> Array2<f64> {
        let mut rng = Random::seed_from_u64(42);
        let mut result = x.clone();

        for elem in result.iter_mut() {
            if rng.gen_range(0.0..1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
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

    fn transformer_layer(&self, x: &Array2<f64>, weights: &TransformerLayerWeights) -> Array2<f64> {
        let attention_output = self.multi_head_attention(x, weights);
        let dropout_attention = self.apply_dropout(&attention_output);
        let residual1 = x + &dropout_attention;
        let norm1 = self.layer_norm(&residual1, weights);

        let ff_output = self.feed_forward(&norm1, weights);
        let dropout_ff = self.apply_dropout(&ff_output);
        let residual2 = &norm1 + &dropout_ff;
        self.layer_norm(&residual2, weights)
    }

    fn global_average_pooling(&self, x: &Array2<f64>) -> Array1<f64> {
        let (seq_len, d_model) = x.dim();
        let mut result = Array1::zeros(d_model);

        for j in 0..d_model {
            let mut sum = 0.0;
            for i in 0..seq_len {
                sum += x[(i, j)];
            }
            result[j] = sum / seq_len as f64;
        }

        result
    }

    fn cls_token_pooling(&self, x: &Array2<f64>) -> Array1<f64> {
        x.row(0).to_owned()
    }

    fn max_pooling(&self, x: &Array2<f64>) -> Array1<f64> {
        let (_, d_model) = x.dim();
        let mut result = Array1::zeros(d_model);

        for j in 0..d_model {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..x.nrows() {
                max_val = max_val.max(x[(i, j)]);
            }
            result[j] = max_val;
        }

        result
    }
}

pub struct FittedTransformerFeatureExtractor {
    extractor: TransformerFeatureExtractor,
    layer_weights: Vec<TransformerLayerWeights>,
    positional_encoding: Array2<f64>,
    input_shape: (usize, usize),
}

impl Estimator<Untrained> for TransformerFeatureExtractor {
    fn estimator_type(&self) -> &'static str {
        "TransformerFeatureExtractor"
    }

    fn complexity(&self) -> f64 {
        (self.n_layers * self.d_model * self.d_ff) as f64
    }
}

impl Fit<Array2<f64>, ()> for TransformerFeatureExtractor {
    type Output = FittedTransformerFeatureExtractor;

    fn fit(&self, x: &Array2<f64>, _y: &()) -> SklResult<Self::Output> {
        let (seq_len, d_input) = x.dim();

        if seq_len == 0 || d_input == 0 {
            return Err(SklearsError::InvalidInput("Empty input data".to_string()));
        }

        if seq_len > self.max_position_embeddings {
            return Err(SklearsError::InvalidInput(format!(
                "Sequence length {} exceeds maximum position embeddings {}",
                seq_len, self.max_position_embeddings
            )));
        }

        if self.d_model % self.n_heads != 0 {
            return Err(SklearsError::InvalidInput(
                "d_model must be divisible by n_heads".to_string()
            ));
        }

        let mut layer_weights = Vec::new();
        for _ in 0..self.n_layers {
            layer_weights.push(self.initialize_layer_weights(self.d_model));
        }

        let positional_encoding = self.positional_encoding(seq_len, self.d_model);

        Ok(FittedTransformerFeatureExtractor {
            extractor: self.clone(),
            layer_weights,
            positional_encoding,
            input_shape: (seq_len, d_input),
        })
    }
}

impl Transform<Array2<f64>, Array1<f64>> for FittedTransformerFeatureExtractor {
    fn transform(&self, x: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (seq_len, d_input) = x.dim();

        if (seq_len, d_input) != self.input_shape {
            return Err(SklearsError::InvalidInput(format!(
                "Input shape {:?} does not match fitted shape {:?}",
                (seq_len, d_input), self.input_shape
            )));
        }

        let mut current = x.clone();

        if d_input != self.extractor.d_model {
            let proj_weights = self.extractor.initialize_weights(d_input, self.extractor.d_model);
            current = x.dot(&proj_weights);
        }

        current = &current + &self.positional_encoding;

        for layer_weight in &self.layer_weights {
            current = self.extractor.transformer_layer(&current, layer_weight);
        }

        Ok(self.extractor.global_average_pooling(&current))
    }
}

impl FittedTransformerFeatureExtractor {
    pub fn get_layer_outputs(&self, x: &Array2<f64>) -> SklResult<Vec<Array2<f64>>> {
        let (seq_len, d_input) = x.dim();

        if (seq_len, d_input) != self.input_shape {
            return Err(SklearsError::InvalidInput("Input shape mismatch".to_string()));
        }

        let mut current = x.clone();
        let mut outputs = vec![current.clone()];

        if d_input != self.extractor.d_model {
            let proj_weights = self.extractor.initialize_weights(d_input, self.extractor.d_model);
            current = x.dot(&proj_weights);
        }

        current = &current + &self.positional_encoding;
        outputs.push(current.clone());

        for layer_weight in &self.layer_weights {
            current = self.extractor.transformer_layer(&current, layer_weight);
            outputs.push(current.clone());
        }

        Ok(outputs)
    }

    pub fn extract_with_pooling(&self, x: &Array2<f64>, pooling_method: &str) -> SklResult<Array1<f64>> {
        let (seq_len, d_input) = x.dim();

        if (seq_len, d_input) != self.input_shape {
            return Err(SklearsError::InvalidInput("Input shape mismatch".to_string()));
        }

        let mut current = x.clone();

        if d_input != self.extractor.d_model {
            let proj_weights = self.extractor.initialize_weights(d_input, self.extractor.d_model);
            current = x.dot(&proj_weights);
        }

        current = &current + &self.positional_encoding;

        for layer_weight in &self.layer_weights {
            current = self.extractor.transformer_layer(&current, layer_weight);
        }

        match pooling_method {
            "mean" | "average" => Ok(self.extractor.global_average_pooling(&current)),
            "cls" => Ok(self.extractor.cls_token_pooling(&current)),
            "max" => Ok(self.extractor.max_pooling(&current)),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown pooling method: {}. Use 'mean', 'cls', or 'max'",
                pooling_method
            ))),
        }
    }

    pub fn get_attention_weights(&self, x: &Array2<f64>, layer_idx: usize) -> SklResult<Vec<Array2<f64>>> {
        if layer_idx >= self.layer_weights.len() {
            return Err(SklearsError::InvalidInput("Layer index out of bounds".to_string()));
        }

        let (seq_len, d_input) = x.dim();
        let mut current = x.clone();

        if d_input != self.extractor.d_model {
            let proj_weights = self.extractor.initialize_weights(d_input, self.extractor.d_model);
            current = x.dot(&proj_weights);
        }

        current = &current + &self.positional_encoding;

        for i in 0..layer_idx {
            current = self.extractor.transformer_layer(&current, &self.layer_weights[i]);
        }

        let weights = &self.layer_weights[layer_idx];
        let q = current.dot(&weights.w_q);
        let k = current.dot(&weights.w_k);

        let d_k = self.extractor.d_model / self.extractor.n_heads;
        let mut attention_maps = Vec::new();

        for head in 0..self.extractor.n_heads {
            let start_idx = head * d_k;
            let end_idx = start_idx + d_k;

            let q_head = self.extractor.slice_head(&q, start_idx, end_idx);
            let k_head = self.extractor.slice_head(&k, start_idx, end_idx);

            let attention_scores = q_head.dot(&k_head.t()) / (d_k as f64).sqrt();
            let attention_weights = self.extractor.apply_softmax_2d(&attention_scores);
            attention_maps.push(attention_weights);
        }

        Ok(attention_maps)
    }

    pub fn get_output_shape(&self) -> usize {
        self.extractor.d_model
    }

    pub fn get_num_layers(&self) -> usize {
        self.extractor.n_layers
    }

    pub fn get_positional_encoding(&self) -> &Array2<f64> {
        &self.positional_encoding
    }
}
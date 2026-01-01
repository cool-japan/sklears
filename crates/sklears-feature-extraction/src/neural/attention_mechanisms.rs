use super::neural_types::*;
use super::neural_activations::*;
use super::neural_utilities::*;

pub struct ScaledDotProductAttention {
    pub d_k: usize,
    pub temperature: f64,
    pub dropout_rate: f64,
}

impl ScaledDotProductAttention {
    pub fn new(d_k: usize, temperature: f64, dropout_rate: f64) -> Self {
        Self {
            d_k,
            temperature,
            dropout_rate,
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let scores = self.compute_attention_scores(q, k);
        let masked_scores = self.apply_mask(&scores, mask);
        let attention_weights = self.apply_softmax(&masked_scores);
        let dropout_weights = self.apply_dropout(&attention_weights);
        let output = dropout_weights.dot(v);

        (output, attention_weights)
    }

    fn compute_attention_scores(&self, q: &Array2<f64>, k: &Array2<f64>) -> Array2<f64> {
        let scores = q.dot(&k.t());
        let scale = (self.d_k as f64).sqrt() * self.temperature;
        &scores / scale
    }

    fn apply_mask(&self, scores: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let mut masked_scores = scores.clone();

        if let Some(mask) = mask {
            for i in 0..scores.nrows() {
                for j in 0..scores.ncols() {
                    if !mask[(i, j)] {
                        masked_scores[(i, j)] = f64::NEG_INFINITY;
                    }
                }
            }
        }

        masked_scores
    }

    fn apply_softmax(&self, scores: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(scores.raw_dim());

        for i in 0..scores.nrows() {
            let row = scores.row(i);
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_row: Vec<f64> = row.iter().map(|&val| (val - max_val).exp()).collect();
            let sum_exp: f64 = exp_row.iter().sum();

            for (j, &exp_val) in exp_row.iter().enumerate() {
                result[(i, j)] = exp_val / sum_exp;
            }
        }

        result
    }

    fn apply_dropout(&self, attention_weights: &Array2<f64>) -> Array2<f64> {
        if self.dropout_rate == 0.0 {
            return attention_weights.clone();
        }

        let mut rng = Random::seed_from_u64(42);
        let mut result = attention_weights.clone();

        for elem in result.iter_mut() {
            if rng.random_range(0.0..1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
        }

        result
    }
}

pub struct MultiHeadAttention {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_k: usize,
    pub d_v: usize,
    pub dropout_rate: f64,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array2<f64>,
    pub w_o: Array2<f64>,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout_rate: f64, random_state: Option<u64>) -> Self {
        let d_k = d_model / n_heads;
        let d_v = d_model / n_heads;

        let w_q = xavier_normal_init(d_model, d_model, random_state);
        let w_k = xavier_normal_init(d_model, d_model, random_state);
        let w_v = xavier_normal_init(d_model, d_model, random_state);
        let w_o = xavier_normal_init(d_model, d_model, random_state);

        Self {
            d_model,
            n_heads,
            d_k,
            d_v,
            dropout_rate,
            w_q,
            w_k,
            w_v,
            w_o,
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        let batch_size = q.nrows();

        let q_proj = q.dot(&self.w_q);
        let k_proj = k.dot(&self.w_k);
        let v_proj = v.dot(&self.w_v);

        let q_heads = self.split_heads(&q_proj, batch_size);
        let k_heads = self.split_heads(&k_proj, batch_size);
        let v_heads = self.split_heads(&v_proj, batch_size);

        let mut head_outputs = Vec::new();
        let mut attention_weights = Vec::new();

        let scaled_attention = ScaledDotProductAttention::new(self.d_k, 1.0, self.dropout_rate);

        for i in 0..self.n_heads {
            let (head_output, head_attention) = scaled_attention.forward(
                &q_heads[i],
                &k_heads[i],
                &v_heads[i],
                mask,
            );

            head_outputs.push(head_output);
            attention_weights.push(head_attention);
        }

        let concatenated = self.concatenate_heads(&head_outputs);
        let output = concatenated.dot(&self.w_o);

        (output, attention_weights)
    }

    fn split_heads(&self, tensor: &Array2<f64>, batch_size: usize) -> Vec<Array2<f64>> {
        let seq_len = tensor.nrows();
        let mut heads = Vec::new();

        for head in 0..self.n_heads {
            let start_idx = head * self.d_k;
            let end_idx = start_idx + self.d_k;
            let mut head_tensor = Array2::zeros((seq_len, self.d_k));

            for i in 0..seq_len {
                for j in 0..self.d_k {
                    head_tensor[(i, j)] = tensor[(i, start_idx + j)];
                }
            }

            heads.push(head_tensor);
        }

        heads
    }

    fn concatenate_heads(&self, heads: &[Array2<f64>]) -> Array2<f64> {
        let seq_len = heads[0].nrows();
        let mut result = Array2::zeros((seq_len, self.d_model));

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
}

pub struct AdditiveAttention {
    pub d_model: usize,
    pub d_hidden: usize,
    pub w_q: Array2<f64>,
    pub w_k: Array2<f64>,
    pub w_v: Array1<f64>,
    pub dropout_rate: f64,
}

impl AdditiveAttention {
    pub fn new(d_model: usize, d_hidden: usize, dropout_rate: f64, random_state: Option<u64>) -> Self {
        let w_q = xavier_normal_init(d_model, d_hidden, random_state);
        let w_k = xavier_normal_init(d_model, d_hidden, random_state);
        let w_v = Array1::zeros(d_hidden);

        Self {
            d_model,
            d_hidden,
            w_q,
            w_k,
            w_v,
            dropout_rate,
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Array2<f64>) {
        let q_proj = q.dot(&self.w_q);
        let k_proj = k.dot(&self.w_k);

        let energy = self.compute_energy(&q_proj, &k_proj);
        let masked_energy = self.apply_mask(&energy, mask);
        let attention_weights = self.apply_softmax(&masked_energy);
        let dropout_weights = self.apply_dropout(&attention_weights);
        let output = dropout_weights.dot(v);

        (output, attention_weights)
    }

    fn compute_energy(&self, q_proj: &Array2<f64>, k_proj: &Array2<f64>) -> Array2<f64> {
        let q_len = q_proj.nrows();
        let k_len = k_proj.nrows();
        let mut energy = Array2::zeros((q_len, k_len));

        for i in 0..q_len {
            for j in 0..k_len {
                let mut sum = 0.0;
                for l in 0..self.d_hidden {
                    sum += self.w_v[l] * (q_proj[(i, l)] + k_proj[(j, l)]).tanh();
                }
                energy[(i, j)] = sum;
            }
        }

        energy
    }

    fn apply_mask(&self, energy: &Array2<f64>, mask: Option<&Array2<bool>>) -> Array2<f64> {
        let mut masked_energy = energy.clone();

        if let Some(mask) = mask {
            for i in 0..energy.nrows() {
                for j in 0..energy.ncols() {
                    if !mask[(i, j)] {
                        masked_energy[(i, j)] = f64::NEG_INFINITY;
                    }
                }
            }
        }

        masked_energy
    }

    fn apply_softmax(&self, energy: &Array2<f64>) -> Array2<f64> {
        let mut result = Array2::zeros(energy.raw_dim());

        for i in 0..energy.nrows() {
            let row = energy.row(i);
            let max_val = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_row: Vec<f64> = row.iter().map(|&val| (val - max_val).exp()).collect();
            let sum_exp: f64 = exp_row.iter().sum();

            for (j, &exp_val) in exp_row.iter().enumerate() {
                result[(i, j)] = exp_val / sum_exp;
            }
        }

        result
    }

    fn apply_dropout(&self, attention_weights: &Array2<f64>) -> Array2<f64> {
        if self.dropout_rate == 0.0 {
            return attention_weights.clone();
        }

        let mut rng = Random::seed_from_u64(42);
        let mut result = attention_weights.clone();

        for elem in result.iter_mut() {
            if rng.random_range(0.0..1.0) < self.dropout_rate {
                *elem = 0.0;
            } else {
                *elem /= 1.0 - self.dropout_rate;
            }
        }

        result
    }
}

pub struct SelfAttention {
    pub attention: MultiHeadAttention,
}

impl SelfAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout_rate: f64, random_state: Option<u64>) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, n_heads, dropout_rate, random_state),
        }
    }

    pub fn forward(
        &self,
        x: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        self.attention.forward(x, x, x, mask)
    }
}

pub struct CrossAttention {
    pub attention: MultiHeadAttention,
}

impl CrossAttention {
    pub fn new(d_model: usize, n_heads: usize, dropout_rate: f64, random_state: Option<u64>) -> Self {
        Self {
            attention: MultiHeadAttention::new(d_model, n_heads, dropout_rate, random_state),
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        kv: &Array2<f64>,
        mask: Option<&Array2<bool>>,
    ) -> (Array2<f64>, Vec<Array2<f64>>) {
        self.attention.forward(q, kv, kv, mask)
    }
}

pub struct LocalAttention {
    pub attention: ScaledDotProductAttention,
    pub window_size: usize,
}

impl LocalAttention {
    pub fn new(d_k: usize, window_size: usize, dropout_rate: f64) -> Self {
        Self {
            attention: ScaledDotProductAttention::new(d_k, 1.0, dropout_rate),
            window_size,
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mask = self.create_local_mask(q.nrows(), k.nrows());
        self.attention.forward(q, k, v, Some(&mask))
    }

    fn create_local_mask(&self, q_len: usize, k_len: usize) -> Array2<bool> {
        let mut mask = Array2::from_elem((q_len, k_len), false);

        for i in 0..q_len {
            let start = if i >= self.window_size { i - self.window_size } else { 0 };
            let end = (i + self.window_size + 1).min(k_len);

            for j in start..end {
                mask[(i, j)] = true;
            }
        }

        mask
    }
}

pub struct SparseAttention {
    pub attention: ScaledDotProductAttention,
    pub sparsity_pattern: String,
    pub block_size: usize,
}

impl SparseAttention {
    pub fn new(d_k: usize, sparsity_pattern: String, block_size: usize, dropout_rate: f64) -> Self {
        Self {
            attention: ScaledDotProductAttention::new(d_k, 1.0, dropout_rate),
            sparsity_pattern,
            block_size,
        }
    }

    pub fn forward(
        &self,
        q: &Array2<f64>,
        k: &Array2<f64>,
        v: &Array2<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let mask = self.create_sparse_mask(q.nrows(), k.nrows());
        self.attention.forward(q, k, v, Some(&mask))
    }

    fn create_sparse_mask(&self, q_len: usize, k_len: usize) -> Array2<bool> {
        match self.sparsity_pattern.as_str() {
            "strided" => self.create_strided_mask(q_len, k_len),
            "fixed" => self.create_fixed_mask(q_len, k_len),
            "random" => self.create_random_mask(q_len, k_len),
            _ => Array2::from_elem((q_len, k_len), true),
        }
    }

    fn create_strided_mask(&self, q_len: usize, k_len: usize) -> Array2<bool> {
        let mut mask = Array2::from_elem((q_len, k_len), false);

        for i in 0..q_len {
            for j in (0..k_len).step_by(self.block_size) {
                if j < k_len {
                    mask[(i, j)] = true;
                }
            }
        }

        mask
    }

    fn create_fixed_mask(&self, q_len: usize, k_len: usize) -> Array2<bool> {
        let mut mask = Array2::from_elem((q_len, k_len), false);

        let num_blocks = (k_len + self.block_size - 1) / self.block_size;
        for i in 0..q_len {
            let block_idx = i / self.block_size;
            let start = block_idx * self.block_size;
            let end = (start + self.block_size).min(k_len);

            for j in start..end {
                mask[(i, j)] = true;
            }

            if block_idx > 0 {
                let prev_start = (block_idx - 1) * self.block_size;
                let prev_end = (prev_start + self.block_size).min(k_len);
                for j in prev_start..prev_end {
                    mask[(i, j)] = true;
                }
            }
        }

        mask
    }

    fn create_random_mask(&self, q_len: usize, k_len: usize) -> Array2<bool> {
        let mut rng = Random::seed_from_u64(42);
        let mut mask = Array2::from_elem((q_len, k_len), false);

        let sparsity = 0.1;
        let num_true = (q_len * k_len as f64 * sparsity) as usize;

        for _ in 0..num_true {
            let i = rng.gen_range(0..q_len);
            let j = rng.gen_range(0..k_len);
            mask[(i, j)] = true;
        }

        mask
    }
}

pub fn create_causal_mask(seq_len: usize) -> Array2<bool> {
    let mut mask = Array2::from_elem((seq_len, seq_len), false);

    for i in 0..seq_len {
        for j in 0..=i {
            mask[(i, j)] = true;
        }
    }

    mask
}

pub fn create_padding_mask(seq_lengths: &[usize], max_len: usize) -> Array2<bool> {
    let batch_size = seq_lengths.len();
    let mut mask = Array2::from_elem((batch_size, max_len), false);

    for (i, &length) in seq_lengths.iter().enumerate() {
        for j in 0..length.min(max_len) {
            mask[(i, j)] = true;
        }
    }

    mask
}

pub fn create_lookahead_mask(seq_len: usize, lookahead: usize) -> Array2<bool> {
    let mut mask = Array2::from_elem((seq_len, seq_len), false);

    for i in 0..seq_len {
        let start = if i >= lookahead { i - lookahead } else { 0 };
        let end = (i + lookahead + 1).min(seq_len);

        for j in start..end {
            mask[(i, j)] = true;
        }
    }

    mask
}

pub fn attention_entropy(attention_weights: &Array2<f64>) -> Array1<f64> {
    let seq_len = attention_weights.nrows();
    let mut entropy = Array1::zeros(seq_len);

    for i in 0..seq_len {
        let mut h = 0.0;
        for j in 0..attention_weights.ncols() {
            let p = attention_weights[(i, j)];
            if p > 0.0 {
                h -= p * p.ln();
            }
        }
        entropy[i] = h;
    }

    entropy
}

pub fn attention_variance(attention_weights: &Array2<f64>) -> Array1<f64> {
    let seq_len = attention_weights.nrows();
    let mut variance = Array1::zeros(seq_len);

    for i in 0..seq_len {
        let mean = attention_weights.row(i).mean().unwrap_or(0.0);
        let mut var = 0.0;

        for j in 0..attention_weights.ncols() {
            let diff = attention_weights[(i, j)] - mean;
            var += diff * diff;
        }

        variance[i] = var / attention_weights.ncols() as f64;
    }

    variance
}
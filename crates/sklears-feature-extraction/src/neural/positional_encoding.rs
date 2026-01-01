use super::neural_types::*;
use super::neural_utilities::*;
use scirs2_core::ndarray::s;

pub fn sinusoidal_positional_encoding(seq_len: usize, d_model: usize) -> Array2<f64> {
    let mut encoding = Array2::zeros((seq_len, d_model));

    for pos in 0..seq_len {
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

pub fn learned_positional_encoding(seq_len: usize, d_model: usize, random_state: Option<u64>) -> Array2<f64> {
    xavier_normal_init(seq_len, d_model, random_state)
}

pub fn relative_positional_encoding(seq_len: usize, d_model: usize, max_relative_position: usize) -> Array3<f64> {
    let mut encoding = Array3::zeros((seq_len, seq_len, d_model));

    for i in 0..seq_len {
        for j in 0..seq_len {
            let relative_pos = ((i as i32 - j as i32).abs() as usize).min(max_relative_position);

            for k in 0..d_model {
                let angle = relative_pos as f64 / (10000.0_f64).powf((2 * (k / 2)) as f64 / d_model as f64);
                if k % 2 == 0 {
                    encoding[(i, j, k)] = angle.sin();
                } else {
                    encoding[(i, j, k)] = angle.cos();
                }
            }
        }
    }

    encoding
}

pub fn rotary_positional_encoding(seq_len: usize, d_model: usize) -> Array2<f64> {
    assert_eq!(d_model % 2, 0, "d_model must be even for rotary encoding");

    let mut encoding = Array2::zeros((seq_len, d_model));

    for pos in 0..seq_len {
        for i in (0..d_model).step_by(2) {
            let theta = pos as f64 / (10000.0_f64).powf(i as f64 / d_model as f64);
            encoding[(pos, i)] = theta.cos();
            encoding[(pos, i + 1)] = theta.sin();
        }
    }

    encoding
}

pub fn apply_rotary_encoding(x: &Array2<f64>, pos_encoding: &Array2<f64>) -> Array2<f64> {
    let (seq_len, d_model) = x.dim();
    let mut result = Array2::zeros((seq_len, d_model));

    for i in 0..seq_len {
        for j in (0..d_model).step_by(2) {
            let cos_theta = pos_encoding[(i, j)];
            let sin_theta = pos_encoding[(i, j + 1)];

            let x1 = x[(i, j)];
            let x2 = x[(i, j + 1)];

            result[(i, j)] = x1 * cos_theta - x2 * sin_theta;
            result[(i, j + 1)] = x1 * sin_theta + x2 * cos_theta;
        }
    }

    result
}

pub fn alibi_positional_bias(seq_len: usize, n_heads: usize) -> Array3<f64> {
    let mut bias = Array3::zeros((n_heads, seq_len, seq_len));

    let slopes = alibi_get_slopes(n_heads);

    for h in 0..n_heads {
        for i in 0..seq_len {
            for j in 0..seq_len {
                if j <= i {
                    bias[(h, i, j)] = -slopes[h] * (i - j) as f64;
                } else {
                    bias[(h, i, j)] = f64::NEG_INFINITY;
                }
            }
        }
    }

    bias
}

fn alibi_get_slopes(n_heads: usize) -> Vec<f64> {
    let mut slopes = Vec::new();

    if n_heads.is_power_of_two() {
        let start = 2.0_f64.powf(-8.0 / n_heads as f64);
        let ratio = 2.0_f64.powf(-1.0);

        for i in 0..n_heads {
            slopes.push(start * ratio.powf(i as f64));
        }
    } else {
        let closest_power_of_2 = 2_usize.pow((n_heads as f64).log2().floor() as u32);
        let extra_heads = n_heads - closest_power_of_2;

        let base_slopes = alibi_get_slopes(closest_power_of_2);
        slopes.extend_from_slice(&base_slopes);

        let start = 2.0_f64.powf(-8.0 / (2 * closest_power_of_2) as f64);
        let ratio = 2.0_f64.powf(-2.0);

        for i in 0..extra_heads {
            slopes.push(start * ratio.powf(i as f64));
        }
    }

    slopes
}

pub struct PositionalEmbedding {
    pub encoding_type: PositionalEncodingType,
    pub seq_len: usize,
    pub d_model: usize,
    pub learned_embeddings: Option<Array2<f64>>,
    pub max_relative_position: usize,
    pub dropout_rate: f64,
}

#[derive(Debug, Clone, Copy)]
pub enum PositionalEncodingType {
    Sinusoidal,
    Learned,
    Relative,
    Rotary,
    ALiBi,
    None,
}

impl PositionalEmbedding {
    pub fn new(
        encoding_type: PositionalEncodingType,
        seq_len: usize,
        d_model: usize,
        random_state: Option<u64>,
    ) -> Self {
        let learned_embeddings = match encoding_type {
            PositionalEncodingType::Learned => Some(learned_positional_encoding(seq_len, d_model, random_state)),
            _ => None,
        };

        Self {
            encoding_type,
            seq_len,
            d_model,
            learned_embeddings,
            max_relative_position: 32,
            dropout_rate: 0.1,
        }
    }

    pub fn with_max_relative_position(mut self, max_relative_position: usize) -> Self {
        self.max_relative_position = max_relative_position;
        self
    }

    pub fn with_dropout(mut self, dropout_rate: f64) -> Self {
        self.dropout_rate = dropout_rate;
        self
    }

    pub fn get_encoding(&self, actual_seq_len: usize) -> Array2<f64> {
        let effective_seq_len = actual_seq_len.min(self.seq_len);

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal => {
                sinusoidal_positional_encoding(effective_seq_len, self.d_model)
            },
            PositionalEncodingType::Learned => {
                if let Some(ref embeddings) = self.learned_embeddings {
                    embeddings.slice(s![..effective_seq_len, ..]).to_owned()
                } else {
                    Array2::zeros((effective_seq_len, self.d_model))
                }
            },
            PositionalEncodingType::Rotary => {
                rotary_positional_encoding(effective_seq_len, self.d_model)
            },
            _ => Array2::zeros((effective_seq_len, self.d_model)),
        }
    }

    pub fn apply_encoding(&self, x: &Array2<f64>) -> Array2<f64> {
        let (seq_len, d_model) = x.dim();

        if d_model != self.d_model {
            panic!("Input d_model {} does not match expected d_model {}", d_model, self.d_model);
        }

        match self.encoding_type {
            PositionalEncodingType::Sinusoidal | PositionalEncodingType::Learned => {
                let encoding = self.get_encoding(seq_len);
                x + &encoding
            },
            PositionalEncodingType::Rotary => {
                let encoding = self.get_encoding(seq_len);
                apply_rotary_encoding(x, &encoding)
            },
            PositionalEncodingType::None => x.clone(),
            _ => x.clone(),
        }
    }

    pub fn get_relative_encoding(&self, seq_len: usize) -> Array3<f64> {
        relative_positional_encoding(seq_len, self.d_model, self.max_relative_position)
    }

    pub fn get_alibi_bias(&self, seq_len: usize, n_heads: usize) -> Array3<f64> {
        alibi_positional_bias(seq_len, n_heads)
    }

    pub fn apply_dropout(&self, x: &Array2<f64>) -> Array2<f64> {
        if self.dropout_rate == 0.0 {
            return x.clone();
        }

        let mut rng = Random::seed_from_u64(42);
        let mut result = x.clone();

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

pub fn create_position_ids(seq_len: usize, batch_size: usize) -> Array2<usize> {
    let mut position_ids = Array2::zeros((batch_size, seq_len));

    for i in 0..batch_size {
        for j in 0..seq_len {
            position_ids[(i, j)] = j;
        }
    }

    position_ids
}

pub fn create_token_type_ids(seq_lengths: &[usize], max_len: usize) -> Array2<usize> {
    let batch_size = seq_lengths.len();
    let mut token_type_ids = Array2::zeros((batch_size, max_len));

    for (i, &length) in seq_lengths.iter().enumerate() {
        for j in 0..length.min(max_len) {
            token_type_ids[(i, j)] = if j < length / 2 { 0 } else { 1 };
        }
    }

    token_type_ids
}

pub fn interpolate_positional_encoding(
    original_encoding: &Array2<f64>,
    new_seq_len: usize,
) -> Array2<f64> {
    let (original_seq_len, d_model) = original_encoding.dim();

    if new_seq_len == original_seq_len {
        return original_encoding.clone();
    }

    let mut interpolated = Array2::zeros((new_seq_len, d_model));

    for i in 0..new_seq_len {
        let position = (i as f64 * (original_seq_len - 1) as f64) / (new_seq_len - 1) as f64;
        let lower_idx = position.floor() as usize;
        let upper_idx = (lower_idx + 1).min(original_seq_len - 1);
        let weight = position - position.floor();

        for j in 0..d_model {
            if lower_idx == upper_idx {
                interpolated[(i, j)] = original_encoding[(lower_idx, j)];
            } else {
                interpolated[(i, j)] = (1.0 - weight) * original_encoding[(lower_idx, j)]
                    + weight * original_encoding[(upper_idx, j)];
            }
        }
    }

    interpolated
}

pub fn extrapolate_positional_encoding(
    original_encoding: &Array2<f64>,
    new_seq_len: usize,
) -> Array2<f64> {
    let (original_seq_len, d_model) = original_encoding.dim();

    if new_seq_len <= original_seq_len {
        return interpolate_positional_encoding(original_encoding, new_seq_len);
    }

    let mut extrapolated = Array2::zeros((new_seq_len, d_model));

    for j in 0..d_model {
        let last_value = original_encoding[(original_seq_len - 1, j)];
        let second_last_value = if original_seq_len > 1 {
            original_encoding[(original_seq_len - 2, j)]
        } else {
            0.0
        };
        let diff = last_value - second_last_value;

        for i in 0..original_seq_len {
            extrapolated[(i, j)] = original_encoding[(i, j)];
        }

        for i in original_seq_len..new_seq_len {
            let steps = (i - original_seq_len + 1) as f64;
            extrapolated[(i, j)] = last_value + diff * steps;
        }
    }

    extrapolated
}

pub fn compute_positional_similarity(
    pos1: usize,
    pos2: usize,
    encoding: &Array2<f64>,
) -> f64 {
    if pos1 >= encoding.nrows() || pos2 >= encoding.nrows() {
        return 0.0;
    }

    let vec1 = encoding.row(pos1);
    let vec2 = encoding.row(pos2);

    cosine_similarity(&vec1.to_owned(), &vec2.to_owned())
}

pub fn positional_encoding_temperature_scaling(
    encoding: &Array2<f64>,
    temperature: f64,
) -> Array2<f64> {
    encoding / temperature
}

pub fn adaptive_positional_encoding(
    seq_len: usize,
    d_model: usize,
    adaptation_factor: f64,
) -> Array2<f64> {
    let base_encoding = sinusoidal_positional_encoding(seq_len, d_model);
    let mut adaptive_encoding = Array2::zeros((seq_len, d_model));

    for i in 0..seq_len {
        let position_weight = 1.0 + adaptation_factor * (i as f64 / seq_len as f64);
        for j in 0..d_model {
            adaptive_encoding[(i, j)] = base_encoding[(i, j)] * position_weight;
        }
    }

    adaptive_encoding
}

pub fn hierarchical_positional_encoding(
    seq_len: usize,
    d_model: usize,
    hierarchy_levels: usize,
) -> Array2<f64> {
    let mut encoding = Array2::zeros((seq_len, d_model));
    let dims_per_level = d_model / hierarchy_levels;

    for level in 0..hierarchy_levels {
        let scale = 2_usize.pow(level as u32);
        let start_dim = level * dims_per_level;
        let end_dim = ((level + 1) * dims_per_level).min(d_model);

        for pos in 0..seq_len {
            for i in start_dim..end_dim {
                let dim_in_level = i - start_dim;
                let angle = (pos / scale) as f64 / (10000.0_f64).powf((2 * (dim_in_level / 2)) as f64 / dims_per_level as f64);

                if dim_in_level % 2 == 0 {
                    encoding[(pos, i)] = angle.sin();
                } else {
                    encoding[(pos, i)] = angle.cos();
                }
            }
        }
    }

    encoding
}

pub fn contextual_positional_encoding(
    context_vectors: &Array2<f64>,
    d_model: usize,
    random_state: Option<u64>,
) -> Array2<f64> {
    let seq_len = context_vectors.nrows();
    let context_dim = context_vectors.ncols();

    let projection_matrix = xavier_normal_init(context_dim, d_model, random_state);
    let base_encoding = sinusoidal_positional_encoding(seq_len, d_model);

    let projected_context = context_vectors.dot(&projection_matrix);
    &base_encoding + &projected_context
}
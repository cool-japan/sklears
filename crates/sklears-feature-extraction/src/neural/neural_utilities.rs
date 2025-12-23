use super::neural_types::*;

pub fn xavier_normal_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.gen_range(-std..std);
        }
    }

    weights
}

pub fn xavier_uniform_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.gen_range(-limit..limit);
        }
    }

    weights
}

pub fn he_normal_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let std = (2.0 / input_dim as f64).sqrt();
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.gen_range(-std..std);
        }
    }

    weights
}

pub fn he_uniform_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let limit = (6.0 / input_dim as f64).sqrt();
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.gen_range(-limit..limit);
        }
    }

    weights
}

pub fn lecun_normal_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let std = (1.0 / input_dim as f64).sqrt();
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.gen_range(-std..std);
        }
    }

    weights
}

pub fn orthogonal_init(input_dim: usize, output_dim: usize, random_state: Option<u64>) -> Array2<f64> {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let min_dim = input_dim.min(output_dim);
    let mut weights = Array2::zeros((input_dim, output_dim));

    for i in 0..input_dim {
        for j in 0..output_dim {
            weights[(i, j)] = rng.random_range(-1.0, 1.0);
        }
    }

    if input_dim >= output_dim {
        qr_decomposition(&mut weights);
    } else {
        let mut weights_t = Array2::zeros((output_dim, input_dim));
        for i in 0..input_dim {
            for j in 0..output_dim {
                weights_t[(j, i)] = weights[(i, j)];
            }
        }
        qr_decomposition(&mut weights_t);
        for i in 0..input_dim {
            for j in 0..output_dim {
                weights[(i, j)] = weights_t[(j, i)];
            }
        }
    }

    weights
}

fn qr_decomposition(matrix: &mut Array2<f64>) {
    let (m, n) = matrix.dim();

    for k in 0..n.min(m) {
        let mut norm = 0.0;
        for i in k..m {
            norm += matrix[(i, k)] * matrix[(i, k)];
        }
        norm = norm.sqrt();

        if norm != 0.0 {
            if matrix[(k, k)] < 0.0 {
                norm = -norm;
            }

            for i in k..m {
                matrix[(i, k)] /= norm;
            }
            matrix[(k, k)] += 1.0;

            for j in (k + 1)..n {
                let mut s = 0.0;
                for i in k..m {
                    s += matrix[(i, k)] * matrix[(i, j)];
                }
                s = -s / matrix[(k, k)];

                for i in k..m {
                    matrix[(i, j)] += s * matrix[(i, k)];
                }
            }
        }
    }

    for i in 0..m {
        for j in 0..n {
            if i > j {
                matrix[(i, j)] = 0.0;
            }
        }
    }
}

pub fn clip_gradients(gradients: &mut Array2<f64>, max_norm: f64) {
    let mut grad_norm = 0.0;
    for grad in gradients.iter() {
        grad_norm += grad * grad;
    }
    grad_norm = grad_norm.sqrt();

    if grad_norm > max_norm {
        let scale = max_norm / grad_norm;
        for grad in gradients.iter_mut() {
            *grad *= scale;
        }
    }
}

pub fn compute_gradient_norm(gradients: &Array2<f64>) -> f64 {
    gradients.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

pub fn l2_regularization_loss(weights: &Array2<f64>, lambda: f64) -> f64 {
    lambda * weights.iter().map(|&x| x * x).sum::<f64>()
}

pub fn l1_regularization_loss(weights: &Array2<f64>, lambda: f64) -> f64 {
    lambda * weights.iter().map(|&x| x.abs()).sum::<f64>()
}

pub fn elastic_net_regularization_loss(weights: &Array2<f64>, l1_ratio: f64, lambda: f64) -> f64 {
    let l1_loss = l1_regularization_loss(weights, lambda * l1_ratio);
    let l2_loss = l2_regularization_loss(weights, lambda * (1.0 - l1_ratio));
    l1_loss + l2_loss
}

pub fn huber_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, delta: f64) -> f64 {
    let mut loss = 0.0;

    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let residual = (true_val - pred_val).abs();
        if residual <= delta {
            loss += 0.5 * residual * residual;
        } else {
            loss += delta * (residual - 0.5 * delta);
        }
    }

    loss / y_true.len() as f64
}

pub fn focal_loss(y_true: &Array1<f64>, y_pred: &Array1<f64>, alpha: f64, gamma: f64) -> f64 {
    let mut loss = 0.0;

    for (&true_val, &pred_val) in y_true.iter().zip(y_pred.iter()) {
        let p_t = if true_val == 1.0 { pred_val } else { 1.0 - pred_val };
        let alpha_t = if true_val == 1.0 { alpha } else { 1.0 - alpha };

        loss += -alpha_t * (1.0 - p_t).powf(gamma) * p_t.ln();
    }

    loss / y_true.len() as f64
}

pub fn cosine_similarity(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    let dot_product = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum::<f64>();
    let norm_a = a.iter().map(|&x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

pub fn euclidean_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub fn manhattan_distance(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .sum::<f64>()
}

pub fn normalize_vector(vector: &mut Array1<f64>) {
    let norm = vector.iter().map(|&x| x * x).sum::<f64>().sqrt();
    if norm > 0.0 {
        for elem in vector.iter_mut() {
            *elem /= norm;
        }
    }
}

pub fn normalize_matrix_rows(matrix: &mut Array2<f64>) {
    let (rows, _) = matrix.dim();

    for i in 0..rows {
        let mut row_norm = 0.0;
        for j in 0..matrix.ncols() {
            row_norm += matrix[(i, j)] * matrix[(i, j)];
        }
        row_norm = row_norm.sqrt();

        if row_norm > 0.0 {
            for j in 0..matrix.ncols() {
                matrix[(i, j)] /= row_norm;
            }
        }
    }
}

pub fn normalize_matrix_cols(matrix: &mut Array2<f64>) {
    let (_, cols) = matrix.dim();

    for j in 0..cols {
        let mut col_norm = 0.0;
        for i in 0..matrix.nrows() {
            col_norm += matrix[(i, j)] * matrix[(i, j)];
        }
        col_norm = col_norm.sqrt();

        if col_norm > 0.0 {
            for i in 0..matrix.nrows() {
                matrix[(i, j)] /= col_norm;
            }
        }
    }
}

pub fn standard_scaler_fit(data: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n_samples, n_features) = data.dim();
    let mut means = Array1::zeros(n_features);
    let mut stds = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut sum = 0.0;
        for i in 0..n_samples {
            sum += data[(i, j)];
        }
        means[j] = sum / n_samples as f64;

        let mut var = 0.0;
        for i in 0..n_samples {
            let diff = data[(i, j)] - means[j];
            var += diff * diff;
        }
        stds[j] = (var / n_samples as f64).sqrt();

        if stds[j] == 0.0 {
            stds[j] = 1.0;
        }
    }

    (means, stds)
}

pub fn standard_scaler_transform(data: &Array2<f64>, means: &Array1<f64>, stds: &Array1<f64>) -> Array2<f64> {
    let (n_samples, n_features) = data.dim();
    let mut scaled = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            scaled[(i, j)] = (data[(i, j)] - means[j]) / stds[j];
        }
    }

    scaled
}

pub fn min_max_scaler_fit(data: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let (n_samples, n_features) = data.dim();
    let mut mins = Array1::zeros(n_features);
    let mut maxs = Array1::zeros(n_features);

    for j in 0..n_features {
        let mut min_val = f64::INFINITY;
        let mut max_val = f64::NEG_INFINITY;

        for i in 0..n_samples {
            min_val = min_val.min(data[(i, j)]);
            max_val = max_val.max(data[(i, j)]);
        }

        mins[j] = min_val;
        maxs[j] = max_val;
    }

    (mins, maxs)
}

pub fn min_max_scaler_transform(data: &Array2<f64>, mins: &Array1<f64>, maxs: &Array1<f64>) -> Array2<f64> {
    let (n_samples, n_features) = data.dim();
    let mut scaled = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            let range = maxs[j] - mins[j];
            if range > 0.0 {
                scaled[(i, j)] = (data[(i, j)] - mins[j]) / range;
            } else {
                scaled[(i, j)] = 0.0;
            }
        }
    }

    scaled
}

pub fn create_batches(data: &Array2<f64>, batch_size: usize) -> Vec<Array2<f64>> {
    let n_samples = data.nrows();
    let n_features = data.ncols();
    let mut batches = Vec::new();

    for start in (0..n_samples).step_by(batch_size) {
        let end = (start + batch_size).min(n_samples);
        let batch_rows = end - start;
        let mut batch = Array2::zeros((batch_rows, n_features));

        for i in 0..batch_rows {
            for j in 0..n_features {
                batch[(i, j)] = data[(start + i, j)];
            }
        }

        batches.push(batch);
    }

    batches
}

pub fn shuffle_data(data: &mut Array2<f64>, random_state: Option<u64>) {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let n_samples = data.nrows();
    let n_features = data.ncols();

    for i in (1..n_samples).rev() {
        let j = rng.gen_range(0..i + 1);

        if i != j {
            for k in 0..n_features {
                let temp = data[(i, k)];
                data[(i, k)] = data[(j, k)];
                data[(j, k)] = temp;
            }
        }
    }
}

pub fn train_test_split(
    data: &Array2<f64>,
    test_size: f64,
    random_state: Option<u64>,
) -> (Array2<f64>, Array2<f64>) {
    let mut rng = match random_state {
        Some(seed) => Random::seed_from_u64(seed),
        None => Random::seed_from_u64(42),
    };

    let n_samples = data.nrows();
    let n_features = data.ncols();
    let n_test = (n_samples as f64 * test_size) as usize;
    let n_train = n_samples - n_test;

    let mut indices: Vec<usize> = (0..n_samples).collect();

    for i in (1..n_samples).rev() {
        let j = rng.gen_range(0..i + 1);
        indices.swap(i, j);
    }

    let mut train_data = Array2::zeros((n_train, n_features));
    let mut test_data = Array2::zeros((n_test, n_features));

    for (i, &idx) in indices[..n_train].iter().enumerate() {
        for j in 0..n_features {
            train_data[(i, j)] = data[(idx, j)];
        }
    }

    for (i, &idx) in indices[n_train..].iter().enumerate() {
        for j in 0..n_features {
            test_data[(i, j)] = data[(idx, j)];
        }
    }

    (train_data, test_data)
}
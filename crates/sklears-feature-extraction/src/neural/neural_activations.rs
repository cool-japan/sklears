use super::neural_types::*;

impl CNNActivation {
    pub fn apply(&self, x: f64) -> f64 {
        match self {
            CNNActivation::ReLU => x.max(0.0),
            CNNActivation::Tanh => x.tanh(),
            CNNActivation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            CNNActivation::LeakyReLU(alpha) => {
                if x > 0.0 { x } else { alpha * x }
            }
        }
    }

    pub fn derivative(&self, x: f64) -> f64 {
        match self {
            CNNActivation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            CNNActivation::Tanh => {
                let tanh_val = x.tanh();
                1.0 - tanh_val * tanh_val
            },
            CNNActivation::Sigmoid => {
                let sig = self.apply(x);
                sig * (1.0 - sig)
            },
            CNNActivation::LeakyReLU(alpha) => {
                if x > 0.0 { 1.0 } else { *alpha }
            }
        }
    }
}

pub fn softmax(x: &Array1<f64>) -> Array1<f64> {
    let max_val = x.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let exp_vals: Array1<f64> = x.mapv(|v| (v - max_val).exp());
    let sum = exp_vals.sum();
    exp_vals.mapv(|v| v / sum)
}

pub fn layer_norm(x: &Array2<f64>) -> Array2<f64> {
    let (n_samples, n_features) = x.dim();
    let mut normalized = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        let row = x.row(i);
        let mean = row.mean().unwrap_or(0.0);
        let variance = row.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n_features as f64;
        let std_dev = (variance + 1e-8).sqrt();

        for j in 0..n_features {
            normalized[(i, j)] = (x[(i, j)] - mean) / std_dev;
        }
    }

    normalized
}
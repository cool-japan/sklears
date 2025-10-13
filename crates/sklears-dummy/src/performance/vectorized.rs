//! Vectorized operations for high-performance dummy estimator computations

use scirs2_core::ndarray::{Array1, Array2, Axis};

/// Vectorized prediction operations
pub fn vectorized_predict_mean(data: &Array2<f64>) -> Array1<f64> {
    data.mean_axis(Axis(1)).unwrap()
}

/// Vectorized statistical computations
pub fn vectorized_stats(data: &Array2<f64>) -> (Array1<f64>, Array1<f64>) {
    let means = data.mean_axis(Axis(0)).unwrap();
    let vars = data.var_axis(Axis(0), 1.0);
    (means, vars)
}

/// Vectorized normalization
pub fn vectorized_normalize(mut data: Array2<f64>) -> Array2<f64> {
    let (means, stds) = vectorized_stats(&data);

    for mut row in data.rows_mut() {
        for (i, val) in row.iter_mut().enumerate() {
            if stds[i] > 1e-8 {
                *val = (*val - means[i]) / stds[i];
            } else {
                *val -= means[i];
            }
        }
    }

    data
}

/// Vectorized distance computations
pub fn vectorized_euclidean_distances(x: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
    let mut distances = Array2::zeros((x.nrows(), y.nrows()));

    for (i, x_row) in x.rows().into_iter().enumerate() {
        for (j, y_row) in y.rows().into_iter().enumerate() {
            let diff = &x_row - &y_row;
            distances[[i, j]] = diff.mapv(|x| x * x).sum().sqrt();
        }
    }

    distances
}

/// Vectorized transformations
pub mod transforms {
    use super::*;

    pub fn log_transform(data: &Array2<f64>) -> Array2<f64> {
        data.mapv(|x| if x > 0.0 { x.ln() } else { f64::NAN })
    }

    pub fn sqrt_transform(data: &Array2<f64>) -> Array2<f64> {
        data.mapv(|x| if x >= 0.0 { x.sqrt() } else { f64::NAN })
    }

    pub fn power_transform(data: &Array2<f64>, power: f64) -> Array2<f64> {
        data.mapv(|x| x.powf(power))
    }
}

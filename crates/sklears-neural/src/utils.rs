//! Neural network utility functions and helper modules.
//!
//! This module provides core neural network utility functions including
//! weight initialization, batch processing, early stopping, encoding utilities,
//! and accuracy metrics.

use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

/// Type alias for neural network weights and biases
pub type WeightsAndBiases = (Vec<Array2<f64>>, Vec<Array1<f64>>);

/// Weight initialization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum WeightInit {
    Zero,
    Random,
    Xavier,
    He,
    Uniform { low: f64, high: f64 },
    Normal { mean: f64, std: f64 },
}

/// Batch configuration for training
#[derive(Debug, Clone)]
pub struct BatchConfig {
    pub batch_size: usize,
    pub shuffle: bool,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            shuffle: true,
        }
    }
}

/// Early stopping configuration and implementation
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    pub patience: usize,
    pub min_delta: f64,
    pub restore_best_weights: bool,
    current_patience: usize,
    best_loss: f64,
    best_weights: Option<WeightsAndBiases>,
}

impl EarlyStopping {
    pub fn new(patience: usize, min_delta: f64, restore_best_weights: bool) -> Self {
        Self {
            patience,
            min_delta,
            restore_best_weights,
            current_patience: 0,
            best_loss: f64::INFINITY,
            best_weights: None,
        }
    }

    pub fn should_stop(
        &mut self,
        loss: f64,
        weights: &[Array2<f64>],
        biases: &[Array1<f64>],
    ) -> bool {
        if loss < self.best_loss - self.min_delta {
            self.best_loss = loss;
            self.current_patience = 0;
            if self.restore_best_weights {
                self.best_weights = Some((weights.to_vec(), biases.to_vec()));
            }
            false
        } else {
            self.current_patience += 1;
            self.current_patience >= self.patience
        }
    }

    pub fn get_best_weights(&self) -> Option<&WeightsAndBiases> {
        self.best_weights.as_ref()
    }
}

/// One-hot encoding utility functions
pub fn one_hot_encode(labels: &[usize], num_classes: Option<usize>) -> Array2<f64> {
    let n_samples = labels.len();
    let n_classes = num_classes.unwrap_or_else(|| labels.iter().max().unwrap_or(&0) + 1);

    let mut encoded = Array2::zeros((n_samples, n_classes));
    for (i, &label) in labels.iter().enumerate() {
        if label < n_classes {
            encoded[[i, label]] = 1.0;
        }
    }
    encoded
}

/// Decode one-hot encoded data back to labels
pub fn one_hot_decode(encoded: &Array2<f64>) -> Vec<usize> {
    encoded
        .rows()
        .into_iter()
        .map(|row| {
            row.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0)
        })
        .collect()
}

/// Calculate accuracy between predictions and true labels
pub fn accuracy(y_true: &[usize], y_pred: &[usize]) -> f64 {
    if y_true.len() != y_pred.len() {
        return 0.0;
    }

    let correct = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(true_val, pred_val)| true_val == pred_val)
        .count();

    correct as f64 / y_true.len() as f64
}

/// Create batches for training data
pub fn create_batches(
    x: &Array2<f64>,
    y: &[usize],
    batch_size: usize,
    shuffle: bool,
) -> Vec<(Array2<f64>, Vec<usize>)> {
    let n_samples = x.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        use scirs2_core::random::seq::SliceRandom;
        let mut rng = scirs2_core::random::thread_rng();
        indices.shuffle(&mut rng);
    }

    let mut batches = Vec::new();
    for chunk in indices.chunks(batch_size) {
        let batch_x = Array2::from_shape_vec(
            (chunk.len(), x.ncols()),
            chunk.iter().flat_map(|&i| x.row(i).to_vec()).collect(),
        )
        .unwrap();

        let batch_y: Vec<usize> = chunk.iter().map(|&i| y[i]).collect();
        batches.push((batch_x, batch_y));
    }

    batches
}

/// Create batches for regression data
pub fn create_batches_regression(
    x: &Array2<f64>,
    y: &Array2<f64>,
    batch_size: usize,
    shuffle: bool,
) -> Vec<(Array2<f64>, Array2<f64>)> {
    let n_samples = x.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        use scirs2_core::random::seq::SliceRandom;
        let mut rng = scirs2_core::random::thread_rng();
        indices.shuffle(&mut rng);
    }

    let mut batches = Vec::new();
    for chunk in indices.chunks(batch_size) {
        let batch_x = Array2::from_shape_vec(
            (chunk.len(), x.ncols()),
            chunk.iter().flat_map(|&i| x.row(i).to_vec()).collect(),
        )
        .unwrap();

        let batch_y = Array2::from_shape_vec(
            (chunk.len(), y.ncols()),
            chunk.iter().flat_map(|&i| y.row(i).to_vec()).collect(),
        )
        .unwrap();
        batches.push((batch_x, batch_y));
    }

    batches
}

/// Initialize weights based on initialization strategy
pub fn initialize_weights(rows: usize, cols: usize, init: &WeightInit) -> Array2<f64> {
    use scirs2_core::random::essentials::{Normal, Uniform};
    use scirs2_core::random::{thread_rng, Rng};

    let mut rng = thread_rng();

    match init {
        WeightInit::Zero => Array2::zeros((rows, cols)),
        WeightInit::Random => {
            let dist = Uniform::new(-1.0, 1.0).unwrap();
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Xavier => {
            let fan_avg = (rows + cols) as f64 / 2.0;
            let bound = (6.0 / fan_avg).sqrt();
            let dist = Uniform::new(-bound, bound).unwrap();
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::He => {
            let std = (2.0 / rows as f64).sqrt();
            let dist = Normal::new(0.0, std).unwrap();
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Uniform { low, high } => {
            let dist = Uniform::new(*low, *high).unwrap();
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).unwrap();
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
    }
}

/// Initialize biases
pub fn initialize_biases(size: usize, init: &WeightInit) -> Array1<f64> {
    use scirs2_core::random::essentials::{Normal, Uniform};
    use scirs2_core::random::{thread_rng, Rng};

    let mut rng = thread_rng();

    match init {
        WeightInit::Zero => Array1::zeros(size),
        WeightInit::Random => {
            let dist = Uniform::new(-1.0, 1.0).unwrap();
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Xavier => {
            let bound = (6.0 / size as f64).sqrt();
            let dist = Uniform::new(-bound, bound).unwrap();
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::He => {
            let std = (2.0 / size as f64).sqrt();
            let dist = Normal::new(0.0, std).unwrap();
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Uniform { low, high } => {
            let dist = Uniform::new(*low, *high).unwrap();
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).unwrap();
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
    }
}

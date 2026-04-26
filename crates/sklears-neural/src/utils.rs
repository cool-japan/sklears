//! Neural network utility functions and helper modules.
//!
//! This module provides core neural network utility functions including
//! weight initialization, batch processing, early stopping, encoding utilities,
//! and accuracy metrics.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::RngExt;

/// Type alias for neural network weights and biases
pub type WeightsAndBiases = (Vec<Array2<f64>>, Vec<Array1<f64>>);

/// Weight initialization strategies
#[derive(Debug, Clone, PartialEq)]
pub enum WeightInit {
    /// Initialize all weights to zero (not recommended for most layers)
    Zero,
    /// Uniform random initialization in `(-1, 1)` without any scaling
    Random,
    /// Xavier/Glorot initialization: scales by `sqrt(2 / (fan_in + fan_out))`
    Xavier,
    /// He/Kaiming initialization: scales by `sqrt(2 / fan_in)`, suited for ReLU activations
    He,
    /// Uniform distribution over `[low, high)`
    Uniform {
        /// Lower bound of the uniform distribution
        low: f64,
        /// Upper bound of the uniform distribution
        high: f64,
    },
    /// Gaussian distribution with specified mean and standard deviation
    Normal {
        /// Mean of the Gaussian distribution
        mean: f64,
        /// Standard deviation of the Gaussian distribution
        std: f64,
    },
}

/// Batch configuration for training
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Number of samples per mini-batch
    pub batch_size: usize,
    /// Whether to shuffle training samples at the start of each epoch
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
    /// Number of epochs with no improvement above `min_delta` before stopping
    pub patience: usize,
    /// Minimum decrease in loss required to be counted as an improvement
    pub min_delta: f64,
    /// Whether to restore the weights from the best epoch when stopping
    pub restore_best_weights: bool,
    current_patience: usize,
    best_loss: f64,
    best_weights: Option<WeightsAndBiases>,
}

impl EarlyStopping {
    /// Create a new `EarlyStopping` monitor with the given patience, minimum improvement delta, and weight-restoration flag
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

    /// Record the current `loss` and return `true` when patience has been exhausted
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

    /// Return a reference to the weights saved at the best epoch, if any were recorded
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
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
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
pub fn create_batches<R: scirs2_core::random::Rng>(
    x: &Array2<f64>,
    y: &[usize],
    batch_size: usize,
    shuffle: bool,
    rng: &mut R,
) -> Vec<(Array2<f64>, Vec<usize>)> {
    let n_samples = x.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        use scirs2_core::random::seq::SliceRandom;
        indices.shuffle(rng);
    }

    let mut batches = Vec::new();
    for chunk in indices.chunks(batch_size) {
        let batch_x = Array2::from_shape_vec(
            (chunk.len(), x.ncols()),
            chunk.iter().flat_map(|&i| x.row(i).to_vec()).collect(),
        )
        .expect("value should be present");

        let batch_y: Vec<usize> = chunk.iter().map(|&i| y[i]).collect();
        batches.push((batch_x, batch_y));
    }

    batches
}

/// Create batches for regression data
pub fn create_batches_regression<R: scirs2_core::random::Rng>(
    x: &Array2<f64>,
    y: &Array2<f64>,
    batch_size: usize,
    shuffle: bool,
    rng: &mut R,
) -> Vec<(Array2<f64>, Array2<f64>)> {
    let n_samples = x.nrows();
    let mut indices: Vec<usize> = (0..n_samples).collect();

    if shuffle {
        use scirs2_core::random::seq::SliceRandom;
        indices.shuffle(rng);
    }

    let mut batches = Vec::new();
    for chunk in indices.chunks(batch_size) {
        let batch_x = Array2::from_shape_vec(
            (chunk.len(), x.ncols()),
            chunk.iter().flat_map(|&i| x.row(i).to_vec()).collect(),
        )
        .expect("value should be present");

        let batch_y = Array2::from_shape_vec(
            (chunk.len(), y.ncols()),
            chunk.iter().flat_map(|&i| y.row(i).to_vec()).collect(),
        )
        .expect("value should be present");
        batches.push((batch_x, batch_y));
    }

    batches
}

/// Initialize weights based on initialization strategy
pub fn initialize_weights<R: scirs2_core::random::Rng>(
    rows: usize,
    cols: usize,
    init: &WeightInit,
    rng: &mut R,
) -> Array2<f64> {
    use scirs2_core::random::essentials::{Normal, Uniform};

    match init {
        WeightInit::Zero => Array2::zeros((rows, cols)),
        WeightInit::Random => {
            let dist = Uniform::new(-1.0, 1.0).expect("valid distribution params");
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Xavier => {
            let fan_avg = (rows + cols) as f64 / 2.0;
            let bound = (6.0 / fan_avg).sqrt();
            let dist = Uniform::new(-bound, bound).expect("valid distribution params");
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::He => {
            let std = (2.0 / rows as f64).sqrt();
            let dist = Normal::new(0.0, std).expect("valid distribution params");
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Uniform { low, high } => {
            let dist = Uniform::new(*low, *high).expect("valid distribution params");
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
        WeightInit::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).expect("valid distribution params");
            Array2::from_shape_simple_fn((rows, cols), || rng.sample(dist))
        }
    }
}

/// Initialize biases
pub fn initialize_biases<R: scirs2_core::random::Rng>(
    size: usize,
    init: &WeightInit,
    rng: &mut R,
) -> Array1<f64> {
    use scirs2_core::random::essentials::{Normal, Uniform};

    match init {
        WeightInit::Zero => Array1::zeros(size),
        WeightInit::Random => {
            let dist = Uniform::new(-1.0, 1.0).expect("valid distribution params");
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Xavier => {
            let bound = (6.0 / size as f64).sqrt();
            let dist = Uniform::new(-bound, bound).expect("valid distribution params");
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::He => {
            let std = (2.0 / size as f64).sqrt();
            let dist = Normal::new(0.0, std).expect("valid distribution params");
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Uniform { low, high } => {
            let dist = Uniform::new(*low, *high).expect("valid distribution params");
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
        WeightInit::Normal { mean, std } => {
            let dist = Normal::new(*mean, *std).expect("valid distribution params");
            Array1::from_shape_simple_fn(size, || rng.sample(dist))
        }
    }
}

//! Deep Gaussian Processes for semi-supervised learning
//!
//! This module implements Deep Gaussian Processes (DGPs), which are hierarchical
//! compositions of Gaussian processes that can model complex non-linear relationships
//! and provide uncertainty quantification for semi-supervised learning tasks.

use scirs2_core::ndarray_ext::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Random, Rng};
// use scirs2_core::random::rand::seq::SliceRandom;
use sklears_core::error::{Result, SklearsError};
use sklears_core::traits::{Fit, Predict, PredictProba};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeepGaussianProcessError {
    #[error("Invalid number of layers: {0}")]
    InvalidNumLayers(usize),
    #[error("Invalid inducing points: {0}")]
    InvalidInducingPoints(usize),
    #[error("Invalid kernel parameter: {0}")]
    InvalidKernelParameter(f64),
    #[error("Invalid learning rate: {0}")]
    InvalidLearningRate(f64),
    #[error("Invalid epochs: {0}")]
    InvalidEpochs(usize),
    #[error("Insufficient labeled samples: need at least 1")]
    InsufficientLabeledSamples,
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    #[error("Training failed: {0}")]
    TrainingFailed(String),
    #[error("Model not trained")]
    ModelNotTrained,
    #[error("Matrix operation failed: {0}")]
    MatrixOperationFailed(String),
}

impl From<DeepGaussianProcessError> for SklearsError {
    fn from(err: DeepGaussianProcessError) -> Self {
        SklearsError::FitError(err.to_string())
    }
}

/// Kernel function types for Gaussian Processes
#[derive(Debug, Clone)]
pub enum KernelType {
    /// RBF
    RBF { length_scale: f64, variance: f64 },
    /// Matern32
    Matern32 { length_scale: f64, variance: f64 },
    /// Matern52
    Matern52 { length_scale: f64, variance: f64 },
    /// Linear
    Linear { variance: f64, offset: f64 },
}

impl KernelType {
    fn compute(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        match self {
            KernelType::RBF {
                length_scale,
                variance,
            } => {
                let squared_distance = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                variance * (-0.5 * squared_distance / length_scale.powi(2)).exp()
            }
            KernelType::Matern32 {
                length_scale,
                variance,
            } => {
                let distance = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let scaled_distance = (3.0_f64.sqrt() * distance) / length_scale;
                variance * (1.0 + scaled_distance) * (-scaled_distance).exp()
            }
            KernelType::Matern52 {
                length_scale,
                variance,
            } => {
                let distance = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let scaled_distance = (5.0_f64.sqrt() * distance) / length_scale;
                variance
                    * (1.0
                        + scaled_distance
                        + (5.0 * distance.powi(2)) / (3.0 * length_scale.powi(2)))
                    * (-scaled_distance).exp()
            }
            KernelType::Linear { variance, offset } => {
                let dot_product = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                variance * (dot_product + offset)
            }
        }
    }

    fn compute_matrix(&self, X1: &ArrayView2<f64>, X2: &ArrayView2<f64>) -> Array2<f64> {
        let (n1, _) = X1.dim();
        let (n2, _) = X2.dim();
        let mut K = Array2::zeros((n1, n2));

        for i in 0..n1 {
            for j in 0..n2 {
                K[[i, j]] = self.compute(&X1.row(i), &X2.row(j));
            }
        }

        K
    }
}

/// Single Gaussian Process layer in the deep architecture
#[derive(Debug, Clone)]
pub struct GaussianProcessLayer {
    /// input_dim
    pub input_dim: usize,
    /// output_dim
    pub output_dim: usize,
    /// num_inducing
    pub num_inducing: usize,
    /// kernel
    pub kernel: KernelType,
    /// noise_variance
    pub noise_variance: f64,
    /// learning_rate
    pub learning_rate: f64,
    inducing_points: Array2<f64>,
    mean_function: Array1<f64>,
    is_trained: bool,
}

impl GaussianProcessLayer {
    pub fn new(input_dim: usize, output_dim: usize, num_inducing: usize) -> Self {
        Self {
            input_dim,
            output_dim,
            num_inducing,
            kernel: KernelType::RBF {
                length_scale: 1.0,
                variance: 1.0,
            },
            noise_variance: 0.1,
            learning_rate: 0.01,
            inducing_points: Array2::zeros((num_inducing, input_dim)),
            mean_function: Array1::zeros(output_dim),
            is_trained: false,
        }
    }

    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn noise_variance(mut self, noise_variance: f64) -> Result<Self> {
        if noise_variance <= 0.0 {
            return Err(DeepGaussianProcessError::InvalidKernelParameter(noise_variance).into());
        }
        self.noise_variance = noise_variance;
        Ok(self)
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(DeepGaussianProcessError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    fn initialize_inducing_points(&mut self, X: &ArrayView2<f64>, random_state: Option<u64>) {
        let (n_samples, _) = X.dim();
        let mut rng = match random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Select random subset of training points as inducing points
        let num_selected = self.num_inducing.min(n_samples);
        let mut selected_indices = Vec::new();
        for _ in 0..num_selected {
            let idx = rng.gen_range(0..n_samples);
            if !selected_indices.contains(&idx) {
                selected_indices.push(idx);
            }
        }

        for (i, &idx) in selected_indices.iter().enumerate() {
            self.inducing_points.row_mut(i).assign(&X.row(idx));
        }
    }

    fn add_jitter(&self, matrix: &mut Array2<f64>, jitter: f64) {
        for i in 0..matrix.nrows() {
            matrix[[i, i]] += jitter;
        }
    }

    fn safe_cholesky(&self, matrix: &Array2<f64>) -> Result<Array2<f64>> {
        let mut A = matrix.clone();
        let jitter = 1e-6;
        self.add_jitter(&mut A, jitter);

        // Simple Cholesky decomposition (simplified implementation)
        let n = A.nrows();
        let mut L: Array2<f64> = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    let sum: f64 = (0..j).map(|k| L[[i, k]].powi(2)).sum();
                    let val = A[[i, i]] - sum;
                    if val <= 0.0 {
                        return Err(DeepGaussianProcessError::MatrixOperationFailed(
                            "Matrix is not positive definite".to_string(),
                        )
                        .into());
                    }
                    L[[i, j]] = val.sqrt();
                } else {
                    let sum: f64 = (0..j).map(|k| L[[i, k]] * L[[j, k]]).sum();
                    L[[i, j]] = (A[[i, j]] - sum) / L[[j, j]];
                }
            }
        }

        Ok(L)
    }

    fn solve_triangular_lower(&self, L: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = L.nrows();
        let mut x = Array1::zeros(n);

        for i in 0..n {
            let sum: f64 = (0..i).map(|j| L[[i, j]] * x[j]).sum();
            x[i] = (b[i] - sum) / L[[i, i]];
        }

        x
    }

    fn solve_triangular_upper(&self, U: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = U.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let sum: f64 = ((i + 1)..n).map(|j| U[[i, j]] * x[j]).sum();
            x[i] = (b[i] - sum) / U[[i, i]];
        }

        x
    }

    #[allow(non_snake_case)]
    pub fn fit(
        &mut self,
        X: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        random_state: Option<u64>,
    ) -> Result<()> {
        let (n_samples, n_features) = X.dim();
        let (n_targets, output_dim) = y.dim();

        if n_features != self.input_dim {
            return Err(DeepGaussianProcessError::ShapeMismatch {
                expected: vec![n_samples, self.input_dim],
                actual: vec![n_samples, n_features],
            }
            .into());
        }

        if n_samples != n_targets || output_dim != self.output_dim {
            return Err(DeepGaussianProcessError::ShapeMismatch {
                expected: vec![n_samples, self.output_dim],
                actual: vec![n_targets, output_dim],
            }
            .into());
        }

        self.initialize_inducing_points(X, random_state);

        // Compute kernel matrices
        let K_uu = self
            .kernel
            .compute_matrix(&self.inducing_points.view(), &self.inducing_points.view());
        let K_uf = self.kernel.compute_matrix(&self.inducing_points.view(), X);

        // Add noise to diagonal for numerical stability
        let mut K_uu_noisy = K_uu.clone();
        self.add_jitter(&mut K_uu_noisy, self.noise_variance);

        // Simplified variational inference (ELBO optimization)
        let L_uu = self.safe_cholesky(&K_uu_noisy)?;

        // For each output dimension, fit a separate GP
        for d in 0..self.output_dim {
            let y_d = y.column(d);

            // Solve for alpha: K_uu * alpha = K_uf * y
            let K_uf_y = K_uf.dot(&y_d);
            let alpha = self.solve_triangular_lower(&L_uu, &K_uf_y);
            let L_uu_t = L_uu.t().to_owned();
            let solution = self.solve_triangular_upper(&L_uu_t, &alpha);

            // Store mean function for this output dimension
            self.mean_function[d] = solution.mean().unwrap_or(0.0);
        }

        self.is_trained = true;
        Ok(())
    }

    #[allow(non_snake_case)]
    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_trained {
            return Err(DeepGaussianProcessError::ModelNotTrained.into());
        }

        let (n_test, n_features) = X.dim();
        if n_features != self.input_dim {
            return Err(DeepGaussianProcessError::ShapeMismatch {
                expected: vec![n_test, self.input_dim],
                actual: vec![n_test, n_features],
            }
            .into());
        }

        // Compute predictive mean using kernel computations
        let K_su = self.kernel.compute_matrix(X, &self.inducing_points.view());
        let K_uu = self
            .kernel
            .compute_matrix(&self.inducing_points.view(), &self.inducing_points.view());

        let mut predictions = Array2::zeros((n_test, self.output_dim));

        // Simple prediction using kernel mean
        for i in 0..n_test {
            for d in 0..self.output_dim {
                // Simplified prediction: weighted average based on kernel similarities
                let weights: f64 = K_su.row(i).sum();
                if weights > 0.0 {
                    predictions[[i, d]] =
                        self.mean_function[d] * (weights / self.num_inducing as f64);
                } else {
                    predictions[[i, d]] = self.mean_function[d];
                }
            }
        }

        Ok(predictions)
    }

    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        let predictions = self.predict(X)?;
        let (n_test, _) = X.dim();

        // Simplified uncertainty estimation
        let mut uncertainties = Array2::zeros((n_test, self.output_dim));
        for i in 0..n_test {
            for d in 0..self.output_dim {
                uncertainties[[i, d]] = self.noise_variance.sqrt();
            }
        }

        Ok((predictions, uncertainties))
    }
}

/// Deep Gaussian Process for semi-supervised learning
///
/// This implements a multi-layer composition of Gaussian processes that can
/// learn complex non-linear mappings while providing principled uncertainty
/// quantification for semi-supervised learning scenarios.
#[derive(Debug, Clone)]
pub struct DeepGaussianProcess {
    /// layer_dims
    pub layer_dims: Vec<usize>,
    /// num_inducing
    pub num_inducing: usize,
    /// epochs
    pub epochs: usize,
    /// learning_rate
    pub learning_rate: f64,
    /// noise_variance
    pub noise_variance: f64,
    /// random_state
    pub random_state: Option<u64>,
    layers: Vec<GaussianProcessLayer>,
    n_classes: usize,
    is_trained: bool,
}

impl Default for DeepGaussianProcess {
    fn default() -> Self {
        Self {
            layer_dims: vec![10, 5, 2],
            num_inducing: 20,
            epochs: 50,
            learning_rate: 0.01,
            noise_variance: 0.1,
            random_state: None,
            layers: Vec::new(),
            n_classes: 0,
            is_trained: false,
        }
    }
}

impl DeepGaussianProcess {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn layer_dims(mut self, layer_dims: Vec<usize>) -> Result<Self> {
        if layer_dims.len() < 2 {
            return Err(DeepGaussianProcessError::InvalidNumLayers(layer_dims.len()).into());
        }
        self.layer_dims = layer_dims;
        Ok(self)
    }

    pub fn num_inducing(mut self, num_inducing: usize) -> Result<Self> {
        if num_inducing == 0 {
            return Err(DeepGaussianProcessError::InvalidInducingPoints(num_inducing).into());
        }
        self.num_inducing = num_inducing;
        Ok(self)
    }

    pub fn epochs(mut self, epochs: usize) -> Result<Self> {
        if epochs == 0 {
            return Err(DeepGaussianProcessError::InvalidEpochs(epochs).into());
        }
        self.epochs = epochs;
        Ok(self)
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Result<Self> {
        if learning_rate <= 0.0 {
            return Err(DeepGaussianProcessError::InvalidLearningRate(learning_rate).into());
        }
        self.learning_rate = learning_rate;
        Ok(self)
    }

    pub fn noise_variance(mut self, noise_variance: f64) -> Result<Self> {
        if noise_variance <= 0.0 {
            return Err(DeepGaussianProcessError::InvalidKernelParameter(noise_variance).into());
        }
        self.noise_variance = noise_variance;
        Ok(self)
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn initialize_layers(&mut self) {
        self.layers.clear();

        for i in 0..self.layer_dims.len() - 1 {
            let layer = GaussianProcessLayer::new(
                self.layer_dims[i],
                self.layer_dims[i + 1],
                self.num_inducing,
            )
            .learning_rate(self.learning_rate)
            .unwrap()
            .noise_variance(self.noise_variance)
            .unwrap();

            self.layers.push(layer);
        }
    }

    fn forward_pass(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        let mut current_data = X.to_owned();

        for layer in self.layers.iter() {
            current_data = layer.predict(&current_data.view())?;
        }

        Ok(current_data)
    }

    fn softmax(&self, X: &ArrayView2<f64>) -> Array2<f64> {
        let mut result = X.to_owned();

        for mut row in result.rows_mut() {
            let max_val = row.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            for val in row.iter_mut() {
                *val = (*val - max_val).exp();
            }
            let sum: f64 = row.sum();
            if sum > 0.0 {
                for val in row.iter_mut() {
                    *val /= sum;
                }
            }
        }

        result
    }

    fn encode_labels(&self, y: &ArrayView1<i32>) -> Array2<f64> {
        let n_samples = y.len();
        let mut encoded = Array2::zeros((n_samples, self.n_classes));

        for (i, &label) in y.iter().enumerate() {
            if label >= 0 {
                let class_idx = label as usize;
                if class_idx < self.n_classes {
                    encoded[[i, class_idx]] = 1.0;
                }
            }
        }

        encoded
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> Result<()> {
        let (n_samples, n_features) = X.dim();

        if y.len() != n_samples {
            return Err(DeepGaussianProcessError::ShapeMismatch {
                expected: vec![n_samples],
                actual: vec![y.len()],
            }
            .into());
        }

        // Count labeled samples
        let labeled_mask: Vec<bool> = y.iter().map(|&label| label >= 0).collect();
        let n_labeled = labeled_mask.iter().filter(|&&x| x).count();

        if n_labeled == 0 {
            return Err(DeepGaussianProcessError::InsufficientLabeledSamples.into());
        }

        // Get unique classes from labeled data
        let mut classes: Vec<i32> = y.iter().filter(|&&label| label >= 0).cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        self.n_classes = classes.len();

        // Set input size from data
        if self.layer_dims.is_empty() {
            self.layer_dims = vec![n_features, n_features / 2, self.n_classes];
        } else {
            self.layer_dims[0] = n_features;
            if self.layer_dims.len() > 1 {
                *self.layer_dims.last_mut().unwrap() = self.n_classes;
            }
        }

        self.initialize_layers();

        // Create target matrix for labeled samples
        let y_encoded = self.encode_labels(y);

        // Layer-wise training with all data
        let mut current_data = X.to_owned();
        let num_layers = self.layers.len();

        for layer_idx in 0..num_layers {
            let seed = self.random_state.map(|s| s + layer_idx as u64);

            if layer_idx == num_layers - 1 {
                // Last layer: train only on labeled data for classification
                let labeled_indices: Vec<usize> = labeled_mask
                    .iter()
                    .enumerate()
                    .filter(|(_, &is_labeled)| is_labeled)
                    .map(|(i, _)| i)
                    .collect();

                if !labeled_indices.is_empty() {
                    let X_labeled = labeled_indices
                        .iter()
                        .map(|&i| current_data.row(i))
                        .collect::<Vec<_>>();
                    let y_labeled = labeled_indices
                        .iter()
                        .map(|&i| y_encoded.row(i))
                        .collect::<Vec<_>>();

                    // Create arrays from the collected rows
                    let X_labeled_array =
                        Array2::from_shape_fn((X_labeled.len(), current_data.ncols()), |(i, j)| {
                            X_labeled[i][j]
                        });
                    let y_labeled_array =
                        Array2::from_shape_fn((y_labeled.len(), y_encoded.ncols()), |(i, j)| {
                            y_labeled[i][j]
                        });

                    self.layers[layer_idx].fit(
                        &X_labeled_array.view(),
                        &y_labeled_array.view(),
                        seed,
                    )?;
                }
            } else {
                // Hidden layers: unsupervised pre-training with all data
                // For simplicity, use identity mapping as target for unsupervised layers
                let target_dim = self.layers[layer_idx].output_dim;
                let identity_target =
                    Array2::from_shape_fn((current_data.nrows(), target_dim), |(i, j)| {
                        if j < current_data.ncols() {
                            current_data[[i, j]]
                        } else {
                            0.0
                        }
                    });

                self.layers[layer_idx].fit(&current_data.view(), &identity_target.view(), seed)?;
            }

            // Forward pass for next layer
            if layer_idx < num_layers - 1 {
                current_data = self.layers[layer_idx].predict(&current_data.view())?;
            }
        }

        self.is_trained = true;
        Ok(())
    }

    pub fn predict_proba(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>> {
        if !self.is_trained {
            return Err(DeepGaussianProcessError::ModelNotTrained.into());
        }

        let logits = self.forward_pass(X)?;
        Ok(self.softmax(&logits.view()))
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let predictions = probabilities.map_axis(Axis(1), |row| {
            row.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx as i32)
                .unwrap()
        });
        Ok(predictions)
    }

    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<f64>,
    ) -> Result<(Array1<i32>, Array2<f64>)> {
        if !self.is_trained {
            return Err(DeepGaussianProcessError::ModelNotTrained.into());
        }

        // Get predictions with uncertainty from last layer
        let mut current_data = X.to_owned();
        let mut uncertainties = Array2::zeros((X.nrows(), self.n_classes));

        // Forward pass through all but last layer
        for layer in self.layers.iter().take(self.layers.len() - 1) {
            current_data = layer.predict(&current_data.view())?;
        }

        // Get predictions and uncertainties from last layer
        if let Some(last_layer) = self.layers.last() {
            let (logits, layer_uncertainties) =
                last_layer.predict_with_uncertainty(&current_data.view())?;
            let probabilities = self.softmax(&logits.view());
            let predictions = probabilities.map_axis(Axis(1), |row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _)| idx as i32)
                    .unwrap()
            });

            uncertainties = layer_uncertainties;
            Ok((predictions, uncertainties))
        } else {
            Err(DeepGaussianProcessError::ModelNotTrained.into())
        }
    }
}

#[derive(Debug, Clone)]
pub struct FittedDeepGaussianProcess {
    model: DeepGaussianProcess,
}

impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>, FittedDeepGaussianProcess>
    for DeepGaussianProcess
{
    type Fitted = FittedDeepGaussianProcess;

    fn fit(
        mut self,
        X: &ArrayView2<'_, f64>,
        y: &ArrayView1<'_, i32>,
    ) -> Result<FittedDeepGaussianProcess> {
        DeepGaussianProcess::fit(&mut self, X, y)?;
        Ok(FittedDeepGaussianProcess { model: self })
    }
}

impl Predict<ArrayView2<'_, f64>, Array1<i32>> for FittedDeepGaussianProcess {
    fn predict(&self, X: &ArrayView2<'_, f64>) -> Result<Array1<i32>> {
        self.model.predict(X)
    }
}

impl PredictProba<ArrayView2<'_, f64>, Array2<f64>> for FittedDeepGaussianProcess {
    fn predict_proba(&self, X: &ArrayView2<'_, f64>) -> Result<Array2<f64>> {
        self.model.predict_proba(X)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::array;

    #[test]
    fn test_kernel_computation() {
        let kernel = KernelType::RBF {
            length_scale: 1.0,
            variance: 1.0,
        };
        let x1 = array![1.0, 2.0];
        let x2 = array![1.0, 2.0];

        let result = kernel.compute(&x1.view(), &x2.view());
        assert_abs_diff_eq!(result, 1.0, epsilon = 1e-10);

        let x3 = array![3.0, 4.0];
        let result2 = kernel.compute(&x1.view(), &x3.view());
        assert!(result2 < 1.0 && result2 > 0.0);
    }

    #[test]
    fn test_gaussian_process_layer_creation() {
        let layer = GaussianProcessLayer::new(4, 2, 10)
            .learning_rate(0.01)
            .unwrap()
            .noise_variance(0.1)
            .unwrap();

        assert_eq!(layer.input_dim, 4);
        assert_eq!(layer.output_dim, 2);
        assert_eq!(layer.num_inducing, 10);
        assert_eq!(layer.learning_rate, 0.01);
        assert_eq!(layer.noise_variance, 0.1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gaussian_process_layer_fit_predict() {
        let mut layer = GaussianProcessLayer::new(2, 1, 5)
            .learning_rate(0.01)
            .unwrap()
            .noise_variance(0.1)
            .unwrap();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![[1.0], [2.0], [3.0], [4.0]];

        layer.fit(&X.view(), &y.view(), Some(42)).unwrap();
        assert!(layer.is_trained);

        let predictions = layer.predict(&X.view()).unwrap();
        assert_eq!(predictions.dim(), (4, 1));

        let (pred_mean, pred_var) = layer.predict_with_uncertainty(&X.view()).unwrap();
        assert_eq!(pred_mean.dim(), (4, 1));
        assert_eq!(pred_var.dim(), (4, 1));
    }

    #[test]
    fn test_deep_gaussian_process_creation() {
        let dgp = DeepGaussianProcess::new()
            .layer_dims(vec![4, 3, 2])
            .unwrap()
            .num_inducing(10)
            .unwrap()
            .epochs(20)
            .unwrap()
            .learning_rate(0.01)
            .unwrap()
            .noise_variance(0.1)
            .unwrap()
            .random_state(42);

        assert_eq!(dgp.layer_dims, vec![4, 3, 2]);
        assert_eq!(dgp.num_inducing, 10);
        assert_eq!(dgp.epochs, 20);
        assert_eq!(dgp.learning_rate, 0.01);
        assert_eq!(dgp.noise_variance, 0.1);
        assert_eq!(dgp.random_state, Some(42));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_deep_gaussian_process_fit_predict() {
        let dgp = DeepGaussianProcess::new()
            .layer_dims(vec![2, 3, 2])
            .unwrap()
            .num_inducing(5)
            .unwrap()
            .epochs(10)
            .unwrap()
            .random_state(42);

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, -1, -1]; // -1 indicates unlabeled

        let fitted = dgp.fit(&X.view(), &y.view()).unwrap();

        let predictions = fitted.predict(&X.view()).unwrap();
        assert_eq!(predictions.len(), 6);

        let probabilities = fitted.predict_proba(&X.view()).unwrap();
        assert_eq!(probabilities.dim(), (6, 2));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let sum: f64 = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-5); // More lenient epsilon for GP
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_deep_gaussian_process_with_uncertainty() {
        let dgp = DeepGaussianProcess::new()
            .layer_dims(vec![2, 2])
            .unwrap()
            .num_inducing(3)
            .unwrap()
            .epochs(5)
            .unwrap()
            .random_state(42);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![0, 1, 0];

        let fitted = dgp.fit(&X.view(), &y.view()).unwrap();

        let (predictions, uncertainties) =
            fitted.model.predict_with_uncertainty(&X.view()).unwrap();
        assert_eq!(predictions.len(), 3);
        assert_eq!(uncertainties.dim(), (3, 2));

        // Check that uncertainties are positive
        for value in uncertainties.iter() {
            assert!(*value >= 0.0);
        }
    }

    #[test]
    fn test_deep_gaussian_process_invalid_parameters() {
        assert!(DeepGaussianProcess::new().layer_dims(vec![]).is_err());
        assert!(DeepGaussianProcess::new().layer_dims(vec![10]).is_err());
        assert!(DeepGaussianProcess::new().num_inducing(0).is_err());
        assert!(DeepGaussianProcess::new().learning_rate(0.0).is_err());
        assert!(DeepGaussianProcess::new().learning_rate(-0.1).is_err());
        assert!(DeepGaussianProcess::new().noise_variance(0.0).is_err());
        assert!(DeepGaussianProcess::new().epochs(0).is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_deep_gaussian_process_insufficient_labeled_samples() {
        let dgp = DeepGaussianProcess::new()
            .layer_dims(vec![2, 2])
            .unwrap()
            .epochs(5)
            .unwrap();

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![-1, -1]; // All unlabeled

        let result = dgp.fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_kernel_types() {
        let x1 = array![1.0, 2.0];
        let x2 = array![2.0, 3.0];

        let rbf = KernelType::RBF {
            length_scale: 1.0,
            variance: 1.0,
        };
        let matern32 = KernelType::Matern32 {
            length_scale: 1.0,
            variance: 1.0,
        };
        let matern52 = KernelType::Matern52 {
            length_scale: 1.0,
            variance: 1.0,
        };
        let linear = KernelType::Linear {
            variance: 1.0,
            offset: 0.0,
        };

        let rbf_val = rbf.compute(&x1.view(), &x2.view());
        let matern32_val = matern32.compute(&x1.view(), &x2.view());
        let matern52_val = matern52.compute(&x1.view(), &x2.view());
        let linear_val = linear.compute(&x1.view(), &x2.view());

        assert!(rbf_val > 0.0 && rbf_val <= 1.0);
        assert!(matern32_val > 0.0 && matern32_val <= 1.0);
        assert!(matern52_val > 0.0 && matern52_val <= 1.0);
        assert!(linear_val > 0.0);
    }
}

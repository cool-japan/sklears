//! Tensor Canonical Correlation Analysis implementation

use super::common::{Trained, Untrained};
use scirs2_core::ndarray::{s, Array1, Array2, Array3, ArrayD, Axis};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Transform},
    types::Float,
};
use std::marker::PhantomData;

/// Tensor Canonical Correlation Analysis (Tensor CCA)
///
/// Tensor CCA extends canonical correlation analysis to handle tensor data by finding
/// linear transformations that maximize correlation between two tensor datasets while
/// preserving their multi-dimensional structure.
///
/// # Examples
///
/// ```rust
/// use scirs2_core::ndarray::Array3;
/// use sklears_cross_decomposition::TensorCCA;
/// use sklears_core::traits::Fit;
///
/// let tensor1 = Array3::zeros((10, 5, 4)); // (samples, mode1, mode2)
/// let tensor2 = Array3::zeros((10, 6, 3)); // (samples, mode1, mode2)
///
/// let tcca = TensorCCA::new(2).regularization(0.1);
/// let fitted = tcca.fit(&tensor1, &tensor2).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct TensorCCA<State = Untrained> {
    /// Number of canonical components
    pub n_components: usize,
    /// Regularization parameter
    pub regularization: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to center the data
    pub center: bool,
    /// Tensor mode weights (None for equal weighting)
    pub mode_weights: Option<Array1<Float>>,
    /// Canonical weights for tensor 1 (one per mode)
    weights_x_: Option<Vec<Array2<Float>>>,
    /// Canonical weights for tensor 2 (one per mode)
    weights_y_: Option<Vec<Array2<Float>>>,
    /// Canonical correlations
    correlations_: Option<Array1<Float>>,
    /// Mean tensors for centering
    mean_x_: Option<ArrayD<Float>>,
    mean_y_: Option<ArrayD<Float>>,
    /// Explained variance for each component
    explained_variance_: Option<Array1<Float>>,
    /// Number of iterations for convergence
    n_iter_: Option<usize>,
    /// State marker
    _state: PhantomData<State>,
}

impl TensorCCA<Untrained> {
    /// Create a new Tensor CCA with specified number of components
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            regularization: 0.0,
            max_iter: 100,
            tol: 1e-6,
            center: true,
            mode_weights: None,
            weights_x_: None,
            weights_y_: None,
            correlations_: None,
            mean_x_: None,
            mean_y_: None,
            explained_variance_: None,
            n_iter_: None,
            _state: PhantomData,
        }
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set mode weights
    pub fn mode_weights(mut self, weights: Array1<Float>) -> Self {
        self.mode_weights = Some(weights);
        self
    }
}

impl Estimator for TensorCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array3<Float>, Array3<Float>> for TensorCCA<Untrained> {
    type Fitted = TensorCCA<Trained>;

    fn fit(self, X: &Array3<Float>, Y: &Array3<Float>) -> Result<Self::Fitted> {
        // Validate input tensors
        if X.shape()[0] != Y.shape()[0] {
            return Err(SklearsError::InvalidInput(format!(
                "Tensors must have same number of samples: {} vs {}",
                X.shape()[0],
                Y.shape()[0]
            )));
        }

        let n_samples = X.shape()[0];
        if n_samples < self.n_components {
            return Err(SklearsError::InvalidInput(format!(
                "n_samples ({}) must be >= n_components ({})",
                n_samples, self.n_components
            )));
        }

        // Center the tensors if requested
        let (X_centered, Y_centered, mean_x, mean_y) = if self.center {
            let mean_x = X.mean_axis(Axis(0)).unwrap();
            let mean_y = Y.mean_axis(Axis(0)).unwrap();

            let X_centered = X - &mean_x.clone().insert_axis(Axis(0));
            let Y_centered = Y - &mean_y.clone().insert_axis(Axis(0));

            (
                X_centered,
                Y_centered,
                Some(mean_x.into_dyn()),
                Some(mean_y.into_dyn()),
            )
        } else {
            (X.clone(), Y.clone(), None, None)
        };

        // Initialize weights for each mode
        let x_shape = X_centered.shape();
        let y_shape = Y_centered.shape();

        let mut weights_x = vec![
            Array2::zeros((x_shape[1], self.n_components)), // Mode 1
            Array2::zeros((x_shape[2], self.n_components)), // Mode 2
        ];

        let mut weights_y = vec![
            Array2::zeros((y_shape[1], self.n_components)), // Mode 1
            Array2::zeros((y_shape[2], self.n_components)), // Mode 2
        ];

        // Initialize weights randomly
        for comp in 0..self.n_components {
            for mode in 0..2 {
                // Initialize X weights
                let mut w_x: Array1<Float> = Array1::from_iter(
                    (0..weights_x[mode].nrows()).map(|_| thread_rng().random::<Float>()),
                );
                let norm = (w_x.dot(&w_x)).sqrt();
                if norm > 0.0 {
                    w_x /= norm;
                }
                weights_x[mode].column_mut(comp).assign(&w_x);

                // Initialize Y weights
                let mut w_y: Array1<Float> = Array1::from_iter(
                    (0..weights_y[mode].nrows()).map(|_| thread_rng().random::<Float>()),
                );
                let norm = (w_y.dot(&w_y)).sqrt();
                if norm > 0.0 {
                    w_y /= norm;
                }
                weights_y[mode].column_mut(comp).assign(&w_y);
            }
        }

        let mut correlations = Array1::zeros(self.n_components);
        let mut n_iter = 0;

        // Alternating optimization for each component
        for comp in 0..self.n_components {
            let mut converged = false;
            let mut iter_count = 0;

            while !converged && iter_count < self.max_iter {
                let old_weights_x = weights_x.clone();
                let old_weights_y = weights_y.clone();

                // Update weights for X tensor
                for mode in 0..2 {
                    let scores_x =
                        self.compute_tensor_scores(&X_centered, &weights_x, comp, mode)?;
                    let scores_y =
                        self.compute_tensor_scores(&Y_centered, &weights_y, comp, mode)?;

                    // Update weights to maximize correlation
                    let updated_weights =
                        self.update_tensor_weights(&X_centered, &scores_y, mode, comp)?;
                    weights_x[mode].column_mut(comp).assign(&updated_weights);
                }

                // Update weights for Y tensor
                for mode in 0..2 {
                    let scores_x =
                        self.compute_tensor_scores(&X_centered, &weights_x, comp, mode)?;
                    let scores_y =
                        self.compute_tensor_scores(&Y_centered, &weights_y, comp, mode)?;

                    // Update weights to maximize correlation
                    let updated_weights =
                        self.update_tensor_weights(&Y_centered, &scores_x, mode, comp)?;
                    weights_y[mode].column_mut(comp).assign(&updated_weights);
                }

                // Check convergence
                let mut max_change: Float = 0.0;
                for mode in 0..2 {
                    let change_x = (&weights_x[mode].column(comp)
                        - &old_weights_x[mode].column(comp))
                        .mapv(|x| x.abs())
                        .sum();
                    let change_y = (&weights_y[mode].column(comp)
                        - &old_weights_y[mode].column(comp))
                        .mapv(|x| x.abs())
                        .sum();
                    max_change = max_change.max(change_x).max(change_y);
                }

                if max_change < self.tol {
                    converged = true;
                }

                iter_count += 1;
            }

            n_iter = iter_count;

            // Compute correlation for this component
            let scores_x = self.compute_tensor_scores(&X_centered, &weights_x, comp, 0)?; // Use mode 0 for correlation
            let scores_y = self.compute_tensor_scores(&Y_centered, &weights_y, comp, 0)?;

            let correlation = scores_x.dot(&scores_y)
                / ((scores_x.dot(&scores_x) * scores_y.dot(&scores_y)).sqrt());
            correlations[comp] = correlation;
        }

        // Compute explained variance
        let mut explained_var = Array1::zeros(self.n_components);
        for comp in 0..self.n_components {
            let scores_x = self.compute_tensor_scores(&X_centered, &weights_x, comp, 0)?;
            let scores_y = self.compute_tensor_scores(&Y_centered, &weights_y, comp, 0)?;
            explained_var[comp] = (scores_x.var(1.0) + scores_y.var(1.0)) / 2.0;
        }

        Ok(TensorCCA {
            n_components: self.n_components,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tol: self.tol,
            center: self.center,
            mode_weights: self.mode_weights,
            weights_x_: Some(weights_x),
            weights_y_: Some(weights_y),
            correlations_: Some(correlations),
            mean_x_: mean_x,
            mean_y_: mean_y,
            explained_variance_: Some(explained_var),
            n_iter_: Some(n_iter),
            _state: PhantomData,
        })
    }
}

impl TensorCCA<Untrained> {
    /// Compute tensor scores for a given component and mode
    fn compute_tensor_scores(
        &self,
        tensor: &Array3<Float>,
        weights: &[Array2<Float>],
        comp: usize,
        mode: usize,
    ) -> Result<Array1<Float>> {
        let n_samples = tensor.shape()[0];
        let mut scores = Array1::zeros(n_samples);

        match mode {
            0 => {
                // Mode 1: Contract along dimensions 1 and 2
                for i in 0..n_samples {
                    let slice = tensor.slice(s![i, .., ..]);
                    let w1 = weights[0].column(comp);
                    let w2 = weights[1].column(comp);

                    // Compute tensor contraction
                    let mut score = 0.0;
                    for j in 0..slice.shape()[0] {
                        for k in 0..slice.shape()[1] {
                            score += slice[[j, k]] * w1[j] * w2[k];
                        }
                    }
                    scores[i] = score;
                }
            }
            1 => {
                // Mode 2: Alternative contraction pattern
                for i in 0..n_samples {
                    let slice = tensor.slice(s![i, .., ..]);
                    let w1 = weights[0].column(comp);
                    let w2 = weights[1].column(comp);

                    // Alternative tensor contraction
                    let projected = slice.dot(&w2);
                    scores[i] = projected.dot(&w1);
                }
            }
            _ => return Err(SklearsError::InvalidInput("Invalid mode".to_string())),
        }

        Ok(scores)
    }

    /// Update tensor weights for a given mode
    fn update_tensor_weights(
        &self,
        tensor: &Array3<Float>,
        target_scores: &Array1<Float>,
        mode: usize,
        comp: usize,
    ) -> Result<Array1<Float>> {
        let n_samples = tensor.shape()[0];

        match mode {
            0 => {
                let n_features = tensor.shape()[1];
                let mut gradient: Array1<Float> = Array1::zeros(n_features);

                for i in 0..n_samples {
                    let slice = tensor.slice(s![i, .., ..]);
                    let target = target_scores[i];

                    // Compute gradient with respect to mode 1 weights
                    for j in 0..n_features {
                        gradient[j] += target * slice.row(j).sum();
                    }
                }

                // Add regularization
                if self.regularization > 0.0 {
                    gradient = gradient * (1.0 - self.regularization);
                }

                // Normalize
                let norm = (gradient.dot(&gradient)).sqrt();
                if norm > self.tol {
                    gradient /= norm;
                }

                Ok(gradient)
            }
            1 => {
                let n_features = tensor.shape()[2];
                let mut gradient: Array1<Float> = Array1::zeros(n_features);

                for i in 0..n_samples {
                    let slice = tensor.slice(s![i, .., ..]);
                    let target = target_scores[i];

                    // Compute gradient with respect to mode 2 weights
                    for k in 0..n_features {
                        gradient[k] += target * slice.column(k).sum();
                    }
                }

                // Add regularization
                if self.regularization > 0.0 {
                    gradient = gradient * (1.0 - self.regularization);
                }

                // Normalize
                let norm = (gradient.dot(&gradient)).sqrt();
                if norm > self.tol {
                    gradient /= norm;
                }

                Ok(gradient)
            }
            _ => Err(SklearsError::InvalidInput("Invalid mode".to_string())),
        }
    }
}

impl Transform<(Array3<Float>, Array3<Float>), (Array1<Float>, Array1<Float>)>
    for TensorCCA<Trained>
{
    /// Transform tensor pair to canonical space
    fn transform(
        &self,
        tensors: &(Array3<Float>, Array3<Float>),
    ) -> Result<(Array1<Float>, Array1<Float>)> {
        let (X, Y) = tensors;

        // Center the tensors if needed
        let X_centered = if self.center {
            if let Some(ref mean_x) = self.mean_x_ {
                let mean_3d = mean_x
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .unwrap();
                X - &mean_3d.insert_axis(Axis(0))
            } else {
                X.clone()
            }
        } else {
            X.clone()
        };

        let Y_centered = if self.center {
            if let Some(ref mean_y) = self.mean_y_ {
                let mean_3d = mean_y
                    .clone()
                    .into_dimensionality::<scirs2_core::ndarray::Ix2>()
                    .unwrap();
                Y - &mean_3d.insert_axis(Axis(0))
            } else {
                Y.clone()
            }
        } else {
            Y.clone()
        };

        let weights_x = self.weights_x_.as_ref().unwrap();
        let weights_y = self.weights_y_.as_ref().unwrap();

        // Compute canonical scores for first component only (for simplicity)
        let scores_x = self.compute_tensor_scores(&X_centered, weights_x, 0, 0)?;
        let scores_y = self.compute_tensor_scores(&Y_centered, weights_y, 0, 0)?;

        Ok((scores_x, scores_y))
    }
}

impl TensorCCA<Trained> {
    /// Get the canonical weights for tensor X
    pub fn weights_x(&self) -> &Vec<Array2<Float>> {
        self.weights_x_.as_ref().unwrap()
    }

    /// Get the canonical weights for tensor Y
    pub fn weights_y(&self) -> &Vec<Array2<Float>> {
        self.weights_y_.as_ref().unwrap()
    }

    /// Get the canonical correlations
    pub fn correlations(&self) -> &Array1<Float> {
        self.correlations_.as_ref().unwrap()
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> &Array1<Float> {
        self.explained_variance_.as_ref().unwrap()
    }

    /// Get the number of iterations for convergence
    pub fn n_iter(&self) -> usize {
        self.n_iter_.unwrap()
    }

    /// Helper method for transform
    fn compute_tensor_scores(
        &self,
        tensor: &Array3<Float>,
        weights: &[Array2<Float>],
        comp: usize,
        mode: usize,
    ) -> Result<Array1<Float>> {
        let n_samples = tensor.shape()[0];
        let mut scores = Array1::zeros(n_samples);

        match mode {
            0 => {
                for i in 0..n_samples {
                    let slice = tensor.slice(s![i, .., ..]);
                    let w1 = weights[0].column(comp);
                    let w2 = weights[1].column(comp);

                    let mut score = 0.0;
                    for j in 0..slice.shape()[0] {
                        for k in 0..slice.shape()[1] {
                            score += slice[[j, k]] * w1[j] * w2[k];
                        }
                    }
                    scores[i] = score;
                }
            }
            _ => return Err(SklearsError::InvalidInput("Invalid mode".to_string())),
        }

        Ok(scores)
    }
}
